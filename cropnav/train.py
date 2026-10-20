"""Supervised navigation training with DeepSpeed ZeRO Stage 2."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import deepspeed
from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import TorchCheckpointEngine
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler

from .data import NavigationDataset, navigation_prompt
from .pretrained import load_assets


class TrainableCheckpointEngine(TorchCheckpointEngine):
    """Keep ZeRO optimizer shards while reusing externally supplied frozen weights."""
    def save(self, state_dict, path):
        if "module" in state_dict:
            names = set(state_dict["trainable_names"])
            state_dict = dict(state_dict)
            state_dict["module"] = {k: v for k, v in state_dict["module"].items() if k in names}
            state_dict["frozen_param_shapes"] = None
            state_dict["frozen_param_fragments"] = None
            state_dict["buffer_names"] = []
            state_dict["shared_params"] = {k: v for k, v in state_dict["shared_params"].items()
                                           if k in names and v in names}
        return super().save(state_dict, path)

    def load(self, path, map_location=None):
        state = super().load(path, map_location)
        if "module" in state and set(state["module"]) != set(state["trainable_names"]):
            raise ValueError("Checkpoint is missing optimized parameters")
        return state


def navigation_loss(prediction, target):
    """Cosine direction loss + L1 distance loss; stationary labels omit direction."""
    prediction, target = prediction.float(), target.float()
    moving = target[:, 3] > 1e-6
    angle = 1 - F.cosine_similarity(prediction[:, :3], target[:, :3], dim=-1)
    angle = (angle * moving).sum() / moving.sum().clamp_min(1)
    distance = F.l1_loss(prediction[:, 3], target[:, 3])
    return angle + distance, angle, distance


class NavigationTrainingModel(nn.Module):
    def __init__(self, policy, encoder, tokenizer, history_frames):
        super().__init__()
        self.policy, self.encoder, self.tokenizer = policy, encoder, tokenizer
        self.history_frames = history_frames

    def train(self, mode=True):
        super().train(mode)
        # Frozen EVA and Q-Former stay deterministic; projection layers still learn.
        self.encoder.eval()
        return self

    def forward(self, samples):
        device = self.policy.action_embedding.device
        instructions = [s["instruction"] for s in samples]
        padding = [self.history_frames + 1 - len(s["images"]) for s in samples]
        spatial = []
        for frame in range(self.history_frames + 1):
            images = torch.stack([s["images"][max(0, frame - pad)] for s, pad in zip(samples, padding)])
            context, tokens = self.encoder(images, instructions)
            spatial.append(tokens)
        history = torch.stack(spatial[:-1], dim=1) if self.history_frames else tokens[:, None][:, :0]
        valid = torch.arange(self.history_frames, device=device)[None] >= torch.tensor(padding, device=device)[:, None]
        memory = self.policy.compression.warmup_memory(history, valid_mask=valid)
        points = [s["waypoints"][-self.policy.compression.config.max_history:] for s in samples]
        points = [p.expand(2, -1) if len(p) == 1 else p for p in points]
        lengths = torch.tensor([len(p) for p in points], device=device)
        waypoints = pad_sequence(points, batch_first=True).to(device)
        compressed = self.policy.compression(context, tokens, waypoints, memory, lengths)
        prompts = [navigation_prompt(self.tokenizer, text, self.policy.markers)[0].to(device) for text in instructions]
        ids = pad_sequence(prompts, batch_first=True, padding_value=0)
        lengths = torch.tensor([len(p) for p in prompts], device=device)
        mask = torch.arange(ids.shape[1], device=device)[None] < lengths[:, None]
        predictions = self.policy.decode(ids, compressed, mask)
        targets = torch.stack([s["target"] for s in samples]).to(device)
        nav, angle, distance = navigation_loss(predictions, targets)
        retention = compressed.retention_loss
        loss = nav + retention
        return loss, torch.stack([loss.detach(), angle.detach(), distance.detach(), retention.detach()])


def configure_training(model, args):
    policy = model.policy
    if args.stage == "selector":
        model.requires_grad_(False)
        policy.compression.view_selector.requires_grad_(True)
    else:
        for name in ("query_projection", "key_projection", "context_projection", "spatial_projection"):
            model.encoder.get_submodule(name).requires_grad_(True)
        from peft import LoraConfig, TaskType, get_peft_model
        policy.decoder = get_peft_model(policy.decoder, LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none", task_type=TaskType.FEATURE_EXTRACTION,
        ))
    policy.decoder.gradient_checkpointing_enable()
    policy.decoder.config.use_cache = False


def parameter_groups(model, args):
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        multiplier = 1.0
        if args.stage == "joint":
            if name.startswith("policy.compression.view_selector."):
                multiplier = 10.0
            elif name.startswith("policy.compression."):
                multiplier = 3.0
            elif name.startswith(("policy.waypoint_head.", "policy.action_embedding")):
                multiplier = 0.5
        decay = args.weight_decay if param.ndim >= 2 and not name.endswith("bias") else 0.0
        groups.setdefault((multiplier, decay), []).append(param)
    return [{"params": params, "lr": args.learning_rate * multiplier, "weight_decay": decay}
            for (multiplier, decay), params in groups.items()]


def learning_rate_factor(step, total, warmup):
    if step < warmup:
        return step / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1 + math.cos(math.pi * progress))


def clip_partitioned_gradients(engine, max_norm):
    """DeepSpeed 0.9.5 omits BF16 norm/clipping; clip the reduced ZeRO2 shards."""
    torch.cuda.synchronize()
    norm = float(engine.optimizer.scaled_global_norm())
    if not math.isfinite(norm) or norm < 0:
        raise FloatingPointError("Nonfinite distributed gradient norm")
    scale = min(1.0, max_norm / (norm + 1e-6))
    for gradients in engine.optimizer.averaged_gradients.values():
        for gradient in gradients:
            gradient.mul_(scale)
    return norm


def save_checkpoint(engine, output, args, cursor):
    rng = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(), "python": random.getstate()}
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, rng)
    step = engine.global_steps
    metadata = {"arguments": vars(args), "cursor": cursor, "rng": states,
                "trainable_names": [n for n, p in engine.module.named_parameters() if p.requires_grad]}
    engine.save_checkpoint(str(output), tag=f"global_step{step}", client_state=metadata)
    if dist.get_rank() == 0:
        weights = {n: p.detach().cpu() for n, p in engine.module.named_parameters() if p.requires_grad}
        temporary = output / "trainable.tmp.pt"
        torch.save({"model": weights, "step": step,
                    "config": asdict(engine.module.policy.compression.config)}, temporary)
        temporary.replace(output / "trainable.pt")
        checkpoints = sorted((p for p in output.glob("global_step*")
                              if p.is_dir() and p.name.removeprefix("global_step").isdigit()),
                             key=lambda p: int(p.name.removeprefix("global_step")))
        for old in checkpoints[:-args.keep_checkpoints]:
            if old.resolve().parent != output.resolve():
                raise ValueError("Checkpoint directory is outside the output directory")
            shutil.rmtree(old)
    dist.barrier()
    if dist.get_rank() == 0:
        print(f"Saved checkpoint: {output}/global_step{step}", flush=True)


def resume_checkpoint(engine, args):
    directory = Path(args.resume)
    tag = directory.name if directory.name.startswith("global_step") else None
    path, state = engine.load_checkpoint(str(directory.parent if tag else directory), tag=tag,
                                         load_module_strict=False)
    if path is None:
        raise ValueError(f"Checkpoint not found: {args.resume}")
    mutable = {"resume", "output", "workers", "save_every", "keep_checkpoints", "local_rank"}
    for key, value in state["arguments"].items():
        if key not in mutable and value != getattr(args, key):
            raise ValueError(f"Resume requires unchanged {key}")
    names = [n for n, p in engine.module.named_parameters() if p.requires_grad]
    if state["trainable_names"] != names or len(state["rng"]) != dist.get_world_size():
        raise ValueError("Checkpoint trainable parameters or GPU count do not match")
    rng = state["rng"][dist.get_rank()]
    torch.set_rng_state(rng["cpu"])
    torch.cuda.set_rng_state(rng["cuda"])
    random.setstate(rng["python"])
    return state["cursor"]


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "data-index", "dataset-root", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--deepspeed", required=True, help="ZeRO Stage 2 JSON configuration")
    parser.add_argument("--stage", choices=["selector", "joint"], default="joint")
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--max-steps", type=int, help="Override epochs with a total number of optimizer updates")
    parser.add_argument("--batch-size", type=int, default=4, help="Samples per GPU per microbatch")
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--history-frames", type=int, default=5)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", help="Run directory or global_step directory")
    parser.add_argument("--init-from", help="trainable.pt from the preceding stage")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    args = parser.parse_args(argv)
    selector = args.stage == "selector"
    if args.epochs is None:
        args.epochs = 2.0 if selector else 0.5
    if args.learning_rate is None:
        args.learning_rate = 3e-4 if selector else 1e-5
    if args.warmup_ratio is None:
        args.warmup_ratio = 0.1 if selector else 0.05
    if min(args.batch_size, args.gradient_accumulation, args.keep_checkpoints, args.lora_rank) < 1:
        parser.error("Batch size, accumulation, checkpoint retention, and LoRA rank must be positive")
    if args.epochs <= 0 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("Epochs and max-steps must be positive")
    if args.workers < 0 or args.history_frames < 0 or args.save_every < 1 or not 0 <= args.warmup_ratio <= 1:
        parser.error("Invalid worker, history, checkpoint, or warm-up setting")
    if args.learning_rate <= 0 or args.max_grad_norm <= 0 or args.weight_decay < 0:
        parser.error("Learning rate and gradient norm must be positive; weight decay must be nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.cuda.set_device(args.local_rank)
    deepspeed.init_distributed(dist_backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Output is nonempty; choose a new directory or use --resume")
    dist.barrier()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    policy, encoder, tokenizer, processor = load_assets(args.assets, torch.device("cuda", args.local_rank))
    model = NavigationTrainingModel(policy, encoder, tokenizer, args.history_frames)
    configure_training(model, args)
    if args.init_from and not args.resume:
        initial = torch.load(args.init_from, map_location="cpu", weights_only=True)
        if initial["config"] != asdict(policy.compression.config):
            raise ValueError("Initialization checkpoint architecture does not match")
        expected = {n for n, p in model.named_parameters() if p.requires_grad}
        selector = {n for n in expected if n.startswith("policy.compression.view_selector.")}
        if set(initial["model"]) not in (expected, selector):
            raise ValueError("Initialization must contain all trained weights or the complete view selector")
        model.load_state_dict(initial["model"], strict=False)
    dataset = NavigationDataset(args.data_index, args.dataset_root, processor, args.history_frames)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=args.seed)
    batches = math.ceil(len(sampler) / args.batch_size)
    steps_per_epoch = math.ceil(batches / args.gradient_accumulation)
    total_steps = args.max_steps or math.ceil(args.epochs * steps_per_epoch)
    warmup_steps = math.ceil(args.warmup_ratio * total_steps)
    config = json.loads(Path(args.deepspeed).read_text())
    if config["zero_optimization"]["stage"] != 2 or not config.get("bf16", {}).get("enabled"):
        raise ValueError("This training entry point requires BF16 and ZeRO Stage 2")
    config.update(train_micro_batch_size_per_gpu=args.batch_size,
                  gradient_accumulation_steps=args.gradient_accumulation,
                  train_batch_size=args.batch_size * args.gradient_accumulation * world,
                  gradient_clipping=args.max_grad_norm)
    optimizer = torch.optim.AdamW(parameter_groups(model, args), foreach=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: learning_rate_factor(step, total_steps, warmup_steps))
    engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, lr_scheduler=scheduler, config=config)
    engine.checkpoint_engine = TrainableCheckpointEngine()
    cursor = resume_checkpoint(engine, args) if args.resume else {"epoch": 0, "batch": 0}
    if engine.global_steps >= total_steps:
        raise ValueError("The checkpoint has already reached the configured training duration")
    # Model initialization is synchronized by DeepSpeed; sampling differs across ranks.
    if not args.resume:
        random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
    if rank == 0:
        (output / "run.json").write_text(json.dumps({**vars(args), "world_size": world,
            "samples": len(dataset), "total_steps": total_steps, "deepspeed_config": config}, indent=2) + "\n")
        print(json.dumps({"stage": args.stage, "world_size": world, "samples": len(dataset),
            "global_batch_size": config["train_batch_size"], "total_steps": total_steps,
            "start_step": engine.global_steps,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}), flush=True)
    engine.train()
    totals = torch.zeros(4, device=engine.device)
    started, accumulated = time.monotonic(), 0
    while engine.global_steps < total_steps:
        epoch, offset = cursor["epoch"], cursor["batch"]
        sampler.set_epoch(epoch)
        # Skip consumed indices before reading images, including when resuming.
        indices = list(iter(sampler))[offset * args.batch_size:]
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=indices,
                            num_workers=args.workers, collate_fn=list, pin_memory=True,
                            multiprocessing_context="spawn" if args.workers else None,
                            generator=torch.Generator().manual_seed(args.seed + epoch + rank))
        remaining = len(loader)
        for index, samples in enumerate(loader):
            group_start = index - index % args.gradient_accumulation
            group_samples = min(args.batch_size * args.gradient_accumulation,
                                len(indices) - group_start * args.batch_size)
            boundary = (index + 1) % args.gradient_accumulation == 0 or index + 1 == remaining
            engine.set_gradient_accumulation_boundary(boundary)
            progress = engine.global_steps / max(1, total_steps - 1)
            model.policy.compression.view_selector.temperature = 1.5 - 0.8 * progress if args.stage == "selector" else 0.7
            loss, metrics = engine(samples)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss on rank {rank} at step {engine.global_steps + 1}")
            # DeepSpeed divides by the configured accumulation count. Normalize
            # the shorter final group to keep the last epoch update well scaled.
            engine.backward(loss * len(samples) * args.gradient_accumulation / group_samples)
            totals += metrics * len(samples)
            accumulated += len(samples)
            if boundary:
                grad_norm = clip_partitioned_gradients(engine, args.max_grad_norm)
            engine.step()
            cursor = {"epoch": epoch + 1, "batch": 0} if index + 1 == remaining else {"epoch": epoch, "batch": offset + index + 1}
            if not boundary:
                continue
            dist.all_reduce(totals)
            totals /= world * accumulated
            if rank == 0:
                record = dict(zip(("loss", "direction_loss", "distance_loss", "retention_loss"), totals.tolist()))
                record.update(step=engine.global_steps, epoch=epoch + (offset + index + 1) / batches,
                              lr=engine.get_lr(), grad_norm=grad_norm,
                              seconds=round(time.monotonic() - started, 2),
                              peak_memory_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3))
                with (output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            totals.zero_()
            started, accumulated = time.monotonic(), 0
            if engine.global_steps % args.save_every == 0 or engine.global_steps == total_steps:
                save_checkpoint(engine, output, args, cursor)
            if engine.global_steps >= total_steps:
                break
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
