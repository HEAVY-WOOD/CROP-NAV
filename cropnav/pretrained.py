"""Load locally supplied backbone assets and compatible navigation components.

The existing EVA and Q-Former implementation is an external dependency; this
module does not copy its architecture or invoke its training pipeline.
"""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn

from .encoding import GoalConditionedEncoder
from .model import CROPNav
from .policy import CROPNavPolicy, WaypointHead


def _subtree(state, prefix):
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


@torch.no_grad()
def _merge_language_adapter(model, checkpoint):
    """Merge linear LoRA pairs directly, avoiding reinitialization of the 7B model."""
    config = json.loads((checkpoint / "adapter_config.json").read_text())
    if (config["peft_type"] != "LORA" or config.get("bias", "none") != "none"
            or config.get("modules_to_save") or config.get("fan_in_fan_out", False)):
        raise ValueError("The asset loader supports linear LoRA with bias=none and no saved extra modules")
    weights = torch.load(checkpoint / "adapter_model.bin", map_location="cpu", weights_only=True)
    weights = {k.removeprefix("base_model.model."): v for k, v in weights.items()}
    used = set()
    for key in sorted(weights):
        if not key.endswith(".lora_A.weight"):
            continue
        name = key.removesuffix(".lora_A.weight")
        b_key = name + ".lora_B.weight"
        layer = model.get_submodule(name)
        if not isinstance(layer, nn.Linear):
            raise ValueError(f"Expected a linear adapter target: {name}")
        # PEFT 0.4 keeps linear adapter parameters in FP32, including when the
        # base decoder uses BF16. Match that accumulation before rounding once.
        a = weights[key].to(device=layer.weight.device, dtype=torch.float32)
        b = weights[b_key].to(device=layer.weight.device, dtype=torch.float32)
        if a.shape[0] != config["r"]:
            raise ValueError(f"Adapter rank mismatch in {name}")
        layer.weight.add_((b @ a) * (config["lora_alpha"] / config["r"]))
        used.update([key, b_key])
    if not used or used != set(weights):
        raise ValueError("Adapter contains unsupported or incomplete weight keys")
    print(f"Merged {len(used) // 2} pretrained linear adapters", flush=True)


def load_assets(config_path, device="cpu", dtype=torch.bfloat16):
    """Return (policy, encoder, language_tokenizer, image_processor).

    Paths are configured by the caller. All pretrained loads are local-only.
    Compatible layers are mapped explicitly; the obsolete learned slot gate is
    excluded because IGTA supplies the paper-defined parameter-free prior.
    """
    from transformers import BertConfig, BertTokenizer, LlamaForCausalLM, AutoTokenizer

    config_path = Path(config_path).resolve()
    assets = json.loads(config_path.read_text(encoding="utf-8"))
    for name in ("implementation_root", "language_model", "navigation_checkpoint",
                 "vision_checkpoint", "image_processor", "view_selector_checkpoint"):
        if assets.get(name):
            path = Path(assets[name]).expanduser()
            assets[name] = str(path if path.is_absolute() else config_path.parent / path)
            if not Path(assets[name]).exists():
                raise FileNotFoundError(f"Missing asset {name}: {assets[name]}")
    sys.path.insert(0, assets["implementation_root"])
    from llamavid.model.multimodal_encoder.eva_vit import EVAVisionTowerLavis
    from llamavid.model.qformer import BertLMHeadModel

    checkpoint = Path(assets["navigation_checkpoint"])
    state = {}
    for filename in ("mm_projector.bin", "non_lora_trainables.bin"):
        weights = torch.load(checkpoint / filename, map_location="cpu", weights_only=True)
        state.update({k.removeprefix("base_model.model."): v for k, v in weights.items()})

    print("Loading pretrained language decoder", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(assets["language_model"], use_fast=False, local_files_only=True)
    language = LlamaForCausalLM.from_pretrained(
        assets["language_model"], torch_dtype=dtype, low_cpu_mem_usage=True, local_files_only=True,
    )
    if "model.embed_tokens.weight" in state:
        language.resize_token_embeddings(state["model.embed_tokens.weight"].shape[0])
    language_keys = language.state_dict().keys()
    language.load_state_dict({k: v for k, v in state.items() if k in language_keys}, strict=False)
    # CPU BF16 adapter merging is extremely slow on CPUs without BF16 kernels.
    # Merge on the requested device before assembling the remaining modules.
    language.to(device=device)
    print(f"Merging pretrained language adapter on {device}", flush=True)
    _merge_language_adapter(language, checkpoint)
    decoder = language.model

    print("Loading pretrained visual encoder and Q-Former", flush=True)
    vision_args = SimpleNamespace(mm_vision_select_layer=-2, mm_vision_select_feature="patch")
    vision = EVAVisionTowerLavis(assets["vision_checkpoint"], assets["image_processor"], vision_args)
    bert_name = assets.get("bert_model", "bert-base-uncased")
    bert_tokenizer = BertTokenizer.from_pretrained(bert_name, truncation_side="left", local_files_only=True)
    bert_tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    bert_config = BertConfig.from_pretrained(bert_name, local_files_only=True)
    bert_config.encoder_width = 1408
    bert_config.add_cross_attention = True
    bert_config.cross_attention_freq = 2
    bert_config.vocab_size = state["model.vlm_att_encoder.bert.embeddings.word_embeddings.weight"].shape[0]
    qformer = BertLMHeadModel(bert_config)
    qformer.cls = None
    qformer_state = _subtree(state, "model.vlm_att_encoder.")
    # Older serialization omitted the deterministic position-id buffer.
    for name, value in qformer.state_dict().items():
        if name.endswith("position_ids") and name not in qformer_state:
            qformer_state[name] = value
    qformer.load_state_dict(qformer_state, strict=True)

    def linear(name):
        values = _subtree(state, f"model.{name}.")
        weight = values["weight"]
        layer = nn.Linear(weight.shape[1], weight.shape[0], bias="bias" in values)
        layer.load_state_dict(values, strict=True)
        return layer

    projection = nn.Sequential(linear("mm_projector.0"), nn.GELU(), linear("mm_projector.2"))
    norm = nn.LayerNorm(1408)
    norm.load_state_dict(_subtree(state, "model.vlm_att_ln."), strict=True)
    encoder = GoalConditionedEncoder(
        vision, qformer, bert_tokenizer, state["model.vlm_att_query"],
        linear("vlm_att_projector"), linear("vlm_att_key_projector"),
        linear("vlm_att_val_projector"), projection, vision_norm=norm,
    )
    core = CROPNav()
    mapped = {
        "memory": "longfly_history_enhancer.shic.",
        "trajectory.res_mlp": "longfly_history_enhancer.ste.res_mlp.",
        "trajectory.projection": "longfly_history_enhancer.ste_to_hidden.",
        "memory_projection": "longfly_history_enhancer.shic_to_hidden.",
        "token_selector": "longfly_history_enhancer.policy_gate.",
    }
    for destination, prefix in mapped.items():
        module = core.get_submodule(destination)
        # Select only the explicitly supported component keys, then require all.
        module.load_state_dict({key: state[prefix + key] for key in module.state_dict()}, strict=True)
    if assets.get("view_selector_checkpoint"):
        selector = Path(assets["view_selector_checkpoint"])
        selector = selector / "view_selector.bin" if selector.is_dir() else selector
        core.view_selector.load_state_dict(torch.load(selector, map_location="cpu", weights_only=True), strict=True)
    head = WaypointHead(core.config.hidden_dim)
    head.features.load_state_dict(_subtree(state, "waypoints_fc."), strict=True)
    head.output.load_state_dict(_subtree(state, "waypoints_output."), strict=True)
    policy = CROPNavPolicy(core, decoder, waypoint_head=head, action_embedding=state["waypoint_emb.weight"])
    del state, language
    # Keep optimized method parameters in FP32; run frozen backbones in BF16/FP32.
    encoder = encoder.to(device=device, dtype=dtype).requires_grad_(False).eval()
    policy.decoder = policy.decoder.to(device=device, dtype=dtype).requires_grad_(False)
    policy.compression.to(device=device, dtype=torch.float32)
    policy.waypoint_head.to(device=device, dtype=torch.float32)
    policy.action_embedding.data = policy.action_embedding.data.to(device=device, dtype=torch.float32)
    return policy, encoder, tokenizer, vision.image_processor
