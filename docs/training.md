# Training

## Assets

Install the package as described in the [README](../README.md), then configure
`assets.local.json` using [the example](../configs/assets.example.json).
`implementation_root` provides the `llamavid` EVA and Q-Former definitions.
The language decoder loads through Transformers. Asset paths resolve relative
to the JSON file; `bert_model` accepts a local path or cached model name.

The navigation checkpoint requires `adapter_config.json`, `adapter_model.bin`,
`mm_projector.bin`, and `non_lora_trainables.bin`. The optional view-selector
checkpoint contains `view_selector.bin`.

## Data

The training index is a JSON list with one-based frame numbers:

```json
[
  {"json": "scene/episode/merged_data.json", "frame": 6}
]
```

Each trajectory JSON contains `trajectory` (XYZ in the first three columns) and
`conversations` (the instruction in a `from: human` entry). Its episode directory
contains either:

- `rgb_imgs.tar`, with zero-based JPEG names such as `000000.frontcamera.jpg`;
- `rgb_imgs.tensor`, with preprocessed images shaped `(T*5, 3, H, W)`.

Camera order is **Front, Left, Right, Rear, Down**. Each sample uses the current
frame and up to five preceding frames; the next position supplies the waypoint
label.

## Optimization

Use the two-stage commands in the [README](../README.md). Both stages use four
GPUs, DeepSpeed ZeRO2, BF16, AdamW, cosine decay, and gradient checkpointing.
Each GPU processes four samples per microbatch and accumulates eight microbatches.
The data are shuffled and distributed across GPUs each epoch.

| Setting | Selector | Joint |
| --- | --- | --- |
| Epochs | 2 | 0.5 |
| Base learning rate | 3e-4 | 1e-5 |
| Warm-up ratio | 0.1 | 0.05 |
| View temperature | 1.5 → 0.7 | 0.7 |

The selector stage trains only the view selector. Joint training also optimizes
the compression/memory modules, trajectory encoder, visual projections, action
query, waypoint head, and decoder LoRA (rank 64, alpha 16). EVA, Q-Former, and
the base decoder remain frozen. Joint learning-rate multipliers are 10 for the
view selector, 3 for the remaining compression modules, and 0.5 for the waypoint
head and action query.

Override settings with `--epochs`, `--batch-size`, `--gradient-accumulation`, or
`--learning-rate`. `--max-steps` overrides epochs with total optimizer updates.
Set `CUDA_VISIBLE_DEVICES` to select four GPUs.

To resume, repeat the original command and add `--resume runs/joint`. Keep the
data, GPU count, stage, and optimization settings unchanged. A specific
`global_stepN` directory can also be passed to `--resume`.

## Outputs

`run.json` records the configuration; `metrics.jsonl` records training losses.
DeepSpeed checkpoints (`global_stepN/`) include optimized parameters, partitioned optimizer,
scheduler, data position, and per-GPU random states. Two checkpoints are kept
by default. Frozen weights are reloaded from the configured assets.
`trainable.pt` exports optimized parameters for stage transfer
or inference with the same configured pretrained assets.
