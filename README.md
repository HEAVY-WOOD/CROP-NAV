# CROP-NAV

**Coarse-to-Fine Token Compression for Long-Horizon UAV Vision-and-Language Navigation**

CROP-NAV compresses five-view observations for waypoint prediction through
adaptive view selection, navigation-conditioned spatial-token selection, and
instability-guided recurrent memory.

## Method

- **Coarse selection:** retain two horizontal views and the downward view.
- **Fine selection:** retain spatial tokens using goal and trajectory context.
- **IGTA memory:** weight visual evidence and update a five-direction slot bank.

The policy combines current visual evidence, recurrent memory, trajectory, and
language to predict a 3D direction and distance.

## Installation

Python 3.10, PyTorch 2.0.1 with CUDA 11.8, and four NVIDIA GPUs are used for training.

```bash
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -e '.[train,backbones]'
cp configs/assets.example.json assets.local.json
```

Set the pretrained model paths in `assets.local.json`. Model weights and datasets
are supplied separately; see [training setup](docs/training.md).

## Training

```bash
bash scripts/train.sh \
  --assets assets.local.json \
  --data-index /path/to/trainset.json \
  --dataset-root /path/to/image-dataset \
  --output runs/selector --stage selector

bash scripts/train.sh \
  --assets assets.local.json \
  --data-index /path/to/trainset.json \
  --dataset-root /path/to/image-dataset \
  --output runs/joint --stage joint \
  --init-from runs/selector/trainable.pt
```

Training uses DeepSpeed ZeRO Stage 2 and BF16 on GPUs `0,1,2,3`, with batch size 4
per GPU and gradient accumulation 8 (global batch 128).
See [training options](docs/training.md) for stage settings and checkpoint
resumption, or [the integration guide](docs/integration.md)
for model inputs and online inference.

## Structure

- `cropnav/`: method modules, pretrained-asset loading, and training entry point.
- `configs/`: pretrained assets and ZeRO configuration.
- `scripts/`: four-GPU training launcher.
- `docs/`: training and integration instructions.

## Citation and license

See [CITATION.cff](CITATION.cff), [Apache-2.0](LICENSE), and [NOTICE](NOTICE).
Externally supplied models and weights retain their own licenses.
