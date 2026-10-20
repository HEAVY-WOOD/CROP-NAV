#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false
exec "${PYTHON_BIN:-python}" -m torch.distributed.run \
  --nproc_per_node=4 --master_port="${MASTER_PORT:-29520}" \
  --module cropnav.train --deepspeed configs/zero2.json "$@"
