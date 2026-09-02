#!/bin/bash
# Sequential exp11 (distance-matched 2x2), one model at a time. Args = model shorts.
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
for m in "$@"; do
  echo "=== $m exp11 start $(date) ==="
  python -u run.py --config configs/default.yaml --model-config "configs/models/$m.yaml" --exps 11 \
    > "logs/${m}_exp11.log" 2>&1 || echo "!!! $m exp11 FAILED"
  echo "=== $m exp11 done $(date) ==="
done
echo "ALL exp11 RUNS DONE $(date)"
