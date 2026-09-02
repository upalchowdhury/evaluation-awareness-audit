#!/bin/bash
# v2 queue: one model at a time. Usage: EXPS=0,1,2,7,8,10,11 scripts/seq_v2.sh qwen15b qwen05b ...
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p results_v2 logs
[ -f results_v2/common_contents.json ] || cp results/common_contents.json results_v2/
EXPS="${EXPS:-0,1,2,7,8,10,11}"
for m in "$@"; do
  echo "=== $m exps $EXPS start $(date) ==="
  python -u run.py --config configs/v2.yaml --model-config "configs/models/$m.yaml" --exps "$EXPS" \
    > "logs/v2_${m}_$(echo $EXPS | tr ',' '_').log" 2>&1 || echo "!!! $m FAILED"
  echo "=== $m done $(date) ==="
done
echo "ALL V2 RUNS DONE $(date)"
