#!/usr/bin/env bash
# End-to-end: run the pipeline for one model, collect assets, compile the PDF.
#
#   ./scripts/build_paper.sh qwen15b                  # full default config
#   ./scripts/build_paper.sh qwen7b  --quick          # smoke sizes, real model
#   COMPARE=qwen15b,qwen7b ./scripts/build_paper.sh qwen7b
#
# Requires: .venv from scripts/setup_mac.sh, and `tectonic` for the PDF
# (brew install tectonic). Falls back to pdflatex+bibtex if tectonic is absent.
set -euo pipefail
cd "$(dirname "$0")/.."

SHORT=${1:?usage: build_paper.sh <model-short> [extra run.py flags]}
shift || true

source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

mkdir -p logs
LOG="logs/${SHORT}_$(date +%Y%m%d_%H%M%S).log"
echo "==> running pipeline for $SHORT (log: $LOG)"
python -u run.py --config configs/default.yaml \
  --model-config "configs/models/${SHORT}.yaml" "$@" 2>&1 | tee "$LOG"

echo "==> collecting paper assets"
python scripts/make_paper_assets.py --model "$SHORT" ${COMPARE:+--compare "$COMPARE"}

echo "==> compiling paper"
cd paper
if command -v tectonic >/dev/null 2>&1; then
  tectonic -X compile main.tex --outdir .
else
  pdflatex -interaction=nonstopmode main && bibtex main \
    && pdflatex -interaction=nonstopmode main \
    && pdflatex -interaction=nonstopmode main
fi
echo "==> done: paper/main.pdf"
