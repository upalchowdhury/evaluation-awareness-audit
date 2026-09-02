#!/usr/bin/env bash
# One-time setup on macOS (Apple Silicon).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python3}
$PY -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Gated models (Llama, Gemma) additionally need:
#   huggingface-cli login     # after accepting each model's license on the Hub
echo
echo "Setup done. Recommended environment for every run on Apple Silicon:"
echo "  export PYTORCH_ENABLE_MPS_FALLBACK=1"
echo
echo "Smoke test (tiny model, validates the wiring):"
echo "  python run.py --config configs/smoke.yaml"
echo
echo "Full primary run:"
echo "  python run.py --config configs/default.yaml --model-config configs/models/qwen7b.yaml"
