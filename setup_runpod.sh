#!/usr/bin/env bash
# One-time environment setup for running SEAL on a RunPod A100 pod.
#
# Usage:
#   cp .env.example .env   # then edit .env to add your HF_TOKEN
#   bash setup_runpod.sh
#
# Assumes a CUDA 12.x PyTorch base image (e.g. runpod/pytorch:2.x-cuda12.1).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# --- Load .env if present ---
if [[ -f .env ]]; then
    set -a; source .env; set +a
    echo "[setup] Loaded .env"
fi

: "${HF_HOME:=/workspace/hf_cache}"
export HF_HOME
mkdir -p "$HF_HOME"
echo "[setup] HF_HOME=$HF_HOME"

# --- Python deps ---
echo "[setup] Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "[setup] Installing requirements.txt..."
pip install -r requirements.txt

# latex2sympy2 must be installed WITHOUT its deps so it can't downgrade
# antlr4-python3-runtime (sympy 1.12's parse_latex needs antlr 4.11.x).
echo "[setup] Installing latex2sympy2 (--no-deps)..."
pip install --no-deps latex2sympy2==1.9.1

# --- Hugging Face auth ---
if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "[setup] Logging in to Hugging Face Hub..."
    # huggingface_hub reads HF_TOKEN automatically, but cache the login too.
    python - <<'PY'
import os
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
print("[setup] HF login OK")
PY
else
    echo "[setup] No HF_TOKEN set — relying on anonymous access (fine for the 1.5B model)."
fi

# --- Sanity checks ---
echo "[setup] Verifying GPU + key imports..."
python - <<'PY'
import torch
print(f"[setup] torch {torch.__version__}  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[setup] GPU: {torch.cuda.get_device_name(0)}  count={torch.cuda.device_count()}")
import vllm, transformers
print(f"[setup] vllm {vllm.__version__}  transformers {transformers.__version__}")
# Grading import chain used by get_math_results.py
from eval_math_rule.evaluation.grader import math_equal
from eval_math_rule.evaluation.parser import extract_answer, strip_string
print("[setup] eval_math_rule import OK")
PY

echo "[setup] Done. Next: bash scripts/runpod_gsm.sh"
