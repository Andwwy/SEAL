#!/usr/bin/env bash
# SEAL end-to-end on RunPod for DeepSeek-R1-Distill-Qwen-1.5B + GSM8K.
#
#   Phase 1  Build the steering vector from MATH train (one-time, per model)
#   Phase 2  Baseline eval on GSM8K  (vLLM, fast)
#   Phase 3  Steered  eval on GSM8K  (custom HF generate — the slow part)
#
# Usage:
#   cp .env.example .env && edit HF_TOKEN
#   bash setup_runpod.sh
#   bash scripts/runpod_gsm.sh                 # full GSM8K test set (1319)
#   GSM_MAX_EXAMPLES=100 bash scripts/runpod_gsm.sh   # quick smoke run first
#
# Re-running is cheap: the steering vector and any completed phase whose output
# already exists are skipped.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# --- Config (override via env) ---
if [[ -f .env ]]; then set -a; source .env; set +a; fi
: "${GPU:=0}"
: "${HF_HOME:=/workspace/hf_cache}"
: "${MODEL:=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
: "${MAX_TOKENS:=10000}"          # generation budget per problem
: "${MATH_TRAIN_CAP:=3000}"       # how many MATH-train problems to generate for the vector
: "${VEC_SAMPLES:=500}"           # correct/incorrect traces used to build the vector
: "${STEER_LAYER:=20}"
: "${STEER_COEF:=-1.0}"
: "${BATCH_SIZE:=25}"             # steering eval batch size (HF generate)
: "${GSM_MAX_EXAMPLES:=0}"        # 0 = full GSM8K test set (1319); else cap

export HF_HOME CUDA_VISIBLE_DEVICES="$GPU"
[[ -n "${HF_TOKEN:-}" ]] && export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

MODEL_TAG="$(basename "$MODEL")"
VEC_BASE="results/MATH_train/${MODEL_TAG}/baseline_${MAX_TOKENS}"
VEC_FILE="${VEC_BASE}/vector_${VEC_SAMPLES}_${VEC_SAMPLES}/layer_${STEER_LAYER}_transition_reflection_steervec.pt"

GSM_MAX_FLAG=""
[[ "$GSM_MAX_EXAMPLES" != "0" ]] && GSM_MAX_FLAG="--max_examples ${GSM_MAX_EXAMPLES}"

echo "=================================================================="
echo " MODEL=$MODEL  GPU=$GPU  MAX_TOKENS=$MAX_TOKENS"
echo " steering: layer=$STEER_LAYER coef=$STEER_COEF  GSM cap=${GSM_MAX_EXAMPLES:-full}"
echo " steering vector -> $VEC_FILE"
echo "=================================================================="

# ------------------------------------------------------------------ #
# Phase 1: steering vector from MATH train
# ------------------------------------------------------------------ #
if [[ -f "$VEC_FILE" ]]; then
    echo "[phase1] Steering vector already exists, skipping."
else
    if [[ -f "${VEC_BASE}/math_eval.jsonl" ]]; then
        echo "[phase1a] MATH-train baseline traces already exist, skipping."
    else
        echo "[phase1a] Generating MATH-train baseline traces (cap=$MATH_TRAIN_CAP)..."
        python eval_MATH_vllm.py \
            --model_name_or_path "$MODEL" \
            --save_dir "$VEC_BASE" \
            --max_tokens "$MAX_TOKENS" \
            --max_examples "$MATH_TRAIN_CAP" \
            --use_chat_format \
            --dataset "MATH_train" \
            --remove_bos
    fi

    # Only keep the steering layer's hidden states (shrinks hidden.pt ~num_layers x);
    # the layer-$STEER_LAYER vector is identical to the all-layers version.
    echo "[phase1b] Extracting hidden states (incorrect, layer $STEER_LAYER only)..."
    python hidden_analysis.py \
        --model_path "$MODEL" \
        --data_path data/MATH/train.jsonl \
        --data_dir "$VEC_BASE" \
        --type incorrect --start 0 --sample "$VEC_SAMPLES" \
        --keep_layers "$STEER_LAYER"

    echo "[phase1b] Extracting hidden states (correct, layer $STEER_LAYER only)..."
    python hidden_analysis.py \
        --model_path "$MODEL" \
        --data_path data/MATH/train.jsonl \
        --data_dir "$VEC_BASE" \
        --type correct --start 0 --sample "$VEC_SAMPLES" \
        --keep_layers "$STEER_LAYER"

    echo "[phase1c] Building steering vector (layer $STEER_LAYER)..."
    python vector_generation.py \
        --data_dir "$VEC_BASE" \
        --prefixs "correct_0_${VEC_SAMPLES}" "incorrect_0_${VEC_SAMPLES}" \
        --layers "$STEER_LAYER" \
        --save_prefix "${VEC_SAMPLES}_${VEC_SAMPLES}"
fi

# ------------------------------------------------------------------ #
# Phase 2: baseline GSM8K (vLLM)
# ------------------------------------------------------------------ #
echo "[phase2] Baseline GSM8K eval..."
python eval_MATH_vllm.py \
    --model_name_or_path "$MODEL" \
    --save_dir "results/GSM/${MODEL_TAG}/baseline_${MAX_TOKENS}" \
    --max_tokens "$MAX_TOKENS" \
    --use_chat_format \
    --dataset "GSM" \
    --remove_bos \
    $GSM_MAX_FLAG

# ------------------------------------------------------------------ #
# Phase 3: steered GSM8K (custom HF generate)
# ------------------------------------------------------------------ #
echo "[phase3] Steered GSM8K eval (coef=$STEER_COEF)..."
python eval_MATH_steering.py \
    --model_name_or_path "$MODEL" \
    --save_dir "results/GSM/${MODEL_TAG}/steer-from-MATH-${MAX_TOKENS}" \
    --max_tokens "$MAX_TOKENS" \
    --use_chat_format \
    --batch_size "$BATCH_SIZE" \
    --dataset "GSM" \
    --remove_bos \
    --steering \
    --steering_vector "$VEC_FILE" \
    --steering_layer "$STEER_LAYER" \
    --steering_coef "$STEER_COEF" \
    --start 0 \
    $GSM_MAX_FLAG

# ------------------------------------------------------------------ #
# Phase 4: visualize (accuracy + token-reduction) for presenting
# ------------------------------------------------------------------ #
echo "[phase4] Building result visualizations..."
python visualize_results.py \
    --baseline_dir "results/GSM/${MODEL_TAG}/baseline_${MAX_TOKENS}" \
    --steered_dir  "results/GSM/${MODEL_TAG}/steer-from-MATH-${MAX_TOKENS}" \
    --model "$MODEL" \
    --dataset "GSM8K" \
    --output_dir "results/GSM/${MODEL_TAG}/figures" || \
    echo "[phase4] viz failed (non-fatal) — check matplotlib install"

echo "=================================================================="
echo " DONE."
echo "   accuracy   : results/GSM/${MODEL_TAG}/*/metrics.json"
echo "   figures    : results/GSM/${MODEL_TAG}/figures/  (dashboard.png + summary.md)"
echo "=================================================================="
grep -r '"acc"' "results/GSM/${MODEL_TAG}/" 2>/dev/null || true
