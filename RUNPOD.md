# Running SEAL on RunPod (A100) — R1-Distill-1.5B / GSM8K

This guide adapts the official [SEAL](https://github.com/VITA-Group/SEAL) repo to run on a
RunPod GPU pod, with a Hugging Face token for model download, targeting
**DeepSeek-R1-Distill-Qwen-1.5B on GSM8K** as the first experiment.

## What SEAL does (1-minute version)
SEAL reduces "overthinking" in reasoning models by steering hidden states away
from reflection/transition steps. The pipeline is:

1. **Build a steering vector** from MATH-train traces (one-time, per model).
2. **Baseline** eval (no steering) — vLLM, fast.
3. **Steered** eval — custom HF `generate()` with the vector applied at a layer.

The steering vector is extracted from MATH (as in the paper) and *applied* to
GSM8K — you don't need GSM-specific vectors.

---

## 1. Launch the pod
- GPU: **1× A100 80GB** (the 1.5B model needs only ~3 GB weights + KV cache for a
  12k context; even A100 40GB or a 24 GB card works — see cost notes).
- Template: a CUDA 12.1 PyTorch image, e.g. `runpod/pytorch:2.4.0-py3.11-cuda12.1`.
- Attach a **persistent network volume mounted at `/workspace`** so model
  downloads and results survive pod restarts.

## 2. Clone + configure
```bash
cd /workspace
git clone https://github.com/VITA-Group/SEAL.git
cd SEAL
cp .env.example .env
# edit .env: set HF_TOKEN=hf_xxx  (HF_HOME defaults to /workspace/hf_cache)
```

## 3. Install
```bash
bash setup_runpod.sh
```
This installs `requirements.txt`, installs `latex2sympy2 --no-deps` (so it can't
downgrade antlr), logs into HF, and runs import/GPU sanity checks.

## 4. Run — smoke test first, then full
```bash
# Quick 100-example smoke test (builds the vector, ~30-45 min total):
GSM_MAX_EXAMPLES=100 bash scripts/runpod_gsm.sh

# Full GSM8K test set (1319 problems):
bash scripts/runpod_gsm.sh
```

Results / accuracy:
- Baseline: `results/GSM/DeepSeek-R1-Distill-Qwen-1.5B/baseline_10000/metrics.json`
- Steered:  `results/GSM/DeepSeek-R1-Distill-Qwen-1.5B/steer-from-MATH-10000/.../metrics.json`

## 5. Visualizations (auto-generated)
Phase 4 of `runpod_gsm.sh` runs `visualize_results.py`, which reproduces SEAL's
headline result — **accuracy held/up while reasoning tokens drop** — into
`results/GSM/<model>/figures/`:
- `dashboard.png` — 2×2 summary for slides (accuracy, generation length,
  length distribution, accuracy-vs-tokens efficiency frontier)
- `accuracy_comparison.png`, `token_usage.png`, `token_distribution.png`
- `summary.md` / `summary.json` — the numbers + a paste-ready table

Run it standalone any time (CPU-only, no GPU needed):
```bash
python visualize_results.py \
  --baseline_dir results/GSM/DeepSeek-R1-Distill-Qwen-1.5B/baseline_10000 \
  --steered_dir  results/GSM/DeepSeek-R1-Distill-Qwen-1.5B/steer-from-MATH-10000 \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --dataset GSM8K \
  --output_dir results/GSM/DeepSeek-R1-Distill-Qwen-1.5B/figures
```
Token counts use the model tokenizer; omit `--model` to fall back to word counts.

### Knobs (env vars)
| Var | Default | Meaning |
|-----|---------|---------|
| `GSM_MAX_EXAMPLES` | `0` (=full 1319) | cap GSM eval size |
| `MAX_TOKENS` | `10000` | generation budget per problem |
| `MATH_TRAIN_CAP` | `3000` | MATH-train problems generated for the vector |
| `STEER_LAYER` | `20` | layer the vector is added at |
| `STEER_COEF` | `-1.0` | steering strength (negative = suppress reflection) |
| `BATCH_SIZE` | `25` | steering-eval batch size |

---

## What was changed for RunPod
- **`requirements.txt`** — consolidated, version-pinned deps (none existed before).
- **`setup_runpod.sh`** — installer + HF login + sanity checks; handles the
  `latex2sympy2`/antlr version conflict.
- **`.env.example`** — `HF_TOKEN`, `HF_HOME`, `GPU`.
- **`scripts/runpod_gsm.sh`** — a single runnable GSM8K pipeline. The stock
  `scripts/*.sh` prefix the Python commands with `echo` (dry-run only) and target
  MATH500/LiveCode; this script actually runs all three phases for GSM8K, skips
  the vector build if it already exists, and caps MATH-train generation.
- **`visualize_results.py`** — plots accuracy + token-reduction (baseline vs
  steered) for presenting; the stock repo only writes `metrics.json`.
- No core logic changed — GSM8K is already supported (`--dataset GSM`).

---

## Cost estimate (A100 80GB @ ~$1.74/hr community / ~$1.89–2.17 secure)

Model is tiny (1.5B), so cost is dominated by **how many tokens are generated**.
The biggest and least predictable cost is Phase 3 (steered eval) because it uses
plain HuggingFace `generate()` (no continuous batching), so each batch is gated
by its longest sequence.

Assumptions: avg ~2k output tok/problem on GSM8K, ~4k on MATH-train; vLLM ~4k
tok/s aggregate; HF-generate steering ~0.8–1.5k tok/s aggregate at batch 25.

| Phase | Work | Est. wall time |
|-------|------|----------------|
| Setup + model/dataset download | one-time | ~0.2–0.4 hr |
| **P1** vector build (MATH-train 3000 vLLM + 1000 HF forward passes) | one-time/model | ~1.0–1.5 hr |
| **P2** baseline GSM8K (1319, vLLM) | per run | ~0.2–0.3 hr |
| **P3** steered GSM8K (1319, HF generate) | per run | ~2–4 hr |
| **Total — full first run** | | **~4–6 hr** |

**Full first run (everything, 1319 examples): ≈ 4–6 hr ⇒ ~$7–11** at $1.74/hr
(~$8–13 on secure cloud). Budget **~$15** to be safe with download/idle overhead.

**100-example smoke test** (still builds the vector once): ≈ **$2–4**.

**Subsequent GSM runs** (vector already cached, P2+P3 only): ≈ 2.5–4.5 hr ⇒ **~$5–8**.

### Cost levers
- The vector build (P1) is **one-time per model** — amortized across all later runs/benchmarks.
- A **cheaper GPU** fits this 1.5B model fine: A100 40GB (~$1.19/hr), L40S/A40
  (~$0.8–1.0/hr), or RTX 4090 24GB (~$0.34–0.69/hr) would cut cost substantially —
  the A100 80GB is overkill here. Use it only if you'll scale to bigger R1-Distill sizes.
- Lower `MAX_TOKENS` (e.g. 4000) if GSM8K completions rarely need 10k — large savings on P3.
- Start with `GSM_MAX_EXAMPLES=100` to validate the whole pipeline before paying for the full set.

> Estimates assume on-demand pricing as of mid-2026; check the live rate in the
> RunPod console at deploy time. Sources:
> [RunPod pricing](https://www.runpod.io/pricing),
> [ComputePrices](https://computeprices.com/providers/runpod).
