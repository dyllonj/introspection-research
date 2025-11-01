# Introspect

Introspect is a zero-fine-tuning evaluation toolkit for probing causal concept representations in open-weight decoder LLMs. It standardises how to build concept vectors, inject them into residual streams, and score downstream behavioural shifts across four complementary benchmark tasks (A–D).

## Feature highlights
- **Deterministic concept vector extraction.** `introspect.src.vectors` builds concept vectors with seeded sampling, caches them under `introspect/results/vectors/`, and records metadata to ease reuse across experiments.
- **Reusable activation injection primitives.** Shared adapters, hook registration, and injection specs provide a consistent surface for manipulating the residual stream across supported architectures.
- **Task suite with automated grading.** Tasks A–D cover detection, thought/text alignment, intent flipping, and activation control. Each script logs JSONL records with parsed responses, grading signals, and seeds for reproducibility.
- **Sweep and plotting utilities.** A single CLI orchestrates multi-model sweeps, produces task-specific JSONL outputs, and renders publication-ready plots.

## Architecture overview
```
introspect/
├─ registry/models.yaml     # model id → adapter/dtype metadata
├─ concepts/words.yaml      # target & baseline concept vocabularies
├─ src/
│  ├─ adapters/             # Base protocol + concrete Llama/Mistral/Falcon/NeoX/Qwen implementations
│  ├─ vectors.py            # concept vector builder, caching helpers, CLI
│  ├─ hooks.py, inject.py   # residual hook registration + injection runtime
│  ├─ eval_common.py        # registry loading, config parsing, vector reuse
│  ├─ eval_[A-D]_.py        # task drivers with grading + JSONL logging
│  ├─ grading.py, prompts.py# prompt templates and rule-based judges
│  ├─ sweep.py              # multi-model/task orchestrator & plotting trigger
│  └─ plotting.py           # Task A–D visualisations saved per model
├─ results/                 # cached vectors, task logs, aggregate plots
└─ tests/                   # unit + smoke coverage for adapters, vectors, tasks
```

## Installation
1. Ensure Python 3.10 or newer is available.
2. Create and activate a virtual environment (`python -m venv .venv && source .venv/bin/activate`).
3. Install the package in editable mode with core dependencies:
   ```bash
   pip install -e .
   ```
4. (Optional) Install Hydra integration for config-driven runs:
   ```bash
   pip install -e .[hydra]
   ```

## Dependencies & environment notes
- Core requirements are pinned in `pyproject.toml` (`torch>=2.2`, `transformers>=4.43`, `accelerate>=0.30`, plus numpy/pandas/matplotlib/pyyaml/tqdm/pytest/ruff).
- GPU runs default to `bfloat16` when supported; CPU-only environments automatically fall back to `float32`.
- Set the `HF_HOME` environment variable if you prefer a custom Hugging Face cache location when downloading models.

## Determinism & safety practices
- Every CLI accepts `--seed` and enables deterministic PyTorch kernels by default; pass `--non-deterministic` only when exact reproducibility is not required.
- Control conditions are logged alongside injections for all tasks. Task A additionally evaluates negative (`-v`) and random vectors as ablations, while Tasks B–C log both control and injected trials before grading.
- JSONL records include runtime metadata (model id, adapter, device map, dtype, seed, library versions) so experiments can be audited post-hoc.

## Configuration, caches, and outputs
- **Model registry:** `introspect/registry/models.yaml` maps model identifiers to adapter classes, default dtypes, and context lengths.
- **Concept vocabulary:** `introspect/concepts/words.yaml` stores `targets` and `baselines` used to build contrastive vectors. Limit subsets with CLI flags such as `--limit-targets`.
- **CLI configuration files:** All CLIs support `--config`, `--config-dir`, and `--config-name` for Hydra/YAML overrides.
- **Vector cache:** Concept vectors are stored as `.pt` files beneath `introspect/results/vectors/<model>/<layer>/<word>.pt` with cached metadata.
- **Task results:** Evaluation scripts default to `introspect/results/task_*.jsonl`; the sweep utility nests outputs under `results/<model_slug>/task_X.jsonl` and creates `plots/` subfolders.
- **Plots:** `introspect/src/plotting.py` produces Task A heatmaps, Task B/C bar charts, and Task D activation curves saved as PNG/SVG files.

## Verified CLI workflow
All commands run from the repository root with the virtual environment active.

```bash
# 1) Build vectors for 10 words on Llama-3-8B at layers [8,16,24,32]
python -m introspect.src.vectors \
  --model meta-llama/Llama-3-8b \
  --layers 8 16 24 32 \
  --words-file introspect/concepts/words.yaml \
  --limit-targets 10
# → caches vectors under introspect/results/vectors/ and records metadata per (model, layer, word).

# 2) Task A sweep (layers×alpha) with 20 concepts
python -m introspect.src.eval_A_injected_report \
  --model meta-llama/Llama-3-8b \
  --layers 0 8 16 24 32 \
  --alphas 1 2 4 8 16 \
  --n-concepts 20 \
  --results-path introspect/results/task_A.jsonl
# → appends control, negative, random, and injected trials to JSONL with grading summaries.

# 3) Task B with multiple-choice grading
python -m introspect.src.eval_B_thoughts_vs_text \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --layers 16 24 \
  --alpha 4 \
  --n-trials 50 \
  --results-path introspect/results/task_B.jsonl
# → logs thought, repetition, and multiple-choice outcomes for both control and injected conditions.

# 4) Task C prefill / intent flip
python -m introspect.src.eval_C_prefill_intent \
  --model tiiuae/falcon-7b \
  --layers 12 24 \
  --alpha 6 \
  --n-trials 50 \
  --results-path introspect/results/task_C.jsonl
# → records baseline vs. injected intent responses per concept.

# 5) Task D intentional control curves
python -m introspect.src.eval_D_intentional_control \
  --model EleutherAI/gpt-neox-20b \
  --layers 0 8 16 24 32 \
  --n-concepts 20 \
  --results-path introspect/results/task_D.jsonl
# → captures per-layer cosine deltas, peaks, and area-under-curve statistics.

# 6) Automated sweep + plots (optional)
python -m introspect.src.sweep \
  --models meta-llama/Llama-3-8b mistralai/Mistral-7B-Instruct-v0.2 \
  --tasks A B C D \
  --layer-grid 10 \
  --task-a-concepts 20 \
  --task-b-trials 50 \
  --task-c-trials 50 \
  --task-d-concepts 20 \
  --run-plots
# → writes per-model task outputs beneath results/<model_slug>/ and saves plots/<...>.png.
```

## Plot review
Regenerate plots from existing JSONL logs by reusing the sweep CLI in plot-only mode:
```bash
python -m introspect.src.sweep \
  --models meta-llama/Llama-3-8b \
  --plot-only \
  --results-root introspect/results
```
Replace the model list with any identifiers present in the results directory. The command skips new trials, calls `generate_all_plots`, and writes heatmaps, bar charts, and cosine curves into `results/<model_slug>/plots/` for inspection and paper figures.

## Contributing & developer notes
- **Adapters:** Implement `BaseModelAdapter` in `introspect/src/adapters/` when adding new architectures. Ensure `load`, `layer_module`, `register_residual_hook`, `tokens_for_spans`, and `generate` obey the protocol, call `seed_everything`, and add the class name to `ADAPTER_REGISTRY`.
- **Task extensions:** Follow existing task scripts as templates—reuse `eval_common.ensure_vector` to respect caching and metadata, log control/injected conditions, and register new grading utilities in `grading.py`.
- **Configuration presets:** Store shared run settings as YAML in a configs directory and pass them via `--config-dir`/`--config-name` for reproducible sweeps.
- **Testing:** Run `pytest` for unit/smoke coverage and `ruff check .` to enforce style before submitting changes.

## FAQ / troubleshooting
- **Why do runs switch to float32 on CPU?** When GPUs are unavailable, adapters request `float32` automatically to maintain numerical stability.
- **How do I handle large model downloads?** Configure Hugging Face cache directories (`HF_HOME`, `TRANSFORMERS_CACHE`) to point at storage with sufficient space.
- **Deterministic decoding still differs across machines.** Verify matching torch/transformers versions, and ensure CUDA/cuDNN allow deterministic kernels; otherwise rerun with `--non-deterministic` disabled and check logs for warnings.

## Testing
Execute the local validation suite after significant changes:
```bash
pytest
ruff check .
```
