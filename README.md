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

## Prompt formatting & decoding controls
- **Chat templates are mandatory.** Task prompts (especially Task A) are authored as chat message lists and must be rendered with the model tokenizer’s `apply_chat_template` or the built-in fallback. The canonical Task A script ends with an empty assistant turn so that the rendered prompt terminates in an `Assistant:` prefix ready for generation.【F:introspect/src/prompts.py†L214-L250】【F:introspect/src/generation.py†L119-L170】
- **Suffix injections rely on explicit token spans.** Utilities such as `token_positions_for_substring` and `token_positions_after` convert character spans (for example, the newline preceding “Trial 1”) into tokenizer indices so injections can begin at the assistant prefix and persist through streamed tokens.【F:introspect/src/inject.py†L332-L413】
- **Stop sequences and logits processors fence in completions.** `StopSequenceCriteria` halts decoding once any configured string or token suffix appears, while `AllowedPrefixLogitsProcessor` restricts outputs until a permitted prefix is emitted. Both are automatically wired up by `prepare_generation_controls` inside `inject_once`/`generate`, keeping verdict formats (`NO_INJECTION`, `INJECTION:`) stable during streaming runs.【F:introspect/src/generation.py†L188-L279】【F:introspect/src/inject.py†L447-L507】

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

# 7) Normalise legacy results layout (file operations only)
python -m introspect.src.tools.normalise_results \
  --source introspect/results \
  --dest introspect/results
# → moves each task_*.jsonl under results/<model_slug>/ without touching cached vectors;
#   no GPU context is required.
```

## Results post-processing (CPU)
These maintenance utilities operate entirely on the filesystem and cached JSONL tables—no GPU
session or model execution is needed. Run them from the repository root after evaluation runs
complete so reviewers can explore metrics and plots quickly.

1. **Normalise per-model directories.** Ensure legacy task outputs are grouped by model slug
   before any downstream filtering:

   ```bash
   python -m introspect.src.tools.normalise_results \
     --source introspect/results \
     --dest introspect/results
   ```

   The command only moves/copies `task_*.jsonl` files. Cached concept vectors in
   `introspect/results/vectors/<model_slug>/` are left untouched so repeated runs never trigger
   recomputation.

2. **Review data-quality flags.** Summarise `grading.notes` annotations (e.g. `invalid`,
   `invalid_format`) before pruning records:

   ```bash
   python -m introspect.src.analysis.data_quality \
     --results-root introspect/results \
     --format table
   ```

   Use the output to decide which notes to drop in the next step.

3. **Regenerate filtered metrics and plots.** Invoke the post-processing CLI to remove noisy
   records, rebuild aggregate tables, and refresh visualisations:

   ```bash
   python -m introspect.src.tools.postprocess \
     --results-root introspect/results \
     --drop-notes invalid invalid_format
   ```

   The utility writes cleaned JSONL/CSV tables under each model slug’s
   `postprocessed/` directory and reuses the filtered data when regenerating plots. Concept vector
   caches continue to be resolved from `results/vectors/<model_slug>/`, avoiding redundant
   extraction work.

### Tooling quick reference
The dedicated `introspect/src/tools/` directory hosts the CPU-only helper CLIs used during
post-processing:

- `python -m introspect.src.tools.normalise_results` — file mover/copy utility with
  `--copy`, `--overwrite`, and `--verbose` flags for tailoring how `task_*.jsonl` files are grouped
  by model slug.
- `python -m introspect.src.analysis.data_quality` — summarises `grading.notes` counts per task and
  mode; accepts `--results-root` and `--format {table,csv}` to control the output destination and
  presentation.
- `python -m introspect.src.tools.postprocess` — filters rows based on `grading.notes`, rebuilds
  metrics tables, and calls the plotting pipeline. Key flags include `--results-root`, optional
  repeated `--drop-notes` values, and `--verbose` for detailed logging.

## Jupyter runbook (GPU servers)
Use the notebook at `notebooks/PrimeIntellect_Runbook.ipynb` to run the full pipeline on a GPU box:

- Installs the package in editable mode (optional if preinstalled).
- Detects GPU and selects `bf16` when supported, else `fp16` (CPU falls back to `fp32`).
- Builds concept vectors and executes Tasks A–D with adjustable layer/alpha/trial counts.
- Optionally triggers multi-model sweeps and regenerates plots from existing JSONL logs.

Open the notebook in JupyterLab/VS Code, adjust the model IDs and counts to match your VRAM, and run cells top to bottom. For private models, export `HUGGING_FACE_HUB_TOKEN` in the environment before starting the kernel.

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
