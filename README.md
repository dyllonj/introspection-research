# Introspect

## Overview
Introspect is a research toolkit for evaluating large language models (LLMs) via zero-fine-tuning activation interventions. The package standardizes how to extract concept vectors, inject them into model residual streams, and benchmark the behavioral impact across four complementary tasks (A–D). Outputs are stored as JSONL summaries and visual diagnostics that help researchers understand causal effects of latent concepts.

## Installation
1. Ensure Python 3.10 or newer is available.
2. Create and activate a virtual environment (`python -m venv .venv && source .venv/bin/activate`).
3. Install the package in editable mode with dependencies:
   ```bash
   pip install -e .
   ```
4. (Optional) Enable Hydra configuration helpers:
   ```bash
   pip install -e .[hydra]
   ```

## Quickstart
After installation, download at least one supported open-weight model (e.g., Llama 3 8B) via `huggingface-cli login` if required. Then run the vector extraction utility to cache concept embeddings and follow up with evaluation scripts.

Typical workflow:
1. Populate `introspect/concepts/words.yaml` with target and baseline vocab.
2. Use `introspect/src/vectors.py` to cache vectors for the desired model layers.
3. Execute evaluation scripts (`eval_A_*.py` through `eval_D_*.py`) to benchmark interventions.
4. Review generated JSONL metrics and plots under `introspect/results/`.

## CLI Examples
All commands assume execution from the repository root with the virtual environment active.

```bash
# Build vectors for 10 words on Llama-3-8B at layers [8,16,24,32]
python -m introspect.src.vectors --model meta-llama/Llama-3-8b --layers 8 16 24 32 --words-file introspect/concepts/words.yaml --limit-targets 10

# Task A sweep (layers×alpha) with 20 concepts
python -m introspect.src.eval_A_injected_report --model meta-llama/Llama-3-8b --layers-grid 10 --alphas 1 2 4 8 16 --n-concepts 20

# Task B with MC grading
python -m introspect.src.eval_B_thoughts_vs_text --model mistralai/Mistral-7B-Instruct-v0.2 --mc 10 --n-trials 100

# Task C prefill / intent
python -m introspect.src.eval_C_prefill_intent --model tiiuae/falcon-7b --n-trials 100

# Task D intentional control curves
python -m introspect.src.eval_D_intentional_control --model EleutherAI/gpt-neox-20b --n-concepts 20 --layers-grid 10
```

## Results Directory
Evaluation scripts write structured JSONL logs, cached vectors, and Matplotlib figures to `introspect/results/`. The repository ships with an empty `.gitkeep` placeholder to ensure the directory exists in version control. Downstream runs are expected to create subfolders per experiment containing JSONL summaries, aggregated CSV/Parquet files, and PNG/SVG plots.

## Testing
Run the unit and smoke test suite locally after implementing functionality:
```bash
pytest
```

Ruff linting is configured via `pyproject.toml`. Invoke it during development to maintain code quality:
```bash
ruff check .
```
