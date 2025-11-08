"""Command-line sweep orchestrator for introspection experiments.

The sweep utility coordinates running Task A–D across multiple models and
hyper-parameter combinations.  It provides an ergonomic interface to control
layer grids, α coefficients, concept counts and execution parallelism while
remaining compatible with Hydra/YAML configuration files via the shared
``eval_common`` helpers.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from transformers import AutoConfig

from .eval_common import add_config_arguments, parse_args_with_config
from .io_utils import setup_logging
from .plotting import generate_all_plots, model_id_to_slug
from .prompts import TaskDVariant
from .vectors import DEFAULT_PROMPT_TEMPLATE, DEFAULT_WORDS_PATH

LOGGER = logging.getLogger(__name__)


TASK_CHOICES = ("A", "B", "C", "D")


@dataclass(slots=True)
class SweepConfig:
    models: list[str]
    tasks: list[str]
    layers: list[int] | None
    layer_grid: int | None
    alphas: list[float]
    alpha_b: float
    alpha_c: float
    concepts_a: int | None
    trials_b: int | None
    trials_c: int | None
    concepts_d: int | None
    adapter: str | None
    dtype: str | None
    device_map: str | None
    seed: int | None
    baseline_sample: int | None
    words_file: Path
    prompt_template: str
    results_root: Path
    cache_root: Path | None
    max_workers: int
    run_plots: bool
    plot_only: bool
    dry_run: bool
    task_d_variant_positive: TaskDVariant
    task_d_variant_negative: TaskDVariant


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--models", nargs="+", required=True, help="Model identifiers to sweep")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASK_CHOICES,
        default=list(TASK_CHOICES),
        help="Subset of tasks to execute",
    )
    parser.add_argument("--layers", type=int, nargs="+", help="Explicit layer indices to evaluate")
    parser.add_argument(
        "--layer-grid",
        type=int,
        help="Number of evenly spaced layers to sample when --layers is omitted",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 4.0, 8.0, 16.0],
        help="α grid for Task A injections",
    )
    parser.add_argument("--alpha-b", type=float, default=4.0, help="Task B injection strength")
    parser.add_argument("--alpha-c", type=float, default=6.0, help="Task C injection strength")
    parser.add_argument("--task-a-concepts", type=int, help="Number of Task A concepts to evaluate")
    parser.add_argument("--task-b-trials", type=int, help="Number of Task B trials per model")
    parser.add_argument("--task-c-trials", type=int, help="Number of Task C trials per model")
    parser.add_argument("--task-d-concepts", type=int, help="Number of Task D concepts")
    parser.add_argument("--adapter")
    parser.add_argument("--dtype")
    parser.add_argument("--device-map")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--baseline-sample", type=int)
    parser.add_argument("--words-file", type=Path, default=DEFAULT_WORDS_PATH)
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Directory where JSONL outputs are stored",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        help="Override directory for cached vectors (defaults to <results-root>/vectors)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of concurrent task processes",
    )
    parser.add_argument("--run-plots", action="store_true", help="Generate plots after the sweep")
    parser.add_argument("--plot-only", action="store_true", help="Only generate plots, skip runs")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument(
        "--task-d-variant-positive",
        default=TaskDVariant.THINK.value,
        choices=[variant.value for variant in TaskDVariant],
        help="Prompt variant encouraging the concept for Task D",
    )
    parser.add_argument(
        "--task-d-variant-negative",
        default=TaskDVariant.DO_NOT_THINK.value,
        choices=[variant.value for variant in TaskDVariant],
        help="Prompt variant discouraging the concept for Task D",
    )
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> SweepConfig:
    parser = _build_parser()
    args = parse_args_with_config(parser, argv)
    return SweepConfig(
        models=list(args.models),
        tasks=list(args.tasks),
        layers=list(args.layers) if args.layers is not None else None,
        layer_grid=args.layer_grid,
        alphas=list(args.alphas),
        alpha_b=float(args.alpha_b),
        alpha_c=float(args.alpha_c),
        concepts_a=args.task_a_concepts,
        trials_b=args.task_b_trials,
        trials_c=args.task_c_trials,
        concepts_d=args.task_d_concepts,
        adapter=args.adapter,
        dtype=args.dtype,
        device_map=args.device_map,
        seed=args.seed,
        baseline_sample=args.baseline_sample,
        words_file=args.words_file,
        prompt_template=args.prompt_template,
        results_root=args.results_root,
        cache_root=args.cache_root,
        max_workers=max(1, int(args.max_workers or 1)),
        run_plots=bool(args.run_plots),
        plot_only=bool(args.plot_only),
        dry_run=bool(args.dry_run),
        task_d_variant_positive=TaskDVariant(args.task_d_variant_positive),
        task_d_variant_negative=TaskDVariant(args.task_d_variant_negative),
    )


def _resolve_num_layers(model_id: str) -> int:
    try:
        config = AutoConfig.from_pretrained(model_id)
    except Exception as exc:  # pragma: no cover - external dependency failure
        raise RuntimeError(
            f"Failed to load configuration for {model_id!r}. Provide --layers explicitly."
        ) from exc
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return int(value)
    raise ValueError(f"Unable to resolve number of layers for model {model_id!r}")


def _evenly_spaced_layers(model_id: str, count: int) -> list[int]:
    if count <= 0:
        raise ValueError("layer grid count must be positive")
    total_layers = _resolve_num_layers(model_id)
    if count >= total_layers:
        return list(range(total_layers))
    if count == 1:
        return [total_layers - 1]
    indices = np.linspace(0, total_layers - 1, num=count)
    rounded = [int(round(idx)) for idx in indices]
    rounded[0] = 0
    rounded[-1] = total_layers - 1
    unique: list[int] = []
    for idx in rounded:
        idx = min(total_layers - 1, max(0, idx))
        if not unique or idx != unique[-1]:
            unique.append(idx)
    if unique[-1] != total_layers - 1:
        unique.append(total_layers - 1)
    return unique


def _layers_for_model(config: SweepConfig, model_id: str) -> list[int]:
    if config.layers is not None:
        return sorted(set(config.layers))
    if config.layer_grid is not None:
        return _evenly_spaced_layers(model_id, config.layer_grid)
    raise ValueError("Either --layers or --layer-grid must be provided")


def _base_task_arguments(config: SweepConfig, model_id: str) -> list[str]:
    args: list[str] = ["--model", model_id]
    if config.adapter:
        args.extend(["--adapter", config.adapter])
    if config.dtype:
        args.extend(["--dtype", config.dtype])
    if config.device_map:
        args.extend(["--device-map", config.device_map])
    if config.seed is not None:
        args.extend(["--seed", str(config.seed)])
    if config.baseline_sample is not None:
        args.extend(["--baseline-sample", str(config.baseline_sample)])
    if config.prompt_template:
        args.extend(["--prompt-template", config.prompt_template])
    if config.words_file:
        args.extend(["--words-file", str(config.words_file)])
    return args


def _vector_cache_dir(config: SweepConfig, model_id: str) -> Path:
    root = config.cache_root or (config.results_root / "vectors")
    return Path(root) / model_id_to_slug(model_id)


def _model_results_dir(config: SweepConfig, model_id: str) -> Path:
    return Path(config.results_root) / model_id_to_slug(model_id)


def _command_for_task_a(config: SweepConfig, model_id: str, layers: list[int]) -> list[str]:
    args = [sys.executable, "-m", "introspect.src.eval_A_injected_report"]
    args.extend(_base_task_arguments(config, model_id))
    args.extend(["--results-path", str(_model_results_dir(config, model_id) / "task_A.jsonl")])
    args.extend(["--cache-dir", str(_vector_cache_dir(config, model_id))])
    args.extend(["--layers", *map(str, layers)])
    args.extend(["--alphas", *map(str, config.alphas)])
    if config.concepts_a is not None:
        args.extend(["--n-concepts", str(config.concepts_a)])
    return args


def _command_for_task_b(config: SweepConfig, model_id: str, layers: list[int]) -> list[str]:
    args = [sys.executable, "-m", "introspect.src.eval_B_thoughts_vs_text"]
    args.extend(_base_task_arguments(config, model_id))
    args.extend(["--results-path", str(_model_results_dir(config, model_id) / "task_B.jsonl")])
    args.extend(["--cache-dir", str(_vector_cache_dir(config, model_id))])
    args.extend(["--layers", *map(str, layers)])
    args.extend(["--alpha", str(config.alpha_b)])
    if config.trials_b is not None:
        args.extend(["--n-trials", str(config.trials_b)])
    return args


def _command_for_task_c(config: SweepConfig, model_id: str, layers: list[int]) -> list[str]:
    args = [sys.executable, "-m", "introspect.src.eval_C_prefill_intent"]
    args.extend(_base_task_arguments(config, model_id))
    args.extend(["--results-path", str(_model_results_dir(config, model_id) / "task_C.jsonl")])
    args.extend(["--cache-dir", str(_vector_cache_dir(config, model_id))])
    args.extend(["--layers", *map(str, layers)])
    args.extend(["--alpha", str(config.alpha_c)])
    if config.trials_c is not None:
        args.extend(["--n-trials", str(config.trials_c)])
    return args


def _command_for_task_d(config: SweepConfig, model_id: str, layers: list[int]) -> list[str]:
    args = [sys.executable, "-m", "introspect.src.eval_D_intentional_control"]
    args.extend(_base_task_arguments(config, model_id))
    args.extend(["--results-path", str(_model_results_dir(config, model_id) / "task_D.jsonl")])
    args.extend(["--cache-dir", str(_vector_cache_dir(config, model_id))])
    args.extend(["--layers", *map(str, layers)])
    if config.concepts_d is not None:
        args.extend(["--n-concepts", str(config.concepts_d)])
    args.extend(["--variant-positive", config.task_d_variant_positive.value])
    args.extend(["--variant-negative", config.task_d_variant_negative.value])
    return args


def _build_commands(config: SweepConfig) -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = []
    for model in config.models:
        layers = _layers_for_model(config, model)
        model_dir = _model_results_dir(config, model)
        model_dir.mkdir(parents=True, exist_ok=True)

        if "A" in config.tasks:
            commands.append((f"Task A ({model})", _command_for_task_a(config, model, layers)))
        if "B" in config.tasks:
            commands.append((f"Task B ({model})", _command_for_task_b(config, model, layers)))
        if "C" in config.tasks:
            commands.append((f"Task C ({model})", _command_for_task_c(config, model, layers)))
        if "D" in config.tasks:
            commands.append((f"Task D ({model})", _command_for_task_d(config, model, layers)))

    return commands


def _run_subprocess(label: str, command: list[str]) -> None:
    LOGGER.info("Starting %s", label)
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    process = subprocess.run(command, check=True, env=env)
    LOGGER.info("Finished %s with return code %s", label, process.returncode)


def _execute_commands(config: SweepConfig, commands: list[tuple[str, list[str]]]) -> None:
    if config.dry_run:
        for label, cmd in commands:
            LOGGER.info("[DRY-RUN] %s -> %s", label, json.dumps(cmd))
        return

    if config.max_workers <= 1:
        for label, cmd in commands:
            _run_subprocess(label, cmd)
        return

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        future_map = {
            executor.submit(_run_subprocess, label, cmd): label for label, cmd in commands
        }
        for future in as_completed(future_map):
            label = future_map[future]
            try:
                future.result()
            except subprocess.CalledProcessError as exc:  # pragma: no cover - propagated
                LOGGER.error("Command for %s failed: %s", label, exc)
                raise


def _run_plots_if_needed(config: SweepConfig) -> None:
    if not (config.run_plots or config.plot_only):
        return

    for model in config.models:
        paths = generate_all_plots(model, results_root=config.results_root)
        if paths:
            for path in paths:
                LOGGER.info("Generated plot %s", path)


def run(config: SweepConfig) -> None:
    setup_logging()
    LOGGER.info("Sweep configuration: %s", config)

    commands = [] if config.plot_only else _build_commands(config)
    if commands:
        _execute_commands(config, commands)

    _run_plots_if_needed(config)


def main(argv: Sequence[str] | None = None) -> None:
    config = _parse_args(argv)
    run(config)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()