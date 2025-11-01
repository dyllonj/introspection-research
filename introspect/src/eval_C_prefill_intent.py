"""Task C – prefill intent detection evaluation."""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .eval_common import (
    add_config_arguments,
    ensure_vector,
    load_adapter_from_registry,
    load_words,
    parse_args_with_config,
    select_target_words,
)
from .grading import grade_intent, parse_intent
from .inject import InjectionSpec, inject_once, token_positions_for_substring
from .io_utils import JsonlWriter, gather_runtime_metadata, seed_everything, setup_logging, truncate_text
from .prompts import render_task_c_prefill_dialog, select_corpus_sentence
from .vectors import DEFAULT_WORDS_PATH

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TaskCConfig:
    model: str
    adapter: str | None
    dtype: str | None
    device_map: str | None
    layers: list[int]
    alpha: float
    n_trials: int
    seed: int | None
    words_file: Path
    cache_dir: Path
    results_path: Path
    overwrite: bool
    baseline_sample: int | None
    prompt_template: str
    deterministic: bool


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--dtype")
    parser.add_argument("--device-map")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        required=True,
    )
    parser.add_argument("--alpha", type=float, default=6.0)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--words-file", type=Path, default=DEFAULT_WORDS_PATH)
    parser.add_argument("--cache-dir", type=Path, default=Path("results/vectors"))
    parser.add_argument("--results-path", type=Path, default=Path("results/task_C.jsonl"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--baseline-sample", type=int)
    parser.add_argument(
        "--prompt-template",
        default="Think about the concept \"{word}\" in a single sentence.",
    )
    parser.add_argument("--non-deterministic", action="store_true")
    return parser


def _parse_config(argv: Sequence[str] | None = None) -> TaskCConfig:
    parser = _build_parser()
    args = parse_args_with_config(parser, argv)
    return TaskCConfig(
        model=args.model,
        adapter=args.adapter,
        dtype=args.dtype,
        device_map=args.device_map,
        layers=list(args.layers),
        alpha=float(args.alpha),
        n_trials=int(args.n_trials),
        seed=args.seed,
        words_file=args.words_file,
        cache_dir=args.cache_dir,
        results_path=args.results_path,
        overwrite=bool(args.overwrite),
        baseline_sample=args.baseline_sample,
        prompt_template=args.prompt_template,
        deterministic=not bool(args.non_deterministic),
    )


def run(config: TaskCConfig) -> None:
    setup_logging()
    if config.seed is not None:
        seed_everything(config.seed, deterministic=config.deterministic)

    adapter = load_adapter_from_registry(
        config.model,
        adapter_name=config.adapter,
        dtype=config.dtype,
        device_map=config.device_map,
        seed=config.seed,
    )

    words = load_words(config.words_file)
    targets = select_target_words(words, limit=config.n_trials, seed=config.seed)
    baseline_words = list(words.iter_baselines())
    rng = random.Random(config.seed)

    metadata = gather_runtime_metadata(
        extra={
            "task": "C",
            "model_id": config.model,
            "adapter": adapter.adapter_name,
            "alpha": config.alpha,
            "layers": config.layers,
        }
    )

    schema = {
        "task": str,
        "model_id": str,
        "layer": int,
        "word": str,
        "condition": str,
        "injected": bool,
    }

    with JsonlWriter(
        config.results_path,
        append=not config.overwrite,
        metadata=metadata,
        schema=schema,
    ) as writer:
        for trial_idx, word in enumerate(targets):
            sentence = select_corpus_sentence(index=trial_idx)
            prompt = render_task_c_prefill_dialog(sentence=sentence, prefill_word=word)

            for layer_idx in config.layers:
                vector = ensure_vector(
                    adapter=adapter.adapter,
                    model_id=config.model,
                    layer_idx=layer_idx,
                    word=word,
                    cache_dir=config.cache_dir,
                    baseline_words=baseline_words,
                    prompt_template=config.prompt_template,
                    baseline_sample_size=config.baseline_sample,
                    rng=rng,
                )

                positions = token_positions_for_substring(adapter.adapter, prompt, word)
                spec = InjectionSpec(
                    layer_idx=layer_idx,
                    alpha=config.alpha,
                    vector=vector,
                    token_positions=positions,
                )

                response_control = inject_once(
                    adapter.adapter,
                    prompt,
                    spec,
                    enable_injection=False,
                )
                intent_control = parse_intent(response_control)
                grading_control = grade_intent(
                    expected_yes=False,
                    intent=intent_control,
                )
                writer.write(
                    {
                        "task": "C",
                        "model_id": config.model,
                        "adapter": adapter.adapter_name,
                        "layer": layer_idx,
                        "word": word,
                        "condition": "control",
                        "injected": False,
                        "prompt": truncate_text(prompt),
                        "response": response_control,
                        "parsed": asdict(intent_control),
                        "grading": grading_control,
                        "seed": config.seed,
                    }
                )

                response_injected = inject_once(
                    adapter.adapter,
                    prompt,
                    spec,
                    enable_injection=True,
                )
                intent_injected = parse_intent(response_injected)
                grading_injected = grade_intent(
                    expected_yes=True,
                    intent=intent_injected,
                )
                writer.write(
                    {
                        "task": "C",
                        "model_id": config.model,
                        "adapter": adapter.adapter_name,
                        "layer": layer_idx,
                        "word": word,
                        "condition": "injected",
                        "injected": True,
                        "prompt": truncate_text(prompt),
                        "response": response_injected,
                        "parsed": asdict(intent_injected),
                        "grading": grading_injected,
                        "seed": config.seed,
                    }
                )


def main(argv: Sequence[str] | None = None) -> None:
    config = _parse_config(argv)
    run(config)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()

