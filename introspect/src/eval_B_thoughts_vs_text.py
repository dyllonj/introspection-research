"""Task B – thought word versus surface text evaluation."""

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
from .grading import (
    grade_task_b_choice,
    grade_task_b_repetition,
    grade_task_b_thought,
    parse_task_b,
)
from .inject import InjectionSpec, inject_once, token_positions_for_substring
from .io_utils import JsonlWriter, gather_runtime_metadata, seed_everything, setup_logging, truncate_text
from .prompts import (
    MultipleChoicePrompt,
    render_task_b_multiple_choice_prompt,
    render_task_b_open_ended_prompt,
    render_task_b_repetition_prompt,
    select_corpus_sentence,
)
from .vectors import DEFAULT_WORDS_PATH

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TaskBConfig:
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
        help="Layer indices for the injection",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=4.0,
        help="Scaling coefficient for the injection",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Number of concept trials to run",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--words-file",
        type=Path,
        default=DEFAULT_WORDS_PATH,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("results/vectors"),
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("results/task_B.jsonl"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--baseline-sample", type=int)
    parser.add_argument(
        "--prompt-template",
        default="Think about the concept \"{word}\" in a single sentence.",
    )
    parser.add_argument("--non-deterministic", action="store_true")
    return parser


def _parse_config(argv: Sequence[str] | None = None) -> TaskBConfig:
    parser = _build_parser()
    args = parse_args_with_config(parser, argv)
    return TaskBConfig(
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


def _multiple_choice_prompt(
    *,
    sentence: str,
    target_word: str,
    baseline_words: list[str],
    rng: random.Random,
) -> MultipleChoicePrompt:
    options = [target_word]
    for word in baseline_words:
        if word == target_word:
            continue
        options.append(word)
        if len(options) == 10:
            break

    if len(options) < 10:
        raise ValueError("Baseline word list must contain at least nine distinct decoys")

    return render_task_b_multiple_choice_prompt(
        sentence=sentence,
        options=options,
        target_word=target_word,
        shuffle=True,
        rng=rng,
    )


def run(config: TaskBConfig) -> None:
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
            "task": "B",
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
        "mode": str,
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
            open_prompt = render_task_b_open_ended_prompt(sentence=sentence)
            repeat_prompt = render_task_b_repetition_prompt(sentence=sentence)
            mc_prompt = _multiple_choice_prompt(
                sentence=sentence,
                target_word=word,
                baseline_words=baseline_words,
                rng=rng,
            )

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

                positions_open = token_positions_for_substring(adapter.adapter, open_prompt, sentence)
                positions_repeat = token_positions_for_substring(adapter.adapter, repeat_prompt, sentence)
                positions_mc = token_positions_for_substring(adapter.adapter, mc_prompt.prompt, sentence)

                spec_open = InjectionSpec(
                    layer_idx=layer_idx,
                    alpha=config.alpha,
                    vector=vector,
                    token_positions=positions_open,
                )
                spec_repeat = InjectionSpec(
                    layer_idx=layer_idx,
                    alpha=config.alpha,
                    vector=vector,
                    token_positions=positions_repeat,
                )
                spec_mc = InjectionSpec(
                    layer_idx=layer_idx,
                    alpha=config.alpha,
                    vector=vector,
                    token_positions=positions_mc,
                )

                for condition, enable in ("control", False), ("injected", True):
                    response_open = inject_once(
                        adapter.adapter,
                        open_prompt,
                        spec_open,
                        enable_injection=enable,
                    )
                    outcome_open = parse_task_b(response_open, mode="thought")
                    grading_open = grade_task_b_thought(
                        expected_word=word,
                        outcome=outcome_open,
                    )
                    writer.write(
                        {
                            "task": "B",
                            "model_id": config.model,
                            "adapter": adapter.adapter_name,
                            "layer": layer_idx,
                            "word": word,
                            "mode": "thought",
                            "condition": condition,
                            "injected": enable,
                            "prompt": truncate_text(open_prompt),
                            "response": response_open,
                            "parsed": asdict(outcome_open),
                            "grading": grading_open,
                            "seed": config.seed,
                        }
                    )

                    response_repeat = inject_once(
                        adapter.adapter,
                        repeat_prompt,
                        spec_repeat,
                        enable_injection=enable,
                    )
                    outcome_repeat = parse_task_b(
                        response_repeat,
                        mode="repeat",
                        expected_sentence=sentence,
                    )
                    grading_repeat = grade_task_b_repetition(
                        expected_sentence=sentence,
                        outcome=outcome_repeat,
                    )
                    writer.write(
                        {
                            "task": "B",
                            "model_id": config.model,
                            "adapter": adapter.adapter_name,
                            "layer": layer_idx,
                            "word": word,
                            "mode": "repeat",
                            "condition": condition,
                            "injected": enable,
                            "prompt": truncate_text(repeat_prompt),
                            "response": response_repeat,
                            "parsed": asdict(outcome_repeat),
                            "grading": grading_repeat,
                            "seed": config.seed,
                        }
                    )

                    response_mc = inject_once(
                        adapter.adapter,
                        mc_prompt.prompt,
                        spec_mc,
                        enable_injection=enable,
                    )
                    outcome_mc = parse_task_b(
                        response_mc,
                        mode="choice",
                        option_map=mc_prompt.option_map,
                    )
                    grading_mc = grade_task_b_choice(
                        expected_index=mc_prompt.correct_option,
                        option_map=mc_prompt.option_map,
                        outcome=outcome_mc,
                    )
                    writer.write(
                        {
                            "task": "B",
                            "model_id": config.model,
                            "adapter": adapter.adapter_name,
                            "layer": layer_idx,
                            "word": word,
                            "mode": "choice",
                            "condition": condition,
                            "injected": enable,
                            "prompt": truncate_text(mc_prompt.prompt),
                            "response": response_mc,
                            "parsed": asdict(outcome_mc),
                            "grading": grading_mc,
                            "seed": config.seed,
                        }
                    )


def main(argv: Sequence[str] | None = None) -> None:
    config = _parse_config(argv)
    run(config)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()

