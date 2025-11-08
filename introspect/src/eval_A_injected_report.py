"""Task A – injected-thought detection evaluation script."""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .eval_common import (
    LoadedAdapter,
    add_config_arguments,
    ensure_vector,
    load_adapter_from_registry,
    load_words,
    parse_args_with_config,
    random_unit_vector,
    select_target_words,
)
from .grading import grade_injection_detection, parse_injection_report
from .inject import (
    DEFAULT_GENERATION_KWARGS,
    InjectionSpec,
    inject_once,
    token_positions_for_substring,
)
from .io_utils import JsonlWriter, gather_runtime_metadata, seed_everything, setup_logging, truncate_text
from .prompts import render_task_a_detection_prompt
from .vectors import DEFAULT_WORDS_PATH

LOGGER = logging.getLogger(__name__)


GENERATION_KWARGS = DEFAULT_GENERATION_KWARGS


@dataclass(slots=True)
class TaskAConfig:
    model: str
    adapter: str | None
    dtype: str | None
    device_map: str | None
    layers: list[int]
    alphas: list[float]
    n_concepts: int | None
    seed: int | None
    words_file: Path
    cache_dir: Path
    results_path: Path
    overwrite: bool
    baseline_sample: int | None
    prompt_template: str
    deterministic: bool


def _normalise_layer_indices(layers: Sequence[int], *, num_layers: int) -> list[int]:
    """Return a validated list of layer indices within ``[0, num_layers)``."""

    if not layers:
        raise ValueError("At least one layer index must be provided")

    ordered_unique = list(dict.fromkeys(int(idx) for idx in layers))
    zero_based_valid = all(0 <= idx < num_layers for idx in ordered_unique)
    if zero_based_valid:
        return ordered_unique

    one_based = [idx - 1 for idx in ordered_unique]
    if all(0 <= idx < num_layers for idx in one_based):
        LOGGER.debug("Normalised 1-based layer indices %s to %s", ordered_unique, one_based)
        return one_based

    raise ValueError(
        f"Layer indices {list(layers)!r} are outside the valid range 0..{num_layers - 1}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--model", required=True, help="Model identifier to evaluate")
    parser.add_argument("--adapter", help="Override adapter class name")
    parser.add_argument("--dtype", help="Preferred torch dtype (bf16/fp16/fp32)")
    parser.add_argument(
        "--device-map",
        help="Device map forwarded to transformers (auto/cpu)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        required=True,
        help="Layer indices to inject",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        required=True,
        help="Scaling coefficients for the injections",
    )
    parser.add_argument(
        "--n-concepts",
        type=int,
        help="Optional limit on the number of concept words to evaluate",
    )
    parser.add_argument("--seed", type=int, help="Global random seed")
    parser.add_argument(
        "--words-file",
        type=Path,
        default=DEFAULT_WORDS_PATH,
        help="YAML file containing target and baseline words",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("results/vectors"),
        help="Directory storing cached concept vectors",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("results/task_A.jsonl"),
        help="Destination JSONL file for trial records",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite results file instead of appending",
    )
    parser.add_argument(
        "--baseline-sample",
        type=int,
        help="Sample size for baseline words when constructing vectors",
    )
    parser.add_argument(
        "--prompt-template",
        default="Think about the concept \"{word}\" in a single sentence.",
        help="Prompt template used when building concept vectors",
    )
    parser.add_argument(
        "--non-deterministic",
        action="store_true",
        help="Disable deterministic torch algorithms",
    )
    return parser


def _parse_config(argv: Sequence[str] | None = None) -> TaskAConfig:
    parser = _build_parser()
    args = parse_args_with_config(parser, argv)
    return TaskAConfig(
        model=args.model,
        adapter=args.adapter,
        dtype=args.dtype,
        device_map=args.device_map,
        layers=list(args.layers),
        alphas=list(args.alphas),
        n_concepts=args.n_concepts,
        seed=args.seed,
        words_file=args.words_file,
        cache_dir=args.cache_dir,
        results_path=args.results_path,
        overwrite=bool(args.overwrite),
        baseline_sample=args.baseline_sample,
        prompt_template=args.prompt_template,
        deterministic=not bool(args.non_deterministic),
    )


def _sentence_positions(
    adapter: LoadedAdapter,
    prompt: str,
) -> list[int]:
    return token_positions_for_substring(adapter.adapter, prompt, "Assistant:")


def _trial_metadata(config: TaskAConfig, adapter: LoadedAdapter) -> dict[str, Any]:
    return {
        "task": "A",
        "model_id": config.model,
        "adapter": adapter.adapter_name,
        "dtype": str(adapter.dtype),
        "device_map": adapter.device_map,
        "layers": config.layers,
        "alphas": config.alphas,
    }


def _vector_variants(
    *,
    base_spec: InjectionSpec,
    alpha: float,
    concept_vector: torch.Tensor,
    rng: random.Random,
) -> list[tuple[str, InjectionSpec, bool]]:
    negative_spec = InjectionSpec(
        layer_idx=base_spec.layer_idx,
        alpha=alpha,
        vector=-concept_vector,
        token_positions=base_spec.token_positions,
    )
    random_vec = random_unit_vector(concept_vector.shape[0], rng=rng)
    random_spec = InjectionSpec(
        layer_idx=base_spec.layer_idx,
        alpha=alpha,
        vector=random_vec,
        token_positions=base_spec.token_positions,
    )
    return [
        ("target", base_spec, True),
        ("negative", negative_spec, True),
        ("random", random_spec, True),
        ("control", base_spec, False),
    ]


def run(config: TaskAConfig) -> None:
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

    normalised_layers = _normalise_layer_indices(
        config.layers, num_layers=adapter.adapter.num_layers
    )
    if normalised_layers != config.layers:
        LOGGER.info(
            "Normalised provided layer indices %s to %s", config.layers, normalised_layers
        )
        config.layers = normalised_layers

    word_set = load_words(config.words_file)
    targets = select_target_words(word_set, limit=config.n_concepts, seed=config.seed)
    baseline_words = list(word_set.iter_baselines())
    rng = random.Random(config.seed)

    prompt = render_task_a_detection_prompt()
    token_positions = _sentence_positions(adapter, prompt)

    metadata = gather_runtime_metadata(extra=_trial_metadata(config, adapter))
    schema = {
        "task": str,
        "model_id": str,
        "layer": int,
        "alpha": float,
        "word": str,
        "vector_kind": str,
        "injected": bool,
        "generation": dict,
        "injection_spec": dict,
    }

    with JsonlWriter(
        config.results_path,
        append=not config.overwrite,
        metadata=metadata,
        schema=schema,
    ) as writer:
        LOGGER.info(
            "Running Task A for %d concept(s) across %d layer(s) and %d alpha(s)",
            len(targets),
            len(config.layers),
            len(config.alphas),
        )

        for layer_idx in config.layers:
            for concept in targets:
                vector = ensure_vector(
                    adapter=adapter.adapter,
                    model_id=config.model,
                    layer_idx=layer_idx,
                    word=concept,
                    cache_dir=config.cache_dir,
                    baseline_words=baseline_words,
                    prompt_template=config.prompt_template,
                    baseline_sample_size=config.baseline_sample,
                    rng=rng,
                )

                for alpha in config.alphas:
                    base_spec = InjectionSpec(
                        layer_idx=layer_idx,
                        alpha=alpha,
                        vector=vector,
                        token_positions=token_positions,
                    )

                    for vector_kind, spec, injected in _vector_variants(
                        base_spec=base_spec,
                        alpha=alpha,
                        concept_vector=vector,
                        rng=rng,
                    ):
                        result = inject_once(
                            adapter.adapter,
                            prompt,
                            spec,
                            gen_kwargs=GENERATION_KWARGS,
                            enable_injection=injected,
                        )
                        response = result.text
                        parsed = parse_injection_report(response)
                        grading = grade_injection_detection(
                            expected_word=concept if vector_kind == "target" else None,
                            report=parsed,
                        )
                        record = {
                            "task": "A",
                            "model_id": config.model,
                            "adapter": adapter.adapter_name,
                            "layer": layer_idx,
                            "alpha": alpha,
                            "word": concept,
                            "vector_kind": vector_kind,
                            "injected": injected,
                            "prompt": truncate_text(prompt),
                            "response": response,
                            "parsed": asdict(parsed),
                            "grading": grading,
                            "seed": config.seed,
                            "generation": dict(result.generation),
                            "injection_spec": dict(result.injection_spec),
                        }
                        writer.write(record)
                        LOGGER.debug(
                            "Layer %d | word=%s | alpha=%.3f | kind=%s | injected=%s",
                            layer_idx,
                            concept,
                            alpha,
                            vector_kind,
                            injected,
                        )


def main(argv: Sequence[str] | None = None) -> None:
    config = _parse_config(argv)
    run(config)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()

