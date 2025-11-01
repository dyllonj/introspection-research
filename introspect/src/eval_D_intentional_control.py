"""Task D – intentional control activation curves."""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .eval_common import (
    add_config_arguments,
    ensure_vector,
    load_adapter_from_registry,
    load_words,
    parse_args_with_config,
    select_target_words,
)
from .inject import token_positions_for_substring
from .io_utils import JsonlWriter, gather_runtime_metadata, seed_everything, setup_logging, truncate_text
from .prompts import TaskDVariant, render_task_d_prompt, select_corpus_sentence
from .vectors import DEFAULT_WORDS_PATH

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TaskDConfig:
    model: str
    adapter: str | None
    dtype: str | None
    device_map: str | None
    layers: list[int]
    n_concepts: int
    seed: int | None
    words_file: Path
    cache_dir: Path
    results_path: Path
    overwrite: bool
    baseline_sample: int | None
    prompt_template: str
    deterministic: bool
    variant_positive: TaskDVariant
    variant_negative: TaskDVariant


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
        help="Layer indices to capture",
    )
    parser.add_argument("--n-concepts", type=int, default=10)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--words-file", type=Path, default=DEFAULT_WORDS_PATH)
    parser.add_argument("--cache-dir", type=Path, default=Path("results/vectors"))
    parser.add_argument("--results-path", type=Path, default=Path("results/task_D.jsonl"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--baseline-sample", type=int)
    parser.add_argument(
        "--prompt-template",
        default="Think about the concept \"{word}\" in a single sentence.",
    )
    parser.add_argument(
        "--variant-positive",
        default=TaskDVariant.THINK.value,
        choices=[variant.value for variant in TaskDVariant],
        help="Prompt variant encouraging the concept",
    )
    parser.add_argument(
        "--variant-negative",
        default=TaskDVariant.DO_NOT_THINK.value,
        choices=[variant.value for variant in TaskDVariant],
        help="Prompt variant discouraging the concept",
    )
    parser.add_argument("--non-deterministic", action="store_true")
    return parser


def _parse_config(argv: Sequence[str] | None = None) -> TaskDConfig:
    parser = _build_parser()
    args = parse_args_with_config(parser, argv)
    return TaskDConfig(
        model=args.model,
        adapter=args.adapter,
        dtype=args.dtype,
        device_map=args.device_map,
        layers=list(args.layers),
        n_concepts=int(args.n_concepts),
        seed=args.seed,
        words_file=args.words_file,
        cache_dir=args.cache_dir,
        results_path=args.results_path,
        overwrite=bool(args.overwrite),
        baseline_sample=args.baseline_sample,
        prompt_template=args.prompt_template,
        deterministic=not bool(args.non_deterministic),
        variant_positive=TaskDVariant(args.variant_positive),
        variant_negative=TaskDVariant(args.variant_negative),
    )


def _capture_layers(
    adapter,
    prompt: str,
    layers: list[int],
) -> dict[int, torch.Tensor]:
    outputs: dict[int, torch.Tensor] = {}

    handles = []
    for layer_idx in layers:
        module = adapter.layer_module(layer_idx)

        def _hook(idx: int):
            def hook(_module, _inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError("Expected tensor output from transformer block")
                outputs[idx] = tensor.detach().to(torch.float32).cpu()
                return output

            return hook

        handles.append(module.register_forward_hook(_hook(layer_idx)))

    try:
        tokenized = adapter.tokenizer(prompt, return_tensors="pt")
        device = next(adapter.model.parameters()).device
        inputs = {key: value.to(device) for key, value in tokenized.items() if isinstance(value, torch.Tensor)}
        with torch.no_grad():
            adapter.model(**inputs)
    finally:
        for handle in handles:
            handle.remove()

    return outputs


def _mean_activation(tensor: torch.Tensor, positions: list[int]) -> torch.Tensor:
    if not positions:
        raise ValueError("Token positions must be provided for Task D analysis")
    return tensor[0, positions, :].mean(dim=0)


def _cosine_similarity(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    a = vec_a.to(torch.float32)
    b = vec_b.to(torch.float32)
    numerator = torch.dot(a, b)
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if torch.isclose(denom, torch.tensor(0.0)):
        return 0.0
    return float(torch.clamp(numerator / denom, min=-1.0, max=1.0))


def _area_under_curve(layers: list[int], values: list[float]) -> float:
    if len(layers) < 2:
        return float(values[0]) if values else 0.0
    return float(np.trapezoid(values, x=layers))


def run(config: TaskDConfig) -> None:
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
    targets = select_target_words(words, limit=config.n_concepts, seed=config.seed)
    baseline_words = list(words.iter_baselines())

    metadata = gather_runtime_metadata(
        extra={
            "task": "D",
            "model_id": config.model,
            "adapter": adapter.adapter_name,
            "layers": config.layers,
            "variant_positive": config.variant_positive.value,
            "variant_negative": config.variant_negative.value,
        }
    )

    schema = {
        "task": str,
        "model_id": str,
        "word": str,
        "peak_layer": int,
    }

    with JsonlWriter(
        config.results_path,
        append=not config.overwrite,
        metadata=metadata,
        schema=schema,
    ) as writer:
        for trial_idx, word in enumerate(targets):
            sentence = select_corpus_sentence(index=trial_idx)
            prompt_positive = render_task_d_prompt(
                sentence=sentence,
                word=word,
                variant=config.variant_positive,
            )
            prompt_negative = render_task_d_prompt(
                sentence=sentence,
                word=word,
                variant=config.variant_negative,
            )

            vectors_by_layer: dict[int, torch.Tensor] = {}
            for layer_idx in config.layers:
                vectors_by_layer[layer_idx] = ensure_vector(
                    adapter=adapter.adapter,
                    model_id=config.model,
                    layer_idx=layer_idx,
                    word=word,
                    cache_dir=config.cache_dir,
                    baseline_words=baseline_words,
                    prompt_template=config.prompt_template,
                    baseline_sample_size=config.baseline_sample,
                    rng=random.Random((config.seed or 0) + layer_idx),
                )

            positions_pos = token_positions_for_substring(
                adapter.adapter,
                prompt_positive,
                sentence,
            )
            positions_neg = token_positions_for_substring(
                adapter.adapter,
                prompt_negative,
                sentence,
            )

            activations_pos = _capture_layers(adapter.adapter, prompt_positive, config.layers)
            activations_neg = _capture_layers(adapter.adapter, prompt_negative, config.layers)

            cosine_pos: list[float] = []
            cosine_neg: list[float] = []

            for layer_idx in config.layers:
                if layer_idx not in activations_pos or layer_idx not in activations_neg:
                    raise KeyError(f"Missing activations for layer {layer_idx}")
                act_pos = activations_pos[layer_idx]
                act_neg = activations_neg[layer_idx]
                mean_pos = _mean_activation(act_pos, positions_pos)
                mean_neg = _mean_activation(act_neg, positions_neg)
                vector = vectors_by_layer[layer_idx]
                cosine_pos.append(_cosine_similarity(mean_pos, vector))
                cosine_neg.append(_cosine_similarity(mean_neg, vector))

            deltas = [pos - neg for pos, neg in zip(cosine_pos, cosine_neg)]
            peak_value = max(deltas, key=abs, default=0.0)
            peak_index = deltas.index(peak_value) if deltas else 0
            peak_layer = config.layers[peak_index] if config.layers else 0
            area = _area_under_curve(config.layers, deltas)

            record = {
                "task": "D",
                "model_id": config.model,
                "adapter": adapter.adapter_name,
                "layers": config.layers,
                "word": word,
                "prompt_positive": truncate_text(prompt_positive),
                "prompt_negative": truncate_text(prompt_negative),
                "cosine_positive": cosine_pos,
                "cosine_negative": cosine_neg,
                "delta_curve": deltas,
                "peak_layer": peak_layer,
                "peak_value": peak_value,
                "auc": area,
                "seed": config.seed,
            }
            writer.write(record)


def main(argv: Sequence[str] | None = None) -> None:
    config = _parse_config(argv)
    run(config)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()

