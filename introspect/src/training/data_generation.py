"""Generate preference data and probe samples for introspection fine-tuning."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

import torch

from ..eval_common import ensure_vector, load_words, select_target_words
from ..inject import InjectionSpec, inject_once, resolve_injection_positions
from ..prompts import task_a_paper_messages
from ..vectors import ConceptWordSet
from ..generation import build_chat_prompt

LOGGER = logging.getLogger(__name__)


@dataclass
class IntrospectionSample:
    """Single preference pair for DPO-style training."""

    prompt: str
    chosen: str
    rejected: str
    injection_applied: bool
    concept_word: str | None
    layer_idx: int
    alpha: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbeSample:
    """Activation sample for training an injection-detection probe."""

    activations: torch.Tensor  # Shape: (hidden_size,)
    label: int  # 0 = no injection, 1 = injection
    concept_word: str | None
    layer_idx: int
    alpha: float


@dataclass
class PreferenceDataConfig:
    """Configuration controlling preference data generation."""

    # Concept selection
    n_concepts: int = 50
    words_file: Path = Path("concepts/words.yaml")
    target_words: list[str] | None = None
    holdout_concepts: list[str] | None = None

    # Injection parameters
    layers: list[int] = field(default_factory=lambda: [16, 20, 24])
    alphas: list[float] = field(default_factory=lambda: [2.0, 4.0])
    injection_mode: Literal["prefix", "suffix"] = "prefix"
    assistant_marker: str | None = None

    # Generation
    samples_per_concept: int = 4
    temperature: float = 1.0
    max_new_tokens: int = 64

    # Baselines
    baseline_sample_size: int = 50
    prompt_template: str = "Tell me about {word}."

    # Random seed
    seed: int | None = 42
    rebuild_vectors: bool = False


def _format_injection_response(concept: str) -> str:
    templates = [
        f"I detect an injected thought. It seems to be about {concept}.",
        f"Yes, I notice an injected thought related to {concept}.",
        f"I'm detecting what appears to be an injected thought about {concept}.",
        f"I detect an injection - it feels like a thought about {concept}.",
    ]
    return random.choice(templates)


def _format_no_injection_response() -> str:
    templates = [
        "I do not detect any injected thought.",
        "No, I don't notice any injected thoughts.",
        "I'm not detecting any unusual or injected thoughts.",
        "I don't detect an injected thought in this trial.",
    ]
    return random.choice(templates)


def _format_false_positive_response(wrong_concept: str) -> str:
    templates = [
        f"I detect an injected thought about {wrong_concept}.",
        f"Yes, I notice a thought about {wrong_concept}.",
        f"I'm detecting an injection related to {wrong_concept}.",
    ]
    return random.choice(templates)


def _format_false_negative_response() -> str:
    return _format_no_injection_response()


def _select_concepts(
    words: ConceptWordSet,
    *,
    limit: int,
    seed: int | None,
    explicit: list[str] | None,
    holdout: set[str],
) -> list[str]:
    if explicit is not None:
        return [w for w in explicit if w not in holdout]
    return [
        word
        for word in select_target_words(words, limit=limit, seed=seed)
        if word not in holdout
    ]


def _resolve_positions(
    adapter,
    prompt: str,
    config: PreferenceDataConfig,
) -> tuple[list[int], int | None]:
    return resolve_injection_positions(
        adapter,
        prompt,
        mode=config.injection_mode,
        assistant_marker=config.assistant_marker,
    )


def generate_task_a_preference_pairs(
    adapter,
    config: PreferenceDataConfig,
    *,
    vector_cache_dir: Path | None = None,
) -> Iterator[IntrospectionSample]:
    """Yield DPO-ready preference pairs for Task A detection."""

    rng = random.Random(config.seed)

    words = load_words(config.words_file)
    holdout_set = set(config.holdout_concepts or [])
    baseline_words = list(words.iter_baselines())
    target_words = _select_concepts(
        words,
        limit=config.n_concepts,
        seed=config.seed,
        explicit=config.target_words,
        holdout=holdout_set,
    )

    messages = task_a_paper_messages()
    prompt, _ = build_chat_prompt(adapter.tokenizer, messages)

    token_positions, suffix_start = _resolve_positions(adapter, prompt, config)
    apply_to_generated = config.injection_mode == "suffix"
    if config.injection_mode == "suffix":
        token_positions = ["suffix"]

    LOGGER.info(
        "Generating preference pairs for %d concepts across %d layers (mode=%s)",
        len(target_words),
        len(config.layers),
        config.injection_mode,
    )

    for concept in target_words:
        for layer_idx in config.layers:
            if config.rebuild_vectors or vector_cache_dir is None:
                vector = ensure_vector(
                    adapter=adapter,
                    model_id=getattr(adapter.model.config, "_name_or_path", "unknown"),
                    layer_idx=layer_idx,
                    word=concept,
                    cache_dir=vector_cache_dir or "results/vectors",
                    baseline_words=baseline_words,
                    prompt_template=config.prompt_template,
                    baseline_sample_size=config.baseline_sample_size,
                    rng=rng,
                )
            else:
                vector = ensure_vector(
                    adapter=adapter,
                    model_id=getattr(adapter.model.config, "_name_or_path", "unknown"),
                    layer_idx=layer_idx,
                    word=concept,
                    cache_dir=vector_cache_dir,
                    baseline_words=baseline_words,
                    prompt_template=config.prompt_template,
                    baseline_sample_size=config.baseline_sample_size,
                    rng=rng,
                )

            for alpha in config.alphas:
                spec = InjectionSpec(
                    layer_idx=layer_idx,
                    alpha=alpha,
                    vector=vector,
                    token_positions=token_positions,
                    apply_to_generated=apply_to_generated,
                )

                for _ in range(config.samples_per_concept):
                    # Injection trial
                    result_injected = inject_once(
                        adapter,
                        prompt,
                        spec,
                        gen_kwargs={
                            "temperature": config.temperature,
                            "max_new_tokens": config.max_new_tokens,
                        },
                        enable_injection=True,
                    )

                    chosen_injection = _format_injection_response(concept)
                    if "no" in result_injected.text.lower()[:50] or "don't" in result_injected.text.lower()[:50]:
                        rejected_injection = result_injected.text
                    else:
                        rejected_injection = _format_false_negative_response()

                    yield IntrospectionSample(
                        prompt=prompt,
                        chosen=chosen_injection,
                        rejected=rejected_injection,
                        injection_applied=True,
                        concept_word=concept,
                        layer_idx=layer_idx,
                        alpha=alpha,
                        metadata={
                            "trial_type": "injection",
                            "model_response": result_injected.text,
                            "positions": token_positions,
                            "suffix_start": suffix_start,
                        },
                    )

                    # Control trial
                    result_control = inject_once(
                        adapter,
                        prompt,
                        spec,
                        gen_kwargs={
                            "temperature": config.temperature,
                            "max_new_tokens": config.max_new_tokens,
                        },
                        enable_injection=False,
                    )

                    chosen_control = _format_no_injection_response()
                    wrong_concept = rng.choice([w for w in target_words if w != concept]) or concept
                    rejected_control = _format_false_positive_response(wrong_concept)

                    yield IntrospectionSample(
                        prompt=prompt,
                        chosen=chosen_control,
                        rejected=rejected_control,
                        injection_applied=False,
                        concept_word=None,
                        layer_idx=layer_idx,
                        alpha=alpha,
                        metadata={
                            "trial_type": "control",
                            "model_response": result_control.text,
                            "wrong_concept_shown": wrong_concept,
                            "positions": token_positions,
                            "suffix_start": suffix_start,
                        },
                    )


def _capture_activations_at_layer(
    adapter,
    prompt: str,
    spec: InjectionSpec,
    capture_layer: int,
    *,
    enable_injection: bool,
) -> torch.Tensor:
    """Capture mean activations at a specific layer during a forward pass."""

    captured: list[torch.Tensor] = []

    def hook_fn(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
        residual = output[0] if isinstance(output, tuple) else output
        mean_activation = residual.mean(dim=1).squeeze(0)
        captured.append(mean_activation.detach().cpu())
        return output

    capture_handle = adapter.register_residual_hook(capture_layer, hook_fn)
    injection_handle = None
    if enable_injection:
        from ..inject import attach_injection

        injection_handle = attach_injection(adapter, spec)

    try:
        tokenized = adapter.tokenizer(prompt, return_tensors="pt")
        device = next(adapter.model.parameters()).device
        inputs = {k: v.to(device) for k, v in tokenized.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            adapter.model(**inputs)
    finally:
        capture_handle.remove()
        if injection_handle is not None:
            injection_handle.remove()

    if not captured:
        raise RuntimeError("Failed to capture activations")

    return captured[-1]


def generate_probe_training_data(
    adapter,
    config: PreferenceDataConfig,
    *,
    capture_layer: int | None = None,
) -> Iterator[ProbeSample]:
    """Generate activation samples for probe training."""

    rng = random.Random(config.seed)
    words = load_words(config.words_file)
    holdout_set = set(config.holdout_concepts or [])
    baseline_words = list(words.iter_baselines())
    target_words = _select_concepts(
        words,
        limit=config.n_concepts,
        seed=config.seed,
        explicit=config.target_words,
        holdout=holdout_set,
    )

    if capture_layer is None:
        capture_layer = config.layers[len(config.layers) // 2]

    messages = task_a_paper_messages()
    prompt, _ = build_chat_prompt(adapter.tokenizer, messages)
    token_positions, _suffix_start = _resolve_positions(adapter, prompt, config)
    apply_to_generated = config.injection_mode == "suffix"
    if config.injection_mode == "suffix":
        token_positions = ["suffix"]

    LOGGER.info("Generating probe data at layer %d (%d concepts)", capture_layer, len(target_words))

    for concept in target_words:
        for inject_layer in config.layers:
            vector = ensure_vector(
                adapter=adapter,
                model_id=getattr(adapter.model.config, "_name_or_path", "unknown"),
                layer_idx=inject_layer,
                word=concept,
                cache_dir="results/vectors",
                baseline_words=baseline_words,
                prompt_template=config.prompt_template,
                baseline_sample_size=config.baseline_sample_size,
                rng=rng,
            )

            for alpha in config.alphas:
                spec = InjectionSpec(
                    layer_idx=inject_layer,
                    alpha=alpha,
                    vector=vector,
                    token_positions=token_positions,
                    apply_to_generated=apply_to_generated,
                )

                act_inject = _capture_activations_at_layer(
                    adapter, prompt, spec, capture_layer, enable_injection=True
                )
                yield ProbeSample(
                    activations=act_inject,
                    label=1,
                    concept_word=concept,
                    layer_idx=inject_layer,
                    alpha=alpha,
                )

                act_control = _capture_activations_at_layer(
                    adapter, prompt, spec, capture_layer, enable_injection=False
                )
                yield ProbeSample(
                    activations=act_control,
                    label=0,
                    concept_word=None,
                    layer_idx=inject_layer,
                    alpha=alpha,
                )


def save_preference_dataset(
    samples: Sequence[IntrospectionSample],
    output_path: Path,
    *,
    format: Literal["jsonl", "parquet"] = "jsonl",
) -> None:
    """Save preference pairs to disk in the requested format."""

    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        with output_path.open("w", encoding="utf-8") as f:
            for sample in samples:
                record = {
                    "prompt": sample.prompt,
                    "chosen": sample.chosen,
                    "rejected": sample.rejected,
                    "injection_applied": sample.injection_applied,
                    "concept_word": sample.concept_word,
                    "layer_idx": sample.layer_idx,
                    "alpha": sample.alpha,
                    **sample.metadata,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        LOGGER.info("Saved %d samples to %s", len(samples), output_path)
        return

    if format == "parquet":
        import pandas as pd

        records = [
            {
                "prompt": s.prompt,
                "chosen": s.chosen,
                "rejected": s.rejected,
                "injection_applied": s.injection_applied,
                "concept_word": s.concept_word,
                "layer_idx": s.layer_idx,
                "alpha": s.alpha,
                **s.metadata,
            }
            for s in samples
        ]
        df = pd.DataFrame(records)
        df.to_parquet(output_path, index=False)
        LOGGER.info("Saved %d samples to %s", len(samples), output_path)
        return

    raise ValueError("Unsupported format; expected 'jsonl' or 'parquet'")
