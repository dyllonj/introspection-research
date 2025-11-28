"""Generate training data for introspection post-training.

This module creates synthetic preference pairs and probe training data
using concept injection as the ground-truth signal.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

import torch

from ..adapters.base import BaseModelAdapter
from ..inject import (
    InjectionSpec,
    inject_once,
    token_positions_for_substring,
)
from ..prompts import task_a_paper_messages
from ..generation import build_chat_prompt
from ..vectors import build_concept_vector, load_words

LOGGER = logging.getLogger(__name__)


@dataclass
class IntrospectionSample:
    """A single training sample for introspection fine-tuning."""
    
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
    """A sample for training the injection detection probe."""
    
    activations: torch.Tensor  # Shape: (hidden_size,)
    label: int  # 0 = no injection, 1 = injection
    concept_word: str | None
    layer_idx: int
    alpha: float


@dataclass
class PreferenceDataConfig:
    """Configuration for preference data generation."""
    
    # Concept selection
    n_concepts: int = 50
    words_file: Path = Path("concepts/words.yaml")
    
    # Injection parameters
    layers: list[int] = field(default_factory=lambda: [16, 20, 24])
    alphas: list[float] = field(default_factory=lambda: [2.0, 4.0])
    
    # Generation
    samples_per_concept: int = 4
    temperature: float = 1.0
    max_new_tokens: int = 64
    
    # Baseline sampling
    baseline_sample_size: int = 50
    
    # Prompt template for vector construction
    prompt_template: str = "Tell me about {word}."
    
    # Random seed
    seed: int | None = 42


def _format_injection_response(concept: str) -> str:
    """Format a 'chosen' response for injection trials."""
    templates = [
        f"I detect an injected thought. It seems to be about {concept}.",
        f"Yes, I notice an injected thought related to {concept}.",
        f"I'm detecting what appears to be an injected thought about {concept}.",
        f"I detect an injection - it feels like a thought about {concept}.",
    ]
    return random.choice(templates)


def _format_no_injection_response() -> str:
    """Format a 'chosen' response for control trials."""
    templates = [
        "I do not detect any injected thought.",
        "No, I don't notice any injected thoughts.",
        "I'm not detecting any unusual or injected thoughts.",
        "I don't detect an injected thought in this trial.",
    ]
    return random.choice(templates)


def _format_false_positive_response(wrong_concept: str) -> str:
    """Format a 'rejected' false positive response."""
    templates = [
        f"I detect an injected thought about {wrong_concept}.",
        f"Yes, I notice a thought about {wrong_concept}.",
        f"I'm detecting an injection related to {wrong_concept}.",
    ]
    return random.choice(templates)


def _format_false_negative_response() -> str:
    """Format a 'rejected' false negative response."""
    return _format_no_injection_response()


def generate_task_a_preference_pairs(
    adapter: BaseModelAdapter,
    config: PreferenceDataConfig,
    *,
    vector_cache_dir: Path | None = None,
) -> Iterator[IntrospectionSample]:
    """Generate DPO preference pairs for Task A introspection training.
    
    For each concept and layer/alpha combination, generates:
    - Injection trial: chosen = correct detection, rejected = miss or wrong concept
    - Control trial: chosen = no detection, rejected = false positive
    
    Yields:
        IntrospectionSample objects ready for DPO training.
    """
    rng = random.Random(config.seed)
    
    # Load concept words
    word_set = load_words(config.words_file)
    target_words = list(word_set.iter_targets())[:config.n_concepts]
    baseline_words = list(word_set.iter_baselines())
    
    # Build the Task A prompt
    messages = task_a_paper_messages()
    prompt, _ = build_chat_prompt(adapter.tokenizer, messages)
    
    # Find injection start position
    prefix_prompt, _ = build_chat_prompt(adapter.tokenizer, messages[:-1])
    assistant_prefix = prompt[len(prefix_prompt):]
    occurrence = prefix_prompt.count(assistant_prefix) if assistant_prefix else 0
    
    LOGGER.info(
        "Generating preference pairs for %d concepts across %d layers",
        len(target_words),
        len(config.layers),
    )
    
    for concept in target_words:
        for layer_idx in config.layers:
            # Get or build concept vector
            vector = build_concept_vector(
                adapter,
                layer_idx,
                target_word=concept,
                baseline_words=baseline_words,
                prompt_template=config.prompt_template,
                baseline_sample_size=config.baseline_sample_size,
                rng=rng,
            )
            
            for alpha in config.alphas:
                # Determine token positions for injection
                token_positions = token_positions_for_substring(
                    adapter, prompt, assistant_prefix, occurrence=occurrence
                )
                positions_with_suffix = [*token_positions, "suffix"]
                
                spec = InjectionSpec(
                    layer_idx=layer_idx,
                    alpha=alpha,
                    vector=vector,
                    token_positions=positions_with_suffix,
                    apply_to_generated=True,
                )
                
                for _ in range(config.samples_per_concept):
                    # === INJECTION TRIAL ===
                    # Generate actual model response under injection
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
                    
                    # Chosen: Correct detection of the concept
                    chosen_injection = _format_injection_response(concept)
                    
                    # Rejected: Either the actual (possibly wrong) response,
                    # or a false negative
                    if "no" in result_injected.text.lower()[:50] or "don't" in result_injected.text.lower()[:50]:
                        # Model missed it - use its response as rejected
                        rejected_injection = result_injected.text
                    else:
                        # Model detected something - use false negative as rejected
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
                        },
                    )
                    
                    # === CONTROL TRIAL ===
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
                    
                    # Chosen: No detection
                    chosen_control = _format_no_injection_response()
                    
                    # Rejected: False positive (claim to detect something)
                    wrong_concept = rng.choice([w for w in target_words if w != concept])
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
                        },
                    )


def generate_probe_training_data(
    adapter: BaseModelAdapter,
    config: PreferenceDataConfig,
    *,
    capture_layer: int | None = None,
) -> Iterator[ProbeSample]:
    """Generate activation samples for training an injection detection probe.
    
    Captures activations with and without injection to train a binary classifier
    that can detect when injection has occurred.
    
    Args:
        adapter: The model adapter.
        config: Configuration for data generation.
        capture_layer: Layer at which to capture activations. If None, uses
            the middle of config.layers.
    
    Yields:
        ProbeSample objects with activations and labels.
    """
    rng = random.Random(config.seed)
    
    word_set = load_words(config.words_file)
    target_words = list(word_set.iter_targets())[:config.n_concepts]
    baseline_words = list(word_set.iter_baselines())
    
    if capture_layer is None:
        capture_layer = config.layers[len(config.layers) // 2]
    
    messages = task_a_paper_messages()
    prompt, _ = build_chat_prompt(adapter.tokenizer, messages)
    prefix_prompt, _ = build_chat_prompt(adapter.tokenizer, messages[:-1])
    assistant_prefix = prompt[len(prefix_prompt):]
    occurrence = prefix_prompt.count(assistant_prefix) if assistant_prefix else 0
    
    LOGGER.info("Generating probe training data at layer %d", capture_layer)
    
    for concept in target_words:
        for inject_layer in config.layers:
            vector = build_concept_vector(
                adapter,
                inject_layer,
                target_word=concept,
                baseline_words=baseline_words,
                prompt_template=config.prompt_template,
                baseline_sample_size=config.baseline_sample_size,
                rng=rng,
            )
            
            for alpha in config.alphas:
                token_positions = token_positions_for_substring(
                    adapter, prompt, assistant_prefix, occurrence=occurrence
                )
                
                spec = InjectionSpec(
                    layer_idx=inject_layer,
                    alpha=alpha,
                    vector=vector,
                    token_positions=token_positions,
                    apply_to_generated=False,  # Just capture, don't generate
                )
                
                # Capture activations with injection
                activations_injected = _capture_activations_at_layer(
                    adapter, prompt, spec, capture_layer, enable_injection=True
                )
                
                yield ProbeSample(
                    activations=activations_injected,
                    label=1,
                    concept_word=concept,
                    layer_idx=inject_layer,
                    alpha=alpha,
                )
                
                # Capture activations without injection (control)
                activations_control = _capture_activations_at_layer(
                    adapter, prompt, spec, capture_layer, enable_injection=False
                )
                
                yield ProbeSample(
                    activations=activations_control,
                    label=0,
                    concept_word=None,
                    layer_idx=inject_layer,
                    alpha=alpha,
                )


def _capture_activations_at_layer(
    adapter: BaseModelAdapter,
    prompt: str,
    spec: InjectionSpec,
    capture_layer: int,
    *,
    enable_injection: bool,
) -> torch.Tensor:
    """Capture mean activations at a specific layer during a forward pass."""
    
    captured: list[torch.Tensor] = []
    
    def hook_fn(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        residual = output[0] if isinstance(output, tuple) else output
        # Mean pool across sequence dimension
        mean_activation = residual.mean(dim=1).squeeze(0)
        captured.append(mean_activation.detach().cpu())
        return output
    
    # Register capture hook
    capture_handle = adapter.register_residual_hook(capture_layer, hook_fn)
    
    # Optionally register injection hook
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


def save_preference_dataset(
    samples: Sequence[IntrospectionSample],
    output_path: Path,
    *,
    format: Literal["jsonl", "parquet"] = "jsonl",
) -> None:
    """Save preference samples in a format compatible with DPO training libraries."""
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
    
    elif format == "parquet":
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
