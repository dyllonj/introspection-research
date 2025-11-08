"""Shared text generation helpers for adapters and injection utilities."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Sequence, TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    from .adapters.base import BaseModelAdapter

__all__ = [
    "DEFAULT_GENERATION_KWARGS",
    "DEFAULT_STOP_SEQUENCES",
    "apply_generation_defaults",
    "decode_generated_tokens",
    "prepare_generation_inputs",
    "trim_stop_sequences",
]


DEFAULT_STOP_SEQUENCES: tuple[str, ...] = (
    "\nAssistant:",
    "\nHuman:",
    "\nUser:",
    "\nSystem:",
)
"""Canonical stop sequences shared across evaluation prompts."""


DEFAULT_GENERATION_KWARGS: Mapping[str, Any] = {
    "max_new_tokens": 64,
    "stop_sequences": DEFAULT_STOP_SEQUENCES,
}
"""Default generation configuration for short, well-scoped completions."""


def _coerce_stop_sequences(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        sequences = (value,)
    else:
        sequences = tuple(str(seq) for seq in value)
    return tuple(seq for seq in sequences if seq)


def trim_stop_sequences(text: str, stop_sequences: Sequence[str]) -> str:
    """Trim ``text`` at the earliest occurrence of any ``stop_sequences``."""

    earliest: int | None = None
    for seq in stop_sequences:
        index = text.find(seq)
        if index != -1 and (earliest is None or index < earliest):
            earliest = index

    if earliest is not None:
        text = text[:earliest]
    return text.rstrip()


def prepare_generation_inputs(
    adapter: "BaseModelAdapter" | Any,
    prompt: str,
) -> tuple[dict[str, torch.Tensor], int]:
    """Tokenize ``prompt`` and move tensors onto the model device."""

    tokenizer_inputs = adapter.tokenizer(prompt, return_tensors="pt")
    input_ids = tokenizer_inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):  # pragma: no cover - defensive
        msg = "Tokenizer must return tensor input_ids for generation"
        raise TypeError(msg)

    device = next(adapter.model.parameters()).device
    tensor_inputs: dict[str, torch.Tensor] = {}
    for name, value in tokenizer_inputs.items():
        if isinstance(value, torch.Tensor):
            tensor_inputs[name] = value.to(device)

    prompt_len = int(input_ids.shape[1])
    return tensor_inputs, prompt_len


def apply_generation_defaults(
    adapter: "BaseModelAdapter" | Any,
    gen_kwargs: MutableMapping[str, Any],
) -> tuple[str, ...]:
    """Populate deterministic generation defaults and return stop sequences."""

    defaults: Mapping[str, Any] = {
        "max_new_tokens": 128,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
    }
    for key, value in defaults.items():
        gen_kwargs.setdefault(key, value)

    tokenizer = adapter.tokenizer
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    if pad_token_id is None and eos_token_id is not None:
        pad_token_id = eos_token_id
    if eos_token_id is None and pad_token_id is not None:
        eos_token_id = pad_token_id

    if pad_token_id is not None:
        gen_kwargs.setdefault("pad_token_id", pad_token_id)
    if eos_token_id is not None:
        gen_kwargs.setdefault("eos_token_id", eos_token_id)

    stop_sequences = _coerce_stop_sequences(gen_kwargs.pop("stop_sequences", ()))
    return stop_sequences


def decode_generated_tokens(
    adapter: "BaseModelAdapter" | Any,
    output_ids: torch.Tensor,
    prompt_len: int,
    *,
    stop_sequences: Sequence[str] = (),
) -> str:
    """Decode only the tokens generated after the prompt and trim by stops."""

    if output_ids.ndim != 2:  # pragma: no cover - defensive
        raise ValueError("Expected 2D tensor of token ids from generate")

    prompt_len = max(0, min(prompt_len, output_ids.shape[1]))
    generated = output_ids[:, prompt_len:]
    if generated.numel() == 0:
        decoded = ""
    else:
        decoded = adapter.tokenizer.decode(generated[0], skip_special_tokens=True)

    return trim_stop_sequences(decoded, stop_sequences)
