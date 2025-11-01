"""Utilities for residual stream injection during inference."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, MutableMapping

import torch
from torch.utils.hooks import RemovableHandle

from .adapters.base import BaseModelAdapter

__all__ = [
    "InjectionSpec",
    "attach_injection",
    "find_substring_span",
    "inject_once",
    "make_residual_hook",
    "token_positions_from_spans",
    "token_positions_for_substring",
]


IntSequence = Sequence[int]


@dataclass(frozen=True)
class InjectionSpec:
    """Configuration describing a single residual injection."""

    layer_idx: int
    alpha: float
    vector: torch.Tensor
    token_positions: Sequence[int] | Sequence[IntSequence]
    apply_on_input: bool = False


def _canonicalize_positions(
    token_positions: Sequence[int] | Sequence[IntSequence],
    batch_size: int,
) -> list[list[int]]:
    """Return a per-batch list of token indices to modify."""

    if len(token_positions) == 0:
        return [[] for _ in range(batch_size)]

    first = token_positions[0]
    if isinstance(first, Iterable) and not isinstance(first, (bytes, str)):
        if len(token_positions) != batch_size:
            msg = (
                "token_positions must provide one sequence per batch element "
                f"when nested; got {len(token_positions)} entries for batch size {batch_size}."
            )
            raise ValueError(msg)
        return [list(int(idx) for idx in positions) for positions in token_positions]  # type: ignore[arg-type]

    shared = [int(idx) for idx in token_positions]  # type: ignore[arg-type]
    return [list(shared) for _ in range(batch_size)]


def _build_modifier(spec: InjectionSpec) -> Callable[[torch.Tensor], tuple[torch.Tensor, bool]]:
    """Create a function that applies the configured injection to hidden states."""

    vector = spec.vector.detach()

    if vector.ndim != 1:
        raise ValueError("Injection vector must be one-dimensional")

    def modifier(hidden: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if hidden.ndim != 3:
            raise ValueError(
                "Residual hook expects tensor of shape (batch, seq_len, hidden_size)"
            )

        batch_size, seq_len, hidden_size = hidden.shape
        if vector.shape[0] != hidden_size:
            raise ValueError(
                "Injection vector hidden size mismatch: "
                f"expected {hidden_size}, got {vector.shape[0]}"
            )

        per_batch = _canonicalize_positions(spec.token_positions, batch_size)
        if any(max(pos, default=-1) >= seq_len for pos in per_batch if pos):
            raise IndexError("Token position out of range for current sequence length")
        if any(min(pos) < 0 for pos in per_batch if pos):
            raise IndexError("Token position must be non-negative")

        scaled_vector = (
            torch.as_tensor(spec.alpha, dtype=hidden.dtype, device=hidden.device)
            * vector.to(device=hidden.device, dtype=hidden.dtype)
        )

        result = hidden
        changed = False
        for batch_idx, positions in enumerate(per_batch):
            if not positions:
                continue
            if not changed:
                result = hidden.clone()
                changed = True
            result[batch_idx, positions, :] += scaled_vector

        return result, changed

    return modifier


def make_residual_hook(spec: InjectionSpec) -> Callable[[torch.nn.Module, tuple, torch.Tensor], torch.Tensor]:
    """Create a forward hook that injects ``alpha · vector`` at selected positions."""

    modifier = _build_modifier(spec)

    def hook(
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        del module, inputs
        if not isinstance(output, torch.Tensor):
            raise TypeError("Residual hook expected tensor output from module")
        updated, _ = modifier(output)
        return updated

    return hook


def attach_injection(adapter: BaseModelAdapter, spec: InjectionSpec) -> RemovableHandle:
    """Register the configured injection on the provided adapter and return the handle."""

    if spec.apply_on_input:
        module = adapter.layer_module(spec.layer_idx)
        modifier = _build_modifier(spec)

        def pre_hook(
            module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
        ) -> tuple[torch.Tensor, ...] | None:
            del module
            if not inputs:
                return None
            hidden = inputs[0]
            if not isinstance(hidden, torch.Tensor):
                raise TypeError("Expected tensor as first argument to transformer block")
            updated, changed = modifier(hidden)
            if not changed:
                return None
            return (updated, *inputs[1:])

        return module.register_forward_pre_hook(pre_hook, with_kwargs=False)

    hook_fn = make_residual_hook(spec)
    return adapter.register_residual_hook(spec.layer_idx, hook_fn)


def find_substring_span(text: str, substring: str, *, occurrence: int = 0) -> tuple[int, int]:
    """Return the character span for ``substring`` within ``text``.

    Args:
        text: Source text that contains ``substring``.
        substring: The fragment whose location should be resolved.
        occurrence: Zero-based index specifying which occurrence to select when the
            substring appears multiple times. The search proceeds left-to-right.

    Returns:
        A ``(start, end)`` tuple describing the substring span.

    Raises:
        ValueError: If the substring cannot be located.
    """

    if substring == "":
        raise ValueError("Substring must be non-empty")

    search_from = 0
    for _ in range(occurrence + 1):
        index = text.find(substring, search_from)
        if index == -1:
            raise ValueError(
                f"Substring {substring!r} (occurrence {occurrence}) not found in text"
            )
        search_from = index + len(substring)

    return index, index + len(substring)


def token_positions_from_spans(
    adapter: BaseModelAdapter,
    text: str,
    span_slices: Sequence[tuple[int, int]],
) -> list[int]:
    """Resolve token indices for the provided ``span_slices`` within ``text``."""

    return adapter.tokens_for_spans(text, span_slices)


def token_positions_for_substring(
    adapter: BaseModelAdapter,
    text: str,
    substring: str,
    *,
    occurrence: int = 0,
) -> list[int]:
    """Return token indices corresponding to a substring occurrence."""

    span = find_substring_span(text, substring, occurrence=occurrence)
    return token_positions_from_spans(adapter, text, [span])


def _prepare_inputs(adapter: BaseModelAdapter, prompt: str) -> dict[str, torch.Tensor]:
    """Tokenize ``prompt`` and move tensors onto the model device."""

    tokenizer_inputs = adapter.tokenizer(prompt, return_tensors="pt")
    device = next(adapter.model.parameters()).device
    tensor_inputs: dict[str, torch.Tensor] = {}
    for name, value in tokenizer_inputs.items():
        if isinstance(value, torch.Tensor):
            tensor_inputs[name] = value.to(device)
    return tensor_inputs


def _apply_generation_defaults(gen_kwargs: MutableMapping[str, Any]) -> None:
    """Populate deterministic generation defaults where not already supplied."""

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


def _normalize_positions(
    token_positions: Sequence[int] | Sequence[IntSequence] | None,
) -> Sequence[int] | Sequence[IntSequence]:
    """Return canonical token positions for a single-element batch."""

    if token_positions is None:
        return []

    if not token_positions:
        return []

    first = token_positions[0]
    if isinstance(first, Iterable) and not isinstance(first, (bytes, str)):
        return [
            [int(idx) for idx in positions]
            for positions in token_positions  # type: ignore[list-item]
        ]

    return [int(idx) for idx in token_positions]  # type: ignore[arg-type]


def inject_once(
    adapter: BaseModelAdapter,
    prompt: str,
    spec: InjectionSpec,
    *,
    token_positions: Sequence[int] | Sequence[IntSequence] | None = None,
    span_slices: Sequence[tuple[int, int]] | None = None,
    gen_kwargs: Mapping[str, Any] | None = None,
    enable_injection: bool = True,
) -> str:
    """Generate text with an optional single residual-stream injection applied.

    Args:
        adapter: Model adapter controlling tokenization, generation, and hooks.
        prompt: Prompt text to feed to the language model.
        spec: Injection configuration describing target layer, scaling, and vector.
        token_positions: Explicit token indices where the vector should be injected.
        span_slices: Optional character spans (start, end) that will be converted to
            token indices via ``adapter.tokens_for_spans`` when ``token_positions`` is
            not provided.
        gen_kwargs: Additional ``generate`` keyword arguments. Deterministic defaults
            are supplied when keys are absent.
        enable_injection: When ``False``, a control run without injection is executed.

    Returns:
        The decoded output string from ``model.generate``.
    """

    effective_positions: Sequence[int] | Sequence[IntSequence] | None = token_positions
    if effective_positions is None and span_slices is not None:
        effective_positions = adapter.tokens_for_spans(prompt, span_slices)
    if effective_positions is None:
        effective_positions = spec.token_positions

    resolved_positions = _normalize_positions(effective_positions)
    resolved_spec = replace(spec, token_positions=resolved_positions)

    mutable_kwargs: MutableMapping[str, Any]
    if gen_kwargs is None:
        mutable_kwargs = {}
    else:
        mutable_kwargs = dict(gen_kwargs)

    _apply_generation_defaults(mutable_kwargs)

    inputs = _prepare_inputs(adapter, prompt)

    handle: RemovableHandle | None = None
    if enable_injection:
        handle = attach_injection(adapter, resolved_spec)

    try:
        with torch.inference_mode():
            adapter.model(**inputs, use_cache=True)
            output_ids = adapter.model.generate(**inputs, **mutable_kwargs)
    finally:
        if handle is not None:
            handle.remove()

    return adapter.tokenizer.decode(output_ids[0], skip_special_tokens=True)

