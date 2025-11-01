"""Utilities for residual stream injection during inference."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.hooks import RemovableHandle

from .adapters.base import BaseModelAdapter

__all__ = [
    "InjectionSpec",
    "attach_injection",
    "make_residual_hook",
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

