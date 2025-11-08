"""Base adapter protocol and shared utilities for model loaders."""
from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Protocol, Sequence, TypeVar, runtime_checkable

import torch
from torch.utils.hooks import RemovableHandle
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..io_utils import seed_everything as _seed_everything

__all__ = [
    "BaseModelAdapter",
    "DeviceMap",
    "HookFn",
    "SpanSlice",
    "seed_everything",
    "select_dtype",
    "select_device_map",
]

LOGGER = logging.getLogger(__name__)

HookFn = Callable[[torch.nn.Module, tuple[torch.Tensor, ...], torch.Tensor], torch.Tensor]
SpanSlice = tuple[int, int]
DeviceMap = str | Mapping[str, Any]

AdapterT = TypeVar("AdapterT", bound="BaseModelAdapter")


@runtime_checkable
class BaseModelAdapter(Protocol):
    """Protocol that all concrete model adapters must implement."""

    name: str
    hidden_size: int
    num_layers: int
    tokenizer: PreTrainedTokenizer
    model: PreTrainedModel

    @classmethod
    def load(
        cls: type[AdapterT],
        model_id: str,
        dtype: torch.dtype,
        device_map: DeviceMap,
        *,
        seed: int | None = None,
    ) -> AdapterT:
        """Load a model and return a fully initialized adapter."""

    def layer_module(self, layer_idx: int) -> torch.nn.Module:
        """Return the transformer block corresponding to ``layer_idx``."""

    def register_residual_hook(self, layer_idx: int, hook_fn: HookFn) -> RemovableHandle:
        """Attach a residual-stream forward hook at ``layer_idx``."""

    def tokens_for_spans(self, text: str, span_slices: Sequence[SpanSlice]) -> list[int]:
        """Map string spans to tokenizer token indices for the provided ``text``."""

    def generate(
        self,
        prompt: str,
        /,
        *,
        stop_sequences: Sequence[str] | None = None,
        **gen_kwargs: Any,
    ) -> str:
        """Generate text from the underlying model with ``prompt`` as context."""


def seed_everything(seed: int) -> None:
    """Seed all RNGs used by the toolkit for deterministic execution."""

    _seed_everything(seed)
    LOGGER.debug("Global seed set to %s", seed)


def select_dtype(requested: torch.dtype | None = None) -> torch.dtype:
    """Select an appropriate floating point dtype for model loading."""

    if requested is not None:
        return requested

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    LOGGER.debug("Falling back to float32 dtype on CPU")
    return torch.float32


def select_device_map(device_map: DeviceMap | None = None) -> DeviceMap:
    """Return a sensible default device map for model loading."""

    if device_map is not None:
        return device_map

    if torch.cuda.is_available():
        return "auto"

    LOGGER.debug("Using CPU device map for model loading")
    return "cpu"
