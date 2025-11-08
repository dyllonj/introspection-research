"""Falcon-family model adapter implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch.utils.hooks import RemovableHandle
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from transformers.utils import ModelOutput

from .base import (
    DeviceMap,
    HookFn,
    SpanSlice,
    seed_everything,
    select_device_map,
    select_dtype,
)
from ..generation import (
    apply_generation_defaults,
    decode_generated_tokens,
    prepare_generation_controls,
    prepare_generation_inputs,
)

__all__ = ["FalconAdapter"]


@dataclass
class FalconAdapter:
    """Adapter for HuggingFace Falcon-style causal language models."""

    model_id: str
    tokenizer: PreTrainedTokenizer
    model: PreTrainedModel

    name: str = "FalconAdapter"

    def __post_init__(self) -> None:
        self.model.eval()
        self.hidden_size: int = int(self.model.config.hidden_size)
        self.num_layers: int = int(self.model.config.num_hidden_layers)

    @classmethod
    def load(
        cls,
        model_id: str,
        dtype: torch.dtype | None,
        device_map: DeviceMap | None,
        *,
        seed: int | None = None,
    ) -> "FalconAdapter":
        if seed is not None:
            seed_everything(seed)

        resolved_dtype = select_dtype(dtype)
        resolved_device_map = select_device_map(device_map)

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        except ValueError:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=resolved_dtype,
            device_map=resolved_device_map,
        )

        return cls(model_id=model_id, tokenizer=tokenizer, model=model)

    def layer_module(self, layer_idx: int) -> torch.nn.Module:
        try:
            return self.model.transformer.h[layer_idx]
        except IndexError as exc:  # pragma: no cover
            raise IndexError(f"Layer index {layer_idx} out of range") from exc

    def register_residual_hook(self, layer_idx: int, hook_fn: HookFn) -> RemovableHandle:
        module = self.layer_module(layer_idx)

        def wrapper(
            mod: torch.nn.Module,
            inputs: tuple[Any, ...],
            output: torch.Tensor | tuple[Any, ...] | ModelOutput,
        ) -> torch.Tensor | tuple[Any, ...] | ModelOutput:
            if isinstance(output, torch.Tensor):
                return hook_fn(mod, inputs, output)

            if isinstance(output, ModelOutput):
                data = list(output.to_tuple())
                data[0] = hook_fn(mod, inputs, data[0])
                return output.__class__(*data)

            if isinstance(output, tuple):
                hidden = output[0]
                modified = hook_fn(mod, inputs, hidden)
                return (modified, *output[1:])

            raise TypeError(
                "Unsupported module output type for residual hook: " f"{type(output)!r}"
            )

        return module.register_forward_hook(wrapper)

    def tokens_for_spans(self, text: str, span_slices: Sequence[SpanSlice]) -> list[int]:
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offsets = encoding.get("offset_mapping")
        if offsets is None:
            raise ValueError("Tokenizer must provide offset mapping for span alignment")

        selected: list[int] = []
        for span in span_slices:
            start, end = span
            if start < 0 or end < start or end > len(text):
                raise ValueError(f"Invalid span slice {span}")

            indices = [
                idx
                for idx, (tok_start, tok_end) in enumerate(offsets)
                if tok_start < end and tok_end > start
            ]

            if not indices:
                raise ValueError(f"No tokens found for span {span}")

            selected.extend(indices)

        return list(dict.fromkeys(selected))

    def generate(
        self,
        prompt: str,
        /,
        *,
        stop_sequences: Sequence[str] | None = None,
        **gen_kwargs: Any,
    ) -> str:
        tensor_inputs, prompt_len = prepare_generation_inputs(self, prompt)
        kwargs = dict(gen_kwargs)
        if stop_sequences is not None and "stop_sequences" not in kwargs:
            kwargs["stop_sequences"] = tuple(stop_sequences)
        stop_sequences = apply_generation_defaults(self, kwargs)

        stop_sequences, _allowed_formats = prepare_generation_controls(
            self.tokenizer,
            prompt_len,
            kwargs,
        )

        with torch.no_grad():
            output_ids = self.model.generate(**tensor_inputs, **kwargs)

        return decode_generated_tokens(
            self,
            output_ids,
            prompt_len,
            stop_sequences=stop_sequences,
        )
