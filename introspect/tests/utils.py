"""Testing utilities providing lightweight adapter and tokenizer stubs."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from introspect.src.adapters.base import BaseModelAdapter

__all__ = ["ToyAdapter", "make_toy_adapter"]


class ToyTokenizer:
    """Very small whitespace tokenizer used in tests.

    The tokenizer lowercases tokens and assigns deterministic incremental token
    identifiers. Offset mappings are computed via regular-expression matches so
    that span selection exercised by the injection utilities can be validated in
    tests without depending on the HuggingFace fast tokenizers.
    """

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {"<pad>": 0, "<eos>": 1}
        self.inv_vocab: dict[int, str] = {0: "<pad>", 1: "<eos>"}
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1

    def _token_matches(self, text: str) -> list[re.Match[str]]:
        return list(re.finditer(r"\S+", text))

    def _token_to_id(self, token: str) -> int:
        lowered = token.lower()
        if lowered not in self.vocab:
            index = len(self.vocab)
            self.vocab[lowered] = index
            self.inv_vocab[index] = lowered
        return self.vocab[lowered]

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        for match in self._token_matches(text):
            token = match.group(0)
            ids.append(self._token_to_id(token))
            offsets.append((match.start(), match.end()))
        return ids, offsets

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
    ) -> dict[str, object]:
        ids, offsets = self.encode_with_offsets(text)
        if add_special_tokens:
            ids.append(self.eos_token_id)
            offsets.append((len(text), len(text)))

        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        result: dict[str, object] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        tokens: list[str] = []
        for idx in token_ids:
            token = self.inv_vocab.get(int(idx), "<unk>")
            if skip_special_tokens and token in {self.pad_token, self.eos_token}:
                continue
            tokens.append(token)
        return " ".join(tokens)


class ToyBlock(nn.Module):
    """Identity-style transformer block used to exercise hooks."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Apply a lightweight affine transformation with residual connection."""

        return hidden_states + self.linear(hidden_states)


@dataclass
class ToyOutput:
    logits: torch.Tensor


class ToyModel(nn.Module):
    """Minimal causal LM that keeps computations inexpensive."""

    def __init__(self, hidden_size: int, num_layers: int, vocab_size: int = 128) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList(ToyBlock(hidden_size) for _ in range(num_layers))
        self.vocab_size = vocab_size
        self.output_weight = nn.Parameter(torch.eye(hidden_size, vocab_size))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **_: object,
    ) -> ToyOutput:
        del attention_mask, use_cache
        hidden = F.one_hot(input_ids, num_classes=self.hidden_size).to(torch.float32)
        for layer in self.layers:
            hidden = layer(hidden)
        logits = torch.einsum("bsh,hv->bsv", hidden, self.output_weight)
        return ToyOutput(logits=logits)

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 1,
        **_: object,
    ) -> torch.Tensor:
        del attention_mask
        outputs = input_ids
        for _ in range(max_new_tokens):
            next_token = torch.full(
                (outputs.size(0), 1),
                1,
                dtype=torch.long,
                device=outputs.device,
            )
            outputs = torch.cat([outputs, next_token], dim=1)
        return outputs


class ToyAdapter(BaseModelAdapter):
    """Adapter exposing the :class:`ToyModel` via the protocol interface."""

    name = "ToyAdapter"

    def __init__(self, hidden_size: int = 8, num_layers: int = 2) -> None:
        self.tokenizer = ToyTokenizer()
        self.model = ToyModel(hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    @classmethod
    def load(
        cls,
        model_id: str,
        dtype: torch.dtype,
        device_map: str | dict[str, object],
        *,
        seed: int | None = None,
    ) -> "ToyAdapter":  # pragma: no cover - not exercised in tests
        del model_id, dtype, device_map, seed
        return cls()

    def layer_module(self, layer_idx: int) -> nn.Module:
        return self.model.layers[layer_idx]

    def register_residual_hook(self, layer_idx: int, hook_fn):
        module = self.layer_module(layer_idx)
        return module.register_forward_hook(hook_fn)

    def tokens_for_spans(self, text: str, span_slices: Iterable[tuple[int, int]]) -> list[int]:
        _, offsets = self.tokenizer.encode_with_offsets(text)
        selected: list[int] = []
        for start, end in span_slices:
            indices = [
                idx
                for idx, (tok_start, tok_end) in enumerate(offsets)
                if tok_start < end and tok_end > start
            ]
            if not indices:
                raise ValueError(f"No tokens found for span {(start, end)}")
            selected.extend(indices)
        return list(dict.fromkeys(selected))

    def generate(self, prompt: str, **gen_kwargs: object) -> str:
        inputs = self.tokenizer(prompt)
        tensor_inputs = {
            key: value
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }
        output_ids = self.model.generate(**tensor_inputs, **gen_kwargs)
        return self.tokenizer.decode(output_ids[0].tolist())


def make_toy_adapter(hidden_size: int = 8, num_layers: int = 2) -> ToyAdapter:
    """Return a freshly initialised :class:`ToyAdapter` instance."""

    return ToyAdapter(hidden_size=hidden_size, num_layers=num_layers)
