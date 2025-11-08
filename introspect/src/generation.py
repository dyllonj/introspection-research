"""Shared text generation helpers for adapters and injection utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Mapping, MutableMapping, TYPE_CHECKING

import torch
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
)

if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    from .adapters.base import BaseModelAdapter

__all__ = [
    "DEFAULT_GENERATION_KWARGS",
    "DEFAULT_STOP_SEQUENCES",
    "AllowedPrefixLogitsProcessor",
    "StopSequenceCriteria",
    "apply_generation_defaults",
    "build_chat_prompt",
    "decode_generated_tokens",
    "prepare_generation_controls",
    "prepare_generation_inputs",
    "trim_stop_sequences",
]


DEFAULT_STOP_SEQUENCES: tuple[str, ...] = (
    "\nAssistant:",
    "\nAssistant:\n",
    "\nHuman:",
    "\nHuman:\n",
    "\nUser:",
    "\nUser:\n",
    "\nSystem:",
    "\nSystem:\n",
    "NO_INJECTION\n",
    "INJECTION:\n",
)
"""Canonical stop sequences shared across evaluation prompts."""


DEFAULT_GENERATION_KWARGS: Mapping[str, Any] = {
    "max_new_tokens": 64,
    "stop_sequences": DEFAULT_STOP_SEQUENCES,
}
"""Default generation configuration for short, well-scoped completions."""


def _normalise_chat_message(message: Any) -> dict[str, str]:
    if isinstance(message, Mapping):
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", ""))
    elif isinstance(message, Sequence) and len(message) == 2:
        role = str(message[0]).strip()
        content = str(message[1])
    else:  # pragma: no cover - defensive
        raise TypeError(
            "Chat messages must be mapping role/content pairs or 2-tuples",
        )

    if not role:  # pragma: no cover - defensive
        raise ValueError("Chat message role must be a non-empty string")

    return {"role": role, "content": content}


def _render_chat_fallback(
    messages: Sequence[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    role_labels = {
        "user": "Human",
        "assistant": "Assistant",
        "system": "System",
        "tool": "Tool",
    }

    lines: list[str] = []
    for message in messages:
        role = message["role"].lower()
        label = role_labels.get(role, message["role"].title())
        content = message.get("content", "")
        if content:
            lines.append(f"{label}: {content}")
        else:
            lines.append(f"{label}:")

    if add_generation_prompt:
        lines.append("Assistant:")

    return "\n\n".join(lines)


def _collect_chat_stop_sequences(tokenizer: Any) -> tuple[str, ...]:
    candidates: list[str] = []

    for attr in ("eos_token", "eot_token", "stop_token"):
        token = getattr(tokenizer, attr, None)
        if isinstance(token, str) and token:
            candidates.append(token)

    special_map = getattr(tokenizer, "special_tokens_map", None)
    if isinstance(special_map, Mapping):
        for key in ("eos_token", "eot_token", "stop_token"):
            token = special_map.get(key)
            if isinstance(token, str) and token:
                candidates.append(token)

    additional = getattr(tokenizer, "additional_special_tokens", None)
    if isinstance(additional, Sequence):
        for token in additional:
            if isinstance(token, str) and token:
                if token not in candidates and "eot" in token.lower():
                    candidates.append(token)

    deduped = list(dict.fromkeys(candidates))
    return tuple(deduped)


def build_chat_prompt(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any] | Sequence[Any]],
) -> tuple[str, tuple[str, ...]]:
    """Render ``messages`` with the tokenizer chat template when available."""

    normalised = [_normalise_chat_message(message) for message in messages]
    if not normalised:
        raise ValueError("At least one chat message is required to build a prompt")

    last_message = normalised[-1]
    final_role = last_message.get("role", "")
    final_content = last_message.get("content", "")
    assistant_pending = final_role == "assistant" and final_content == ""
    add_generation_prompt = not assistant_pending

    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(  # type: ignore[call-arg]
            normalised,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        if not isinstance(prompt, str):  # pragma: no cover - defensive
            raise TypeError("Tokenizer.apply_chat_template must return a string")
    else:
        prompt = _render_chat_fallback(
            normalised,
            add_generation_prompt=add_generation_prompt,
        )

    stop_sequences = _collect_chat_stop_sequences(tokenizer)
    return prompt, stop_sequences


def _coerce_stop_sequences(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        sequences = (value,)
    else:
        sequences = tuple(str(seq) for seq in value)
    return tuple(seq for seq in sequences if seq)


def _coerce_allowed_formats(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        formats = (value,)
    else:
        formats = tuple(str(fmt) for fmt in value)
    return tuple(fmt for fmt in formats if fmt)


def _tokenize_to_tuple(tokenizer: Any, text: str) -> tuple[int, ...]:
    tokenized = tokenizer(
        text,
        add_special_tokens=False,
    )
    token_ids = tokenized.get("input_ids")
    if isinstance(token_ids, torch.Tensor):
        raw = token_ids.tolist()
        if raw and isinstance(raw[0], list):
            flattened = [int(token) for sublist in raw for token in sublist]
        else:
            flattened = [int(token) for token in raw]
        return tuple(flattened)
    if isinstance(token_ids, Iterable):
        flattened: list[int] = []
        for token in token_ids:
            if isinstance(token, Iterable) and not isinstance(token, (bytes, bytearray)):
                flattened.extend(int(t) for t in token)
            else:
                flattened.append(int(token))
        return tuple(flattened)
    return ()


class StopSequenceCriteria(StoppingCriteria):
    """Stop generation when any of the provided string sequences appear."""

    def __init__(
        self,
        tokenizer: Any,
        stop_sequences: Sequence[str],
        prompt_len: int,
    ) -> None:
        super().__init__()
        self._tokenizer = tokenizer
        self._prompt_len = prompt_len
        self._stop_sequences = tuple(seq for seq in stop_sequences if seq)
        self._token_sequences: list[tuple[int, ...]] = []
        self._string_sequences: list[str] = []
        for sequence in self._stop_sequences:
            tokens = _tokenize_to_tuple(tokenizer, sequence)
            if tokens:
                self._token_sequences.append(tokens)
            else:
                self._string_sequences.append(sequence)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **_: Any,
    ) -> bool:
        if input_ids.ndim != 2:
            return False

        generated = input_ids[:, self._prompt_len :]
        if generated.shape[1] == 0:
            return False

        generated_tokens = generated.tolist()
        for row_tokens in generated_tokens:
            for sequence in self._token_sequences:
                seq_len = len(sequence)
                if seq_len == 0 or len(row_tokens) < seq_len:
                    continue
                if row_tokens[-seq_len:] == list(sequence):
                    return True

        if not self._string_sequences:
            return False

        decoded = self._tokenizer.decode(
            generated[0],
            skip_special_tokens=False,
        )
        return any(sequence in decoded for sequence in self._string_sequences)


class AllowedPrefixLogitsProcessor(LogitsProcessor):
    """Restrict the vocabulary until a permitted prefix has been emitted."""

    def __init__(
        self,
        tokenizer: Any,
        allowed_formats: Sequence[str],
        prompt_len: int,
    ) -> None:
        self._tokenizer = tokenizer
        self._prompt_len = prompt_len
        self._allowed_formats = tuple(fmt for fmt in allowed_formats if fmt)
        self._allowed_token_sequences: list[tuple[int, ...]] = []
        for fmt in self._allowed_formats:
            tokens = _tokenize_to_tuple(tokenizer, fmt)
            if tokens:
                self._allowed_token_sequences.append(tokens)
        self._max_prefix_len = max(
            (len(sequence) for sequence in self._allowed_token_sequences),
            default=0,
        )

    @property
    def has_prefixes(self) -> bool:
        return bool(self._allowed_token_sequences)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if not self._allowed_token_sequences:
            return scores
        if input_ids.ndim != 2 or scores.ndim != 2:
            return scores

        constrained = scores
        batch_size = input_ids.shape[0]
        for batch_index in range(batch_size):
            generated = input_ids[batch_index, self._prompt_len :]
            generated_len = int(generated.shape[0])
            if generated_len == 0:
                prefix_tokens: list[int] = []
            else:
                prefix_tokens = generated.tolist()

            release_constraints = False
            allowed_next: set[int] = set()

            for sequence in self._allowed_token_sequences:
                seq_len = len(sequence)
                if generated_len >= seq_len:
                    if prefix_tokens[:seq_len] == list(sequence):
                        release_constraints = True
                        break
                    continue
                if prefix_tokens == list(sequence[:generated_len]):
                    allowed_next.add(sequence[generated_len])

            if release_constraints or not allowed_next:
                continue

            mask_value = torch.finfo(constrained.dtype).min
            updated = torch.full_like(constrained[batch_index], mask_value)
            indices = sorted(allowed_next)
            updated[indices] = constrained[batch_index, indices]
            constrained[batch_index] = updated

        return constrained


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
    ignored_keys = {"token_type_ids"}
    tensor_inputs: dict[str, torch.Tensor] = {}
    for name, value in tokenizer_inputs.items():
        if name in ignored_keys:
            continue
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

    stop_value = gen_kwargs.get("stop_sequences", ())
    stop_sequences = (
        _coerce_stop_sequences(stop_value) if stop_value else DEFAULT_STOP_SEQUENCES
    )
    gen_kwargs["stop_sequences"] = stop_sequences

    if "allowed_formats" in gen_kwargs:
        gen_kwargs["allowed_formats"] = _coerce_allowed_formats(gen_kwargs["allowed_formats"])

    return stop_sequences


def _collect_stopping_criteria(value: Any) -> list[StoppingCriteria]:
    if value is None:
        return []
    if isinstance(value, StoppingCriteriaList):
        return list(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [criterion for criterion in value]
    return [value]


def _collect_logits_processors(value: Any) -> list[LogitsProcessor]:
    if value is None:
        return []
    if isinstance(value, LogitsProcessorList):
        return list(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [processor for processor in value]
    return [value]


def prepare_generation_controls(
    tokenizer: Any,
    prompt_len: int,
    gen_kwargs: MutableMapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Register stopping criteria and logits processors for generation."""

    stop_sequences = tuple(gen_kwargs.get("stop_sequences", ()))
    allowed_formats = tuple(gen_kwargs.get("allowed_formats", ()))

    stopping_entries = []
    stopping_entries.extend(_collect_stopping_criteria(gen_kwargs.pop("stopping_criteria", None)))
    if stop_sequences:
        stopping_entries.append(StopSequenceCriteria(tokenizer, stop_sequences, prompt_len))
    if stopping_entries:
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList(stopping_entries)

    processor_entries: list[LogitsProcessor] = []
    processor_entries.extend(_collect_logits_processors(gen_kwargs.pop("logits_processor", None)))
    processor_entries.extend(_collect_logits_processors(gen_kwargs.pop("logits_processors", None)))
    if allowed_formats:
        format_processor = AllowedPrefixLogitsProcessor(
            tokenizer,
            allowed_formats,
            prompt_len,
        )
        if format_processor.has_prefixes:
            processor_entries.append(format_processor)
    if processor_entries:
        gen_kwargs["logits_processor"] = LogitsProcessorList(processor_entries)

    gen_kwargs.pop("stop_sequences", None)
    gen_kwargs.pop("allowed_formats", None)

    return stop_sequences, allowed_formats


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
