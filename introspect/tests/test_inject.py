from __future__ import annotations

import torch

from introspect.src.inject import InjectionSpec, inject_once, injection_context
from introspect.tests.utils import make_toy_adapter


def _prepare_inputs(adapter, prompt: str) -> dict[str, torch.Tensor]:
    tokenized = adapter.tokenizer(prompt)
    return {
        key: tensor
        for key, tensor in tokenized.items()
        if isinstance(tensor, torch.Tensor)
    }


def test_injection_changes_logits() -> None:
    adapter = make_toy_adapter(hidden_size=8, num_layers=2)
    prompt = "alpha beta"
    inputs = _prepare_inputs(adapter, prompt)

    with torch.no_grad():
        baseline = adapter.model(**inputs).logits.detach().clone()

    last_token_index = inputs["input_ids"].shape[-1] - 2
    vector = torch.zeros(adapter.hidden_size)
    vector[0] = 5.0
    spec = InjectionSpec(
        layer_idx=adapter.num_layers - 1,
        alpha=1.0,
        vector=vector,
        token_positions=[last_token_index],
    )

    with injection_context(adapter, spec, enable=True):
        with torch.no_grad():
            injected = adapter.model(**inputs).logits.detach().clone()

    with injection_context(adapter, spec, enable=False):
        with torch.no_grad():
            control = adapter.model(**inputs).logits.detach().clone()

    baseline_slice = baseline[0, last_token_index]
    injected_slice = injected[0, last_token_index]
    assert not torch.allclose(injected_slice, baseline_slice)
    assert torch.allclose(control, baseline)


def test_inject_once_decodes_new_tokens_and_applies_stops(monkeypatch) -> None:
    adapter = make_toy_adapter(hidden_size=8, num_layers=2)
    prompt = "System: check\nAssistant:"
    vector = torch.zeros(adapter.hidden_size)
    spec = InjectionSpec(
        layer_idx=0,
        alpha=1.0,
        vector=vector,
        token_positions=[0],
    )

    completion_ids, _ = adapter.tokenizer.encode_with_offsets(
        "NO_INJECTION Assistant: trailing"
    )
    completion_ids.append(adapter.tokenizer.eos_token_id)
    completion = torch.tensor([completion_ids], dtype=torch.long)

    recorded_kwargs: dict[str, object] = {}

    def _fake_generate(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        del attention_mask
        recorded_kwargs.clear()
        recorded_kwargs.update(kwargs)
        return torch.cat(
            [input_ids, completion.to(input_ids.device)],
            dim=1,
        )

    monkeypatch.setattr(adapter.model, "generate", _fake_generate)

    response = inject_once(
        adapter,
        prompt,
        spec,
        gen_kwargs={
            "stop_sequences": [" assistant:"],
            "max_new_tokens": 5,
        },
        enable_injection=False,
    )

    assert response.strip() == "no_injection"
    assert "assistant:" not in response.lower()
    assert recorded_kwargs.get("pad_token_id") == adapter.tokenizer.pad_token_id
    assert recorded_kwargs.get("eos_token_id") == adapter.tokenizer.eos_token_id
