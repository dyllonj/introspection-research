from __future__ import annotations

import torch

from introspect.src.inject import InjectionSpec, injection_context
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
