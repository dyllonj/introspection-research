from __future__ import annotations

import os

import pytest
import torch
from requests import exceptions as requests_exceptions

from introspect.src.adapters.base import BaseModelAdapter
from introspect.src.adapters.llama import LlamaAdapter
from introspect.tests.utils import make_toy_adapter


@pytest.fixture(scope="session")
def adapter_under_test() -> BaseModelAdapter:
    if os.environ.get("INTROSPECT_TEST_REAL_ADAPTER") == "1":
        pytest.importorskip("transformers")
        model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        try:
            return LlamaAdapter.load(model_id, dtype=torch.float32, device_map="cpu", seed=0)
        except (requests_exceptions.RequestException, OSError):
            pytest.skip("Unable to download reference adapter in this environment")

    return make_toy_adapter(hidden_size=8, num_layers=2)


def test_layer_module_indices(adapter_under_test: BaseModelAdapter) -> None:
    adapter = adapter_under_test
    assert adapter.num_layers > 0
    first_block = adapter.layer_module(0)
    assert isinstance(first_block, torch.nn.Module)
    with pytest.raises(IndexError):
        adapter.layer_module(adapter.num_layers)


def test_residual_hook_modifies_logits(adapter_under_test: BaseModelAdapter) -> None:
    adapter = adapter_under_test
    tokenized = adapter.tokenizer("A test prompt.", return_tensors="pt")
    inputs = {key: tensor for key, tensor in tokenized.items() if isinstance(tensor, torch.Tensor)}
    with torch.no_grad():
        base_logits = adapter.model(**inputs).logits.detach().clone()

    def hook_fn(_module, _inputs, output):
        delta = torch.ones_like(output) * 0.5
        return output + delta

    handle = adapter.register_residual_hook(adapter.num_layers - 1, hook_fn)
    try:
        with torch.no_grad():
            modified = adapter.model(**inputs).logits.detach()
    finally:
        handle.remove()

    assert not torch.allclose(modified, base_logits)
