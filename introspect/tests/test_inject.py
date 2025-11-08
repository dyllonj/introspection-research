from __future__ import annotations

import torch

from introspect.src.inject import (
    InjectionSpec,
    _build_modifier,
    describe_injection_spec,
    inject_once,
    injection_context,
    token_positions_after,
)
from introspect.src.prompts import render_task_a_detection_prompt
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

    result = inject_once(
        adapter,
        prompt,
        spec,
        gen_kwargs={
            "stop_sequences": [" assistant:"],
            "max_new_tokens": 5,
        },
        enable_injection=False,
    )

    response = result.text
    assert response.strip() == "no_injection"
    assert "assistant:" not in response.lower()
    assert recorded_kwargs.get("pad_token_id") == adapter.tokenizer.pad_token_id
    assert recorded_kwargs.get("eos_token_id") == adapter.tokenizer.eos_token_id
    assert result.generation == {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "max_new_tokens": 5,
        "do_sample": False,
        "stop_sequences": [" assistant:"],
    }
    assert result.injection_spec["layer_idx"] == 0
    assert result.injection_spec["alpha"] == 1.0
    assert result.injection_spec["token_positions"] == [0]
    assert result.injection_spec["apply_on_input"] is False
    assert result.injection_spec["apply_to_generated"] is False
    assert result.injection_spec["vector_dim"] == adapter.hidden_size
    assert result.injection_spec["vector_norm"] == 0.0


def test_token_positions_after_task_a_prompt_suffix_range() -> None:
    adapter = make_toy_adapter(hidden_size=4, num_layers=1)
    prompt = render_task_a_detection_prompt()
    marker = "Trial 1:"

    positions = token_positions_after(adapter, prompt, marker)

    marker_index = prompt.index(marker)
    newline_index = prompt.rfind("\n", 0, marker_index)
    span_start = newline_index + 1 if newline_index != -1 else 0
    expected = adapter.tokens_for_spans(prompt, [(span_start, len(prompt))])

    assert positions == expected

    token_ids, offsets = adapter.tokenizer.encode_with_offsets(prompt)
    del token_ids
    first_span = offsets[positions[0]]
    assert prompt[first_span[0] : first_span[1]].lower().startswith("trial")


def test_suffix_sentinel_injects_prefill_and_generated_tokens() -> None:
    adapter = make_toy_adapter(hidden_size=6, num_layers=1)
    vector = torch.zeros(adapter.hidden_size)
    vector[0] = 3.0
    spec = InjectionSpec(
        layer_idx=0,
        alpha=1.0,
        vector=vector,
        token_positions=[0, "suffix"],
    )

    modifier = _build_modifier(spec)

    prefill_hidden = torch.zeros((1, 5, adapter.hidden_size))
    updated_prefill, changed_prefill = modifier(prefill_hidden)
    assert changed_prefill
    assert torch.allclose(updated_prefill[0, 0], vector)
    assert torch.allclose(updated_prefill[0, 4], vector)
    assert torch.allclose(updated_prefill[0, 1], torch.zeros_like(vector))

    stream_hidden = torch.zeros((1, 1, adapter.hidden_size))
    updated_stream, changed_stream = modifier(stream_hidden)
    assert changed_stream
    assert torch.allclose(updated_stream[0, 0], vector)


def test_describe_injection_spec_serializes_suffix() -> None:
    vector = torch.ones(4)
    spec = InjectionSpec(
        layer_idx=1,
        alpha=0.5,
        vector=vector,
        token_positions=[-1],
        apply_on_input=True,
        apply_to_generated=True,
    )

    metadata = describe_injection_spec(spec)

    assert metadata["token_positions"] == ["suffix"]
    assert metadata["apply_on_input"] is True
    assert metadata["apply_to_generated"] is True


def test_apply_to_generated_reuses_suffix_positions() -> None:
    adapter = make_toy_adapter(hidden_size=5, num_layers=1)
    vector = torch.zeros(adapter.hidden_size)
    vector[1] = 2.5
    spec = InjectionSpec(
        layer_idx=0,
        alpha=1.0,
        vector=vector,
        token_positions=[1, 2, 3],
        apply_to_generated=True,
    )

    modifier = _build_modifier(spec)

    prefill_hidden = torch.zeros((1, 6, adapter.hidden_size))
    updated_prefill, changed_prefill = modifier(prefill_hidden)
    assert changed_prefill
    for index in (1, 2, 3):
        assert torch.allclose(updated_prefill[0, index], vector)

    stream_hidden = torch.zeros((1, 1, adapter.hidden_size))
    updated_stream, changed_stream = modifier(stream_hidden)
    assert changed_stream
    assert torch.allclose(updated_stream[0, 0], vector)


def test_inject_once_boosts_generated_token_probability(monkeypatch) -> None:
    adapter = make_toy_adapter(hidden_size=8, num_layers=2)
    prompt = "alpha beta"

    token_ids = adapter.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    assert isinstance(token_ids, torch.Tensor)
    last_prompt_token = int(token_ids[0, -1])
    assert last_prompt_token < adapter.hidden_size

    vector = torch.zeros(adapter.hidden_size)
    vector[last_prompt_token] = 6.0

    spec = InjectionSpec(
        layer_idx=adapter.num_layers - 1,
        alpha=1.0,
        vector=vector,
        token_positions=["suffix"],
        apply_to_generated=True,
    )

    captured: dict[str, list[torch.Tensor]] = {}

    def _patched_generate(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        max_new_tokens: int = 2,
        **_: object,
    ) -> torch.Tensor:
        del attention_mask
        history: list[torch.Tensor] = []
        outputs = input_ids
        for _ in range(int(max_new_tokens)):
            result = adapter.model(input_ids=outputs)
            logits = result.logits[:, -1, :].detach().clone()
            history.append(logits)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            outputs = torch.cat([outputs, next_token], dim=1)
        captured["history"] = history
        return outputs

    monkeypatch.setattr(adapter.model, "generate", _patched_generate)

    def _run(enable: bool) -> torch.Tensor:
        captured["history"] = []
        inject_once(
            adapter,
            prompt,
            spec,
            gen_kwargs={"max_new_tokens": 2},
            enable_injection=enable,
        )
        history = captured.get("history")
        assert history, "patched generate must record logits"
        return torch.stack(history)

    control_logits = _run(enable=False)
    injected_logits = _run(enable=True)

    control_probs = control_logits.softmax(dim=-1)
    injected_probs = injected_logits.softmax(dim=-1)

    control_generated = control_probs[1, 0, last_prompt_token]
    injected_generated = injected_probs[1, 0, last_prompt_token]

    assert injected_generated > control_generated


def test_inject_once_removes_hook_exactly_once(monkeypatch) -> None:
    adapter = make_toy_adapter(hidden_size=4, num_layers=1)
    prompt = "alpha"
    spec = InjectionSpec(
        layer_idx=0,
        alpha=1.0,
        vector=torch.zeros(adapter.hidden_size),
        token_positions=[0],
    )

    remove_calls: list[int] = []

    class _DummyHandle:
        def __init__(self) -> None:
            self._removed = False

        def remove(self) -> None:
            if self._removed:
                raise AssertionError("Hook handle removed more than once")
            self._removed = True
            remove_calls.append(1)

    def _fake_register(layer_idx: int, hook_fn: object) -> _DummyHandle:
        assert layer_idx == spec.layer_idx
        assert callable(hook_fn)
        return _DummyHandle()

    monkeypatch.setattr(adapter, "register_residual_hook", _fake_register)

    inject_once(
        adapter,
        prompt,
        spec,
        gen_kwargs={"max_new_tokens": 1},
    )

    assert remove_calls == [1]
