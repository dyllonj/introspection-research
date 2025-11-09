from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from introspect.src.eval_A_injected_report import TaskAConfig
from introspect.src.eval_A_injected_report import run as run_task_a
from introspect.src.eval_common import LoadedAdapter
from introspect.src.grading import (
    injection_format_precision,
    is_valid_injection_report,
    parse_injection_report,
)
from introspect.src.inject import (
    DEFAULT_STOP_SEQUENCES,
    InjectionResult,
    describe_injection_spec,
)
from introspect.src.vectors import ConceptWordSet
from introspect.tests.utils import make_toy_adapter


EXPECTED_GENERATION = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "max_new_tokens": 64,
    "do_sample": False,
    "stop_sequences": list(DEFAULT_STOP_SEQUENCES),
    "allowed_formats": ["NO_INJECTION", "INJECTION: "],
}


def _read_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_task_a_chat_prompt_sanity(tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture) -> None:
    adapter = make_toy_adapter(hidden_size=6, num_layers=2)
    loaded = LoadedAdapter(
        adapter=adapter,
        adapter_name="ToyAdapter",
        dtype=adapter.model.output_weight.dtype,
        device_map="cpu",
    )
    word_set = ConceptWordSet(targets=["alpha"], baselines=[f"base{i}" for i in range(5)])

    completions: list[str] = []
    prompts: list[str] = []

    monkeypatch.setattr(
        "introspect.src.eval_A_injected_report.load_adapter_from_registry",
        lambda *args, **kwargs: loaded,
    )
    monkeypatch.setattr(
        "introspect.src.eval_A_injected_report.ensure_vector",
        lambda **_kwargs: adapter.model.output_weight[:, 0].clone().detach(),
    )
    monkeypatch.setattr(
        "introspect.src.eval_A_injected_report.load_words",
        lambda *_args, **_kwargs: word_set,
    )
    monkeypatch.setattr(
        "introspect.src.eval_A_injected_report.token_positions_for_substring",
        lambda *_args, **_kwargs: [0],
    )
    monkeypatch.setattr(
        "introspect.src.eval_A_injected_report.setup_logging",
        lambda *args, **kwargs: None,
    )

    def _inject_stub(_adapter, prompt: str, spec, *, enable_injection: bool = True, **_kwargs):
        prompts.append(prompt)
        text = "INJECTION: alpha" if enable_injection else "NO_INJECTION"
        completions.append(text)
        return InjectionResult(
            text=text,
            generation=dict(EXPECTED_GENERATION),
            injection_spec=describe_injection_spec(spec),
        )

    monkeypatch.setattr("introspect.src.eval_A_injected_report.inject_once", _inject_stub)

    results_path = tmp_path / "task_a.jsonl"
    config = TaskAConfig(
        model="toy",
        adapter="ToyAdapter",
        dtype=None,
        device_map=None,
        layers=[0],
        alphas=[2.0],
        n_concepts=1,
        seed=42,
        words_file=Path("unused"),
        cache_dir=tmp_path,
        results_path=results_path,
        overwrite=True,
        baseline_sample=None,
        prompt_template="Think about the concept '{word}'.",
        deterministic=True,
        use_llm_judge=False,
    )

    with caplog.at_level(logging.DEBUG, logger="introspect.src.eval_A_injected_report"):
        run_task_a(config)

    records = _read_records(results_path)
    assert len(records) == 4
    assert len(prompts) == len(records)
    assert completions == [record["response"] for record in records]

    responses_by_kind = {record["vector_kind"]: record["response"] for record in records}
    assert responses_by_kind["control"] == "NO_INJECTION"
    for kind in ("target", "negative", "random"):
        assert responses_by_kind[kind] == "INJECTION: alpha"

    for record in records:
        response = str(record["response"])
        assert response in {"NO_INJECTION", "INJECTION: alpha"}
        assert "Human:" not in response
        parsed = record["parsed"]
        assert parsed["label"] in {"no_injection", "injection"}
        spec = record["injection_spec"]
        assert spec["token_positions"] == [0, "suffix"]
        assert spec["apply_to_generated"] is True
        assert record["generation"]["allowed_formats"] == ["NO_INJECTION", "INJECTION: "]

    reports = [parse_injection_report(record["response"]) for record in records]
    valid = sum(1 for report in reports if is_valid_injection_report(report))
    precision = injection_format_precision(valid, len(reports))
    assert precision == pytest.approx(1.0)

    completion_logs = [
        message for message in caplog.messages if "Task A completion" in message
    ]
    assert len(completion_logs) == len(records)
    assert "echoed prompt" not in caplog.text

    # Ensure the rendered prompt uses the chat template with an assistant suffix.
    assert any("Assistant:" in prompt for prompt in prompts)
