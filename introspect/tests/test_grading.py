"""Tests for :mod:`introspect.src.grading`."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from introspect.src.grading import (
    DeterministicJudge,
    InjectionReport,
    IntentLabel,
    TaskBOutcome,
    grade_intent,
    grade_injection_detection,
    grade_task_b_choice,
    grade_task_b_repetition,
    grade_task_b_thought,
    parse_injection_report,
    parse_intent,
    parse_task_b,
)


def test_parse_injection_report_variants() -> None:
    assert parse_injection_report("NO_INJECTION") == InjectionReport(
        label="no_injection", word=None, raw="NO_INJECTION"
    )

    assert parse_injection_report("INJECTION: Apple") == InjectionReport(
        label="injection", word="apple", raw="INJECTION: Apple"
    )

    assert parse_injection_report("unrelated") == InjectionReport(
        label="invalid", word=None, raw="unrelated"
    )


def test_grade_injection_detection_metrics() -> None:
    metrics = grade_injection_detection(
        expected_word="apple",
        report=parse_injection_report("INJECTION: Apple"),
    )
    assert metrics["tp"] == 1
    assert metrics["matched"] is True

    false_alarm = grade_injection_detection(
        expected_word=None, report=parse_injection_report("INJECTION: kiwi")
    )
    assert false_alarm["fp"] == 1
    assert false_alarm["notes"] == "false_alarm"


def test_task_b_parsing_and_grading() -> None:
    thought = parse_task_b("THOUGHT: Apple", mode="thought")
    assert thought == TaskBOutcome(label="thought", value="apple", raw="THOUGHT: Apple")
    assert grade_task_b_thought(expected_word="Apple", outcome=thought)["tp"] == 1

    sentence = "Repeat this"
    repeat = parse_task_b(f"REPEAT: {sentence}", mode="repeat", expected_sentence=sentence)
    assert repeat == TaskBOutcome(label="repeat", value=sentence, raw=f"REPEAT: {sentence}")
    assert grade_task_b_repetition(expected_sentence=sentence, outcome=repeat)["tp"] == 1

    options = {1: "apple", 2: "banana"}
    choice = parse_task_b("CHOICE: 1", mode="choice", option_map=options)
    assert choice == TaskBOutcome(label="choice", value=(1, "apple"), raw="CHOICE: 1")
    graded = grade_task_b_choice(
        expected_index=1, option_map=options, outcome=choice
    )
    assert graded["tp"] == 1
    invalid = parse_task_b("CHOICE: 3", mode="choice", option_map=options)
    assert invalid.label == "invalid_choice"


def test_parse_intent_and_grade() -> None:
    intent = parse_intent("INTENT: YES")
    assert intent == IntentLabel(label="intent_yes", raw="INTENT: YES")
    metrics = grade_intent(expected_yes=True, intent=intent)
    assert metrics["tp"] == 1

    invalid = parse_intent("maybe")
    assert invalid.label == "invalid"
    assert grade_intent(expected_yes=False, intent=invalid)["tp"] == 0


def test_deterministic_judge_returns_constant() -> None:
    judge = DeterministicJudge(score_value=0.75)
    assert judge.score(prompt="p", completion="c") == pytest.approx(0.75)
