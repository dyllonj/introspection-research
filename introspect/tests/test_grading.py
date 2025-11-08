from __future__ import annotations

import pytest

from introspect.src.grading import (
    DeterministicJudge,
    injection_format_precision,
    is_valid_injection_report,
    grade_injection_detection,
    grade_intent,
    grade_task_b_choice,
    grade_task_b_repetition,
    grade_task_b_thought,
    parse_injection_report,
    parse_intent,
    parse_task_b,
)


def test_parse_and_grade_injection_detection() -> None:
    report = parse_injection_report("INJECTION: Apple")
    metrics = grade_injection_detection(expected_word="apple", report=report)
    assert metrics["tp"] == 1 and metrics["matched"]

    miss_report = parse_injection_report("NO_INJECTION")
    miss_metrics = grade_injection_detection(expected_word="apple", report=miss_report)
    assert miss_metrics["fn"] == 1

    invalid = parse_injection_report("unexpected format")
    invalid_metrics = grade_injection_detection(expected_word=None, report=invalid)
    assert invalid_metrics["notes"] == "invalid_format"


def test_task_b_parsers_and_graders() -> None:
    thought = parse_task_b("THOUGHT: apple", mode="thought")
    thought_grade = grade_task_b_thought(expected_word="apple", outcome=thought)
    assert thought_grade["tp"] == 1

    repeat = parse_task_b(
        "REPEAT: Hello world",
        mode="repeat",
        expected_sentence="Hello world",
    )
    repeat_grade = grade_task_b_repetition(expected_sentence="Hello world", outcome=repeat)
    assert repeat_grade["tp"] == 1

    choice = parse_task_b(
        "CHOICE: 2",
        mode="choice",
        option_map={1: "alpha", 2: "beta"},
        num_options=2,
    )
    choice_grade = grade_task_b_choice(
        expected_index=2,
        option_map={1: "alpha", 2: "beta"},
        outcome=choice,
    )
    assert choice_grade["tp"] == 1

    with pytest.raises(ValueError):
        parse_task_b("Assistant: ???", mode="unknown")


def test_intent_parsing_and_grading() -> None:
    intent_yes = parse_intent("INTENT: YES")
    grade_yes = grade_intent(expected_yes=True, intent=intent_yes)
    assert grade_yes["tp"] == 1

    intent_no = parse_intent("INTENT: NO")
    grade_no = grade_intent(expected_yes=False, intent=intent_no)
    assert grade_no["tp"] == 1

    invalid = parse_intent("maybe")
    grade_invalid = grade_intent(expected_yes=True, intent=invalid)
    assert grade_invalid["notes"] == "invalid"


def test_deterministic_judge_returns_constant() -> None:
    judge = DeterministicJudge(score_value=0.42)
    assert judge.score(prompt="p", completion="c") == pytest.approx(0.42)


def test_injection_format_precision_helpers() -> None:
    reports = [
        parse_injection_report("INJECTION: apple"),
        parse_injection_report("NO_INJECTION"),
        parse_injection_report("unexpected"),
    ]
    valid = sum(1 for report in reports if is_valid_injection_report(report))
    precision = injection_format_precision(valid, len(reports))
    assert precision == pytest.approx(2 / 3)
