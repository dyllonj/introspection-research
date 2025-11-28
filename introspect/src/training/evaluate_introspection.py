"""Evaluation script to compare introspection before/after training."""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..eval_common import (
    ensure_vector,
    load_adapter_from_registry,
    load_words,
    select_target_words,
)
from ..generation import build_chat_prompt
from ..grading import grade_injection_detection, llm_judge_task_a, parse_injection_report
from ..inject import (
    DEFAULT_GENERATION_KWARGS,
    InjectionSpec,
    inject_once,
    resolve_injection_positions,
)
from ..io_utils import seed_everything, setup_logging
from ..prompts import task_a_paper_messages
from ..vectors import DEFAULT_WORDS_PATH

LOGGER = logging.getLogger(__name__)


@dataclass
class IntrospectionMetrics:
    true_positive_rate: float = 0.0
    false_positive_rate: float = 0.0
    true_negative_rate: float = 0.0
    false_negative_rate: float = 0.0
    concept_accuracy: float = 0.0
    net_introspection_score: float = 0.0
    f1_detection: float = 0.0
    n_injection_trials: int = 0
    n_control_trials: int = 0
    n_correct_detection: int = 0
    n_correct_identification: int = 0
    n_false_alarms: int = 0
    layer_metrics: dict[int, dict[str, float]] = field(default_factory=dict)


@dataclass
class EvalConfig:
    model: str
    model_b: str | None = None
    adapter: str | None = None
    dtype: str | None = None
    n_concepts: int = 20
    layers: list[int] | None = None
    alphas: list[float] = field(default_factory=lambda: [2.0, 4.0])
    words_file: Path = Path(DEFAULT_WORDS_PATH)
    cache_dir: Path = Path("results/vectors")
    output: Path = Path("results/eval_introspection.json")
    use_llm_judge: bool = True
    seed: int = 42
    holdout_concepts: list[str] | None = None
    injection_mode: str = "prefix"
    assistant_marker: str | None = None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-b", help="Second model for comparison")
    parser.add_argument("--adapter")
    parser.add_argument("--dtype")
    parser.add_argument("--n-concepts", type=int, default=20)
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--alphas", type=float, nargs="+", default=[2.0, 4.0])
    parser.add_argument("--words-file", type=Path, default=DEFAULT_WORDS_PATH)
    parser.add_argument("--cache-dir", type=Path, default=Path("results/vectors"))
    parser.add_argument("--output", type=Path, default=Path("results/eval_introspection.json"))
    parser.add_argument("--no-llm-judge", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-list", type=Path, help="Optional JSON/line file of held-out concepts")
    parser.add_argument("--injection-mode", choices=["prefix", "suffix"], default="prefix")
    parser.add_argument("--assistant-marker", help="Override detected assistant marker")
    return parser


def _parse_config(argv: Sequence[str] | None = None) -> EvalConfig:
    parser = _build_parser()
    args = parser.parse_args(argv)

    holdout_concepts: list[str] | None = None
    if args.holdout_list is not None and args.holdout_list.exists():
        content = args.holdout_list.read_text(encoding="utf-8").splitlines()
        holdout_concepts = [line.strip() for line in content if line.strip()]

    return EvalConfig(
        model=args.model,
        model_b=args.model_b,
        adapter=args.adapter,
        dtype=args.dtype,
        n_concepts=args.n_concepts,
        layers=args.layers,
        alphas=args.alphas,
        words_file=args.words_file,
        cache_dir=args.cache_dir,
        output=args.output,
        use_llm_judge=not args.no_llm_judge,
        seed=args.seed,
        holdout_concepts=holdout_concepts,
        injection_mode=args.injection_mode,
        assistant_marker=args.assistant_marker,
    )


def _infer_layers(model_config: object) -> list[int]:
    n_layers = getattr(model_config, "num_hidden_layers", 32)
    center = (2 * n_layers) // 3
    return [
        max(0, center - 4),
        center - 2,
        center,
        center + 2,
        min(n_layers - 1, center + 4),
    ]


def evaluate_model(adapter, config: EvalConfig, model_id: str) -> tuple[IntrospectionMetrics, list[dict[str, Any]]]:
    rng = random.Random(config.seed)

    word_set = load_words(config.words_file)
    targets = select_target_words(word_set, limit=config.n_concepts, seed=config.seed)
    if config.holdout_concepts:
        targets = [t for t in targets if t not in set(config.holdout_concepts)]
    baseline_words = list(word_set.iter_baselines())

    layers = config.layers or _infer_layers(adapter.model.config)
    LOGGER.info("Evaluating on layers: %s", layers)

    messages = task_a_paper_messages()
    prompt, _ = build_chat_prompt(adapter.tokenizer, messages)
    token_positions, suffix_start = resolve_injection_positions(
        adapter,
        prompt,
        mode=config.injection_mode,
        assistant_marker=config.assistant_marker,
    )
    apply_to_generated = config.injection_mode == "suffix"
    if config.injection_mode == "suffix":
        token_positions = ["suffix"]

    trial_question = messages[-2]["content"]

    all_trials: list[dict[str, Any]] = []
    metrics = IntrospectionMetrics()
    layer_results: dict[int, dict[str, list[int]]] = {
        layer: {"tp": [], "fp": [], "tn": [], "fn": [], "concept_correct": []}
        for layer in layers
    }

    for layer_idx in layers:
        for concept in targets:
            vector = ensure_vector(
                adapter=adapter,
                model_id=model_id,
                layer_idx=layer_idx,
                word=concept,
                cache_dir=config.cache_dir,
                baseline_words=baseline_words,
                prompt_template="Tell me about {word}.",
                baseline_sample_size=50,
                rng=rng,
            )

            for alpha in config.alphas:
                spec = InjectionSpec(
                    layer_idx=layer_idx,
                    alpha=alpha,
                    vector=vector,
                    token_positions=token_positions,
                    apply_to_generated=apply_to_generated,
                )

                result_inject = inject_once(
                    adapter,
                    prompt,
                    spec,
                    gen_kwargs=DEFAULT_GENERATION_KWARGS,
                    enable_injection=True,
                )

                if config.use_llm_judge:
                    report_inject, judge_text, judge_json = llm_judge_task_a(
                        adapter,
                        trial_question=trial_question,
                        subject_response=result_inject.text,
                    )
                else:
                    report_inject = parse_injection_report(result_inject.text)
                    judge_text = ""
                    judge_json = {}

                grading_inject = grade_injection_detection(
                    expected_word=concept,
                    report=report_inject,
                )

                detected = grading_inject.get("tp", 0) == 1
                correct_concept = grading_inject.get("matched", False)

                if detected:
                    metrics.n_correct_detection += 1
                    layer_results[layer_idx]["tp"].append(1)
                    layer_results[layer_idx]["fn"].append(0)
                    if correct_concept:
                        metrics.n_correct_identification += 1
                        layer_results[layer_idx]["concept_correct"].append(1)
                    else:
                        layer_results[layer_idx]["concept_correct"].append(0)
                else:
                    layer_results[layer_idx]["tp"].append(0)
                    layer_results[layer_idx]["fn"].append(1)
                    layer_results[layer_idx]["concept_correct"].append(0)

                metrics.n_injection_trials += 1

                all_trials.append({
                    "model": model_id,
                    "layer": layer_idx,
                    "alpha": alpha,
                    "concept": concept,
                    "condition": "injection",
                    "response": result_inject.text,
                    "detected": detected,
                    "correct_concept": correct_concept,
                    "grading": grading_inject,
                    "judge_json": judge_json,
                    "positions": token_positions,
                    "suffix_start": suffix_start,
                })

                result_control = inject_once(
                    adapter,
                    prompt,
                    spec,
                    gen_kwargs=DEFAULT_GENERATION_KWARGS,
                    enable_injection=False,
                )

                if config.use_llm_judge:
                    report_control, _, control_json = llm_judge_task_a(
                        adapter,
                        trial_question=trial_question,
                        subject_response=result_control.text,
                    )
                else:
                    report_control = parse_injection_report(result_control.text)
                    control_json = {}

                grading_control = grade_injection_detection(
                    expected_word=None,
                    report=report_control,
                )

                false_alarm = grading_control.get("fp", 0) == 1

                if false_alarm:
                    metrics.n_false_alarms += 1
                    layer_results[layer_idx]["fp"].append(1)
                    layer_results[layer_idx]["tn"].append(0)
                else:
                    layer_results[layer_idx]["fp"].append(0)
                    layer_results[layer_idx]["tn"].append(1)

                metrics.n_control_trials += 1

                all_trials.append({
                    "model": model_id,
                    "layer": layer_idx,
                    "alpha": alpha,
                    "concept": concept,
                    "condition": "control",
                    "response": result_control.text,
                    "false_alarm": false_alarm,
                    "grading": grading_control,
                    "positions": token_positions,
                    "suffix_start": suffix_start,
                    "judge_json": control_json,
                })

    if metrics.n_injection_trials > 0:
        metrics.true_positive_rate = metrics.n_correct_detection / metrics.n_injection_trials
        metrics.false_negative_rate = 1 - metrics.true_positive_rate

    if metrics.n_control_trials > 0:
        metrics.false_positive_rate = metrics.n_false_alarms / metrics.n_control_trials
        metrics.true_negative_rate = 1 - metrics.false_positive_rate

    if metrics.n_correct_detection > 0:
        metrics.concept_accuracy = metrics.n_correct_identification / metrics.n_correct_detection

    metrics.net_introspection_score = metrics.true_positive_rate - metrics.false_positive_rate

    precision = (
        metrics.n_correct_detection / (metrics.n_correct_detection + metrics.n_false_alarms)
        if (metrics.n_correct_detection + metrics.n_false_alarms) > 0 else 0
    )
    recall = metrics.true_positive_rate
    metrics.f1_detection = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0
    )

    for layer_idx, results in layer_results.items():
        n_inject = len(results["tp"])
        n_control = len(results["fp"])

        tpr = sum(results["tp"]) / n_inject if n_inject > 0 else 0
        fpr = sum(results["fp"]) / n_control if n_control > 0 else 0
        concept_acc = sum(results["concept_correct"]) / sum(results["tp"]) if sum(results["tp"]) > 0 else 0

        metrics.layer_metrics[layer_idx] = {
            "tpr": tpr,
            "fpr": fpr,
            "net_score": tpr - fpr,
            "concept_accuracy": concept_acc,
            "n_trials": n_inject + n_control,
        }

    return metrics, all_trials


def run(config: EvalConfig) -> None:
    setup_logging()
    seed_everything(config.seed)

    results: dict[str, Any] = {
        "config": asdict(config),
        "models": {},
    }

    LOGGER.info("Evaluating model: %s", config.model)
    loaded_a = load_adapter_from_registry(
        config.model,
        adapter_name=config.adapter,
        dtype=config.dtype,
        seed=config.seed,
    )
    metrics_a, trials_a = evaluate_model(loaded_a.adapter, config, config.model)
    results["models"][config.model] = {
        "metrics": asdict(metrics_a),
        "trials": trials_a,
    }

    if config.model_b:
        LOGGER.info("Evaluating comparison model: %s", config.model_b)
        loaded_b = load_adapter_from_registry(
            config.model_b,
            adapter_name=config.adapter,
            dtype=config.dtype,
            seed=config.seed,
        )
        metrics_b, trials_b = evaluate_model(loaded_b.adapter, config, config.model_b)
        results["models"][config.model_b] = {
            "metrics": asdict(metrics_b),
            "trials": trials_b,
        }

        delta_net = metrics_b.net_introspection_score - metrics_a.net_introspection_score
        delta_tpr = metrics_b.true_positive_rate - metrics_a.true_positive_rate
        delta_fpr = metrics_b.false_positive_rate - metrics_a.false_positive_rate

        results["comparison"] = {
            "delta_net_score": delta_net,
            "delta_tpr": delta_tpr,
            "delta_fpr": delta_fpr,
            "improvement": delta_net > 0,
        }

    config.output.parent.mkdir(parents=True, exist_ok=True)
    with config.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    LOGGER.info("Results saved to %s", config.output)


def main(argv: Sequence[str] | None = None) -> None:
    config = _parse_config(argv)
    run(config)


if __name__ == "__main__":
    main()
