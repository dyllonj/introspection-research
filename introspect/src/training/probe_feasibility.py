"""Feasibility check: can a linear probe detect injections?"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch

from ..eval_common import load_adapter_from_registry, load_words, select_target_words, ensure_vector
from ..generation import build_chat_prompt
from ..inject import InjectionSpec, attach_injection, resolve_injection_positions
from ..io_utils import seed_everything, setup_logging
from ..prompts import task_a_paper_messages
from ..vectors import DEFAULT_WORDS_PATH
from .probes import InjectionProbe, ProbeMetrics, evaluate_probe, train_injection_probe

LOGGER = logging.getLogger(__name__)


def _capture_activations(
    adapter,
    prompt: str,
    inject_layer: int,
    capture_layer: int,
    spec: InjectionSpec | None,
    enable_injection: bool,
) -> torch.Tensor:
    """Capture activations downstream of an injection.

    Hooks are ordered so the injection fires before the capture hook. Captures
    the last-token residual to emphasise accumulated signal rather than
    mean-pooled values that can wash out the effect.
    """

    captured: list[torch.Tensor] = []

    def hook_fn(_module, _inputs, output):
        residual = output[0] if isinstance(output, tuple) else output
        last_token = residual[0, -1, :].detach().cpu()
        captured.append(last_token)
        return output

    inject_handle = None
    if enable_injection and spec is not None:
        inject_handle = attach_injection(adapter, spec)

    capture_handle = adapter.register_residual_hook(capture_layer, hook_fn)

    try:
        tokenized = adapter.tokenizer(prompt, return_tensors="pt")
        device = next(adapter.model.parameters()).device
        inputs = {k: v.to(device) for k, v in tokenized.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            adapter.model(**inputs)
    finally:
        capture_handle.remove()
        if inject_handle:
            inject_handle.remove()

    if not captured:
        raise RuntimeError("Failed to capture activations")
    return captured[-1].to(torch.float32)


def run_probe_analysis(
    model: str,
    *,
    n_concepts: int = 30,
    layers: list[int] | None = None,
    alpha: float = 2.0,
    seed: int = 42,
    injection_mode: str = "prefix",
    output: Path | None = None,
) -> dict:
    """Run probe feasibility analysis and return metrics."""

    try:  # pragma: no cover - optional dependency
        from sklearn.model_selection import train_test_split
    except Exception as exc:  # pragma: no cover - defensive
        raise ImportError(
            "probe_feasibility requires scikit-learn. Install with `pip install scikit-learn` "
            "or use the `[train]` extra."
        ) from exc

    setup_logging()
    seed_everything(seed)
    rng = random.Random(seed)

    LOGGER.info("Loading model: %s", model)
    loaded = load_adapter_from_registry(model, seed=seed)
    adapter = loaded.adapter

    n_layers = adapter.num_layers
    if layers is None:
        center = (2 * n_layers) // 3
        layers = [max(0, center - 2), center, min(n_layers - 1, center + 2)]

    LOGGER.info("Testing layers: %s (model has %d layers)", layers, n_layers)

    word_set = load_words(DEFAULT_WORDS_PATH)
    concepts = select_target_words(word_set, limit=n_concepts, seed=seed)
    baseline_words = list(word_set.iter_baselines())

    messages = task_a_paper_messages()
    prompt, _ = build_chat_prompt(adapter.tokenizer, messages)
    token_positions, _suffix_start = resolve_injection_positions(
        adapter,
        prompt,
        mode="prefix" if injection_mode not in {"prefix", "suffix"} else injection_mode,
    )
    apply_to_generated = injection_mode == "suffix"
    if injection_mode == "suffix":
        token_positions = ["suffix"]

    results = {"model": model, "layers": {}, "recommendation": ""}
    best_accuracy = 0.0
    best_layer = layers[0]

    for layer_idx in layers:
        capture_layer = min(layer_idx + 4, n_layers - 1)
        LOGGER.info(
            "Collecting activations at inject layer %d, capture layer %d...",
            layer_idx,
            capture_layer,
        )

        activations = []
        labels = []

        for concept in concepts:
            vector = ensure_vector(
                adapter=adapter,
                model_id=model,
                layer_idx=layer_idx,
                word=concept,
                cache_dir="results/vectors",
                baseline_words=baseline_words,
                prompt_template="Tell me about {word}.",
                baseline_sample_size=50,
                rng=rng,
            )

            spec = InjectionSpec(
                layer_idx=layer_idx,
                alpha=alpha,
                vector=vector,
                token_positions=token_positions,
                apply_to_generated=apply_to_generated,
            )

            act_inject = _capture_activations(
                adapter,
                prompt,
                inject_layer=layer_idx,
                capture_layer=capture_layer,
                spec=spec,
                enable_injection=True,
            )
            activations.append(act_inject)
            labels.append(1)

            act_control = _capture_activations(
                adapter,
                prompt,
                inject_layer=layer_idx,
                capture_layer=capture_layer,
                spec=spec,
                enable_injection=False,
            )
            activations.append(act_control)
            labels.append(0)

        X = torch.stack(activations)
        y = torch.tensor(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            X.numpy(),
            y.numpy(),
            test_size=0.3,
            random_state=seed,
            stratify=y.numpy(),
        )
        X_train = torch.from_numpy(X_train)
        X_test = torch.from_numpy(X_test)
        y_train = torch.from_numpy(y_train)
        y_test = torch.from_numpy(y_test)

        hidden_size = X.shape[1]
        probe = InjectionProbe(hidden_size)

        LOGGER.info("Training probe...")
        _history = train_injection_probe(
            probe,
            X_train,
            y_train,
            val_activations=X_test,
            val_labels=y_test,
            epochs=100,
            lr=1e-3,
            early_stopping_patience=15,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        metrics: ProbeMetrics = evaluate_probe(
            probe,
            X_test,
            y_test,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        LOGGER.info(
            "Layer %d: Accuracy=%.1f%%, AUROC=%.3f, TPR=%.1f%%, FPR=%.1f%%",
            layer_idx,
            metrics.accuracy * 100,
            metrics.auroc,
            metrics.tpr * 100,
            metrics.fpr * 100,
        )

        results["layers"][layer_idx] = {
            "accuracy": metrics.accuracy,
            "auroc": metrics.auroc,
            "tpr": metrics.tpr,
            "fpr": metrics.fpr,
            "f1": metrics.f1,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

        if metrics.accuracy > best_accuracy:
            best_accuracy = metrics.accuracy
            best_layer = layer_idx

    results["best_layer"] = best_layer
    results["best_accuracy"] = best_accuracy

    if best_accuracy >= 0.85:
        results["recommendation"] = "STRONG_SIGNAL"
        results["message"] = (
            f"Probe achieves {best_accuracy:.1%} accuracy at layer {best_layer}. "
            "Strong signal - proceed with DPO."
        )
    elif best_accuracy >= 0.70:
        results["recommendation"] = "MODERATE_SIGNAL"
        results["message"] = (
            f"Probe achieves {best_accuracy:.1%} accuracy at layer {best_layer}. "
            "Moderate signal - proceed with caution."
        )
    elif best_accuracy >= 0.60:
        results["recommendation"] = "WEAK_SIGNAL"
        results["message"] = (
            f"Probe achieves {best_accuracy:.1%} accuracy at layer {best_layer}. "
            "Weak signal - training unlikely to help."
        )
    else:
        results["recommendation"] = "NO_SIGNAL"
        results["message"] = (
            f"Probe achieves only {best_accuracy:.1%} accuracy (near chance). "
            "No detectable signal."
        )

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        LOGGER.info("Results saved to %s", output)

    return results


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model to analyze")
    parser.add_argument("--n-concepts", type=int, default=30)
    parser.add_argument("--layers", type=int, nargs="+", help="Layers to test")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--injection-mode", choices=["prefix", "suffix"], default="prefix")
    parser.add_argument("--output", type=Path, default=Path("results/probe_feasibility.json"))

    args = parser.parse_args(argv)

    run_probe_analysis(
        model=args.model,
        n_concepts=args.n_concepts,
        layers=args.layers,
        alpha=args.alpha,
        seed=args.seed,
        injection_mode=args.injection_mode,
        output=args.output,
    )


if __name__ == "__main__":
    main()
