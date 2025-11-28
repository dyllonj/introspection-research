"""Quick probe analysis to validate introspection training feasibility.

This is the FIRST thing to run. If the probe can't detect injection,
training won't help.

Usage:
    python -m introspect.src.training.probe_feasibility \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --n-concepts 30 \
        --output results/probe_feasibility.json

Expected output:
    - Probe accuracy > 80%: Strong signal, proceed with DPO
    - Probe accuracy 65-80%: Weak signal, might work with careful training
    - Probe accuracy < 65%: No signal, training unlikely to help
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch
from sklearn.model_selection import train_test_split

from ..adapters.base import BaseModelAdapter
from ..eval_common import load_adapter_from_registry, load_words, select_target_words
from ..generation import build_chat_prompt
from ..inject import InjectionSpec, token_positions_for_substring, attach_injection
from ..io_utils import seed_everything, setup_logging
from ..prompts import task_a_paper_messages
from ..vectors import build_concept_vector, DEFAULT_WORDS_PATH
from .probes import InjectionProbe, train_injection_probe, evaluate_probe, ProbeMetrics

LOGGER = logging.getLogger(__name__)


def _capture_activations(
    adapter: BaseModelAdapter,
    prompt: str,
    layer_idx: int,
    spec: InjectionSpec | None,
    enable_injection: bool,
) -> torch.Tensor:
    """Capture mean-pooled activations at a layer during forward pass."""
    
    captured: list[torch.Tensor] = []
    
    def hook_fn(module, inputs, output):
        residual = output[0] if isinstance(output, tuple) else output
        # Mean pool over sequence, take first batch element
        mean_act = residual.mean(dim=1).squeeze(0).detach().cpu()
        captured.append(mean_act)
        return output
    
    # Register capture hook
    capture_handle = adapter.register_residual_hook(layer_idx, hook_fn)
    
    # Optionally register injection
    inject_handle = None
    if enable_injection and spec is not None:
        inject_handle = attach_injection(adapter, spec)
    
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
    
    return captured[-1].to(torch.float32)


def run_probe_analysis(
    model: str,
    n_concepts: int = 30,
    layers: list[int] | None = None,
    alpha: float = 2.0,
    seed: int = 42,
    output: Path | None = None,
) -> dict:
    """Run probe feasibility analysis."""
    
    setup_logging()
    seed_everything(seed)
    rng = random.Random(seed)
    
    # Load model
    LOGGER.info("Loading model: %s", model)
    loaded = load_adapter_from_registry(model, seed=seed)
    adapter = loaded.adapter
    
    # Infer layers if not specified
    n_layers = adapter.num_layers
    if layers is None:
        # Focus on the 2/3 point where introspection typically peaks
        center = (2 * n_layers) // 3
        layers = [center - 2, center, center + 2]
    
    LOGGER.info("Testing layers: %s (model has %d layers)", layers, n_layers)
    
    # Load concepts
    word_set = load_words(DEFAULT_WORDS_PATH)
    concepts = select_target_words(word_set, limit=n_concepts, seed=seed)
    baseline_words = list(word_set.iter_baselines())
    
    # Build prompt
    messages = task_a_paper_messages()
    prompt, _ = build_chat_prompt(adapter.tokenizer, messages)
    prefix_prompt, _ = build_chat_prompt(adapter.tokenizer, messages[:-1])
    assistant_prefix = prompt[len(prefix_prompt):]
    occurrence = prefix_prompt.count(assistant_prefix) if assistant_prefix else 0
    
    results = {"model": model, "layers": {}, "recommendation": ""}
    best_accuracy = 0.0
    best_layer = layers[0]
    
    for layer_idx in layers:
        LOGGER.info("Collecting activations at layer %d...", layer_idx)
        
        activations = []
        labels = []
        
        for concept in concepts:
            # Build concept vector
            vector = build_concept_vector(
                adapter,
                layer_idx,
                target_word=concept,
                baseline_words=baseline_words,
                prompt_template="Tell me about {word}.",
                baseline_sample_size=50,
                rng=rng,
            )
            
            # Get token positions
            token_positions = token_positions_for_substring(
                adapter, prompt, assistant_prefix, occurrence=occurrence
            )
            
            spec = InjectionSpec(
                layer_idx=layer_idx,
                alpha=alpha,
                vector=vector,
                token_positions=token_positions,
                apply_to_generated=False,
            )
            
            # Capture with injection (label=1)
            act_inject = _capture_activations(
                adapter, prompt, layer_idx, spec, enable_injection=True
            )
            activations.append(act_inject)
            labels.append(1)
            
            # Capture without injection (label=0)
            act_control = _capture_activations(
                adapter, prompt, layer_idx, spec, enable_injection=False
            )
            activations.append(act_control)
            labels.append(0)
        
        # Stack into tensors
        X = torch.stack(activations)
        y = torch.tensor(labels)
        
        LOGGER.info("Collected %d samples at layer %d", len(X), layer_idx)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X.numpy(), y.numpy(), test_size=0.3, random_state=seed, stratify=y.numpy()
        )
        X_train = torch.from_numpy(X_train)
        X_test = torch.from_numpy(X_test)
        y_train = torch.from_numpy(y_train)
        y_test = torch.from_numpy(y_test)
        
        # Train probe
        hidden_size = X.shape[1]
        probe = InjectionProbe(hidden_size)
        
        LOGGER.info("Training probe...")
        history = train_injection_probe(
            probe,
            X_train, y_train,
            val_activations=X_test,
            val_labels=y_test,
            epochs=100,
            lr=1e-3,
            early_stopping_patience=15,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        # Evaluate
        metrics = evaluate_probe(
            probe, X_test, y_test,
            device="cuda" if torch.cuda.is_available() else "cpu"
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
    
    # Generate recommendation
    results["best_layer"] = best_layer
    results["best_accuracy"] = best_accuracy
    
    if best_accuracy >= 0.85:
        results["recommendation"] = "STRONG_SIGNAL"
        results["message"] = (
            f"Probe achieves {best_accuracy:.1%} accuracy at layer {best_layer}. "
            "Strong signal - proceed with DPO training. The model's activations "
            "clearly distinguish injection from control trials."
        )
    elif best_accuracy >= 0.70:
        results["recommendation"] = "MODERATE_SIGNAL"
        results["message"] = (
            f"Probe achieves {best_accuracy:.1%} accuracy at layer {best_layer}. "
            "Moderate signal - DPO training may help but results uncertain. "
            "Consider using higher alpha values or different layers."
        )
    elif best_accuracy >= 0.60:
        results["recommendation"] = "WEAK_SIGNAL"
        results["message"] = (
            f"Probe achieves {best_accuracy:.1%} accuracy at layer {best_layer}. "
            "Weak signal - training unlikely to produce strong introspection. "
            "The injection may not be sufficiently distinguishable in this model."
        )
    else:
        results["recommendation"] = "NO_SIGNAL"
        results["message"] = (
            f"Probe achieves only {best_accuracy:.1%} accuracy (near chance). "
            "No detectable signal - DPO training will not help. "
            "The model cannot distinguish injection from baseline activations."
        )
    
    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("RECOMMENDATION: %s", results["recommendation"])
    LOGGER.info(results["message"])
    LOGGER.info("=" * 60)
    
    # Save results
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as f:
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
    parser.add_argument("--output", type=Path, default=Path("results/probe_feasibility.json"))
    
    args = parser.parse_args(argv)
    
    run_probe_analysis(
        model=args.model,
        n_concepts=args.n_concepts,
        layers=args.layers,
        alpha=args.alpha,
        seed=args.seed,
        output=args.output,
    )


if __name__ == "__main__":
    main()
