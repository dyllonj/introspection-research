"""Tests covering helper logic inside evaluation scripts."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from introspect.src.eval_A_injected_report import _vector_variants
from introspect.src.eval_D_intentional_control import _area_under_curve, _cosine_similarity
from introspect.src.inject import InjectionSpec


@pytest.mark.parametrize("alpha", [0.5, 1.0])
def test_task_a_vector_variants_include_control_and_ablation(alpha: float) -> None:
    vector = torch.tensor([0.25, -0.5, 0.75, 1.0], dtype=torch.float32)
    spec = InjectionSpec(layer_idx=3, alpha=alpha, vector=vector, token_positions=[1, 2])
    rng = random.Random(123)

    variants = _vector_variants(
        base_spec=spec,
        alpha=alpha,
        concept_vector=vector,
        rng=rng,
    )

    kinds = [kind for kind, _spec, _injected in variants]
    assert kinds == ["target", "negative", "random", "control"]

    for kind, variant_spec, injected in variants:
        assert variant_spec.token_positions == spec.token_positions
        if kind == "target":
            assert injected is True
            assert torch.allclose(variant_spec.vector, vector)
        elif kind == "negative":
            assert injected is True
            assert torch.allclose(variant_spec.vector, -vector)
        elif kind == "random":
            assert injected is True
            # All random vectors are unit norm by construction.
            assert torch.allclose(
                torch.linalg.vector_norm(variant_spec.vector),
                torch.tensor(1.0),
                atol=1e-6,
            )
        else:  # control
            assert kind == "control"
            assert injected is False
            # Control path reuses the base specification without modification.
            assert variant_spec is spec


def test_task_d_cosine_and_area_metrics() -> None:
    vec_a = torch.tensor([1.0, 0.0, 0.0])
    vec_b = torch.tensor([1.0, 1.0, 0.0])
    assert _cosine_similarity(vec_a, vec_a) == pytest.approx(1.0)
    assert _cosine_similarity(vec_a, vec_b) == pytest.approx(1 / np.sqrt(2))

    layers = [0, 2, 4, 6]
    values = [0.0, 0.5, 1.0, 0.0]
    expected = float(np.trapezoid(values, x=layers))
    assert _area_under_curve(layers, values) == pytest.approx(expected)

