"""Tests for sweep utilities and layer handling."""

from __future__ import annotations

import pytest

from introspect.src import sweep
from introspect.src.eval_A_injected_report import _normalise_layer_indices


def test_evenly_spaced_layers_include_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sweep, "_resolve_num_layers", lambda _model_id: 8)

    assert sweep._evenly_spaced_layers("dummy", 1) == [7]

    indices = sweep._evenly_spaced_layers("dummy", 3)
    assert indices[0] == 0
    assert indices[-1] == 7
    assert len(indices) == 3

    assert sweep._evenly_spaced_layers("dummy", 10) == list(range(8))


def test_normalise_layer_indices_supports_one_based() -> None:
    assert _normalise_layer_indices([1, 4, 8], num_layers=8) == [0, 3, 7]
    assert _normalise_layer_indices([0, 2, 5], num_layers=8) == [0, 2, 5]

    with pytest.raises(ValueError):
        _normalise_layer_indices([9], num_layers=8)
