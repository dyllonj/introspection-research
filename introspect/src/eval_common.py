"""Shared helpers for evaluation entry points.

This module keeps the evaluation scripts lightweight by centralising common
operations such as registry lookups, adapter loading, concept vector caching
and configuration parsing.  The functions are intentionally dependency-free so
they can be used both from production code and within the unit tests.
"""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from .adapters import BaseModelAdapter, select_device_map, select_dtype
from .vectors import (
    ConceptWordSet,
    build_concept_vector,
    cache_path,
    load_vector,
    load_words,
    save_vector,
)

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "registry" / "models.yaml"


try:  # pragma: no cover - optional dependency
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover - broad to avoid hard dependency
    OmegaConf = None  # type: ignore[assignment]


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``--config`` style arguments shared across CLIs."""

    parser.add_argument("--config", type=Path, help="Path to a YAML/Hydra config")
    parser.add_argument(
        "--config-dir",
        type=Path,
        help="Directory containing configuration files (used with --config-name)",
    )
    parser.add_argument(
        "--config-name",
        help="Configuration file name (without extension) resolved within --config-dir",
    )


def _load_config_mapping(path: Path) -> Mapping[str, Any]:
    if OmegaConf is not None:  # pragma: no branch - simple gate
        cfg = OmegaConf.load(path)  # type: ignore[call-arg]
        data = OmegaConf.to_container(cfg, resolve=True)
    else:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

    if not isinstance(data, Mapping):
        raise ValueError(f"Configuration at {path} must be a mapping")
    return data


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse CLI arguments optionally backed by a Hydra/YAML config file."""

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    pre_parser.add_argument("--config-dir", type=Path)
    pre_parser.add_argument("--config-name")
    known, remaining = pre_parser.parse_known_args(argv)

    config_path: Path | None = None
    if known.config is not None:
        config_path = known.config
    elif known.config_name is not None:
        config_dir = known.config_dir or Path.cwd()
        config_path = config_dir / f"{known.config_name}.yaml"

    if config_path is not None:
        config_data = _load_config_mapping(config_path)
        parser.set_defaults(**config_data)

    args = parser.parse_args(argv)
    return args


def _load_registry_entry(model_id: str) -> Mapping[str, Any]:
    with REGISTRY_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    for entry in data.get("models", []):
        if entry.get("id") == model_id:
            return entry

    msg = f"Model {model_id!r} not present in registry {REGISTRY_PATH}"
    raise KeyError(msg)


def _resolve_adapter_class(name: str) -> type[BaseModelAdapter]:
    # Import lazily to avoid circular imports.
    from . import vectors as vector_module

    try:
        return vector_module.ADAPTER_REGISTRY[name]
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise KeyError(f"Unknown adapter class {name!r}") from exc


def _resolve_dtype(name: str | None) -> torch.dtype | None:
    if name is None:
        return None

    from . import vectors as vector_module

    key = name.lower()
    try:
        return vector_module.DTYPE_ALIASES[key]
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported dtype alias {name!r}") from exc


@dataclass(slots=True)
class LoadedAdapter:
    adapter: BaseModelAdapter
    adapter_name: str
    dtype: torch.dtype
    device_map: str | Mapping[str, Any]


def load_adapter_from_registry(
    model_id: str,
    *,
    adapter_name: str | None = None,
    dtype: str | None = None,
    device_map: str | Mapping[str, Any] | None = None,
    seed: int | None = None,
) -> LoadedAdapter:
    """Instantiate an adapter using information from the registry."""

    registry_entry: Mapping[str, Any] | None = None
    if adapter_name is None or dtype is None:
        registry_entry = _load_registry_entry(model_id)

    if adapter_name is None and registry_entry is not None:
        adapter_name = registry_entry.get("adapter")
    if adapter_name is None:
        raise ValueError("Adapter class must be specified either via CLI or registry")

    adapter_cls = _resolve_adapter_class(adapter_name)

    requested_dtype = dtype
    if requested_dtype is None and registry_entry is not None:
        requested_dtype = registry_entry.get("dtype")

    torch_dtype = select_dtype(_resolve_dtype(requested_dtype))
    if not torch.cuda.is_available() and torch_dtype in {torch.bfloat16, torch.float16}:
        LOGGER.info("Falling back to float32 on CPU for model %s", model_id)
        torch_dtype = torch.float32

    resolved_device_map = select_device_map(device_map)

    adapter = adapter_cls.load(
        model_id,
        dtype=torch_dtype,
        device_map=resolved_device_map,
        seed=seed,
    )

    return LoadedAdapter(
        adapter=adapter,
        adapter_name=adapter_name,
        dtype=torch_dtype,
        device_map=resolved_device_map,
    )


def ensure_vector(
    *,
    adapter: BaseModelAdapter,
    model_id: str,
    layer_idx: int,
    word: str,
    cache_dir: str | Path,
    baseline_words: Sequence[str],
    prompt_template: str,
    baseline_sample_size: int | None,
    rng: random.Random | None,
) -> torch.Tensor:
    """Load a cached concept vector or build and cache it."""

    path = cache_path(model_id, layer_idx, word, cache_dir=cache_dir)
    if path.exists():
        vector, _ = load_vector(path)
        LOGGER.debug("Loaded cached vector %s", path)
        return vector

    vector, sampled_baselines = build_concept_vector(
        adapter,
        layer_idx,
        target_word=word,
        baseline_words=baseline_words,
        prompt_template=prompt_template,
        baseline_sample_size=baseline_sample_size,
        rng=rng,
        return_sampled_baselines=True,
    )

    metadata = {
        "model_id": model_id,
        "layer": layer_idx,
        "word": word,
        "baseline_count": len(baseline_words),
        "baseline_sample_count": len(sampled_baselines),
        "prompt_template": prompt_template,
    }
    save_vector(vector, path, metadata=metadata)
    return vector


def select_target_words(
    words: ConceptWordSet,
    *,
    limit: int | None,
    seed: int | None,
) -> list[str]:
    """Return the list of target words respecting ``limit`` and ``seed``."""

    targets = list(words.iter_targets())
    if limit is None or limit >= len(targets):
        return targets

    if limit <= 0:
        raise ValueError("limit must be positive when provided")

    rng = random.Random(seed)
    return rng.sample(targets, limit)


def random_unit_vector(hidden_size: int, *, rng: random.Random | None = None) -> torch.Tensor:
    """Return a unit-norm random vector with deterministic sampling."""

    generator = random.Random() if rng is None else rng
    values = [generator.gauss(0.0, 1.0) for _ in range(hidden_size)]
    tensor = torch.tensor(values, dtype=torch.float32)
    norm = torch.linalg.vector_norm(tensor)
    if torch.isclose(norm, torch.tensor(0.0)):
        raise ValueError("Cannot normalise a zero vector")
    return tensor / norm


__all__ = [
    "LoadedAdapter",
    "add_config_arguments",
    "ensure_vector",
    "load_adapter_from_registry",
    "load_words",
    "parse_args_with_config",
    "random_unit_vector",
    "select_target_words",
]

