"""Concept vector construction, caching, and command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping, Sequence

import numpy as np
import torch
import yaml

from .adapters import (
    BaseModelAdapter,
    FalconAdapter,
    LlamaAdapter,
    MistralAdapter,
    NeoXAdapter,
    QwenAdapter,
    seed_everything,
    select_device_map,
    select_dtype,
)

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORDS_PATH = PROJECT_ROOT / "concepts" / "words.yaml"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "results" / "vectors"
DEFAULT_PROMPT_TEMPLATE = "Think about the concept \"{word}\" in a single sentence."
METADATA_VERSION = 1

AdapterRegistry = Mapping[str, type[BaseModelAdapter]]

ADAPTER_REGISTRY: AdapterRegistry = {
    "LlamaAdapter": LlamaAdapter,
    "MistralAdapter": MistralAdapter,
    "FalconAdapter": FalconAdapter,
    "NeoXAdapter": NeoXAdapter,
    "QwenAdapter": QwenAdapter,
}

DTYPE_ALIASES: Mapping[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "full": torch.float32,
}

__all__ = [
    "ConceptWordSet",
    "build_concept_vector",
    "cache_path",
    "METADATA_VERSION",
    "load_cached_words",
    "load_vector",
    "load_words",
    "save_vector",
]


@dataclass(slots=True)
class ConceptWordSet:
    """Container for target and baseline words loaded from YAML."""

    targets: list[str]
    baselines: list[str]

    def iter_targets(self) -> Iterator[str]:
        """Yield target words in their configured order."""

        yield from self.targets

    def iter_baselines(self) -> Iterator[str]:
        """Yield baseline words in their configured order."""

        yield from self.baselines


def load_words(
    yaml_path: str | Path = DEFAULT_WORDS_PATH,
    *,
    limit_targets: int | None = None,
    limit_baselines: int | None = None,
) -> ConceptWordSet:
    """Load concept word lists from ``yaml_path``.

    Parameters
    ----------
    yaml_path:
        Path to a YAML file containing ``targets`` and ``baselines`` lists.
    limit_targets:
        Optional limit on the number of target words to return.
    limit_baselines:
        Optional limit on the number of baseline words to return.
    """

    path = Path(yaml_path)
    if not path.exists():  # pragma: no cover - guard
        msg = f"Word list YAML does not exist: {path}" 
        raise FileNotFoundError(msg)

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    targets = list(map(str, data.get("targets", [])))
    baselines = list(map(str, data.get("baselines", [])))

    if not targets:
        raise ValueError("YAML file must define at least one target word")
    if not baselines:
        raise ValueError("YAML file must define at least one baseline word")

    if limit_targets is not None:
        targets = targets[:limit_targets]
    if limit_baselines is not None:
        baselines = baselines[:limit_baselines]

    LOGGER.debug(
        "Loaded %d target and %d baseline words from %s",
        len(targets),
        len(baselines),
        path,
    )

    return ConceptWordSet(targets=targets, baselines=baselines)


def load_cached_words(
    yaml_path: str | Path = DEFAULT_WORDS_PATH,
    *,
    limit_targets: int | None = None,
    limit_baselines: int | None = None,
) -> ConceptWordSet:
    """Alias maintained for backwards compatibility with earlier naming."""

    return load_words(
        yaml_path=yaml_path,
        limit_targets=limit_targets,
        limit_baselines=limit_baselines,
    )


def _select_baselines(
    baselines: Sequence[str],
    sample_size: int | None,
    *,
    rng: random.Random | None = None,
) -> list[str]:
    """Return either the full ``baselines`` list or a deterministic sample."""

    words = list(baselines)
    if sample_size is None or sample_size >= len(words):
        return words
    if sample_size <= 0:
        raise ValueError("Baseline sample size must be positive when provided")

    sampler: random.Random
    if rng is None:
        sampler = random
    else:
        sampler = rng

    # random.sample keeps order within the sampled subset deterministic for a
    # fixed RNG state while avoiding replacement.
    return sampler.sample(words, sample_size)


def _prompt_from_template(template: str, word: str) -> str:
    try:
        return template.format(word=word)
    except KeyError as exc:  # pragma: no cover - format guard
        raise ValueError("Prompt template must expose '{word}' placeholder") from exc


def _capture_residual(
    adapter: BaseModelAdapter,
    layer_idx: int,
    inputs: MutableMapping[str, torch.Tensor],
) -> torch.Tensor:
    """Run a forward pass capturing the residual stream at ``layer_idx``."""

    captured: list[torch.Tensor] = []

    def hook_fn(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        residual = output[0] if isinstance(output, tuple) else output
        if not isinstance(residual, torch.Tensor):  # pragma: no cover - guard
            raise TypeError("Residual hook expected tensor output")
        captured.append(residual.detach())
        return output

    handle = adapter.register_residual_hook(layer_idx, hook_fn)
    was_training = adapter.model.training
    adapter.model.eval()
    try:
        with torch.no_grad():
            adapter.model(**inputs)
    finally:
        handle.remove()
        if was_training:
            adapter.model.train()

    if not captured:
        raise RuntimeError(
            "Residual hook did not capture any activations; verify layer index"
        )

    return captured[-1]


def _activation_for_prompt(
    adapter: BaseModelAdapter,
    layer_idx: int,
    prompt: str,
) -> torch.Tensor:
    tokenized = adapter.tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
    )
    device = next(adapter.model.parameters()).device
    inputs = {key: tensor.to(device) for key, tensor in tokenized.items()}

    residual = _capture_residual(adapter, layer_idx, inputs)

    if "attention_mask" in inputs:
        lengths = inputs["attention_mask"].sum(dim=1) - 1
    else:
        lengths = torch.full((residual.size(0),), residual.size(1) - 1, device=residual.device)

    index = lengths.long().clamp(min=0)
    batch_indices = torch.arange(residual.size(0), device=residual.device)
    final_states = residual[batch_indices, index, :]

    return final_states.detach().to(torch.float32).cpu()


def build_concept_vector(
    adapter: BaseModelAdapter,
    layer_idx: int,
    *,
    target_word: str,
    baseline_words: Sequence[str],
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    baseline_sample_size: int | None = None,
    rng: random.Random | None = None,
    return_sampled_baselines: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[str]]:
    """Construct a centered, unit-normalized concept vector.

    The vector is computed as the difference between the mean activation of
    ``target_word`` prompts and the mean activation of prompts built from
    ``baseline_words`` at the specified ``layer_idx``. The final vector is
    centered by subtracting its mean value and normalised to unit length. When
    ``return_sampled_baselines`` is ``True`` the function returns a tuple of the
    vector and the list of sampled baseline words.
    """

    if not baseline_words:
        raise ValueError("At least one baseline word is required")

    prompts = [_prompt_from_template(prompt_template, target_word)]
    target_acts = [
        _activation_for_prompt(adapter, layer_idx, prompt)
        for prompt in prompts
    ]

    sampled_baselines = _select_baselines(
        baseline_words, baseline_sample_size, rng=rng
    )
    baseline_acts = [
        _activation_for_prompt(
            adapter,
            layer_idx,
            _prompt_from_template(prompt_template, baseline),
        )
        for baseline in sampled_baselines
    ]

    target_stack = torch.stack(target_acts)
    baseline_stack = torch.stack(baseline_acts)

    target_mean = target_stack.mean(dim=0)
    baseline_mean = baseline_stack.mean(dim=0)

    vector = target_mean - baseline_mean
    vector -= vector.mean()

    norm = torch.linalg.vector_norm(vector)
    if torch.isclose(norm, torch.tensor(0.0, dtype=norm.dtype)):
        raise ValueError("Cannot normalize zero vector; adjust baseline set")

    normalized = (vector / norm).to(torch.float32).squeeze()
    if return_sampled_baselines:
        return normalized, sampled_baselines
    return normalized


def cache_path(
    model_id: str,
    layer_idx: int,
    word: str,
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> Path:
    """Return the canonical cache path for ``(model_id, layer_idx, word)``."""

    safe_model = model_id.replace("/", "--")
    safe_word = word.replace(" ", "_")
    directory = Path(cache_dir) / safe_model
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"layer{layer_idx:04d}_{safe_word}.npy"


def save_vector(
    vector: torch.Tensor | np.ndarray,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Persist ``vector`` to ``path`` alongside optional metadata."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    array = (
        vector.detach().cpu().numpy()
        if isinstance(vector, torch.Tensor)
        else np.asarray(vector)
    )

    np.save(file_path, array)
    LOGGER.info("Saved concept vector to %s", file_path)

    meta_dict: dict[str, Any] | None
    if metadata is None:
        meta_dict = {}
    else:
        meta_dict = dict(metadata)

    if meta_dict is not None:
        meta_dict.setdefault("dtype", str(array.dtype))
        meta_dict.setdefault("shape", list(array.shape))
        meta_dict.setdefault("version", METADATA_VERSION)

        def _json_safe(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, torch.dtype):
                return str(value)
            if isinstance(value, (np.generic,)):
                return value.item()
            return value

        serializable = {key: _json_safe(val) for key, val in meta_dict.items()}
        meta_path = file_path.with_suffix(".json")
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(serializable, fh, indent=2, sort_keys=True)
        LOGGER.debug("Wrote metadata to %s", meta_path)


def load_vector(
    path: str | Path,
    *,
    map_location: torch.device | str | None = None,
) -> tuple[torch.Tensor, dict[str, Any] | None]:
    """Load a cached vector and its metadata, if available."""

    file_path = Path(path)
    array = np.load(file_path)
    tensor = torch.from_numpy(array)
    if map_location is not None:
        tensor = tensor.to(map_location)

    meta_path = file_path.with_suffix(".json")
    metadata = None
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as fh:
            metadata = json.load(fh)
        if not isinstance(metadata, dict):
            raise ValueError(f"Metadata at {meta_path} must be a JSON object")

        recorded_shape = metadata.get("shape")
        if recorded_shape is not None and tuple(recorded_shape) != tuple(array.shape):
            raise ValueError(
                "Metadata shape does not match stored vector array"
            )

        recorded_dtype = metadata.get("dtype")
        if recorded_dtype is not None and recorded_dtype != str(array.dtype):
            raise ValueError(
                "Metadata dtype does not match stored vector array"
            )

    return tensor, metadata


def _load_registry_entry(model_id: str) -> dict[str, Any]:
    registry_path = PROJECT_ROOT / "registry" / "models.yaml"
    with registry_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    for entry in data.get("models", []):
        if entry.get("id") == model_id:
            return entry

    msg = f"Model {model_id!r} not found in registry {registry_path}"
    raise KeyError(msg)


def _resolve_adapter_class(name: str) -> type[BaseModelAdapter]:
    try:
        return ADAPTER_REGISTRY[name]
    except KeyError as exc:  # pragma: no cover - guard
        raise KeyError(f"Unknown adapter {name!r}") from exc


def _resolve_dtype(value: str | None) -> torch.dtype | None:
    if value is None:
        return None
    key = value.lower()
    if key not in DTYPE_ALIASES:
        raise ValueError(f"Unrecognized dtype alias: {value}")
    return DTYPE_ALIASES[key]


def _run_cli(args: argparse.Namespace) -> None:
    if args.seed is not None:
        seed_everything(args.seed)

    if args.adapter is None:
        entry = _load_registry_entry(args.model)
        adapter_name = entry.get("adapter")
        dtype_name = entry.get("dtype")
    else:
        adapter_name = args.adapter
        dtype_name = args.dtype

    if adapter_name is None:
        raise ValueError(
            "Adapter class must be provided via registry or --adapter option"
        )

    adapter_cls = _resolve_adapter_class(adapter_name)
    requested_dtype = _resolve_dtype(dtype_name) if dtype_name else None
    dtype = select_dtype(requested_dtype)
    if not torch.cuda.is_available() and dtype in {torch.bfloat16, torch.float16}:
        LOGGER.warning(
            "Falling back to float32 for CPU execution (requested %s)",
            dtype,
        )
        dtype = torch.float32
    device_map = select_device_map(None if args.device_map == "auto" else args.device_map)

    adapter = adapter_cls.load(
        args.model,
        dtype=dtype,
        device_map=device_map,
        seed=args.seed,
    )

    words = load_words(
        args.words_file,
        limit_targets=args.limit_targets,
        limit_baselines=args.limit_baselines,
    )

    baseline_words = list(words.iter_baselines())
    baseline_rng = random.Random(args.seed) if args.seed is not None else None

    for layer in args.layers:
        for word in words.iter_targets():
            path = cache_path(args.model, layer, word, cache_dir=args.cache_dir)
            if path.exists() and not args.force:
                LOGGER.info("Skipping existing vector %s", path)
                continue

            vector, sampled_baselines = build_concept_vector(
                adapter,
                layer,
                target_word=word,
                baseline_words=baseline_words,
                prompt_template=args.prompt_template,
                baseline_sample_size=args.baseline_sample,
                rng=baseline_rng,
                return_sampled_baselines=True,
            )

            metadata = {
                "model_id": args.model,
                "adapter": adapter_name,
                "layer": layer,
                "word": word,
                "prompt_template": args.prompt_template,
                "baseline_count": len(baseline_words),
                "baseline_sample_requested": args.baseline_sample,
                "baseline_sampled": sampled_baselines,
                "baseline_sample_count": len(sampled_baselines),
                "target_count": 1,
                "model_dtype": str(dtype),
                "seed": args.seed,
            }
            save_vector(vector, path, metadata=metadata)


def _parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model identifier to load")
    parser.add_argument(
        "--adapter",
        help="Override adapter class name (defaults to registry lookup)",
    )
    parser.add_argument(
        "--dtype",
        help="Preferred torch dtype (bf16, fp16, fp32). Used when --adapter is set.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map passed to transformers (auto or cpu)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        required=True,
        help="Layer indices to process",
    )
    parser.add_argument(
        "--words-file",
        type=Path,
        default=DEFAULT_WORDS_PATH,
        help="Path to YAML file containing target/baseline words",
    )
    parser.add_argument(
        "--limit-targets",
        type=int,
        help="Optional limit on the number of target words",
    )
    parser.add_argument(
        "--limit-baselines",
        type=int,
        help="Optional limit on the number of baseline words",
    )
    parser.add_argument(
        "--baseline-sample",
        type=int,
        help=(
            "Sample size for baseline words when computing each vector. "
            "Defaults to using the full baseline list."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory where vectors will be cached",
    )
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Template used to construct prompts (must include {word})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute vectors even when cache files already exist",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_cli(argv)
    _run_cli(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    logging.basicConfig(level=logging.INFO)
    main()
