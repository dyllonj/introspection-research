"""DPO fine-tuning pipeline for introspection (optional Phase 2)."""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from ..eval_common import load_adapter_from_registry, load_words
from ..io_utils import seed_everything, setup_logging
from ..vectors import DEFAULT_WORDS_PATH
from .data_generation import (
    PreferenceDataConfig,
    generate_task_a_preference_pairs,
    save_preference_dataset,
)
from .probe_feasibility import run_probe_analysis

LOGGER = logging.getLogger(__name__)


@dataclass
class DPOTrainingConfig:
    """Configuration for DPO training."""

    model: str
    adapter: str | None = None
    dtype: str | None = None

    # Data generation
    n_concepts: int = 50
    layers: list[int] | None = None
    alphas: list[float] | None = None
    samples_per_concept: int = 4
    injection_mode: str = "prefix"
    assistant_marker: str | None = None
    rebuild_vectors: bool = False

    # Held-out concepts
    holdout_concepts: list[str] | None = None
    holdout_count: int = 0

    # Training
    output_dir: Path = Path("results/introspection_dpo")
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-7
    beta: float = 0.1
    max_length: int = 512
    max_prompt_length: int = 384

    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None

    # Misc
    seed: int = 42
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    generate_only: bool = False
    dry_run: bool = False
    skip_probe_check: bool = False
    probe_threshold: float = 0.70


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model identifier")
    parser.add_argument("--adapter", help="Adapter class name")
    parser.add_argument("--dtype", help="Data type (bf16/fp16/fp32)")

    parser.add_argument("--n-concepts", type=int, default=50)
    parser.add_argument("--layers", type=int, nargs="+", help="Layers for injection")
    parser.add_argument("--alphas", type=float, nargs="+", default=[2.0, 4.0])
    parser.add_argument("--samples-per-concept", type=int, default=4)
    parser.add_argument("--injection-mode", choices=["prefix", "suffix"], default="prefix")
    parser.add_argument("--assistant-marker", help="Override detected assistant marker")
    parser.add_argument("--rebuild-vectors", action="store_true", help="Regenerate vectors even if cached")

    parser.add_argument("--holdout-concepts", type=Path, help="Path to line-delimited holdout concept list")
    parser.add_argument("--holdout-count", type=int, default=0, help="Sample this many concepts for holdout")

    parser.add_argument("--output-dir", type=Path, default=Path("results/introspection_dpo"))
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=0.1)

    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generate-only", action="store_true", help="Only generate preference data")
    parser.add_argument("--dry-run", action="store_true", help="Parse/configure but skip generation and training")
    parser.add_argument("--skip-probe-check", action="store_true", help="Skip probe feasibility gate")
    return parser


def _parse_config(argv: Sequence[str] | None = None) -> DPOTrainingConfig:
    parser = _build_parser()
    args = parser.parse_args(argv)

    holdout_concepts: list[str] | None = None
    if args.holdout_concepts is not None and args.holdout_concepts.exists():
        holdout_concepts = [
            line.strip()
            for line in args.holdout_concepts.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    return DPOTrainingConfig(
        model=args.model,
        adapter=args.adapter,
        dtype=args.dtype,
        n_concepts=args.n_concepts,
        layers=args.layers,
        alphas=args.alphas,
        samples_per_concept=args.samples_per_concept,
        injection_mode=args.injection_mode,
        assistant_marker=args.assistant_marker,
        rebuild_vectors=args.rebuild_vectors,
        holdout_concepts=holdout_concepts,
        holdout_count=args.holdout_count,
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
        generate_only=args.generate_only,
        dry_run=args.dry_run,
        skip_probe_check=args.skip_probe_check,
    )


def _infer_optimal_layers(model_config: object) -> list[int]:
    n_layers = getattr(model_config, "num_hidden_layers", 32)
    layer_mid = n_layers // 2
    layer_two_thirds = (2 * n_layers) // 3
    layer_late = (3 * n_layers) // 4
    return [layer_mid, layer_two_thirds, layer_late]


def _resolve_holdout(words_path: Path, config: DPOTrainingConfig) -> list[str]:
    words = load_words(words_path)
    if config.holdout_concepts:
        return config.holdout_concepts
    if config.holdout_count <= 0:
        return []
    rng = random.Random(config.seed + 137)
    targets = list(words.iter_targets())
    count = min(config.holdout_count, len(targets))
    return rng.sample(targets, count)


def run_data_generation(config: DPOTrainingConfig) -> tuple[Path, list[str]]:
    LOGGER.info("Loading model %s for data generation", config.model)
    loaded = load_adapter_from_registry(
        config.model,
        adapter_name=config.adapter,
        dtype=config.dtype,
        seed=config.seed,
    )
    adapter = loaded.adapter

    layers = config.layers
    if layers is None:
        layers = _infer_optimal_layers(adapter.model.config)
        LOGGER.info("Auto-selected layers: %s", layers)

    holdout = _resolve_holdout(Path(DEFAULT_WORDS_PATH), config)
    if holdout:
        LOGGER.info("Holding out %d concepts for eval: %s", len(holdout), ", ".join(holdout))

    data_config = PreferenceDataConfig(
        n_concepts=config.n_concepts,
        layers=layers,
        alphas=config.alphas or [2.0, 4.0],
        samples_per_concept=config.samples_per_concept,
        seed=config.seed,
        injection_mode=config.injection_mode,
        assistant_marker=config.assistant_marker,
        holdout_concepts=holdout,
        rebuild_vectors=config.rebuild_vectors,
    )

    LOGGER.info("Generating preference pairs...")
    samples = list(generate_task_a_preference_pairs(adapter, data_config))
    LOGGER.info("Generated %d preference pairs", len(samples))

    safe_model = config.model.replace("/", "--")
    output_root = config.output_dir / safe_model
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "preference_data.jsonl"
    save_preference_dataset(samples, output_path, format="jsonl")

    if holdout:
        holdout_path = output_root / "holdout_concepts.json"
        holdout_path.write_text(json.dumps(holdout, indent=2), encoding="utf-8")

    return output_path, holdout


def run_dpo_training(config: DPOTrainingConfig, data_path: Path, model_root: Path) -> None:
    try:  # pragma: no cover - optional deps
        from trl import DPOTrainer, DPOConfig
        from peft import LoraConfig, get_peft_model
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.error(
            "DPO training requires `trl`, `peft`, `datasets`, and `transformers`. "
            "Install with `pip install trl peft datasets transformers` or the `[train]` extra."
        )
        raise

    LOGGER.info("Loading model for training: %s", config.model)
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.use_lora:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules or [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=str(data_path), split="train")
    dataset = dataset.train_test_split(test_size=0.1, seed=config.seed)

    def format_example(example):
        return {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }

    train_dataset = dataset["train"].map(format_example)
    eval_dataset = dataset["test"].map(format_example)

    LOGGER.info("Training on %d examples, evaluating on %d", len(train_dataset), len(eval_dataset))

    dpo_config = DPOConfig(
        output_dir=str(model_root / "checkpoints"),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        beta=config.beta,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    LOGGER.info("Starting DPO training...")
    trainer.train()

    final_path = model_root / "final_model"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    LOGGER.info("Training complete. Model saved to %s", final_path)


def main(argv: Sequence[str] | None = None) -> None:
    setup_logging()
    config = _parse_config(argv)

    seed_everything(config.seed)

    if config.dry_run:
        LOGGER.info("Dry run: parsed config and seeded RNG. Exiting.")
        return

    # Probe gate
    if not config.skip_probe_check:
        try:
            probe_results = run_probe_analysis(
                model=config.model,
                n_concepts=min(config.n_concepts, 20),
                layers=config.layers,
                alpha=config.alphas[0] if config.alphas else 2.0,
                seed=config.seed,
                injection_mode=config.injection_mode,
                output=None,
            )
            best_acc = probe_results.get("best_accuracy", 0.0)
            if best_acc < config.probe_threshold:
                LOGGER.warning(
                    "Probe accuracy %.1f%% below threshold %.0f%%. "
                    "Use --skip-probe-check to bypass.",
                    best_acc * 100,
                    config.probe_threshold * 100,
                )
                return
        except ImportError as exc:
            LOGGER.warning("Probe feasibility skipped (dependency missing): %s", exc)

    data_path, _holdout = run_data_generation(config)

    if config.generate_only:
        LOGGER.info("Data generation complete. Exiting (--generate-only).")
        return

    safe_model = config.model.replace("/", "--")
    model_root = config.output_dir / safe_model
    model_root.mkdir(parents=True, exist_ok=True)

    run_dpo_training(config, data_path, model_root)


if __name__ == "__main__":
    main()
