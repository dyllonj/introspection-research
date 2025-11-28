"""DPO training for introspection capabilities.

This script fine-tunes a model using Direct Preference Optimization
to improve its introspective awareness, using concept injection as
the ground-truth signal for preference pairs.

Usage:
    python -m introspect.src.training.train_dpo \
        --model meta-llama/Llama-2-7b-hf \
        --output-dir results/introspection_dpo \
        --n-concepts 50 \
        --layers 12 16 20 24 \
        --alphas 2.0 4.0
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from transformers import TrainingArguments

from ..adapters import LlamaAdapter, MistralAdapter
from ..eval_common import load_adapter_from_registry
from ..io_utils import seed_everything, setup_logging
from .data_generation import (
    PreferenceDataConfig,
    generate_task_a_preference_pairs,
    save_preference_dataset,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class DPOTrainingConfig:
    """Configuration for DPO training."""
    
    # Model
    model: str
    adapter: str | None = None
    dtype: str | None = None
    
    # Data generation
    n_concepts: int = 50
    layers: list[int] | None = None
    alphas: list[float] | None = None
    samples_per_concept: int = 4
    
    # Training
    output_dir: Path = Path("results/introspection_dpo")
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-7
    beta: float = 0.1  # DPO temperature parameter
    max_length: int = 512
    max_prompt_length: int = 384
    
    # LoRA (recommended for efficiency)
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    
    # Model args
    parser.add_argument("--model", required=True, help="Model identifier")
    parser.add_argument("--adapter", help="Adapter class name")
    parser.add_argument("--dtype", help="Data type (bf16/fp16/fp32)")
    
    # Data generation args
    parser.add_argument("--n-concepts", type=int, default=50)
    parser.add_argument("--layers", type=int, nargs="+", help="Layers for injection")
    parser.add_argument("--alphas", type=float, nargs="+", default=[2.0, 4.0])
    parser.add_argument("--samples-per-concept", type=int, default=4)
    
    # Training args
    parser.add_argument("--output-dir", type=Path, default=Path("results/introspection_dpo"))
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=0.1)
    
    # LoRA args
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generate-only", action="store_true", 
                       help="Only generate preference data, don't train")
    
    return parser


def _parse_config(argv: Sequence[str] | None = None) -> DPOTrainingConfig:
    parser = _build_parser()
    args = parser.parse_args(argv)
    
    return DPOTrainingConfig(
        model=args.model,
        adapter=args.adapter,
        dtype=args.dtype,
        n_concepts=args.n_concepts,
        layers=args.layers,
        alphas=args.alphas,
        samples_per_concept=args.samples_per_concept,
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
    )


def _infer_optimal_layers(model_config: object) -> list[int]:
    """Infer good injection layers based on model architecture."""
    n_layers = getattr(model_config, "num_hidden_layers", 32)
    
    # Target roughly 1/2 and 2/3 through the model (based on Anthropic findings)
    layer_mid = n_layers // 2
    layer_two_thirds = (2 * n_layers) // 3
    layer_late = (3 * n_layers) // 4
    
    return [layer_mid, layer_two_thirds, layer_late]


def run_data_generation(config: DPOTrainingConfig) -> Path:
    """Generate preference data for DPO training."""
    
    LOGGER.info("Loading model %s for data generation", config.model)
    
    loaded = load_adapter_from_registry(
        config.model,
        adapter_name=config.adapter,
        dtype=config.dtype,
        seed=config.seed,
    )
    
    # Infer layers if not specified
    layers = config.layers
    if layers is None:
        layers = _infer_optimal_layers(loaded.adapter.model.config)
        LOGGER.info("Auto-selected layers: %s", layers)
    
    # Configure data generation
    data_config = PreferenceDataConfig(
        n_concepts=config.n_concepts,
        layers=layers,
        alphas=config.alphas or [2.0, 4.0],
        samples_per_concept=config.samples_per_concept,
        seed=config.seed,
    )
    
    # Generate samples
    LOGGER.info("Generating preference pairs...")
    samples = list(generate_task_a_preference_pairs(loaded.adapter, data_config))
    
    LOGGER.info("Generated %d preference pairs", len(samples))
    
    # Save to disk
    output_path = config.output_dir / "preference_data.jsonl"
    save_preference_dataset(samples, output_path, format="jsonl")
    
    return output_path


def run_dpo_training(config: DPOTrainingConfig, data_path: Path) -> None:
    """Run DPO training using TRL library."""
    
    try:
        from trl import DPOTrainer, DPOConfig
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        LOGGER.error(
            "DPO training requires 'trl' and 'peft' packages. "
            "Install with: pip install trl peft"
        )
        raise e
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    
    LOGGER.info("Loading model for training: %s", config.model)
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Apply LoRA if enabled
    if config.use_lora:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules or [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Load dataset
    dataset = load_dataset("json", data_files=str(data_path), split="train")
    
    # Split into train/eval
    dataset = dataset.train_test_split(test_size=0.1, seed=config.seed)
    
    # Format for DPO
    def format_example(example):
        return {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }
    
    train_dataset = dataset["train"].map(format_example)
    eval_dataset = dataset["test"].map(format_example)
    
    LOGGER.info(
        "Training on %d examples, evaluating on %d",
        len(train_dataset),
        len(eval_dataset),
    )
    
    # Configure DPO training
    dpo_config = DPOConfig(
        output_dir=str(config.output_dir / "checkpoints"),
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
    
    # Create trainer
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    # Train
    LOGGER.info("Starting DPO training...")
    trainer.train()
    
    # Save final model
    final_path = config.output_dir / "final_model"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    
    LOGGER.info("Training complete. Model saved to %s", final_path)


def main(argv: Sequence[str] | None = None) -> None:
    setup_logging()
    config = _parse_config(argv)
    
    seed_everything(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Generate preference data
    data_path = run_data_generation(config)
    
    if getattr(config, "generate_only", False):
        LOGGER.info("Data generation complete. Exiting (--generate-only).")
        return
    
    # Step 2: Run DPO training
    run_dpo_training(config, data_path)


if __name__ == "__main__":
    main()
