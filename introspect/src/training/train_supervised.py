"""Supervised introspection training with explicit concept labels."""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from peft import LoraConfig, TaskType, get_peft_model

    PEFT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PEFT_AVAILABLE = False

from ..adapters.base import BaseModelAdapter
from ..eval_common import ensure_vector, load_adapter_from_registry, load_words
from ..generation import build_chat_prompt
from ..io_utils import seed_everything, setup_logging
from ..inject import InjectionSpec, attach_injection, resolve_injection_positions
from ..prompts import task_a_paper_messages
from ..vectors import DEFAULT_WORDS_PATH
from .introspection_head import IntrospectionHead, IntrospectionHeadConfig

LOGGER = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SupervisedTrainingConfig:
    """Configuration for supervised introspection."""

    # Model + phase
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    phase: str = "head_only"  # "head_only" | "joint"

    # Injection settings
    layers: list[int] = field(default_factory=lambda: [14, 16, 18])
    alphas: list[float] = field(default_factory=lambda: [2.0, 4.0])
    capture_offset: int = 4
    injection_mode: str = "prefix"
    assistant_marker: Optional[str] = None

    # Data
    n_concepts: int = 50
    samples_per_concept: int = 10
    words_file: Path = DEFAULT_WORDS_PATH
    baseline_sample_size: int = 50
    prompt_template: str = "Tell me about {word}."
    val_ratio: float = 0.1

    # Head architecture
    head_intermediate_size: Optional[int] = None
    head_dropout: float = 0.1

    # Training
    batch_size: int = 8
    head_lr: float = 1e-3
    learning_rate: float = 1e-4
    num_epochs: int = 10
    warmup_steps: int = 0
    weight_decay: float = 0.01

    # Loss weights
    detection_weight: float = 0.3
    concept_weight: float = 0.7
    generation_weight: float = 0.5  # Only applied for joint training

    # LoRA (joint training)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Paths
    output_dir: Path = Path("results/supervised")
    head_checkpoint: Optional[Path] = None
    vector_cache_dir: Path = Path("results/vectors")

    # Misc
    seed: int = 42
    device: str = "cuda"


# =============================================================================
# Dataset
# =============================================================================


@dataclass
class SupervisedSample:
    """Single training sample with explicit labels."""

    prompt: str
    concept_word: str
    concept_id: int
    layer_idx: int
    alpha: float
    is_injection: bool
    target_response: str


class IntrospectionDataset(Dataset):
    """Dataset for supervised introspection."""

    def __init__(self, samples: list[SupervisedSample], tokenizer, max_length: int = 512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        prompt_tokens = self.tokenizer(
            sample.prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding=False,
        )

        full_text = sample.prompt + sample.target_response
        full_tokens = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding=False,
        )

        return {
            "input_ids": prompt_tokens["input_ids"].squeeze(0),
            "attention_mask": prompt_tokens["attention_mask"].squeeze(0),
            "full_input_ids": full_tokens["input_ids"].squeeze(0),
            "full_attention_mask": full_tokens["attention_mask"].squeeze(0),
            "concept_id": sample.concept_id,
            "is_injection": 1 if sample.is_injection else 0,
            "layer_idx": sample.layer_idx,
            "alpha": sample.alpha,
            "concept_word": sample.concept_word,
            "prompt_length": prompt_tokens["input_ids"].shape[1],
        }


def collate_fn(batch: list[dict], pad_token_id: int) -> dict:
    """Pad batch samples."""

    max_len = max(item["input_ids"].shape[0] for item in batch)
    max_full_len = max(item["full_input_ids"].shape[0] for item in batch)

    def _pad_ids(tensor: torch.Tensor, target_len: int, pad_value: int) -> torch.Tensor:
        pad_len = target_len - tensor.shape[0]
        if pad_len <= 0:
            return tensor
        return torch.nn.functional.pad(tensor, (0, pad_len), value=pad_value)

    return {
        "input_ids": torch.stack([_pad_ids(item["input_ids"], max_len, pad_token_id) for item in batch]),
        "attention_mask": torch.stack([_pad_ids(item["attention_mask"], max_len, 0) for item in batch]),
        "full_input_ids": torch.stack([_pad_ids(item["full_input_ids"], max_full_len, pad_token_id) for item in batch]),
        "full_attention_mask": torch.stack(
            [_pad_ids(item["full_attention_mask"], max_full_len, 0) for item in batch]
        ),
        "concept_id": torch.tensor([item["concept_id"] for item in batch]),
        "is_injection": torch.tensor([item["is_injection"] for item in batch]),
        "layer_idx": torch.tensor([item["layer_idx"] for item in batch]),
        "alpha": torch.tensor([item["alpha"] for item in batch]),
        "prompt_length": torch.tensor([item["prompt_length"] for item in batch]),
        "concept_word": [item["concept_word"] for item in batch],
    }


# =============================================================================
# Data generation
# =============================================================================


def build_prompt_and_positions(
    adapter: BaseModelAdapter,
    injection_mode: str,
    assistant_marker: Optional[str],
) -> tuple[str, list[int | str], bool]:
    """Return the Task A prompt plus token positions for injection."""

    messages = task_a_paper_messages()
    prompt, _ = build_chat_prompt(adapter.tokenizer, messages)
    token_positions, _ = resolve_injection_positions(
        adapter,
        prompt,
        mode=injection_mode,
        assistant_marker=assistant_marker,
    )
    apply_to_generated = injection_mode == "suffix"
    if injection_mode == "suffix":
        token_positions = ["suffix"]  # sentinel understood by InjectionSpec helpers
    return prompt, token_positions, apply_to_generated


def build_concept_vocabulary(words_file: Path, n_concepts: int) -> tuple[list[str], dict[str, int]]:
    words = load_words(words_file)
    targets = list(words.iter_targets())[:n_concepts]
    concept_to_id = {word: idx for idx, word in enumerate(targets)}
    return targets, concept_to_id


def prebuild_concept_vectors(
    adapter: BaseModelAdapter,
    concepts: list[str],
    layers: list[int],
    *,
    baseline_words: list[str],
    prompt_template: str,
    baseline_sample_size: int,
    cache_dir: Path,
) -> dict[tuple[str, int], torch.Tensor]:
    """Build or load all (concept, layer) vectors and cache to disk."""

    rng = random.Random(13)  # deterministic across runs
    model_id = getattr(adapter.model.config, "_name_or_path", "unknown")
    vectors: dict[tuple[str, int], torch.Tensor] = {}

    for concept in tqdm(concepts, desc="Building concept vectors"):
        for layer in layers:
            vector = ensure_vector(
                adapter=adapter,
                model_id=model_id,
                layer_idx=layer,
                word=concept,
                cache_dir=cache_dir,
                baseline_words=baseline_words,
                prompt_template=prompt_template,
                baseline_sample_size=baseline_sample_size,
                rng=rng,
            )
            vectors[(concept, layer)] = vector

    return vectors


def generate_supervised_samples(
    adapter: BaseModelAdapter,
    config: SupervisedTrainingConfig,
    concept_to_id: dict[str, int],
) -> tuple[list[SupervisedSample], list[int], bool]:
    """Generate labeled samples for supervised training."""

    prompt, token_positions, apply_to_generated = build_prompt_and_positions(
        adapter,
        injection_mode=config.injection_mode,
        assistant_marker=config.assistant_marker,
    )

    rng = random.Random(config.seed)
    word_set = load_words(config.words_file)
    concepts = list(word_set.iter_targets())[: config.n_concepts]

    samples: list[SupervisedSample] = []
    for concept in concepts:
        concept_id = concept_to_id[concept]
        for layer_idx in config.layers:
            for alpha in config.alphas:
                for _ in range(config.samples_per_concept):
                    samples.append(
                        SupervisedSample(
                            prompt=prompt,
                            concept_word=concept,
                            concept_id=concept_id,
                            layer_idx=layer_idx,
                            alpha=alpha,
                            is_injection=True,
                            target_response=f"I detect an injected thought. It seems to be about {concept}.",
                        )
                    )
                    samples.append(
                        SupervisedSample(
                            prompt=prompt,
                            concept_word=concept,
                            concept_id=concept_id,
                            layer_idx=layer_idx,
                            alpha=0.0,
                            is_injection=False,
                            target_response="I do not detect any injected thought.",
                        )
                    )

    rng.shuffle(samples)
    return samples, token_positions, apply_to_generated


# =============================================================================
# Trainer
# =============================================================================


class SupervisedTrainer:
    """Trainer handling head-only or joint LoRA training."""

    def __init__(
        self,
        adapter: BaseModelAdapter,
        head: IntrospectionHead,
        config: SupervisedTrainingConfig,
        concept_vectors: dict[tuple[str, int], torch.Tensor],
        *,
        token_positions: list[int | str],
        apply_to_generated: bool,
    ) -> None:
        self.adapter = adapter
        self.head = head
        self.config = config
        self.concept_vectors = concept_vectors
        self.token_positions = token_positions
        self.apply_to_generated = apply_to_generated
        self.device = config.device
        self.global_step = 0
        self.allow_model_grads = config.phase != "head_only"

        self.head.to(self.device)

        if config.phase == "head_only":
            self.adapter.model.eval()
            for param in self.adapter.model.parameters():
                param.requires_grad = False
            self.optimizer = torch.optim.AdamW(
                self.head.parameters(),
                lr=config.head_lr,
                weight_decay=config.weight_decay,
            )
        else:
            if not PEFT_AVAILABLE:
                raise ImportError("peft is required for joint training")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
            )
            self.adapter.model = get_peft_model(self.adapter.model, lora_config)
            self.adapter.model.train()
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": self.head.parameters(), "lr": config.head_lr},
                    {"params": self.adapter.model.parameters(), "lr": config.learning_rate},
                ],
                weight_decay=config.weight_decay,
            )

    def _vector_for(self, concept: str, layer_idx: int) -> torch.Tensor:
        key = (concept, layer_idx)
        if key not in self.concept_vectors:
            raise KeyError(f"Missing concept vector for {key}")
        return self.concept_vectors[key].to(self.device)

    def _forward_single(
        self,
        full_input_ids: torch.Tensor,
        full_attention_mask: torch.Tensor,
        prompt_length: int,
        concept_word: str,
        layer_idx: int,
        alpha: float,
        is_injection: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run one sample with injection and capture last-token hidden."""

        capture_layer = layer_idx + self.config.capture_offset
        n_layers = getattr(self.adapter.model.config, "num_hidden_layers", None)
        if n_layers is not None:
            capture_layer = max(0, min(capture_layer, n_layers - 1))
        captured: dict[str, torch.Tensor] = {}

        labels: Optional[torch.Tensor] = None
        if self.config.phase == "joint":
            labels = full_input_ids.clone()
            labels[:prompt_length] = -100
            labels = labels.unsqueeze(0)

        injection_handle = None
        if is_injection and alpha > 0:
            spec = InjectionSpec(
                layer_idx=layer_idx,
                alpha=alpha,
                vector=self._vector_for(concept_word, layer_idx),
                token_positions=self.token_positions,
                apply_to_generated=self.apply_to_generated,
            )
            # Regression guard: injection hook must be registered before capture hook.
            injection_handle = attach_injection(self.adapter, spec)

        def capture_fn(_module, _inputs, output):
            residual = output[0] if isinstance(output, tuple) else output
            captured["hidden"] = residual[0, prompt_length - 1, :]
            return output

        capture_handle = self.adapter.register_residual_hook(capture_layer, capture_fn)

        try:
            with torch.set_grad_enabled(self.allow_model_grads):
                outputs = self.adapter.model(
                    input_ids=full_input_ids.unsqueeze(0).to(self.device),
                    attention_mask=full_attention_mask.unsqueeze(0).to(self.device),
                    labels=labels.to(self.device) if labels is not None else None,
                )
            if "hidden" not in captured:
                raise RuntimeError("Failed to capture hidden state; check hook ordering")
            generation_loss = outputs.loss if labels is not None else None
            return captured["hidden"], generation_loss
        finally:
            capture_handle.remove()
            if injection_handle is not None:
                injection_handle.remove()

    def train_step(self, batch: dict) -> dict[str, float]:
        self.optimizer.zero_grad()
        self.head.train()

        batch_size = batch["input_ids"].shape[0]
        hidden_states: list[torch.Tensor] = []
        generation_losses: list[torch.Tensor] = []

        for i in range(batch_size):
            hidden, gen_loss = self._forward_single(
                full_input_ids=batch["full_input_ids"][i],
                full_attention_mask=batch["full_attention_mask"][i],
                prompt_length=int(batch["prompt_length"][i].item()),
                concept_word=batch["concept_word"][i],
                layer_idx=int(batch["layer_idx"][i].item()),
                alpha=float(batch["alpha"][i].item()),
                is_injection=bool(batch["is_injection"][i].item()),
            )
            hidden_states.append(hidden)
            if gen_loss is not None:
                generation_losses.append(gen_loss)

        hidden_stack = torch.stack(hidden_states, dim=0)
        concept_ids = batch["concept_id"].to(self.device)
        is_injection = batch["is_injection"].to(self.device).float()

        loss_dict = self.head.compute_loss(
            hidden_states=hidden_stack,
            concept_labels=concept_ids,
            is_injection=is_injection,
            detection_weight=self.config.detection_weight,
            concept_weight=self.config.concept_weight,
        )

        total_loss = loss_dict["loss"]
        if self.config.phase == "joint" and generation_losses:
            gen_loss = torch.stack(generation_losses).mean()
            loss_dict["generation_loss"] = gen_loss
            total_loss = total_loss + self.config.generation_weight * gen_loss
            loss_dict["loss"] = total_loss

        total_loss.backward()
        self.optimizer.step()
        self.global_step += 1

        return {k: v.item() for k, v in loss_dict.items()}

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        metrics = {
            "loss": 0.0,
            "detection_loss": 0.0,
            "concept_loss": 0.0,
            "detection_acc": 0.0,
            "concept_acc": 0.0,
        }
        n_batches = 0

        for batch in tqdm(dataloader, desc="Training"):
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            batch_metrics = self.train_step(batch)
            for key, value in batch_metrics.items():
                metrics[key] = metrics.get(key, 0.0) + value
            n_batches += 1
        if n_batches == 0:
            return metrics
        return {k: v / n_batches for k, v in metrics.items()}

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        self.head.eval()
        detection_preds: list[int] = []
        detection_labels: list[int] = []
        concept_preds: list[int] = []
        concept_labels: list[int] = []
        is_injection_flags: list[int] = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                hidden_states: list[torch.Tensor] = []

                for i in range(batch["input_ids"].shape[0]):
                    hidden, _ = self._forward_single(
                        full_input_ids=batch["full_input_ids"][i],
                        full_attention_mask=batch["full_attention_mask"][i],
                        prompt_length=int(batch["prompt_length"][i].item()),
                        concept_word=batch["concept_word"][i],
                        layer_idx=int(batch["layer_idx"][i].item()),
                        alpha=float(batch["alpha"][i].item()),
                        is_injection=bool(batch["is_injection"][i].item()),
                    )
                    hidden_states.append(hidden)

                preds = self.head.predict(torch.stack(hidden_states, dim=0))
                detection_preds.extend(preds["detected"].cpu().int().tolist())
                detection_labels.extend(batch["is_injection"].cpu().int().tolist())
                concept_preds.extend(preds["concept_id"].cpu().int().tolist())
                concept_labels.extend(batch["concept_id"].cpu().int().tolist())
                is_injection_flags.extend(batch["is_injection"].cpu().int().tolist())

        detection_preds_t = torch.tensor(detection_preds)
        detection_labels_t = torch.tensor(detection_labels)
        concept_preds_t = torch.tensor(concept_preds)
        concept_labels_t = torch.tensor(concept_labels)
        is_injection_mask = torch.tensor(is_injection_flags).bool()

        tp = ((detection_preds_t == 1) & (detection_labels_t == 1)).sum().item()
        fp = ((detection_preds_t == 1) & (detection_labels_t == 0)).sum().item()
        tn = ((detection_preds_t == 0) & (detection_labels_t == 0)).sum().item()
        fn = ((detection_preds_t == 0) & (detection_labels_t == 1)).sum().item()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        correct_detections = (detection_preds_t == 1) & (detection_labels_t == 1)
        if correct_detections.any():
            concept_acc = (
                (concept_preds_t[correct_detections] == concept_labels_t[correct_detections])
                .float()
                .mean()
                .item()
            )
        else:
            concept_acc = 0.0

        return {
            "detection_accuracy": (tp + tn) / len(detection_labels_t) if detection_labels_t.numel() else 0.0,
            "tpr": tpr,
            "fpr": fpr,
            "net_score": tpr - fpr,
            "concept_accuracy": concept_acc,
            "n_correct_detections": int(correct_detections.sum().item()),
        }

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.head.save(path / "introspection_head")
        torch.save(
            {"optimizer": self.optimizer.state_dict(), "global_step": self.global_step},
            path / "trainer_state.pt",
        )
        with (path / "config.json").open("w", encoding="utf-8") as fh:
            json.dump(asdict(self.config), fh, indent=2, default=str)
        LOGGER.info("Saved checkpoint to %s", path)


# =============================================================================
# Entry point
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--phase", choices=["head_only", "joint"], default="head_only")
    parser.add_argument("--layers", type=int, nargs="+", default=[14, 16, 18])
    parser.add_argument("--alphas", type=float, nargs="+", default=[2.0, 4.0])
    parser.add_argument("--n-concepts", type=int, default=50)
    parser.add_argument("--samples-per-concept", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=Path, default=Path("results/supervised"))
    parser.add_argument("--head-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--injection-mode", choices=["prefix", "suffix"], default="prefix")
    parser.add_argument("--assistant-marker")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    seed_everything(args.seed)

    config = SupervisedTrainingConfig(
        model_name=args.model,
        phase=args.phase,
        layers=args.layers,
        alphas=args.alphas,
        n_concepts=args.n_concepts,
        samples_per_concept=args.samples_per_concept,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        head_lr=args.head_lr,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        head_checkpoint=args.head_checkpoint,
        seed=args.seed,
        injection_mode=args.injection_mode,
        assistant_marker=args.assistant_marker,
        device=args.device,
    )

    LOGGER.info("Loading model: %s", config.model_name)
    loaded = load_adapter_from_registry(
        config.model_name,
        device_map=config.device,
        seed=config.seed,
    )
    adapter = loaded.adapter

    concepts, concept_to_id = build_concept_vocabulary(config.words_file, config.n_concepts)
    word_set = load_words(config.words_file)
    baseline_words = list(word_set.iter_baselines())

    concept_vectors = prebuild_concept_vectors(
        adapter,
        concepts,
        config.layers,
        baseline_words=baseline_words,
        prompt_template=config.prompt_template,
        baseline_sample_size=config.baseline_sample_size,
        cache_dir=config.vector_cache_dir,
    )

    samples, token_positions, apply_to_generated = generate_supervised_samples(
        adapter, config, concept_to_id
    )

    split_idx = int(len(samples) * (1 - config.val_ratio))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    train_dataset = IntrospectionDataset(train_samples, adapter.tokenizer)
    val_dataset = IntrospectionDataset(val_samples, adapter.tokenizer)

    if adapter.tokenizer.pad_token_id is None:
        adapter.tokenizer.pad_token = adapter.tokenizer.eos_token

    collate = lambda batch: collate_fn(batch, adapter.tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate)

    hidden_size = adapter.model.config.hidden_size
    head_config = IntrospectionHeadConfig(
        hidden_size=hidden_size,
        n_concepts=len(concepts),
        intermediate_size=config.head_intermediate_size,
        dropout=config.head_dropout,
    )

    if config.head_checkpoint:
        LOGGER.info("Loading head checkpoint from %s", config.head_checkpoint)
        head = IntrospectionHead.load(config.head_checkpoint, device=config.device)
    else:
        head = IntrospectionHead(head_config)

    trainer = SupervisedTrainer(
        adapter,
        head,
        config,
        concept_vectors,
        token_positions=token_positions,
        apply_to_generated=apply_to_generated,
    )

    best_net = -1.0
    output_dir = config.output_dir / config.phase

    for epoch in range(config.num_epochs):
        LOGGER.info("Epoch %d/%d", epoch + 1, config.num_epochs)
        train_metrics = trainer.train_epoch(train_loader)
        LOGGER.info("Train metrics: %s", train_metrics)

        val_metrics = trainer.evaluate(val_loader)
        LOGGER.info("Val metrics: %s", val_metrics)

        if val_metrics["net_score"] > best_net:
            best_net = val_metrics["net_score"]
            trainer.save(output_dir / "best")
            LOGGER.info("New best net score: %.4f", best_net)

    trainer.save(output_dir / "final")
    LOGGER.info("Training complete. Best net score: %.4f", best_net)


if __name__ == "__main__":
    main()
