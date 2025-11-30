"""Introspection head for classifying injected concepts."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


@dataclass
class IntrospectionHeadConfig:
    """Configuration for the introspection head."""

    hidden_size: int
    n_concepts: int
    intermediate_size: Optional[int] = None
    dropout: float = 0.1
    use_layer_norm: bool = True

    def __post_init__(self) -> None:
        if self.intermediate_size is None:
            self.intermediate_size = self.hidden_size // 2


class IntrospectionHead(nn.Module):
    """Two-head MLP for detection + concept classification."""

    def __init__(self, config: IntrospectionHeadConfig) -> None:
        super().__init__()
        self.config = config

        self.layer_norm = (
            nn.LayerNorm(config.hidden_size) if config.use_layer_norm else nn.Identity()
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.n_concepts),
        )
        self.detection_head = nn.Linear(config.hidden_size, 2)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_detection: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Run the head on captured hidden states."""

        h = self.layer_norm(hidden_states)
        outputs: dict[str, torch.Tensor] = {
            "concept_logits": self.classifier(h),
        }

        if return_detection:
            outputs["detection_logits"] = self.detection_head(h)

        return outputs

    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        concept_labels: torch.Tensor,
        is_injection: torch.Tensor,
        *,
        detection_weight: float = 0.3,
        concept_weight: float = 0.7,
    ) -> dict[str, torch.Tensor]:
        """Compute detection + concept losses."""

        outputs = self.forward(hidden_states, return_detection=True)

        detection_loss = F.cross_entropy(
            outputs["detection_logits"],
            is_injection.long(),
        )
        detection_preds = outputs["detection_logits"].argmax(dim=-1)
        detection_acc = (detection_preds == is_injection.long()).float().mean()

        injection_mask = is_injection.bool()
        if injection_mask.any():
            concept_logits = outputs["concept_logits"][injection_mask]
            concept_labels_inj = concept_labels[injection_mask]
            concept_loss = F.cross_entropy(concept_logits, concept_labels_inj)
            concept_preds = concept_logits.argmax(dim=-1)
            concept_acc = (concept_preds == concept_labels_inj).float().mean()
        else:
            concept_loss = torch.tensor(0.0, device=hidden_states.device)
            concept_acc = torch.tensor(0.0, device=hidden_states.device)

        total = detection_weight * detection_loss + concept_weight * concept_loss

        return {
            "loss": total,
            "detection_loss": detection_loss,
            "concept_loss": concept_loss,
            "detection_acc": detection_acc,
            "concept_acc": concept_acc,
        }

    def predict(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return argmax predictions and confidences."""

        with torch.no_grad():
            outputs = self.forward(hidden_states, return_detection=True)
            detection_probs = F.softmax(outputs["detection_logits"], dim=-1)
            concept_probs = F.softmax(outputs["concept_logits"], dim=-1)

            return {
                "detected": detection_probs[:, 1] > 0.5,
                "detection_confidence": detection_probs[:, 1],
                "concept_id": concept_probs.argmax(dim=-1),
                "concept_confidence": concept_probs.max(dim=-1).values,
                "concept_probs": concept_probs,
            }

    def save(self, path: Path) -> None:
        """Persist weights + config."""

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "head_weights.pt")
        with (path / "head_config.json").open("w", encoding="utf-8") as fh:
            json.dump(vars(self.config), fh, indent=2)
        LOGGER.info("Saved introspection head to %s", path)

    @classmethod
    def load(cls, path: Path, *, device: str = "cpu") -> "IntrospectionHead":
        """Load a saved head checkpoint."""

        path = Path(path)
        with (path / "head_config.json").open(encoding="utf-8") as fh:
            config_dict = json.load(fh)
        config = IntrospectionHeadConfig(**config_dict)

        head = cls(config)
        state = torch.load(path / "head_weights.pt", map_location=device)
        head.load_state_dict(state)
        head.to(device)
        LOGGER.info("Loaded introspection head from %s", path)
        return head
