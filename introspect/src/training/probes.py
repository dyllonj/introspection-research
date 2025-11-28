"""Linear probes for detecting concept injection in activations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

LOGGER = logging.getLogger(__name__)


@dataclass
class ProbeMetrics:
    """Evaluation metrics for the injection probe."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    auroc: float

    tpr: float  # True positive rate
    fpr: float  # False positive rate
    tnr: float  # True negative rate
    fnr: float  # False negative rate


class InjectionProbe(nn.Module):
    """A linear probe to detect concept injection from activations."""

    def __init__(self, hidden_size: int, *, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        if activations.dim() == 1:
            activations = activations.unsqueeze(0)
        return self.classifier(activations)

    def predict_proba(self, activations: torch.Tensor) -> torch.Tensor:
        logits = self.forward(activations)
        return torch.sigmoid(logits)

    def predict(self, activations: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        probs = self.predict_proba(activations)
        return (probs >= threshold).long()


class ConceptIdentificationProbe(nn.Module):
    """Multi-class probe to identify which concept was injected."""

    def __init__(self, hidden_size: int, n_concepts: int, *, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_concepts = n_concepts
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_concepts + 1),
        )

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        if activations.dim() == 1:
            activations = activations.unsqueeze(0)
        return self.classifier(activations)

    def predict(self, activations: torch.Tensor) -> torch.Tensor:
        logits = self.forward(activations)
        return logits.argmax(dim=-1)


def train_injection_probe(
    probe: InjectionProbe,
    train_activations: torch.Tensor,
    train_labels: torch.Tensor,
    *,
    val_activations: torch.Tensor | None = None,
    val_labels: torch.Tensor | None = None,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 10,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, list[float]]:
    """Train the injection detection probe."""

    device = torch.device(device)
    probe = probe.to(device)

    train_dataset = TensorDataset(
        train_activations.to(device),
        train_labels.float().to(device),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    has_val = val_activations is not None and val_labels is not None
    if has_val:
        val_dataset = TensorDataset(
            val_activations.to(device),
            val_labels.float().to(device),
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        probe.train()
        train_loss = 0.0
        for batch_acts, batch_labels in train_loader:
            optimizer.zero_grad()
            logits = probe(batch_acts).squeeze(-1)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_acts)

        train_loss /= len(train_dataset)
        history["train_loss"].append(train_loss)

        if has_val:
            probe.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_acts, batch_labels in val_loader:
                    logits = probe(batch_acts).squeeze(-1)
                    loss = criterion(logits, batch_labels)
                    val_loss += loss.item() * len(batch_acts)
                    preds = (torch.sigmoid(logits) >= 0.5).float()
                    correct += (preds == batch_labels).sum().item()
                    total += len(batch_labels)

            val_loss /= len(val_dataset)
            val_accuracy = correct / total
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    LOGGER.info("Early stopping at epoch %d", epoch + 1)
                    break

    if best_state is not None:
        probe.load_state_dict(best_state)
        probe = probe.to(device)

    return history


def evaluate_probe(
    probe: InjectionProbe,
    activations: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
) -> ProbeMetrics:
    """Evaluate the probe on a test set."""

    try:  # pragma: no cover - optional dependency
        from sklearn.metrics import roc_auc_score
    except Exception as exc:  # pragma: no cover - defensive
        raise ImportError(
            "evaluate_probe requires scikit-learn. Install with `pip install scikit-learn` "
            "or the `[train]` extra."
        ) from exc

    device = torch.device(device)
    probe = probe.to(device).eval()

    activations = activations.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        probs = probe.predict_proba(activations).squeeze(-1)
        preds = (probs >= 0.5).long()

    labels = labels.long()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    total = len(labels)
    n_positive = (labels == 1).sum().item()
    n_negative = (labels == 0).sum().item()

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    tpr = tp / n_positive if n_positive > 0 else 0.0
    fpr = fp / n_negative if n_negative > 0 else 0.0
    tnr = tn / n_negative if n_negative > 0 else 0.0
    fnr = fn / n_positive if n_positive > 0 else 0.0

    try:
        auroc = roc_auc_score(labels.cpu().numpy(), probs.cpu().numpy())
    except Exception:  # pragma: no cover - defensive
        auroc = 0.0

    return ProbeMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auroc=auroc,
        tpr=tpr,
        fpr=fpr,
        tnr=tnr,
        fnr=fnr,
    )


def save_probe(probe: InjectionProbe, path: Path) -> None:
    """Save probe weights and configuration."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": probe.state_dict(), "hidden_size": probe.hidden_size}, path)
    LOGGER.info("Saved probe to %s", path)


def load_probe(path: Path, device: str | torch.device = "cpu") -> InjectionProbe:
    """Load a saved probe."""

    checkpoint = torch.load(path, map_location=device)
    probe = InjectionProbe(checkpoint["hidden_size"])
    probe.load_state_dict(checkpoint["state_dict"])
    return probe.to(device)
