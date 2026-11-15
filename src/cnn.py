"""CNN : architecture, entraînement avec callbacks, évaluation, sauvegarde (étapes 2, 3, 4 et 6)."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from load_data import CLASS_NAMES, get_dataloaders

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = PROJECT_ROOT / "models"
REPORTS = PROJECT_ROOT / "reports"


# --------------------------------------------------------------------------- #
# Étape 2 : architecture
# --------------------------------------------------------------------------- #
def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    """Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2 (divise la taille par 2)."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class BrainTumorCNN(nn.Module):
    """4 blocs de convolution (32 -> 64 -> 128 -> 256 filtres par défaut), puis classifieur.

    Entrée (B, 1, 224, 224) -> sortie (B, 4) : un score (logit) par classe.
    """

    def __init__(self, filters: int = 32, dense: int = 128, dropout: float = 0.3,
                 n_classes: int = 4):
        super().__init__()
        f = [filters, filters * 2, filters * 4, filters * 8]
        self.features = nn.Sequential(
            conv_block(1, f[0]),     # 224 -> 112
            conv_block(f[0], f[1]),  # 112 -> 56
            conv_block(f[1], f[2]),  # 56 -> 28
            conv_block(f[2], f[3]),  # 28 -> 14  (dernière conv : utilisée par Grad-CAM)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling : (f[3], 14, 14) -> (f[3])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(f[3], dense),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


def build_model(config: dict) -> BrainTumorCNN:
    """Construit le modèle depuis la section 'model' de la config (utilisé aussi par l'appli)."""
    m = config.get("model", config)
    return BrainTumorCNN(filters=m.get("filters", 32), dense=m.get("dense", 128),
                         dropout=m.get("dropout", 0.3), n_classes=len(CLASS_NAMES))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- #
# Étape 3 : entraînement
# --------------------------------------------------------------------------- #
def class_weights(y: np.ndarray) -> torch.Tensor:
    """Poids = inverse de la fréquence : une erreur sur une classe rare compte plus."""
    counts = np.bincount(y, minlength=len(CLASS_NAMES))
    return torch.tensor(len(y) / (len(CLASS_NAMES) * counts), dtype=torch.float32)


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Une passe sur le loader. Avec optimizer : entraînement, sinon : évaluation."""
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def loader_options(config: dict) -> dict:
    """Options de prétraitement de la config (section 'preprocessing')."""
    pre = config.get("preprocessing", {})
    return {"per_image": pre.get("per_image_norm", False),
            "strong_aug": pre.get("strong_augment", False)}


def train_model(config: dict, data, device, verbose: bool = True, trial=None):
    """Entraîne le modèle avec 3 callbacks :

    - ModelCheckpoint : on garde les poids de la meilleure epoch (val_loss la plus basse)
    - ReduceLROnPlateau : learning rate divisé par 2 si la val_loss stagne 3 epochs
    - EarlyStopping : arrêt si la val_loss ne s'améliore plus pendant `patience` epochs

    Renvoie (modèle avec les meilleurs poids, historique).
    """
    t = config["training"]
    torch.manual_seed(config["project"]["seed"])
    train_loader, val_loader, _ = get_dataloaders(
        data, batch_size=t["batch_size"], augment=t["augment"], num_workers=t["num_workers"],
        seed=config["project"]["seed"], **loader_options(config))

    model = build_model(config).to(device)
    weights = class_weights(data.y_train).to(device) if t.get("class_weights", True) else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=t["learning_rate"],
                                 weight_decay=t.get("weight_decay", 0.0))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                           patience=3)

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "lr": []}
    best_loss, best_state, best_epoch, wait = float("inf"), None, 0, 0
    for epoch in range(1, t["epochs"] + 1):
        start = time.time()
        loss, acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        for k, v in zip(history, [loss, acc, val_loss, val_acc, lr]):
            history[k].append(v)

        improved = val_loss < best_loss
        if improved:  # ModelCheckpoint
            best_loss, best_epoch, wait = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
        if verbose:
            print(f"epoch {epoch:2d} | loss {loss:.4f} acc {acc:.3f} | val_loss {val_loss:.4f} "
                  f"val_acc {val_acc:.3f} | lr {lr:.1e} | {time.time() - start:.0f}s"
                  + ("  *" if improved else ""))
        if trial is not None:  # Optuna : arrêt des essais peu prometteurs
            trial.report(val_loss, epoch)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
        if wait >= t.get("patience", 7):  # EarlyStopping
            if verbose:
                print(f"early stopping : pas d'amélioration depuis {wait} epochs")
            break

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    return model, history


# --------------------------------------------------------------------------- #
# Étape 4 : évaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model, loader, device):
    """Renvoie (labels vrais, probabilités (N, 4))."""
    model.eval()
    ys, probs = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), dim=1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(probs).numpy()


def evaluate(model, loader, device) -> dict:
    """Accuracy, loss, rapport de classification par classe et matrice de confusion."""
    from sklearn.metrics import classification_report, confusion_matrix, log_loss

    y, p = predict(model, loader, device)
    pred = p.argmax(1)
    return {
        "accuracy": float((pred == y).mean()),
        "loss": float(log_loss(y, p, labels=range(len(CLASS_NAMES)))),
        "classification_report": classification_report(
            y, pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "class_names": CLASS_NAMES,
    }


# --------------------------------------------------------------------------- #
# Étape 6 : sauvegarde / chargement
# --------------------------------------------------------------------------- #
def save_model(model, config: dict, stats: dict, path: Path = MODELS / "best_model.pt",
               metrics: dict | None = None) -> Path:
    """Sauvegarde les poids + tout ce qu'il faut pour reconstruire et utiliser le modèle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": config,
        "class_names": CLASS_NAMES,
        "stats": {**{k: stats[k] for k in ["image_size", "mean", "std", "channels"]},
                  "per_image_norm": loader_options(config)["per_image"]},
        "metrics": metrics,
    }, path)
    (path.parent / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def load_model(path: Path = MODELS / "best_model.pt", device="cpu"):
    """Recharge le modèle sauvegardé. Renvoie (modèle en mode eval, checkpoint)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
