"""Étapes 2, 3, 4 et 6 : le CNN, son entraînement, son évaluation et sa sauvegarde."""

import copy
import json
import time
from pathlib import Path

import torch
from torch import nn

from load_data import CLASS_NAMES, get_dataloaders

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = PROJECT_ROOT / "models"


# ---------- étape 2 : l'architecture ----------

def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),  # padding=1 : la taille ne change pas
        nn.BatchNorm2d(c_out),  # stabilise l'entraînement
        nn.ReLU(),
        nn.MaxPool2d(2),  # divise la hauteur et la largeur par 2
    )


class BrainTumorCNN(nn.Module):
    """4 blocs de convolution puis un classifieur. Entrée (B, 1, 224, 224), sortie (B, 4)."""

    def __init__(self, filters=32, dense=128, dropout=0.3):
        super().__init__()
        f = [filters, filters * 2, filters * 4, filters * 8]  # on double à chaque bloc
        self.features = nn.Sequential(
            conv_block(1, f[0]),     # 224 -> 112
            conv_block(f[0], f[1]),  # 112 -> 56
            conv_block(f[1], f[2]),  # 56 -> 28
            conv_block(f[2], f[3]),  # 28 -> 14, la dernière conv sert au Grad-CAM
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # moyenne de chaque carte 14x14 -> 1 nombre par filtre
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(f[3], dense),
            nn.ReLU(),
            nn.Dropout(dropout),  # éteint des neurones au hasard, contre le surapprentissage
            nn.Linear(dense, len(CLASS_NAMES)),  # 1 score par classe (softmax fait dans la loss)
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_model(config):
    m = config.get("model", {})
    return BrainTumorCNN(m.get("filters", 32), m.get("dense", 128), m.get("dropout", 0.3))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# ---------- étape 3 : l'entraînement ----------

def run_epoch(model, loader, criterion, device, optimizer=None):
    """Une passe sur toutes les images. Avec optimizer on entraîne, sans on évalue."""
    train = optimizer is not None
    model.train(train)  # active ou coupe le dropout
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()  # sinon les gradients s'additionnent d'un batch à l'autre
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def train_model(config, data, device, verbose=True, trial=None):
    """Entraîne le modèle, renvoie (modèle avec ses meilleurs poids, historique)."""
    t = config["training"]
    torch.manual_seed(config["project"]["seed"])
    train_loader, val_loader, _ = get_dataloaders(
        data, t["batch_size"], t["augment"], t["num_workers"], config["project"]["seed"])

    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=t["learning_rate"],
                                 weight_decay=t["weight_decay"])
    # callback 1 : learning rate divisé par 2 si la val_loss stagne 3 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

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
        if improved:  # callback 2 : on garde les poids de la meilleure epoch
            best_loss, best_epoch, wait = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
        if verbose:
            print(f"epoch {epoch:2d} | loss {loss:.4f} acc {acc:.3f} | val_loss {val_loss:.4f} "
                  f"val_acc {val_acc:.3f} | lr {lr:.1e} | {time.time() - start:.0f}s"
                  + ("  *" if improved else ""))
        if trial is not None:  # Optuna : on arrête les essais qui partent mal
            trial.report(val_loss, epoch)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
        if wait >= t["patience"]:  # callback 3 : early stopping
            if verbose:
                print(f"early stopping : pas d'amélioration depuis {wait} epochs")
            break

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    return model, history


# ---------- étape 4 : l'évaluation ----------

@torch.no_grad()
def predict(model, loader, device):
    """Renvoie (vrais labels, probabilités de chaque classe)."""
    model.eval()
    ys, probs = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), dim=1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(probs).numpy()


def evaluate(model, loader, device):
    from sklearn.metrics import classification_report, confusion_matrix, log_loss

    y, p = predict(model, loader, device)
    pred = p.argmax(1)  # classe prédite = la plus probable
    return {
        "accuracy": float((pred == y).mean()),
        "loss": float(log_loss(y, p, labels=range(len(CLASS_NAMES)))),
        "classification_report": classification_report(
            y, pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "class_names": CLASS_NAMES,
    }


# ---------- étape 6 : sauvegarde et rechargement ----------

def save_model(model, config, stats, metrics=None, path=MODELS / "best_model.pt"):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),  # les poids
        "config": config,  # de quoi reconstruire la même architecture
        "class_names": CLASS_NAMES,
        "stats": {"image_size": stats["image_size"], "channels": stats["channels"]},
        "metrics": metrics,
    }, path)
    (path.parent / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def load_model(path=MODELS / "best_model.pt", device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
