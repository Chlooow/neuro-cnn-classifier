"""Chargement et prétraitement du dataset Brain Tumor MRI (étape 1 du sujet).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DATASET = "masoudnickparvar/brain-tumor-mri-dataset"

# Ordre alphabétique = ordre des dossiers = ordre des sorties du modèle.
# C'est la seule source de vérité pour l'ordre des classes (modèle, appli, PDF).
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


# --------------------------------------------------------------------------- #
# Lecture du dataset
# --------------------------------------------------------------------------- #
def load_brain_tumor_data() -> Path:
    """Télécharge le dataset via kagglehub (en cache après le 1er appel), renvoie son chemin."""
    import kagglehub

    return Path(kagglehub.dataset_download(KAGGLE_DATASET))


def list_images(root: Path, split: str) -> tuple[list[Path], np.ndarray]:
    """Liste les images d'un split ('Training' ou 'Testing') et leurs labels (indices de classe).

    ImageFolder déduit la classe du nom du dossier, dans l'ordre alphabétique.
    """
    folder = ImageFolder(root / split)
    assert folder.classes == CLASS_NAMES, folder.classes
    return [Path(p) for p, _ in folder.samples], np.array(folder.targets, dtype=np.int64)


def remove_duplicates(paths: list[Path], labels: np.ndarray) -> tuple[list[Path], np.ndarray]:
    """Supprime les doublons exacts (même contenu de fichier), en gardant la 1re occurrence.

    Sans ça, une même image pourrait se retrouver à la fois en train et en validation,
    ce qui surestimerait la performance en validation.
    """
    seen, keep = set(), []
    for i, p in enumerate(paths):
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h not in seen:
            seen.add(h)
            keep.append(i)
    return [paths[i] for i in keep], labels[keep]


def near_duplicates(x: np.ndarray, ref: np.ndarray | None = None,
                    threshold: float = 0.02) -> np.ndarray:
    """Repère les images quasi identiques (même IRM enregistrée à une autre taille, etc.).

    On compare des miniatures 32x32 : la distance est l'écart moyen entre pixels (0 à 1).
    Sous 0.02, c'est la même image (vérifié à l'œil) ; au-dessus, ce sont des coupes voisines.
    - ref donné : True si l'image de x a une quasi-copie dans ref
    - ref absent : True si l'image a une quasi-copie plus tôt dans x (on garde la 1re)
    """
    def thumbs(a):
        t = torch.from_numpy(a).float().unsqueeze(1) / 255
        return torch.nn.functional.adaptive_avg_pool2d(t, 32).flatten(1)

    a = thumbs(x)
    b = a if ref is None else thumbs(ref)
    dup = np.zeros(len(a), dtype=bool)
    for s in range(0, len(a), 512):
        d = torch.cdist(a[s:s + 512], b) / 32
        if ref is None:  # on ne compare qu'aux images précédentes
            rows = torch.arange(s, s + len(d))[:, None]
            d[torch.arange(len(b))[None, :] >= rows] = 1.0
        dup[s:s + len(d)] = (d.min(dim=1).values < threshold).numpy()
    return dup


# --------------------------------------------------------------------------- #
# Prétraitement d'une image
# --------------------------------------------------------------------------- #
def crop_brain(gray: np.ndarray, threshold: int = 20, margin: int = 4) -> np.ndarray:
    """Recadre l'image sur la zone non noire (le crâne), pour ne pas gaspiller de pixels en fond."""
    mask = gray > threshold
    if mask.sum() < 0.05 * mask.size:  # image presque noire : on ne touche à rien
        return gray
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    r0, r1 = max(rows[0] - margin, 0), min(rows[-1] + margin + 1, gray.shape[0])
    c0, c1 = max(cols[0] - margin, 0), min(cols[-1] + margin + 1, gray.shape[1])
    return gray[r0:r1, c0:c1]


def prepare_image(image: Image.Image | str | Path, image_size: int) -> np.ndarray:
    """Étapes 1 à 3 : image PIL ou chemin -> tableau uint8 (image_size, image_size)."""
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    gray = np.asarray(image.convert("L"))
    cropped = Image.fromarray(crop_brain(gray))
    # ImageOps.pad : redimensionne sans déformer, puis complète avec du noir (color=0)
    square = ImageOps.pad(cropped, (image_size, image_size), method=Image.BILINEAR, color=0)
    return np.array(square, dtype=np.uint8)


def normalization(mean: float, std: float) -> v2.Transform:
    """Étape 4 : uint8 [0, 255] -> float [0, 1] -> standardisé (moyenne 0, écart-type 1)."""
    return v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize([mean], [std])])


def preprocess_for_inference(image: Image.Image, stats: dict) -> torch.Tensor:
    """Image PIL -> tenseur (1, 1, H, W) prêt pour model(x). À utiliser dans l'appli Streamlit."""
    arr = torch.from_numpy(prepare_image(image, stats["image_size"]))[None]  # (1, H, W)
    return normalization(stats["mean"], stats["std"])(arr)[None]


# --------------------------------------------------------------------------- #
# Cache sur disque
# --------------------------------------------------------------------------- #
@dataclass
class PreparedData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    stats: dict  # image_size, mean, std, class_names, tailles des splits


def _cache_path(image_size: int) -> Path:
    return PROJECT_ROOT / "data" / "processed" / f"mri_{image_size}.npz"


def stats_path() -> Path:
    return PROJECT_ROOT / "data" / "processed" / "stats.json"


def prepare_dataset(
    image_size: int = 224, val_size: float = 0.15, seed: int = 42, force: bool = False
) -> PreparedData:
    """Prépare (ou relit depuis le cache) les jeux train / validation / test.

    - Train + validation viennent de Training/ (doublons retirés, découpage stratifié).
    - Test = Testing/ tel quel : il n'est utilisé qu'à la toute fin (étape 4).
    - mean / std sont calculés sur le train uniquement (pas de fuite d'information).
    """
    cache = _cache_path(image_size)
    if cache.exists() and stats_path().exists() and not force:
        d = np.load(cache, allow_pickle=False)
        stats = json.loads(stats_path().read_text(encoding="utf-8"))
        same = stats.get("image_size") == image_size and stats.get("seed") == seed
        if same and "n_near_test_removed" in stats:  # sinon : ancien cache, on refait
            return PreparedData(*(d[k] for k in ["x_train", "y_train", "x_val", "y_val",
                                                  "x_test", "y_test"]), stats=stats)

    root = load_brain_tumor_data()
    train_paths, train_labels = list_images(root, "Training")
    n_before = len(train_paths)
    train_paths, train_labels = remove_duplicates(train_paths, train_labels)
    test_paths, test_labels = list_images(root, "Testing")

    def load(paths):
        return np.stack([prepare_image(p, image_size) for p in paths])

    x_all, x_te = load(train_paths), load(test_paths)
    # quasi-doublons : on retire de Training les copies d'images du test (fuite vers le test)
    # puis les copies internes (fuite train -> validation). Le test officiel reste intact.
    in_test = near_duplicates(x_all, x_te)
    keep = ~in_test
    keep[keep] = ~near_duplicates(x_all[keep])
    x_all, train_labels = x_all[keep], train_labels[keep]
    train_paths = [p for p, k in zip(train_paths, keep) if k]

    idx = np.arange(len(train_paths))
    i_tr, i_va = train_test_split(idx, test_size=val_size, stratify=train_labels,
                                  random_state=seed)
    tr_p, va_p = [train_paths[i] for i in i_tr], [train_paths[i] for i in i_va]
    x_tr, y_tr, x_va, y_va = x_all[i_tr], train_labels[i_tr], x_all[i_va], train_labels[i_va]
    scaled = x_tr.astype(np.float32) / 255.0
    stats = {
        "image_size": image_size,
        "channels": 1,
        "mean": float(scaled.mean()),
        "std": float(scaled.std()),
        "class_names": CLASS_NAMES,
        "seed": seed,
        "val_size": val_size,
        "n_train_raw": n_before,
        "n_duplicates_removed": n_before - len(keep),
        "n_near_test_removed": int(in_test.sum()),
        "n_near_duplicates_removed": int((~in_test).sum() - keep.sum()),
        "n_train": len(tr_p),
        "n_val": len(va_p),
        "n_test": len(test_paths),
        "test_paths": [str(p.relative_to(root)) for p in test_paths],
    }

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, x_train=x_tr, y_train=y_tr, x_val=x_va, y_val=y_va,
             x_test=x_te, y_test=test_labels)
    stats_path().write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return PreparedData(x_tr, y_tr, x_va, y_va, x_te, test_labels, stats)


# --------------------------------------------------------------------------- #
# Dataset PyTorch + DataLoaders
# --------------------------------------------------------------------------- #
def train_augmentation() -> v2.Transform:
    """Augmentations légères, réalistes pour des IRM, appliquées au train uniquement.

    - flip horizontal : le cerveau est à peu près symétrique gauche/droite et une tumeur
      peut être d'un côté comme de l'autre (pas de flip vertical : non réaliste)
    - petite rotation / translation / zoom : variations de positionnement du patient
    - léger changement de luminosité / contraste : variations entre machines IRM
    """
    return v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        v2.ColorJitter(brightness=0.1, contrast=0.1),
    ])


class MRIDataset(Dataset):
    """Renvoie (image (1, H, W) normalisée, label), avec augmentation optionnelle."""

    def __init__(self, x: np.ndarray, y: np.ndarray, mean: float, std: float,
                 augment: bool = False):
        self.x = torch.from_numpy(x).unsqueeze(1)  # (N, H, W) -> (N, 1, H, W), uint8
        self.y = torch.from_numpy(y)
        steps = [train_augmentation()] if augment else []
        self.transform = v2.Compose(steps + [normalization(mean, std)])

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return self.transform(self.x[i]), self.y[i]


def get_dataloaders(data: PreparedData, batch_size: int = 32, augment: bool = True,
                    num_workers: int = 0, seed: int = 42):
    """Renvoie (train_loader, val_loader, test_loader)."""
    m, s = data.stats["mean"], data.stats["std"]
    g = torch.Generator().manual_seed(seed)
    opts = dict(batch_size=batch_size, num_workers=num_workers,
                pin_memory=torch.cuda.is_available())
    train = DataLoader(MRIDataset(data.x_train, data.y_train, m, s, augment=augment),
                       shuffle=True, generator=g, **opts)
    val = DataLoader(MRIDataset(data.x_val, data.y_val, m, s), **opts)
    test = DataLoader(MRIDataset(data.x_test, data.y_test, m, s), **opts)
    return train, val, test
