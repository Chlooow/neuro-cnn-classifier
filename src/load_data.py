"""Étape 1 : chargement et prétraitement des IRM."""

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

# ordre alphabétique = ordre des dossiers = ordre des sorties du modèle
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


def load_brain_tumor_data() -> Path:
    import kagglehub

    return Path(kagglehub.dataset_download(KAGGLE_DATASET))  # téléchargé une seule fois


def list_images(root: Path, split: str):
    """Chemins et labels d'un split ('Training' ou 'Testing')."""
    folder = ImageFolder(root / split)  # la classe = le nom du sous-dossier
    assert folder.classes == CLASS_NAMES
    return [Path(p) for p, _ in folder.samples], np.array(folder.targets)


def remove_duplicates(paths, labels):
    """Enlève les fichiers strictement identiques (même empreinte MD5)."""
    seen, keep = set(), []
    for i, p in enumerate(paths):
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h not in seen:  # on garde la 1re occurrence
            seen.add(h)
            keep.append(i)
    return [paths[i] for i in keep], labels[keep]


def near_duplicates(x, ref=None, threshold=0.02):
    """True pour les images de x qui ont une quasi-copie dans ref.

    Sans ref : on cherche les quasi-copies à l'intérieur de x (on garde la 1re).
    """
    def thumbs(a):  # miniatures 32x32 aplaties, pixels entre 0 et 1
        t = torch.from_numpy(a).float().unsqueeze(1) / 255
        return torch.nn.functional.adaptive_avg_pool2d(t, 32).flatten(1)

    a = thumbs(x)
    b = a if ref is None else thumbs(ref)
    dup = np.zeros(len(a), dtype=bool)
    for s in range(0, len(a), 512):  # par paquets de 512 pour ne pas saturer la mémoire
        d = torch.cdist(a[s:s + 512], b) / 32  # écart moyen entre pixels (0 = identiques)
        if ref is None:
            rows = torch.arange(s, s + len(d))[:, None]
            d[torch.arange(len(b))[None, :] >= rows] = 1.0  # on ne compare qu'aux précédentes
        dup[s:s + len(d)] = (d.min(dim=1).values < threshold).numpy()  # 0.02 : vérifié à l'œil
    return dup


def crop_brain(gray, threshold=20, margin=4):
    """Recadre sur la zone non noire (le crâne)."""
    mask = gray > threshold
    if mask.sum() < 0.05 * mask.size:  # image presque noire : on ne touche à rien
        return gray
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    r0, r1 = max(rows[0] - margin, 0), min(rows[-1] + margin + 1, gray.shape[0])
    c0, c1 = max(cols[0] - margin, 0), min(cols[-1] + margin + 1, gray.shape[1])
    return gray[r0:r1, c0:c1]


def prepare_image(image, image_size):
    """Image PIL ou chemin -> tableau uint8 (image_size, image_size) en niveaux de gris."""
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    gray = np.asarray(image.convert("L"))  # 1 canal
    cropped = Image.fromarray(crop_brain(gray))
    # pad = redimensionne sans déformer puis complète avec du noir
    square = ImageOps.pad(cropped, (image_size, image_size), method=Image.BILINEAR, color=0)
    return np.array(square, dtype=np.uint8)


def normalize(x):
    """uint8 [0, 255] -> float, standardisé avec la moyenne et l'écart-type de l'image.

    Par image et pas sur tout le train : ça efface les écarts de luminosité et de
    contraste entre les sources d'images (sinon le modèle s'en sert pour tricher).
    """
    x = x.float() / 255
    return (x - x.mean()) / (x.std() + 1e-6)  # + 1e-6 pour ne jamais diviser par 0


def preprocess_for_inference(image, stats):
    """Image PIL -> tenseur (1, 1, H, W), même traitement qu'à l'entraînement (pour l'appli)."""
    arr = torch.from_numpy(prepare_image(image, stats["image_size"]))
    return normalize(arr)[None, None]


@dataclass
class PreparedData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    stats: dict  # taille d'image, effectifs, chemins des images de val et de test


def _cache_path(image_size):
    return PROJECT_ROOT / "data" / "processed" / f"mri_{image_size}.npz"


def stats_path():
    return PROJECT_ROOT / "data" / "processed" / "stats.json"


def prepare_dataset(image_size=224, val_size=0.15, seed=42, force=False):
    """Prépare train / validation / test (ou les relit depuis le cache)."""
    cache = _cache_path(image_size)
    if cache.exists() and stats_path().exists() and not force:
        stats = json.loads(stats_path().read_text(encoding="utf-8"))
        ok = stats.get("image_size") == image_size and stats.get("seed") == seed
        if ok and stats.get("version") == 3:  # sinon c'est un ancien cache, on refait
            d = np.load(cache)
            keys = ["x_train", "y_train", "x_val", "y_val", "x_test", "y_test"]
            return PreparedData(*(d[k] for k in keys), stats=stats)

    root = load_brain_tumor_data()
    train_paths, y = list_images(root, "Training")
    test_paths, y_test = list_images(root, "Testing")
    n_raw = len(train_paths)
    train_paths, y = remove_duplicates(train_paths, y)
    n_md5 = n_raw - len(train_paths)

    def load(paths):
        return np.stack([prepare_image(p, image_size) for p in paths])

    x, x_test = load(train_paths), load(test_paths)

    # on enlève de Training les copies d'images du test, puis les copies internes
    # (le test officiel reste tel quel)
    in_test = near_duplicates(x, x_test)
    keep = ~in_test
    keep[keep] = ~near_duplicates(x[keep])
    x, y = x[keep], y[keep]
    train_paths = [p for p, k in zip(train_paths, keep) if k]

    # stratify : même proportion de chaque classe en train et en validation
    i_tr, i_va = train_test_split(np.arange(len(y)), test_size=val_size, stratify=y,
                                  random_state=seed)
    stats = {
        "version": 3,
        "image_size": image_size,
        "channels": 1,
        "class_names": CLASS_NAMES,
        "seed": seed,
        "val_size": val_size,
        "n_train_raw": n_raw,
        "n_duplicates_removed": n_md5,
        "n_near_test_removed": int(in_test.sum()),
        "n_near_duplicates_removed": int((~in_test).sum() - keep.sum()),
        "n_train": len(i_tr),
        "n_val": len(i_va),
        "n_test": len(test_paths),
        "val_paths": [str(train_paths[i].relative_to(root)) for i in i_va],
        "test_paths": [str(p.relative_to(root)) for p in test_paths],
    }
    data = PreparedData(x[i_tr], y[i_tr], x[i_va], y[i_va], x_test, y_test, stats)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, x_train=data.x_train, y_train=data.y_train, x_val=data.x_val,
             y_val=data.y_val, x_test=x_test, y_test=y_test)
    stats_path().write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return data


def train_augmentation():
    """Petites variations réalistes pour une IRM, tirées au hasard à chaque epoch."""
    return v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),  # pas de flip vertical : une IRM à l'envers n'existe pas
        v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        v2.ColorJitter(brightness=0.1, contrast=0.1),
    ])


class MRIDataset(Dataset):
    def __init__(self, x, y, augment=False):
        self.x = torch.from_numpy(x).unsqueeze(1)  # (N, H, W) -> (N, 1, H, W)
        self.y = torch.from_numpy(y)
        self.augment = train_augmentation() if augment else None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = self.x[i]
        if self.augment is not None:  # seulement pour le train
            img = self.augment(img)
        return normalize(img), self.y[i]


def get_dataloaders(data, batch_size=32, augment=True, num_workers=0, seed=42):
    """Renvoie (train_loader, val_loader, test_loader)."""
    g = torch.Generator().manual_seed(seed)  # mélange reproductible
    opts = dict(batch_size=batch_size, num_workers=num_workers,
                pin_memory=torch.cuda.is_available())
    train = DataLoader(MRIDataset(data.x_train, data.y_train, augment=augment),
                       shuffle=True, generator=g, **opts)
    val = DataLoader(MRIDataset(data.x_val, data.y_val), **opts)
    test = DataLoader(MRIDataset(data.x_test, data.y_test), **opts)
    return train, val, test
