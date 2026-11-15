"""Petites fonctions communes : config, graine aléatoire, GPU."""

import random
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(name: str = "baseline") -> dict:
    with open(PROJECT_ROOT / "configs" / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    # même graine partout = mêmes résultats à chaque lancement
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
