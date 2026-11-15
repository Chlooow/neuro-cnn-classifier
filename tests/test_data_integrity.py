"""Vérifications automatiques sur les données préparées (nécessite le cache data/processed/)."""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from load_data import (  # noqa: E402
    CLASS_NAMES,
    _cache_path,
    get_dataloaders,
    load_brain_tumor_data,
    near_duplicates,
    prepare_dataset,
    preprocess_for_inference,
)

pytestmark = pytest.mark.skipif(not _cache_path(224).exists(),
                                reason="lancer d'abord main.py prepare")


@pytest.fixture(scope="module")
def data():
    return prepare_dataset()


def hashes(x):
    return {hashlib.md5(img.tobytes()).hexdigest() for img in x}


def test_aucune_image_commune_entre_les_jeux(data):
    tr, va, te = hashes(data.x_train), hashes(data.x_val), hashes(data.x_test)
    assert not (tr & va) and not (tr & te) and not (va & te)


def test_aucune_quasi_copie_du_test(data):
    assert not near_duplicates(data.x_train, data.x_test).any()
    assert not near_duplicates(data.x_val, data.x_test).any()
    assert not near_duplicates(data.x_val, data.x_train).any()


def test_toutes_les_classes_presentes_et_split_stratifie(data):
    p_tr = np.bincount(data.y_train, minlength=4) / len(data.y_train)
    p_va = np.bincount(data.y_val, minlength=4) / len(data.y_val)
    assert (p_tr > 0).all() and np.abs(p_tr - p_va).max() < 0.02
    assert data.stats["class_names"] == CLASS_NAMES


def test_batch_au_bon_format(data):
    x, y = next(iter(get_dataloaders(data, batch_size=8)[0]))
    assert x.shape == (8, 1, 224, 224) and x.dtype == torch.float32
    assert torch.isfinite(x).all() and y.max() < 4


def test_preprocessing_appli_identique_au_test(data):
    root = load_brain_tumor_data()
    test_loader = get_dataloaders(data, batch_size=8)[2]
    for k in [0, 500, 1500]:
        x_app = preprocess_for_inference(Image.open(root / data.stats["test_paths"][k]), data.stats)
        assert torch.allclose(x_app[0], test_loader.dataset[k][0])
