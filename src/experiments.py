"""Comparaison de variantes contre le raccourci « source de l'image ».

    python src/experiments.py

Critère de choix (validation uniquement) : accuracy globale ET accuracy sur les tumeurs
dont le fichier est en RGB (même format que les images notumor : c'est là que le
raccourci fait des erreurs). Le test est affiché pour information, pas pour choisir.
"""

import copy
import json

import numpy as np
import torch
from PIL import Image

from cnn import MODELS, REPORTS, loader_options, predict, save_model, train_model
from load_data import CLASS_NAMES, get_dataloaders, load_brain_tumor_data, prepare_dataset
from utils import get_device, load_config, set_seed

VARIANTS = {
    "baseline": {},
    "norm_par_image": {"per_image_norm": True},
    "norm_par_image+augm_forte": {"per_image_norm": True, "strong_augment": True},
}


def scores(model, loader, y_paths, root, device):
    y, p = predict(model, loader, device)
    pred = p.argmax(1)
    rgb = np.array([Image.open(root / q).mode == "RGB" for q in y_paths])
    tumor_rgb = rgb & (y != CLASS_NAMES.index("notumor"))
    return {"accuracy": float((pred == y).mean()),
            "acc_tumeurs_rgb": float((pred == y)[tumor_rgb].mean()),
            "n_tumeurs_rgb": int(tumor_rgb.sum())}


def main():
    base = load_config()
    set_seed(base["project"]["seed"])
    data = prepare_dataset()
    root, device = load_brain_tumor_data(), get_device()
    results = {}
    for name, pre in VARIANTS.items():
        config = copy.deepcopy(base)
        config["preprocessing"] = pre
        path = MODELS / f"exp_{name}.pt"
        print(f"=== {name}", flush=True)
        model, history = train_model(config, data, device, verbose=False)
        save_model(model, config, data.stats, path=path)
        per_image = loader_options(config)["per_image"]
        _, val, test = get_dataloaders(data, batch_size=64, per_image=per_image)
        results[name] = {"best_epoch": history["best_epoch"],
                         "val": scores(model, val, data.stats["val_paths"], root, device),
                         "test": scores(model, test, data.stats["test_paths"], root, device)}
        print(json.dumps(results[name]), flush=True)
        torch.cuda.empty_cache()
    (REPORTS / "experiments.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
