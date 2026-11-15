"""Point d'entrée en ligne de commande.

    python src/main.py prepare                 # télécharge + prétraite le dataset (cache)
    python src/main.py train                   # entraîne avec configs/baseline.yaml
    python src/main.py train --config best     # entraîne avec la config trouvée par Optuna
    python src/main.py evaluate                # évalue models/best_model.pt sur le test
    python src/main.py optimize --trials 15    # recherche d'hyperparamètres (Optuna)
"""

import argparse
import json

from load_data import prepare_dataset
from utils import PROJECT_ROOT, get_device, load_config, set_seed

REPORTS = PROJECT_ROOT / "reports"
FIG = REPORTS / "figures"


def plot_history(history: dict, path) -> None:
    """Courbes loss et accuracy (train / validation) au fil des epochs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in [(axes[0], "loss", "Loss"), (axes[1], "accuracy", "Accuracy")]:
        ax.plot(epochs, history[key], label="train")
        ax.plot(epochs, history[f"val_{key}"], label="validation")
        ax.axvline(history["best_epoch"], color="grey", ls="--", label="meilleure epoch")
        ax.set(title=title, xlabel="epoch")
        ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion(metrics: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(6, 5))
    names = metrics["class_names"]
    sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names)
    plt.xlabel("classe prédite")
    plt.ylabel("vraie classe")
    plt.title(f"Matrice de confusion (test) - accuracy {metrics['accuracy']:.1%}")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "train", "evaluate", "optimize"])
    parser.add_argument("--config", default="baseline")
    parser.add_argument("--force", action="store_true", help="refaire le cache")
    parser.add_argument("--trials", type=int, default=15, help="nombre d'essais Optuna")
    args = parser.parse_args()
    config = load_config(args.config)
    cfg = config["project"]
    set_seed(cfg["seed"])
    data = prepare_dataset(cfg["image_size"], cfg["val_size"], cfg["seed"], force=args.force)
    s = data.stats
    FIG.mkdir(parents=True, exist_ok=True)

    if args.command == "prepare":
        print(f"train={s['n_train']}  val={s['n_val']}  test={s['n_test']}  "
              f"(retirés : {s['n_duplicates_removed']} doublons, "
              f"{s['n_near_test_removed']} copies d'images du test, "
              f"{s['n_near_duplicates_removed']} quasi-doublons)")
        print(f"mean={s['mean']:.4f}  std={s['std']:.4f}  taille={s['image_size']}px")

    elif args.command == "train":
        from cnn import (
            count_parameters,
            evaluate,
            get_dataloaders,
            loader_options,
            save_model,
            train_model,
        )

        device = get_device()
        model, history = train_model(config, data, device)
        print(f"paramètres : {count_parameters(model):,} | "
              f"meilleure epoch : {history['best_epoch']}")
        opts = {"per_image": loader_options(config)["per_image"]}
        _, val_loader, test_loader = get_dataloaders(data, batch_size=64, **opts)
        metrics = {"config": args.config, "best_epoch": history["best_epoch"],
                   "validation": evaluate(model, val_loader, device),
                   "test": evaluate(model, test_loader, device)}
        save_model(model, config, s, metrics=metrics)
        (REPORTS / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        plot_history(history, FIG / "training_curves.png")
        plot_confusion(metrics["test"], FIG / "confusion_matrix.png")
        print(f"val acc {metrics['validation']['accuracy']:.3f} | "
              f"test acc {metrics['test']['accuracy']:.3f}")

    elif args.command == "evaluate":
        from cnn import evaluate, get_dataloaders, load_model

        device = get_device()
        model, ckpt = load_model(device=device)
        test_loader = get_dataloaders(data, batch_size=64,
                                      per_image=ckpt["stats"].get("per_image_norm", False))[2]
        m = evaluate(model, test_loader, device)
        print(f"test accuracy {m['accuracy']:.4f} | loss {m['loss']:.4f}")
        for name in m["class_names"]:
            r = m["classification_report"][name]
            print(f"  {name:11s} precision {r['precision']:.3f}  recall {r['recall']:.3f}  "
                  f"f1 {r['f1-score']:.3f}")

    elif args.command == "optimize":
        from optimize import run_study

        run_study(config, data, n_trials=args.trials)


if __name__ == "__main__":
    main()
