"""Point d'entrée du projet.

    python src/main.py prepare                 # télécharge et prétraite le dataset
    python src/main.py optimize --trials 15    # cherche les hyperparamètres (Optuna)
    python src/main.py train --config best     # entraîne avec configs/best.yaml
    python src/main.py evaluate                # évalue models/best_model.pt sur le test
"""

import argparse
import json

from load_data import get_dataloaders, prepare_dataset
from utils import PROJECT_ROOT, get_device, load_config, set_seed

REPORTS = PROJECT_ROOT / "reports"
FIG = REPORTS / "figures"


def plot_history(history, path):
    import matplotlib
    matplotlib.use("Agg")  # pas de fenêtre, on enregistre juste l'image
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key in zip(axes, ["loss", "accuracy"]):
        ax.plot(epochs, history[key], label="train")
        ax.plot(epochs, history[f"val_{key}"], label="validation")
        ax.axvline(history["best_epoch"], color="grey", ls="--", label="meilleure epoch")
        ax.set(title=key.capitalize(), xlabel="epoch")
        ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion(metrics, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    names = metrics["class_names"]
    plt.figure(figsize=(6, 5))
    sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names)
    plt.xlabel("classe prédite")
    plt.ylabel("vraie classe")
    plt.title(f"Matrice de confusion (test) - accuracy {metrics['accuracy']:.1%}")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def print_report(metrics):
    print(f"accuracy {metrics['accuracy']:.4f} | loss {metrics['loss']:.4f}")
    for name in metrics["class_names"]:
        r = metrics["classification_report"][name]
        print(f"  {name:11s} précision {r['precision']:.3f}  rappel {r['recall']:.3f}  "
              f"f1 {r['f1-score']:.3f}")


def main():
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
        print(f"train={s['n_train']}  val={s['n_val']}  test={s['n_test']}")
        print(f"retirés de Training : {s['n_duplicates_removed']} doublons exacts, "
              f"{s['n_near_test_removed']} copies d'images du test, "
              f"{s['n_near_duplicates_removed']} quasi-doublons")

    elif args.command == "train":
        from cnn import count_parameters, evaluate, save_model, train_model

        device = get_device()
        model, history = train_model(config, data, device)
        _, val_loader, test_loader = get_dataloaders(data, batch_size=64)
        metrics = {
            "config": args.config,
            "model": {**config["model"], "parameters": count_parameters(model)},
            "best_epoch": history["best_epoch"],
            "validation": evaluate(model, val_loader, device),
            "test": evaluate(model, test_loader, device),  # le test ne sert qu'ici, à la fin
        }
        save_model(model, config, s, metrics)
        (REPORTS / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        plot_history(history, FIG / "training_curves.png")
        plot_confusion(metrics["test"], FIG / "confusion_matrix.png")
        print(f"paramètres : {metrics['model']['parameters']:,} | "
              f"meilleure epoch : {history['best_epoch']}")
        print(f"validation : accuracy {metrics['validation']['accuracy']:.4f}")
        print("test :", end=" ")
        print_report(metrics["test"])

    elif args.command == "evaluate":
        from cnn import evaluate, load_model

        device = get_device()
        model, _ = load_model(device=device)  # on repart du fichier sauvegardé
        test_loader = get_dataloaders(data, batch_size=64)[2]
        print_report(evaluate(model, test_loader, device))

    elif args.command == "optimize":
        from optimize import run_study

        run_study(config, data, n_trials=args.trials)


if __name__ == "__main__":
    main()
