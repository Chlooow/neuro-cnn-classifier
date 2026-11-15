"""Étape 5 : recherche des hyperparamètres avec Optuna (lr, batch, filtres, dropout)."""

import copy

import optuna
import yaml

from cnn import train_model
from utils import PROJECT_ROOT, get_device

SEARCH_EPOCHS = 15  # entraînements courts : on compare juste des réglages
SEARCH_PATIENCE = 5


def suggest_config(trial, base):
    """Un jeu d'hyperparamètres à essayer."""
    config = copy.deepcopy(base)
    # log=True : on explore autant entre 0.0001 et 0.001 qu'entre 0.001 et 0.01
    config["training"]["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    config["training"]["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64])
    config["model"]["filters"] = trial.suggest_categorical("filters", [16, 32, 48])
    config["model"]["dropout"] = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
    config["training"]["epochs"] = SEARCH_EPOCHS
    config["training"]["patience"] = SEARCH_PATIENCE
    return config


def run_study(base, data, n_trials=15):
    device = get_device()

    def objective(trial):
        config = suggest_config(trial, base)
        _, history = train_model(config, data, device, verbose=False, trial=trial)
        best = history["best_epoch"] - 1
        trial.set_user_attr("val_accuracy", history["val_accuracy"][best])
        print(f"essai {trial.number:2d} | {trial.params} | val_loss {history['val_loss'][best]:.4f}"
              f" | val_acc {history['val_accuracy'][best]:.3f}", flush=True)
        return history["val_loss"][best]  # on juge sur la validation, jamais sur le test

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=base["project"]["seed"]),
        # le pruner coupe les essais qui partent mal (moins bons que la médiane)
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=5),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)

    reports = PROJECT_ROOT / "reports"
    study.trials_dataframe().to_csv(reports / "optuna_trials.csv", index=False)
    save_figures(study, reports / "figures")

    # on écrit la config gagnante dans configs/best.yaml
    best = copy.deepcopy(base)
    p = study.best_params
    best["training"].update(learning_rate=p["learning_rate"], batch_size=p["batch_size"])
    best["model"].update(filters=p["filters"], dropout=p["dropout"])
    path = PROJECT_ROOT / "configs" / "best.yaml"
    path.write_text("# généré par src/optimize.py (meilleur essai Optuna)\n"
                    + yaml.safe_dump(best, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"meilleur essai : {p} | val_loss {study.best_value:.4f} -> {path}")
    return best


def save_figures(study, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from optuna.visualization import matplotlib as ovm

    for name, plot in [("optuna_history", ovm.plot_optimization_history),
                       ("optuna_importance", ovm.plot_param_importances)]:
        try:
            plot(study)
            plt.tight_layout()
            plt.savefig(fig_dir / f"{name}.png", dpi=150)
            plt.close("all")
        except Exception as e:  # ex. pas assez d'essais terminés
            print(f"figure {name} non générée : {e}")
