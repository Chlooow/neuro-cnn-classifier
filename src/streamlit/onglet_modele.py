"""Onglet 2 : paramètres du modèle, métriques et courbes d'apprentissage."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from common import CLASS_NAMES, REPORTS

import streamlit as st


def render(model, stats, config, metrics, history):
    st.header("Synthèse des Performances du CNN")
    st.subheader("Architecture et hyperparamètres")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Paramètres", f"{metrics['model']['parameters']:,}".replace(",", " "))
    c2.metric("Filtres de base", config["model"]["filters"])
    c3.metric("Neurones Dense", config["model"]["dense"])
    c4.metric("Dropout", config["model"]["dropout"])

    t = config["training"]
    f = config["model"]["filters"]
    st.write(
        f"4 blocs Conv 3×3 + BatchNorm + ReLU + MaxPool ({f}, {2 * f}, {4 * f} puis {8 * f} filtres), "
        "puis moyenne globale (GlobalAveragePooling) et 2 couches Dense avec Dropout. "
        f"Learning rate {t['learning_rate']:.2e}, batch {t['batch_size']}, weight decay {t['weight_decay']}, "
        f"{t['epochs']} epochs maximum (patience {t['patience']}), meilleure epoch : {metrics['best_epoch']}."
    )
    with st.expander("Voir les couches du modèle (architecture complète)"):
        size = stats["image_size"]
        st.caption(f"Entrée : image {size}×{size}, 1 canal (niveaux de gris) · Sortie : 4 scores (une par classe)")
        st.code(str(model), language="text")

    st.subheader("Métriques")
    jeu = st.radio("Jeu d'évaluation", ["test", "validation"], horizontal=True)
    m = metrics[jeu]
    col_acc, col_loss = st.columns(2)
    col_acc.metric("Accuracy", f"{m['accuracy']:.2%}")
    col_loss.metric("Loss", f"{m['loss']:.3f}")

    col_rep, col_conf = st.columns(2)
    with col_rep:
        st.write("Rapport de classification")
        lignes = CLASS_NAMES + ["macro avg", "weighted avg"]
        rapport = pd.DataFrame({nom: m["classification_report"][nom] for nom in lignes}).T
        rapport = rapport[["precision", "recall", "f1-score", "support"]].round(3)
        rapport["support"] = rapport["support"].astype(int)
        st.dataframe(rapport)
    with col_conf:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(m["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set(xlabel="classe prédite", ylabel="vraie classe", title=f"Matrice de confusion ({jeu})")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Courbes d'apprentissage")
    st.caption(
        "Courbes **train** et **validation** : la validation (15 % de Training) pilote les callbacks "
        "(réduction du learning rate, early stopping, meilleure epoch). Le jeu de **test** reste isolé "
        "et n'est évalué qu'une fois à la fin : sa valeur finale est tracée en trait constant "
        "(« test (final) ») pour la comparer aux courbes."
    )
    epochs = list(range(1, len(history["loss"]) + 1))
    test_m = metrics["test"]
    col_loss_curve, col_acc_curve = st.columns(2)
    with col_loss_curve:
        st.caption("Loss")
        st.line_chart(pd.DataFrame({"train": history["loss"], "validation": history["val_loss"],
                                    "test (final)": [test_m["loss"]] * len(epochs)}, index=epochs))
    with col_acc_curve:
        st.caption("Accuracy")
        st.line_chart(pd.DataFrame({"train": history["accuracy"], "validation": history["val_accuracy"],
                                    "test (final)": [test_m["accuracy"]] * len(epochs)}, index=epochs))

    with st.expander("Optimisation des hyperparamètres (Optuna)"):
        for nom in ["optuna_history.png", "optuna_importance.png"]:
            figure = REPORTS / "figures" / nom
            if figure.exists():
                st.image(str(figure))
