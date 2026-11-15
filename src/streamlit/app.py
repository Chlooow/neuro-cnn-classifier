"""Application Streamlit : exploration, performances du modèle et diagnostic IRM."""

import json
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image

# on remonte jusqu'à la racine du projet (pyproject.toml) pour rendre src/ importable
ROOT = Path(__file__).resolve().parent
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.append(str(ROOT / "src"))

from cnn import load_model
from gen_pdf import generate_medical_pdf
from grad_cam import explain_image
from load_data import CLASS_NAMES, load_brain_tumor_data

CHECKPOINT = ROOT / "models" / "best_model.pt"
REPORTS = ROOT / "reports"

# Configuration générale de la page
st.set_page_config(
    page_title="CerebroAI - Diagnostic IRM",
    page_icon="🧠",
    layout="wide"
)


@st.cache_resource
def charger_modele():
    """Renvoie (modèle, checkpoint). Le checkpoint contient la config, les stats et les métriques."""
    return load_model(CHECKPOINT)


def lire_json(nom):
    return json.loads((REPORTS / nom).read_text(encoding="utf-8"))


@st.cache_resource
def dossier_dataset():
    return load_brain_tumor_data()


# Chemin du logo CerebroAI
LOGO_PATH = Path(__file__).parent / "assets" / "logo-cerebroai.png"


# En-tête principal de la page
col_logo, col_title = st.columns([1, 5])
with col_logo:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=100)
with col_title:
    st.title("CerebroAI")
    st.write("Système de classification et d'explicabilité par analyse d'imagerie IRM cérébrale.")

st.divider()

if not CHECKPOINT.exists():
    st.error(f"Modèle introuvable : `{CHECKPOINT}`")
    st.stop()

model, checkpoint = charger_modele()
stats = checkpoint["stats"]
config = checkpoint["config"]
metrics = lire_json("metrics.json")
history = lire_json("history.json")

# Structure en onglets
tab_explore, tab_model, tab_predict = st.tabs([
    "Galerie d'Exploration",
    "Modèle & Performances",
    "Diagnostic Temps Réel"
])

# ------------------------------------------------------------------------------
# ONGLET 1 : EXPLORATION DES DONNÉES
# ------------------------------------------------------------------------------
with tab_explore:
    st.subheader("Exploration des coupes IRM par pathologie")
    st.write("Parcourez les séries d'images issues du dataset de test pour évaluer les pathologies.")

    classes = CLASS_NAMES
    selected_class = st.selectbox(
        "Sélectionner la classe à observer :",
        classes,
        format_func=lambda x: x.capitalize()
    )

    # Chemin vers le dossier de test
    data_dir = dossier_dataset() / "Testing" / selected_class

    if data_dir.exists():
        image_files = list(data_dir.glob("*.jpg"))[:8]

        if image_files:
            cols = st.columns(4)
            for idx, img_path in enumerate(image_files):
                col = cols[idx % 4]
                with col:
                    img = Image.open(img_path)
                    st.image(img, caption=f"{selected_class.capitalize()} #{idx+1}", width='stretch')
                    if st.button(f"Agrandir #{idx+1}", key=f"btn_{selected_class}_{idx}"):
                        st.session_state["selected_image"] = img_path
                        st.session_state["selected_class_name"] = selected_class

            # Fenêtre/Zone d'agrandissement sous la grille
            if st.session_state.get("selected_image"):
                st.divider()
                st.subheader(f"Vue agrandie - {st.session_state['selected_class_name'].capitalize()}")

                col_det1, col_det2 = st.columns([2, 3])
                with col_det1:
                    st.image(st.session_state["selected_image"], width=380)
                with col_det2:
                    st.write(f"**Nom du fichier :** `{st.session_state['selected_image'].name}`")
                    st.write(f"**Classe diagnostiquée :** {st.session_state['selected_class_name'].capitalize()}")
                    st.info("Ce zoom permet un contrôle visuel détaillé des coupes coronales et axiales.")
                    if st.button("Fermer la vue agrandie"):
                        st.session_state["selected_image"] = None
                        st.rerun()
        else:
            st.warning(f"Aucune image `.jpg` n'a été trouvée dans le dossier `{data_dir}`.")
    else:
        st.info(f"Le dossier `{data_dir}` est introuvable.")

# ------------------------------------------------------------------------------
# ONGLET 2 : PERFORMANCES DU MODÈLE
# ------------------------------------------------------------------------------
with tab_model:
    st.header("Synthèse des Performances du CNN")
    st.subheader("Architecture et hyperparamètres")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Paramètres", f"{metrics['model']['parameters']:,}".replace(",", " "))
    c2.metric("Filtres de base", config["model"]["filters"])
    c3.metric("Neurones Dense", config["model"]["dense"])
    c4.metric("Dropout", config["model"]["dropout"])

    t = config["training"]
    st.write(
        "4 blocs Conv 3×3 + BatchNorm + ReLU + MaxPool, puis moyenne globale et 2 couches Dense. "
        f"Learning rate {t['learning_rate']:.2e}, batch {t['batch_size']}, weight decay {t['weight_decay']}, "
        f"{t['epochs']} epochs maximum (patience {t['patience']}), meilleure epoch : {metrics['best_epoch']}."
    )

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
    epochs = list(range(1, len(history["loss"]) + 1))
    col_loss_curve, col_acc_curve = st.columns(2)
    with col_loss_curve:
        st.caption("Loss")
        st.line_chart(pd.DataFrame({"train": history["loss"], "validation": history["val_loss"]}, index=epochs))
    with col_acc_curve:
        st.caption("Accuracy")
        st.line_chart(pd.DataFrame({"train": history["accuracy"], "validation": history["val_accuracy"]}, index=epochs))

    with st.expander("Optimisation des hyperparamètres (Optuna)"):
        for nom in ["optuna_history.png", "optuna_importance.png"]:
            figure = REPORTS / "figures" / nom
            if figure.exists():
                st.image(str(figure))

# ------------------------------------------------------------------------------
# ONGLET 3 : DIAGNOSTIC TEMPS RÉEL (Prédiction + Grad-CAM + Export PDF)
# ------------------------------------------------------------------------------
with tab_predict:
    st.header("Analyse d'IRM en Temps Réel")

    # Avertissement de responsabilité médicale
    st.warning(
        "⚠️ **Avertissement :** "
        "Cette application est un outil expérimental issu d'un projet académique d'intelligence artificielle. "
        "Elle ne constitue en aucun cas un dispositif médical ni un outil d'aide au diagnostic validé. "
        "Les prédictions générées ne remplacent sous aucun prétexte l'avis, l'expertise ou l'évaluation d'un professionnel de santé qualifié."
    )

    uploaded_file = st.file_uploader("Importer une coupe IRM au format JPG ou PNG", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col_img, col_cam, col_res = st.columns(3)

        # Sauvegarde temporaire pour traitement & PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_orig:
            image = Image.open(uploaded_file)
            image.save(tmp_orig.name)
            orig_path = tmp_orig.name

        # prédiction et Grad-CAM, avec le même prétraitement qu'à l'entraînement
        res = explain_image(model, image, stats)
        pred_class = res["pred_class"]
        probs = res["probabilities"]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_heat:
            res["overlay"].save(tmp_heat.name)
            heat_path = tmp_heat.name

        with col_img:
            st.image(image, caption="IRM Patient chargée", width='stretch')

        with col_cam:
            st.image(res["overlay"], caption="Grad-CAM", width='stretch')

        with col_res:
            st.subheader("Résultats d'Inférence")

            for cls_name, prob in probs.items():
                st.write(f"**{cls_name.capitalize()}**")
                st.progress(prob, text=f"{prob * 100:.1f}%")

            st.success(f"Diagnostic principal : **{pred_class.capitalize()}** ({probs[pred_class]*100:.1f}%)")

        st.divider()

        # ----------------------------------------------------------------------
        # SECTION EXPORT PDF PERSONNALISABLE
        # ----------------------------------------------------------------------
        st.subheader("Configuration & Génération du Rapport PDF")

        with st.form("pdf_configuration_form"):
            patient_id = st.text_input("Identifiant / Code Patient :", value="PAT-2026-8894")

            st.markdown("**Éléments à inclure dans le bilan PDF :**")
            cb_col1, cb_col2 = st.columns(2)

            with cb_col1:
                inc_patient_info = st.checkbox("Informations de la session & Identifiant Patient", value=True)
                inc_probs = st.checkbox("Distribution complète des probabilités (4 classes)", value=True)
                inc_metrics = st.checkbox("Métriques et paramètres du modèle", value=True)

            with cb_col2:
                # Option spécifique demandée
                inc_heatmap = st.checkbox("Ajouter la heatmap dans le rapport PDF destiné au praticien", value=True)
                inc_disclaimer = st.checkbox("Avertissement de non-responsabilité médicale", value=True)

            notes_medecin = st.text_area("Remarques / Observations du praticien (optionnel) :", placeholder="Saisir ici les commentaires cliniques...")

            btn_generate = st.form_submit_button("⛨ Générer le Rapport PDF")

        if btn_generate:
            include_options = {
                "patient_info": inc_patient_info,
                "probabilities": inc_probs,
                "heatmap": inc_heatmap,
                "metrics": inc_metrics,
                "disclaimer": inc_disclaimer
            }

            # Envoi au générateur PDF
            pdf_bytes = generate_medical_pdf(
                patient_id=patient_id,
                pred_class=pred_class,
                probabilities=probs,
                original_img_path=orig_path,
                heatmap_img_path=heat_path if inc_heatmap else None,
                notes=notes_medecin,
                include_options=include_options,
                model_info={"config": config, "metrics": metrics["test"],
                            "n_parameters": metrics["model"]["parameters"]}
            )

            st.success("Rapport PDF généré avec succès !")
            st.download_button(
                label="Télécharger le Rapport PDF",
                data=pdf_bytes,
                file_name=f"rapport_CerebroAI_{patient_id}.pdf",
                mime="application/pdf"
            )


# ------------------------------------------------------------------------------
# FOOTER PERMANENT
# ------------------------------------------------------------------------------
footer_html = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f8f9fa;
    color: #6c757d;
    text-align: center;
    padding: 8px 0;
    font-size: 13px;
    border-top: 1px solid #e9ecef;
    z-index: 999;
}
.footer p {
    margin: 0;
}
</style>
<div class="footer">
    <p>🧠 <b>CerebroAI</b> - Projet Académique M2 Big Data, Chloé MAKOUNDOU et Assalas ARAB | À usage d'aide à la décision uniquement</p>
</div>
"""

st.markdown(footer_html, unsafe_allow_html=True)