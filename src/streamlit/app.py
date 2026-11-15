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

IMAGES_PAR_PAGE = 12
EXTENSIONS = {".jpg", ".jpeg", ".png"}

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


@st.cache_resource(show_spinner="Chargement du dataset (kagglehub)...")
def _telecharger_dataset():
    return load_brain_tumor_data()


def dossier_dataset():
    """Renvoie (dossier, erreur). Ne plante jamais : sans internet ni cache kagglehub, on renvoie
    (None, message) et seuls les onglets qui ont besoin du dataset sont limités."""
    if st.session_state.get("dataset_erreur"):
        return None, st.session_state["dataset_erreur"]
    try:
        return _telecharger_dataset(), None
    except Exception as exc:  # pas de réseau, pas de cache, identifiants Kaggle manquants...
        st.session_state["dataset_erreur"] = f"{type(exc).__name__} : {exc}"
        return None, st.session_state["dataset_erreur"]


@st.cache_data(show_spinner=False)
def lister_images(dossier: str):
    """Fichiers images d'un dossier, triés (ordre stable d'un rerun à l'autre)."""
    return sorted(p for p in Path(dossier).iterdir() if p.suffix.lower() in EXTENSIONS)


def avertir_dataset_absent(erreur, cle):
    st.warning(
        "Le dataset n'est pas disponible (pas de connexion et rien dans le cache `kagglehub`). "
        "Les fonctions qui en dépendent sont désactivées ; le reste de l'application fonctionne. "
        "Pour remplir le cache, lancez l'application une fois avec internet."
    )
    with st.expander("Détail de l'erreur"):
        st.code(erreur)
    if st.button("Réessayer le chargement", key=f"retry_{cle}"):
        st.session_state.pop("dataset_erreur", None)
        st.rerun()


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
    st.write("Parcourez les images du dataset (jeu d'entraînement ou de test), classe par classe.")

    racine, erreur_dataset = dossier_dataset()
    if racine is None:
        avertir_dataset_absent(erreur_dataset, "explorer")
    else:
        col_split, col_cls, col_page = st.columns([1, 1, 1])
        split = col_split.radio("Jeu d'images", ["Testing", "Training"], horizontal=True,
                                format_func=lambda x: {"Testing": "Test", "Training": "Entraînement"}[x])
        selected_class = col_cls.selectbox("Classe :", CLASS_NAMES, format_func=str.capitalize)

        data_dir = racine / split / selected_class
        if not data_dir.exists():
            st.info(f"Le dossier `{data_dir}` est introuvable.")
        else:
            image_files = lister_images(str(data_dir))
            if not image_files:
                st.warning(f"Aucune image n'a été trouvée dans le dossier `{data_dir}`.")
            else:
                n_pages = (len(image_files) + IMAGES_PAR_PAGE - 1) // IMAGES_PAR_PAGE
                # une clé par (jeu, classe) : changer de classe ramène à la page 1
                page = col_page.number_input(f"Page (sur {n_pages})", min_value=1, max_value=n_pages,
                                             value=1, step=1, key=f"page_{split}_{selected_class}")
                debut = (page - 1) * IMAGES_PAR_PAGE
                fin = min(debut + IMAGES_PAR_PAGE, len(image_files))
                st.caption(f"Images {debut + 1} à {fin} sur {len(image_files)} "
                           f"({selected_class.capitalize()}, jeu {'de test' if split == 'Testing' else 'd’entraînement'})")

                cols = st.columns(4)
                for idx, img_path in enumerate(image_files[debut:fin]):
                    with cols[idx % 4]:
                        st.image(Image.open(img_path), caption=f"{selected_class.capitalize()} #{debut + idx + 1}",
                                 width='stretch')
                        if st.button(f"Agrandir #{debut + idx + 1}", key=f"btn_{split}_{selected_class}_{debut + idx}"):
                            st.session_state["selected_image"] = img_path
                            st.session_state["selected_class_name"] = selected_class
                            st.session_state["selected_split"] = split

        # Zone d'agrandissement sous la grille
        if st.session_state.get("selected_image"):
            st.divider()
            st.subheader(f"Vue agrandie - {st.session_state['selected_class_name'].capitalize()}")

            col_det1, col_det2 = st.columns([2, 3])
            with col_det1:
                st.image(st.session_state["selected_image"], width=380)
            with col_det2:
                st.write(f"**Nom du fichier :** `{st.session_state['selected_image'].name}`")
                st.write(f"**Jeu :** {st.session_state.get('selected_split', '')}")
                st.write(f"**Classe :** {st.session_state['selected_class_name'].capitalize()}")
                st.info("Ce zoom permet un contrôle visuel détaillé des coupes coronales et axiales.")
                if st.button("Fermer la vue agrandie"):
                    st.session_state["selected_image"] = None
                    st.rerun()

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

    # --- choix de l'image : jeu de test (classe réelle connue) ou fichier importé
    image, vraie_classe, nom_image = None, None, None
    racine, erreur_dataset = dossier_dataset()
    sources = ["Importer un fichier"]
    if racine is not None:
        sources = ["Image du jeu de test", "Importer un fichier"]
    source = st.radio("Source de l'image", sources, horizontal=True)

    if source == "Image du jeu de test":
        c_cls, c_img = st.columns([1, 3])
        vraie_classe = c_cls.selectbox("Classe réelle :", CLASS_NAMES, format_func=str.capitalize, key="pred_cls")
        fichiers = lister_images(str(racine / "Testing" / vraie_classe))
        choix = c_img.selectbox(f"Image ({len(fichiers)} disponibles) :", fichiers,
                                format_func=lambda p: p.name, key=f"pred_img_{vraie_classe}")
        if choix is not None:
            image = Image.open(choix).convert("RGB")
            nom_image = choix.name
    else:
        if racine is None:
            avertir_dataset_absent(erreur_dataset, "predire")
        uploaded_file = st.file_uploader("Importer une coupe IRM au format JPG ou PNG", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            nom_image = uploaded_file.name

    if image is not None:
        # Grad-CAM : par défaut la classe prédite, mais on peut expliquer n'importe quelle classe
        options_cam = ["Classe prédite (automatique)"] + CLASS_NAMES
        choix_cam = st.selectbox("Classe expliquée par Grad-CAM :", options_cam,
                                 format_func=lambda x: x if x in options_cam[:1] else x.capitalize())
        cam_idx = None if choix_cam == options_cam[0] else CLASS_NAMES.index(choix_cam)

        col_img, col_cam, col_res = st.columns(3)

        # Sauvegarde temporaire pour traitement & PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_orig:
            image.save(tmp_orig.name)
            orig_path = tmp_orig.name

        # prédiction et Grad-CAM, avec le même prétraitement qu'à l'entraînement
        res = explain_image(model, image, stats, class_idx=cam_idx)
        pred_class = res["pred_class"]
        explained_class = res["explained_class"]
        probs = res["probabilities"]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_heat:
            res["overlay"].save(tmp_heat.name)
            heat_path = tmp_heat.name

        with col_img:
            st.image(image, caption=f"IRM chargée ({nom_image})", width='stretch')

        with col_cam:
            st.image(res["overlay"], caption=f"Grad-CAM - classe expliquée : {explained_class.capitalize()}",
                     width='stretch')
            st.caption(
                f"Rouge = zones les plus influentes pour la classe **{explained_class}** · "
                "Bleu = zones peu influentes. La carte explique une seule classe à la fois."
            )

        with col_res:
            st.subheader("Résultats d'Inférence")

            for cls_name, prob in probs.items():
                st.write(f"**{cls_name.capitalize()}**")
                st.progress(prob, text=f"{prob * 100:.1f}%")

            st.success(f"Diagnostic principal : **{pred_class.capitalize()}** ({probs[pred_class]*100:.1f}%)")
            if vraie_classe is not None:
                if vraie_classe == pred_class:
                    st.info(f"Classe réelle : **{vraie_classe.capitalize()}** prédiction correcte")
                else:
                    st.error(f"Classe réelle : **{vraie_classe.capitalize()}** prédiction incorrecte")

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
                heatmap_class=explained_class,
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