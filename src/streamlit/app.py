"""Streamlit entry point; application views will be added as project features are built."""

import streamlit as st
from PIL import Image
import tempfile
from pathlib import Path
from gen_pdf import generate_medical_pdf

# Configuration générale de la page
st.set_page_config(
    page_title="CerebroAI — Diagnostic IRM",
    page_icon="🧠",
    layout="wide"
)

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
    
    classes = ["glioma", "meningioma", "notumor", "pituitary"]
    selected_class = st.selectbox(
        "Sélectionner la classe à observer :", 
        classes, 
        format_func=lambda x: x.capitalize()
    )
    
    # Chemin vers le dossier de test
    data_dir = Path("data/Testing") / selected_class
    
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
        st.info(f"Placez votre dossier d'images dans `{data_dir.parents[1]}` pour alimenter la galerie.")

# ------------------------------------------------------------------------------
# ONGLET 2 : PERFORMANCES DU MODÈLE
# ------------------------------------------------------------------------------
with tab_model:
    st.header("Synthèse des Performances du CNN")
    st.info("Espace réservé à l'affichage des métriques (Accuracy, Loss, Matrice de confusion) transmises par le Binôme A.")

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
    st.info("Espace réservé à l'importation de l'image IRM patient, à l'affichage des probabilités, du Grad-CAM et à l'export du rapport PDF.")

    uploaded_file = st.file_uploader("Importer une coupe IRM au format JPG ou PNG", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col_img, col_res = st.columns([1, 1])
        
        # Sauvegarde temporaire pour traitement & PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_orig:
            image = Image.open(uploaded_file)
            image.save(tmp_orig.name)
            orig_path = tmp_orig.name

        with col_img:
            st.image(image, caption="IRM Patient chargée", use_container_width=True)

        with col_res:
            st.subheader("Résultats d'Inférence")
            
            # Probabilités factices (remplacées lors du raccordement par les prédictions du réseau)
            mock_probs = {
                "glioma": 0.04,
                "meningioma": 0.06,
                "notumor": 0.02,
                "pituitary": 0.88
            }
            pred_class = max(mock_probs, key=mock_probs.get)

            for cls_name, prob in mock_probs.items():
                st.write(f"**{cls_name.capitalize()}**")
                st.progress(prob, text=f"{prob * 100:.1f}%")

            st.success(f"Diagnostic principal : **{pred_class.capitalize()}** ({mock_probs[pred_class]*100:.1f}%)")

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
                "disclaimer": inc_disclaimer
            }

            # Envoi au générateur PDF (path heatmap mis à None ici en attendant le module Grad-CAM)
            pdf_bytes = generate_medical_pdf(
                patient_id=patient_id,
                pred_class=pred_class,
                probabilities=mock_probs,
                original_img_path=orig_path,
                heatmap_img_path=orig_path if inc_heatmap else None,  # Remplacer par le chemin de la heatmap réelle
                notes=notes_medecin,
                include_options=include_options
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