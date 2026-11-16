"""Onglet 3 : diagnostic sur une image (prédiction, Grad-CAM et rapport PDF)."""

import tempfile

from common import (
    CLASS_NAMES,
    avertir_dataset_absent,
    charger_neurologie,
    dossier_dataset,
    lister_images,
)
from gen_pdf import generate_medical_pdf
from PIL import Image

import streamlit as st
from grad_cam import explain_image


def render(model, stats, config, metrics):
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

        note_neuro, etablissements = charger_neurologie()
        if etablissements:
            with st.expander("Services de neurologie de référence en Île-de-France"):
                st.caption(note_neuro)
                for e in etablissements:
                    lieu = ", ".join(x for x in (e.get("adresse"), e.get("ville")) if x)
                    resp = f" · {e['responsable']}" if e.get("responsable") else ""
                    st.markdown(f"- **{e['etablissement']}**, {lieu}{resp} "
                                f"([source]({e['source']}))")
        sites = ["Non précisé"] + [e["etablissement"] for e in etablissements] + ["Autre"]

        with st.form("pdf_configuration_form"):
            patient_id = st.text_input("Identifiant / Code Patient :", value="PAT-2026-8894")

            col_doc, col_site = st.columns(2)
            nom_medecin = col_doc.text_input("Médecin générant le rapport (nom) :",
                                             placeholder="Dr Prénom NOM")
            site_choisi = col_site.selectbox("Établissement / service :", sites)
            # dans un formulaire, un champ conditionnel n'apparaîtrait qu'après l'envoi : il est donc toujours affiché
            site_libre = col_site.text_input("Autre établissement (utilisé si « Autre » est choisi) :")

            st.markdown("**Éléments à inclure dans le bilan PDF :**")
            cb_col1, cb_col2 = st.columns(2)

            with cb_col1:
                inc_patient_info = st.checkbox("Informations de la session & Identifiant Patient", value=True)
                inc_probs = st.checkbox("Distribution complète des probabilités (4 classes)", value=True)
                inc_metrics = st.checkbox("Métriques et paramètres du modèle", value=True)

            with cb_col2:
                inc_heatmap = st.checkbox("Ajouter la heatmap dans le rapport PDF destiné au praticien", value=True)
                inc_disclaimer = st.checkbox("Avertissement de non-responsabilité médicale", value=True)
                inc_doctor = st.checkbox("Nom du médecin, établissement et zone de signature", value=True)

            notes_medecin = st.text_area("Remarques / Observations du praticien (optionnel) :", placeholder="Saisir ici les commentaires cliniques...")

            btn_generate = st.form_submit_button("⛨ Générer le Rapport PDF")

        if btn_generate:
            include_options = {
                "patient_info": inc_patient_info,
                "probabilities": inc_probs,
                "heatmap": inc_heatmap,
                "metrics": inc_metrics,
                "disclaimer": inc_disclaimer,
                "doctor": inc_doctor
            }
            site_final = site_libre if site_choisi == "Autre" else (
                "" if site_choisi == "Non précisé" else site_choisi)

            # Envoi au générateur PDF
            pdf_bytes = generate_medical_pdf(
                patient_id=patient_id,
                pred_class=pred_class,
                probabilities=probs,
                original_img_path=orig_path,
                heatmap_img_path=heat_path if inc_heatmap else None,
                notes=notes_medecin,
                include_options=include_options,
                doctor_name=nom_medecin.strip(),
                doctor_site=site_final.strip(),
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
