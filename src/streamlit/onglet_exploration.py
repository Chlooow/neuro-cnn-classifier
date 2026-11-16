"""Onglet 1 : exploration des images du dataset."""

from common import (
    CLASS_NAMES,
    IMAGES_PAR_PAGE,
    avertir_dataset_absent,
    dossier_dataset,
    lister_images,
)
from PIL import Image

import streamlit as st


def render():
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
