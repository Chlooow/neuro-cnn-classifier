"""Application Streamlit : exploration, performances du modèle et diagnostic IRM.
"""

import json

import streamlit as st

# common.py en premier : il ajoute src/ au chemin d'import (cnn, load_data, grad_cam)
from common import CHECKPOINT, LOGO_PATH, REPORTS, load_model
from onglet_diagnostic import render as onglet_diagnostic
from onglet_exploration import render as onglet_exploration
from onglet_modele import render as onglet_modele

# Config de la page
st.set_page_config(
    page_title="CerebroAI - Diagnostic IRM",
    layout="wide"
)


@st.cache_resource
def charger_modele():
    """Renvoie (modèle, checkpoint). Le checkpoint contient la config, les stats et les métriques."""
    return load_model(CHECKPOINT)


def lire_json(nom):
    return json.loads((REPORTS / nom).read_text(encoding="utf-8"))


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

with tab_explore:
    onglet_exploration()

with tab_model:
    onglet_modele(model, stats, config, metrics, history)

with tab_predict:
    onglet_diagnostic(model, stats, config, metrics)


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
