"""Éléments communs à l'application : chemins, constantes et accès au dataset."""

import json
import sys
from pathlib import Path

import streamlit as st

# on remonte jusqu'à la racine du projet (pyproject.toml) pour rendre src/ importable
ROOT = Path(__file__).resolve().parent
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.append(str(ROOT / "src"))

from cnn import load_model  # noqa: E402
from load_data import CLASS_NAMES, load_brain_tumor_data  # noqa: E402

IMAGES_PAR_PAGE = 12
EXTENSIONS = {".jpg", ".jpeg", ".png"}

CHECKPOINT = ROOT / "models" / "best_model.pt"
REPORTS = ROOT / "reports"
LOGO_PATH = Path(__file__).parent / "assets" / "logo-cerebroai.png"
NEURO_PATH = Path(__file__).parent / "assets" / "neurologie_idf.json"

__all__ = ["load_model", "CLASS_NAMES", "IMAGES_PAR_PAGE", "CHECKPOINT", "REPORTS", "LOGO_PATH", "ROOT"]


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
    except Exception as exc:  # pas de réseau, pas de cache, identifiants Kaggle manquants au cas ou pour la presentation
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


def charger_neurologie():
    """Renvoie (note, liste d'établissements) depuis assets/neurologie_idf.json (modifiable à la main)."""
    try:
        data = json.loads(NEURO_PATH.read_text(encoding="utf-8"))
        return data.get("note", ""), data.get("etablissements", [])
    except (OSError, ValueError):
        return "", []
