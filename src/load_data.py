# sert à charger le jeu de données Brain Tumor MRI depuis Kaggle via kagglehub
import os
import pathlib
import kagglehub

def load_brain_tumor_data() -> pathlib.Path:
    """Télécharge le jeu de données Brain Tumor MRI via kagglehub et retourne son chemin d'accès."""
    dataset_path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
    return pathlib.Path(dataset_path)