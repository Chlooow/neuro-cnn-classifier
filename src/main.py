# le main 

import os
from load_data import load_brain_tumor_data

if __name__ == "__main__":
    data_path = load_brain_tumor_data()
    print(f"Le jeu de données Brain Tumor MRI a été téléchargé et est disponible dans : {data_path}")