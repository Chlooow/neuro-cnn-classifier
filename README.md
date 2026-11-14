# Classification de tumeurs cérébrales par IRM

Projet de classification multi-classes d'images IRM en quatre catégories : `glioma`, `meningioma`, `pituitary` et `notumor`.

## Démarrage rapide

Prérequis : Python 3.11 ou 3.12 et [uv](https://docs.astral.sh/uv/).

```powershell
uv sync --group dev
uv run pytest
uv run streamlit run src/streamlit/app.py
```

Le guide [docs/uv-guide.md](docs/uv-guide.md) explique les commandes et le fonctionnement de `uv` pour le binôme.

## Données

Télécharger le dataset « Brain Tumor MRI Dataset » depuis [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), puis placer les répertoires `Training/` et `Testing/` dans `data/raw/`. Ne pas ajouter les images au dépôt Git. La préparation (redimensionnement, normalisation et séparation) sera documentée dans le code et les notebooks.

## Structure

```text
.
|-- configs/                 # configurations d'expériences
|-- data/
|   |-- raw/                  # données Kaggle originales (non versionnées)
|   |-- interim/              # données intermédiaires
|   |-- processed/            # données prêtes pour le modèle
|   `-- external/             # éventuelles données externes
|-- docs/                     # documentation, dont le guide uv
|-- models/                   # checkpoints (non versionnés)
|-- notebooks/                # exploration et analyses exploratoires
|-- reports/figures/          # courbes, matrices de confusion, Grad-CAM
|-- src/
|   |-- load_data.py          # téléchargement et prétraitement des images
|   |-- cnn.py                # architecture CNN, entraînement, évaluation
|   |-- optimize.py           # optimisation des hyperparamètres (Optuna)
|   |-- grad_cam.py           # Grad-CAM (bonus)
|   |-- report_pdf.py         # génération du rapport PDF
|   |-- utils.py              # fonctions utilitaires
|   |-- main.py               # point d'entrée en ligne de commande
|   `-- streamlit/app.py      # interface Streamlit
|-- tests/
|-- pyproject.toml            # projet, dépendances et outils
`-- uv.lock                  # versions exactes résolues par uv
```

## Travail d'équipe

- Garder les chemins et paramètres d'expérience dans `configs/`, pas dispersés dans les notebooks.
- Utiliser les notebooks pour explorer; mettre le code réutilisable dans `src/`.
- Enregistrer les figures dans `reports/figures/` et les modèles dans `models/`.
- Avant de partager un changement : `uv run ruff check .` et `uv run pytest`.
- Les données, environnements, checkpoints et rapports générés sont ignorés par Git.

## Sujet et critères

Le projet suit les étapes du sujet : préparation du dataset, CNN d'au moins quatre convolutions, entraînement avec callbacks, évaluation (accuracy, loss, rapport de classification et matrice de confusion), optimisation des hyperparamètres, sauvegarde/rechargement, interface Streamlit et rapport PDF. Grad-CAM est prévu comme bonus.
# CerebroAI

<img width="2000" height="2000" alt="logo-cerebroai" src="https://github.com/user-attachments/assets/111ff1f4-c2d7-43dc-b577-646c1c257a1d" />

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

lien des slides : https://canva.link/qz8yfu72lgudejd
