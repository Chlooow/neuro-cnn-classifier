# CerebroAI: Classification de tumeurs cérébrales par IRM

<img width="2000" height="2000" alt="logo-cerebroai" src="https://github.com/user-attachments/assets/111ff1f4-c2d7-43dc-b577-646c1c257a1d" />

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

lien des slides : https://canva.link/qz8yfu72lgudejd

---

Projet de classification multi-classes d'images IRM en quatre catégories : `glioma`, `meningioma`, `pituitary` et `notumor`.

## Démarrage rapide

Prérequis : Python 3.11 ou 3.12 et [uv](https://docs.astral.sh/uv/).

```powershell
uv sync --group dev
uv run pytest
uv run streamlit run src/streamlit/app.py
```

Le guide [docs/uv-guide.md](docs/uv-guide.md) explique les commandes et le fonctionnement de `uv` pour le binôme. La documentation technique complète est dans [docs/documentation.md](docs/documentation.md).

## Fonctionnalités de l'application

- **Galerie d'Exploration** : choix du jeu (Test ou Entraînement) et de la classe, pagination, vue agrandie d'une image.
- **Modèle & Performances** : paramètres et hyperparamètres, couches réelles du modèle (volet repliable), métriques (rapport de classification, matrice de confusion), courbes train / validation avec la valeur finale du test, résultats Optuna.
- **Diagnostic Temps Réel** : image du jeu de test (classe réelle affichée) ou fichier importé, probabilités des 4 classes, Grad-CAM avec légende et choix de la classe expliquée.
- **Rapport PDF** : image testée, heatmap Grad-CAM, probabilités, métriques et paramètres du modèle (images à leur ratio d'origine).

## Données

Le dataset « Brain Tumor MRI Dataset » ([Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)) est téléchargé automatiquement par `kagglehub` au premier lancement, puis réutilisé depuis le cache. Ne pas ajouter les images au dépôt Git.

**Avant la soutenance :** lancer l'application une fois avec internet pour remplir le cache. Sans internet et sans cache, l'application ne plante pas : la galerie et le sélecteur d'images de test sont désactivés avec un message explicatif, mais l'onglet « Modèle & Performances » et l'import d'un fichier restent utilisables.

## Commandes du pipeline

```powershell
uv run python src/main.py prepare                 # télécharge et prétraite le dataset
uv run python src/main.py optimize --trials 15    # recherche d'hyperparamètres (Optuna)
uv run python src/main.py train --config best     # entraîne avec configs/best.yaml
uv run python src/main.py evaluate                # évalue models/best_model.pt sur le test
```

## Résultats (modèle optimisé)

| Jeu | Accuracy | Loss |
|---|---|---|
| Validation | 95,97 % | 0,116 |
| Test | 89,75 % | 0,517 |

Configuration retenue : 48 filtres de base, Dense 128, dropout 0,1, learning rate ≈ 4,7e-4, batch 16 (923 300 paramètres).

## Structure

```text
.
|-- configs/                 # configurations d'expériences
|-- data/                     # données (non versionnées)
|-- docs/                     # documentation (documentation.md, guide uv)
|-- models/                   # checkpoints
|-- notebooks/                # EDA, prétraitement, modèle
|-- reports/                  # métriques, historique, figures (courbes, matrice, Grad-CAM)
|-- src/
|   |-- load_data.py          # téléchargement et prétraitement des images
|   |-- cnn.py                # architecture CNN, entraînement, évaluation
|   |-- optimize.py           # optimisation des hyperparamètres (Optuna)
|   |-- grad_cam.py           # Grad-CAM (bonus)
|   |-- utils.py              # fonctions utilitaires
|   |-- main.py               # point d'entrée en ligne de commande
|   `-- streamlit/
|       |-- app.py            # interface Streamlit
|       `-- gen_pdf.py        # génération du rapport PDF
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

Le projet suit les étapes du sujet : préparation du dataset, CNN d'au moins quatre convolutions, entraînement avec callbacks, évaluation (accuracy, loss, rapport de classification et matrice de confusion), optimisation des hyperparamètres, sauvegarde/rechargement, interface Streamlit et rapport PDF. Le bonus Grad-CAM est intégré à l'interface et au rapport PDF.



