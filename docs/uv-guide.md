# Guide uv pour le binôme

## À quoi sert uv ?

`uv` est l'outil qui gère ici les dépendances Python et l'environnement virtuel du projet. Il lit `pyproject.toml`, résout des versions compatibles, puis les inscrit dans `uv.lock`. Ce lockfile permet aux deux membres du binôme d'installer les mêmes versions. Le dossier `.venv/` contient l'environnement local; il n'est pas partagé dans Git.

## Installation

Installer uv une seule fois sur son ordinateur en suivant la [documentation officielle](https://docs.astral.sh/uv/getting-started/installation/). Vérifier l'installation dans PowerShell :

```powershell
uv --version
```

Python 3.11 ou 3.12 doit être disponible. uv peut aussi gérer Python; par exemple `uv python install 3.11` installe cette version si nécessaire.

## Premier démarrage après avoir récupéré le dépôt

Depuis la racine du projet :

```powershell
uv sync --group dev
```

Cette commande crée `.venv/`, installe les dépendances du projet et celles de développement, puis respecte `uv.lock`. Il n'est pas nécessaire d'activer manuellement l'environnement : préfixer les commandes par `uv run`.

```powershell
uv run python --version
uv run pytest
uv run ruff check .
uv run streamlit run src/app.py
```

## Ajouter ou modifier une dépendance

Pour une bibliothèque utilisée par l'application ou les notebooks :

```powershell
uv add nom-du-paquet
```

Pour un outil seulement utile au développement :

```powershell
uv add --group dev nom-du-paquet
```

uv met à jour `pyproject.toml` et `uv.lock`. Il faut versionner ces deux fichiers ensemble. Un binôme qui récupère ensuite ces changements relance `uv sync --group dev`.

Pour retirer une bibliothèque : `uv remove nom-du-paquet` (ou ajouter `--group dev` si elle est dans le groupe de développement).

## Bonnes habitudes à deux

- Ne pas versionner `.venv/`, le dataset Kaggle ni les poids du modèle.
- Ne pas éditer `uv.lock` à la main : laisser uv le régénérer.
- Après une mise à jour de `pyproject.toml` ou `uv.lock`, exécuter `uv sync --group dev`.
- Si l'installation est incohérente, `uv sync --reinstall` force la réinstallation; à garder pour le dépannage.
- Pour vérifier la résolution sans modifier l'environnement : `uv lock --check`.

## Commandes utiles

| Commande | Utilité |
| --- | --- |
| `uv sync --group dev` | Synchroniser l'environnement du projet |
| `uv run <commande>` | Lancer une commande dans l'environnement |
| `uv add <paquet>` | Ajouter une dépendance de production |
| `uv add --group dev <paquet>` | Ajouter une dépendance de développement |
| `uv remove <paquet>` | Supprimer une dépendance |
| `uv lock` | Résoudre/mettre à jour le lockfile |
| `uv lock --check` | Vérifier que le lockfile est à jour |