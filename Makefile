# Makefile CerebroAI : une commande par tâche (nécessite make et uv)
# Exemple : make app

.PHONY: help install test lint format prepare optimize train evaluate app pipeline clean

UV ?= uv run
TRIALS ?= 15
CONFIG ?= best

help:  ## Affiche les commandes disponibles
	@grep -E "^[a-z-]+:.*##" $(MAKEFILE_LIST) | sed -E "s/:.*## /\t/"

install:  ## Installe les dépendances (avec les outils de dev)
	uv sync --group dev

test:  ## Lance les tests
	$(UV) pytest

lint:  ## Vérifie le style du code (ruff)
	$(UV) ruff check .

format:  ## Corrige automatiquement le style du code
	$(UV) ruff check --fix .

prepare:  ## Télécharge et prétraite le dataset (remplit aussi le cache kagglehub)
	$(UV) python src/main.py prepare

optimize:  ## Recherche d'hyperparamètres Optuna (TRIALS=15 par défaut)
	$(UV) python src/main.py optimize --trials $(TRIALS)

train:  ## Entraîne le modèle (CONFIG=best par défaut) et sauvegarde models/best_model.pt
	$(UV) python src/main.py train --config $(CONFIG)

evaluate:  ## Évalue models/best_model.pt sur le jeu de test
	$(UV) python src/main.py evaluate

app:  ## Lance l'application Streamlit
	$(UV) streamlit run src/streamlit/app.py

pipeline: prepare train evaluate  ## Prépare, entraîne puis évalue

clean:  ## Supprime les caches Python et pytest
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \) -prune -exec rm -rf {} +