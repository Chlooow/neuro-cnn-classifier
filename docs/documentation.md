# Documentation technique — CerebroAI

Classification d'IRM cérébrales en 4 classes (`glioma`, `meningioma`, `notumor`, `pituitary`) avec un CNN PyTorch, une application Streamlit et un Grad-CAM.

## 1. Pipeline

| Étape | Fichier | Rôle |
|---|---|---|
| Données | `src/load_data.py` | Téléchargement Kaggle (`kagglehub`), doublons, recadrage, redimensionnement 224×224, normalisation |
| Modèle | `src/cnn.py` | CNN, entraînement avec callbacks, évaluation, sauvegarde/rechargement |
| Optimisation | `src/optimize.py` | Recherche Optuna (learning rate, batch, filtres, dropout) |
| Interprétabilité | `src/grad_cam.py` | Grad-CAM sur la dernière convolution |
| Application | `src/streamlit/app.py` et `onglet_*.py` | Interface en 3 onglets, un fichier par onglet |
| Rapport PDF | `src/streamlit/gen_pdf.py` | Rapport ReportLab généré en mémoire |
| CLI | `src/main.py` | `prepare`, `train`, `evaluate`, `optimize` |

### Prétraitement
- Conversion en niveaux de gris (1 canal), recadrage sur la zone non noire, redimensionnement **sans déformation** (padding noir) à 224×224.
- Normalisation par image (moyenne/écart-type de chaque image) pour gommer les écarts de luminosité entre sources.
- Suppression des doublons exacts (MD5) et des quasi-copies (entre Training et Testing, puis internes à Training).
- Validation = 15 % de Training (stratifiée). Le jeu **Testing** officiel n'est utilisé qu'à l'évaluation finale.

### Architecture
4 blocs `Conv 3×3 (padding 1) → BatchNorm → ReLU → MaxPool 2×2`, avec un nombre de filtres doublé à chaque bloc (f, 2f, 4f, 8f), puis `AdaptiveAvgPool2d(1)` et un classifieur `Linear → ReLU → Dropout → Linear(4)`. Meilleure configuration trouvée par Optuna : 48 filtres de base, Dense 128, dropout 0,1, learning rate d'environ 4,7e-4, batch 16 (923 300 paramètres).

### Entraînement
Adam avec weight decay ; callbacks : `ReduceLROnPlateau` (lr ÷ 2 après 3 epochs sans amélioration), early stopping (patience 10) et conservation des poids de la meilleure epoch (selon la `val_loss`).

## 2. Application Streamlit

Lancer : `uv run streamlit run src/streamlit/app.py`.

### Organisation des fichiers
| Fichier | Rôle |
|---|---|
| `app.py` | En-tête, chargement du modèle, création des trois onglets, pied de page |
| `common.py` | Chemins, constantes, accès au dataset protégé par `try/except`, lecture de la liste de neurologie |
| `onglet_exploration.py` | Onglet 1 : galerie |
| `onglet_modele.py` | Onglet 2 : modèle et performances |
| `onglet_diagnostic.py` | Onglet 3 : prédiction, Grad-CAM, formulaire PDF |
| `gen_pdf.py` | Génération du rapport PDF |
| `assets/neurologie_idf.json` | Liste indicative de services de neurologie d'Île-de-France |

Chaque onglet expose une fonction `render(...)` appelée par `app.py`. Ne pas créer de `__init__.py` dans `src/streamlit/`.

### Onglet « Galerie d'Exploration »
- Choix du jeu (**Test** ou **Entraînement**) et de la classe.
- Pagination : 12 images par page, sélecteur de page ; changer de classe revient à la page 1.
- Bouton « Agrandir » sous chaque image → vue agrandie avec nom de fichier, jeu et classe.

### Onglet « Modèle & Performances »
- Paramètres : nombre de paramètres, filtres, Dense, dropout, learning rate, batch, weight decay, epochs.
- Volet repliable « Voir les couches du modèle » : `str(model)` (couches réelles).
- Métriques sur test ou validation : accuracy, loss, rapport de classification, matrice de confusion.
- Courbes loss/accuracy : **train** et **validation** par epoch, plus la valeur finale du **test** en trait constant (le test n'est évalué qu'une fois, après l'entraînement, pour éviter toute fuite vers le choix du modèle).
- Résultats Optuna (historique et importance des hyperparamètres).

### Onglet « Diagnostic Temps Réel »
- Source de l'image : **image du jeu de test** (on choisit la classe réelle puis le fichier ; la classe réelle est comparée à la prédiction) ou **import d'un fichier**.
- Probabilités des 4 classes et classe prédite.
- **Grad-CAM** : sélecteur de classe expliquée (par défaut la classe prédite) et légende « rouge = zones les plus influentes pour la classe expliquée ; bleu = peu influentes ».
- Volet « Services de neurologie de référence en Île-de-France » : liste indicative avec sources, modifiable dans `assets/neurologie_idf.json` (pas une recommandation, ces services n'ont pas participé au projet).
- Génération du rapport PDF (voir ci-dessous) avec, en option, le nom du médecin, son établissement et une zone de signature.

### Robustesse sans internet
Le dataset est chargé via `kagglehub` dans un `try/except`. Sans connexion et sans cache :
- un message explique la situation (avec le détail de l'erreur et un bouton « Réessayer ») ;
- la galerie et le sélecteur d'images de test sont désactivés ;
- l'onglet « Modèle & Performances » et l'import d'un fichier continuent de fonctionner.

**Avant la soutenance !!! :** il faut lancer l'application une fois avec internet pour remplir le cache `kagglehub` (le dataset restera ensuite disponible hors ligne).

## 3. Rapport PDF
Contenu : image testée, heatmap Grad-CAM (avec la classe expliquée et sa légende), probabilités, métriques et paramètres du modèle, remarques du praticien, nom du médecin et établissement (facultatifs, avec zone de signature) et avertissement. Les images conservent leur **ratio d'origine** (boîte maximale 240×240, sans déformation).

## 4. Grad-CAM
1. Hook sur la dernière couche `Conv2d` (cartes 28×28, avant BatchNorm, ReLU et MaxPool).
2. Gradient du score de la classe expliquée par rapport aux cartes d'activation.
3. Poids de chaque carte = moyenne spatiale de son gradient ; somme pondérée, ReLU, redimensionnement à 224×224, normalisation 0–1.
4. Superposition (colormap `jet`, alpha 0,4) sur l'IRM en niveaux de gris.

La carte n'explique qu'une classe à la fois ; `explain_image(..., class_idx=...)` permet de choisir laquelle.

## 5. Points à préciser à l'oral
- Grad-CAM est une aide d'interprétation, pas une preuve médicale ; l'application n'est pas un dispositif médical.

## 6. Dépannage
| Problème | Cause probable | Solution |
|---|---|---|
| « Le dataset n'est pas disponible » | Pas d'internet et cache `kagglehub` vide | Il faut lancer l'app une fois en ligne ; ou utiliser l'import de fichier |
| « Modèle introuvable » | `models/best_model.pt` absent | `uv run python src/main.py train --config best` |
| Cache de données obsolète | Changement de taille d'image ou de seed | `uv run python src/main.py prepare --force` |
