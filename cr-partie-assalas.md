---
title: Untitled

---

# Compte-rendu partie Assalas

## EDA

Suite à ma première relecture, j’ai retravaillé l’EDA avec l’aide d’une IA et je l’ai comparée à une checklist classique pour des IRM. Voici où on en est.

Ce qu’on a ajouté à l’EDA : le croisement mode, taille, ratio et poids de fichier avec la classe, les stats d’intensité par classe (écart-type, percentiles, % de fond noir), l’histogramme moyen, les images moyennes et d’écart-type, et une PCA sur les pixels. Le résultat principal : notumor se démarque des trois autres classes sur presque tout. 98 % sont en RGB, 1 % seulement en 512×512, 68 % ne sont pas carrées, et elles sont plus claires et plus contrastées. La PCA suit le mode du fichier. Les images moyennes suggèrent aussi un plan de coupe différent (à confirmer à l’œil). Le modèle risque donc de reconnaître notumor grâce à la source de l’image plutôt qu’à la tumeur. À garder en tête pour l’évaluation : regarder les métriques par classe, pas seulement l’accuracy.

Ton EDA me semblait peut-être un peut trop simple,

je pense pour montrer qu'on a bien observer nos images il ne faut pas hesiter à aller dans le detail

Ce qui manque encore dans l’EDA :


| Élément | Priorité | Mon avis |
|---|---|---|
| Qualité des images | Indispensable | Il faut vérifier que les images sont exploitables. |
| Vérification des anomalies détectées | Indispensable | Une anomalie statistique doit être expliquée ou au moins investiguée. |
| Comparaison Training / Testing | Très recommandé | Permet de vérifier que les deux ensembles sont comparables. |
| Détection des quasi-doublons | Très recommandé | Particulièrement importante pour éviter une fuite de données. |
| Section sur les limites du dataset | Très recommandé | Indispensable pour présenter honnêtement les limites méthodologiques du projet. |
| Inspection visuelle des annotations et images atypiques | Indispensable | Il faut s'assurer que le modèle n'apprend pas des annotations ou artefacts. |
| Vérification R = G = B | Très recommandé | Permet de confirmer que les images RGB sont réellement monochromes. |


un point à update : 
- le tableau de bilan de la section 8, à mettre à jour avec les nouveaux constats.

## preproc 
Pour le prétraitement, le coeur est bien : niveaux de gris, recadrage, padding, normalisation calculée sur le train, augmentation sur le train uniquement, forme des tenseurs. Il manque peut-être : 


| Élément | Priorité | Mon avis |
|---|---|---|
| Nettoyage des données | Indispensable | Compléter le nettoyage avec la détection des fichiers illisibles, des images vides ou presque noires, la vérification des labels et la détection des quasi-doublons. |
| Validation du recadrage | Indispensable | Vérifier le recadrage sur l'ensemble du dataset, mesurer la surface conservée, identifier les cas extrêmes et s'assurer qu'aucun tissu cérébral n'est coupé. Comparer le seuil fixe de 20 avec un seuil adaptatif comme Otsu. |
| Risque de biais lié au padding | Indispensable | Les images non carrées sont surreprésentées dans la classe No Tumor. Mesurer la proportion d'images avec des bandes noires après padding pour chaque classe et évaluer le risque de raccourci appris par le CNN. |
| Équilibre des classes après nettoyage | Très recommandé | Présenter les nouvelles distributions après suppression des doublons. Les classes ne sont plus parfaitement équilibrées. Évaluer la nécessité d'utiliser des class weights. |
| Justification du rehaussement des images | Très recommandé | Justifier l'absence de débruitage, de CLAHE et d'égalisation d'histogramme, ou comparer les performances avec et sans ces transformations. |
| Normalisation des intensités | Très recommandé | Mentionner l'influence du fond noir sur la moyenne et l'écart-type. Comparer éventuellement les statistiques avec et sans fond noir et examiner les distributions normalisées par classe. |
| Validation de la Data Augmentation | Très recommandé | Tester des variations plus importantes de luminosité et de contraste, ainsi qu'un gamma aléatoire. Vérifier les distributions d'intensités après augmentation et s'assurer que les transformations ne coupent pas les tumeurs. |
| Contrôles d'intégrité automatisés | Indispensable | Ajouter des tests automatiques pour vérifier l'absence de chemins et de hashes communs entre les ensembles, l'absence de NaN et d'infinis, la cohérence des proportions, la reproductibilité du split et la compatibilité entre le preprocessing d'inférence et celui du test. |
| Limites du dataset et du preprocessing | Indispensable | Signaler l'absence d'identifiants patients, qui empêche une séparation par patient. Mentionner également les doublons présents dans le Testing officiel (27 en Glioma et 4 en Meningioma), conservés pour respecter le jeu de test d'origine. |


## Verdict
C'est bien ! ta bien faiit de faire le guide ca va nous aider à resumer quand on aura tout terminé pour l'oral

### vigilance : 
IA ca se voit trop que c'est une IA qui a fait le travail du coup je ne sais pas comment reagit le prof donc n'hesite pas à conclure avec tes mots ou bien demande à Claude d'ecrire une conclusion simple sans mise en page comme un etudiant


