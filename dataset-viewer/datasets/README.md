# Dataset `training`

Le fichier `training.csv` est le manifeste du jeu de données. Ajoute une ligne par image :

- `image_path` : chemin de l'image, relatif à ce dossier (par exemple `images/sprite-001.png`)
- `label` : classe ou catégorie à apprendre
- `split` : `train`, `validation` ou `test`
- `notes` : commentaire facultatif

Place les images dans un sous-dossier `images/`, puis ouvre `training.csv` dans le Dataset Viewer pour parcourir les annotations. Le manifeste ne contient pas encore d'exemples, car aucune image ni étiquette n'a été fournie.
