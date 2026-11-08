# Dataset `training`

Le fichier `training.csv` est un manifeste séparé pour tes propres annotations. Ajoute une ligne par image :

- `image_path` : chemin de l'image, relatif à ce dossier (par exemple `images/sprite-001.png`)
- `label` : classe ou catégorie à apprendre
- `split` : `train`, `validation` ou `test`
- `notes` : commentaire facultatif

Place les images dans un sous-dossier `images/`, puis ouvre `training.csv` dans le Dataset Viewer pour parcourir les annotations.

La copie privée Hugging Face est téléchargée séparément dans `pixelart-dataset/` par l'application. Ce dossier est ignoré par Git afin que le contenu du dataset ne soit pas poussé dans le dépôt du projet.
