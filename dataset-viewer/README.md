# Dataset Viewer

Un visualiseur dans le navigateur pour le dataset Hugging Face `ethanzxv/pixelart-dataset` et des fichiers CSV, TSV ou JSON.

## Dataset Hugging Face

Ouvre `index.html`, saisis un jeton Hugging Face autorisé à lire le dataset, puis clique sur **Connecter**. Choisis une configuration et un split ; le tableau charge 100 lignes à la fois et permet la recherche dans les colonnes textuelles.

Le jeton est utilisé en mémoire pendant la session de l'onglet. Il n'est pas écrit dans le dépôt ni conservé dans le stockage du navigateur. Les requêtes authentifiées sont envoyées directement à l'API Dataset Viewer de Hugging Face.

## Fichiers locaux

Tu peux aussi choisir un fichier ou le déposer dans la zone d'import. Le tableau permet de rechercher dans toutes les colonnes et de parcourir les résultats par pages de 100 lignes. Les données locales sont lues par le navigateur et ne sont envoyées à aucun serveur.

Pour les fichiers JSON, le visualiseur accepte un tableau d'objets, un objet contenant un tableau `data` ou `rows`, ou un objet unique.

Le dossier `datasets/` contient le manifeste de départ `training.csv` pour annoter des images.
