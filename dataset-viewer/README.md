# Dataset Viewer

Un petit visualiseur autonome pour parcourir des fichiers CSV, TSV et JSON dans le navigateur.

## Utilisation

Ouvre `index.html` dans un navigateur, puis choisis un fichier ou dépose-le dans la zone d'import. Le tableau permet de rechercher dans toutes les colonnes et de parcourir les résultats par pages de 100 lignes.

Les données sont lues localement par le navigateur et ne sont envoyées à aucun serveur. Pour les fichiers JSON, le visualiseur accepte un tableau d'objets, un objet contenant un tableau `data` ou `rows`, ou un objet unique.
