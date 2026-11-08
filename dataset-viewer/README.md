# Dataset Viewer · Nuxt + Vue

Visualiseur Nuxt pour le dataset Hugging Face `ethanzxv/pixelart-dataset`, avec prise en charge des fichiers CSV, TSV et JSON.

## Lancer l'application

Depuis ce dossier :

```sh
npm install
npm run dev
```

Puis ouvre l'adresse locale affichée par Nuxt.

## Dataset Hugging Face

Saisis un jeton Hugging Face qui a accès en lecture au dataset, puis clique sur **Connecter**. Choisis une configuration et un split ; le tableau charge les lignes par pages de 100 et permet la recherche dans les colonnes textuelles.

Le jeton reste en mémoire pendant la session. Il n'est pas écrit dans le dépôt ni conservé dans le stockage du navigateur. Les requêtes passent par le serveur Nuxt vers l'API Dataset Viewer de Hugging Face.

## Fichiers locaux

Tu peux aussi choisir ou déposer un fichier CSV, TSV ou JSON. Le tableau permet de rechercher dans toutes les colonnes et de parcourir les résultats par pages de 100. Les données locales sont lues par le navigateur et ne sont envoyées à aucun serveur.

Pour les fichiers JSON, le visualiseur accepte un tableau d'objets, un objet contenant un tableau `data` ou `rows`, ou un objet unique.

Le dossier `datasets/` contient le manifeste de départ `training.csv` pour annoter des images.
