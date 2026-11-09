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

Si le dépôt est déjà téléchargé, l'application ouvre sa copie locale par défaut. Sinon, saisis un jeton qui a accès au dépôt et clique sur **Télécharger localement**. L'application récupère les fichiers du Hub, indexe les images PNG avec leur légende TXT, puis affiche le dataset depuis le disque.

Le jeton est transmis au téléchargeur local en mémoire et n'est écrit ni dans le dépôt ni dans le stockage du navigateur. Le contenu privé sous `datasets/pixelart-dataset/` est ignoré par Git et ne sera pas poussé sur GitHub. Python 3, Git et Git LFS doivent être disponibles pour le téléchargement.

Quand la copie est chargée, le panneau **Répartition globale du dataset** présente d’un côté les tags structurés des annotations disponibles et de l’autre les caractéristiques explicites tirées des légendes de toutes les images locales, indépendamment de la recherche et de la page affichée. La source HF contient actuellement un sous-ensemble d’images annotées; le viewer affiche sa couverture exacte. Les catégories de légendes incluent la forme, la vue, la pose/action, le fond, les dimensions, le type d’image et le dossier/jeu. Le champ de rôle de jeu est exclu; « character » s’affiche comme « humanoid » pour la forme du sujet. Les graphiques sont recalculés quand les fichiers locaux changent.

## Fichiers locaux

Tu peux aussi choisir ou déposer un fichier CSV, TSV ou JSON. Le tableau permet de rechercher dans toutes les colonnes et de parcourir les résultats par pages de 100. Les données locales sont lues par le navigateur et ne sont envoyées à aucun serveur.

Pour les fichiers JSON, le visualiseur accepte un tableau d'objets, un objet contenant un tableau `data` ou `rows`, ou un objet unique.

Le dossier `datasets/` contient le manifeste de départ `training.csv` pour annoter des images.
