# Dataset Lab

- analysis/: `analyse_dataset.ipynb` pour les annotations Qwen, `analyse_hex_colors.ipynb` pour les couleurs et leurs combinaisons, et `data_structure.ipynb` pour la taxonomie hiérarchique.
- taxonomy/: taxonomies visuelles, palettes et annotations Qwen/Strata.
- sync_dataset_lab.ipynb: prévisualiser puis synchroniser les annotations vers Hugging Face; les assets bruts sont sélectionnés explicitement et les uploads sont append-only.

Le notebook de sync démarre en mode prévisualisation. Vérifie les comptes et chemins avant d’activer la publication.

## Palettes exactes du dataset

Les couleurs sont calculées directement depuis les PNG, sans être estimées par le modèle à partir de la caption. Depuis la racine du dépôt, mesure les images annotées par Qwen3.8 :

```sh
python notebook/dataset/analysis/enrich_qwen_exact_palettes.py
```

Pour mesurer toutes les images du dataset, y compris celles qui n’ont pas encore d’annotation Qwen, exécute :

```sh
python notebook/dataset/analysis/enrich_qwen_exact_palettes.py --all-dataset-assets
```

Le second mode lit `metadata.jsonl` et écrit un index compact dans `taxonomy_work/dataset_color_palettes.jsonl.gz`. Il utilise le checkout local quand il contient les vrais PNG et récupère les fichiers Git LFS manquants depuis le dépôt privé avec l’authentification Hugging Face. Le JSONL Qwen reste un fichier dérivé séparé, sans modifier le fichier écrit par le runner.

Dans chaque entrée, `color_measurements.palette_hex` donne les codes RGB réellement présents (avec alpha, nombre de pixels et fréquence); `training_canvas_16.palette_hex` décrit les couleurs après composition sur blanc et padding multiple de 16. Le notebook `sync_dataset_lab.ipynb` sait publier le snapshot Qwen et l’index compressé du dataset après vérification de son aperçu.
