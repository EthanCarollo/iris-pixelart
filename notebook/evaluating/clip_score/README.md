# Comparaison CLIP des modèles de pixel art

Ouvrir [clip_score.ipynb](clip_score.ipynb). Les prompts sont dans [prompts.jsonl](prompts.jsonl). Générer à l'avance les images pour Iris de base, Iris avec LoRA et les modèles à comparer, avec les mêmes prompts et les mêmes nombres d'essais. Le notebook crée un manifeste CSV puis vérifie que chaque modèle couvre les mêmes couples prompt/graine.

Le calcul utilise `openai/clip-vit-base-patch32` sur CPU par défaut et ne lance aucune génération ; il faut télécharger CLIP au premier usage. Les dépendances sont PyTorch, Transformers, Pillow, NumPy, pandas et Matplotlib. Installer `transformers` dans le kernel si nécessaire.

La mesure affichée est **100 × max(0, cosinus CLIP)**, pas la formule ou l'échelle exacte du CLIPScore académique. Elle mesure l'accord entre image et texte, pas la qualité du pixel art. L'aplatissement sur blanc et le prétraitement CLIP sont identiques pour tous les modèles ; ses redimensionnements peuvent lisser les pixels. Les différences entre modèles utilisent des moyennes par prompt, puis un bootstrap apparié des prompts.

Les données générées et les résultats sont écrits dans `work/`, ignoré par Git. L'évaluation ne publie rien et ne compare pas de modèles pour lesquels les images manquent.

Références : [documentation CLIP de Transformers](https://huggingface.co/docs/transformers/model_doc/clip) et [article CLIPScore](https://aclanthology.org/2021.emnlp-main.595/).
