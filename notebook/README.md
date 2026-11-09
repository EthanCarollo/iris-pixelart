# Entraînement LoRA Iris-3B pour sprites

Le notebook [`iris3b_sprite_lora.ipynb`](training/iris3b_sprite_lora.ipynb) prépare et entraîne un LoRA Iris-3B à partir du pilote d'annotations Qwen3-VL-4B.

## Lancer

1. Ouvrir le notebook dans Google Colab ou Jupyter sur Linux avec une carte NVIDIA CUDA. Un GPU de 24 Go ou plus est recommandé.
2. Activer le runtime GPU.
3. Exécuter les cellules dans l'ordre. Le notebook lit le jeton Hugging Face depuis la variable d'environnement `HF_TOKEN` ou depuis le fichier local `.env`.
4. Inspecter les images/captions affichées, renseigner `REJECT_ASSET_IDS` si nécessaire, puis lancer la préparation et l'entraînement.

Le notebook ne télécharge que les images annotées retenues au lieu du dépôt complet. Les fichiers préparés et poids sont écrits dans `training/iris_sprite_lora_work/` en local. Le jeton reste dans l'environnement local ou le fichier `.env` et n'est pas intégré aux notebooks ni au dataset.

Le fichier `.env` local est ignoré par Git. Le pilote ne marque pas encore ses captions comme approuvées, et les droits des sources sont notés `unknown`. Vérifier les captions et licences avant tout partage du modèle. Iris-3B génère des images RGB : le fond d'entraînement est blanc et la transparence nécessite un post-traitement.

## Inférence

Ouvrir [`inference_iris3b_sprite_lora.ipynb`](inference/inference_iris3b_sprite_lora.ipynb) dans le même environnement GPU, avec `iris_sprite_lora_work/` et le checkpoint LoRA disponibles. Le notebook charge en priorité le run hiérarchique `iris_pixelart_sprite_96_humanoid`, démarre le moteur au premier essai et affiche l'avancement. Les images générées sont enregistrées dans `training/iris_sprite_lora_work/inference_iris3b_sprite_lora/generated/`. Il active les poids FP8 et l'offload complet des couches du transformer et de l'encodeur texte vers la RAM système. Cela réduit l'usage de VRAM, au prix d'une génération plus lente et d'une forte consommation de RAM. Après nettoyage, relancer la cellule de génération redémarre le moteur automatiquement.

## Lancer avec Conda

Le notebook a besoin de Python 3.12, JupyterLab et PyTorch. L'environnement est défini dans [`environment.yml`](environment.yml). Sur une nouvelle machine, crée-le puis démarre JupyterLab :

```bash
conda env create -f environment.yml
conda activate iris-lora
python -m ipykernel install --user --name iris-lora --display-name "Python (Iris LoRA)"
jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

Si l'environnement `iris-lora` existe déjà, active-le directement et lance `jupyter lab`. Dans le notebook, sélectionne le kernel **Python (Iris LoRA)**.

L'entraînement local demande Linux, un GPU NVIDIA avec pilote compatible CUDA et assez de VRAM. Le notebook tente FP8 et low-VRAM, mais 24 Go restent recommandés ; sur 8 Go, le chargement ou l'entraînement peut échouer. L'environnement installe PyTorch avec les bibliothèques CUDA 13.0 ; cela ne remplace pas le pilote NVIDIA de la machine.

Pour une machine distante, lance JupyterLab avec `--no-browser --ip=127.0.0.1 --port=8888`, puis ouvre un tunnel SSH depuis ton PC en remplaçant `utilisateur` et `hote` :

```powershell
ssh -N -L 8888:127.0.0.1:8888 utilisateur@hote
```

Garde cette fenêtre ouverte, puis ouvre `http://localhost:8888` dans ton navigateur et entre le token affiché par Jupyter sur la machine distante. Le notebook clone AI Toolkit sous son dossier de travail.

## Analyse et taxonomie visuelle

Ouvrir `dataset/analyse_dataset.ipynb`. La hiérarchie préparée utilise `sprite → vue → pose` ; les rôles de jeu sont retirés des tags et des captions. Les descriptions automatiques peuvent encore comporter des erreurs et nécessitent une vérification visuelle. Le JSONL original est conservé sous `qwen3vl_sprite_training_original.jsonl`. `sprite_taxonomy.py` doit rester à côté du notebook d’entraînement. Les checkpoints existants ne sont pas réentraînés automatiquement.

Les images préparées sont placées sur un canevas blanc de 96 × 96 sans agrandissement : les sources qui tiennent dans le canevas conservent leurs pixels 1:1. Les sources dépassant 96 pixels sont réduites en plus proche voisin pour cette configuration ; elles perdent donc des détails. La mention `96x96` dans le prompt désigne le canevas, pas la résolution native du sprite. Les anciens checkpoints ont été entraînés sur la préparation précédente.

## Annotation des taxonomies

Le notebook [`taxonomie_dataset.ipynb`](dataset/taxonomie_dataset.ipynb) annote les images du dataset local avec Qwen3-VL selon les hiérarchies de [`taxonomies.json`](dataset/taxonomies.json). Chaque annotation contient les étiquettes et leurs chemins complets. Commence avec le petit lot configuré par défaut, vérifie les propositions, puis mets `MAX_IMAGES = None` pour traiter toutes les images. La progression est conservée dans `dataset/taxonomy_work/` et la dernière cellule publie le lot dans un nouveau dossier du dataset Hugging Face. Les résultats restent à vérifier et ne remplacent pas les annotations existantes.

Le notebook [`arbre_taxonomie.ipynb`](dataset/arbre_taxonomie.ipynb) affiche les arbres et explique leurs dimensions. Chaque arbre est indépendant : une vue et une pose peuvent être combinées sans former une seule catégorie.

La taxonomie distingue la forme du sujet (`subject_type`), sa structure corporelle (`body_plan`) et les ailes visibles (`wings`), afin de combiner par exemple humanoïde et bipède. Les annotations des anciennes versions restent dans leurs fichiers de travail.

Les définitions de chaque branche et valeur sont dans `value_descriptions` du JSON ; elles sont affichées dans le notebook des arbres et transmises au modèle d’annotation.
