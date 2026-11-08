# Entraînement LoRA Iris-3B pour sprites

Le notebook [`iris3b_sprite_lora.ipynb`](iris3b_sprite_lora.ipynb) prépare et entraîne un LoRA Iris-3B à partir du pilote d'annotations Qwen3-VL-4B.

## Lancer

1. Ouvrir le notebook dans Google Colab ou Jupyter sur Linux avec une carte NVIDIA CUDA. Un GPU de 24 Go ou plus est recommandé.
2. Activer le runtime GPU.
3. Exécuter les cellules dans l'ordre et fournir un jeton Hugging Face en lecture au dépôt privé `ethanzxv/pixelart-dataset`.
4. Inspecter les images/captions affichées, renseigner `REJECT_ASSET_IDS` si nécessaire, puis lancer la préparation et l'entraînement.

Le notebook ne télécharge que les images annotées retenues au lieu du dépôt complet. Les fichiers préparés et poids sont écrits dans `iris_sprite_lora_work/` dans l'environnement d'exécution. Le jeton n'est pas enregistré.

Le pilote ne marque pas encore ses captions comme approuvées, et les droits des sources sont notés `unknown`. Vérifier les captions et licences avant tout partage du modèle. Iris-3B génère des images RGB : le fond d'entraînement est blanc et la transparence nécessite un post-traitement.

## Lancer sur un PC distant

Le PC distant doit avoir Linux, Git, Python 3.11 ou plus et CUDA. Le notebook tente FP8 et low-VRAM pour les GPU modestes, mais 24 Go de VRAM restent recommandés ; sur 8 Go, le chargement ou l'entraînement peut encore échouer. Clone ce dépôt sur ce PC, puis installe JupyterLab dans son environnement Python :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip jupyterlab
jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

Depuis ton PC, ouvre un tunnel SSH en remplaçant `utilisateur` et `hote` par les informations de la machine distante :

```powershell
ssh -N -L 8888:127.0.0.1:8888 utilisateur@hote
```

Garde cette fenêtre ouverte, puis ouvre `http://localhost:8888` dans ton navigateur et entre le token affiché par Jupyter sur la machine distante. Le notebook clone AI Toolkit sous son dossier de travail, donc il fonctionne hors de Colab.
