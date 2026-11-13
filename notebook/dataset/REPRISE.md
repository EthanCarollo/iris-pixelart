# Reprendre le dataset et les annotations

État sauvegardé le 13 novembre 2026 : 6 905 annotations Qwen3.8 Gyro-S et 6 905 palettes exactes, avec un objectif de 10 000. Les annotations sont automatiques et restent à vérifier. Le dataset source est privé : `ethanzxv/pixelart-dataset`. Les images source se trouvent déjà dans ce dépôt Hugging Face ; elles ne sont pas copiées dans le dépôt Git du code.

Le checkpoint privé est `annotations/checkpoints/20261113T093216Z`, au commit dataset `b399daded3dce679e6d4c264c357dc14d132df90`. Son manifeste indique les empreintes des fichiers et la révision des images source : `095c6f0773a9b1ebd7bb6a00334254f04ac8e085`. Le modèle, le token et les mots de passe ne sont pas inclus dans Git ou le checkpoint.

## Installer dans un autre environnement

1. Cloner le dépôt Git du projet, puis installer Python 3, Git, Git LFS, Node.js 22 et npm. Installer les dépendances Python :

   ```bash
   python -m pip install huggingface_hub pillow tqdm requests python-dotenv matplotlib pandas pyyaml
   ```

2. Copier `dataset-viewer/.env.example` vers `dataset-viewer/.env` et y renseigner `HF_TOKEN` avec un token ayant accès au dataset privé. Depuis `dataset-viewer/`, exécuter :

   ```bash
   npm ci
   npm run download:full
   ```

   Cette commande télécharge la copie locale dans `dataset-viewer/datasets/pixelart-dataset/.gitrepo`. Elle utilise la révision courante du dataset ; pour reproduire exactement le corpus source du checkpoint, vérifier que son commit correspond à `095c6f0773a9b1ebd7bb6a00334254f04ac8e085` ou récupérer cette révision avec Git et Git LFS avant de reprendre. Garder le dépôt local dans un état propre.

3. Depuis la racine du projet, restaurer les annotations et palettes :

   ```bash
   python notebook/dataset/taxonomy/scripts/restore_checkpoint.py
   ```

   Le script vérifie les empreintes et refuse d’écraser un fichier local différent. Les JSONL sont restaurés dans `notebook/dataset/taxonomy/taxonomy_work/`.

4. Démarrer le viewer avec `cd dataset-viewer && npm run dev`, puis ouvrir http://localhost:3000. Pour consulter les données : `notebook/dataset/analysis/analyse_dataset.ipynb` et `analyse_hex_colors.ipynb`.

## Reprendre Qwen3.8 sur Strata

Démarrer séparément un serveur Strata compatible vision avec le modèle `qwen3.8-flash-next-gyro-gyro-s`. Le dernier run utilisait trois requêtes simultanées, `medium`, 1 024 tokens de sortie et un contexte serveur de 7 168 tokens. Si le serveur est distant, ouvrir un tunnel SSH vers son port 8080, par exemple `ssh -N -L 8080:127.0.0.1:8080 utilisateur@machine` en ajoutant le port SSH approprié. Utiliser ses propres identifiants ; aucun mot de passe n’est stocké ici.

Depuis la racine du projet :

```bash
python -u notebook/dataset/taxonomy/scripts/resume_annotations.py --target 10000 2>&1 | tee -a notebook/dataset/taxonomy/taxonomy_work/run_resume.log
```

La commande lit les lignes déjà sauvegardées et n’annote que les images manquantes jusqu’au total demandé. Le fichier d’annotations est écrit progressivement. Les fichiers JSONL locaux évoluent après reprise : pour les republier, utiliser `notebook/dataset/sync_dataset_lab.ipynb` après vérification, ou la cellule de synchronisation du notebook de taxonomie. Avant une nouvelle préparation d’entraînement, régénérer le snapshot enrichi :

```bash
python notebook/dataset/analysis/enrich_qwen_exact_palettes.py
```

Le fichier `dataset_color_palettes.jsonl.gz` calculé avant ce checkpoint était incomplet ; le recréer avec `--all-dataset-assets` après téléchargement des images si une analyse globale des couleurs est nécessaire.
