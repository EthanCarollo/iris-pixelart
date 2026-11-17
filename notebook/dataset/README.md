# Dataset Lab

Pour installer le projet ailleurs et reprendre les annotations sauvegardées, voir [REPRISE.md](REPRISE.md).

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

## Nouvelles annotations LFM3-VL-2B

Ce pipeline repart des images, sans charger les annotations ni les captions Qwen. Il écrit dans `taxonomy/lfm3_work/`, ignoré par Git. Aucun upload vers Hugging Face n'est effectué. Le token reste dans `dataset-viewer/.env` ou `HF_TOKEN`.

Le commit dataset `4048251856d8e082ab9a0b31615f91a0f33f99ac` contient **166 203 images dans `data/` et 20 449 images dans `source_sheets/`**, soit **186 652 fichiers source** hors rapports. Ces comptes ont été vérifiés à la fois avec l'arbre Git complet et l'API HF paginée. `RepoInfo.siblings` tronque la liste des grands dépôts et ne doit pas servir à compter le corpus.

### Configuration actuelle : BF16 sur le serveur NVIDIA

Le run courant utilise le modèle officiel **BF16 non quantifié** sur la RTX 5060 Ti 16 Go, avec **vLLM 0.34.0**, batching continu, cache de préfixe, CUDA graphs, FlashAttention 2 et sampler FlashInfer. Le modèle est épinglé à `23387beda43c60310348e83e25f78f33c3140d3a`. Les 424 tenseurs du fichier Safetensors ont été vérifiés en BF16 ; le serveur utilise `dtype=bfloat16`, sans quantification, avec cache KV automatique de même précision. Les poids Q8 du test Mac ne sont pas utilisés par ce run.

Le GPU calcule sur le serveur configuré par `IRIS_SSH_HOST` (ou `--host`). Les images et résultats restent sur le Mac ; l'API vLLM écoute seulement sur le localhost distant et passe par un tunnel SSH local sur 8092. Les téléchargements manquants sont temporaires et supprimés après traitement. Les deux machines ayant peu d'espace libre, la base compresse ses résultats et l'export est **`annotations_lfm3.jsonl.gz`**. Les découpes restent des PNG individuels. Les compteurs de palette restent exacts ; les mesures jointes à chaque annotation conservent les 64 couleurs les plus fréquentes, avec un indicateur de troncature. L'index complet des palettes se produit séparément avec le script d'enrichissement.

Chaque image passe deux fois dans LFM3 : brouillon complet, puis **relecture avec la même image et le brouillon**. Le second appel corrige la caption et les étiquettes. Le résultat conserve le brouillon, les champs changés et les contrôles visuels. La grammaire de relecture tient compte du nombre de sujets : une image identifiée comme sujet unique ne peut pas être étiquetée comme planche. Les fichiers sous `source_sheets/` passent aussi par une vérification géométrique indépendante de la classification du modèle, pour ne pas ignorer une planche mal étiquetée. La relecture automatique ne vaut pas validation humaine.

Benchmarks du 17 novembre 2026, génération + relecture, sorties validées :

| Capacité serveur | Requêtes client | Images testées | Tokens de sortie/s cumulés | Images/s |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 16 | 64 | 912 | 2,44 |
| 32 | 32 | 64 | 1 756 | 4,73 |
| 32 | 64 | 64 | 1 804 | 4,84 |
| 128 | 128 | 128 | 2 189 | 5,93 |

Le meilleur test donne environ **8,7 heures pour les fichiers sources**, avant ajout des crops et éventuelles reprises. Le débit du corpus complet dépend aussi des téléchargements et de la complexité des sources. Les moyennes sont calculées sur la durée de chaque lot, démarrage du modèle exclu ; les pics du journal serveur sont distincts. Les lots n'ont pas tous la même taille et réutilisent des images déjà échauffées. Le nombre final d'images après découpe reste à établir.

Pour reprendre le traitement distant déjà préparé :

```bash
notebook/dataset/taxonomy/lfm3_work/.venv/bin/python notebook/dataset/taxonomy/scripts/start_lfm3_remote.py --all --workers 128
```

L'authentification du tunnel est lue dans `lfm3_work/remote/ssh-password`, fichier local ignoré par Git et protégé en mode 600. Aucun mot de passe n'est inscrit dans le code ni les commandes. Le tunnel et le runner persistent après fermeture du terminal. Le runner utilise `caffeinate -i` sur le Mac. Les résultats BF16 sont dans **`taxonomy/lfm3_work/remote/full_bf16/`**, séparés des essais Mac/Qwen. `progress.json` donne le débit des images, celui des tokens de sortie et l'ETA de la file connue ; `self_review` contient la première proposition et les corrections. Un seuil d'espace libre arrête la soumission de nouvelles tâches avant de remplir le disque.

Sur le serveur, l'environnement est `/home/eth/iris-lfm3-vllm/.venv` et les poids sont dans `/home/eth/iris-lfm3-vllm/model`. Le démarrage reproductible utilise `scripts/serve_lfm3_vllm.py`. Installer `requirements-vllm-lfm3.txt` dans un environnement Python 3.12 isolé : NVCC, CRT et NVVM sont épinglés ensemble à CUDA 13.0 pour correspondre au runtime PyTorch. Le script expose le layout `lib64` attendu par le linker CUDA à partir des wheels NVIDIA.

```bash
# Sur le serveur, avec les fichiers de scripts et dépendances copiés :
nohup /home/eth/iris-lfm3-vllm/.venv/bin/python /home/eth/iris-lfm3-vllm/serve_lfm3_vllm.py --model /home/eth/iris-lfm3-vllm/model --max-seqs 128 > /home/eth/iris-lfm3-vllm/server.log 2>&1 &
```

Pour exporter un snapshot compressé pendant le run distant :

```bash
notebook/dataset/taxonomy/lfm3_work/.venv/bin/python notebook/dataset/taxonomy/scripts/annotate_lfm3.py --backend vllm --export-only --output notebook/dataset/taxonomy/lfm3_work/remote/full_bf16
```

### Session CPU auxiliaire

Une instance Transformers **BF16 non quantifiée** peut tourner à côté de vLLM sur le Ryzen 7 9800X3D. Le test avec grammaire xgrammar et relecture donne 405 tokens en 64,5 secondes, soit environ **6,3 tokens/s et une image/minute**. Ce gain est inférieur à 1 % du débit GPU. Le réglage utilisé est de six threads, affinité sur les CPU 2–7, priorité `nice 19`, attente OpenMP passive ; les CPU 0–1 restent disponibles pour alimenter le GPU. Le GPU et les autres services gardent leur priorité normale.

Le service `/home/eth/iris-lfm3-vllm/serve_lfm3_cpu.py` écoute uniquement sur le localhost distant, port 8093, exposé par un second tunnel SSH. Il masque CUDA et vérifie que les poids chargés sont en BF16 sur CPU. Il réutilise les mêmes poids et schémas ; chaque image reçoit aussi la double passe. Le client auxiliaire est :

```bash
notebook/dataset/taxonomy/lfm3_work/.venv/bin/python notebook/dataset/taxonomy/scripts/run_lfm3_cpu_worker.py --output notebook/dataset/taxonomy/lfm3_work/remote/full_bf16
```

Ce client utilise la même base SQLite. Il prend prioritairement les images en échec non encore testées sur CPU, puis la fin de la file pendant que le GPU traite son début. Les revendications sont transactionnelles et utilisent l'état distinct `cpu_running` ; le worker cesse de prendre des images ordinaires lorsque moins de 16 384 restent en attente. Les résultats CPU sont ajoutés à `annotations_lfm3.cpu.jsonl.gz` pour ne pas corrompre l'export gzip du processus GPU par des écritures simultanées. L'export final ou un snapshot de la base fusionne les deux. `cpu_progress.json` donne l'état et les mesures du worker auxiliaire.

Pour éviter de bloquer toute la file sur une erreur propre à une image, les échecs de validation restent à reprendre, tandis que les échecs d'infrastructure répétés déclenchent l'arrêt. Les répétitions identiques d'une couleur sont supprimées selon la sémantique d'ensemble multi-étiquette, avec l'avant/après conservé dans `self_review.normalizations`. Les valeurs invalides, les sentinelles mélangées et les contradictions sont toujours rejetées.

Le cache Safetensors de cette tâche a été déplacé en RAM dans `/dev/shm/iris-lfm3-bf16-23387b/`, avec vérification SHA-256 et un lien depuis le dossier du modèle, pour libérer 3,6 Go sur le disque du serveur. Ce cache est volatil au redémarrage : les poids publics doivent alors être retéléchargés avant de relancer les services. Les annotations et les checkpoints restent persistants sur le Mac.

### Synchroniser les résultats vers le dataset privé

Le script suivant prend un snapshot SQLite cohérent pendant que GPU et CPU continuent à écrire, valide chaque annotation terminée et publie un nouveau checkpoint sur `main` du dataset privé :

```bash
notebook/dataset/taxonomy/lfm3_work/.venv/bin/python notebook/dataset/taxonomy/scripts/sync_lfm3_checkpoint.py --upload
```

Sans `--upload`, il prépare seulement les fichiers localement. Le checkpoint est ajouté sous `annotations/lfm3_vl_bf16/checkpoints/<horodatage>/`. Il contient l'export fusionné CPU/GPU, la base de reprise compressée, l'inventaire, les taxonomies, les schémas, un état de couverture et des empreintes SHA-256. Les états en cours et les budgets de reprise sont réinitialisés seulement dans la copie destinée à la restauration.

Toutes les images dérivées référencées dans le snapshot, y compris celles encore en attente d'annotation, sont publiées sous `data/lfm3_extracted/<sha256>.png`. Les images identiques sont partagées ; `derived_assets.jsonl.gz` conserve chaque frame, sa boîte et son parent. Le champ `source.dataset_path` dans les annotations donne le chemin image sur le Hub. Pour restaurer les crops, télécharger ces chemins vers leurs `checkpoint_path`, puis rétablir les `local_path` de la base relativement au nouveau dossier de travail. Les images originales restent liées à la révision source indiquée dans le manifeste.

`experiments.tar.gz` conserve séparément les essais Mirage, les benchmarks et les tests de description ; ces résultats ne sont pas mélangés à l'export d'annotations courant. La sélection des fichiers est explicite et exclut les tokens, mots de passe, environnements, processus et fichiers temporaires. La publication ne remplace aucune annotation Qwen et ne modifie pas la visibilité privée du dépôt. Un checkpoint intermédiaire est marqué incomplet tant que la file ou la revue des découpes reste ouverte.

### Variante locale Mac

Sur Apple Silicon, installer les dépendances dans un environnement dédié, puis tester :

```bash
python3 -m venv notebook/dataset/taxonomy/lfm3_work/.venv
notebook/dataset/taxonomy/lfm3_work/.venv/bin/pip install -r notebook/dataset/taxonomy/requirements-lfm3.txt
notebook/dataset/taxonomy/lfm3_work/.venv/bin/python notebook/dataset/taxonomy/scripts/start_lfm3_local.py --benchmark 12 --output notebook/dataset/taxonomy/lfm3_work/test
```

Pour traiter **tout le corpus**, sans filtre de qualité, d'éligibilité à l'entraînement ou de couverture Qwen :

```bash
notebook/dataset/taxonomy/lfm3_work/.venv/bin/python notebook/dataset/taxonomy/scripts/start_lfm3_local.py --all
```

Le launcher télécharge le runtime officiel llama.cpp `b12359`, les poids officiels LFM3-VL-2B Q8 et le projecteur F16, puis démarre un serveur Metal local sur le port 8091 et le traitement en arrière-plan. Il utilise `caffeinate -i` pendant le calcul. La copie locale `../ds/mother3-sync-latest-worktree` est utilisée uniquement si son commit correspond à celui du Hub ; les vrais pixels Git LFS absents sont téléchargés à la demande. `--local-root` permet de changer ce chemin. Relancer la même commande reprend la file persistante ; un verrou interdit deux runners dans le même dossier.

La description est produite dans un appel dédié : sujet, silhouette/vue, détails distinctifs et positions relatives. Elle commence directement par **A** ou **An**, sans « The image displays ». Un second appel choisit les 16 dimensions de taxonomie. Les deux appels utilisent des JSON Schemas compilés en GBNF par llama.cpp pour masquer les tokens interdits pendant la génération. Les valeurs autorisées, les cardinalités et les dépendances fond/décor sont contraintes ; la transparence et la taille de palette viennent des pixels. La validation Python rejette les doublons de couleurs, les sentinelles mélangées, les captions incomplètes et les réponses tronquées. `annotation.schema.json` et `annotation.gbnf` documentent le résultat combiné ; chaque résultat conserve les empreintes des schémas réellement utilisés par étape.

Les PNG/BMP/GIF et autres formats d'image dans les deux racines sont inventoriés. Chaque frame d'un GIF animé est extraite en PNG et annotée séparément. Les liens de découpes déjà vérifiées sont repris depuis les métadonnées et manifestes, sans réutiliser leurs descriptions. Les nouvelles planches sont séparées par les lignes/colonnes entièrement vides, en conservant chaque région visible, son parent et sa boîte source. Les cellules peuvent aussi être définies par `--split-overrides`, un JSON du type :

```json
{"source_sheets/example.png": {"cell_width": 32, "cell_height": 32, "margin_x": 0, "margin_y": 0, "spacing_x": 0, "spacing_y": 0}}
```

Les grilles qui perdent des pixels sont refusées. Les dispositions ambiguës restent explicitement `pending_geometry`, les crops encore classés comme planches restent `pending_crop_review`, et les découpes automatiques portent `auto_split_unreviewed`. Une séparation par espaces peut isoler des fragments ou du texte : couverture des pixels ne signifie pas que chaque crop est un sprite complet. Ces cas doivent être revus pour atteindre une couverture individuelle complète. Aucune planche ambiguë n'est déclarée entièrement découpée. Un benchmark annote seulement les sources sélectionnées et prépare ses crops sans lancer leur annotation.

Dans `lfm3_work/full/`, consulter :

- `progress.json` : état de la file, débit mesuré et ETA des tâches actuellement connues ;
- `annotations_lfm3.jsonl` : export lisible des nouvelles annotations ;
- `annotations.sqlite3` : état durable et résultats, en mode WAL ;
- `run.log` : progression et échecs, réessayés jusqu'à trois fois ;
- `crops/` : images individuelles extraites ;
- `review.html` : aperçu des 24 premières images (avec `--keep-previews` pour conserver toutes les autres).

Le coût total augmente avec les nouvelles découpes. Les estimations du benchmark sont des extrapolations, pas des garanties. La grammaire garantit la structure et les valeurs autorisées, **pas la justesse visuelle** : des erreurs de sujet et des détails inventés ont été observés, surtout sur les petites icônes. Les annotations restent des propositions non vérifiées et n'héritent pas d'une autorisation d'entraînement ou de droits sur les images.

Pour exporter un snapshot cohérent sans arrêter un traitement :

```bash
notebook/dataset/taxonomy/lfm3_work/.venv/bin/python notebook/dataset/taxonomy/scripts/annotate_lfm3.py --export-only --output notebook/dataset/taxonomy/lfm3_work/full
```

Cette commande écrit `annotations_lfm3.snapshot.jsonl` pour préserver l'export alimenté par le runner.

Pour arrêter proprement, envoyer SIGINT au `pid` du runner indiqué dans `progress.json`. Les résultats validés restent dans la base ; le serveur local peut rester chargé pour une reprise. Les changements de consigne, de taxonomie ou de modèle exigent un nouveau dossier de sortie. Le modèle, ses poids quantifiés et le runtime proviennent de [LiquidAI/LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B), du [dépôt GGUF officiel](https://huggingface.co/LiquidAI/LFM3-VL-2B-GGUF) et de [llama.cpp](https://github.com/ggml-org/llama.cpp).
