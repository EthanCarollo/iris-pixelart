# Charge le token depuis l’environnement ou les fichiers .env connus, sans jamais l’imprimer.
import textwrap
for env_file in (PROJECT_ROOT / '.env', PROJECT_ROOT / 'training/.env', Path.home() / 'iris-sprite-lora/training/.env'):
    if not env_file.is_file():
        continue
    for line in env_file.read_text(encoding='utf-8').splitlines():
        key, separator, value = line.partition('=')
        if separator and key.strip() in {'HF_TOKEN', 'HUGGINGFACE_HUB_TOKEN'}:
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
            break
HF_TOKEN = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN') or True


cache_roots = [PROJECT_ROOT / 'notebook/training/iris_sprite_lora_work/source_images']
image_sample = random.Random(GRAINE).sample(qwen_rows, min(N_EXEMPLES, len(qwen_rows)))
image_records = []
for row in image_sample:
    source = row.get('source', {})
    relative_path = str(source.get('relative_path', '')).replace('\\', '/')
    root_name = str(source.get('root', 'data'))
    image_path = next((root / root_name / relative_path for root in cache_roots if (root / root_name / relative_path).is_file()), None)
    if image_path is None and TELECHARGER_IMAGES_HF and relative_path:
        try:
            image_path = Path(hf_hub_download(
                repo_id=DATASET_REPO, repo_type='dataset',
                filename=f'data/{relative_path}', token=HF_TOKEN,
            ))
        except Exception:
            image_path = None
    image_records.append((row, image_path))

# Matplotlib : chaque sprite a une zone de texte distincte juste en dessous.
import matplotlib.pyplot as plt

columns = 2
rows_count = (len(image_records) + columns - 1) // columns
fig = plt.figure(figsize=(16, rows_count * 7.3), facecolor='#f5f5f2')
outer = fig.add_gridspec(rows_count, columns, wspace=0.12, hspace=0.16)
for index, (row, image_path) in enumerate(image_records):
    outer_row, outer_col = divmod(index, columns)
    card = outer[outer_row, outer_col].subgridspec(2, 1, height_ratios=[3.0, 2.3], hspace=0.035)
    image_ax = fig.add_subplot(card[0])
    text_ax = fig.add_subplot(card[1])
    image_ax.set_facecolor('#e9e9e5')
    image_ax.axis('off')
    asset_id = str(row.get('asset_id', '(id absent)'))
    source_path = str(row.get('source', {}).get('relative_path', '(source absente)'))
    if image_path is not None and image_path.is_file():
        try:
            with Image.open(image_path) as image:
                image_ax.imshow(image.convert('RGBA'), interpolation='nearest')
                dimensions = f'{image.width} × {image.height} px'
        except OSError:
            dimensions = 'image illisible'
            image_ax.text(.5, .5, dimensions, ha='center', va='center', transform=image_ax.transAxes)
    else:
        dimensions = 'image indisponible'
        image_ax.text(.5, .5, dimensions, ha='center', va='center', transform=image_ax.transAxes)
    image_ax.set_title(asset_id, fontsize=9, loc='left', pad=5, color='#262a25')

    labels = row.get('labels', {})
    caption = str(row.get('caption_candidate') or 'Description Qwen absente.')
    colors = labels.get('dominant_color_families') or []
    color_text = ', '.join(map(str, colors)) if isinstance(colors, list) else str(colors or '—')
    measured_palette = row.get('color_measurements', {}).get('palette_hex', [])
    exact_colors = [
        f"{entry.get('hex')} ({entry.get('pixel_count', 0):,} px)"
        for entry in measured_palette[:12] if entry.get('hex')
    ]
    exact_palette_text = ', '.join(exact_colors) or 'non mesurée'
    if len(measured_palette) > len(exact_colors):
        exact_palette_text += f" · +{len(measured_palette) - len(exact_colors)} autres couleurs"
    description = (
        f"CAPTION QWEN\n{textwrap.fill(caption, width=68)}\n\n"
        f"Source : {textwrap.fill(source_path, width=68)}\n"
        f"Résolution : {dimensions}   ·   Qualité : {labels.get('visual_quality', 'inconnue')}   ·   Rework : {row.get('need_rework', 'inconnu')}\n"
        f"Sujet : {labels.get('subject_type', '—')}   ·   Vue : {labels.get('view', '—')}   ·   Pose : {labels.get('pose', '—')}\n"
        f"Familles : {textwrap.fill(color_text, width=68)}\n"
        f"Hex exacts (12 couleurs dominantes) : {textwrap.fill(exact_palette_text, width=68)}\n"
        f"Palette : {labels.get('palette_size', '—')} / {labels.get('palette_structure', '—')}   ·   Contraste : {labels.get('palette_contrast', '—')}"
    )
    text_ax.axis('off')
    text_ax.set_facecolor('white')
    text_ax.text(
        0.025, 0.98, description, transform=text_ax.transAxes,
        ha='left', va='top', fontsize=11, color='#292d28',
        linespacing=1.3, wrap=True,
        bbox={'boxstyle':'round,pad=0.7', 'facecolor':'white', 'edgecolor':'#d8dcd4', 'linewidth':0.8},
    )

for index in range(len(image_records), rows_count * columns):
    ax = fig.add_subplot(outer[index // columns, index % columns])
    ax.axis('off')
fig.suptitle('Images source et descriptions Qwen3.8 Gyro-S', fontsize=16, fontweight='bold', y=.995, color='#20231f')
fig.subplots_adjust(left=.025, right=.985, top=.975, bottom=.02)
display(fig)
plt.close(fig)
