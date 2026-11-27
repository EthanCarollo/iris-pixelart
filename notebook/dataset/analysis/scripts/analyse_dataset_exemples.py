import textwrap

image_sample = random.Random(GRAINE).sample(qwen_rows, min(N_EXEMPLES, len(qwen_rows)))
asset_paths = [annotation_asset_path(row) for row in image_sample]
if TELECHARGER_IMAGES_ATLAS:
    image_paths = atlas_dataset.hydrate(asset_paths, allow_missing=True)
else:
    image_paths = {}
    for name in asset_paths:
        path = atlas_dataset.path(name)
        if path.is_file() and not atlas_dataset.is_pointer(path):
            image_paths[name] = path
image_records = [(row, image_paths.get(name)) for row, name in zip(image_sample, asset_paths)]

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
