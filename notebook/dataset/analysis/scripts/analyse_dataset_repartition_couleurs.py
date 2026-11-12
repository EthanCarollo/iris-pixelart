from collections import Counter

fig, axes = plt.subplots(len(image_records), 1, figsize=(13, max(3, len(image_records) * 1.55)), squeeze=False)
fig.patch.set_facecolor('white')
for ax, (row, _) in zip(axes.flat, image_records):
    rgb = Counter()
    for color in row.get('color_measurements', {}).get('palette_hex', []):
        if color.get('hex'):
            rgb[color['hex']] += color.get('pixel_count', 0)
    total = sum(rgb.values())
    source_path = row.get('source', {}).get('relative_path', row.get('asset_id', 'source inconnue'))
    ax.set_title(str(source_path), loc='left', fontsize=10, pad=5)
    if not total:
        ax.text(.5, .5, 'Palette non mesurée ou aucun pixel visible', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        continue
    dominant = rgb.most_common(MAX_COULEURS_AFFICHÉES)
    autres = total - sum(count for _, count in dominant)
    segments = [(hex_code, count, hex_code) for hex_code, count in dominant]
    if autres:
        segments.append(('Autres', autres, '#a0a0a0'))
    left = 0
    legend_handles = []
    from matplotlib.patches import Patch
    for label, count, hex_code in segments:
        fraction = count / total
        ax.barh(0, fraction, left=left, height=.65, color=hex_code, edgecolor='white', linewidth=.6)
        legend_handles.append(Patch(facecolor=hex_code, edgecolor='#888', label=f'{label} : {fraction:.1%}'))
        left += fraction
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1], ['0 %', '25 %', '50 %', '75 %', '100 %'])
    ax.set_yticks([])
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0, -.25), ncol=4, fontsize=8, frameon=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
fig.subplots_adjust(hspace=1.5, left=.08, right=.98, top=.97, bottom=.06)
display(fig)
plt.close(fig)