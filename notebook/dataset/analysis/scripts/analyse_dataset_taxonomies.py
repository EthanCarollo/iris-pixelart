TAXONOMY_FIELDS = [
    'asset_type', 'subject_type', 'view', 'pose', 'action',
    'background_type', 'render_family', 'body_plan', 'wings',
    'background_scene', 'visual_quality', 'dominant_color_families',
    'palette_size', 'palette_temperature', 'palette_structure', 'palette_contrast',
]

def values_for(row, field):
    value = row.get('labels', {}).get(field)
    values = value if isinstance(value, list) else [value]
    return [str(value) if value is not None else '(absent)' for value in values]

fig, axes = plt.subplots(4, 4, figsize=(22, 20))
for ax, field in zip(axes.flat, TAXONOMY_FIELDS):
    counts = Counter()
    for row in qwen_rows:
        counts.update(set(values_for(row, field)))
    top = counts.most_common(12)
    if top:
        names, numbers = zip(*top)
        ax.barh(list(names)[::-1], list(numbers)[::-1], color='#5686c5')
        ax.set_xlim(0, max(numbers) * 1.18)
    ax.set_title(field, loc='left', fontsize=10, fontweight='bold')
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(axis='x', alpha=.2)
    ax.set_axisbelow(True)
axes.flat[-1].axis('off')
fig.suptitle('Annotations Qwen3.8 Gyro-S · répartition par champ', fontsize=15, y=1.01)
fig.tight_layout()
plt.show()
