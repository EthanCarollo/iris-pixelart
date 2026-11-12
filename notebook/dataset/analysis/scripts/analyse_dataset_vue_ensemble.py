asset_ids = [row.get('asset_id') for row in qwen_rows]
model_counts = Counter(row.get('model', '(absent)') for row in qwen_rows)
review_counts = Counter(row.get('review_status', '(absent)') for row in qwen_rows)
quality_counts = Counter(row.get('labels', {}).get('visual_quality', '(absent)') for row in qwen_rows)
rework_counts = Counter(row.get('need_rework', '(absent)') for row in qwen_rows)
timestamps = [row.get('annotated_at_utc') for row in qwen_rows if row.get('annotated_at_utc')]
summary = pd.DataFrame([
    ('Lignes Qwen3.8 Gyro-S', len(qwen_rows)),
    ('asset_id uniques', len(set(asset_ids))),
    ('asset_id dupliqués', len(asset_ids) - len(set(asset_ids))),
    ('Première annotation (UTC)', min(timestamps) if timestamps else 'inconnue'),
    ('Dernière annotation (UTC)', max(timestamps) if timestamps else 'inconnue'),
    ('Modèles', ', '.join(f'{k}: {v}' for k, v in model_counts.items())),
], columns=['Mesure', 'Valeur'])
display(summary)
print('Statuts de revue :', dict(review_counts))
print('Qualité visuelle :', dict(quality_counts))
print('need_rework :', dict(rework_counts))
