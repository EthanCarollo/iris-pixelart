quality_audit = Counter(
    (row.get('labels', {}).get('visual_quality', '(absent)'), row.get('need_rework', '(absent)'))
    for row in qwen_rows
)
audit_df = pd.DataFrame([
    {'visual_quality': quality, 'need_rework': rework, 'images': count}
    for (quality, rework), count in quality_audit.most_common()
])
display(audit_df)

coverage_rows = []
for field in TAXONOMY_FIELDS:
    missing = sum(not row.get('labels', {}).get(field) for row in qwen_rows)
    unknown = sum('unknown' in values_for(row, field) for row in qwen_rows)
    not_applicable = sum('not_applicable' in values_for(row, field) for row in qwen_rows)
    coverage_rows.append({
        'champ': field,
        'manquant': missing,
        'unknown': unknown,
        'not_applicable': not_applicable,
        'renseigné': len(qwen_rows) - missing,
    })
display(pd.DataFrame(coverage_rows).set_index('champ'))
