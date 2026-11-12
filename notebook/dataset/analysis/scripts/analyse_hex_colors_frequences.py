images_par_couleur = Counter()
pixels_par_couleur = Counter()
paires = Counter()
palettes = Counter()
tailles = Counter()
statuts = Counter()
ouvre = gzip.open if source.suffix == '.gz' else open
with ouvre(source, 'rt', encoding='utf-8') as stream:
    for line in stream:
        if not line.strip():
            continue
        row = json.loads(line)
        mesures = row.get('color_measurements') or {}
        statut = mesures.get('status', 'absent')
        statuts[statut] += 1
        if statut != 'measured':
            continue
        rgb = Counter()
        for couleur in mesures.get('palette_hex', []):
            if couleur.get('hex'):
                rgb[couleur['hex'].upper()] += int(couleur.get('pixel_count', 0))
        if not rgb:
            continue
        images_par_couleur.update(rgb.keys())
        pixels_par_couleur.update(rgb)
        tailles[len(rgb)] += 1
        palettes[tuple(sorted(rgb))] += 1
        dominantes = sorted(rgb, key=lambda code: (-rgb[code], code))[:TOP_COULEURS_PAR_IMAGE]
        paires.update(combinations(sorted(dominantes), 2))
print('Statuts :', dict(statuts))
print(f"{sum(tailles.values()):,} images avec couleurs visibles · {len(images_par_couleur):,} codes RGB distincts · {len(palettes):,} palettes distinctes")