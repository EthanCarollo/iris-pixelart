if tailles:
    fig, ax = plt.subplots(figsize=(10, 4))
    x = sorted(tailles)
    ax.bar(x, [tailles[k] for k in x], width=1)
    ax.set(xlabel='Nombre de couleurs RGB distinctes', ylabel='Images', title='Taille des palettes source')
    ax.set_yscale('log'); plt.tight_layout(); plt.show()
    display(pd.DataFrame([{'couleurs': len(p), 'images': count, 'palette hex': ', '.join(p)} for p, count in palettes.most_common(TOP_AFFICHAGE)]))
else:
    print('Aucune palette visible mesurée.')