top_paires = paires.most_common(TOP_AFFICHAGE)
if not top_paires:
    print('Aucune paire disponible.')
else:
    fig, ax = plt.subplots(figsize=(11, max(4, len(top_paires)*.42)))
    for y, ((a, b), count) in enumerate(reversed(top_paires)):
        ax.barh(y, count, color='#b9bec8')
        ax.scatter([-0.08, -0.04], [y, y], c=[a, b], s=180, marker='s', edgecolors='#777', transform=ax.get_yaxis_transform(), clip_on=False)
    ax.set_yticks(range(len(top_paires)), [f'{a} + {b}' for (a,b), _ in reversed(top_paires)])
    ax.set(xlabel='Nombre d’images', title=f'Paires parmi les {TOP_COULEURS_PAR_IMAGE} couleurs dominantes')
    fig.subplots_adjust(left=.27); plt.show()
    display(pd.DataFrame([{'hex 1': a, 'hex 2': b, 'images': count} for (a,b), count in top_paires]))