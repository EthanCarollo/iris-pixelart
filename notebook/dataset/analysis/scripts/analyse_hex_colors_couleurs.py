couleurs = pd.DataFrame([{'hex': c, 'images': n, 'pixels visibles': pixels_par_couleur[c]} for c, n in images_par_couleur.most_common()])
if couleurs.empty:
    print('Aucune couleur visible mesurée.')
else:
    top = couleurs.head(TOP_AFFICHAGE).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top['hex'], top['images'], color=top['hex'], edgecolor='#888888')
    ax.set(title='Couleurs RGB présentes dans le plus d’images', xlabel='Nombre d’images')
    plt.tight_layout(); plt.show()
    display(couleurs.head(100))