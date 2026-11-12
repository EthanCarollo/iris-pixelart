EXAMPLE_IMAGE = dataset_dir.parent / "training/iris_sprite_lora_work/dataset_96/4e7203a272af59b647a90108.png"
example_labels = {
    "asset_type": "single_sprite", "subject_type": "unknown", "view": "unknown",
    "pose": "unknown", "action": "unknown", "background_type": "solid_color",
    "render_family": "pixel_art", "body_plan": "unknown", "wings": "unknown",
    "background_scene": "not_applicable", "visual_quality": "unknown",
}

def find_path(tree, value, prefix=()):
    for name, children in tree.items():
        path = (*prefix, name)
        if not children and name == value:
            return list(path)
        found = find_path(children, value, path)
        if found is not None:
            return found
    return None

paths = {key: find_path(hierarchies[key], value) for key, value in example_labels.items()}
if any(path is None for path in paths.values()):
    raise ValueError("L’exemple ne correspond plus à taxonomies.json ; mets ses étiquettes à jour.")
example = {
    "image": str(EXAMPLE_IMAGE.relative_to(dataset_dir.parent)),
    "labels": example_labels,
    "taxonomy_paths": paths,
    "need_rework": None,
    "free_caption": "A small pixel-art figure with orange, red and blue details on a white background.",
    "review_status": "illustrative_example_not_reviewed",
}
if EXAMPLE_IMAGE.is_file():
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(plt.imread(EXAMPLE_IMAGE), interpolation="nearest")
    ax.set_title("Sprite d’exemple")
    ax.axis("off")
    plt.show()
else:
    print(f"Image d’exemple indisponible : {EXAMPLE_IMAGE}. Prépare le dataset d’entraînement pour l’afficher.")
print(json.dumps(example, ensure_ascii=False, indent=2))
