from pathlib import Path
import json
color_taxonomy = json.loads((taxonomy_dir / "taxonomies_couleurs.json").read_text(encoding="utf-8"))
for dimension, tree in color_taxonomy["hierarchies"].items():
    info = color_taxonomy["dimensions"][dimension]
    display(Markdown(f"### {info['label']} ({dimension}) — {info['cardinality']}\n\n{info['definition']}\n\n**Application :** {info['applicability']}"))
    draw_tree(dimension, flatten_hierarchy(tree, {}))
