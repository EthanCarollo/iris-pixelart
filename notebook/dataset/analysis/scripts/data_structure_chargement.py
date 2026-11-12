import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from IPython.display import display, Markdown

cwd = Path.cwd().resolve()
def find_taxonomy_dir(start):
    for directory in (start, *start.parents):
        for candidate in (directory, directory / "taxonomy", directory / "dataset/taxonomy", directory / "notebook/dataset/taxonomy"):
            if (candidate / "taxonomies.json").is_file():
                return candidate
    raise FileNotFoundError("Taxonomie introuvable ; ouvre le notebook dans le projet iris-pixelart.")
taxonomy_dir = find_taxonomy_dir(cwd)
dataset_dir = taxonomy_dir.parent
taxonomy = json.loads((taxonomy_dir / "taxonomies.json").read_text(encoding="utf-8"))
print(f"Taxonomie version {taxonomy['version']} · {len(taxonomy['hierarchies'])} dimensions")

def flatten_hierarchy(nodes, descriptions):
    result = {}
    for name, node in nodes.items():
        descriptions[name] = node["description"]
        result[name] = flatten_hierarchy(node["children"], descriptions)
    return result
value_descriptions = {}
hierarchies = {}
for dimension, tree in taxonomy["hierarchies"].items():
    value_descriptions[dimension] = {}
    hierarchies[dimension] = flatten_hierarchy(tree, value_descriptions[dimension])
