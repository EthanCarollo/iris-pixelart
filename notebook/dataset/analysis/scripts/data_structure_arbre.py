def draw_tree(dimension, tree):
    # Place leaves on separate rows, then center each branch above its children.
    nodes, edges = [], []
    next_row = 0
    def visit(name, children, depth):
        nonlocal next_row
        positions = []
        for child, subtree in children.items():
            child_y = visit(child, subtree, depth + 1)
            positions.append((child, child_y))
        if positions:
            y = sum(pos for _, pos in positions) / len(positions)
            edges.extend((depth, y, depth + 1, child_y) for _, child_y in positions)
        else:
            y = next_row
            next_row += 1
        nodes.append((name, depth, y, not bool(children)))
        return y
    visit(dimension, tree, 0)
    depth_max = max(depth for _, depth, _, _ in nodes)
    fig, ax = plt.subplots(figsize=(max(9, (depth_max + 1) * 3.0), max(2.6, next_row * 0.52 + 1.0)))
    ax.set_xlim(-0.55, depth_max + 0.85)
    ax.set_ylim(next_row - 0.35, -0.65)
    ax.axis("off")
    for x1, y1, x2, y2 in edges:
        mid = (x1 + x2) / 2
        ax.plot([x1, mid, mid, x2], [y1, y1, y2, y2], color="#94a3b8", lw=1.1, zorder=1)
    for name, x, y, is_leaf in nodes:
        ax.text(x, y, name, ha="center", va="center", fontsize=9,
                color="#1e3a8a" if is_leaf else "#334155",
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "#dbeafe" if is_leaf else "#f1f5f9", "edgecolor": "#93c5fd" if is_leaf else "#cbd5e1"}, zorder=2)
    plt.tight_layout()
    plt.show()

for dimension, tree in hierarchies.items():
    info = taxonomy["dimensions"][dimension]
    display(Markdown(f"### {info['label']} (`{dimension}`)\n\n{info['definition']}\n\n**Application :** {info['applicability']}"))
    draw_tree(dimension, tree)
    def describe_nodes(nodes, depth=0):
        lines = []
        for name, children in nodes.items():
            kind = "branche" if children else "valeur"
            lines.append(f"{'  ' * depth}- `{name}` ({kind}) : {value_descriptions[dimension][name]}")
            lines.extend(describe_nodes(children, depth + 1))
        return lines
    display(Markdown("\n".join(describe_nodes(tree))))
