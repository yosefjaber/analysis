import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

# ------------------------------------------------------------------
# Helper: draw a fully‑connected feed‑forward network on a given Axes
# ------------------------------------------------------------------
def draw_network(ax, layer_sizes, node_radius=0.12, layer_h=1.0, x_pad=1.5):
    """
    layer_sizes : list[int]  – number of neurons in each layer (left→right)
    node_radius : float      – circle radius in axis units
    layer_h     : float      – vertical spacing between neurons
    x_pad       : float      – horizontal spacing between layers
    """
    # Compute x positions for each layer
    xs = np.arange(len(layer_sizes)) * x_pad

    # Draw nodes
    for i, (x, n_nodes) in enumerate(zip(xs, layer_sizes)):
        y_top = (n_nodes - 1) * layer_h / 2
        for j in range(n_nodes):
            y = y_top - j * layer_h
            circ = Circle((x, y), node_radius, ec='k', fc='w', zorder=3)
            ax.add_patch(circ)

    # Draw connections
    for (x0, n0), (x1, n1) in zip(zip(xs, layer_sizes), zip(xs[1:], layer_sizes[1:])):
        y0s = np.linspace((n0 - 1) * layer_h / 2, -(n0 - 1) * layer_h / 2, n0)
        y1s = np.linspace((n1 - 1) * layer_h / 2, -(n1 - 1) * layer_h / 2, n1)
        for y0 in y0s:
            for y1 in y1s:
                ax.plot([x0, x1], [y0, y1], color='k', lw=0.6, zorder=1)

    # Cosmetic tweaks
    ax.set_aspect('equal')
    ax.axis('off')
    # Tighten view limits a bit
    ax.set_xlim(xs[0] - x_pad * 0.4, xs[-1] + x_pad * 0.4)
    max_nodes = max(layer_sizes)
    ax.set_ylim(-max_nodes * layer_h / 2 - layer_h,
                 max_nodes * layer_h / 2 + layer_h)


# ------------------------------------------------------------------
# Network layouts (feel free to tweak the exact sizes)
# ------------------------------------------------------------------
networks = {
    "(a) Right triangle": [6, 4, 2, 1],
    "(b) Left triangle" : [1, 2, 4, 6],
    "(c) Diamond"       : [1, 3, 5 ,3, 1],
    "(d) Block"         : [1, 6, 6, 6, 1],
}

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 7), facecolor="#d3d3d3")
axes = axes.flatten()

for ax, (title, layers) in zip(axes, networks.items()):
    draw_network(ax, layers)
    # Put the label below the panel
    ax.set_title(title, y=-0.12, fontsize=18, fontweight='bold')

plt.tight_layout()
plt.show()
