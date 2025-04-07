import numpy as np
import matplotlib.pyplot as plt

# x‑axis range
x = np.linspace(-5, 5, 1000)

# dictionary of activation functions
activations = {
    "Sigmoid":        lambda z: 1 / (1 + np.exp(-z)),
    "Tanh":           np.tanh,
    "ReLU":           lambda z: np.maximum(0, z),
    "Leaky ReLU":     lambda z, α=0.01: np.where(z > 0, z, α * z),
    "ELU":            lambda z, α=1.0: np.where(z > 0, z, α * (np.exp(z) - 1)),
    "Swish":          lambda z, β=1.0: z / (1 + np.exp(-β * z)),
}

# 2 × 3 subplot grid
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
axes = axes.flatten()

for ax, (name, func) in zip(axes, activations.items()):
    y = func(x)                    # compute activation
    ax.plot(x, y, lw=2)
    ax.set_title(name)
    ax.axhline(0, color="black", lw=.5)
    ax.axvline(0, color="black", lw=.5)
    ax.set_xlim(-5, 5)
    ax.grid(True)

fig.suptitle("Common Neural‑Network Activation Functions", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
