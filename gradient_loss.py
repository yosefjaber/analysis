import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

# ── Random loss surface ────────────────────────────────────────────────────────
rng = np.random      # remove the argument to get a new surface each run
K   = 4                               # how many sinusoidal components

freqs  = rng.uniform(0.5, 0.6, size=(K, 2))   # (fx, fy) pairs
phases = rng.uniform(0, 0.1*np.pi, size=K)
amps   = rng.uniform(0.4, 0.5, size=K)

def loss(x, y):
    """Random sinusoid mixture + small quadratic to keep things bounded."""
    z = sum(
        a * np.sin(fx * x + p) * np.cos(fy * y + p)
        for (fx, fy), a, p in zip(freqs, amps, phases)
    )
    return z + 0.05 * (x**2 + y**2)

def grad(x, y):
    """Analytic gradient of the above loss."""
    gx = sum(
        a * fx * np.cos(fx * x + p) * np.cos(fy * y + p)
        for (fx, fy), a, p in zip(freqs, amps, phases)
    ) + 0.10 * x
    gy = sum(
        -a * fy * np.sin(fx * x + p) * np.sin(fy * y + p)
        for (fx, fy), a, p in zip(freqs, amps, phases)
    ) + 0.10 * y
    return gx, gy

# ── Gradient‑descent parameters ────────────────────────────────────────────────
lr       = 0.05      # learning rate
n_steps  = 80
# x0, y0   = rng.uniform(-5, 5, size=2)   # random start
x0, y0   = -4.5,4.5

path = [(x0, y0, loss(x0, y0))]
x, y = x0, y0
for _ in range(n_steps):
    gx, gy = grad(x, y)
    x -= lr * gx
    y -= lr * gy
    path.append((x, y, loss(x, y)))

px, py, pz = map(np.array, zip(*path))

# ── Mesh for surface ──────────────────────────────────────────────────────────
grid_lim = 6
X = np.linspace(-grid_lim, grid_lim, 300)
Y = np.linspace(-grid_lim, grid_lim, 300)
X, Y = np.meshgrid(X, Y)
Z = loss(X, Y)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 8))
ax  = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(
    X, Y, Z,
    cmap='viridis',
    alpha=0.8,
    rstride=5, cstride=5,
    linewidth=0, antialiased=True
)

ax.plot(
    px, py, pz,
    color='crimson', marker='o', markersize=4,
    linewidth=2, label='Gradient descent path'
)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Loss')
ax.set_title('Gradient Descent on a Random Loss Surface')
ax.legend()
plt.tight_layout()
plt.show()
