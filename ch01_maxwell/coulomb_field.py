"""
Chapter 1.1 - Coulomb's law (1785): the inverse-square electric field.

Phenomenon
----------
Two charges repel or attract along the line joining them, with a force
that falls off as 1/r^2. Coulomb measured this with a torsion balance —
the same instrument Cavendish used for gravity 13 years later.

  F = k * q1 * q2 / r^2

The 1/r^2 is not arbitrary: it is the geometric signature of three-dimensional
space. The flux through a sphere of radius r must be conserved, and a sphere's
area is 4*pi*r^2 — so the field intensity must dilute as 1/r^2.

We compute and visualize the electric field from a configuration of point
charges by superposition (a key idea — Maxwell's equations are linear).
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

torch.set_default_dtype(torch.float32)

# Charges:  (x, y, q)
charges = torch.tensor([
    [-0.5,  0.0, +1.0],   # +q
    [+0.5,  0.0, -1.0],   # -q   --> dipole
    [ 0.0, +1.2, +0.5],   # extra positive far above
])

# Grid of evaluation points
n = 200
xs = torch.linspace(-2, 2, n)
ys = torch.linspace(-2, 2, n)
X, Y = torch.meshgrid(xs, ys, indexing="xy")

# Vectorized superposition of point-charge fields:  E = k q (r - r_q) / |r - r_q|^3
Ex = torch.zeros_like(X)
Ey = torch.zeros_like(Y)
phi = torch.zeros_like(X)  # scalar potential (for the contour plot)
k = 1.0
for cx, cy, q in charges:
    dx = X - cx
    dy = Y - cy
    r2 = dx * dx + dy * dy + 1e-3   # softened to avoid singularity at the charges
    r = torch.sqrt(r2)
    Ex = Ex + k * q * dx / r2 / r
    Ey = Ey + k * q * dy / r2 / r
    phi = phi + k * q / r

E_mag = torch.sqrt(Ex * Ex + Ey * Ey)

# ---------- Plot ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# (a) Streamlines + charges
ax = axes[0]
ax.streamplot(
    X.numpy(), Y.numpy(), Ex.numpy(), Ey.numpy(),
    color=torch.log(E_mag + 1e-2).numpy(), cmap="viridis", density=1.6, linewidth=0.8,
)
for cx, cy, q in charges:
    ax.scatter(cx, cy, s=300, c=("#b71540" if q > 0 else "#0a3d62"),
               edgecolors="white", linewidths=2, zorder=5)
    ax.annotate(f"{'+' if q>0 else ''}{q:.1f}", (cx, cy),
                color="white", ha="center", va="center", fontsize=10, weight="bold", zorder=6)
ax.set_title("E-field streamlines  (Faraday's lines of force, 1830s)")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_aspect("equal")

# (b) Equipotential contours — what 19th century physicists actually drew
ax = axes[1]
phi_clipped = torch.clamp(phi, -5, 5)
levels = np.linspace(-3, 3, 25)
cs = ax.contour(X.numpy(), Y.numpy(), phi_clipped.numpy(), levels=levels, cmap="RdBu_r")
ax.contourf(X.numpy(), Y.numpy(), phi_clipped.numpy(), levels=levels, cmap="RdBu_r", alpha=0.4)
for cx, cy, q in charges:
    ax.scatter(cx, cy, s=300, c=("#b71540" if q > 0 else "#0a3d62"),
               edgecolors="white", linewidths=2, zorder=5)
ax.set_title("Equipotential contours $\\phi(x,y)$  (Green's function 1828)")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_aspect("equal")

fig.suptitle("Coulomb 1785: superposition of point-charge fields  ($1/r^2$ = 3-D geometry)", fontsize=12)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "coulomb_field.png")
fig.savefig(out, dpi=120)
print(f"saved {out}")
