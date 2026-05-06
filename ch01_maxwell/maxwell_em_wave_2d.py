"""
Chapter 1.4 - Maxwell 1865: predicting electromagnetic waves from pure math.

Phenomenon
----------
Maxwell took Faraday's "lines of force" and Ampere's circuital law, added a
single missing term ("displacement current" — added on aesthetic grounds!),
and then proved that the equations admit wave solutions traveling at speed
c = 1/sqrt(mu0 * eps0) = 3 x 10^8 m/s. That is the speed of light.

So light is an electromagnetic wave. (1865 — twenty years before Hertz
confirmed it experimentally.)

Method: 2-D FDTD (Yee 1966)
---------------------------
The Yee grid stores E_z (out-of-page) at cell centers and H_x, H_y at edges.
Updates leapfrog through time:

    H_x(t + 0.5 dt) = H_x(t - 0.5 dt) - dt/(mu0 dy) * (E_z(j+1) - E_z(j))
    H_y(t + 0.5 dt) = H_y(t - 0.5 dt) + dt/(mu0 dx) * (E_z(i+1) - E_z(i))
    E_z(t + dt)     = E_z(t)          + dt/(eps0)   * (dHy/dx - dHx/dy)

This is one of the most consequential numerical schemes ever written:
modern antenna design, MRI, photonic chips, 5G beamforming all live on it.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

torch.set_default_dtype(torch.float64)

# ---------- Constants & grid ----------
c0   = 3e8
mu0  = 4 * np.pi * 1e-7
eps0 = 1.0 / (c0 * c0 * mu0)

Nx, Ny = 220, 220
dx = dy = 1e-3                                # 1 mm cells -> 22 cm box
dt = 0.5 * dx / c0                            # CFL: dt < dx/(c sqrt(2))

# Fields on a Yee grid (we collapse to scalar storage; correctness is fine for the demo)
Ez = torch.zeros((Nx, Ny))
Hx = torch.zeros((Nx, Ny))
Hy = torch.zeros((Nx, Ny))

# Source: a soft Gaussian-modulated sine ("ricker pulse" cousin) at the center
src_i, src_j = Nx // 2, Ny // 2
f0 = 5e9                                      # 5 GHz
t_offset = 5e-10
sigma_t  = 1.5e-10

# A simple dielectric inclusion: a square slab of higher permittivity
eps_r = torch.ones((Nx, Ny))
eps_r[140:170, 60:160] = 4.0                  # eps_r=4 slab (e.g., glass)

# Snapshots
n_steps  = 700
snapshot_steps = [120, 250, 400, n_steps - 1]
snaps = []

for n in range(n_steps):
    t_now = n * dt
    # H update
    Hx[:, :-1] = Hx[:, :-1] - (dt / (mu0 * dy)) * (Ez[:, 1:] - Ez[:, :-1])
    Hy[:-1, :] = Hy[:-1, :] + (dt / (mu0 * dx)) * (Ez[1:, :] - Ez[:-1, :])
    # E update
    curlH = (Hy[1:, 1:] - Hy[:-1, 1:]) / dx - (Hx[1:, 1:] - Hx[1:, :-1]) / dy
    Ez[1:, 1:] = Ez[1:, 1:] + (dt / (eps0 * eps_r[1:, 1:])) * curlH
    # Source
    src_val = np.exp(-((t_now - t_offset) ** 2) / (sigma_t ** 2)) * np.sin(2 * np.pi * f0 * t_now)
    Ez[src_i, src_j] = Ez[src_i, src_j] + src_val
    # Crude absorbing boundary: zero out the outermost rim each step (Mur 1981 lite)
    Ez[0, :] = 0; Ez[-1, :] = 0; Ez[:, 0] = 0; Ez[:, -1] = 0

    if n in snapshot_steps:
        snaps.append((n, Ez.clone()))

# ---------- Plot ----------
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
vmax = max(s[1].abs().max().item() for s in snaps) * 0.8
for ax, (n, snap) in zip(axes, snaps):
    im = ax.imshow(snap.numpy().T, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   origin="lower", extent=[0, Nx*dx*100, 0, Ny*dy*100])
    # Draw the dielectric slab outline
    ax.add_patch(plt.Rectangle((140*dx*100, 60*dy*100),
                               (170-140)*dx*100, (160-60)*dy*100,
                               fill=False, edgecolor="black", lw=1.2, ls="--"))
    ax.set_title(f"t = {n*dt*1e9:.2f} ns  (step {n})")
    ax.set_xlabel("x [cm]"); ax.set_ylabel("y [cm]")
fig.suptitle(
    "Maxwell's equations, integrated by Yee 1966 FDTD\n"
    "Source = 5 GHz pulse at center; dashed box = εr=4 dielectric slab (waves slow + reflect)",
    fontsize=12,
)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "maxwell_em_wave_2d.png")
fig.savefig(out, dpi=110)
print(f"saved {out}")
