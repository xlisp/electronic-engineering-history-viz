"""
Chapter 1.3 - Faraday's law (1831): a changing magnetic flux induces an EMF.

Phenomenon
----------
Faraday's actual 1831 experiment: a bar magnet plunges through a coil of
wire, and a galvanometer needle deflects.  No battery, no chemical source —
the *motion* of the magnet is the energy supply.

  EMF = -d(Phi)/dt        with Phi = integral of B over the coil area

This single equation is the parent of every transformer, every motor, every
generator on the planet, and (through Maxwell's symmetric extension) of the
electromagnetic wave itself.

Setup
-----
We model the magnet as a point dipole moving along the coil axis.  The
flux through the coil is computed by integrating B_z over a disk; the EMF
is the time derivative of that flux, computed via PyTorch autograd.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

torch.set_default_dtype(torch.float64)

# ---------- Geometry ----------
coil_radius = 0.05         # 5 cm
coil_z      = 0.0
N_turns     = 200          # solenoid wraps

# Magnet trajectory:  z(t) = z0 + v t   (passing through coil at t=0)
v = 0.5                    # m/s
z0 = -0.3
t = torch.linspace(-1.0, 1.0, 1500)   # 2 seconds

# Magnetic dipole moment along z
m_dip = 1.0                           # A m^2

# B_z on axis of a dipole at distance d:  B_z = (mu0/(4 pi)) * 2m / d^3
# but the coil has nonzero radius, so we integrate over the disk.
# We sample the disk with a polar grid:
n_r, n_th = 30, 36
rr = torch.linspace(0.001, coil_radius, n_r)
th = torch.linspace(0, 2 * np.pi, n_th)
R, TH = torch.meshgrid(rr, th, indexing="ij")
dx = R * torch.cos(TH)
dy = R * torch.sin(TH)
dA = (coil_radius / n_r) * (2 * np.pi / n_th) * R   # area element

mu0 = 4 * np.pi * 1e-7

def flux_through_coil(z_magnet):
    # distance from dipole (at (0,0,z_magnet)) to point (dx,dy,0) on the coil disk
    rho2 = dx * dx + dy * dy
    dz = -z_magnet
    r2 = rho2 + dz * dz
    r5 = r2 ** 2.5
    # On-axis-ish dipole field, B_z component:
    #   B_z = (mu0/(4 pi)) * (3 dz^2 - r^2) m / r^5
    Bz = (mu0 / (4 * np.pi)) * (3 * dz * dz - r2) * m_dip / r5
    Phi = (Bz * dA).sum()
    return Phi * N_turns   # total flux linkage of N turns

# Compute Phi(t)
zm_t = z0 + v * t
Phi_t = torch.stack([flux_through_coil(z) for z in zm_t])

# EMF = -dPhi/dt  (use autograd to be cute, but a finite difference is simpler here)
emf_t = -torch.gradient(Phi_t, spacing=(t[1] - t[0]).item())[0]

# Current with a 10-ohm load (just to make the picture more concrete)
R_load = 10.0
i_t = emf_t / R_load

# ---------- Plot ----------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

ax = axes[0, 0]
ax.plot(t.numpy(), zm_t.numpy() * 100, color="#0a3d62")
ax.axhspan(coil_z * 100 - 1, coil_z * 100 + 1, color="#b71540", alpha=0.25, label="coil position")
ax.set_xlabel("time [s]"); ax.set_ylabel("magnet position z [cm]")
ax.set_title("Magnet plunging through the coil")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(t.numpy(), Phi_t.numpy() * 1e3, color="#0a3d62")
ax.set_xlabel("time [s]"); ax.set_ylabel("flux linkage Nφ [mWb]")
ax.set_title("Magnetic flux through the coil")
ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.plot(t.numpy(), emf_t.numpy() * 1e3, color="#b71540")
ax.set_xlabel("time [s]"); ax.set_ylabel("EMF [mV]")
ax.set_title("Induced EMF = -dΦ/dt   (Faraday 1831)")
ax.grid(alpha=0.3)
ax.axhline(0, color="black", lw=0.5)

ax = axes[1, 1]
ax.plot(t.numpy(), i_t.numpy() * 1e3, color="#e58e26")
ax.set_xlabel("time [s]"); ax.set_ylabel("current [mA]")
ax.set_title(f"Current in {R_load:.0f} Ω load (the deflecting galvanometer)")
ax.grid(alpha=0.3)
ax.axhline(0, color="black", lw=0.5)

fig.suptitle("Faraday 1831: motion of a magnet → flux change → induced current\n"
             "(this single experiment is the ancestor of every generator and transformer)",
             fontsize=12)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "faraday_induction.png")
fig.savefig(out, dpi=120)
print(f"saved {out}")
