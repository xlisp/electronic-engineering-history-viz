"""
Chapter 0 - Field vs Circuit: Two ways of looking at the same LC oscillator.

Historical context
------------------
1845: Kirchhoff proved that when circuit dimensions << wavelength, Maxwell's
       equations reduce to two algebraic relations: KCL and KVL.
       This is what makes "circuit diagrams" possible at all — the circuit
       is a low-frequency approximation of the electromagnetic field.

Phenomenon
----------
An ideal LC circuit oscillates: charge sloshes between the capacitor (electric
field energy) and the inductor (magnetic field energy) at frequency
omega = 1/sqrt(L*C).

We compute the same oscillation two ways:
  (a) Circuit view (KVL): a 2nd-order ODE for q(t) — what every EE student knows.
  (b) Field view: track total electric-field energy in C and magnetic-field
      energy in L over time — what Maxwell would have written.
Both produce identical waveforms because we are inside the regime where the
circuit approximation holds.

When the approximation breaks
-----------------------------
Circuit theory ignores radiation resistance. We add a tiny radiation-loss
term and watch energy bleed away — that's the field reaching out beyond the
loop and never coming back. The radiated component is exactly what made
Hertz's 1888 experiment possible: an antenna is a circuit with deliberately
bad lumped-parameter approximation.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

# ---------- Setup ----------
L = 1e-3        # 1 mH
C = 1e-9        # 1 nF
q0 = 1e-9       # initial charge on capacitor (1 nC)

omega = 1.0 / np.sqrt(L * C)
T = 2 * np.pi / omega
dt = T / 500.0
n_steps = 2000

# ---------- (a) Circuit view: KVL gives  L * d^2 q/dt^2 + q/C = 0 ----------
# We integrate as a 1st-order system using leapfrog, no SciPy needed.
q = torch.tensor(q0)
i = torch.tensor(0.0)             # current = dq/dt
q_hist, i_hist = [], []
for _ in range(n_steps):
    # KVL:  L*di/dt = -q/C   =>   di/dt = -q/(L*C)
    i = i + (-q / (L * C)) * dt
    q = q + i * dt
    q_hist.append(q.item())
    i_hist.append(i.item())

q_hist = np.array(q_hist)
i_hist = np.array(i_hist)
t = np.arange(n_steps) * dt

# ---------- (b) Field view: track energy in each field ----------
# Electric field energy in C:  U_E = q^2 / (2C)
# Magnetic field energy in L:  U_B = L * i^2 / 2
U_E = q_hist**2 / (2 * C)
U_B = L * i_hist**2 / 2
U_total = U_E + U_B

# ---------- (c) Lossy variant: add a small radiation resistance R ----------
# This is what circuit theory cannot directly model from first principles —
# it has to be inserted by hand because the field is leaking into space.
R = 0.5  # ohms (small)
q2 = torch.tensor(q0)
i2 = torch.tensor(0.0)
q2_hist, i2_hist = [], []
for _ in range(n_steps):
    # L*di/dt + R*i + q/C = 0
    di = (-q2 / (L * C) - (R / L) * i2) * dt
    i2 = i2 + di
    q2 = q2 + i2 * dt
    q2_hist.append(q2.item())
    i2_hist.append(i2.item())

q2_hist = np.array(q2_hist)
i2_hist = np.array(i2_hist)
U_total_lossy = q2_hist**2 / (2 * C) + L * i2_hist**2 / 2

# ---------- Plot ----------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

ax = axes[0, 0]
ax.plot(t * 1e6, q_hist * 1e9, label="charge q(t) [nC]", color="#0a3d62")
ax.plot(t * 1e6, i_hist * 1e3, label="current i(t) [mA]", color="#b71540", alpha=0.8)
ax.set_xlabel("time [µs]")
ax.set_title("Circuit view (KVL): q and i oscillate 90° out of phase")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(t * 1e6, U_E * 1e12, label="$U_E = q^2/(2C)$  (capacitor electric field)", color="#3c6382")
ax.plot(t * 1e6, U_B * 1e12, label="$U_B = Li^2/2$  (inductor magnetic field)", color="#e58e26")
ax.plot(t * 1e6, U_total * 1e12, "--k", label="total (conserved)", lw=1)
ax.set_xlabel("time [µs]")
ax.set_ylabel("energy [pJ]")
ax.set_title("Field view: energy sloshes between E-field and B-field")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.plot(q_hist * 1e9, i_hist * 1e3, color="#0a3d62", lw=0.7)
ax.set_xlabel("q [nC]")
ax.set_ylabel("i [mA]")
ax.set_title("Phase portrait: closed ellipse = energy conservation")
ax.grid(alpha=0.3)
ax.set_aspect("auto")

ax = axes[1, 1]
ax.plot(t * 1e6, U_total * 1e12, label="lossless: circuit picture is exact", color="#0a3d62")
ax.plot(t * 1e6, U_total_lossy * 1e12, label="with radiation R: energy leaks (Hertz 1888)", color="#b71540")
ax.set_xlabel("time [µs]")
ax.set_ylabel("total energy [pJ]")
ax.set_title("When the field leaks out, the circuit picture starts to crack")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle(
    "LC oscillator: same physics, two languages\n"
    f"L = {L*1e3:.0f} mH, C = {C*1e9:.0f} nF, f = {omega/(2*np.pi)/1e3:.1f} kHz",
    fontsize=12,
)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "field_vs_circuit.png")
fig.savefig(out, dpi=120)
print(f"saved {out}")
