"""
Chapter 2.3 - The Shockley diode equation (1949).

Phenomenon
----------
A diode conducts in one direction and blocks in the other — the simplest
nonlinear electronic device, and the door into all of solid-state electronics.
Shockley wrote down the I-V relation while at Bell Labs, two years after the
transistor was demonstrated:

    I = I_s * (exp(q V / (n k T)) - 1)

  - I_s : reverse saturation current  (~10^-12 A for Si)
  - n   : ideality factor (~1 for ideal diode)
  - kT/q : thermal voltage, ~25.85 mV at room temp

The exponential is what makes diodes interesting:  every 60 mV of forward
bias multiplies current by 10x.  This is the slope behind the BJT, behind
log/anti-log amplifiers, and behind why solar cells have an open-circuit
voltage that drifts with temperature.

We compute and plot:
  (a) I-V curves at three temperatures
  (b) the same plot on a semilog scale to see the decade-per-60mV slope
  (c) a half-wave rectifier: feed a 50 Hz sine through a single diode
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

torch.set_default_dtype(torch.float64)

q  = 1.602e-19
kB = 1.381e-23
Is = 1e-12        # 1 pA reverse saturation current
n  = 1.0

V = torch.linspace(-0.5, 0.8, 600)

def shockley(V, T):
    Vt = kB * T / q
    return Is * (torch.exp(V / (n * Vt)) - 1)

I_300 = shockley(V, 300.0)
I_350 = shockley(V, 350.0)
I_250 = shockley(V, 250.0)

# ---------- Half-wave rectifier ----------
# Source: 5 V peak sine, in series with a 1 kOhm load and our diode.
# Solve   v_in = v_diode + i * R  with i = Shockley(v_diode)  via Newton iteration.
R_load = 1e3
fs = 50.0
T = 1.0 / fs
t = torch.linspace(0, 2 * T, 800)
v_in = 5.0 * torch.sin(2 * np.pi * fs * t)

def solve_node(v_src, T_op=300.0):
    Vt = kB * T_op / q
    vd = torch.full_like(v_src, 0.6)
    for _ in range(40):
        i  = Is * (torch.exp(vd / (n * Vt)) - 1)
        f  = v_src - vd - i * R_load
        di_dv = (Is / (n * Vt)) * torch.exp(vd / (n * Vt))
        df_dv = -1 - di_dv * R_load
        vd = vd - f / df_dv
    return vd, Is * (torch.exp(vd / (n * Vt)) - 1)

vd, i_diode = solve_node(v_in)
v_load = i_diode * R_load

# ---------- Plot ----------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

ax = axes[0, 0]
for I, T_, color in [(I_250, 250, "#0a3d62"), (I_300, 300, "#b71540"), (I_350, 350, "#e58e26")]:
    ax.plot(V.numpy(), (I*1e3).numpy(), color=color, lw=2, label=f"T = {T_} K")
ax.set_xlabel("V [V]"); ax.set_ylabel("I [mA]")
ax.set_ylim(-1, 50)
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
ax.set_title("Shockley diode I-V (linear)")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[0, 1]
for I, T_, color in [(I_250, 250, "#0a3d62"), (I_300, 300, "#b71540"), (I_350, 350, "#e58e26")]:
    ax.semilogy(V.numpy(), torch.clamp(I, 1e-15, None).numpy(), color=color, lw=2, label=f"T = {T_} K")
ax.set_xlabel("V [V]"); ax.set_ylabel("I [A], log scale")
ax.set_title("On a log axis: ~one decade per 60 mV  (kT/q ≈ 25.85 mV @ 300 K)")
ax.legend(); ax.grid(alpha=0.3, which="both")

ax = axes[1, 0]
ax.plot(t.numpy() * 1e3, v_in.numpy(), label="$v_{in}$ (5 V peak, 50 Hz)", color="#0a3d62")
ax.plot(t.numpy() * 1e3, v_load.numpy(), label="$v_{load}$ (after diode)", color="#b71540")
ax.set_xlabel("time [ms]"); ax.set_ylabel("voltage [V]")
ax.set_title("Half-wave rectifier: diode + 1 kΩ load")
ax.legend(); ax.grid(alpha=0.3)
ax.axhline(0, color="k", lw=0.5)

ax = axes[1, 1]
ax.plot(v_in.numpy(), v_load.numpy(), color="#0a3d62")
ax.set_xlabel("$v_{in}$ [V]"); ax.set_ylabel("$v_{load}$ [V]")
ax.set_title("Transfer characteristic: diode 'kink' at ~0.6 V")
ax.grid(alpha=0.3); ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)

fig.suptitle("Diode = the first nonlinear element  (Shockley 1949, Bell Labs)", fontsize=12)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "diode_iv.png")
fig.savefig(out, dpi=120)
print(f"saved {out}")
