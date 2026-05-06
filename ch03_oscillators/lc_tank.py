"""
Chapter 3.1 - The LC tank: every radio's heartbeat.

Phenomenon
----------
Charge a capacitor, then close a switch connecting it to an inductor:
the energy ping-pongs back and forth at angular frequency

    omega_0 = 1 / sqrt(L C)

This is the same equation as a frictionless mass on a spring.  Marconi's
1901 transatlantic transmitter, your FM radio, and a smartphone's RF
front-end all start from this picture.

Real LC tanks have small resistance, so they ring down — that's the
"damped harmonic oscillator", same form as RLC = mass-spring-damper.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

torch.set_default_dtype(torch.float64)

L = 100e-6
C = 1e-9
R_list = [0.0, 5.0, 30.0]    # different damping levels in ohms
omega0 = 1.0 / np.sqrt(L * C)
T0 = 2 * np.pi / omega0

dt = T0 / 200
N = 6000
t = torch.arange(N) * dt

results = {}
for R in R_list:
    q = torch.tensor(1e-9)
    i = torch.tensor(0.0)
    qs, is_ = [], []
    for _ in range(N):
        # L di/dt + R i + q/C = 0
        di = (-q / (L * C) - (R / L) * i) * dt
        i = i + di
        q = q + i * dt
        qs.append(q.item()); is_.append(i.item())
    results[R] = (np.array(qs), np.array(is_))

# ---------- Plot ----------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

ax = axes[0, 0]
for R, color in zip(R_list, ["#0a3d62", "#b71540", "#e58e26"]):
    qs, _ = results[R]
    ax.plot(t.numpy() * 1e6, qs * 1e9, color=color, lw=1.4, label=f"R = {R:.0f} Ω")
ax.set_xlabel("time [µs]"); ax.set_ylabel("charge q [nC]")
ax.set_title(f"LC ringing  (f₀ = {omega0/(2*np.pi)/1e3:.1f} kHz)")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[0, 1]
qs, is_ = results[0.0]
U_E = qs**2 / (2 * C)
U_B = L * is_**2 / 2
ax.plot(t.numpy() * 1e6, U_E * 1e9, label="$U_E$ in capacitor", color="#3c6382")
ax.plot(t.numpy() * 1e6, U_B * 1e9, label="$U_B$ in inductor",  color="#e58e26")
ax.plot(t.numpy() * 1e6, (U_E + U_B) * 1e9, "--k", lw=1, label="total")
ax.set_xlabel("time [µs]"); ax.set_ylabel("energy [nJ]")
ax.set_title("Energy sloshes: E-field ↔ B-field")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1, 0]
for R, color in zip(R_list, ["#0a3d62", "#b71540", "#e58e26"]):
    qs, is_ = results[R]
    ax.plot(qs * 1e9, is_ * 1e3, color=color, lw=0.8, label=f"R = {R:.0f} Ω")
ax.set_xlabel("q [nC]"); ax.set_ylabel("i [mA]")
ax.set_title("Phase portrait: spiral into origin = energy loss")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 1]
# FFT of the lossless ringdown to recover f0
qs, _ = results[0.0]
spec = np.abs(np.fft.rfft(qs))
freqs = np.fft.rfftfreq(N, d=dt)
ax.plot(freqs / 1e3, spec / spec.max(), color="#0a3d62")
ax.axvline(omega0/(2*np.pi)/1e3, ls="--", color="#b71540", lw=1,
           label=f"theory f₀ = {omega0/(2*np.pi)/1e3:.1f} kHz")
ax.set_xlim(0, omega0/(2*np.pi)/1e3 * 3)
ax.set_xlabel("frequency [kHz]"); ax.set_ylabel("|FFT| (normalized)")
ax.set_title("Spectrum: a single sharp tone at $1/(2\\pi\\sqrt{LC})$")
ax.legend(); ax.grid(alpha=0.3)

fig.suptitle("LC tank: the resonator at the heart of every radio", fontsize=12)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "lc_tank.png")
fig.savefig(out, dpi=120)
print(f"saved {out}")
