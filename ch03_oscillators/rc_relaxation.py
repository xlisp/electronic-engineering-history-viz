"""
Chapter 3.2 - 555 timer: an RC relaxation oscillator.

Phenomenon
----------
A capacitor charges through R_a + R_b, gets compared to two thresholds
(1/3 and 2/3 of V_cc), and a comparator-driven flip-flop discharges it
through R_b.  The result is a square wave that dominated 1970s-80s
electronics ("the most popular IC ever made", >1 billion sold per year).

Hans Camenzind designed the 555 at Signetics in 1971 — same year as
Intel's 4004 microprocessor.  Both are still in production.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

torch.set_default_dtype(torch.float64)

# 555 in astable mode
Vcc = 5.0
R_a = 10e3
R_b = 47e3
C   = 100e-9

# Hand-derived periods (just to compare)
T_high = np.log(2.0) * (R_a + R_b) * C
T_low  = np.log(2.0) * R_b * C
period = T_high + T_low
duty   = T_high / period

# Simulate the relaxation directly: state machine with charging / discharging RC
dt = period / 200
N = 4000
t  = torch.arange(N) * dt

vC = torch.zeros(N)
out = torch.zeros(N)
state = "charge"
v = 0.0
o = Vcc
for k in range(N):
    if state == "charge":
        tau = (R_a + R_b) * C
        v_target = Vcc
        v += (v_target - v) * dt / tau
        if v >= 2.0/3.0 * Vcc:
            state = "discharge"; o = 0.0
    else:
        tau = R_b * C
        v_target = 0.0
        v += (v_target - v) * dt / tau
        if v <= 1.0/3.0 * Vcc:
            state = "charge"; o = Vcc
    vC[k] = v
    out[k] = o

# ---------- Plot ----------
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

ax = axes[0]
ax.plot(t.numpy() * 1e3, vC.numpy(), color="#0a3d62", lw=1.5, label="$v_C(t)$")
ax.axhline(2.0/3.0 * Vcc, ls="--", color="#b71540", lw=0.8, label="2 V_cc / 3 (upper threshold)")
ax.axhline(1.0/3.0 * Vcc, ls="--", color="#e58e26", lw=0.8, label="V_cc / 3 (lower threshold)")
ax.set_ylabel("v_C [V]")
ax.set_title(f"555 astable:  R_a={R_a/1e3:.0f}k, R_b={R_b/1e3:.0f}k, C={C*1e9:.0f}nF\n"
             f"Period ≈ {period*1e3:.2f} ms, duty = {duty*100:.1f}%, f ≈ {1.0/period:.0f} Hz")
ax.legend(loc="upper right"); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(t.numpy() * 1e3, out.numpy(), color="#b71540", lw=1.5)
ax.set_xlabel("time [ms]"); ax.set_ylabel("V_out [V]")
ax.set_title("Square-wave output (the 'tick' that drives PWM, blinkers, beepers, …)")
ax.grid(alpha=0.3)
ax.set_ylim(-0.3, Vcc + 0.3)

fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "rc_relaxation.png")
fig.savefig(out_path, dpi=120)
print(f"saved {out_path}")
