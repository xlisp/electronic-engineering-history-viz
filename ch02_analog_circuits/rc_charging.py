"""
Chapter 2.2 - RC charging: the same ODE that runs your phone, your bank
account, and Chernobyl.

Phenomenon
----------
Connect a capacitor to a battery through a resistor.  The capacitor doesn't
charge instantly — it follows an exponential curve set by the time
constant tau = R*C.

  dq/dt = (V_in - q/C) / R

This is THE simplest first-order linear ODE in EE.  It is the same equation
that governs:
    radioactive decay (just flip the sign)
    compound interest in a bank account
    Newton's law of cooling
    drug elimination from the bloodstream
    the membrane voltage of a single neuron

Different physics, same skeleton.  EEs see this everywhere; if you can solve
RC, you have solved a quarter of a physicist's toolkit.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

torch.set_default_dtype(torch.float64)

# ---------- Phenomenon: charge a capacitor through a resistor ----------
V_in = 5.0          # battery voltage
R    = 1e3          # 1 kOhm
C    = 1e-6         # 1 uF
tau  = R * C        # 1 ms time constant

dt   = tau / 200
N    = 2500
t    = torch.arange(N) * dt
q    = torch.zeros(N)

# Forward Euler integration of dq/dt = (V_in - q/C)/R
for k in range(N - 1):
    dqdt = (V_in - q[k] / C) / R
    q[k+1] = q[k] + dqdt * dt

v_cap = q / C
i     = (V_in - v_cap) / R

# ---------- Same ODE, three other contexts ----------
# y' = -k*y  (decay) or y' = k*(target - y)  (charging-like)
T_grid = torch.linspace(0, 5 * tau, N)
decay  = V_in * torch.exp(-T_grid / tau)              # radioactive isotope analog
cool   = 30 + (90 - 30) * torch.exp(-T_grid / tau)    # coffee in 30°C room cooling from 90°C
bank   = 100 * torch.exp(+T_grid / tau / 50)          # 2% annual interest, scaled

# ---------- Plot ----------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

ax = axes[0, 0]
ax.plot(t.numpy() * 1e3, v_cap.numpy(), color="#0a3d62", lw=2, label="$v_C(t)$")
ax.axhline(V_in, ls="--", color="gray", lw=1, label=f"V_in = {V_in} V")
ax.axvline(tau * 1e3, ls=":", color="#b71540", lw=1)
ax.text(tau*1e3 + 0.05, 0.5, f"τ = RC = {tau*1e3:.1f} ms", color="#b71540")
ax.set_xlabel("time [ms]"); ax.set_ylabel("v_C [V]")
ax.set_title("Capacitor voltage: $v_C(t) = V_{in}(1 - e^{-t/\\tau})$")
ax.grid(alpha=0.3); ax.legend(loc="lower right")

ax = axes[0, 1]
ax.plot(t.numpy() * 1e3, i.numpy() * 1e3, color="#b71540", lw=2)
ax.set_xlabel("time [ms]"); ax.set_ylabel("current [mA]")
ax.set_title("Current: peaks at t=0, decays to 0 (capacitor 'fills up')")
ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.plot(T_grid.numpy() * 1e3, decay.numpy(), label="radioactive decay $V e^{-t/\\tau}$", color="#0a3d62")
ax.plot(T_grid.numpy() * 1e3, cool.numpy(),  label="Newton's cooling (30°C ambient)", color="#e58e26")
ax.set_xlabel("time [ms]"); ax.set_ylabel("(arbitrary units)")
ax.set_title("Same ODE shape — physics doesn't care what the variable means")
ax.grid(alpha=0.3); ax.legend()

ax = axes[1, 1]
# Log-linear plot of (V_in - v_cap): should be a straight line
gap = V_in - v_cap
ax.semilogy(t.numpy() * 1e3, gap.numpy(), color="#0a3d62", lw=2)
slope = -1.0 / tau
ax.set_xlabel("time [ms]"); ax.set_ylabel("$V_{in} - v_C$  [V, log scale]")
ax.set_title(f"On a log axis the exponential is a line with slope -1/τ")
ax.grid(alpha=0.3, which="both")

fig.suptitle("RC charging: the universal first-order linear system", fontsize=12)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "rc_charging.png")
fig.savefig(out, dpi=120)
print(f"saved {out}")
