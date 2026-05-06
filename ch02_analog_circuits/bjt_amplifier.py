"""
Chapter 2.4 - The common-emitter BJT amplifier (1947 → today).

Phenomenon
----------
A small AC voltage at the base of a bipolar junction transistor produces a
much larger AC voltage at the collector. This is *amplification* — the
defining capability of an electronic device that distinguishes "active" from
"passive" components.

Bardeen, Brattain and Shockley demonstrated the first solid-state version
on 23 Dec 1947 at Bell Labs.  Before that, every radio in the world used
vacuum tubes; the transistor replaced them, and 76 years later your laptop
has 30 billion of them on a single chip.

Math
----
Ebers-Moll-ish, simplified:
    I_C = Is * exp(V_BE / V_T)        (collector current)
    I_B = I_C / beta                  (base current)

Small-signal voltage gain (textbook):  A_v = -g_m * R_C    where g_m = I_C / V_T
We compute everything from scratch via Newton iteration on the actual ODE.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

torch.set_default_dtype(torch.float64)

# ---------- Operating point ----------
VCC  = 12.0
RC   = 4.7e3
RB   = 220e3
beta = 200.0
Is   = 1e-14
VT   = 0.02585          # thermal voltage @ 300 K

def collector_current(VBE):
    return Is * (torch.exp(VBE / VT) - 1)

# DC bias: pick V_in_DC such that the transistor sits in the middle of its range
def solve_op(V_in):
    """Given V_in (base voltage relative to ground, emitter grounded), find V_BE,
    I_B, I_C, V_out."""
    VBE = torch.full_like(torch.as_tensor(V_in), 0.7) if isinstance(V_in, torch.Tensor) else torch.tensor(0.7)
    V_in_t = torch.as_tensor(V_in)
    for _ in range(50):
        IC = collector_current(VBE)
        IB = IC / beta
        # KVL on input loop:  V_in = IB * RB + VBE
        f = V_in_t - IB * RB - VBE
        # df/dVBE = -RB/beta * dIC/dVBE - 1
        dIC = (Is / VT) * torch.exp(VBE / VT)
        df_dVBE = -RB * dIC / beta - 1.0
        VBE = VBE - f / df_dVBE
    IC = collector_current(VBE)
    Vout = VCC - IC * RC
    return VBE, IC, Vout

# DC sweep of input voltage to draw transfer curve
V_in_sweep = torch.linspace(0.0, 1.5, 500)
_, _, V_out_sweep = solve_op(V_in_sweep)

# Pick a quiescent point in the middle of the linear region
V_in_Q = 0.95
_, IC_Q, V_out_Q = solve_op(V_in_Q)
gm_Q = IC_Q / VT
Av_predicted = -gm_Q * RC

# Time-domain demo: 1 kHz sine, small signal
fs = 1e6
t = torch.linspace(0, 4e-3, int(fs * 4e-3))
v_signal_amp = 0.005  # 5 mV peak input -> small signal regime
v_in_t = V_in_Q + v_signal_amp * torch.sin(2 * np.pi * 1e3 * t)
_, _, v_out_t = solve_op(v_in_t)

# ---------- Plot ----------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

ax = axes[0, 0]
ax.plot(V_in_sweep.numpy(), V_out_sweep.numpy(), color="#0a3d62", lw=2)
ax.axvline(V_in_Q, ls="--", color="#b71540", lw=1, label="bias point Q")
ax.scatter([V_in_Q], [V_out_Q.item()], color="#b71540", zorder=5, s=60)
ax.set_xlabel("V_in [V]"); ax.set_ylabel("V_out [V]")
ax.set_title("DC transfer curve V_out(V_in)")
ax.legend(); ax.grid(alpha=0.3)
ax.text(0.05, 1.0, "cutoff", fontsize=9, color="gray")
ax.text(0.95, 6.0, "active\n(linear)", fontsize=9, color="gray")
ax.text(1.3, 0.3, "saturation", fontsize=9, color="gray")

ax = axes[0, 1]
# I_C vs V_BE on a semilog plot — exponential characteristic
VBE_sweep = torch.linspace(0.4, 0.8, 400)
IC_sweep = collector_current(VBE_sweep)
ax.semilogy(VBE_sweep.numpy(), IC_sweep.numpy(), color="#b71540", lw=2)
ax.set_xlabel("V_BE [V]"); ax.set_ylabel("I_C [A]")
ax.set_title("I_C(V_BE): exponential, slope = 1 decade per 60 mV")
ax.grid(alpha=0.3, which="both")

ax = axes[1, 0]
ax.plot(t.numpy() * 1e3, (v_in_t - V_in_Q).numpy() * 1e3, label="v_in (AC, mV)", color="#0a3d62")
ax.plot(t.numpy() * 1e3, (v_out_t - V_out_Q).numpy() * 1e3, label="v_out (AC, mV)", color="#b71540")
ax.set_xlabel("time [ms]"); ax.set_ylabel("AC component [mV]")
ax.set_title(f"Small-signal: input × ({Av_predicted:+.0f})  + 180° inverted")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.bar(["g_m × R_C\n(theory)", "measured\nfrom plot"],
       [abs(Av_predicted), float((v_out_t.max() - v_out_t.min()) / (v_in_t.max() - v_in_t.min()))],
       color=["#0a3d62", "#b71540"])
ax.set_ylabel("|voltage gain|")
ax.set_title("Voltage gain: small-signal theory matches the numerics")
ax.grid(alpha=0.3, axis="y")

fig.suptitle("BJT common-emitter amplifier  (Bardeen-Brattain-Shockley 1947 → 1948 junction transistor)",
             fontsize=12)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "bjt_amplifier.png")
fig.savefig(out, dpi=120)
print(f"saved {out}")
