"""
Chapter 4.3 - Setup / hold time: why your synthesis tool yells at you.

Phenomenon
----------
A D flip-flop has a *capture window* around the clock edge: the data input
must be stable from t_setup *before* the edge until t_hold *after* it.
If the data changes inside this window, the flip-flop can metastable —
its output dwells at an indeterminate voltage for an unbounded time
(unbounded *in theory*; unbounded *in practice* up to many nanoseconds).

This is the constraint behind every "static timing analysis" report:

    t_clk_period >= t_clk-q + t_combinational + t_setup + t_skew

If this inequality fails, your design is broken at that frequency.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

torch.set_default_dtype(torch.float64)

# Timing parameters (made up but realistic for a slowish process)
t_period       = 10.0   # ns (100 MHz)
t_setup        = 1.5    # ns
t_hold         = 0.5    # ns
t_clk_to_q     = 1.0    # ns
t_combo_short  = 5.0    # ns  -> meets timing
t_combo_long   = 8.0    # ns  -> setup violation

t = np.linspace(0, 4 * t_period, 4000)

def square_clock(t, period):
    return 0.5 * (np.sign(np.sin(2 * np.pi * t / period)) + 1)

clk = square_clock(t, t_period)

# Find rising clock edges
edges = []
for k in range(1, len(clk)):
    if clk[k] > 0.5 and clk[k-1] <= 0.5:
        edges.append(t[k])
edges = np.array(edges)

# CASE A: short combinational path -- data stable well before setup window
data_A = np.zeros_like(t)
for k, e in enumerate(edges):
    # data flips t_clk_to_q + t_combo_short after the previous edge
    if k == 0: continue
    flip_t = edges[k-1] + t_clk_to_q + t_combo_short
    data_A[t >= flip_t] = (k % 2)
# CASE B: long combinational path -- data flips inside the setup window
data_B = np.zeros_like(t)
for k, e in enumerate(edges):
    if k == 0: continue
    flip_t = edges[k-1] + t_clk_to_q + t_combo_long
    data_B[t >= flip_t] = (k % 2)

# ---------- Plot ----------
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

for ax, data, title, ok in zip(
    axes,
    [data_A, data_B],
    [f"PASS: t_combo={t_combo_short} ns  →  data stable {t_period - t_clk_to_q - t_combo_short - t_setup:.1f} ns before edge",
     f"FAIL: t_combo={t_combo_long} ns  →  data flips inside setup window"],
    [True, False],
):
    ax.plot(t, clk + 2.5, color="#0a3d62", lw=1.4, label="CLK")
    ax.plot(t, data, color="#b71540", lw=1.4, label="D")
    # Highlight setup windows
    for e in edges[1:]:
        ax.axvspan(e - t_setup, e + t_hold, color=("#7bed9f" if ok else "#ff6b6b"), alpha=0.35)
        ax.axvline(e, ls="--", color="black", lw=0.6)
    ax.set_title(title, color=("green" if ok else "red"))
    ax.set_ylim(-0.4, 4.0)
    ax.set_yticks([0, 1, 2.5, 3.5])
    ax.set_yticklabels(["0", "1", "0", "1"])
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
axes[1].set_xlabel("time [ns]")

fig.suptitle("Setup/hold window: D must stay stable inside the green/red band\n"
             "(metastability is what happens when D flips inside red)",
             fontsize=12)
fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "setup_hold_timing.png")
fig.savefig(out_path, dpi=120)
print(f"saved {out_path}")
