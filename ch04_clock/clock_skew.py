"""
Chapter 4.1 - Clock skew: an H-tree vs a serial bus.

Phenomenon
----------
On a modern CPU die, ~30 billion transistors must all see the rising edge
of the clock at *almost the same time* — within picoseconds.  Naively
running a single wire to every flip-flop is hopeless: the wire near the
clock generator gets the edge first, the wire across the chip gets it
hundreds of picoseconds later.

The standard fix is the **H-tree**: a recursively split tree where every
leaf is the same wire-length away from the root.  All leaves see the edge
at the same time (modulo manufacturing variation).

This is one of those quiet engineering inventions that makes GHz CPUs
possible at all.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

torch.set_default_dtype(torch.float64)

rng = np.random.default_rng(0)

# ---------- Generate a 64-leaf grid of flip-flops ----------
n_side = 8                              # 8x8 = 64 flip-flops
xs = np.linspace(0, 10, n_side)
ys = np.linspace(0, 10, n_side)
X, Y = np.meshgrid(xs, ys)
leaves = np.stack([X.ravel(), Y.ravel()], axis=1)   # shape (64, 2)

# Wire delay model: 50 ps per mm
ps_per_mm = 50.0

# (a) Naive bus: clock source at (0,0), wire goes straight from source to each leaf
src = np.array([0.0, 0.0])
delay_bus = np.linalg.norm(leaves - src, axis=1) * ps_per_mm

# (b) H-tree: total wire length from source to every leaf is identical = 2*W + 2*W/2 + ...
# We build the H-tree geometry recursively to draw it; for delay, every leaf is identical (modulo a small mismatch we add)
def htree_segments(center, w, h, depth, segments):
    cx, cy = center
    # horizontal bar
    segments.append(((cx - w/2, cy), (cx + w/2, cy)))
    # two vertical bars at the bar's ends
    segments.append(((cx - w/2, cy - h/2), (cx - w/2, cy + h/2)))
    segments.append(((cx + w/2, cy - h/2), (cx + w/2, cy + h/2)))
    if depth == 0:
        return [(cx - w/2, cy - h/2), (cx + w/2, cy - h/2),
                (cx - w/2, cy + h/2), (cx + w/2, cy + h/2)]
    leaves_pts = []
    for nx, ny in [(cx-w/2, cy-h/2), (cx+w/2, cy-h/2),
                   (cx-w/2, cy+h/2), (cx+w/2, cy+h/2)]:
        leaves_pts += htree_segments((nx, ny), w/2, h/2, depth - 1, segments)
    return leaves_pts

segments = []
htree_leaves = htree_segments((5, 5), 8, 8, depth=2, segments=segments)
htree_leaves = np.array(htree_leaves)
# Ideal H-tree delay = constant; real has process variation
ideal_htree_delay = (8 + 4 + 2) * ps_per_mm   # rough total path length
delay_htree = ideal_htree_delay + rng.normal(0, 5, size=len(htree_leaves))

# ---------- Plot ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
# bus visualization: all leaves as a star from origin
for lx, ly in leaves:
    ax.plot([src[0], lx], [src[1], ly], color="gray", lw=0.5, alpha=0.5)
sc = ax.scatter(leaves[:,0], leaves[:,1], c=delay_bus, cmap="plasma", s=120, edgecolor="black")
ax.scatter([src[0]], [src[1]], s=250, c="red", marker="*", zorder=10, label="clock source")
ax.set_title(f"Naive 'star' clock bus\nskew = {delay_bus.max()-delay_bus.min():.0f} ps  (max - min)")
ax.set_aspect("equal"); ax.legend(loc="upper right")
plt.colorbar(sc, ax=ax, label="delay [ps]")

ax = axes[1]
for (x1, y1), (x2, y2) in segments:
    ax.plot([x1, x2], [y1, y2], color="gray", lw=1.0)
ax.scatter([5], [5], s=250, c="red", marker="*", zorder=10, label="clock source")
sc2 = ax.scatter(htree_leaves[:,0], htree_leaves[:,1], c=delay_htree, cmap="plasma",
                 s=140, edgecolor="black", vmin=delay_htree.min(), vmax=delay_htree.max())
ax.set_title(f"H-tree clock distribution\nskew ≈ {delay_htree.max()-delay_htree.min():.0f} ps  (process noise only)")
ax.set_aspect("equal"); ax.legend(loc="upper right")
plt.colorbar(sc2, ax=ax, label="delay [ps]")

fig.suptitle("Clock skew: same chip, two distribution networks", fontsize=12)
fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "clock_skew.png")
fig.savefig(out_path, dpi=120)
print(f"saved {out_path}")
