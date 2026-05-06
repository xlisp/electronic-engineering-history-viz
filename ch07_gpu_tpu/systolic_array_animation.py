"""
Chapter 7.3 - The systolic array: how Google's TPU computes 65,536 MACs in parallel.

Phenomenon
----------
H.T. Kung & Charles Leiserson published "Systolic Arrays for VLSI" in 1978.
The idea: arrange a grid of simple processing elements (PEs), each
multiplying-and-accumulating; the data flows through them like blood
through a heart (hence "systolic").  Result: a matrix multiply
in O(n) time on n^2 PEs, with each datum read from memory exactly once.

40 years later, Google built the TPU v1 (2015, deployed 2016) around a
256x256 systolic array of int8 multiply-accumulators — the same idea Kung
described, scaled up.  This is the central engine of every Transformer
inference at Google.

We compute C = A x B (3x3) on a 3x3 systolic array, snapshot every cycle,
verify the result against torch.matmul, and visualize the data wavefronts.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

torch.set_default_dtype(torch.float32)

# ---------- 3x3 matrices ----------
N = 3
A = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.],
                  [7., 8., 9.]])
B = torch.tensor([[1., 0., 1.],
                  [0., 1., 1.],
                  [1., 1., 0.]])

# ---------- Systolic schedule ----------
# A streams in from the left, one row at a time, with a staggered delay so that
# row i enters i cycles late.  B streams in from the top, one column at a time,
# similarly staggered.  Each PE accumulates a*b every cycle.
n_cycles = 3 * N - 2
A_in = torch.zeros((N, n_cycles))   # A_in[i, t] = element fed into row i at cycle t
B_in = torch.zeros((N, n_cycles))   # B_in[j, t] = element fed into column j at cycle t

for i in range(N):
    for k in range(N):
        A_in[i, i + k] = A[i, k]
for j in range(N):
    for k in range(N):
        B_in[j, j + k] = B[k, j]

# ---------- Simulate cycle by cycle ----------
# PEs hold (a_local, b_local, accum)
# At each cycle: each PE accumulates a*b, then propagates a to the right and b down.
PE = np.zeros((N, N))                # accumulators
A_grid = np.zeros((N, N))            # current 'a' value passing through PE
B_grid = np.zeros((N, N))
snapshots = []
for t in range(n_cycles):
    # Insert new inputs at column 0 / row 0 boundary
    A_in_col = A_in[:, t].numpy()    # length N, one per row
    B_in_row = B_in[:, t].numpy()    # length N, one per column

    # Snapshot of current state BEFORE this cycle's compute
    snapshots.append({"PE": PE.copy(), "A": A_grid.copy(), "B": B_grid.copy(), "t": t})

    # Compute: each PE multiplies its current a*b and adds to accumulator
    PE = PE + A_grid * B_grid

    # Shift A grid right by one (data leaves the right edge), B grid down by one
    new_A = np.zeros_like(A_grid)
    new_B = np.zeros_like(B_grid)
    new_A[:, 1:] = A_grid[:, :-1]
    new_B[1:, :] = B_grid[:-1, :]
    # Inject new boundary inputs
    new_A[:, 0] = A_in_col
    new_B[0, :] = B_in_row
    A_grid = new_A
    B_grid = new_B

snapshots.append({"PE": PE.copy(), "A": A_grid.copy(), "B": B_grid.copy(), "t": n_cycles})

# Verify against torch.matmul
C_ref = (A @ B).numpy()
print("systolic C =\n", PE)
print("torch.matmul C =\n", C_ref)
assert np.allclose(PE, C_ref), "systolic result should match torch.matmul"

# ---------- Plot ----------
n_show = min(7, len(snapshots))
fig, axes = plt.subplots(2, n_show, figsize=(2.6 * n_show, 6))
for col, snap in enumerate(snapshots[:n_show]):
    ax = axes[0, col]
    ax.set_title(f"cycle {snap['t']}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.5, N + 0.5); ax.set_ylim(-0.5, N + 0.5)
    ax.invert_yaxis()
    # Draw the grid of PEs
    for i in range(N):
        for j in range(N):
            ax.add_patch(plt.Rectangle((j - 0.4, i - 0.4), 0.8, 0.8,
                                       fill=False, edgecolor="black"))
            ax.text(j, i - 0.1, f"{snap['PE'][i,j]:.0f}", ha="center", va="center",
                    fontsize=10, color="#0a3d62", weight="bold")
            if abs(snap["A"][i, j]) > 1e-6:
                ax.text(j - 0.25, i + 0.2, f"a={snap['A'][i,j]:.0f}", ha="center", va="center",
                        fontsize=7, color="#b71540")
            if abs(snap["B"][i, j]) > 1e-6:
                ax.text(j + 0.25, i + 0.2, f"b={snap['B'][i,j]:.0f}", ha="center", va="center",
                        fontsize=7, color="#e58e26")
    if col == 0:
        ax.set_ylabel("3×3 PE array\n(blue = accum)")

# Bottom row: heatmaps of the accumulator state (more visual)
for col, snap in enumerate(snapshots[:n_show]):
    ax = axes[1, col]
    im = ax.imshow(snap["PE"], cmap="viridis", vmin=0, vmax=PE.max())
    for i in range(N):
        for j in range(N):
            ax.text(j, i, f"{snap['PE'][i,j]:.0f}", ha="center", va="center",
                    color="white" if snap["PE"][i,j] < PE.max()/2 else "black", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    if col == 0:
        ax.set_ylabel("accumulator")
    ax.set_title(f"cycle {snap['t']}", fontsize=9)

fig.suptitle("Systolic array computing C = A × B  (Kung & Leiserson 1978 → Google TPU v1 2015)\n"
             "data wavefronts: A flows left → right, B flows top → bottom,\n"
             "each PE accumulates a*b once per cycle",
             fontsize=11)
fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "systolic_array_animation.png")
fig.savefig(out_path, dpi=120)
print(f"saved {out_path}")
