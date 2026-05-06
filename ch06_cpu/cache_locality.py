"""
Chapter 6.4 - Cache locality: why row-major beats column-major 10x.

Phenomenon
----------
DRAM is ~100x slower than the CPU.  A modern CPU survives that gap with
3-4 levels of on-die SRAM cache, exploiting *locality of reference*:
  - temporal locality: data just used will be used again
  - spatial locality: data near recently used data will be used soon

The classic demo: traverse a 2-D matrix two ways.  Row-major access reads
contiguous bytes (cache-line friendly).  Column-major access strides
through memory and trashes the cache.  Same number of memory operations,
~10x time difference.

We measure this using PyTorch tensors (which are row-major / C-order in
memory by default).
"""
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

torch.set_default_dtype(torch.float32)

# ---------- Setup ----------
sizes = [128, 256, 512, 1024, 2048]
trials = 2

results = {"row": [], "col": [], "size": []}
for N in sizes:
    A = torch.rand(N, N)
    # Sum row-by-row (good locality)
    best = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        s = 0.0
        for i in range(N):
            s += A[i, :].sum().item()
        best = min(best, time.perf_counter() - t0)
    results["row"].append(best)
    # Sum col-by-col (bad locality)
    best = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        s = 0.0
        for j in range(N):
            s += A[:, j].sum().item()
        best = min(best, time.perf_counter() - t0)
    results["col"].append(best)
    results["size"].append(N)
    print(f"N={N}: row={results['row'][-1]*1000:.1f} ms, col={results['col'][-1]*1000:.1f} ms")

# ---------- Plot ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(sizes, np.array(results["row"]) * 1000, "o-", color="#0a3d62", label="row-major (sum row by row)")
ax.plot(sizes, np.array(results["col"]) * 1000, "s-", color="#b71540", label="column-major (sum col by col)")
ax.set_xlabel("matrix side N"); ax.set_ylabel("time [ms]")
ax.set_title("Row vs column access on an N×N tensor")
ax.legend(); ax.grid(alpha=0.3)
ax.set_xscale("log", base=2)

ax = axes[1]
ratio = np.array(results["col"]) / np.array(results["row"])
ax.plot(sizes, ratio, "o-", color="#e58e26", lw=2, ms=8)
ax.axhline(1, ls="--", color="gray", lw=1, label="ratio = 1 (no cache effect)")
ax.set_xlabel("matrix side N"); ax.set_ylabel("col-time / row-time")
ax.set_title("Slowdown ratio: cache misses dominate at large N")
ax.set_xscale("log", base=2)
ax.legend(); ax.grid(alpha=0.3)

fig.suptitle("Cache locality: same arithmetic, different memory pattern\n"
             "(temporal + spatial locality is what makes the CPU-DRAM gap survivable)",
             fontsize=12)
fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "cache_locality.png")
fig.savefig(out_path, dpi=120)
print(f"saved {out_path}")
