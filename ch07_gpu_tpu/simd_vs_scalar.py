"""
Chapter 7.1 - SIMD vs scalar: why one Python `+` beats a thousand.

Phenomenon
----------
Modern CPUs have wide SIMD registers (Intel AVX-512: 512 bits = 16 floats
in flight per instruction).  GPUs take this idea to an extreme: thousands
of arithmetic units running the same instruction on different data.

You see this every time you swap a Python `for` loop for a NumPy or
PyTorch vectorized expression.  We benchmark four versions of the same
operation: y = a*x + b (the famous "axpy" kernel from BLAS).
"""
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

torch.set_default_dtype(torch.float32)

sizes = [10_000, 100_000, 1_000_000, 10_000_000]
trials = 3

results = {"python": [], "numpy": [], "torch_cpu": [], "torch_gpu": []}
gpu_available = torch.cuda.is_available()

a, b = 2.0, 3.0
for N in sizes:
    x_list = list(np.random.rand(N))
    x_np   = np.array(x_list, dtype=np.float32)
    x_t    = torch.from_numpy(x_np)
    x_gpu  = x_t.cuda() if gpu_available else None

    # 1) Pure Python loop
    if N <= 1_000_000:
        best = float("inf")
        for _ in range(trials):
            t0 = time.perf_counter()
            y = [a * v + b for v in x_list]
            best = min(best, time.perf_counter() - t0)
        results["python"].append(best)
    else:
        results["python"].append(np.nan)   # too slow

    # 2) NumPy
    best = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        y = a * x_np + b
        best = min(best, time.perf_counter() - t0)
    results["numpy"].append(best)

    # 3) PyTorch CPU
    best = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        y = a * x_t + b
        best = min(best, time.perf_counter() - t0)
    results["torch_cpu"].append(best)

    # 4) PyTorch GPU
    if gpu_available:
        torch.cuda.synchronize()
        best = float("inf")
        for _ in range(trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            y = a * x_gpu + b
            torch.cuda.synchronize()
            best = min(best, time.perf_counter() - t0)
        results["torch_gpu"].append(best)
    else:
        results["torch_gpu"].append(np.nan)

    print(f"N={N}: " + ", ".join(f"{k}={v*1000:.2f} ms" for k, v in
                                  zip(results.keys(), [results[k][-1] for k in results])))

# ---------- Plot ----------
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
labels = {"python": "pure Python loop", "numpy": "NumPy",
          "torch_cpu": "PyTorch CPU (SIMD)",
          "torch_gpu": "PyTorch GPU (CUDA)" if gpu_available else "PyTorch GPU (n/a here)"}
colors = {"python": "#b71540", "numpy": "#e58e26",
          "torch_cpu": "#0a3d62", "torch_gpu": "#3c6382"}
for k, v in results.items():
    arr = np.array(v, dtype=float)
    if np.all(np.isnan(arr)):
        continue
    ax.loglog(sizes, arr * 1000, "o-", color=colors[k], label=labels[k], lw=2, ms=8)
ax.set_xlabel("N (vector length)"); ax.set_ylabel("time [ms, log scale]")
ax.set_title("y = a*x + b   on N elements\n(SIMD = same instruction, multiple data — "
             "the parallel side of the silicon)")
ax.grid(alpha=0.3, which="both"); ax.legend()

fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "simd_vs_scalar.png")
fig.savefig(out_path, dpi=120)
print(f"saved {out_path}")
