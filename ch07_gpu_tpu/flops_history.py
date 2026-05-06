"""
Chapter 7.4 - 1971-2024: peak FLOPS per chip, and what the deep-learning
era did to the curve.

Phenomenon
----------
Until ~2010 the FLOPS-per-chip curve climbed at Moore's-law pace ~1.5x/yr.
Then deep learning hit, demand for matrix-multiply spiked, and the curve
*bent upward*: GPU/TPU peak throughput started doubling every ~7 months
(Sevilla et al. 2022, "Compute Trends Across Three Eras of Machine Learning").

The reason isn't smaller transistors — it's specialization. CPUs spend
~99% of their die area on cache, control logic, branch prediction. AI
accelerators spend it on multiply-accumulate units. Same fab, different
floorplan.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# (year, peak FLOPS, name, type)
chips = [
    (1971, 9.2e4,    "Intel 4004",         "cpu"),     # ~92 kIPS
    (1978, 3.3e5,    "Intel 8086",         "cpu"),
    (1989, 1.5e7,    "Intel i860",         "cpu"),
    (1993, 1e8,      "Pentium",            "cpu"),
    (1999, 6.6e9,    "Pentium III @ 1 GHz","cpu"),
    (2003, 1.2e10,   "Pentium 4 / SSE2",   "cpu"),
    (2010, 1e11,     "Core i7-980X",       "cpu"),
    (2017, 1e12,     "Xeon Platinum 8180", "cpu"),
    (2023, 3e12,     "Xeon Sapphire Rapids","cpu"),

    (2001, 5e9,      "GeForce 3",          "gpu"),
    (2006, 5e11,     "G80 (CUDA)",         "gpu"),
    (2012, 4e12,     "GTX 580 (AlexNet!)", "gpu"),
    (2016, 1.1e13,   "P100",               "gpu"),
    (2020, 3.1e14,   "A100 (TF32)",        "gpu"),
    (2022, 1e15,     "H100 (FP8)",         "gpu"),
    (2024, 2e16,     "Blackwell B200 (FP4)","gpu"),

    (2017, 1.8e14,   "TPU v2",             "tpu"),
    (2021, 2.75e14,  "TPU v4",             "tpu"),
    (2023, 9.18e14,  "TPU v5p",            "tpu"),
]

cpu = [(y, f) for y, f, _, t in chips if t == "cpu"]
gpu = [(y, f) for y, f, _, t in chips if t == "gpu"]
tpu = [(y, f) for y, f, _, t in chips if t == "tpu"]

# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(11, 7))

for series, color, label, marker in [
    (cpu, "#0a3d62", "CPU peak", "o"),
    (gpu, "#b71540", "GPU peak (training/inference precision)", "s"),
    (tpu, "#e58e26", "Google TPU peak", "^"),
]:
    ys = np.array([y for y, _ in series])
    fs = np.array([f for _, f in series])
    ax.semilogy(ys, fs, marker, color=color, ms=10, lw=0, label=label)
    # Linear fit in log space
    slope, intercept = np.polyfit(ys, np.log10(fs), 1)
    yline = np.linspace(ys.min(), 2024, 50)
    ax.semilogy(yline, 10 ** (slope * yline + intercept), "--", color=color, lw=1, alpha=0.7)
    print(f"{label}: doubling every {np.log10(2)/slope:.2f} years")

# Annotations
notable = [
    (2012, 4e12,   "AlexNet — DL revolution starts here"),
    (2017, 1.8e14, "TPU v2 — Google bets on AI silicon"),
    (2024, 2e16,   "Blackwell B200 (FP4)"),
    (1971, 9.2e4,  "Intel 4004"),
]
for y, f, name in notable:
    ax.annotate(name, (y, f), textcoords="offset points", xytext=(8, 8), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.5))

ax.set_xlabel("year"); ax.set_ylabel("peak FLOPS / chip (log scale)")
ax.set_title("Peak floating-point throughput per chip, 1971-2024\n"
             "After 2012 the GPU/TPU curve bent upward — that bend is the deep-learning era")
ax.grid(alpha=0.3, which="both"); ax.legend(loc="lower right")

fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "flops_history.png")
fig.savefig(out_path, dpi=120)
print(f"saved {out_path}")
