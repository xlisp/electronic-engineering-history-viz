"""
Chapter 6.5 - Moore's Law (1965-2024): is it dead yet?

Phenomenon
----------
In 1965 Gordon Moore wrote a 4-page article in *Electronics* magazine
predicting that the number of components per IC would double every year
(later revised to ~every two years).  The "law" turned into a *target*
that the industry organized itself around for 60 years.

The plot below shows real CPUs (Intel, AMD, IBM, Apple, NVIDIA GPUs) over
time.  The line is still climbing, but:
  - Single-thread frequency is flat since ~2005 (the "frequency wall")
  - Single-die transistor counts keep rising via more cores + 3D stacking
  - Energy per op keeps falling (Koomey's law), but slower than before
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# Approximate transistor counts of selected milestones
# (year, transistors, name)
chips = [
    (1971, 2300,        "Intel 4004"),
    (1974, 6000,        "Intel 8080"),
    (1978, 29000,       "Intel 8086"),
    (1982, 134000,      "i286"),
    (1985, 275000,      "i386"),
    (1989, 1180000,     "i486"),
    (1993, 3.1e6,       "Pentium"),
    (1995, 5.5e6,       "Pentium Pro"),
    (1999, 9.5e6,       "Pentium III"),
    (2000, 42e6,        "Pentium 4"),
    (2006, 291e6,       "Core 2 Duo"),
    (2008, 781e6,       "Core i7"),
    (2010, 1.17e9,      "Westmere-EX"),
    (2012, 1.4e9,       "Ivy Bridge-EX"),
    (2014, 5.5e9,       "18-core Haswell"),
    (2016, 7.2e9,       "22-core Broadwell"),
    (2017, 19.2e9,      "32-core Epyc"),
    (2019, 39.5e9,      "AWS Graviton2"),
    (2020, 16e9,        "Apple M1"),
    (2022, 114e9,       "M1 Ultra"),
    (2023, 134e9,       "M2 Ultra"),
    (2023, 80e9,        "NVIDIA H100 GPU"),
    (2024, 208e9,       "NVIDIA Blackwell B200"),
]

years = np.array([c[0] for c in chips])
trans = np.array([c[1] for c in chips])
names = [c[2] for c in chips]

# Fit a line in log space:  log(T) = a*(year-1971) + b
# Doubling every ~2 years means slope = log10(2)/2 ≈ 0.15 per year
log_t = np.log10(trans)
slope, intercept = np.polyfit(years, log_t, 1)
doubling_period = np.log10(2) / slope

# Reference line: 2-year doubling from the 4004
ref_years = np.linspace(years.min(), years.max() + 2, 100)
ref_2yr   = 2300 * 2 ** ((ref_years - 1971) / 2)

# ---------- Plot ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.semilogy(years, trans, "o", ms=8, color="#0a3d62")
ax.semilogy(ref_years, ref_2yr, "--", color="#b71540", lw=1.5,
            label=f"reference: 2× / 2 yr from 4004")
ax.semilogy(ref_years, 10 ** (slope * ref_years + intercept), "-",
            color="#e58e26", lw=1.5, label=f"fit: 2× every {doubling_period:.1f} yr")
for y, t, name in chips:
    if name in ("Intel 4004", "Pentium 4", "M1 Ultra", "NVIDIA Blackwell B200", "Pentium"):
        ax.annotate(name, (y, t), textcoords="offset points", xytext=(8, -8), fontsize=8)
ax.set_xlabel("year"); ax.set_ylabel("transistors per chip")
ax.set_title("Moore's Law (1965 prediction → 2024 reality)")
ax.legend(); ax.grid(alpha=0.3, which="both")

# Frequency stagnation since 2005
freq_chips = [
    (1971, 0.74,   "4004"),
    (1978, 5,      "8086"),
    (1989, 25,     "486"),
    (1993, 60,     "Pentium"),
    (2000, 1500,   "Pentium 4"),
    (2003, 3200,   "P4 Prescott"),
    (2006, 2933,   "Core 2 Duo"),
    (2010, 3460,   "Core i7"),
    (2017, 4500,   "i7-7700K"),
    (2020, 5300,   "i9-10900K"),
    (2024, 6000,   "i9-14900KS"),
]
fy = np.array([c[0] for c in freq_chips])
ff = np.array([c[1] for c in freq_chips])

ax = axes[1]
ax.plot(fy, ff, "o-", color="#b71540", ms=8, lw=2)
ax.set_yscale("log")
ax.set_xlabel("year"); ax.set_ylabel("max single-thread clock [MHz]")
ax.set_title("The frequency wall: clocks stopped scaling around 2005\n"
             "(power density hit ~100 W/cm² — same as a hot plate)")
ax.axvspan(2005, 2024, alpha=0.15, color="#b71540")
ax.text(2014, 100, "single-thread\nfrequency plateau", color="#b71540",
        ha="center", fontsize=10)
ax.grid(alpha=0.3, which="both")

fig.suptitle("Moore's Law is alive (transistor count) but its old companion (clock speed) is not",
             fontsize=12)
fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "moores_law.png")
fig.savefig(out_path, dpi=120)
print(f"saved {out_path}")
