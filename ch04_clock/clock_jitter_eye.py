"""
Chapter 4.2 - Eye diagrams: signal integrity at high speed.

Phenomenon
----------
At GHz data rates, the question "is this bit a 1 or a 0?" stops being
trivial.  Wire bandwidth, ringing, jitter and noise all conspire to make
the signal *almost* unrecognizable at the receiver.  The eye diagram
overlays many bit periods on top of each other; if the central "eye"
is open, the receiver can sample reliably; if it's closed, you've lost.

This is the standard sanity-check for every high-speed link engineer
(USB, PCIe, DDR, SerDes, …).
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(1)

# ---------- Generate a random bitstream and pass it through a channel ----------
n_bits   = 1000
samples_per_bit = 32
fs = samples_per_bit  # samples per UI

bits = rng.integers(0, 2, size=n_bits)
nrz  = np.repeat(bits * 2.0 - 1.0, samples_per_bit)         # ±1 NRZ

# Channel: a 1st-order RC low-pass (limited bandwidth) + Gaussian noise + jitter
# Bandwidth limit -> simple IIR
alpha = 0.35   # IIR coeff; higher = more bandwidth
y = np.zeros_like(nrz)
y[0] = nrz[0]
for k in range(1, len(nrz)):
    y[k] = alpha * nrz[k] + (1 - alpha) * y[k-1]
# Add Gaussian noise
y_noisy = y + rng.normal(0, 0.06, size=y.shape)

# Add timing jitter: shift each bit boundary by a small random amount
# (we approximate by resampling: small per-bit time offsets)
jitter_pp = 0.06   # peak-peak jitter as fraction of UI
shifts = rng.normal(0, jitter_pp / 3, size=n_bits)   # one shift per bit
shifted = np.zeros_like(y_noisy)
t_idx = np.arange(len(y_noisy))
for k in range(n_bits):
    seg = slice(k*samples_per_bit, (k+1)*samples_per_bit)
    # interpolate the segment with a small fractional shift
    t_seg = np.linspace(0, 1, samples_per_bit) + shifts[k]
    src   = y_noisy[seg]
    if k > 0:
        # use one previous sample for left context
        prev = y_noisy[k*samples_per_bit - 1]
        ext  = np.concatenate([[prev], src])
        t_ref = np.linspace(-1.0/samples_per_bit, 1, samples_per_bit + 1)
        shifted[seg] = np.interp(t_seg, t_ref, ext)
    else:
        shifted[seg] = src

# ---------- Build the eye diagram by slicing 2-UI windows ----------
window_len = 2 * samples_per_bit
n_windows = (len(shifted) - window_len) // samples_per_bit
eye = np.stack([shifted[k*samples_per_bit : k*samples_per_bit + window_len]
                for k in range(n_windows)])
t_eye = np.linspace(-1, 1, window_len)   # UI on x-axis

# ---------- Plot ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(np.arange(20*samples_per_bit) / samples_per_bit,
        nrz[:20*samples_per_bit], color="#0a3d62", lw=1, label="ideal NRZ")
ax.plot(np.arange(20*samples_per_bit) / samples_per_bit,
        shifted[:20*samples_per_bit], color="#b71540", lw=1, alpha=0.8, label="received (BW + noise + jitter)")
ax.set_xlabel("bit periods (UI)"); ax.set_ylabel("voltage [a.u.]")
ax.set_title("First 20 bits of the link")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
for w in eye:
    ax.plot(t_eye, w, color="#0a3d62", lw=0.3, alpha=0.15)
ax.set_xlabel("time within 2 UI"); ax.set_ylabel("voltage [a.u.]")
ax.set_title(f"Eye diagram: {n_windows} bits overlaid\n(open eye = receiver can sample reliably)")
ax.grid(alpha=0.3)
# Sample point indicator
ax.axvline(0, ls="--", color="#b71540", lw=1)
ax.text(0.02, 0.9, "ideal\nsample\npoint", color="#b71540", fontsize=9)

fig.suptitle("Eye diagram: the high-speed engineer's microscope", fontsize=12)
fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "clock_jitter_eye.png")
fig.savefig(out_path, dpi=120)
print(f"saved {out_path}")
