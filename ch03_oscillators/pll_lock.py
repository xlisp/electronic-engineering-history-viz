"""
Chapter 3.4 - PLL (Phase-Locked Loop): how a chip locks to an external clock.

Phenomenon
----------
A PLL is a feedback control system whose state variable is *phase*, not
amplitude.  It compares the phase of an input reference signal with the
phase of an internal voltage-controlled oscillator (VCO), and tweaks the
VCO until the two phases match.  Once 'locked', the VCO output tracks the
reference frequency exactly, but with much cleaner edges.

PLLs were invented by Henri de Bellescize in 1932 (for synchronizing radio
receivers).  They are now in *everything*: every CPU multiplies a slow
crystal reference up to GHz, every SerDes recovers clock from data with one,
every cell phone uses them for frequency synthesis.

Math (linearized loop, Type-II)
-------------------------------
    d(theta_VCO)/dt = K_VCO * v_ctrl
    v_ctrl(t)       = K_PD * loop_filter( theta_ref - theta_VCO )

We integrate this directly and watch the error settle.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

torch.set_default_dtype(torch.float64)

# ---------- Parameters ----------
f_ref = 10e6           # 10 MHz reference (e.g. crystal)
f_vco_init = 9.5e6     # VCO starts a bit off
K_VCO = 2 * np.pi * 1e6   # rad/s per volt
K_PD  = 1.0
# PI loop filter: tau1, tau2
tau1 = 1e-5
tau2 = 1e-6

dt = 1e-9
N  = 25000
t  = torch.arange(N) * dt

theta_ref = torch.zeros(N)
theta_vco = torch.zeros(N)
v_ctrl    = torch.zeros(N)
integ     = 0.0

# at step ~ N/2 we kick the reference frequency to test pull-in
omega_ref = 2 * np.pi * f_ref
omega_vco_free = 2 * np.pi * f_vco_init

th_r = 0.0
th_v = 0.0
for k in range(N):
    if k > N // 2:
        omega_ref_now = 2 * np.pi * (f_ref + 0.2e6)   # ref jumps by 200 kHz at t = N/2
    else:
        omega_ref_now = omega_ref
    th_r += omega_ref_now * dt
    # phase detector: sin of difference (mixer-style)
    pd = K_PD * np.sin(th_r - th_v)
    # PI filter:  v = pd + (1/tau1) * integral(pd)
    integ += pd * dt
    v = pd * (tau2 / tau1) + integ / tau1
    # VCO
    th_v += (omega_vco_free + K_VCO * v) * dt

    theta_ref[k] = th_r
    theta_vco[k] = th_v
    v_ctrl[k]    = v

phase_err = (theta_ref - theta_vco).numpy()
# wrap to [-pi, pi] for plotting clarity
phase_err_wrapped = ((phase_err + np.pi) % (2 * np.pi)) - np.pi

# Instantaneous frequencies (numerical derivative)
f_inst_vco = np.gradient(theta_vco.numpy(), dt) / (2 * np.pi)
f_inst_ref = np.gradient(theta_ref.numpy(), dt) / (2 * np.pi)

# ---------- Plot ----------
fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

ax = axes[0]
ax.plot(t.numpy() * 1e6, f_inst_ref / 1e6, label="reference f_ref(t)", color="#0a3d62")
ax.plot(t.numpy() * 1e6, f_inst_vco / 1e6, label="VCO output f_vco(t)", color="#b71540")
ax.set_ylabel("frequency [MHz]")
ax.set_title("PLL acquisition + tracking through a reference jump")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(t.numpy() * 1e6, phase_err_wrapped, color="#e58e26")
ax.set_ylabel("phase error [rad]")
ax.set_title("Phase error θ_ref − θ_vco  (locked when this stays near zero)")
ax.grid(alpha=0.3); ax.axhline(0, color="black", lw=0.5)

ax = axes[2]
ax.plot(t.numpy() * 1e6, v_ctrl.numpy(), color="#0a3d62")
ax.set_xlabel("time [µs]"); ax.set_ylabel("VCO control voltage [V]")
ax.set_title("Control voltage drives the VCO toward the right frequency")
ax.grid(alpha=0.3)

fig.suptitle("PLL: feedback in the *phase* domain  (de Bellescize 1932)", fontsize=12)
fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "pll_lock.png")
fig.savefig(out_path, dpi=120)
print(f"saved {out_path}")
