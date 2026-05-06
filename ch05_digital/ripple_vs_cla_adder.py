"""
Chapter 5.3 - Ripple-carry vs carry-look-ahead: how adders got fast.

Phenomenon
----------
The simplest n-bit adder chains n full-adders, each waiting for the
carry-out of the previous one.  Total delay scales as O(n) — for a 64-bit
add, that's 64 gate delays just for the addition.  CPUs need to add in
~1 cycle, not 64.

Carry-look-ahead (Weinberger & Smith, 1958) precomputes the carries in
parallel using two signals per bit:
    g_i = a_i AND b_i      (generate: this bit definitely creates a carry)
    p_i = a_i XOR b_i      (propagate: this bit passes a carry along)
With those, c_{i+1} = g_i + p_i * c_i can be unrolled:
    c_4 = g_3 + p_3 g_2 + p_3 p_2 g_1 + p_3 p_2 p_1 g_0 + p_3 p_2 p_1 p_0 c_0
which can be computed in O(log n) gate delays.  This is *the* idea that
made multi-GHz arithmetic possible.

We measure both, with a simple "1 unit per gate" delay model.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

def ripple_carry_delay(n):
    # Each full-adder ≈ 2 gate-delays in the carry path
    return 2 * n

def cla_block_delay(n, block_size=4):
    # 1 layer for g/p, log_{block_size}(n) carry-tree layers, 1 final sum layer
    layers = int(np.ceil(np.log(n) / np.log(block_size)))
    return 1 + 2 * layers + 1

ns = np.array([4, 8, 16, 32, 64, 128, 256])
ripple = np.array([ripple_carry_delay(n) for n in ns])
cla    = np.array([cla_block_delay(n)    for n in ns])

# ---------- Functional model: do an actual add and verify both produce the same answer ----------
rng = np.random.default_rng(42)

def to_bits(x, n):
    return [(x >> i) & 1 for i in range(n)]
def from_bits(bits):
    x = 0
    for i, b in enumerate(bits):
        x |= int(b) << i
    return x

def ripple_carry_add(a, b, n):
    out = []
    cin = 0
    for i in range(n):
        s = (a[i] ^ b[i]) ^ cin
        cout = (a[i] & b[i]) | (cin & (a[i] ^ b[i]))
        out.append(s); cin = cout
    return out, cin

def cla_add(a, b, n):
    p = [a[i] ^ b[i] for i in range(n)]
    g = [a[i] & b[i] for i in range(n)]
    c = [0] * (n + 1)
    for i in range(n):
        c[i+1] = g[i] | (p[i] & c[i])
    s = [p[i] ^ c[i] for i in range(n)]
    return s, c[n]

# Verify on random inputs (Python ints to avoid int64 overflow at n=64)
import random
random.seed(0)
for n in [8, 16, 32, 63]:
    for _ in range(50):
        a = random.randrange(0, 1 << n)
        b = random.randrange(0, 1 << n)
        a_bits, b_bits = to_bits(a, n), to_bits(b, n)
        s1, c1 = ripple_carry_add(a_bits, b_bits, n)
        s2, c2 = cla_add(a_bits, b_bits, n)
        assert from_bits(s1) + (c1 << n) == from_bits(s2) + (c2 << n) == (a + b)

# ---------- Plot ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(ns, ripple, "o-", label="ripple-carry  O(n)", color="#b71540", lw=2, ms=8)
ax.plot(ns, cla,    "s-", label="carry-look-ahead  O(log n)", color="#0a3d62", lw=2, ms=8)
ax.set_xlabel("adder width n [bits]"); ax.set_ylabel("worst-case gate delay (units)")
ax.set_title("Adder delay vs width")
ax.legend(); ax.grid(alpha=0.3)
ax.set_xscale("log", base=2); ax.set_xticks(ns); ax.set_xticklabels(ns)

ax = axes[1]
# Same data on log-log to see the slopes clearly
ax.loglog(ns, ripple, "o-", color="#b71540", lw=2, ms=8, label="ripple-carry")
ax.loglog(ns, cla,    "s-", color="#0a3d62", lw=2, ms=8, label="carry-look-ahead")
ax.set_xlabel("n [bits]"); ax.set_ylabel("delay (log scale)")
ax.set_title("Same plot, log-log: slopes reveal asymptotic class")
ax.legend(); ax.grid(alpha=0.3, which="both")

fig.suptitle("Why your CPU can add 64-bit numbers in one clock\n"
             "(Weinberger & Smith 1958: collapse the carry chain in parallel)",
             fontsize=12)
fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "ripple_vs_cla_adder.png")
fig.savefig(out_path, dpi=120)
print(f"saved {out_path}")
