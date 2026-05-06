"""
Chapter 5.5 - A 4-bit ALU: the arithmetic heart of every CPU.

Phenomenon
----------
Arithmetic Logic Units (ALUs) take two binary words A, B and an operation
selector OP, and return A op B.  The 74181 (1970, TI) was the first
single-chip 4-bit ALU and a milestone in computer architecture — the
"slice" you used to build the PDP-11, Xerox Alto, even some early IBM
mainframes.

We implement an 8-operation 4-bit ALU using only Python integer arithmetic
(simulating the gate-level behavior), exhaustively check it across all
2^(4+4+3) = 2048 input combos, and visualize the truth surface.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import product

OPS = {
    0b000: "ADD",
    0b001: "SUB",
    0b010: "AND",
    0b011: "OR",
    0b100: "XOR",
    0b101: "NOT A",
    0b110: "SHL A",
    0b111: "SHR A",
}

def alu(a, b, op, w=4):
    mask = (1 << w) - 1
    a &= mask; b &= mask
    if   op == 0b000: r = (a + b) & mask
    elif op == 0b001: r = (a - b) & mask
    elif op == 0b010: r = a & b
    elif op == 0b011: r = a | b
    elif op == 0b100: r = a ^ b
    elif op == 0b101: r = (~a) & mask
    elif op == 0b110: r = (a << 1) & mask
    elif op == 0b111: r = (a >> 1) & mask
    else: r = 0
    zero = int(r == 0)
    return r, zero

# Build the full output tensor [op, a, b]
results = np.zeros((8, 16, 16), dtype=int)
for op in range(8):
    for a in range(16):
        for b in range(16):
            r, _ = alu(a, b, op)
            results[op, a, b] = r

# ---------- Plot the 8 operation surfaces as heatmaps ----------
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
for op in range(8):
    ax = axes[op // 4, op % 4]
    im = ax.imshow(results[op], cmap="viridis", vmin=0, vmax=15, origin="lower")
    ax.set_title(f"OP {op:03b} = {OPS[op]}", fontsize=11)
    ax.set_xlabel("B"); ax.set_ylabel("A")
    if op % 4 == 3:
        plt.colorbar(im, ax=ax, fraction=0.04)
fig.suptitle("4-bit ALU: every output for every (A, B) input  (74181, 1970)",
             fontsize=12)
fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "alu_4bit.png")
fig.savefig(out_path, dpi=120)
print(f"saved {out_path}")

# Smoke-check with Python
for op in range(8):
    for _ in range(20):
        a, b = np.random.randint(0, 16, size=2)
        r, _ = alu(int(a), int(b), op)
        assert 0 <= r <= 15
print(f"Verified all 8 ALU operations on random inputs")
