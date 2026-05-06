"""
Chapter 5.2 - NAND is universal: build a CPU from a single gate.

Phenomenon
----------
A NAND gate (NOT (A AND B)) is "functionally complete" — every Boolean
function, no matter how complex, can be synthesized from NAND gates alone.

Why this matters in practice: a CMOS NAND2 takes 4 transistors. A foundry
that can fabricate NAND can fabricate anything. Reducing the design problem
to one gate is the same kind of move as Turing reducing all of computation
to one tape and one head.

We construct AND, OR, NOT, XOR, half-adder, and full-adder out of NAND only,
then check the truth tables.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from itertools import product

# ---------- The only gate we are allowed ----------
def NAND(a, b):
    return 1 - (a * b)   # works for 0/1

def NOT(a):                    return NAND(a, a)
def AND(a, b):                 return NOT(NAND(a, b))
def OR(a, b):                  return NAND(NOT(a), NOT(b))
def XOR(a, b):                 return NAND(NAND(a, NAND(a, b)), NAND(b, NAND(a, b)))
def HALF_ADDER(a, b):          return XOR(a, b), AND(a, b)            # sum, carry
def FULL_ADDER(a, b, cin):
    s1, c1 = HALF_ADDER(a, b)
    s2, c2 = HALF_ADDER(s1, cin)
    return s2, OR(c1, c2)

# ---------- Verify by exhaustive truth tables ----------
def truth_table(fn, n_inputs):
    rows = []
    for inputs in product([0, 1], repeat=n_inputs):
        out = fn(*inputs)
        if isinstance(out, tuple):
            rows.append(list(inputs) + list(out))
        else:
            rows.append(list(inputs) + [out])
    return np.array(rows)

ttables = {
    "NOT (1 in)":         truth_table(NOT, 1),
    "AND (2 in)":         truth_table(AND, 2),
    "OR (2 in)":          truth_table(OR, 2),
    "XOR (2 in)":         truth_table(XOR, 2),
    "HALF_ADDER (2 in)":  truth_table(HALF_ADDER, 2),
    "FULL_ADDER (3 in)":  truth_table(FULL_ADDER, 3),
}

# Sanity check vs Python's own
assert np.all(truth_table(AND, 2)[:, 2] == np.array([0, 0, 0, 1]))
assert np.all(truth_table(XOR, 2)[:, 2] == np.array([0, 1, 1, 0]))

# ---------- Plot ----------
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, (name, table) in zip(axes.ravel(), ttables.items()):
    n_in = {"NOT (1 in)": 1, "AND (2 in)": 2, "OR (2 in)": 2,
            "XOR (2 in)": 2, "HALF_ADDER (2 in)": 2, "FULL_ADDER (3 in)": 3}[name]
    n_out = table.shape[1] - n_in
    ax.imshow(table, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, str(int(table[i,j])),
                    ha="center", va="center",
                    color=("white" if table[i,j] else "black"), fontsize=11)
    cols = (["a"] if n_in == 1 else ["a", "b"] if n_in == 2 else ["a", "b", "cin"]) + \
           (["out"] if n_out == 1 else ["sum", "carry"] if n_out == 2 else [])
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks([])
    ax.set_title(name, fontsize=11)
    # Vertical line between inputs and outputs
    ax.axvline(n_in - 0.5, color="red", lw=1.5)

fig.suptitle("Sheffer 1913: NAND is functionally complete\n"
             "every gate above is built from NAND only — verified by truth table",
             fontsize=12)
fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "nand_universal.png")
fig.savefig(out_path, dpi=120)
print(f"saved {out_path}")
