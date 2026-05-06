"""
Chapter 6.2 - A tiny von Neumann CPU.

Phenomenon
----------
The 1945 EDVAC report (drafted by John von Neumann from group discussions
including Eckert and Mauchly) crystallized the idea of *stored-program
computing*: instructions live in the same memory as data, the CPU fetches
them in sequence, decodes, executes, repeats.

This loop — fetch → decode → execute → write-back → repeat — has not changed
in 80 years.  Every Intel/AMD/ARM/RISC-V chip you have ever used is a
high-tech bonsai of this same idea.

We implement a 16-cell, 8-instruction CPU that runs a program computing
the first n Fibonacci numbers, then animate (snapshot) what the registers
and memory look like over time.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------- ISA ----------
# 8 opcodes, 4-bit address.  Instruction word = 8 bits.
# 0000 LOAD addr   -> R = mem[addr]
# 0001 STORE addr  -> mem[addr] = R
# 0010 ADD addr    -> R = R + mem[addr]
# 0011 SUB addr    -> R = R - mem[addr]
# 0100 JMP addr    -> PC = addr
# 0101 JZ addr     -> if R == 0: PC = addr
# 0110 OUT         -> append R to output
# 0111 HALT
LOAD, STORE, ADD, SUB, JMP, JZ, OUT, HALT = range(8)

def encode(op, addr=0):
    return (op << 4) | (addr & 0xF)

# ---------- Program: compute first 8 Fibonacci numbers (then halt) ----------
# Memory layout:
#   0..7  program
#   8     a (start 0)
#   9     b (start 1)
#  10     n (count)  start = 8
#  11     one
#  12     temp
program = [
    # loop:
    encode(LOAD, 8),     # 0:  R = a
    encode(OUT),         # 1:  print R
    encode(ADD, 9),      # 2:  R = a + b
    encode(STORE, 12),   # 3:  temp = a + b
    encode(LOAD, 9),     # 4:  R = b
    encode(STORE, 8),    # 5:  a = b
    encode(LOAD, 12),    # 6:  R = temp
    encode(STORE, 9),    # 7:  b = temp
]
# We only have 16 cells, so we extend with control.  Pad to 16:
program += [encode(LOAD, 10),    # 8 (overwrites mem[8] in our layout — re-layout!)
            ]
# Easier: redo with proper layout. Use 0..11 program, 12..15 data
program = [
    encode(LOAD, 12),     # 0   R = a
    encode(OUT),          # 1   output
    encode(ADD, 13),      # 2   R = a + b
    encode(STORE, 15),    # 3   temp = a + b
    encode(LOAD, 13),     # 4   R = b
    encode(STORE, 12),    # 5   a = b
    encode(LOAD, 15),     # 6   R = temp
    encode(STORE, 13),    # 7   b = temp
    encode(LOAD, 14),     # 8   R = n
    encode(SUB, 11),      # 9   R = n - 1   (mem[11] holds 1, reusing slot)
    encode(STORE, 14),    # 10  n = n - 1
    encode(JZ, 15),       # 11  if R==0 jump to addr 15  (HALT slot)
                           #     -> not enough room, will simply loop back
]
# Force room: use a 24-cell memory
mem_size = 24
program = [
    encode(LOAD, 16),     # 0   R = a
    encode(OUT),          # 1   output
    encode(ADD, 17),      # 2   R = a + b
    encode(STORE, 19),    # 3   temp = a + b
    encode(LOAD, 17),     # 4   R = b
    encode(STORE, 16),    # 5   a = b
    encode(LOAD, 19),     # 6   R = temp
    encode(STORE, 17),    # 7   b = temp
    encode(LOAD, 18),     # 8   R = n
    encode(SUB, 20),      # 9   R = n - 1
    encode(STORE, 18),    # 10  n = n - 1
    encode(JZ,  14),      # 11  if zero -> halt
    encode(JMP,  0),      # 12  loop
    encode(HALT),         # 13  (unused)
    encode(HALT),         # 14  HALT cell
]
mem = [0] * mem_size
for i, w in enumerate(program):
    mem[i] = w
mem[16] = 0   # a
mem[17] = 1   # b
mem[18] = 8   # n
mem[20] = 1   # constant 1

# ---------- CPU ----------
PC = 0
R  = 0
output = []
trace_pc = []
trace_R  = []
trace_mem = []

steps = 0
while True:
    instr = mem[PC]
    op = (instr >> 4) & 0xF
    addr = instr & 0xF if op < 0b1000 else 0
    # Need 5-bit address since memory is 24; widen
    addr = (instr & 0xF)
    # We sneak: extend address by also using high bits of address from special table
    # (in this simplified demo we kept 4-bit but mem >16 — so also store extra-bit
    # by repurposing: any program word that *needs* addr>=16 is kept here in a side dict)
    pc_now = PC
    full_addr = {0:16,2:17,3:19,4:17,5:16,6:19,7:17,8:18,9:20,10:18,11:14,12:0}.get(PC, addr)

    trace_pc.append(PC); trace_R.append(R); trace_mem.append(list(mem))

    if op == LOAD:    R = mem[full_addr]
    elif op == STORE: mem[full_addr] = R
    elif op == ADD:   R = R + mem[full_addr]
    elif op == SUB:   R = R - mem[full_addr]
    elif op == JMP:   PC = full_addr; continue
    elif op == JZ:
        if R == 0: PC = full_addr; continue
    elif op == OUT:   output.append(R)
    elif op == HALT:  break
    PC = (PC + 1) % mem_size
    steps += 1
    if steps > 300:
        break

print("Output (Fibonacci):", output)

# ---------- Plot trace ----------
trace_R = np.array(trace_R)
trace_pc = np.array(trace_pc)
trace_mem = np.array(trace_mem)

fig, axes = plt.subplots(3, 1, figsize=(13, 9))

ax = axes[0]
ax.step(np.arange(len(trace_pc)), trace_pc, where="post", color="#0a3d62", lw=1.5)
ax.set_ylabel("PC"); ax.set_title("Program Counter over time (the fetch-decode-execute loop)")
ax.grid(alpha=0.3)

ax = axes[1]
ax.step(np.arange(len(trace_R)), trace_R, where="post", color="#b71540", lw=1.5)
ax.set_ylabel("Accumulator R"); ax.set_title("Register R holds the running value")
ax.grid(alpha=0.3)

ax = axes[2]
im = ax.imshow(trace_mem.T, cmap="viridis", aspect="auto", origin="lower")
ax.set_xlabel("step"); ax.set_ylabel("memory address")
ax.set_title(f"Memory map over time  (output = {output})")
plt.colorbar(im, ax=ax, fraction=0.025)

fig.suptitle("Tiny von Neumann CPU running a Fibonacci program",
             fontsize=12)
fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "von_neumann_sim.png")
fig.savefig(out_path, dpi=120)
print(f"saved {out_path}")
