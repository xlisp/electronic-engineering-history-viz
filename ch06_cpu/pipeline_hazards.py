"""
Chapter 6.3 - Pipeline hazards: why CPUs are not as fast as they look.

Phenomenon
----------
A 5-stage pipeline (IF / ID / EX / MEM / WB) lets the CPU work on 5
instructions in parallel.  Ideal throughput: 1 instruction per cycle.

Reality: data hazards force stalls (RAW: read-after-write).  Branch
mispredictions force flushes.  We render Gantt-style pipeline diagrams
showing the bubbles forming and the IPC (instructions per cycle) dropping.

Hennessy & Patterson popularized this picture in *Computer Architecture:
A Quantitative Approach* (1990) — required reading for every chip designer.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

STAGES = ["IF", "ID", "EX", "MEM", "WB"]

def schedule_no_forwarding(program):
    """Return a per-instruction (start_cycle_for_each_stage) schedule.
       program is a list of dicts with 'reads', 'writes', 'is_branch', 'mispred'."""
    n = len(program)
    starts = np.full((n, 5), -1)
    write_done = {}   # reg -> cycle WB completes
    cycle = 0
    for i, instr in enumerate(program):
        # Each instruction enters IF one cycle after the previous one
        if i == 0:
            ifc = 0
        else:
            ifc = starts[i-1, 0] + 1
        # ID stage waits until source registers' write_done <= ID cycle
        idc = max(ifc + 1, *(write_done.get(r, -1) + 1 for r in instr["reads"]))
        exc = idc + 1
        memc = exc + 1
        wbc  = memc + 1
        starts[i] = [ifc, idc, exc, memc, wbc]
        for w in instr["writes"]:
            write_done[w] = wbc
        # if this is a branch and mispredicts, the next instruction is squashed:
        # easiest model: the next instr gets pushed by 2 cycles (flush penalty)
        if instr.get("is_branch") and instr.get("mispred"):
            # bump the next IF cycle by 2
            if i + 1 < n:
                program[i+1]["_extra_if_offset"] = 2
        # apply pending offset
        if instr.get("_extra_if_offset"):
            starts[i, 0] += instr["_extra_if_offset"]
            # also ripple subsequent stages
            for s in range(1, 5):
                starts[i, s] = max(starts[i, s], starts[i, s-1] + 1)
    return starts

# Construct a small program: a RAW hazard, then a branch with mispredict
program = [
    dict(reads=["r1", "r2"], writes=["r3"], name="ADD r3, r1, r2"),    # 0
    dict(reads=["r3", "r4"], writes=["r5"], name="ADD r5, r3, r4"),    # 1: RAW on r3
    dict(reads=["r5"], writes=[],   is_branch=True, mispred=True,
         name="BNE r5, label"),                                        # 2: branch mispredict
    dict(reads=["r6"], writes=["r7"], name="ADD r7, r6, r6"),          # 3
    dict(reads=["r7"], writes=["r8"], name="ADD r8, r7, r7"),          # 4: RAW on r7
    dict(reads=["r9"], writes=["r10"], name="ADD r10, r9, r9"),        # 5
]

sched = schedule_no_forwarding(program)
n_cycles = sched.max() + 1
n = len(program)
ipc = n / n_cycles

# ---------- Plot ----------
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Ideal pipeline (no hazards): each instr starts 1 cycle after the previous
ideal = np.zeros((n, 5), dtype=int)
for i in range(n):
    for s in range(5):
        ideal[i, s] = i + s
ipc_ideal = n / (ideal.max() + 1)

stage_colors = ["#0a3d62", "#3c6382", "#b71540", "#e58e26", "#7bed9f"]

for ax, schedule, title, ipc_v in [
    (axes[0], ideal, f"Ideal pipeline (no hazards), IPC = {ipc_ideal:.2f}", ipc_ideal),
    (axes[1], sched, f"With RAW + branch mispredict, IPC = {ipc:.2f}", ipc),
]:
    for i in range(n):
        for s, stage_name in enumerate(STAGES):
            cycle = schedule[i, s]
            ax.barh(i, 1.0, left=cycle, color=stage_colors[s], edgecolor="white")
            ax.text(cycle + 0.5, i, stage_name, ha="center", va="center", color="white", fontsize=8)
    ax.set_xlim(-0.5, max(ideal.max(), sched.max()) + 1.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([p["name"] for p in program], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("cycle")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)

handles = [mpatches.Patch(color=c, label=s) for c, s in zip(stage_colors, STAGES)]
fig.legend(handles=handles, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Pipeline Gantt chart: ideal vs real with hazards\n"
             "(Hennessy & Patterson, 'Computer Architecture: A Quantitative Approach')",
             fontsize=12)
fig.tight_layout()
fig.subplots_adjust(bottom=0.13)
out_path = os.path.join(os.path.dirname(__file__), "pipeline_hazards.png")
fig.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"saved {out_path}")
