# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project intent

A Python-based project that re-teaches electronic engineering by retracing its **historical discovery path** (Coulomb's law → Faraday's coil → Maxwell's equations → Bell Labs transistor → Intel 4004 → Google TPU). Every visualization should connect a modern technique to the historical figure / real-world phenomenon that produced it. This framing — "EE as a 200-year human story, not a stack of textbook abstractions" — is the load-bearing design constraint, not a marketing tagline. Code, comments, and animations should make the historical lineage visible (e.g., `systolic_array_animation.py` deliberately maps a TPU's MAC array back to H.T. Kung's 1978 paper).

This project is the EE companion to `../math-history-viz`. The pedagogical philosophy is identical; only the subject matter differs.

## Architecture

Code is organized by **historical chapter**, not by technique. Each chapter is a self-contained directory with runnable scripts:

```
ch00_perspectives/      Field-vs-circuit duality (when does Maxwell collapse to KVL/KCL?)
ch01_maxwell/           Coulomb, Oersted, Faraday, Maxwell — fields, induction, EM waves (FDTD)
ch02_analog_circuits/   Ohm, Kirchhoff, BJT, op-amp — diode I-V, RC, amplifiers
ch03_oscillators/       LC tank, RC relaxation, quartz, PLL — every electronic device's heartbeat
ch04_clock/             Skew, jitter, setup/hold, CDC — synchronizing 30B transistors
ch05_digital/           Boole→Shannon→NAND→adder→FSM — abstraction ladder of digital logic
ch06_cpu/               Turing → von Neumann → pipeline → cache → branch prediction
ch07_gpu_tpu/           SIMD, GPU matmul, systolic arrays, attention as matmul
```

Chapter 7 is the keystone: it reuses primitives from earlier chapters to make the claim that AI hardware is the convergence of two centuries of EE. When adding to ch07, prefer demonstrating the lineage over introducing new techniques.

## Pedagogical style: simulation over formulas, always visualize, phenomenon first

This project teaches EE through **executable code and visualization**, not through formula transcription. Three hard rules apply to every script:

1. **Prefer simulation to formulas.** Express ideas as runnable PyTorch / NumPy code rather than LaTeX. A capacitor charging is `dq/dt = (V - q/C)/R` integrated numerically, not $v(t)=V_0(1-e^{-t/RC})$ in a docstring. An electromagnetic wave is an FDTD update loop on a 2D grid, not a $\nabla^2 \vec{E} = \mu_0\varepsilon_0 \partial_t^2 \vec{E}$ block. When a formula is unavoidable (to name a historical equation), keep it to one line and immediately follow it with the code that computes it. The reader should be able to delete every formula in the file and still understand the physics from the code alone.
2. **Every script must produce a visualization.** No script is complete if it only prints numbers. Use Matplotlib / Seaborn for static plots and waveforms, Matplotlib's animation API for time-domain dynamics (capacitor charging, EM wave propagation, systolic array data flow). The visualization is the deliverable — the code exists to generate it.
3. **Phenomenon first, problem-driven, not tool-driven.** Almost every EE concept was originally invented to answer a concrete physical or industrial question. Lead with the real-world phenomenon (oscillating circuit, radio antenna, CPU pipeline stall, GPU memory wall…), simulate it first, and let the math drop out as the explanation of what the simulation is doing. The order is **phenomenon → simulation → dissection → formula**, never the reverse. "Disassemble the circuit, look at every part, then reassemble" is the working metaphor.

Lean on PyTorch even where NumPy would suffice: it makes ch07 (GPU/TPU/deep-learning) connect cleanly back to ch01 (Maxwell), since the same `torch.Tensor` represents both a 2D EM field grid and a neural-network activation.

## Tech stack

- `PyTorch` — numerics + autograd; intentionally used throughout, even for ODEs/PDEs, to thread the same data type from Maxwell's grid to Transformer's attention
- `NumPy` / `SciPy` — when PyTorch is overkill (e.g., `scipy.integrate.solve_ivp` for circuit ODEs)
- `Matplotlib` — static plots + animations (FuncAnimation for FDTD, systolic flow, charging curves)
- `SymPy` — symbolic verification of hand derivations (rare; only when needed)

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python ch01_maxwell/coulomb_field.py            # representative entry point
python ch01_maxwell/maxwell_em_wave_2d.py       # the project's "wow" demo
python ch07_gpu_tpu/systolic_array_animation.py # the project's "synthesis" demo
```

Each script is **self-contained** and **single-file**: no shared utility modules between chapters. Duplication is OK — readability and copy-pasteability matter more than DRY in a teaching repo.

## Conventions

- Each script saves its plot as `<script_name>.png` next to itself (so README image links work).
- Each script has a top-of-file docstring with: (a) the historical figure / year, (b) the physical phenomenon, (c) the modern relevance.
- Comments are sparse but **historical** — when a piece of code corresponds to a specific scientist's contribution, name them: `# Heaviside 1884: vector form` rather than just `# divergence`.
- All text in scripts is in English; user-facing docs (README) are bilingual but the reading order is Chinese.
