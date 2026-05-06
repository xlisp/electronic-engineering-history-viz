"""
Chapter 6.1 - The Turing machine that everything descends from.

Phenomenon
----------
In 1936 Alan Turing proposed a hypothetical machine: an infinite tape, a
read/write head, a finite state register, and a transition table. He used
it to settle Hilbert's *Entscheidungsproblem* (NO — there is no
algorithm that decides every mathematical statement).

That same paper turned out to define what "computable" means. Every CPU
that has ever been built is, formally, a finite approximation of this
mathematical object.

Demo: a Turing machine that recognizes the language a^n b^n (an equal
number of a's followed by b's). It crosses out matching pairs from the
outside in until the tape is empty.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

BLANK = "_"

# Transition table:  (state, symbol)  ->  (write, move, next_state)
table = {
    ("q0", "a"): ("X", +1, "q1"),
    ("q0", "Y"): ("Y", +1, "q3"),       # already crossed out, scan to end
    ("q0", "_"): ("_",  0, "q_accept"), # empty -> accept

    ("q1", "a"): ("a", +1, "q1"),
    ("q1", "Y"): ("Y", +1, "q1"),
    ("q1", "b"): ("Y", -1, "q2"),
    ("q1", "_"): ("_",  0, "q_reject"),

    ("q2", "a"): ("a", -1, "q2"),
    ("q2", "Y"): ("Y", -1, "q2"),
    ("q2", "X"): ("X", +1, "q0"),

    ("q3", "Y"): ("Y", +1, "q3"),
    ("q3", "_"): ("_",  0, "q_accept"),
}

def run(tape, max_steps=200):
    tape = list(tape) + [BLANK] * 5
    head = 0
    state = "q0"
    history = [(state, head, list(tape))]
    for _ in range(max_steps):
        if state.startswith("q_"):
            break
        symbol = tape[head]
        if (state, symbol) not in table:
            state = "q_reject"
            history.append((state, head, list(tape)))
            break
        write, move, nxt = table[(state, symbol)]
        tape[head] = write
        head = max(0, head + move)
        if head >= len(tape) - 1:
            tape.append(BLANK)
        state = nxt
        history.append((state, head, list(tape)))
    return history

def plot_run(history, title, ax):
    width = max(len(h[2]) for h in history)
    grid = np.zeros((len(history), width))
    chars = [["" for _ in range(width)] for _ in history]
    for i, (state, head, tape) in enumerate(history):
        for j, c in enumerate(tape[:width]):
            chars[i][j] = c
            grid[i, j] = {"a": 1, "b": 2, "X": 3, "Y": 4, BLANK: 0}.get(c, 0)
    cmap = plt.matplotlib.colors.ListedColormap(
        ["white", "#ffd166", "#06d6a0", "#118ab2", "#ef476f"])
    ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=4)
    for i, (state, head, tape) in enumerate(history):
        for j, c in enumerate(tape[:width]):
            ax.text(j, i, c, ha="center", va="center", fontsize=10)
        # Triangle marker for the head
        ax.scatter([head], [i], marker="v", color="red", s=60, zorder=5)
        ax.text(width + 0.5, i, state, va="center", fontsize=9)
    ax.set_xticks(range(width))
    ax.set_yticks(range(len(history)))
    ax.set_yticklabels([f"step {i}" for i in range(len(history))], fontsize=8)
    ax.set_xlabel("tape cell")
    ax.set_title(title)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
plot_run(run("aaabbb"), "Input 'aaabbb' → ACCEPT", axes[0])
plot_run(run("aabbb"),  "Input 'aabbb' (unbalanced) → REJECT", axes[1])

# Legend
legend_handles = [mpatches.Patch(color="#ffd166", label="a"),
                  mpatches.Patch(color="#06d6a0", label="b"),
                  mpatches.Patch(color="#118ab2", label="X (matched 'a')"),
                  mpatches.Patch(color="#ef476f", label="Y (matched 'b')"),
                  mpatches.Patch(color="white", ec="gray", label="blank '_'")]
fig.legend(handles=legend_handles, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Turing machine recognizing $a^n b^n$  (Turing 1936)\n"
             "head crosses 'a' on the left, walks right, crosses matching 'b', repeats",
             fontsize=12)
fig.tight_layout()
fig.subplots_adjust(bottom=0.12)
out_path = os.path.join(os.path.dirname(__file__), "turing_machine.png")
fig.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"saved {out_path}")
