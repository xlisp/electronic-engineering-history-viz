"""
Chapter 7.5 - Attention is just three matrix multiplies.

Phenomenon
----------
The Transformer (Vaswani et al. 2017) is essentially three torch.matmul
calls per attention head:

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    A = softmax(Q @ K^T / sqrt(d_k))    # attention weights
    Y = A @ V

Every step is a matrix multiplication.  And matrix multiplication is the
operation a TPU/GPU does best.  *That* is why "scaling laws" worked: as
the silicon got better at matmul, neural networks built out of matmul got
smarter, and the same hardware ran them.

We compute self-attention from scratch, plot the attention matrix on a toy
sentence, and trace which token is paying attention to which.
"""
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

torch.manual_seed(0)
torch.set_default_dtype(torch.float32)

# ---------- Toy "sentence" ----------
tokens = ["The", "cat", "sat", "on", "the", "mat", "."]
T = len(tokens)
d_model = 32
d_k = 16

# Random token embeddings X (T, d_model). In a real model these come from an embedding table.
X = torch.randn(T, d_model) * 0.5

# Random projection matrices for Q, K, V
W_Q = torch.randn(d_model, d_k) / np.sqrt(d_model)
W_K = torch.randn(d_model, d_k) / np.sqrt(d_model)
W_V = torch.randn(d_model, d_k) / np.sqrt(d_model)

# Force a clearer attention pattern: tie tokens that are *the same word*
# ("The" and "the" should attend to each other).
X[4] = X[0] + 0.1 * torch.randn(d_model)   # second "the" is near first "The"

# Forward pass
Q = X @ W_Q                              # (T, d_k)
K = X @ W_K
V = X @ W_V
scores = Q @ K.transpose(0, 1) / np.sqrt(d_k)   # (T, T)
A = F.softmax(scores, dim=-1)            # (T, T) — attention weights
Y = A @ V                                # (T, d_k)

# ---------- Plot ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
im = ax.imshow(scores.numpy(), cmap="RdBu_r")
ax.set_xticks(range(T)); ax.set_yticks(range(T))
ax.set_xticklabels(tokens, rotation=45); ax.set_yticklabels(tokens)
for i in range(T):
    for j in range(T):
        ax.text(j, i, f"{scores[i,j]:+.1f}", ha="center", va="center", fontsize=8)
ax.set_title("Raw scores  Q · K^T / √d_k")
ax.set_xlabel("attended-to token (key)"); ax.set_ylabel("attending token (query)")
plt.colorbar(im, ax=ax, fraction=0.04)

ax = axes[1]
im = ax.imshow(A.numpy(), cmap="viridis", vmin=0, vmax=A.max().item())
ax.set_xticks(range(T)); ax.set_yticks(range(T))
ax.set_xticklabels(tokens, rotation=45); ax.set_yticklabels(tokens)
for i in range(T):
    for j in range(T):
        ax.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center",
                color="white" if A[i,j] < A.max()/2 else "black", fontsize=8)
ax.set_title("Attention weights  softmax(scores)\n(rows sum to 1)")
ax.set_xlabel("attended-to token"); ax.set_ylabel("attending token")
plt.colorbar(im, ax=ax, fraction=0.04)

# Right panel: bar chart of attention from token "the" (index 4) to all others
ax = axes[2]
weights = A[4].numpy()
ax.bar(range(T), weights, color=["#b71540" if t == 0 else "#0a3d62" for t in range(T)])
ax.set_xticks(range(T)); ax.set_xticklabels(tokens, rotation=45)
ax.set_ylabel("attention weight")
ax.set_title(f"Attention from '{tokens[4]}'\n(should attend strongly to '{tokens[0]}' since we tied embeddings)")
ax.grid(alpha=0.3, axis="y")

fig.suptitle("Self-attention = three matrix multiplies + a softmax\n"
             "(Vaswani et al. 2017 — every Transformer ever shipped runs this loop)",
             fontsize=12)
fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "attention_as_matmul.png")
fig.savefig(out_path, dpi=120)
print(f"saved {out_path}")
