import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data extracted from the table
data = {
    "Model": [
        "Llama3 8B",
        "Llama3 70B",
        "Qwen2.5 7B",
        "Qwen2.5 32B",
        "Qwen1.5 110B",
        "Qwen3 8B",
        "Qwen3 30B‑A3B",
    ],
    "Min": [-38.75, -38.25, -77, -73, -39, -141, -38],
    "Max": [26.125, 27.45, 80, 130, 28, 106, 56.75],
}

df = pd.DataFrame(data)

# Plot: vertical range (min↔max) + endpoints, one per model
fig, ax = plt.subplots(figsize=(8, 4))

positions = np.arange(len(df))
for pos, (mn, mx) in enumerate(zip(df["Min"], df["Max"])):
    ax.plot([pos, pos], [mn, mx], linewidth=5, alpha=0.5)          # vertical line showing range
    ax.scatter(pos, mn, s=150)                    # min point
    ax.scatter(pos, mx, s=150)                    # max point

ax.set_xticks(positions)
ax.set_xticklabels(df["Model"], rotation=45, ha="right")
# ax.set_ylabel("Silu activation value")
# ax.set_title("Range of Silu activations across models")
ax.grid(True, axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("plots/figure1.png")