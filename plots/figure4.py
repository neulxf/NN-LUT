import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme(style="whitegrid", font_scale=1.1)

# Set Times New Roman font
plt.rcParams['font.family'] = 'Times New Roman'

# ------------ data -------------

# ------------ data -------------
data = {
    "Model": [
        "Llama3 8B",
        "Llama3 70B",
        "Qwen2.5 7B",
        "Qwen2.5 32B",
        "Qwen1.5 110B",
        "Qwen3 8B",
        "Qwen3 30B-A3B",
    ],
    "NLI":     [0.91,  7.97, 0.79, 3.64,  800, 0.90, 3.44],
    "NN-LUT":   [0.99,  8.63, 0.87, 3.96, 860, 0.98, 3.83],
    "RI-LUT":      [1, 8.66, 0.88, 4,  880, 1.01, 3.85],
}
df = pd.DataFrame(data)

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(10, 5))

# Set grid lines to black and add thicker outer border
ax.grid(color='black', alpha=0.3)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('black')

models = df["Model"]
x = np.arange(len(models))
width = 0.25

style = {
    "NLI": {"color": "#ff6200", "hatch": None},      # Dark blue
    "NN-LUT": {"color": "#F7BBBD", "hatch": None},   # Light blue
    "RI-LUT": {"color": "#9FB99C", "hatch": None},   # Pink
}

for idx, method in enumerate(["NLI", "NN-LUT", "RI-LUT"]):
    props = style[method]
    bars = ax.bar(
        x + (idx-1)*width,
        df[method],
        width,
        label=method,
        color=props["color"],
        edgecolor="black",
        hatch=props["hatch"],
        linewidth=0.6,
    )
    # for bar, val in zip(bars, df[method]):
    #     ax.text(
    #         bar.get_x() + bar.get_width() / 2,
    #         val * 1.2,
    #         f"{val:.2g}",
    #         ha="center",
    #         va="bottom",
    #         fontsize=8,
    #     )

ax.set_yscale("log", base=10)
ax.set_ylabel("Latency")
ax.set_xticks(x)
ax.set_xticklabels(models, ha="right")

# Add legend to the upper right corner
# ax.legend(loc="upper right", frameon=True, framealpha=0.9)

plt.tight_layout()
plt.savefig("plots/figure4.png")