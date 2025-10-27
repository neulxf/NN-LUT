import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme(style="whitegrid", font_scale=1.1)

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
    "BF16":     [6.137, 2.856, 7.457, 5.319, 4.811,  9.72, 8.70],
    # "NN-LUT":   [8.281, 5.126, 28194, 70360, 6.833, 825.31, 10.76],
    "NN-LUT":   [28.194, 5.126, 28194, 70360, 6.833, 825.31, 10.76],
    "NLI":      [6.138, 2.857, 7.457, 5.320, 4.813,  9.73, 8.70],
}
df = pd.DataFrame(data)

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(10, 5))

models = df["Model"]
x = np.arange(len(models))
width = 0.25

style = {
    "BF16": {"color": "#09ADB8", "hatch": None},   # 浅灰
    # 更浅、更灰的珊瑚（可再调）
    "NN-LUT": {"color": "#F7BBBD", "hatch": None},
    "NLI":    {"color": "#ff6200", "hatch": None},
}

for idx, method in enumerate(["BF16", "NN-LUT", "NLI"]):
    props = style[method]
    bars = ax.bar(
        x + (idx-1)*width,
        df[method],
        width,
        label=method,
        color=props["color"],
        edgecolor="#444444",
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
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=30, ha="right")

plt.tight_layout()
plt.savefig("plots/figure2.png")