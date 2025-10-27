import torch
import matplotlib.pyplot as plt
import seaborn as sns

# 设置 Seaborn 风格与上下文，营造柔和的审美
sns.set(style="whitegrid", context="talk")

# 加载数据
silu_data = torch.load("silu_i_3.pth")
attn_data = torch.load("attn.pt")

print("silu_data shape:", silu_data.shape)  # 示例输出: torch.Size([1, 166, 18944])
print("attn_data shape:", attn_data.shape)

# 展平数据并转换为 numpy 数组
silu_flat = silu_data.float().flatten().cpu().numpy()
attn_flat = attn_data.float().flatten().cpu().numpy()

# 创建1行2列的画布
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 绘制 SiLU 输入数据的直方图（橙色）
sns.histplot(silu_flat, bins=100, stat="density", color="orange",
             kde=True, kde_kws={"bw_adjust": 0.7}, ax=axes[0], edgecolor="none")
axes[0].set_title("SiLU Input Distribution", fontsize=16)
axes[0].set_xlabel("Value", fontsize=14)
axes[0].set_ylabel("Density", fontsize=14)

# 绘制 Softmax 输入数据的直方图（蓝色）
sns.histplot(attn_flat, bins=100, stat="density", color="blue",
             kde=True, kde_kws={"bw_adjust": 0.7}, ax=axes[1], edgecolor="none")
axes[1].set_title("Softmax Input Distribution", fontsize=16)
axes[1].set_xlabel("Value", fontsize=14)
axes[1].set_ylabel("Density", fontsize=14)

plt.tight_layout()
plt.savefig("combined_histograms_adjusted.png", dpi=300)
plt.show()