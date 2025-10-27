import matplotlib.pyplot as plt

models = ['SAM2-T','SAM2-S','SAM2-B','SAM2-L','SAM-B','SAM-L','SAM-H']

gflops_fp32 = [96.6, 125.4, 246.9, 755.3, 346.0, 1224.6, 2548.9]
gflops_w4aw = [33.31, 42.5, 60.0, 168.0, 99.0, 245.0, 509.0]

storage_fp32 = [0.15, 0.18, 0.2, 0.9, 0.36, 1.2, 2.5]
storage_w4aw = [0.05, 0.06, 0.08, 0.33, 0.12, 0.45, 0.9]  # SAM-B 这行建议核对

speedup = [a/b for a,b in zip(gflops_fp32, gflops_w4aw)]
stor_reduction = [1 - (b/a) for a,b in zip(storage_fp32, storage_w4aw)]

# 图1：GFLOPs 折线 + 加速倍率
plt.figure(figsize=(9, 4.5))
x = list(range(len(models)))
plt.plot(x, gflops_fp32, marker='o', label='GFLOPs FP32')
plt.plot(x, gflops_w4aw, marker='o', label='GFLOPs W4AW')
for xi, y, sp in zip(x, gflops_w4aw, speedup):
    plt.text(xi, y, f'×{sp:.2f}', fontsize=9, ha='center', va='bottom')
plt.xticks(x, models, rotation=20)
plt.ylabel('GFLOPs (lower is better)')
plt.title('Compute vs. Model (FP32 vs W4AW)')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('gflops_lines.png', dpi=300)

# 图2：存储折线 + 压缩比例
plt.figure(figsize=(9, 4.5))
plt.plot(x, storage_fp32, marker='o', label='Storage FP32 (GB)')
plt.plot(x, storage_w4aw, marker='o', label='Storage W4AW (GB)')
for xi, y, red in zip(x, storage_w4aw, stor_reduction):
    plt.text(xi, y, f'-{red*100:.0f}%', fontsize=9, ha='center', va='bottom')
plt.xticks(x, models, rotation=20)
plt.ylabel('Storage (GB)')
plt.title('Model Size vs. Model (FP32 vs W4AW)')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('storage_lines.png', dpi=300)