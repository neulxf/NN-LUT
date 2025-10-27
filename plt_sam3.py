# Create a single chart with dual y-axes:
# - Bars (left y-axis) for Storage (GB) in blue hues
# - Lines (right y-axis) for Compute (TFLOPs) in warm hues
# Convert GFLOPs to TFLOPs in the plot.
import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['SAM2-T','SAM2-S','SAM2-B','SAM2-L','SAM-B','SAM-L','SAM-H']

gflops_fp32 = np.array([96.6, 125.4, 246.9, 755.3, 346.0, 1224.6, 2548.9])
gflops_w4aw = np.array([33.31, 42.5, 60.0, 168.0, 99.0, 245.0, 509.0])

# Convert to TFLOPs
tflops_fp32 = gflops_fp32 / 1000.0
tflops_w4aw = gflops_w4aw / 1000.0

storage_fp32 = np.array([0.15, 0.18, 0.2, 0.9, 0.36, 1.2, 2.5])
storage_w4aw = np.array([0.05, 0.06, 0.08, 0.33, 0.12, 0.45, 0.9])  # Verify SAM-B (0.9) later

# Derived metrics
speedup = gflops_fp32 / gflops_w4aw
savings_pct = 100 * (1 - storage_w4aw / storage_fp32)  # may be negative if W4AW>FP32

# Positions and widths
x = np.arange(len(models))
bar_width = 0.35

# Colors
color_storage_fp32 = '#1f77b4'  # blue
color_storage_w4aw = '#17becf'  # teal
color_tflops_fp32  = '#ff7f0e'  # orange
color_tflops_w4aw  = '#d62728'  # red

fig, ax_left = plt.subplots(figsize=(10, 5))

# Left axis: Storage bars
bars1 = ax_left.bar(x - bar_width/2, storage_fp32, width=bar_width, label='Storage FP32 (GB)',
                    color=color_storage_fp32, edgecolor='black', linewidth=0.8)
bars2 = ax_left.bar(x + bar_width/2, storage_w4aw, width=bar_width, label='Storage W4AW (GB)',
                    color=color_storage_w4aw, edgecolor='black', linewidth=0.8)

ax_left.set_ylabel('Storage (GB)')
ax_left.set_xlabel('Model')
ax_left.set_xticks(x, models, rotation=20)
ax_left.grid(True, axis='y', linestyle='--', linewidth=0.6, alpha=0.5)

# Annotate storage savings on W4AW bars
for rect, sv in zip(bars2, savings_pct):
    height = rect.get_height()
    ax_left.text(rect.get_x() + rect.get_width()/2, height, f'{sv:+.0f}%', ha='center', va='bottom', fontsize=9, color=color_storage_w4aw)

# Right axis: TFLOPs lines
ax_right = ax_left.twinx()
line1, = ax_right.plot(x, tflops_fp32, marker='o', label='TFLOPs FP32',
                        color=color_tflops_fp32, linewidth=1.6)
line2, = ax_right.plot(x, tflops_w4aw, marker='o', label='TFLOPs W4AW',
                        color=color_tflops_w4aw, linewidth=1.6)
ax_right.set_ylabel('TFLOPs (lower is better)')
ax_right.grid(False)

# Annotate speedup at W4AW TFLOPs points
for xi, y, sp in zip(x, tflops_w4aw, speedup):
    ax_right.text(xi, y, f'×{sp:.2f}', ha='center', va='bottom', fontsize=9, color=color_tflops_w4aw)

# Legend (combine)
handles_left, labels_left = ax_left.get_legend_handles_labels()
handles_right, labels_right = ax_right.get_legend_handles_labels()
ax_left.legend(handles_left + handles_right, labels_left + labels_right, loc='upper left', ncol=2, frameon=True)

plt.title('Storage (GB) and Compute (TFLOPs): FP32 vs W4AW')
plt.tight_layout()
plt.savefig('compute_storage_combo_color.png', dpi=300)
plt.savefig('compute_storage_combo_color.pdf')
# plt.show()

# print("Saved: compute_storage_combo_color.png")
# print("Saved: /mnt/data/compute_storage_combo_color.pdf")