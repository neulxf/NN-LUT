import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

def create_scientific_line_plot(array1, array2, 
                               labels=None, 
                               title="Encoder Sensitivity Analysis", 
                               xlabel="Block Index", 
                               ylabel="mAP",
                               figsize=(10, 6),
                               save_path=None,
                               dpi=300):
    """
    创建符合科研规范的优美折线图

    Args:
        array1, array2: 长度为12的数组，值在0-1之间
        labels: 数据标签列表，默认为["Series 1", "Series 2"]
        title: 图表标题
        xlabel, ylabel: x轴和y轴标签
        figsize: 图片尺寸
        save_path: 保存路径（可选）
        dpi: 图片分辨率
    """

    # 设置科研论文风格
    plt.style.use('default')
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    rcParams['font.size'] = 12
    rcParams['axes.linewidth'] = 1.2
    rcParams['grid.linewidth'] = 0.8
    rcParams['lines.linewidth'] = 2
    rcParams['lines.markersize'] = 6

    # 验证输入数据
    assert len(array1) == 12 and len(array2) == 12, "数组长度必须为12"
    assert all(0 <= x <= 1 for x in array1), "array1的值必须在0-1之间"
    assert all(0 <= x <= 1 for x in array2), "array2的值必须在0-1之间"

    if labels is None:
        labels = ["Series 1", "Series 2"]

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # x轴数据（索引）
    x = np.arange(1, 13)  # 1到12，更符合科研习惯

    # 定义科研常用的颜色和标记 EB6B67  #FF3232
    colors = ['#5B9BD5', "orange"]  # 深蓝色和深红色
    markers = ['o', 's']  # 圆形和方形
    linestyles = ['-', '-']  # 实线和虚线

    # 绘制折线图
    line1 = ax.plot(x, array1, 
                    color=colors[0], 
                    marker=markers[0], 
                    linestyle=linestyles[0],
                    label=labels[0],
                    # markerfacecolor='white',
                    markeredgecolor=colors[0],
                    markeredgewidth=2,
                    markersize=7,
                    linewidth=3.5,
                    alpha=0.9)

    line2 = ax.plot(x, array2, 
                    color=colors[1], 
                    marker=markers[1], 
                    linestyle=linestyles[1],
                    label=labels[1],
                    # markerfacecolor='white',
                    markeredgecolor=colors[1],
                    markeredgewidth=2,
                    markersize=7,
                    linewidth=3.5,
                    alpha=0.9)

    # 设置坐标轴
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # 设置x轴刻度
    ax.set_xticks(x)
    ax.set_xlim(0.5, 12.5)

    # 设置y轴范围，留出适当边距
    y_min = min(min(array1), min(array2))
    y_max = max(max(array1), max(array2))
    margin = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
    ax.set_ylim(max(0, y_min - margin), min(1, y_max + margin))

    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)

    # 添加图例
    # legend = ax.legend(loc='best', 
    #                   frameon=True, 
    #                   fancybox=True, 
    #                   shadow=True,
    #                   framealpha=0.9,
    #                   fontsize=12)
    # legend.get_frame().set_facecolor('white')
    # legend.get_frame().set_edgecolor('gray')

    # 美化坐标轴
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    # 设置刻度参数
    ax.tick_params(axis='both', which='major', labelsize=11, 
                   direction='out', length=6, width=1.2)

    # 调整布局
    plt.tight_layout()

    # 保存图片（如果指定了路径）
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"图片已保存至: {save_path}")

    return fig, ax

if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)

    # 创建两个示例数组（长度为12，值在0-1之间）
    array1 = np.random.beta(2, 2, 12)  # beta分布，更自然的0-1数据
    array2 = np.random.beta(3, 1.5, 12)

    # 为了演示，稍微调整数据使其更有趣
    array1 = np.clip(array1 + 0.1 * np.sin(np.linspace(0, 2*np.pi, 12)), 0, 1)
    array2 = np.clip(array2 + 0.15 * np.cos(np.linspace(0, 2*np.pi, 12)), 0, 1)

    array1 = [0.311,    0.51,   0.54,   0.544,  0.548,  0.55,   0.551,  0.548,  0.549,  0.551,  0.559,  0.558]
    array2 = [0.299,    0.377,  0.555,  0.364,  0.536,  0.536,  0.559,  0.563,  0.564,  0.561,  0.574,  0.574]

    # 创建图表
    fig, ax = create_scientific_line_plot(
        np.array(array1), np.array(array2),
        labels=["SAM-ViT", "SAM2-Hiera"],
        figsize=(12, 7),
        save_path="fig2_Sensitivity_analysis_v5.png",
    )

    plt.show()

    # # 额外的统计信息
    # print(f"\nStatistical Summary:")
    # print(f"Array 1 - Mean: {np.mean(array1):.3f}, Std: {np.std(array1):.3f}")
    # print(f"Array 2 - Mean: {np.mean(array2):.3f}, Std: {np.std(array2):.3f}")

    # # 创建另一个风格的图表示例
    # fig2, ax2 = create_scientific_line_plot(
    #     array1, array2,
    #     labels=["Method A", "Method B"],
    #     title="Performance Comparison",
    #     xlabel="Time Point",
    #     ylabel="Efficiency Score",
    #     figsize=(10, 6)
    # )

    # plt.show()