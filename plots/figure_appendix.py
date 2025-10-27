import math
from functools import partial
from typing import List, Optional, Union, overload
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import os

from fplut.nl.generate_split_points import generate_lut_table_v2
from fplut.nl.base import verbose_lut


from fplut.nl.base import lut_v2_fast
from fplut.util import generate_all_fp16_values     
FP16_ALL = torch.tensor(generate_all_fp16_values(), dtype=torch.float16).sort()[0]

OP_CFG = {
    "hardswish": {"func": lambda x: F.hardswish(x),
                  "range": (-150., 150.),
                  "init": "hardswish",
                  "num_points": 32,
                  "table_cfg": dict(func_name="hardswish", unique_name="hardswish", func=F.hardswish, x_range=(-3, 65504), left_edge=-3, right_edge=65504)},
    "sigmoid": {"func": lambda x: F.sigmoid(x),
                  "range": (-150., 150.),
                  "init": "sigmoid",
                  "num_points": 32,
                  "table_cfg": dict(func_name="sigmoid", unique_name="sigmoid", func=F.sigmoid, x_range=(-17.34375, 8.3203125), left_edge=-17.34375, right_edge=8.3203125)},
    "tanh": {"func": lambda x: F.tanh(x),
                  "range": (-150., 150.),
                  "init": "tanh",
                  "num_points": 32,
                  "table_cfg": dict(func_name="tanh", unique_name="tanh", func=F.tanh, x_range=(-4.5078,4.5078), left_edge=-4.5078, right_edge=4.5078)},
    "mish": {"func": lambda x: F.mish(x),
                  "range": (-150., 150.),
                  "init": "mish",
                  "num_points": 32,
                  "table_cfg": dict(func_name="mish", unique_name="mish", func=F.mish, x_range=(-20.3438,65504), left_edge=-20.3438, right_edge=65504)},
}

def test_op(op_key, hidden, save_dir, plot_large):
    cfg   = OP_CFG[op_key]
    func  = cfg["func"]

    table = generate_lut_table_v2(cfg["table_cfg"])
    lut_cut_points, lut_values, lut_scale = verbose_lut(table)

    # 3.1 构造输入样本

    lo, hi  = cfg["range"]
    mask    = (FP16_ALL >= lo) & (FP16_ALL <= hi)
    x_train = FP16_ALL[mask]
    x_train = x_train.cuda()

    x_train = x_train.unsqueeze(1)            # shape [N, 1]
    y_train = func(x_train.float())
    y = lut_v2_fast(x_train, lut_cut_points, lut_values, lut_scale)

    # --- 定义每个算子的可视化区间 ---
    if plot_large:
        PLOT_RANGE = {
            "hardswish":   (-150., 150.),
            "sigmoid":   (-150., 150.),
            "tanh":   (-150., 150.),
            "mish":   (-150., 150.),
            "softplus":   (-150., 150.),
            "hard_sigmoid":   (-150., 150.),
        }
    else:
        PLOT_RANGE = {
            "hardswish":   (-5., 5.),
            "sigmoid":   (-5., 5.),
            "tanh":   (-5., 5.),
            "mish":   (-5., 5.),
            "softplus":   (-5., 5.),
            "hard_sigmoid":   (-5., 5.),
        }
    xs_plot = x_train.detach().cpu().numpy()
    ys_true = y_train.detach().cpu().numpy()
    ys_pred = y.detach().cpu().numpy()
    abs_err_np = np.abs(ys_true - ys_pred)

    x_lo, x_hi = PLOT_RANGE.get(op_key, (xs_plot.min(), xs_plot.max()))

    mask_vis = (FP16_ALL >= x_lo) & (FP16_ALL <= x_hi)
    xs_vis   = FP16_ALL[mask_vis]
    ys_true_vis = func(xs_vis)
    ys_pred_vis = lut_v2_fast(xs_vis, lut_cut_points, lut_values, lut_scale)
    abs_err_vis = np.abs(ys_true_vis - ys_pred_vis)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(6, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )

    # --- Top: ground truth & piece‑wise linear ---
    ax_top.plot(xs_vis, ys_true_vis, label="ground‑truth", color="#09ADB8")

    # 逐段画直线，并在 cut‑point 处用大圆点标示
    flag = True
    for i in range(len(table.table)):
        if i == len(table.table) - 1:
            if op_key not in ["tanh", "sigmoid"]:
                continue
            else:
                x0 = table.index[i]
                x1 = torch.tensor(x_hi)
                y0 = table.table[i]
                y1 = func(x1)
        else:
            x0, x1 = table.index[i], table.index[i + 1]
            y0, y1 = table.table[i], table.table[i + 1]

        if i == 0 and op_key in ["exp", "gelu", "silu", "hardswish", "sigmoid", "tanh", "mish"]:
            x1 = x0
            x0 = torch.tensor(x_lo)

        if x0 < x_lo:
            continue
        if x0 > x_hi:
            break
        if x1 > x_hi:
            x1 = torch.tensor(x_hi)
        if x0 < x_lo:
            # continue
            x0 = torch.tensor(x_lo)

        y0, y1 = func(x0), func(x1)


        # 计算该段在可视区间内的端点
        x_start = x0.item()
        x_end = x1.item()
        y_start = y0.item()
        y_end = y1.item()

        # print(x_start, x_end, y_start, y_end)

        label = "NLI approx" if flag else "_nolegend_"  # “第一次”才进图例
        ax_top.plot(
            [x_start, x_end],
            [y_start, y_end],
            color="#ff6200",
            linewidth=1.3,
            label=label,  # 仅首段加 label
        )
        flag = False    

    # cut‑points 圆点
    cps_in = table.cut_points[(table.cut_points >= x_lo) & (table.cut_points <= x_hi)]
    if len(cps_in) > 0:
        # 取对应段计算 ŷ
        y_cps = []
        for cp_val in cps_in:
            y_cps.append(func(cp_val).item())
        # 实心小圆点标注 cut‑points
        ax_top.scatter(cps_in, y_cps, s=15, color="red", zorder=3)

    ax_top.set_ylabel("y")
    # ax_top.set_ylim([ys_true_vis.min().item(), ys_true_vis.max().item() + 0.5])
    ax_top.set_title(f"{op_key} | H={hidden}")
    ax_top.set_xlim(x_lo, x_hi)
    # ax_top.legend()

    # --- Bottom: absolute error ---
    abs_err_vis[abs_err_vis == 0] = 1e-12
    ax_bot.plot(xs_vis, abs_err_vis, color="tab:green")
    ax_bot.set_ylabel("|error|")
    ax_bot.set_xlabel("x")
    ax_bot.set_yscale("log")
    ax_bot.set_xlim(x_lo, x_hi)
    ax_bot.grid(True, linestyle=":")

    os.makedirs(save_dir, exist_ok=True)
    plot_fname = f"{save_dir}/{op_key}_H{hidden}.png"
    fig.savefig(plot_fname, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="plot/NLI_lut_plot")
    parser.add_argument("--plot_large", action="store_true")
    args = parser.parse_args()
    for op_key in OP_CFG.keys():
        print(f"Testing {op_key}...")
        test_op(op_key, 11, args.save_dir, args.plot_large)