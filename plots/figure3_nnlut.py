# run_nnlut_bench.py  ——  训练 NN‑LUT，评测误差，同时保存切分点/slopes/bias 以及可视化 PNG
import math, bisect, argparse, itertools, json
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import pathlib
import numpy as np
# -------------------------------------------------------------------------
# Choose computation device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Running on {DEVICE}")
# -------------------------------------------------------------------------
from fplut.util import generate_all_fp16_values     
from nnlut.nnlut import OneHiddenReLU, export_lut, lut_forward_fast   
################################################################################
# 1. 准备 FP16 值全集（≈1e5 个）
################################################################################
FP16_ALL = torch.tensor(generate_all_fp16_values(), dtype=torch.float16) \
                .float().sort()[0]            # [-65504, 65504]

################################################################################
# 2. 不同非线性算子的配置（来自论文 Table 1）
################################################################################
OP_CFG = {
    "silu":      {"func": F.silu,
                  "range": (-150., 150),    
                   "init": "silu"},
    "gelu":      {"func": lambda x: F.gelu(x, approximate="none"),
                  "range": (-150., 150.),     "init": "gelu"},
    "exp":       {"func": torch.exp,      "range": (-256., 0.), "init": "softmax"},
    "divide":    {"func": lambda x: 1./x, "range": (1., 1024.), "init": "divide"},
    "invsqrt":   {"func": lambda x: 1./torch.sqrt(x),
                  "range": (0.1, 1024.),  "init": "layernorm"},
}

################################################################################
# 3. 训练 + 导出 LUT + 评测
################################################################################
def train_and_eval(op_key: str, hidden: int, full_fp16=False,
                   steps=15_000, batch=2048, lr=1e-3, seed=0, save_dir="nnlut_bench"):
    torch.manual_seed(seed)
    cfg   = OP_CFG[op_key]
    func  = cfg["func"]

    # 3.1 构造输入样本
    if full_fp16:
        x_train = FP16_ALL.clone()
        mask    = torch.ones_like(x_train, dtype=torch.bool)   # 全域评测
    else:
        lo, hi  = cfg["range"]
        mask    = (FP16_ALL >= lo) & (FP16_ALL <= hi)
        x_train = FP16_ALL[mask]
        if op_key not in ["divide", "invsqrt"]:
            x_train = torch.linspace(lo, hi, 1000000, device=DEVICE)

    x_train = x_train.unsqueeze(1)            # shape [N, 1]
    y_train = func(x_train)

    # 将数据搬到 GPU / 目标设备
    x_train = x_train.to(DEVICE)
    y_train = y_train.to(DEVICE)

    # 3.2 构建/初始化模型
    net = OneHiddenReLU(hidden).to(DEVICE)
    net.init_custom_weights(cfg["init"], cfg.get("range", None), use_curvature=False)

    # 3.3 训练
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=5e-6)
    for s in range(steps+1):
        idx   = torch.randint(x_train.shape[0], (batch,), device=DEVICE)
        loss  = (net(x_train[idx]) - y_train[idx]).abs().mean()
        loss.backward(); opt.step(); opt.zero_grad(); sch.step()
        if s % 3000 == 0:
            print(f"[{op_key:8s}|H={hidden:4d}] step {s:>5d} | mean AE = {loss.item():.4e}")

    # 3.4 导出 & 前向
    net_cpu = net.to("cpu")
    s, t, d = export_lut(net_cpu)
    def lut(x):
        # x: 1-D Tensor
        return torch.tensor([s[bisect.bisect_right(d, v.item())-1]*v + t[bisect.bisect_right(d, v.item())-1]
                             for v in x], dtype=torch.float32)

    with torch.no_grad():
        x_eval = FP16_ALL if full_fp16 else FP16_ALL[mask]
        x_eval_cpu = x_eval.cpu()          # bisect 在 CPU 上运行
        pred = lut_forward_fast(x_eval_cpu, s, t, d)
        gt   = func(x_eval_cpu).float()

        abs_err   = (pred - gt).abs()
        rel_err   = abs_err / gt.abs().clamp_min(2**-14)   # 与论文一致的下限

    # ---------- 结果统计 ----------
    AE_mean = abs_err.mean().item()        # 绝对误差均值

    # ---------- 保存 LUT 细节 ----------
    detail_fname = f"{save_dir}/lut_details_{op_key}_H{hidden}_{'full' if full_fp16 else 'sub'}.json"
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(detail_fname, "w") as f:
        json.dump({
            "op": op_key,
            "hidden": hidden,
            "cut_points": d,
            "slopes": s,
            "biases": t
        }, f, indent=2)

    # ---------- 绘图 ----------
    xs_plot = x_eval_cpu.numpy()
    ys_true = gt.numpy()
    ys_pred = pred.numpy()
    abs_err_np = np.abs(ys_true - ys_pred)

    # --- 定义每个算子的可视化区间 ---
    PLOT_RANGE_SMALL = {
        "silu":   (-5., 5.),
        "gelu":   (-6., 6.),
        "exp":    (-20., 3.),
        "divide": (1, 40.),
        "invsqrt": (0.1, 40.),
    }
    PLOT_RANGE_LARGE = {
        "silu":   (-150., 150.),
        "gelu":   (-150., 150.),
        "exp":    (-256., 3.),
        "divide": (1., 1024.),
        "invsqrt": (0.1, 1024.),
    }
    for PLOT_RANGE in [PLOT_RANGE_SMALL, PLOT_RANGE_LARGE]:
        x_lo, x_hi = PLOT_RANGE.get(op_key, (xs_plot.min(), xs_plot.max()))

        mask_vis = (FP16_ALL >= x_lo) & (FP16_ALL <= x_hi)
        xs_vis   = FP16_ALL[mask_vis]
        ys_true_vis   = func(xs_vis).float()
        ys_pred_vis   = lut_forward_fast(xs_vis, s, t, d)
        abs_err_vis = np.abs(ys_true_vis - ys_pred_vis)

        # 准备 cut‑points / 线段
        cps = np.array(d)
        slopes = np.array(s)
        biases = np.array(t)

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
        for i in range(len(slopes)):
            x0, x1 = cps[i], cps[i + 1]
            # 若该段完全在可视区间之外，跳过
            if x1 < x_lo or x0 > x_hi:
                continue
            seg_x = np.array([max(x0, x_lo), min(x1, x_hi)])
            seg_y = slopes[i] * seg_x + biases[i]
            ax_top.plot(
                seg_x,
                seg_y,
                color="#F7BBBD",
                linewidth=1.3,
                label="NN‑LUT approx" if flag else None,  # 仅首段加 label
            )
            flag = False

        # cut‑points 圆点
        cps_in = cps[(cps >= x_lo) & (cps <= x_hi)]
        if len(cps_in) > 0:
            # 取对应段计算 ŷ
            y_cps = []
            for cp_val in cps_in:
                idx = max(0, np.searchsorted(cps, cp_val) - 1)
                y_cps.append(slopes[idx] * cp_val + biases[idx])
            # 实心小圆点标注 cut‑points
            ax_top.scatter(cps_in, y_cps, s=15, color="red", zorder=3)

        ax_top.set_ylabel("y")
        ax_top.set_title(f"{op_key} | H={hidden}")
        ax_top.set_xlim(x_lo, x_hi)
        # ax_top.legend()

        # --- Bottom: absolute error ---
        ax_bot.plot(xs_vis, abs_err_vis, color="tab:green")
        ax_bot.set_ylabel("|error|")
        ax_bot.set_xlabel("x")
        ax_bot.set_yscale("log")
        ax_bot.set_xlim(x_lo, x_hi)
        ax_bot.grid(True, linestyle=":")

        plot_fname = f"{save_dir}/lut_plot_{op_key}_H{hidden}_{'full' if full_fp16 else 'sub'}_{'small' if PLOT_RANGE == PLOT_RANGE_SMALL else 'large'}.png" 
        fig.savefig(plot_fname, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return dict(
        op        = op_key,
        hidden    = hidden,
        segs      = hidden,           # 段数 = H
        cut_pts   = hidden + 1,
        AE_max    = abs_err.max().item(),
        AE_mean   = AE_mean,
        ARE_mean  = rel_err.mean().item(),
        detail    = detail_fname,
        plot      = plot_fname,
        full_fp16 = full_fp16,
    )

################################################################################
# 4. CLI & 主流程
################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops",   nargs="+", default=list(OP_CFG.keys()),
                        help="要跑的非线性算子")
    parser.add_argument("--cuts",  nargs="+", type=int,
                        default=[17],   # 论文默认：段数=16 → cut=17
                        help="切分点数量列表（N）→ 隐层大小= N-1")
    parser.add_argument("--extra_fp16", action="store_true",
                        help="另外在全 FP16 空间上测试 11 与 259 cut-points")
    parser.add_argument("--steps", type=int, default=45000,
                        help="训练步数")
    parser.add_argument("--batch", type=int, default=4096,
                        help="批量大小")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="学习率")
    parser.add_argument("--seed", type=int, default=0,
                        help="随机种子")
    parser.add_argument("--save_dir", type=str, default="nnlut_bench",
                        help="保存结果的目录")
    args = parser.parse_args()

    exp_cuts = args.cuts
    if args.extra_fp16:
        exp_cuts += [11, 259]

    results = []
    for op, N in itertools.product(args.ops, exp_cuts):
        res = train_and_eval(op_key=op, hidden=N-1,
                             full_fp16=(N in [11, 259] and args.extra_fp16),
                             steps=args.steps, batch=args.batch, lr=args.lr, seed=args.seed,
                             save_dir=args.save_dir)
        results.append(res)

    # 打印表格 / 保存 JSON
    import pandas as pd, tabulate, json, pathlib
    df = pd.DataFrame(results)
    print(tabulate.tabulate(df, headers="keys", tablefmt="github", floatfmt=".3e"))
    pathlib.Path(args.save_dir, "nnlut_bench.json").write_text(json.dumps(results, indent=2))