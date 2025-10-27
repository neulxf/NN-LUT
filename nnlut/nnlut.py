import torch, torch.nn as nn, torch.nn.functional as F
from fplut.util import generate_all_fp16_values

class OneHiddenReLU(nn.Module):
    def __init__(self, hidden, in_scale=1.0):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden, bias=True)  # 输出隐藏层 y_i
        self.fc2 = nn.Linear(hidden, 1, bias=False) # 权重 m_i
        self.in_scale = in_scale                    # 可选：输入缩放
    def forward(self, x):
        x = x * self.in_scale
        h = F.relu(self.fc1(x))
        return self.fc2(h)
    
    def init_custom_weights(self, op_type, input_range=None, x_samples=None,
                            use_curvature=False, num_pts=100_000):
        """
        根据op_type初始化权重和偏置：
        op_type: 'gelu', 'softmax', 'divide', 'layernorm', 'exp', 'sigmoid', 'silu
        新增参数：
        use_curvature: 是否使用二阶导数驱动切分
        num_pts: 采样点数
        """
        # === ② 二阶导数驱动切分 ===
        if use_curvature and input_range is not None and op_type in ["gelu", "silu"]:
            low, high = input_range
            H = self.fc1.out_features
            xs = torch.linspace(low, high, num_pts)
            d = _second_deriv_abs(op_type, xs, H)
            
            n = torch.ones(H)
            b = -n * d

            with torch.no_grad():
                self.fc1.weight.copy_(n.view(H, 1))
                self.fc1.bias.copy_(b)
            return

        if op_type in ['gelu', 'silu']:
            # 权重、偏置均为标准正态分布
            if True:
                nn.init.normal_(self.fc1.weight)
                nn.init.normal_(self.fc1.bias)
                
            # ---- 手动固定 SiLU 关键断点：-8, 7, 以及输入区间端点 ----
            # if op_type == "silu" and input_range is not None:
            #     low, high = input_range
            #     fixed_ds = [-8.0, 7.0]
            #     # 如果区间超出 [-8,7]，则也把区间端点加入
            #     if low < -8.0 or high > 7.0:
            #         fixed_ds += [low, high]
            #     # 去重并保证不超过 hidden 单元数
            #     fixed_ds = list(dict.fromkeys(fixed_ds))[: self.fc1.out_features]

            #     # 先确保权重不为 0，若为 0 则设为 1
            #     w = self.fc1.weight.data
            #     b = self.fc1.bias.data
            #     for i, d_val in enumerate(fixed_ds):
            #         if w[i, 0] == 0.0:
            #             w[i, 0] = 1.0
            #         # 令 -b_i / w_i = d_val  ⇒  b_i = -d_val * w_i
            #         b[i] = -d_val * w[i, 0]
                
        elif op_type == 'softmax':
            # 权重、偏置均为正随机（均匀分布在(0, 1)）
            nn.init.uniform_(self.fc1.weight, a=0.01, b=1.0)
            nn.init.uniform_(self.fc1.bias, a=0.01, b=1.0)
        elif op_type == 'divide':
            # 权重为负随机，偏置为正随机
            nn.init.uniform_(self.fc1.weight, a=-1.0, b=-0.01)
            nn.init.uniform_(self.fc1.bias, a=0.01, b=1.0)
        elif op_type == 'layernorm':
            # 权重为负随机，偏置为正随机
            nn.init.uniform_(self.fc1.weight, a=-1.0, b=-0.01)
            nn.init.uniform_(self.fc1.bias, a=0.01, b=1.0)
        else:
            raise NotImplementedError(f"Not implemented for op_type: {op_type}")

@torch.no_grad()
def _second_deriv_abs(op_type: str, xs: torch.Tensor, num_segs: int) -> torch.Tensor:
    xs = xs.clone()
    if op_type == "silu":
        y = xs * torch.sigmoid(xs)
    elif op_type == "gelu":
        # 使用近似 Gelu = x * 0.5 * (1 + tanh(√(2/π)(x + 0.044715x³)))
        y = F.gelu(xs)
    else:
        raise ValueError(op_type)
    dv = xs[1] - xs[0]
    fp = (y[2:] - y[:-2]) / (2*dv)
    fp = torch.cat([fp[:1], fp, fp[-1:]])

    fpp     = (y[:-2] - 2*y[1:-1] + y[2:]) / dv**2
    fpp     = torch.cat([fpp[:1], fpp, fpp[-1:]])
    rho     = (fpp.abs() + 1e-16).pow(1/3)
    rho     /= rho.sum()
    cdf     = torch.cumsum(rho, 0)
    cdf     = torch.cat([torch.zeros(1).to(xs.device), cdf])
    edges_idx  = torch.searchsorted(cdf, torch.linspace(0, 1, num_segs).to(xs.device))
    edges_idx[-1] = xs.shape[0] - 1
    return xs[edges_idx]

def export_lut(model):
    # 取参数
    n  = model.fc1.weight.data.squeeze()   # shape [H]
    b  = model.fc1.bias.data.squeeze()     # shape [H]
    m  = model.fc2.weight.data.squeeze()   # shape [H]
    # 断点 d_i = -b_i / n_i
    d = -b / n
    order = torch.argsort(d)
    n, b, m, d = [x[order] for x in (n, b, m, d)]

    # ---- 2. 预计算正/负贡献 ----
    pos = (n > 0)
    neg = ~pos

    pos_slope      = (m * n)              * pos      # m_i n_i  (pos only)
    pos_intercept  = (m * b)              * pos      # m_i b_i
    neg_slope      = (m * n)              * neg
    neg_intercept  = (m * b)              * neg

    # 前缀(≤k) 与后缀(>k) 累和
    prefix_pos_slope      = torch.cat([torch.zeros(1), pos_slope.cumsum(0)])
    prefix_pos_intercept  = torch.cat([torch.zeros(1), pos_intercept.cumsum(0)])

    suffix_neg_slope      = torch.cat([neg_slope.flip(0).cumsum(0).flip(0), torch.zeros(1)])
    suffix_neg_intercept  = torch.cat([neg_intercept.flip(0).cumsum(0).flip(0), torch.zeros(1)])

    # ---- 3. 逐段组合 ----
    # segment k 对应区间 (d_k , d_{k+1}), k = 0..H
    s = (prefix_pos_slope      + suffix_neg_slope)     .tolist()
    t = (prefix_pos_intercept  + suffix_neg_intercept) .tolist()

    # ---- 4. 添加首尾无穷断点 ----
    d = torch.cat([torch.tensor([-float("inf")]), d, torch.tensor([float("inf")])]).tolist()
    return s, t, d

import bisect
def lut_forward(x, s, t, d):
    idx = bisect.bisect_right(d, x) - 1
    return s[idx] * x + t[idx]

def lut_forward_fast(x, s, t, d):
    cut_indices = (
        torch.bucketize(x, torch.tensor(d), right=True).clamp(
            max=len(s)
        )
        - 1
    ).clip(0)
    y = torch.tensor(s)[cut_indices] * x + torch.tensor(t)[cut_indices]
    return y

def test_exp(hidden):
    net = OneHiddenReLU(hidden)
    x = fp16_values[fp16_values >= -200]
    x = x[x <= 0].float().unsqueeze(1)
    y = torch.exp(x)
    net.init_custom_weights('softmax')

    # 2) 训练
    # 学习率余弦衰减
    max_step = 15000
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_step)

    for step in range(max_step + 1):
        idx = torch.randint(len(x), (2048,))
        loss = (net(x[idx]) - y[idx]).abs().max()
        if step % 1000 == 0:
            print(f"step {step} loss: {loss.item()}")
        loss.backward(); opt.step(); opt.zero_grad()

    # 3) 导出 LUT
    s, t, d = export_lut(net)

    # 4) 误差检查
    with torch.no_grad():
        pred = torch.tensor([lut_forward(v.item(), s, t, d) for v in x])
    err = (pred - y.squeeze()).abs().max().item()

    pred = torch.tensor([lut_forward(v.item(), s, t, d) for v in fp16_values[fp16_values <= 0].item()])
    y = torch.exp(fp16_values[fp16_values<=0])
    err = (pred - y).abs() / y.abs().clamp(min=2**-14)
    print(f"max |error| = {err:.6f}")   

if __name__ == "__main__":
    import torch, math, matplotlib.pyplot as plt
    torch.manual_seed(0)

    fp16_values = generate_all_fp16_values()
    fp16_values = torch.tensor(fp16_values, dtype=torch.float16).sort(descending=False)[0]

    cut_points = [16, 32, 64, 128, 256]
    for hidden in cut_points:
        test_exp(hidden)