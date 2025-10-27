from math import pi,sqrt
from typing import Callable, Literal, Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from tqdm import tqdm

M_MAX = 11
REL_L1_EPS = 2 ** (-14.0)
ERR_TYPE = "fp32"
Torchfunc2Name = dict()
Name2TorchFunc = dict()
Name2TritonFunc = dict()


def register_triton_func(torch_func: Callable, func_name: str):
    def wrapper(triton_func: Callable):
        Torchfunc2Name[torch_func] = func_name
        Name2TritonFunc[func_name] = triton_func
        Name2TorchFunc[func_name] = torch_func
        return triton_func

    return wrapper


@register_triton_func(F.celu, "celu")
@triton.jit
def _celu(x, alpha: tl.constexpr = 1.0):
    if alpha is None:
        alpha = 1.0
    return tl.where(x > 0, x, alpha * (tl.exp(x / alpha) - 1))


@register_triton_func(F.tanh, "tanh")
@triton.jit
def _tanh(x):
    return 2* tl.sigmoid(2*x) - 1 # stable
    # return (1-tl.exp(-2*x))/(1+tl.exp(-2*x))
    # exp_val = tl.exp(2.0 * x)
    # return (exp_val - 1.0) / (exp_val + 1.0)


@register_triton_func(F.silu, "silu")
@triton.jit
def _silu(x):
    return tl.sigmoid(x) * x


@register_triton_func(F.gelu, "gelu")
@register_triton_func(F.gelu, "gelu_standard")
@triton.jit
def _gelu_standard(x):
    sqrt2 = tl.sqrt(2.0)
    cdf = 0.5 * (1.0 + tl.erf(x / sqrt2))
    return x * cdf


from functools import partial


@register_triton_func(partial(F.gelu, approximate="tanh"), "gelu_tanh")
@triton.jit
def _gelu_tanh(x):
    # alpha = 0.7978845608028654
    # alpha = tl.sqrt(2 / pi)
    alpha = tl.sqrt(0.6366197723675814)
    beta = 0.044714998453855515
    return 0.5 * x * (1.0 + _tanh(alpha * (x + beta * x * x * x)))


@register_triton_func(F.gelu, "gelu")
@triton.jit
def _gelu(x, approximate: tl.constexpr = "tanh"):
    if approximate == "tanh":
        return _gelu_tanh(x)
    else:
        return _gelu_standard(x)


@register_triton_func(F.hardsigmoid, "hardsigmoid")
@triton.jit
def _hard_sigmoid(x, alpha: tl.constexpr, beta: tl.constexpr):
    if alpha is None:
        alpha = 1 / 6
    if beta is None:
        beta = 0.5
    return tl.clamp(x * alpha + beta, 0, 1)


@register_triton_func(F.hardswish, "hardswish")
@triton.jit
def _hard_swish(x):
    mask1 = x <= -3
    mask2 = x >= 3
    x = tl.where(mask1, 0, x)
    x = tl.where(mask2, x, x * (x + 3) / 6)
    return x


@register_triton_func(F.relu, "relu")
@triton.jit
def _relu(x):
    return tl.where(x > 0, x, 0)


@register_triton_func(F.leaky_relu, "leaky_relu")
@triton.jit
def _leaky_relu(x, negative_slope):
    return tl.where(x > 0, x, x * negative_slope)


@register_triton_func(torch.exp, "softmax_exp")
@register_triton_func(torch.exp, "exp")
@triton.jit
def _exp(x):
    return tl.exp(x)


@register_triton_func(torch.reciprocal, "reciprocal")
@triton.jit
def _reciprocal(x):
    return 1./x

@register_triton_func(lambda x: 1./torch.sqrt(x), "rsqrt")
@triton.jit
def _rsqrt(x):
    return 1./tl.sqrt(x)

@register_triton_func(torch.log, "log")
@triton.jit
def _log(x):
    return tl.log(x)


@register_triton_func(None, "inverse_sigmoid")
@triton.jit
def _inverse_sigmoid(x):
    return tl.log(x / (1 - x))


@register_triton_func(F.elu, "elu")
@triton.jit
def _elu(x, alpha: tl.constexpr):
    return tl.where(x > 0, x, alpha * (tl.exp(x) - 1))


@register_triton_func(F.selu, "selu")
@triton.jit
def _selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tl.where(x > 0, x, alpha * (tl.exp(x) - 1))


@register_triton_func(F.softplus, "softplus")
@triton.jit
def _softplus(x):
    return tl.log(1 + tl.exp(x))


@register_triton_func(lambda x:F.softsign(x.float()).half(), "softsign")
@triton.jit
def _softsign(x):
    return x / (1 + tl.abs(x))

@register_triton_func(F.mish, "mish")
@triton.jit
def _mish(x):
    return x * _tanh(_softplus(x))


@register_triton_func(torch.sin, "sin")
@triton.jit
def _sin(x):
    return tl.sin(x)


@register_triton_func(torch.cos, "cos")
@triton.jit
def _cos(x):
    return tl.cos(x)


@register_triton_func(torch.log, "log")
@triton.jit
def _log(x):
    return tl.log(x)


@register_triton_func(torch.log2, "log2")
@triton.jit
def _log2(x):
    return tl.log2(x)


@register_triton_func(torch.sinh, "sinh")
@triton.jit
def _sinh(x):
    return (tl.exp(x) - tl.exp(-x)) / 2


@register_triton_func(torch.sigmoid, "sigmoid")
@triton.jit
def _sigmoid(x):
    return tl.sigmoid(x)


@register_triton_func(torch.cosh, "cosh")
@triton.jit
def _cosh(x):
    return (tl.exp(x) + tl.exp(-x)) / 2.0


@triton.jit
def _FUNC_MAP(FUNC: tl.constexpr, x, FUNC_ARG1: tl.constexpr, FUNC_ARG2: tl.constexpr):
    if FUNC == "cos":
        return _cos(x)
    elif FUNC == "cosh":
        return _cosh(x)
    elif FUNC == "celu":
        return _celu(x,FUNC_ARG1)
    elif FUNC == "elu":
        return _elu(x, 1.0)
    elif FUNC == "exp" or FUNC == "softmax_exp":
        return _exp(x)
    elif FUNC == "gelu_standard":
        return _gelu_standard(x)
    elif FUNC == "gelu_tanh":
        return _gelu_tanh(x)
    elif FUNC == "gelu":
        return _gelu(x, FUNC_ARG1)
    elif FUNC == "gelu":
        if FUNC_ARG1 == "tanh":
            return _gelu_tanh(x)
        else:
            return _gelu_standard(x)
    elif FUNC == "hardsigmoid":
        if FUNC_ARG1 is None:
            FUNC_ARG1, FUNC_ARG2 = 1 / 6, 0.5
            return _hard_sigmoid(x, FUNC_ARG1, FUNC_ARG2)
        return _hard_sigmoid(x, FUNC_ARG1, FUNC_ARG2)
    elif FUNC == "hardswish":
        return _hard_swish(x)
    elif FUNC == "inverse_sigmoid":
        return _inverse_sigmoid(x)
    elif FUNC == "leakyrelu":
        return _leaky_relu(x, FUNC_ARG1)
    elif FUNC == "log":
        return _log(x)
    elif FUNC == "log2":
        return _log2(x)
    elif FUNC == "mish":
        return _mish(x)
    elif FUNC == "relu":
        return _relu(x)
    elif FUNC == "silu":
        return _silu(x)
    elif FUNC == "sigmoid":
        return _sigmoid(x)
    elif FUNC == "selu":
        return _selu(x)
    elif FUNC == "sin":
        return _sin(x)
    elif FUNC == "sinh":
        return _sinh(x)
    elif FUNC == "softplus":
        return _softplus(x)
    elif FUNC == "softsign":
        return _softsign(x)
    elif FUNC == "tanh":
        return _tanh(x)
    elif FUNC == "reciprocal":
        return _reciprocal(x)   
    elif FUNC == "rsqrt":
        return _rsqrt(x)
    else:
        raise ValueError(f"Unknown function: {FUNC}")


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 2048}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 512}),
    ],
    key=["BLOCK_SIZE"],
)
@triton.jit
def dp_step_kernel(
    d_ptr,
    path_ptr,
    fp16_values_ptr,  # 输入: FP16 值的指针
    f_values_ptr,  # 输入: 非线性函数值的指针
    n,  # 输入: FP16 值的总数
    loss: tl.constexpr,
    m: tl.constexpr,  # 输入: 当前的段索引 (m)
    BLOCK_SIZE: tl.constexpr,  # 常量: K 维度的并行块大小
    REL_L1_EPS: tl.constexpr,
    FUNC: tl.constexpr,
    FUNC_ARG1: tl.constexpr,
    FUNC_ARG2: tl.constexpr,
    ERR_TYPE: tl.constexpr = "fp64",
):
    # tl.static_print(FUNC,FUNC_ARG1,FUNC_ARG2)
    k = tl.program_id(0)
    # if (k >= n) or (k < m - 1):
    if (k>=n):
        return
    d_curr_ptr = d_ptr + (m * n)
    d_prev_ptr = d_ptr + (m - 1) * n
    path_curr_ptr = path_ptr + (m * n)

    if m == 0:
        xk = tl.load(f_values_ptr + k).to(tl.float32)
        best_err = 0.0
        best_err = best_err.to(tl.float32 if ERR_TYPE == "fp32" else tl.float64)
        for j in range(0, k, BLOCK_SIZE):
            j_offsets = tl.arange(0, BLOCK_SIZE) + j
            j_mask = j_offsets < k
            xs = tl.load(f_values_ptr + j_offsets, mask=j_mask, other=0.0).to(tl.float32)
            minus = tl.where(j_mask, xk - xs, 0).to(tl.float32 if ERR_TYPE == "fp32" else tl.float64)
            if loss == "l2":
                best_err += tl.sum(minus * minus)
            elif loss == "l1":
                best_err += tl.sum(tl.abs(minus))
            elif loss == "rel_l1":
                best_err += tl.sum(tl.abs(minus) / tl.maximum(tl.abs(xs), REL_L1_EPS))
        tl.store(d_curr_ptr + k, best_err)
        tl.store(path_curr_ptr + k, k)
        return

    best_err = float("inf")
    best_err = best_err.to(tl.float32 if ERR_TYPE == "fp32" else tl.float64)
    best_i = -1

    xk = tl.load(fp16_values_ptr + k).to(tl.float32)
    # for i in tl.range(k - 1, -1, -1):
    for i in tl.range(0, k):
        xi = tl.load(fp16_values_ptr + i).to(tl.float32)
        if (m == 1) or (m == 10):
            interval = xk - xi
            idx_min, idx_max = 0.0, 1.0
        else:
            interval = (xk - xi) / 32.0
            idx_min, idx_max = 0.0, 32.0
        # to_mul = tl.clamp((1.0 / interval),2**(-16),2**(15)).to(tl.float16, fp_downcast_rounding="rtne")
        to_mul = 1.0 / interval

        # 0. pre_err
        pre_err = tl.load(d_prev_ptr + i)
        # 1. tail_err
        tail_err = 0.0
        tail_err = tail_err.to(tl.float32 if ERR_TYPE == "fp32" else tl.float64)
        if m == (10):
            final_value = tl.load(f_values_ptr + k).to(tl.float32)
            for tail_start in tl.range(k, n, BLOCK_SIZE):
                tail_offsets = tail_start + tl.arange(0, BLOCK_SIZE)
                tail_mask = tail_offsets < n
                tail_vals = tl.load(f_values_ptr + tail_offsets, mask=tail_mask).to(tl.float32)
                minus = tl.where(tail_mask, tail_vals - final_value, tl.zeros_like(tail_vals)).to(
                    tl.float32 if ERR_TYPE == "fp32" else tl.float64
                )
                if loss == "l2":
                    tail_err = tail_err + tl.sum(minus * minus)
                elif loss == "l1":
                    tail_err = tail_err + tl.sum(tl.abs(minus))
                elif loss == "rel_l1":
                    tail_err = tail_err + (tl.sum(tl.abs(minus) / tl.maximum(tl.abs(tail_vals), REL_L1_EPS)))
        if (pre_err + tail_err) < best_err:
            # 2. seg_err
            seg_err = 0.0
            seg_err = seg_err.to(tl.float32 if ERR_TYPE == "fp32" else tl.float64)
            for j in range(
                i,
                k,
                BLOCK_SIZE,
            ):
                j_offsets = j + tl.arange(0, BLOCK_SIZE)
                j_mask = j_offsets < k

                xs = tl.load(fp16_values_ptr + j_offsets, mask=j_mask, other=0.0).to(tl.float32)
                ys = tl.load(f_values_ptr + j_offsets, mask=j_mask, other=0.0).to(tl.float32)

                # 1. seg err
                # position1 = tl.cast(xs - xi, tl.float16, fp_downcast_rounding="rtne")
                position1 = xs - xi
                position2 = position1 * to_mul
                position = position2
                idxs = tl.clamp(tl.floor(position2), idx_min, idx_max)  # f32
                idxs_plus1 = tl.minimum(idxs + 1, idx_max)

                table_x_low = xi + idxs * interval
                table_x_high = xi + idxs_plus1 * interval
                table_y_low = _FUNC_MAP(FUNC, table_x_low, FUNC_ARG1, FUNC_ARG2)
                table_y_high = _FUNC_MAP(FUNC, table_x_high, FUNC_ARG1, FUNC_ARG2)

                m1 = tl.clamp(position - idxs, 0.0, 1.0)

                right_sub_left = (table_y_high - table_y_low).to(tl.float32)

                # m2 = tl.clamp(1 - m1, 0.0, 1.0).to(tl.float32)
                # pred_y = m1 * table_y_high + m2 * table_y_low  # fp32
                pred_y = right_sub_left * (m1.to(tl.float32)) + (
                    table_y_low.to(tl.float32)
                )  # 实际是复用dot单元计算的因此会存在误差
                pred_y = pred_y

                ys_minus_pred_y = tl.where(j_mask, pred_y - ys, 0.0).to(
                    tl.float32 if ERR_TYPE == "fp32" else tl.float64
                )
                if loss == "l2":
                    seg_err = seg_err + tl.sum(ys_minus_pred_y * ys_minus_pred_y)
                elif loss == "l1":
                    seg_err = seg_err + tl.sum(tl.abs(ys_minus_pred_y))
                elif loss == "rel_l1":
                    seg_err = seg_err + tl.sum(tl.abs(ys_minus_pred_y) / tl.maximum(tl.abs(ys), REL_L1_EPS))

            current_err = seg_err + pre_err + tail_err
            if current_err < best_err:
                best_err = current_err
                best_i = i
    tl.store(d_curr_ptr + k, best_err)
    tl.store(path_curr_ptr + k, best_i)


def _generate_fp16_values():
    x = torch.tensor(range(65536), dtype=torch.uint16).view(torch.float16)
    mask = x.isnan() | x.isinf()
    x = x[~mask].sort()[0].contiguous()
    return x


def if_can_v2(func_name: str):
    if_can = True
    try:
        import triton
        import torch

        if_can = torch.cuda.is_available()
    except:
        if_can = False
    return if_can and (func_name in Name2TritonFunc)


def generate_cut_points_triton(
    func: Callable,
    fp16_values=None,
    func_name=None,
    loss: Literal["l1", "l2", "rel_l1"] = "rel_l1",
    left_edge=None,
    right_edge=None,
    max_search_points=25000,
    func_arg1: Any = None,
    func_arg2: Any = None,
):
    global REL_L1_EPS
    global FuncName2TltFunc
    assert loss in ["l1", "l2", "rel_l1"]
    if fp16_values is None:
        fp16_values = _generate_fp16_values()
    if max_search_points is None:
        max_search_points = 25000
    fp16_values = fp16_values.cuda().half().unique().sort()[0]
    f_values = func(fp16_values).half()
    mask = f_values.isnan() | f_values.isinf()
    fp16_values = fp16_values[~mask].half().contiguous()
    while fp16_values.numel() > (max_search_points + 1):
        n = fp16_values.numel()
        indices = torch.linspace(0, n - 1, n // 2, device="cuda").round_().int().unique()
        fp16_values = fp16_values[indices]
    fp16_values = fp16_values.half().contiguous()
    f_values = func(fp16_values).half()
    n = fp16_values.numel()
    # m 从0-10
    MAX_M = 11
    d = torch.full(
        (MAX_M, n), float("inf"), device="cuda", dtype=torch.float32 if ERR_TYPE == "fp32" else torch.float64
    ).contiguous()
    path = torch.zeros_like(d, dtype=torch.int16, device="cuda").contiguous()
    # m=0~10
    for m in tqdm(range(0, M_MAX), desc="dp_step_kernel"):
        torch.cuda.synchronize()
        grid_dp = lambda meta: (n,)
        dp_step_kernel[grid_dp](
            d_ptr=d,
            path_ptr=path,
            fp16_values_ptr=fp16_values,
            f_values_ptr=f_values,
            n=n,
            m=m,
            FUNC=func_name if func_name is not None else Torchfunc2Name[func],
            loss=loss,
            REL_L1_EPS=REL_L1_EPS,
            ERR_TYPE=ERR_TYPE,
            FUNC_ARG1=func_arg1,
            FUNC_ARG2=func_arg2,
        )
        torch.cuda.synchronize()
        if m == 0 and left_edge is not None:
            idx = (fp16_values - left_edge).abs().argmin()
            value = d[0, idx].item()
            d[0].fill_(float("inf"))
            d[0, idx] = value
        if m == MAX_M - 1 and right_edge is not None:
            idx = (fp16_values - right_edge).abs().argmin()
            value = d[MAX_M - 1, idx].item()
            d[MAX_M - 1].fill_(float("inf"))
            d[MAX_M - 1, idx] = value
    # 进行回溯
    best_points = []
    min_err = d[-1].min()
    last_point = d[-1].argmin()
    best_points.append(last_point)

    for i in reversed(range(1, MAX_M)):
        last_point = path[i][last_point]
        best_points.append(last_point)
    best_points.reverse()
    best_splits = [fp16_values[i].item() for i in best_points]
    best_fvalues = [f_values[i].item() for i in best_points]
    for i in range(len(best_points)):
        print(best_splits[i], best_fvalues[i])
    return min_err, best_splits, best_fvalues


def compare():
    torch.manual_seed(0)
    x = torch.arange(0, 65536).view(torch.float16)
    y = F.silu(x).half()
    mask = x.isnan() | x.isinf() | y.isnan() | y.isinf()
    x = x[~mask]
    x = torch.linspace(0, 10, 100)
    # x = torch.linspace(1,10000,10000,device="cuda")
    generate_cut_points_triton(F.silu, x, loss="l2")


if __name__ == "__main__":
    compare()
    exit(0)
    x = torch.arange(0, 65536).view(torch.float16)
    y = F.silu(x).half()
    mask = x.isnan() | x.isinf() | y.isnan() | y.isinf() | (x < -100) | (x > 100)
    x = x[~mask]
    # y = y[~mask]
    # x = torch.linspace(1,10000,10000,device="cuda")
    # x = torch.linspace(-32768,32767,100000,device="cuda")
    # for t in ["1e-5", "1e-4", "1e-3", "1e-2", "1e-1", "1"]:
    #     REL_L1_EPS = float(t)
    #     generate_cut_points_triton(F.silu, x, loss="rel_l1")
    # x = x[x.numel()//2-100:x.numel()//2+100]
    generate_cut_points_triton(F.silu, x, loss="rel_l1")
    generate_cut_points_triton(F.silu, x, loss="l2")
    generate_cut_points_triton(F.silu, x, loss="l1")

# TODO 增加更多的误差评估手段
