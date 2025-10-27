import os
from functools import partial
from pathlib import Path
from typing import Dict, Optional

import appdirs
import torch
import torch.nn.functional as F

from ...base.fp16_lut_mid_search import generate_all_fp16_values, get_best_luts, mod_data
from ...base.tools import check_should_equal, init_tensor_with_fix_val, unit_test
from ...base.xh2_fp_lut import NewTable, cut_points_dict

_best_cut_points_cache: Dict[str, list] = {}

def generate_cut_points(function, name: Optional[str] = None, device: torch.device = torch.device("cpu")):
    fp16_values = generate_all_fp16_values()
    fp16_values = torch.tensor(fp16_values, dtype=torch.float16).sort(descending=False)[0]
    data = fp16_values

    if name is None:
        if isinstance(function, partial):
            func = function.func
        else:
            func = function

    if name is None:
        name = ""

    if name in ["log", "divide"]:
        data = data[data > 0]
    elif "inversesigmoid" in name:
        data = data[data.abs() < 1]
    elif name in ["exp"]:
        data = data[data <= 0]
    elif name in ["sin", "cos"]:
        data = mod_data(data)
    elif name in ["sin_update", "cos_update", "tanh"]:
        data = mod_data(data, positive_only=True)
    elif name in ["tan", "tan_update"]:
        data = mod_data(data, cycle=torch.pi)
    elif name in ["acos", "asin", "atanh"]:  # [-1, 1]
        data = data[data <= 1]
        data = data[data >= -1]
    elif name in ["acosh"]:
        data = data[data >= 1]
    else:
        out = function(data)
        is_valid_mask = torch.isnan(out).logical_not()
        data = data[is_valid_mask]

    global _best_cut_points_cache
    if name is None or len(name) == 0:
        best_cut_points, _, _ = get_best_luts(function, data.to(device), device=device)
    elif name in _best_cut_points_cache:
        best_cut_points = _best_cut_points_cache[name]
    else:
        cache_dir = appdirs.user_cache_dir("xhquantool")
        cut_points_file = Path(cache_dir) / f"lut_cut_points-{name}"
        cut_points_file.parent.mkdir(parents=True, exist_ok=True)
        if cut_points_file.exists():
            with open(cut_points_file, "r") as f:
                best_cut_points = f.readlines()
            best_cut_points = [float(i) for i in best_cut_points]
            _best_cut_points_cache[name] = best_cut_points
        else:
            best_cut_points, _, _ = get_best_luts(function, data.to(device), device=device)
            _best_cut_points_cache[name] = best_cut_points
            with open(cut_points_file, "w") as f:
                for i in best_cut_points:
                    if isinstance(i, torch.Tensor):
                        i = i.item()
                    f.write(str(i) + "\n")

    return best_cut_points


def arbitrary_xh2a_default(
    x: torch.Tensor, function, table=None, name=None, use_gpu_lut: bool = True, cut_points: list = None
):
    assert x.dtype in [torch.float16], "unexpected dtype {}".format(x.dtype)

    device = x.device
    if table is None or not isinstance(table, NewTable):
        self = torch.nn.Module()
        assert function is not None and callable(function), "function is invaild"

        if cut_points is None:
            cut_points = generate_cut_points(function, name)

        setattr(self, "table", None)
        self.table = NewTable(function, cut_points, table_size=259, device=device)
        table = self.table

    if x.is_cuda:
        x = x.clamp(min=table.cut_points[0], max=table.cut_points[-1])
    else:
        x = x.clone()
        x[x < table.cut_points[0]] = table.cut_points[0]
        x[x > table.cut_points[-1]] = table.cut_points[-1]

    y = table(x, use_gpu_lut=use_gpu_lut)
    y[y == 0] = 0  # convert all neg zero to zero

    if y.is_cuda:
        y = y.clamp(-65504, 65504)
    else:
        y[y < -65504] = -65504
        y[y > 65504] = 65504
    return y


def arbitrary_xh2a_fast(
    x: torch.Tensor, function, table=None, name=None, use_gpu_lut: bool = True, cut_points: list = None
):
    assert x.dtype in [torch.float16], "unexpected dtype {}".format(x.dtype)
    y = function(x)
    return y


def test_cos(func=torch.cos):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if name in cut_points_dict:

        out = cos_xh2a_default(x)
    else:
        out = arbitrary_xh2a_default(x, func)

    check_should_equal(out.view(torch.int16)[1, 0, 3], [15357, 15357, 15357, 15357, 15357])
    check_should_equal(out.view(torch.int16)[0, 0, 0], [15360, -17408, 15360, -17408, -17408])


def test_softplus():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    func = partial(F.softplus, beta=1.0, threshold=20)
    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if name in cut_points_dict:
        from ..nl.softplus import softplus_xh2a_default

        out = softplus_xh2a_default(x)
    else:
        out = arbitrary_xh2a_default(x, func)

    check_should_equal(out.view(torch.int16)[1, 0, 3], [14777, 14778, 14779, 14779, 14780])
    check_should_equal(out.view(torch.int16)[0, 0, 0], [14732, 0, 14732, 31743, 0])

    func = partial(F.softplus, beta=2.0, threshold=20)
    out = arbitrary_xh2a_default(x, func)  # out.view(torch.int16)[0, 1, 2] == [13741, 13741, 13743, 13745, 13745]
    check_should_equal(out.view(torch.int16)[1, 0, 3], [13796, 13798, 13800, 13800, 13802])
    check_should_equal(out.view(torch.int16)[0, 0, 0], [13706, 0, 13706, 31743, 0])

    func = partial(F.softplus, beta=2.0, threshold=10)
    out = arbitrary_xh2a_default(x, func)  # out.view(torch.int16)[1, 0, 3] == [13796, 13798, 13802, 13802, 13804]
    check_should_equal(out.view(torch.int16)[1, 0, 3], [13796, 13798, 13802, 13802, 13804])
    check_should_equal(out.view(torch.int16)[0, 0, 0], [13705, 0, 13705, 31743, 0])


def test_exp():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    func = torch.exp
    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if name in cut_points_dict:
        from ..nl.exp import exp_xh2a_default

        out = exp_xh2a_default(x)
    else:
        out = arbitrary_xh2a_default(x, func)
    check_should_equal(out.view(torch.int16)[1, 0, 3], [15406, 15407, 15409, 15409, 15410])
    check_should_equal(out.view(torch.int16)[0, 0, 0], [15360, 0, 15360, 31743, 0])

    cut_points = [
        -65504.0,
        -18.453125,
        -11.015625,
        -7.7109375,
        -5.6875,
        -4.3046875,
        -2.435546875,
        -1.2529296875,
        -0.49072265625,
        -0.10150146484375,
        0.0,
    ]
    out = arbitrary_xh2a_default(x, func, cut_points=cut_points)

    from ..nl.base import lut

    out_ = lut(x, name="exp", cut_points=cut_points)
    assert (out != out_).sum() == 0, "not pass"

    check_should_equal(out.view(torch.int16)[1, 0, 3], [15360, 15360, 15360, 15360, 15360])
    check_should_equal(out.view(torch.int16)[0, 0, 0], [15360, 0, 15360, 15360, 0])


def test_dummy():
    raise RuntimeError("not pass")


def test_elu(func=F.elu):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if name in cut_points_dict:
        from ..nl.elu import elu_xh2a_default

        out = elu_xh2a_default(x)
    else:
        out = arbitrary_xh2a_default(x, func)

    check_should_equal(out.view(torch.int16)[1, 0, 3], [10601, 10601, 10601, 10729, 10729])
    check_should_equal(out.view(torch.int16)[0, 0, 0], [-28286, -17408, -28286, 31743, -17408])


def test_erf(func=torch.erf):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if name in cut_points_dict:
        from ..nl.elu import elu_xh2a_default

        out = elu_xh2a_default(x)
    else:
        out = arbitrary_xh2a_default(x, func)

    check_should_equal(out.view(torch.int16)[1, 0, 3], [10601, 10601, 10601, 10729, 10729])
    check_should_equal(out.view(torch.int16)[0, 0, 0], [-28286, -17408, -28286, 31743, -17408])


def test_gelu(func=F.gelu):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if name in cut_points_dict:
        from ..nl.gelu import gelu_xh2a_default

        out = gelu_xh2a_default(x)
    else:
        out = arbitrary_xh2a_default(x, func)

    check_should_equal(out.view(torch.int16)[1, 0, 3], [9710, 9729, 9751, 9772, 9794])
    check_should_equal(out.view(torch.int16)[0, 0, 0], [59, 0, 59, 31743, 0])


def test_gelu_tanh():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    func = partial(F.gelu, approximate="tanh")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    name = "gelu_tanh_update"
    if name in cut_points_dict:
        from ..nl.gelu import gelu_xh2a_default

        out = gelu_xh2a_default(x)
    else:
        out = arbitrary_xh2a_default(x, func, name=name)

    # check_should_equal(out.view(torch.int16)[1, 0, 3], [9710, 9729, 9751, 9772, 9794])
    # check_should_equal(out.view(torch.int16)[0, 0, 0], [59, 0, 59, 31743, 0])


def test_NewGELUActivation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import math

    def gelu_forward(input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

    func = gelu_forward

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    name = "NewGELUActivation"
    if name in cut_points_dict:
        from ..nl.gelu import gelu_xh2a_default

        out = gelu_xh2a_default(x)
    else:
        out = arbitrary_xh2a_default(x, func, name=name)

    # check_should_equal(out.view(torch.int16)[1, 0, 3], [9710, 9729, 9751, 9772, 9794])
    # check_should_equal(out.view(torch.int16)[0, 0, 0], [59, 0, 59, 31743, 0])


def test_hardsigmoid(func=F.hardsigmoid):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if name in cut_points_dict:
        from ..nl.hardsigmoid import hardsigmoid_xh2a_default

        out = hardsigmoid_xh2a_default(x)
    else:
        out = arbitrary_xh2a_default(x, func)

    check_should_equal(out.view(torch.int16)[1, 0, 3], [14351, 14351, 14351, 14351, 14352])
    check_should_equal(out.view(torch.int16)[0, 0, 0], [14336, 0, 14336, 15360, 0])


def test_hardswish(func=F.hardswish):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if name in cut_points_dict:
        from ..nl.hardswish import hardswish_xh2a_default

        out = hardswish_xh2a_default(x)
    else:
        out = arbitrary_xh2a_default(x, func)

    check_should_equal(out.view(torch.int16)[1, 0, 3], [9677, 9697, 9718, 9738, 9758])
    check_should_equal(out.view(torch.int16)[0, 0, 0], [-32667, 0, -32667, 31743, 0])


def test_leaky_relu(func=F.leaky_relu):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if name in cut_points_dict:
        from ..nl.leaky_relu import leaky_relu_xh2a_default

        out = leaky_relu_xh2a_default(x)
    else:
        out = arbitrary_xh2a_default(x, func)

    check_should_equal(out.view(torch.int16)[1, 0, 3], [10679, 10699, 10719, 10738, 10758])
    check_should_equal(out.view(torch.int16)[0, 0, 0], [0, -7907, 0, 31743, -7907])


def test_log(func=torch.log):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if name in cut_points_dict:
        from ..nl.log import log_xh2a_default

        out = log_xh2a_default(x)
    else:
        out = arbitrary_xh2a_default(x, func)

    check_should_equal(out.view(torch.int16)[1, 0, 3], [-15817, -15824, -15830, -15836, -15843])
    check_should_equal(out.view(torch.int16)[0, 0, 0], [-14433, -14433, -14433, 18828, -14433])


def test_mish(func=F.mish):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if name in cut_points_dict:
        from ..nl.mish import mish_xh2a_default

        out = mish_xh2a_default(x)
    else:
        out = arbitrary_xh2a_default(x, func)

    check_should_equal(out.view(torch.int16)[1, 0, 3], [9994, 10019, 10044, 10068, 10093])
    check_should_equal(out.view(torch.int16)[0, 0, 0], [-32177, 0, -32177, 31743, 0])


def test_pow2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    func = partial(torch.pow, exponent=2.0)
    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if name in cut_points_dict:
        from ..nl.pow2 import pow2_xh2a_default

        out = pow2_xh2a_default(x)
    else:
        out = arbitrary_xh2a_default(x, func)

    check_should_equal(out.view(torch.int16)[1, 0, 3], [6185, 6212, 6244, 6267, 6294])
    check_should_equal(out.view(torch.int16)[0, 0, 0], [448, 31743, 448, 31743, 31743])


def test_pow0dot4():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    func = partial(torch.pow, exponent=0.4)
    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    name = "pow0.4"

    if name in cut_points_dict:
        raise NotImplemented("function already ready")
    else:
        out = arbitrary_xh2a_default(x, func, name=name)

    # check_should_equal(out.view(torch.int16)[1, 0, 3], [6185, 6212, 6244, 6267, 6294])
    # check_should_equal(out.view(torch.int16)[0, 0, 0], [448, 31743, 448, 31743, 31743])


def test_prelu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    weight = torch.zeros(1, dtype=torch.float32)
    weight.fill_(x[0, 1, 2, 3])

    func = torch.prelu
    func = partial(func, weight=weight)

    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if name in cut_points_dict:
        from ..nl.prelu import prelu_xh2a_default

        out = prelu_xh2a_default(x)
    else:
        out = arbitrary_xh2a_default(x, func)

    check_should_equal(out.view(torch.int16)[1, 0, 3], [10680, 10698, 10719, 10738, 10757])
    check_should_equal(out.view(torch.int16)[0, 0, 0], [0, -7573, 0, 31743, -7573])


def test_reciprocal():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    func = torch.reciprocal
    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if name in cut_points_dict:
        from ..nl.reciprocal import reciprocal_xh2a_default

        out = reciprocal_xh2a_default(x)
    else:
        out = arbitrary_xh2a_default(x, func)

    check_should_equal(out.view(torch.int16)[1, 0, 3], [19865, 19847, 19827, 19810, 19792])
    check_should_equal(out.view(torch.int16)[0, 0, 0], [26620, 26620, 26620, 256, 26620])


def test_relu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    func = torch.relu
    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if name in cut_points_dict:
        from ..nl.relu import relu_xh2a_default

        out = relu_xh2a_default(x)
    else:
        out = arbitrary_xh2a_default(x, func)

    check_should_equal(out.view(torch.int16)[1, 0, 3], [10679, 10698, 10719, 10738, 10757])
    check_should_equal(out.view(torch.int16)[0, 0, 0], [0, 0, 0, 31743, 0])


def test_sigmoid():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    func = torch.sigmoid
    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if name in cut_points_dict:
        from ..nl.sigmoid import sigmoid_xh2a_default

        out = sigmoid_xh2a_default(x)
    else:
        out = arbitrary_xh2a_default(x, func)

    check_should_equal(out.view(torch.int16)[1, 0, 3], [14358, 14359, 14359, 14359, 14360])
    check_should_equal(out.view(torch.int16)[0, 0, 0], [14336, 0, 14336, 15360, 0])


def test_silu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    func = F.silu
    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if name in cut_points_dict:
        from ..nl.silu import silu_xh2a_default

        out = silu_xh2a_default(x)
    else:
        out = arbitrary_xh2a_default(x, func)

    check_should_equal(out.view(torch.int16)[1, 0, 3], [9686, 9708, 9730, 9748, 9770])
    check_should_equal(out.view(torch.int16)[0, 0, 0], [-32517, 0, -32517, 31743, 0])


def test_sin(update_cut_point=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    func = torch.sin
    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if update_cut_point:
        name = name + "_update"

    if name in cut_points_dict:
        from ..nl.sin import sin_xh2a_default

        out = sin_xh2a_default(x)
    else:
        out = arbitrary_xh2a_default(x, func, name=name)

    check_should_equal(out.view(torch.int16)[1, 0, 3], [10678, 10698, 10718, 10735, 10755])
    check_should_equal(out.view(torch.int16)[0, 0, 0], [-32736, -27667, -32736, 5101, -27667])


def test_cos(update_cut_point=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    func = torch.cos
    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if update_cut_point:
        name = name + "_update"

    if name in cut_points_dict:
        from ..nl.cos import cos_xh2a_default

        out = cos_xh2a_default(x)
    else:
        out = arbitrary_xh2a_default(x, func, name=name)

    # check_should_equal(out.view(torch.int16)[1, 0, 3], [10678, 10698, 10718, 10735, 10755])
    # check_should_equal(out.view(torch.int16)[0, 0, 0], [-32736, -27667, -32736, 5101, -27667])


def test_tanh():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    func = torch.tanh
    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if name in cut_points_dict:
        from ..nl.tanh import tanh_xh2a_default

        out = tanh_xh2a_default(x)
    else:
        out = arbitrary_xh2a_default(x, func)

    check_should_equal(out.view(torch.int16)[1, 0, 3], [10677, 10693, 10714, 10730, 10746])
    check_should_equal(out.view(torch.int16)[0, 0, 0], [-30460, -17408, -30460, 15360, -17408])


def test_pow3():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    func = lambda a: pow(a, 3)

    out = arbitrary_xh2a_default(x, func)

    check_should_equal(out.view(torch.int16)[1, 0, 3], [10677, 10693, 10714, 10730, 10746])
    check_should_equal(out.view(torch.int16)[0, 0, 0], [-30460, -17408, -30460, 15360, -17408])


def test_acos(update_cut_point=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    func = torch.acos
    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if update_cut_point:
        name = name + "_update"

    if name in cut_points_dict:
        raise NotImplemented("function already ready")
    else:
        out = arbitrary_xh2a_default(x, func, name=name)


def test_acosh(update_cut_point=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    func = torch.acosh
    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if update_cut_point:
        name = name + "_update"

    if name in cut_points_dict:
        raise NotImplemented("function already ready")
    else:
        out = arbitrary_xh2a_default(x, func, name=name)


def test_asin(update_cut_point=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    func = torch.asin
    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if update_cut_point:
        name = name + "_update"

    if name in cut_points_dict:
        raise NotImplemented("function already ready")
    else:
        out = arbitrary_xh2a_default(x, func, name=name)


def test_asinh(update_cut_point=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    func = torch.asinh
    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if update_cut_point:
        name = name + "_update"

    if name in cut_points_dict:
        raise NotImplemented("function already ready")
    else:
        out = arbitrary_xh2a_default(x, func, name=name)


def test_atan(update_cut_point=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    func = torch.atan
    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if update_cut_point:
        name = name + "_update"

    if name in cut_points_dict:
        raise NotImplemented("function already ready")
    else:
        out = arbitrary_xh2a_default(x, func, name=name)


def test_atanh(update_cut_point=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    func = torch.atanh
    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if update_cut_point:
        name = name + "_update"

    if name in cut_points_dict:
        raise NotImplemented("function already ready")
    else:
        out = arbitrary_xh2a_default(x, func, name=name)


def test_celu(update_cut_point=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    func = torch.celu
    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if update_cut_point:
        name = name + "_update"

    if name in cut_points_dict:
        raise NotImplemented("function already ready")
    else:
        out = arbitrary_xh2a_default(x, func, name=name)


def test_cosh(update_cut_point=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    func = torch.cosh
    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if update_cut_point:
        name = name + "_update"

    if name in cut_points_dict:
        raise NotImplemented("function already ready")
    else:
        out = arbitrary_xh2a_default(x, func, name=name)


def test_prelu(update_cut_point=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    func = torch.prelu
    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if update_cut_point:
        name = name + "_update"

    if name in cut_points_dict:
        raise NotImplemented("function already ready")
    else:
        out = arbitrary_xh2a_default(x, func, name=name)


def test_selu(update_cut_point=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    func = torch.selu
    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if update_cut_point:
        name = name + "_update"

    if name in cut_points_dict:
        raise NotImplemented("function already ready")
    else:
        out = arbitrary_xh2a_default(x, func, name=name)


def test_sinh(update_cut_point=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    func = torch.sinh
    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if update_cut_point:
        name = name + "_update"

    if name in cut_points_dict:
        raise NotImplemented("function already ready")
    else:
        out = arbitrary_xh2a_default(x, func, name=name)


def test_softsign(update_cut_point=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    func = torch.nn.Softsign()
    name = "softsign"
    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if update_cut_point:
        name = name + "_update"

    if name in cut_points_dict:
        raise NotImplemented("function already ready")
    else:
        out = arbitrary_xh2a_default(x, func, name=name)


def test_tan(update_cut_point=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    func = torch.tan
    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if update_cut_point:
        name = name + "_update"

    if name in cut_points_dict:
        raise NotImplemented("function already ready")
    else:
        out = arbitrary_xh2a_default(x, func, name=name)


def test_thresholdedrelu(update_cut_point=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((2, 3, 4, 5), dtype=torch.float16)
    x = x.to(device)
    x = init_tensor_with_fix_val(x)
    x.mul_(10000)
    x[0, 0, 0, 0] = torch.nan
    x[0, 0, 0, 1] = -torch.inf
    x[0, 0, 0, 2] = torch.nan
    x[0, 0, 0, 3] = 65504.0
    x[0, 0, 0, 4] = -65504.0

    func = torch.threshold
    if isinstance(func, partial) and hasattr(func.func, "__name__"):
        name = func.func.__name__
    if hasattr(func, "__name__"):
        name = func.__name__

    if update_cut_point:
        name = name + "_update"

    if name in cut_points_dict:
        raise NotImplemented("function already ready")
    else:
        out = arbitrary_xh2a_default(x, func, name=name)


def test_lut():
    # unit_test(test_dummy)
    unit_test(test_cos)
    unit_test(test_elu)
    unit_test(test_erf)
    unit_test(test_exp)
    unit_test(test_gelu)
    unit_test(test_hardsigmoid)
    unit_test(test_hardswish)
    unit_test(test_leaky_relu)
    unit_test(test_log)
    unit_test(test_mish)
    unit_test(test_pow2)
    unit_test(test_reciprocal)
    unit_test(test_relu)
    unit_test(test_sigmoid)
    unit_test(test_silu)
    unit_test(test_sin)
    unit_test(test_tanh)
    unit_test(test_prelu)
    unit_test(test_softplus)


if __name__ == "__main__":
    # test_pow3()
    # test_lut()
    # test_sin(update_cut_point=True)
    # test_cos(update_cut_point=True)
    # test_cos()
    # test_sin()
    # test_acos()
    # test_asin()
    # test_asinh()
    # test_atan()
    # test_atanh()
    # test_celu()
    # test_cosh()
    # test_selu()
    # test_sinh()
    # test_softsign()
    # test_tan(update_cut_point=True)
    # test_acosh()
    # test_prelu()
    # test_NewGELUActivation()
    # test_gelu_tanh()
    test_pow0dot4()
