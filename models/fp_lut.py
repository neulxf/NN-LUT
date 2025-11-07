import torch, torch.nn as nn, torch.nn.functional as F
from .triton_binary import float16_add_triton, float16_mul_triton
from fplut.nl._lut_fast_imp import lut_fast

# Nnum_points=16
Nnum_points=4

Ntable_size=Nnum_points*8+3

cut_points_dict = {
    "leaky_relu": [
        -65472.0,
        -2976.0,
        -132.5,
        -14.359375,
        -3.421875,
        -0.1968994140625,
        -0.00872802734375,
        0.0,
        257.75,
        41696.0,
        65504.0,
    ],
    "exp": [
        -17.34375,
        -15.171875,
        -8.890625,
        -5.2734375,
        -2.35546875,
        -0.3583984375,
        0.91650390625,
        3.451171875,
        6.84765625,
        10.9453125,
        11.0859375,
    ],
    "log": [
        4.8923e-04,
        5.8651e-04,
        1.7609e-02,
        2.3706e-01,
        1.0225e00,
        2.1523e00,
        6.9180e00,
        1.5075e02,
        1.6960e03,
        4.7616e04,
        6.5504e04,
    ],
    "divide": [
        4.8923e-04,
        5.3644e-04,
        8.6746e-03,
        6.8298e-02,
        2.6929e-01,
        2.1758e00,
        6.0938e01,
        2.6750e02,
        2.9740e03,
        3.0016e04,
        6.5504e04,
    ],
    "silu": [
        -20.359375,
        -17.109375,
        -8.3671875,
        -1.9755859375,
        -0.255615234375,
        -0.007244110107421875,
        0.0072174072265625,
        0.228515625,
        1.58203125,
        10.46875,
        65504.0,
    ],
    "sigmoid": [
        -65504.0,
        -19.5625,
        -10.4453125,
        -7.62890625,
        -6.3203125,
        -4.59375,
        -2.890625,
        -1.2880859375,
        2.068359375,
        8.6484375,
        65504.0,
    ],
    "swish": [
        -6.5504e04,
        -1.1766e01,
        -5.7539e00,
        -1.2461e00,
        -5.4834e-01,
        -7.9590e-02,
        1.1444e-02,
        1.3538e-01,
        1.2451e00,
        3.4922e00,
        6.5504e04,
    ],
    "inversesigmoid": [
        -8.2471e-01,
        8.7023e-06,
        7.6294e-04,
        2.3556e-03,
        1.1185e-02,
        4.0863e-02,
        2.0654e-01,
        6.1230e-01,
        8.9893e-01,
        9.9316e-01,
        9.9951e-01,
    ],
    "pow_2.0": [
        -6.5504e04,
        -2.5800e02,
        -1.8312e01,
        -4.6133e00,
        -1.1533e00,
        -1.4368e-01,
        2.4695e-01,
        2.2246e00,
        3.4469e01,
        2.5725e02,
        6.5504e04,
    ],
    "hardswish": [
        -41760.0,
        -3.03515625,
        -2.44140625,
        -1.2587890625,
        -0.33642578125,
        -0.037445068359375,
        0.009674072265625,
        0.1990966796875,
        1.0771484375,
        3.03515625,
        65504.0,
    ],
    "gelu": [
        -6.5504e04,
        -6.0273e00,
        -3.9453e00,
        -3.0000e00,
        -1.2539e00,
        -1.4355e-01,
        -2.2369e-02,
        8.5449e-02,
        6.7334e-01,
        3.0117e00,
        6.5504e04,
    ],
    "tanh": [
        -6.5504e04,
        -4.6641e00,
        -2.5703e00,
        -1.5195e00,
        -6.0205e-01,
        -2.5977e-01,
        7.3792e-02,
        7.9443e-01,
        2.5000e00,
        4.8047e00,
        6.5504e04,
    ],
    "mish": [
        -6.5504e04,
        -2.1859e01,
        -1.2477e01,
        -6.5898e00,
        -1.9648e00,
        -7.7051e-01,
        -8.3008e-02,
        3.8422e-02,
        9.0771e-01,
        3.3340e00,
        6.5504e04,
    ],
    "elu": [
        -6.5504e04,
        -8.9141e00,
        -4.1836e00,
        -2.4336e00,
        -1.1104e00,
        -7.7588e-01,
        -3.0908e-01,
        -8.6792e-02,
        -1.3252e-02,
        -6.7282e-04,
        6.5504e04,
    ],
    "softplus": [
        -65504.0,
        -19.5625,
        -10.4453125,
        -7.62890625,
        -6.31640625,
        -4.58984375,
        -2.8828125,
        -1.0712890625,
        0.921875,
        5.5234375,
        65504.0,
    ],
    "sin": [
        -3.140625,
        -3.125,
        -2.828125,
        -2.109375,
        -1.234375,
        -0.078125,
        0.765625,
        1.765625,
        2.859375,
        3.109375,
        3.140625,
    ],
    "cos": [
        -3.140625,
        -3.083984375,
        -1.859375,
        -1.421875,
        -0.515625,
        0.296875,
        1.421875,
        1.859375,
        2.234375,
        3.046875,
        3.140625,
    ],
    "hardsigmoid": [
        -6.5504e04,
        -3.2752e04,
        -3.0000e00,
        -2.0000e00,
        -1.0000e00,
        0.0000e00,
        1.0000e00,
        2.0000e00,
        3.0000e00,
        3.2752e04,
        6.5504e04,
    ],
    "relu": [
        -65504.0,
        0.0,
        0.11663818359375,
        0.1688232421875,
        0.229248046875,
        1.740234375,
        25.859375,
        132.875,
        3854.0,
        29280.0,
        65504.0,
    ],
}


def mod_data(x, cycle=None):
    if cycle is None:
        pi_int_2 = (
            2 * torch.pi
        )  # torch.tensor(torch.pi, dtype=torch.float16).view(torch.int16)
        mod_data = x % pi_int_2
        mod_data = (mod_data - torch.pi).sort(descending=False)[0]
        return mod_data

    return x % cycle


def pack_float16(mans, exp):
    out = (mans / 1024.0) * exp.exp2()
    return out.to(torch.float16)


def naive_msb(man, exp, bit):
    ones = torch.ones_like(man)
    msb = torch.zeros_like(exp)
    for i in range(1, bit):
        mask = ones << (i - 1)
        msb.masked_fill_((man & mask) != 0, i)
    return msb


def normalization_hmfp(man, exp, bit, output_format=(5, 12)):
    bit = int(bit)
    e, m = output_format
    assert man.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]
    assert exp.dtype in [torch.int8, torch.int16]
    assert e > 0 and e <= 16
    assert m > 4 and m <= 64

    # fast return
    if man.min() >= -pow(2, m - 1) and man.max() < pow(2, m - 1):
        return pack_float16(man, exp)

    sign = man >= 0
    neg = man.abs()
    neg.sub_(1).clip_(min=2)
    tmp = torch.where(sign, man, neg)

    msb = naive_msb(tmp, exp, bit)

    shift = msb.sub(m - 1).clip_(min=0)
    man_ = man >> shift
    exp = exp.add(shift)

    # assert man_.min() >= -pow(2, m-1) and man_.max() < pow(2, m-1)
    # assert exp.min() >= -pow(2, e-1) and exp.max() < pow(2, e-1)
    return pack_float16(man_, exp)


def float16_mul(a, b, input_format=(5, 12), output_format=(5, 12)):
    a_int = a.view(torch.int16)
    b_int = b.view(torch.int16)

    # 提取符号、指数和尾数
    sign_mask = 0x8000
    exp_mask = 0x7C00
    mantissa_mask = 0x03FF

    a_sign = (a_int & sign_mask) >> 15
    a_sign = torch.where(a_sign == 0, 1, -1)

    b_sign = (b_int & sign_mask) >> 15
    b_sign = torch.where(b_sign == 0, 1, -1)

    a_exp = (a_int & exp_mask) >> 10
    b_exp = (b_int & exp_mask) >> 10

    a_mantissa = a_int & mantissa_mask
    a_mantissa = torch.where(
        a_exp != 0, a_mantissa.add(1024) * a_sign, a_mantissa * a_sign
    )

    b_mantissa = b_int & mantissa_mask
    b_mantissa = torch.where(
        b_exp != 0, b_mantissa.add(1024) * b_sign, b_mantissa * b_sign
    )

    a_exp = torch.where(a_exp != 0, a_exp - 15, a_exp - 15 + 1)
    b_exp = torch.where(b_exp != 0, b_exp - 15, b_exp - 15 + 1)
    exp = a_exp + b_exp
    shift = torch.tensor(10).exp2()
    out = (a_mantissa * b_mantissa).div(shift, rounding_mode="trunc").to(torch.int32)
    return normalization_hmfp(out, exp, bit=32, output_format=output_format)


def float16_add(a, b, input_format=(5, 12), output_format=(5, 12)):
    a_int = a.view(torch.int16)
    b_int = b.view(torch.int16)

    # 提取符号、指数和尾数
    sign_mask = 0x8000
    exp_mask = 0x7C00
    mantissa_mask = 0x03FF

    a_sign = (a_int & sign_mask) >> 15
    a_sign = torch.where(a_sign == 0, 1, -1)

    b_sign = (b_int & sign_mask) >> 15
    b_sign = torch.where(b_sign == 0, 1, -1)

    a_exp = (a_int & exp_mask) >> 10
    b_exp = (b_int & exp_mask) >> 10

    a_mantissa = a_int & mantissa_mask
    a_mantissa = torch.where(
        a_exp != 0, a_mantissa.add(1024) * a_sign, a_mantissa * a_sign
    )

    b_mantissa = b_int & mantissa_mask
    b_mantissa = torch.where(
        b_exp != 0, b_mantissa.add(1024) * b_sign, b_mantissa * b_sign
    )

    a_exp = torch.where(a_exp != 0, a_exp - 15, a_exp - 15 + 1)
    b_exp = torch.where(b_exp != 0, b_exp - 15, b_exp - 15 + 1)

    shift = torch.maximum(a_exp, b_exp) - torch.minimum(a_exp, b_exp)
    shift = shift.exp2()
    # 进行对齐指数  小的指数进行移位
    a_mantissa = torch.where(
        a_exp > b_exp, a_mantissa, a_mantissa.div(shift, rounding_mode="trunc")
    )
    b_mantissa = torch.where(
        a_exp > b_exp, b_mantissa.div(shift, rounding_mode="trunc"), b_mantissa
    )

    out = (a_mantissa + b_mantissa).to(torch.int16)
    return normalization_hmfp(
        out, torch.maximum(a_exp, b_exp), bit=32, output_format=output_format
    )
    
    
# def float16_add_triton(a, b):
#     return a + b

# def float16_mul_triton(a, b):
#     return a * b

class NewTable(nn.Module):
    def __init__(
        self,
        func,
        cut_points,
        # table_size=259,
		table_size=Ntable_size,
        # num_points=32,
		num_points=Nnum_points,
        min=-65504,
        max=65504,
        device="cpu",
        # use_gpu_lut=False,
    ) -> None:
        super().__init__()
        self.func = func
        self.cut_points = torch.tensor(cut_points, dtype=torch.float16).to(device)
        assert len(self.cut_points) == 11, "cut_points must be 11 points"
        self.num_tables = len(self.cut_points) - 1
        self.table_size = table_size
        self.y_min = min
        self.y_max = max
        self.num_points = num_points
        self.device = device
        # self.use_gpu_lut = use_gpu_lut
        self.create_table()

    def create_table(self):
        # 生成插值表
        self.table = torch.zeros(self.table_size, dtype=torch.float32).to(self.device)
        self.index = torch.zeros(self.table_size, dtype=torch.float32).to(self.device)
        self.mul_scale = torch.zeros(len(self.cut_points) - 1, dtype=torch.float32).to(
            self.device
        )  # 乘子系数
        self.mul_scale[0] = 1 / (self.cut_points[1] - self.cut_points[0])

        for i in range(1, len(self.cut_points[:-2])):
            start = self.cut_points[i]
            end = self.cut_points[i + 1]
            self.mul_scale[i] = self.num_points / (end - start)
        self.mul_scale[-1] = 1 / (self.cut_points[-1] - self.cut_points[-2])

        self.index[0] = self.cut_points[0]
        for i in range(1, len(self.cut_points) - 2):
            start = self.cut_points[i]
            end = self.cut_points[i + 1]

            x = torch.linspace(start, end, self.num_points + 1).to(self.device)

            if i != len(self.cut_points) - 3:
                self.index[(i - 1) * self.num_points + 1 : i * self.num_points + 1] = x[
                    :-1
                ].to(torch.float32)
            else:
                self.index[(i - 1) * self.num_points + 1 : -1] = x.to(torch.float32)

        self.index[-1] = self.cut_points[-1]

        self.table = self.func(self.index.float())
        self.table = self.table.clamp(min=self.y_min, max=self.y_max)

    def fp32_to_fp16_floor(self, x: torch.Tensor):
        int_tensor = x.view(torch.int32)

        # 提取指数位和mantissa位
        sign_mask = 0x80000000
        exp_mask = 0x7F800000
        mantissa_mask = 0x007FFFFF

        sign_bit = int_tensor & sign_mask
        exp = int_tensor & exp_mask
        mantissa = int_tensor & mantissa_mask

        mantissa = mantissa >> 13 << 13
        result = torch.zeros_like(int_tensor, dtype=torch.int32)
        result = result | sign_bit
        result = result | exp
        result = result | mantissa
        result = result.view(torch.float32).half()
        return result

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1)
        if torch.numel(x) > pow(2, 28):
            split = 32
        elif torch.numel(x) > pow(2, 25):
            split = 8
        else:
            split = 1
        try:
            out = self.split_forward(
                x, split
            )
        except Exception as e:
            print(
                "shape: {} (total: {}), split: {}".format(shape, torch.numel(x), split)
            )
            raise e
        out = out.reshape(shape)
        return out

    def _lut_fast(self, x):
        return lut_fast(x, self.cut_points, self.table, self.mul_scale)

    def _forward(
        self,
        x,
    ):
        torch.cuda.set_device(x.device) 
        x = torch.nan_to_num(
            x, nan=0.0, neginf=self.cut_points[0], posinf=self.cut_points[-1]
        )
        # 根据切分点，找到x所在的区间
        cut_indices = (
            torch.bucketize(x.float(), self.cut_points, right=True).clamp(
                max=self.num_tables
            )
            - 1
        ).clip(0)

        dval = float16_add_triton(x, -self.cut_points[cut_indices])
        temp = float16_mul_triton(dval, self.mul_scale[cut_indices])
        if temp.is_cuda:
            index = temp.floor().to(torch.int16)
        else:
            index = temp.float().floor().to(torch.int16)

        mask_last_table = (cut_indices == 9) * (index == 1)
        index = torch.where(mask_last_table, torch.zeros_like(index), index)

        decimal = float16_add_triton(temp, -index.to(torch.float16))

        y = torch.zeros_like(x, dtype=torch.float16)
        indices = torch.zeros_like(cut_indices, dtype=torch.int64)
        indices[cut_indices == 0] = (0 + index[cut_indices == 0]).long()
        indices[cut_indices >= 1] = (
            1
            + (cut_indices[cut_indices >= 1] - 1) * self.num_points
            + index[cut_indices >= 1]
        ).long()
        
        left = self.table[indices.long()]
        right = self.table[indices.long() + 1]

        interval = float16_add_triton(right, -left)

        y = float16_add_triton(
            left,
            interval * decimal,
        )

        y[x <= self.cut_points[0]] = self.table[0]
        y[x >= self.cut_points[-1]] = self.table[-1]
        return y

    def split_forward(
        self, x, split=4, inplace=False, test_lut=False, use_gpu_lut=True
    ):
        if inplace:
            out = x
        else:
            out = x.clone()
        length = int(len(x) / split)
        split = int(split)
        for i in range(split):
            if i == (split - 1):
                local = x[i * length :]
                out[i * length :] = self._lut_fast(
                    local
                )
            else:
                local = x[i * length : (i + 1) * length]
                out[i * length : (i + 1) * length] = self._lut_fast(
                    local,
                )
        return out
    
    
class FPLUT(nn.Module):
    def __init__(
        self,
        function,
        func_name,
    ) -> None:
        """
        Base look up table
        """
        super().__init__()
        self.function = function
        self.func_name = func_name
        self.table = None
        self.op_class = "FP16_LUT"

    def raw_forward(self, x):
        return self.function(x)

    def forward(self, x):
        if self.table is None:
            cut_points = cut_points_dict[self.func_name]
            self.table = NewTable(
                # self.function, cut_points, table_size=259, device=x.device
				self.function, cut_points, table_size=Ntable_size, device=x.device
            )

        if self.func_name in ["cos", "sin"]:
            x = mod_data(x)
        dtype = x.dtype
        x = x.clamp(self.table.cut_points[0], self.table.cut_points[-1])
        y = self.table(x.half()).clamp(-65504, 65504).to(dtype)
        return y



def divide(x):
    return 1 / x


class FPSoftMax(nn.Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis
        self.op_class = "Fp16Softmax"
        self.lut_exp = None
        self.lut_div = None

    def raw_forward(self, x):
        return F.softmax(x, dim=self.axis)

    def forward(self, x):
        if self.lut_exp is None:
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
            self.lut_exp = NewTable(
                # torch.exp, cut_points, table_size=259, device=x.device
				torch.exp, cut_points, table_size=Ntable_size, device=x.device
            )

        # if self.lut_div is None:
        #     cut_points = cut_points_dict["divide"]
        #     self.lut_div = NewTable(divide, cut_points, table_size=259, device=x.device)
            
        dtype = x.dtype
        re_x = x - x.max(dim=self.axis, keepdim=True)[0]
        exp_out = self.lut_exp(re_x.half())
        exp_sum = exp_out.sum(dim=self.axis, keepdim=True, dtype=torch.float16).float()
        # y = exp_out * self.lut_div(exp_sum)
        y = exp_out / exp_sum
        return y.to(dtype)


if __name__ == "__main__":
    a = torch.tensor([504.0, 124.1], dtype=torch.float16)
    b = torch.tensor([-5.3516, 124.1], dtype=torch.float16)

    func = nn.SiLU()
    model = FPLUT(func, func_name="silu")
    for ai in [b]:
        f_out = model.raw_forward(ai)
        q_out = model(ai)
        print(f_out, q_out)