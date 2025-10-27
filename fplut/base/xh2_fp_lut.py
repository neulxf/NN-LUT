import os

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import fp24
    import xh2a_vp

    from tools import comparison, dec_string, hex_string
except Exception as e:
    try:
        from . import fp24 as fp24
        from . import xh2a_vp as xh2a_vp
        from .tools import comparison, dec_string, hex_string
    except Exception as e:
        # from hmquant.ptq.sefp import xh2a_vp
        # from hmquant.ptq.sefp import fp24
        # from hmquant.ptq.sefp.tools import hex_string, dec_string, comparison
        raise e

try:
    from .hsum import hsum
except:
    hsum = None

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
        -65504.0,
        -14.53125,
        -8.640625,
        -4.5859375,
        -2.416015625,
        -1.0654296875,
        2.99609375,
        5.97265625,
        9.53125,
        11.5234375,
        65504.0,
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
        -6.5504e04,
        -1.8719e01,
        -1.1141e01,
        -7.7461e00,
        -2.0840e00,
        -8.2617e-01,
        -6.8420e-02,
        1.4856e-01,
        1.1289e00,
        5.3516e00,
        6.5504e04,
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
    "pow_3.0": [
        -41760.0,
        -35.0,
        -7.8671875,
        -2.55078125,
        -1.0859375,
        -0.28466796875,
        0.556640625,
        2.11328125,
        7.87109375,
        40.5,
        65504.0,
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
    "sin_old": [
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
    "sin": [
        -0.0,
        0.0004878044128417969,
        0.125,
        0.53125,
        1.625,
        2.8125,
        3.5,
        4.46875,
        5.875,
        6.234375,
        6.28125,
    ],
    "cos_old": [
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
    "cos": [
        -0.0,
        0.04937744140625,
        1.40625,
        2.25,
        3.125,
        4.0,
        4.625,
        4.9375,
        5.46875,
        6.15625,
        6.28125,
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
    # "relu": [
    #     -65504.0,
    #     0.0,
    #     0.11663818359375,
    #     0.1688232421875,
    #     0.229248046875,
    #     1.740234375,
    #     25.859375,
    #     132.875,
    #     3854.0,
    #     29280.0,
    #     65504.0,
    # ],
    "relu":[
        0.0,
        0.001953125,
        0.00389862060546875,
        0.03515625,
        0.234375,
        0.75,
        256.75,
        496.75,
        16880.0,
        32240.0,
        65504.0,
    ],
    "celu": [
        -65504.0,
        -8.6640625,
        -3.919921875,
        -1.2646484375,
        -0.3349609375,
        -0.07342529296875,
        0.01531219482421875,
        18.109375,
        995.5,
        41696.0,
        65504.0,
    ],
    "selu": [
        -65504.0,
        -8.6328125,
        -3.919921875,
        -1.0654296875,
        -0.5498046875,
        -0.17138671875,
        -0.0308074951171875,
        -0.004917144775390625,
        0.0,
        62336.0,
        65504.0,
    ],
    "softsign": [
        -55648.0,
        -35.65625,
        -3.9375,
        -1.060546875,
        -0.281005859375,
        0.0263671875,
        1.0615234375,
        7.7890625,
        35.71875,
        857.5,
        65504.0,
    ],
    "asinh": [
        -65504.0,
        -41408.0,
        -1988.0,
        -70.5,
        -2.060546875,
        -0.06182861328125,
        2.904296875,
        36.125,
        439.75,
        41440.0,
        65504.0,
    ],
    "atan": [
        -56256.0,
        -159.25,
        -10.984375,
        -2.41796875,
        -0.2003173828125,
        -0.003520965576171875,
        0.3916015625,
        7.78515625,
        115.0625,
        1203.0,
        65504.0,
    ],
    "acos": [
        -1.0,
        -0.982421875,
        0.45751953125,
        0.49609375,
        0.54296875,
        0.65185546875,
        0.77099609375,
        0.841796875,
        0.89453125,
        0.982421875,
        1.0,
    ],
    "asin": [
        -1.0,
        -0.982421875,
        -0.89501953125,
        -0.0484619140625,
        -0.003734588623046875,
        0.00406646728515625,
        0.085693359375,
        0.6005859375,
        0.89453125,
        0.982421875,
        1.0,
    ],
    "atanh": [
        -1.0,
        -0.99951171875,
        -0.8984375,
        -0.60302734375,
        -0.0214691162109375,
        -0.0005216598510742188,
        0.07220458984375,
        0.54345703125,
        0.89794921875,
        0.99951171875,
        1.0,
    ],
    "tan": [
        0.0,
        0.00024890899658203125,
        1.234375,
        1.421875,
        1.46875,
        1.53125,
        1.59375,
        1.6875,
        2.109375,
        3.03125,
        3.140625,
    ],
    "sinh": [
        -65504.0,
        -12.59375,
        -9.7265625,
        -5.96875,
        -1.0888671875,
        -0.01345062255859375,
        1.515625,
        4.21484375,
        7.57421875,
        12.5859375,
        65504.0,
    ],
    "cosh": [
        -65504.0,
        -12.59375,
        -9.7265625,
        -5.96875,
        -2.994140625,
        1.0849609375,
        2.99609375,
        5.97265625,
        9.734375,
        12.5859375,
        65504.0,
    ],
    "acosh": [
        1.0185546875,
        1.1123046875,
        1.556640625,
        4.99609375,
        14.9921875,
        143.625,
        383.75,
        2302.0,
        8752.0,
        59392.0,
        65504.0,
    ],
}


def mod_data(x, cycle=None):
    if cycle is None:
        pi_int_2 = 2 * torch.pi  # torch.tensor(torch.pi, dtype=torch.float16).view(torch.int16)
        mod_data = x % pi_int_2
        mod_data = (mod_data - torch.pi).sort(descending=False)[0]
        return mod_data

    return x % cycle


def float16_mul_(a, b, use_ip=True, test_lut=False):
    if hsum is not None and use_ip and a.is_cuda:
        a = a.reshape(-1, 1)
        b = b.reshape(-1, 1)
        val = hsum.dot(a, b, dim=-1, keepdim=True, round_mode=1, is_lut=True, debug=0)
        if test_lut:
            baseline = xh2a_vp.dot(a, b, dim=-1, keepdim=True, round_mode=1, is_lut=1, debug=0)
            comparison(baseline, val)
            baseline = None

        out = val.reshape(-1)
    elif xh2a_vp is not None and xh2a_vp.ip is not None and use_ip:
        a = a.reshape(-1, 1)
        b = b.reshape(-1, 1)
        val = xh2a_vp.dot(a, b, dim=-1, keepdim=True, round_mode=1, is_lut=1, debug=0)
        out = val.reshape(-1)
    else:
        val = a.float() * b.float()  # val.half(rounding_mode='rtz')
        out = fp24.fp24_to_fp16(val)
    return out


def float16_add_(a, b, use_ip=True, test_lut=False):
    if hsum is not None and use_ip:
        a = a.reshape(-1, 1)
        b = b.reshape(-1, 1)
        tensor1 = torch.concat((a, b), dim=-1)
        tensor2 = torch.ones_like(tensor1)
        val = hsum.dot(tensor1, tensor2, dim=-1, keepdim=True, round_mode=1, is_lut=True, debug=0)
        if test_lut:
            baseline = xh2a_vp.dot(tensor1, tensor2, dim=-1, keepdim=True, round_mode=1, is_lut=1, debug=0)
            comparison(baseline, val)
            baseline = None
        tensor1 = None
        tensor2 = None
        out = val.reshape(-1)
    elif xh2a_vp is not None and xh2a_vp.ip is not None and use_ip:
        a = a.reshape(-1, 1)
        b = b.reshape(-1, 1)
        tensor1 = torch.concat((a, b), dim=-1)
        tensor2 = torch.ones_like(tensor1)
        val = xh2a_vp.dot(tensor1, tensor2, dim=-1, keepdim=True, round_mode=1, is_lut=True)
        tensor1 = None
        tensor2 = None
        out = val.reshape(-1)
    else:
        val = a.float() + b.float()  # val.half(rounding_mode='rtz')
        out = fp24.fp24_to_fp16(val)
    return out


class NewTable(nn.Module):
    def __init__(
        self,
        func,
        cut_points,
        table_size=259,
        num_points=32,
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
        # self.index2 = torch.zeros(self.table_size, dtype=torch.float32).to(self.device)

        self.mul_scale = torch.zeros(len(self.cut_points) - 1, dtype=torch.float32).to(self.device)  # 乘子系数

        self.mul_scale[0] = (
            1 / (self.cut_points[1].float() - self.cut_points[0].float())
        )  # .clip(-65504, 65504).half()
        # self.cut_points[0] = (self.cut_points[1] - (1 / self.mul_scale[0])).clip(-65504, 65504).half().float()

        for i in range(1, len(self.cut_points[:-2])):
            start = self.cut_points[i]
            end = self.cut_points[i + 1]
            self.mul_scale[i] = (self.num_points / (end.float() - start.float())).item()
        self.mul_scale[-1] = (1 / (self.cut_points[-1].float() - self.cut_points[-2].float())).item()

        self.index[0] = self.cut_points[0]
        # self.index2[0] = self.cut_points[0]
        for i in range(1, len(self.cut_points) - 2):
            start = self.cut_points[i]
            end = self.cut_points[i + 1]

            # for j in range(self.num_points):
            #     self.index[(i - 1) * self.num_points + j + 1] = start + j / self.mul_scale[i]

            x = torch.linspace(start, end, self.num_points + 1, dtype=torch.float32).to(self.device)

            if i != len(self.cut_points) - 3:
                self.index[(i - 1) * self.num_points + 1 : i * self.num_points + 1] = x[:-1].to(torch.float32)
            else:
                self.index[(i - 1) * self.num_points + 1 : -1] = x.to(torch.float32)

        self.index[-1] = self.cut_points[-1]
        # self.index2[-1] = self.cut_points[-1]
        if self.table.is_cuda:
            self.table = self.func(self.index.float())
            self.table = self.table.clamp(min=self.y_min, max=self.y_max)
        else:
            # self.table = self.func(self.index) # not support for CPU, thus convert to FP32 and back
            self.table = self.func(self.index.float())
            self.table[self.table < self.y_min] = self.y_min
            self.table[self.table > self.y_max] = self.y_max

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

    def forward(self, x, inplace=False, test_lut=False, use_gpu_lut=True):
        shape = x.shape
        x = x.reshape(-1)
        if torch.numel(x) > pow(2, 28):
            split = 32
        elif torch.numel(x) > pow(2, 25):
            split = 8
        else:
            split = 1

        # x = x.clone() # must clone as we revise x inplace # already clone outside
        try:
            out = self.split_forward(x, split, inplace=inplace, test_lut=test_lut, use_gpu_lut=use_gpu_lut)
        except Exception as e:
            print("shape: {} (total: {}), split: {}".format(shape, torch.numel(x), split))
            raise e

        out = out.reshape(shape)
        return out

    def _forward(
        self,
        x,
        test_lut=False,
        use_gpu_lut=True,
        debug=False,
        error_indice=0,
    ):
        if xh2a_vp is not None and xh2a_vp.ip is not None:
            pass
        else:
            test_lut = False

        x = torch.nan_to_num(x, nan=0.0, neginf=self.cut_points[0], posinf=self.cut_points[-1])
        x = x.clamp(min=self.cut_points[0], max=self.cut_points[-1])
        hres = y = None
        if use_gpu_lut and hsum is not None and hasattr(hsum, "lut") and not test_lut and not debug and x.is_cuda:
            try:
                assert (
                    x.device == self.cut_points.device == self.table.device == self.mul_scale.device
                ), f"{self.cut_points.device} {self.table.device} {self.mul_scale.device} {x.device}, is not same device"
                with torch.cuda.device_of(x):
                    hres = hsum.lut(
                        x,
                        self.cut_points,
                        self.table,
                        self.mul_scale,
                        self.table_size,
                        self.num_points,
                    )
            except IndexError:
                hres = None

        if hres is None:
            if debug:
                print(self.cut_points[-1].item(), self.cut_points[0].item())
                print(hex_string(self.cut_points))
                print("")

            # 根据切分点，找到x所在的区间
            cut_indices = (torch.bucketize(x.float(), self.cut_points, right=True).clamp(max=self.num_tables) - 1).clip(
                0
            )

            temp = float16_mul_(
                float16_add_(x, -self.cut_points[cut_indices], test_lut=test_lut),
                self.mul_scale[cut_indices],
                test_lut=test_lut,
            )

            if temp.is_cuda:
                index = temp.floor().to(torch.int16)
            else:
                index = temp.float().floor().to(torch.int16)

            decimal = float16_add_(temp, -index.to(torch.float16), test_lut=test_lut)

            y = torch.zeros_like(x, dtype=torch.float16)
            indices = torch.zeros_like(cut_indices, dtype=torch.int64)
            indices[cut_indices == 0] = (0 + index[cut_indices == 0]).long()
            indices[cut_indices >= 1] = (
                1 + (cut_indices[cut_indices >= 1] - 1) * self.num_points + index[cut_indices >= 1]
            ).long()

            if not debug:
                temp = None
                index = None
                cut_indices = None

            decimal_is_zero = True
            if decimal_is_zero:
                y[decimal == 0] = self.table[indices[decimal == 0]]
                left = self.table[indices[decimal != 0].long()]
                right = self.table[indices[decimal != 0].long() + 1]
            else:
                if debug:
                    unexcept = indices >= 258
                    if unexcept.sum().item() > 0:
                        check = decimal[unexcept] == 0
                        print(check.sum().item() == unexcept.sum().item())
                        print(decimal[unexcept])
                        print(cut_indices[unexcept])
                        print(index[unexcept])
                left = self.table[indices.long()]
                right = self.table[indices.long() + 1]

            if not (decimal_is_zero and right.numel() == 0):
                interval = float16_add_(right, -left, test_lut=test_lut)
                right = None

                if decimal_is_zero:
                    distance = decimal[decimal != 0]
                else:
                    distance = decimal

                if hsum is not None and left.is_cuda:
                    left = left.reshape(-1, 1)
                    interval = interval.reshape(-1, 1)
                    distance = distance.reshape(-1, 1)
                    coefficient = torch.ones_like(distance)

                    tensor1 = torch.concat((interval, left), dim=-1)
                    tensor2 = torch.concat((distance, coefficient), dim=-1)
                    if not debug:
                        coefficient = None
                        interval = None
                        left = None
                        distance = None

                    value = hsum.dot(
                        tensor1,
                        tensor2,
                        dim=-1,
                        keepdim=True,
                        round_mode=1,
                        is_lut=True,
                    )

                    if test_lut:
                        baseline = xh2a_vp.dot(
                            tensor1,
                            tensor2,
                            dim=-1,
                            keepdim=True,
                            round_mode=1,
                            is_lut=1,
                        )
                        comparison(baseline, value)
                        baseline = None

                    if not debug:
                        tensor1 = None
                        tensor2 = None

                    # value = xh2a_vp.dot(tensor1[0:1], tensor2[0:1], dim=-1, keepdim=True, round_mode=1, debug=3)
                    if decimal_is_zero:
                        y[decimal != 0] = value.reshape(-1)
                    else:
                        y = value.reshape(-1)
                elif xh2a_vp is not None and xh2a_vp.ip is not None:
                    left = left.reshape(-1, 1)
                    interval = interval.reshape(-1, 1)
                    distance = distance.reshape(-1, 1)
                    coefficient = torch.ones_like(distance)

                    tensor1 = torch.concat((interval, left), dim=-1)
                    tensor2 = torch.concat((distance, coefficient), dim=-1)
                    if not debug:
                        coefficient = None
                        interval = None
                        left = None
                        distance = None

                    value = xh2a_vp.dot(
                        tensor1,
                        tensor2,
                        dim=-1,
                        keepdim=True,
                        round_mode=1,
                        is_lut=1,
                        debug=0,
                    )
                    if not debug:
                        tensor1 = None
                        tensor2 = None
                    if decimal_is_zero:
                        y[decimal != 0] = value.reshape(-1)
                    else:
                        y = value.reshape(-1)
                else:
                    if decimal_is_zero:
                        y[decimal != 0] = float16_add_(
                            left,
                            interval * distance,
                        )
                    else:
                        y = float16_add_(
                            left,
                            interval * distance,
                        )

                    if not debug:
                        left = None
                        interval = None
                        distance = None

            y[x <= self.cut_points[0]] = self.table[0]
            y[x >= self.cut_points[-1]] = self.table[-1]

            if debug:  # debug_i could be 1. a integer or 2: a list of int or 3. slice()
                # debug_i = 0
                # debug_i = [0, 20]
                # debug_i = slice(None)
                debug_i = int(error_indice)

                print("x.shape: {}, cut_indices.shape = {}, debug_i: {}".format(x.shape, cut_indices.shape, debug_i))
                print("idxIn10:", dec_string(cut_indices[debug_i]))
                print("indices:", dec_string(indices[debug_i]))
                print("x      :", hex_string(x[debug_i]))
                # print("-self.cut_points[cut_indices]:", hex_string(-self.cut_points[cut_indices][debug_i]))
                # tmp = float16_add_(x, -self.cut_points[cut_indices],)
                # print("x - self.cut_points[cut_indices]:", hex_string(tmp[debug_i]))
                # tmp = self.mul_scale[cut_indices]
                # print("ratio:", hex_string(tmp[debug_i]))
                # print("temp (mul):", hex_string(temp[debug_i]))
                # print("index:", dec_string(index[debug_i]))
                print("decimal == 0:", decimal[debug_i] == 0)
                print("decimal:", hex_string(decimal[debug_i]))
                print("left   :", hex_string(self.table[indices.long()][debug_i]))
                print("y      :", hex_string(y[debug_i]))

                if (indices >= 258).sum().item() == 0:
                    print("right  :", hex_string(self.table[indices.long() + 1][debug_i]))
                    tmp = float16_add_(
                        self.table[indices.long() + 1],
                        -self.table[indices.long()],
                    )
                    print("add1   :", hex_string(tmp[debug_i]))
                    value = xh2a_vp.dot(
                        tensor1[debug_i],
                        tensor2[debug_i],
                        dim=-1,
                        keepdim=True,
                        round_mode=1,
                        is_lut=1,
                        enable_openmp=0,
                        debug=3,
                    )

            if test_lut and hsum is not None and hasattr(hsum, "lut"):
                assert (
                    x.device == self.cut_points.device == self.table.device == self.mul_scale.device
                ), f"{x.device} {self.cut_points.device} {self.table.device} {self.mul_scale.device}, is not the same device"
                with torch.cuda.device_of(x):
                    hres = hsum.lut(
                        x,
                        self.cut_points,
                        self.table,
                        self.mul_scale,
                        self.table_size,
                        self.num_points,
                    )
                try:
                    if hres is not None:
                        comparison(hres, y)
                except AssertionError as e:
                    name = ""
                    if hasattr(self.func, "__name__"):
                        name = self.func.__name__
                    elif hasattr(self.func, "__class__"):
                        name = self.func.__class__.__name__
                    name = name.lower()
                    print("gpu lut failed for function {} !".format(name))
                    torch.save(x, "xh2a_fp_lut.{}.x.pth".format(name))
                    raise e

        return hres if hres is not None else y

    def split_forward(self, x, split=4, inplace=False, test_lut=False, use_gpu_lut=True):
        if inplace:
            out = x
        else:
            out = x.clone()
        length = int(len(x) / split)
        split = int(split)
        for i in range(split):
            if i == (split - 1):
                local = x[i * length :]
                out[i * length :] = self._forward(local, test_lut=test_lut, use_gpu_lut=use_gpu_lut)
            else:
                local = x[i * length : (i + 1) * length]
                out[i * length : (i + 1) * length] = self._forward(
                    local,
                    test_lut=test_lut,
                    use_gpu_lut=use_gpu_lut,
                )
        return out

    def create_table_float32(self):
        # 生成插值表
        self.table = torch.zeros(self.table_size, dtype=torch.float32).to(self.device)
        self.index = torch.zeros(self.table_size, dtype=torch.float32).to(self.device)
        # self.index2 = torch.zeros(self.table_size, dtype=torch.float32).to(self.device)

        self.mul_scale = torch.zeros(len(self.cut_points) - 1, dtype=torch.float32).to(self.device)  # 乘子系数
        self.mul_scale[0] = 1 / (self.cut_points[1] - self.cut_points[0])
        for i in range(1, len(self.cut_points[:-2])):
            start = self.cut_points[i]
            end = self.cut_points[i + 1]
            self.mul_scale[i] = self.num_points / (end - start)
        self.mul_scale[-1] = 1 / (self.cut_points[-1] - self.cut_points[-2])

        self.index[0] = self.cut_points[0]
        # self.index2[0] = self.cut_points[0]
        for i in range(1, len(self.cut_points) - 2):
            start = self.cut_points[i]
            end = self.cut_points[i + 1]

            # for j in range(self.num_points):
            #     self.index[(i - 1) * self.num_points + j + 1] = start + j / self.mul_scale[i]

            x = torch.linspace(start, end, self.num_points + 1).to(self.device)

            if i != len(self.cut_points) - 3:
                self.index[(i - 1) * self.num_points + 1 : i * self.num_points + 1] = x[:-1].to(torch.float32)
            else:
                self.index[(i - 1) * self.num_points + 1 : -1] = x.to(torch.float32)

        self.index[-1] = self.cut_points[-1]
        # self.index2[-1] = self.cut_points[-1]
        self.table = self.func(self.index).clamp(self.y_min, self.y_max)

    def get_hardware_params(self):
        param_dict = dict()
        name = ""
        if hasattr(self.func, "__name__"):
            name = self.func.__name__
        elif hasattr(self.func, "__class__"):
            name = self.func.__class__.__name__
        name = name.lower()
        param_dict["table_name"] = name
        param_dict["table_size"] = self.table_size
        param_dict["cut_points"] = self.cut_points.cpu().to(torch.float16).detach().numpy()
        param_dict["table"] = self.table.cpu().to(torch.float16).detach().numpy()

        param_dict["y_min"] = self.y_min
        param_dict["y_max"] = self.y_max
        param_dict["num_points"] = self.num_points

        if hasattr(self, "i_bit"):
            delattr(self, "i_bit")
        if hasattr(self, "o_bit"):
            delattr(self, "o_bit")
        return param_dict


class HXTable(NewTable):
    def __init__(
        self,
        lut_cut_points,
        lut_table,
        lut_scale,
        table_size=259,
        num_points=32,
        min=-65504,
        max=65504,
        device="cpu",
    ):
        nn.Module.__init__(self)
        self.y_min = min
        self.y_max = max
        self.num_tables = len(lut_cut_points) - 1
        self.cut_points = torch.tensor(lut_cut_points, dtype=torch.float16, device=device)
        self.table = torch.tensor(lut_table, dtype=torch.float16, device=device)
        self.mul_scale = torch.tensor(lut_scale, dtype=torch.float16, device=device)
        self.table_size = table_size
        self.num_points = num_points
        self.device = device


if __name__ == "__main__":
    a = torch.tensor([65504.0, 124.1], dtype=torch.float16)
    b = torch.tensor([-5.3516, 124.1], dtype=torch.float16)

    # print(a + b)
    a = torch.load("float16_add_.tensor1.pth")
    b = torch.load("float16_add_.tensor2.pth")
    print(a.shape, a.dtype)
    print(b.shape, b.dtype)
    print(float16_add_(a, b, use_ip=True, test_lut=True))

    # print(a * b)
    print(float16_mul_(a, b))
