import torch


def check(x, bit_sign=1, bit_exp=8, bit_mas=15):
    assert x.dtype in [torch.int32, torch.float32], "format invalid"
    assert bit_exp == 8, "format invalid"
    assert bit_mas <= 23, "format invalid"


def fp24toe10m17(x, bias=511 - 15):
    assert x.dtype == torch.float32, "input should be FP24"
    x_int = x.view(torch.int32)
    mask = 0x00FF
    x_int_mas = x_int & mask
    assert x_int_mas.abs().sum().item() == 0, "input should be FP24"

    mask = 0x7F800000
    x_int_exp = x_int & mask
    x_int_exp = x_int_exp >> 23

    mask = 0x007FFF00
    x_int_mas = x_int & mask
    x_int_mas = x_int_mas >> 8
    x_int_mas[x_int_exp != 0] = x_int_mas[x_int_exp != 0] + pow(2, 15)
    x_int_mas[x < 0] = x_int_mas[x < 0] * -1

    x_int_exp = x_int_exp - 127
    x_int_exp[x_int_exp < -127] = -127
    x_int_exp = x_int_exp + bias
    return x_int_mas.to(torch.int32), x_int_exp.to(torch.int16)


def fp16toe10m17(x, bias=511):
    x = x.to(torch.float32)
    return fp24toe10m17(x, bias)


def fp32toe10m17(x, bias=511):
    x = x.to(torch.float32)
    return fp24toe10m17(x, bias)


def fp32_to_fp24(x, bit_sign=1, bit_exp=8, bit_mas=15, round_mode="trunc", debug=False):
    # round_mode in ['trunc', 'RNE']
    check(x, bit_sign, bit_exp, bit_mas)
    x = x.to(torch.float32)
    x_int = x.view(torch.int32)
    mask = 0xFFFFFF00
    x_int_1 = x_int & mask
    round_mode = round_mode.lower()
    if round_mode == "trunc":
        x_int = x_int_1
    elif round_mode == "rne":
        mask = 0x7F800000
        x_int_exp = x_int & mask
        mask = 0x007FFF00
        x_int_mas = x_int & mask
        x_int_mas = x_int_mas + 0x00000100
        x_int_exp += (x_int_mas == 0x00800000).to(torch.int32) << 23
        x_int_mas[x_int_mas == 0x00800000] = 0
        mask = 0x80000000
        x_int_up = x_int & mask
        x_int_up = x_int_up | x_int_exp | x_int_mas
        x_int_exp = x_int_mas = None

        mask = 0x000000FF
        x_int_2 = x_int & mask
        round_up = (x_int_2 > 0x80) | ((x_int_2 == 0x80) & ((x_int_1 & 0x0100) == 0x0100))
        x_int_2 = None
        x_int = torch.where(round_up, x_int_up, x_int_1)
        x_int_up = x_int_1 = None
    elif round_mode == "floor":
        mask = 0x7F800000
        x_int_exp = x_int & mask
        mask = 0x007FFF00
        x_int_mas = x_int & mask
        x_int_mas = x_int_mas + 0x00000100
        x_int_exp += (x_int_mas == 0x00800000).to(torch.int32) << 23
        x_int_mas[x_int_mas == 0x00800000] = 0
        mask = 0x80000000
        x_int_up = x_int & mask
        x_int_up = x_int_up | x_int_exp | x_int_mas

        if debug:
            print(x_int & mask & 0xFF)
            print(x_int_exp & 0xFF)
            print(x_int_mas & 0xFF)
            print(fp24toe10m17(x_int_up.view(torch.float32)))
            print(fp24toe10m17(x_int_1.view(torch.float32)))

        mask = 0x000000FF
        x_int_2 = x_int & mask
        round_up = (x < 0) & (x_int_2 != 0)
        x_int = torch.where(round_up, x_int_up, x_int_1)
    else:
        raise RuntimeError("unkonwn rounding mode {}".format(round_mode))

    x = x_int.view(torch.float32)
    return x


def E10M17_to_fp16(exp, man, bias=0, rounding_mode="trunc", enable_overflow=False):  # bias = 511 in certain case
    is_free_cuda = man.is_cuda and man.nelement() > 100000000

    exp = exp.to(torch.int16)
    man = man.to(torch.int32)

    exp = exp.sub(bias)

    assert exp.min().item() > -pow(2, 9) and exp.max().item() <= pow(2, 9), "exp shape {}".format(exp.shape)
    assert man.min().item() >= -pow(2, 16) and man.max().item() < pow(2, 16), "man shape {}".format(man.shape)

    sign = torch.zeros(exp.shape, dtype=torch.int16, device=exp.device)
    man_ = torch.zeros(exp.shape, dtype=torch.int16, device=exp.device)
    exp_ = torch.zeros(exp.shape, dtype=torch.int16, device=exp.device)
    sign.masked_fill_(man < 0, 1)
    sign = sign << 15

    man_2 = None
    rounding_mode = rounding_mode.lower()
    assert rounding_mode in ["trunc", "rne", "round", "rne-hardware"]
    if rounding_mode == "rne":
        man = man.to(torch.float32).div(32).round().to(torch.int16)
        if not enable_overflow:
            # man = man.clip(max=pow(2, 11) - 1, min=-pow(2, 11) + 1)
            man = man.clip(max=pow(2, 11) - 1)
        man = man.abs()
    elif rounding_mode == "trunc":
        man = man.abs()
        man = man >> (15 - 10)
    elif rounding_mode == "rne-hardware":
        man_2 = man.abs()
        man_2 = man_2 >> (15 - 10)
        fix = True
        if fix:
            man = man.abs()
            man = man.to(torch.float32).div(32)
            mask = man >= 2048
            man = man.round().to(torch.int16)
            if not enable_overflow:
                man = man.clip(max=pow(2, 11) - 1)
            man[mask] = 2048
        else:
            man = man.to(torch.float32).div(32).round().to(torch.int16)
            if not enable_overflow:
                man = man.clip(max=pow(2, 11) - 1, min=-pow(2, 11) + 1)
            man = man.abs()
    else:
        raise RuntimeError("to be support: {}".format(rounding_mode))
    man = man.to(torch.int16)
    if man_2 is not None:
        man_2 = man_2.to(torch.int16)

    if is_free_cuda:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # fix corner case
    exp[man == 2048] = exp[man == 2048] + 1
    man[man == 2048] = 1024

    # norm
    man_[man >= 1024] = man[man >= 1024] - 1024
    exp_ = exp + (23 - 8) + 15

    # INF
    mask = exp + (23 - 8) > 15  # INF
    man_[mask] = 0
    exp_[mask] = 16 + 15

    # sub-norm
    mask = exp + (23 - 8) < -14  # sub-norm
    shift = -14 - (exp + (23 - 8))
    exp_[mask] = 0
    if rounding_mode == "rne-hardware":
        man_[mask] = man_2[mask] >> shift[mask]
    else:
        man_[mask] = man[mask] >> shift[mask]

    exp_ = exp_ << 10
    result = sign | exp_ | man_
    result = result.view(torch.float16)
    return result


def E10M17_to_fp32(exp, man, bias=0):  # bias = 511 in certain case
    exp = exp.to(torch.int32)
    man = man.to(torch.int32)

    exp = exp.sub(bias)

    assert exp.min().item() > -pow(2, 9) and exp.max().item() <= pow(2, 9), "exp shape {}".format(exp.shape)
    assert man.min().item() >= -pow(2, 16) and man.max().item() < pow(2, 16), "man shape {}".format(man.shape)

    sign = torch.zeros(exp.shape, dtype=torch.int32, device=exp.device)
    man_ = torch.zeros(exp.shape, dtype=torch.int32, device=exp.device)
    exp_ = torch.zeros(exp.shape, dtype=torch.int32, device=exp.device)
    sign.masked_fill_(man < 0, 1)
    sign = sign << 31

    man = man.abs()  # 16bit
    man = man << 8  # 24bit

    # fix corner case
    exp[man == pow(2, 24)] = exp[man == pow(2, 24)] + 1
    man[man == pow(2, 24)] = pow(2, 23)

    # norm
    man_[man >= pow(2, 23)] = man[man >= pow(2, 23)] - pow(2, 23)  # 23bit
    exp_ = exp + 15 + 127

    # INF
    mask = exp + 15 > 127  # INF
    # clip to max value
    if True:
        man_[mask] = 0x007FFFFF
        exp_[mask] = 127 + 127
    else:
        man_[mask] = 0
        exp_[mask] = 128 + 127

    # sub-norm
    mask = exp + 15 < -127  # sub-norm
    shift = -127 - (exp + 15)
    exp_[mask] = 0
    man_[mask] = man[mask] >> shift[mask]

    exp_ = exp_ << 23
    result = sign | exp_ | man_
    result = result.view(torch.float32)
    return result


# convert E10M17/ FP24 to FP16, round_mode is fixed to 'trunc'
def fp24_to_fp16(x):  # 0x3c11bc00
    x = x.to(torch.float32)
    x_int = x.view(torch.int32)

    mask = 0x7F800000
    x_int_exp = x_int & mask
    x_int_exp = x_int_exp >> 23

    mask = 0x007FFF00
    x_int_mas = x_int & mask
    x_int_mas = x_int_mas
    x_int_mas = x_int_mas >> 8

    x_int_mas = x_int_mas.to(torch.int32)
    implicit = torch.zeros_like(x_int_mas)
    implicit.masked_fill_(x_int_exp != 0, 0x8000)
    x_int_mas.add_(implicit)
    implicit = None

    sign = x < 0
    x_int_mas[sign] = -x_int_mas[sign]
    sign = None

    x_int_exp.sub_(127)
    x_int_exp.clamp_min_(-126)

    # scale down 'exp' as we scale up 'mantissa'
    x_int_exp.sub_(23 - 8)
    x_int_exp = x_int_exp.to(torch.int16)

    # x_int_mas: M17; x_int_exp: E8; Value = x_int_mas (M17) * pow(2, x_int_exp)
    result = E10M17_to_fp16(x_int_exp, x_int_mas)
    return result


import ctypes
import os
import pathlib

import numpy as np

from .xh2a_vp import load_so_file

softfloat_lib = ["../xh_refmodel/vector/softfloat.so", "softfloat.so", "jinghai/softfloat.so"]
softfloat_lib.insert(0, os.path.join(pathlib.Path(__file__).parent.resolve(), "softfloat.so"))
softfloat_lib.insert(0, os.path.join(pathlib.Path(__file__).parent.resolve(), "jinghai/softfloat.so"))

softfloat = load_so_file(softfloat_lib)
if softfloat is None:
    print("[fp24.py] softfloat is None")


def softfloat_fp32_add(
    input1,
    input2,
    round_mode=1,  # 只要支持round_mode = 1
    length=None,
    enable_openmp=0,
    debug=0,
):
    if softfloat is None:
        raise RuntimeError("softfloat not support")

    device = "cpu"
    assert isinstance(input1, torch.Tensor)
    device = input1.device
    input1 = input1.detach().cpu()
    assert input1.dtype in [torch.float32, torch.int32]

    assert isinstance(input2, torch.Tensor)
    input2 = input2.detach().cpu()
    assert input2.dtype in [torch.float32, torch.int32]
    assert len(input1.shape) == len(input2.shape)
    if input1.shape != input2.shape:
        shape = [int(input1.shape[i] / input2.shape[i]) for i in range(len(input2.shape))]
        input2 = input2.repeat(shape)
    assert input1.shape == input2.shape, f"inputs shape mismatch: {input1.shape} vs {input2.shape}"

    input1 = input1.numpy()
    input2 = input2.numpy()

    input1_c = ctypes.c_void_p(input1.view(np.uint32).ctypes.data)
    input2_c = ctypes.c_void_p(input2.view(np.uint32).ctypes.data)

    output = np.zeros(input1.shape, dtype=np.uint32)
    output_c = ctypes.c_void_p(output.ctypes.data)

    if length is None:
        length = int(input1.size)

    softfloat.float32_add_wrapper(
        input1_c,
        input2_c,
        output_c,
        ctypes.c_uint32(length),
        ctypes.c_uint8(round_mode),
        ctypes.c_uint8(enable_openmp),
        ctypes.c_uint8(debug),
    )

    output = torch.from_numpy(output.view(np.float32)).to(device=device)
    return output


def ExMx_to_FP32(
    mans,
    exp,
    round_mode=1,  # 只要支持round mode = 1
    length=None,
    enable_openmp=1,
    debug=0,
):
    if softfloat is None:
        raise RuntimeError("softfloat not support")

    device = "cpu"
    if isinstance(mans, torch.Tensor):
        device = mans.device
        assert mans.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]
        mans = mans.to(torch.int64)
        mans = mans.detach().cpu().numpy()

    if isinstance(exp, torch.Tensor):
        assert exp.dtype in [torch.int8, torch.int16]
        exp = exp.to(torch.int16)
        exp = exp.detach().cpu().numpy()
        assert mans.shape == exp.shape, f"inputs shape mismatch: {mans.shape} vs {exp.shape}"

    input1_c = ctypes.c_void_p(mans.ctypes.data)
    input2_c = ctypes.c_void_p(exp.ctypes.data)

    output = np.zeros(mans.shape, dtype=np.uint32)
    output_c = ctypes.c_void_p(output.ctypes.data)

    if length is None:
        length = int(mans.size)

    softfloat.ExMx_to_FP32_wrapper(
        input1_c,
        input2_c,
        output_c,
        ctypes.c_uint32(length),
        ctypes.c_uint8(round_mode),
        ctypes.c_uint8(enable_openmp),
        ctypes.c_uint8(debug),
    )

    output = torch.from_numpy(output.view(np.float32)).to(device=device)
    return output


def test_fp32tofp24():
    fp32 = torch.rand(100).to(torch.float32)

    fp24 = fp32_to_fp24(fp32, round_mode="rne")
    div = fp24 / fp32
    print(
        div.min().item(),
        div.max().item(),
    )
    print((fp24 - fp32).abs().max().item(), (fp24 - fp32).abs().sum().item())

    fp24_2 = fp32_to_fp24(fp32)
    div = fp24_2 / fp32
    print(
        div.min().item(),
        div.max().item(),
    )
    print((fp24_2 - fp32).abs().max().item(), (fp24_2 - fp32).abs().sum().item())

    print((fp24_2 - fp24).abs().max().item(), (fp24_2 - fp24).abs().sum().item())
    print("")


def test_fp16_exp_log():
    a = torch.ones(1, dtype=torch.int16).cuda()
    for i in range(-128, 128):
        a.fill_(i)
        if a.half().exp2().log2().item() != i:
            print("mismatch {}".format(i))
        else:
            pass  # from -24 to 15

        if a.half().exp2().log2().round().item() != i:
            print("mismatch with round {}".format(i))
        else:
            pass  # from -24 to 15


def test_fp32_exp_log():
    a = torch.ones(1, dtype=torch.int32).cuda()
    for i in range(-128, 128):
        a.fill_(i)
        if a.float().exp2().log2().item() != i:
            print("mismatch {}".format(i))
        else:
            pass  # from -128 to 128


def test_fp32_fp16():
    a = torch.zeros(1)
    a = a.to(torch.float32)
    a_int = a.view(torch.int32)
    a_int[0] = 0x3C11BC00

    b = fp24_to_fp16(a)
    b_int = b.view(torch.int16)
    print(b_int)  # 0x208d(8333) vs 0x208e(8334) should be former
    print(a, b)


def test_E10M17_to_fp16():
    from tools import hex_string

    exp = torch.zeros(1, dtype=torch.int16)
    man = torch.zeros(1, dtype=torch.int32)

    # exp[0] = 0x1fb
    # exp.sub_(511)
    # man[0] = 0x7FFF0000
    # man[0] = 0x80000000 | man[0]
    # print(hex_string(man, bytes=4, nbits=17))
    # print(hex_string(exp + 511, bytes=2, nbits=10))
    # fp16 = E10M17_to_fp16(exp, man, rounding_mode='RNE-hardware')
    # print(hex_string(fp16))

    # exp[0] = 0x1F8
    # exp.sub_(511)
    # man[0] = 0x7FFF0007
    # man[0] = 0x80000000 | man[0]
    # print(hex_string(man, bytes=4, nbits=17))
    # print(hex_string(exp + 511, bytes=2, nbits=10))
    # fp16 = E10M17_to_fp16(exp, man, rounding_mode='RNE-hardware')
    # print(hex_string(fp16))

    # exp[0] = 0x1F8
    # exp.sub_(511)
    # man[0] = 0xFFFF
    # print(hex_string(man, bytes=4, nbits=17))
    # print(hex_string(exp + 511, bytes=2, nbits=10))
    # fp16 = E10M17_to_fp16(exp, man, rounding_mode='RNE-hardware')
    # print(hex_string(fp16))

    exp[0] = 0x1FA
    exp.sub_(511)
    man[0] = 0x7FFF4012
    man[0] = 0x80000000 | man[0]
    print(hex_string(man, bytes=4, nbits=17))
    print(hex_string(exp + 511, bytes=2, nbits=10))
    fp16 = E10M17_to_fp16(exp, man, rounding_mode="RNE-hardware")
    print(hex_string(fp16))

    exp[0] = 0x1FA
    exp.sub_(511)
    man[0] = 0x7FFF4010
    man[0] = 0x80000000 | man[0]
    print(hex_string(man, bytes=4, nbits=17))
    print(hex_string(exp + 511, bytes=2, nbits=10))
    fp16 = E10M17_to_fp16(exp, man, rounding_mode="RNE-hardware")
    print(hex_string(fp16))


def test_E10M17_to_fp32():
    from tools import hex_string

    exp = torch.zeros(1, dtype=torch.int16)
    man = torch.zeros(1, dtype=torch.int32)

    # exp.fill_(-28)
    # man.fill_(43520)
    # fp16 = E10M17_to_fp16(exp, man, rounding_mode="RNE-hardware")
    # fp32 = E10M17_to_fp32(exp, man)
    # print(fp16.to(torch.float32).item(), fp32.item())
    # print(fp16.to(torch.float32).mul(1024).item(), fp32.mul(1024).item())

    # exp.fill_(0x0201 - 511)
    # man.fill_(0x7FFF60F4)
    # man[0] = 0x80000000 | man[0]
    # print(hex_string(man, bytes=4, nbits=17))
    # print(hex_string(exp + 511, bytes=2, nbits=10))
    # fp32 = E10M17_to_fp32(exp, man)
    # print(hex_string(fp32.view(torch.int32), bytes=4))

    exp.fill_(0x0200 - 511)
    man.fill_(0x7FFF0000)
    man[0] = 0x80000000 | man[0]
    print(hex_string(man, bytes=4, nbits=17))
    print(hex_string(exp + 511, bytes=2, nbits=10))
    fp32 = E10M17_to_fp32(exp, man)
    print(hex_string(fp32.view(torch.int32), bytes=4))


if __name__ == "__main__":
    # test_fp32tofp24()
    # test_fp16_exp_log()
    # test_fp32_exp_log()
    # test_fp32_fp16()
    # test_E10M17_to_fp16()
    test_E10M17_to_fp32()
