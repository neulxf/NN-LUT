import ctypes
import os
import pathlib

import numpy as np
import torch
from loguru import logger

__ALL__ = ["sum", "dot", "ip", "fp16_sum"]


def load_so_file(filename_list=[]):
    dll = None
    for filename in filename_list:
        if os.path.isfile(filename):
            try:
                dll = ctypes.CDLL(filename)
                # logger.info(
                #     "[xh2_vp.py] load so file ({}) from working path {}".format(
                #         filename,
                #         os.path.abspath(os.getcwd()),
                #     )
                # )
                return dll
            except Exception as e:
                logger.info(
                    "[xh2_vp.py] Load so file {} from working path {} failed with error {}".format(
                        filename, os.path.abspath(os.getcwd()), e
                    )
                )
    return dll


accu_dll = ["../xh_refmodel/accu/fp16_sum.so", "fp16_sum.so"]
ip_dll = ["../xh_refmodel/ip/fp16_ip.so", "fp16_ip.so"]
accu_dll.insert(0, os.path.join(pathlib.Path(__file__).parent.resolve(), "fp16_sum.so"))
ip_dll.insert(0, os.path.join(pathlib.Path(__file__).parent.resolve(), "fp16_ip.so"))

fp16_sum = load_so_file(accu_dll)
if fp16_sum is None:
    print("[xh2_vp.py] fp16_sum is None")

ip = load_so_file(ip_dll)
if ip is None:
    print("[xh2_vp.py] ip is None")


def sum(
    input: torch.Tensor,
    dim,
    keepdim=True,
    to_fp32=False,
    fp16_exp_sub_value=0,
    round_mode=0,
):
    dev = input.device
    if fp16_sum is None:
        raise RuntimeError("module of VP accu not available")
    if not input.is_contiguous():
        input = input.contiguous()
    float_out_type = 0 if to_fp32 in [0, False] else 1
    neg_zero = 1
    assert isinstance(round_mode, int) and isinstance(fp16_exp_sub_value, int) and isinstance(float_out_type, int)
    assert round_mode in [0, 1, 2, 3, 4]
    assert float_out_type in [0, 1]
    assert fp16_exp_sub_value >= 0 and fp16_exp_sub_value <= 64
    assert keepdim in [0, 1, True, False]

    assert isinstance(input, torch.Tensor) and input.dtype == torch.float16, "invalid input"
    if isinstance(dim, int):
        dim = [dim]
    reduce_dim = set()
    for d in dim:
        if d < -input.ndim or d >= input.ndim:
            raise Exception(
                "Range error! Valid dim should be [%d, %d], but found %d" % (-input.ndim, input.ndim - 1),
                d,
            )
        if d < 0:
            d += input.ndim
        reduce_dim.add(d)
    # target_shapes = [input.shape[i] for i in range(input.ndim)] # target_shape用来
    target_shapes = list()
    reserve_shapes = []
    reduce_shapes = []
    outer_dims = 1
    inner_dims = 1
    for i in range(input.ndim):
        if i in reduce_dim:
            if keepdim:
                target_shapes.append(1)
            reduce_shapes.append(i)
            inner_dims *= input.shape[i]
        else:
            target_shapes.append(input.shape[i])
            outer_dims *= input.shape[i]
            reserve_shapes.append(i)
    final_permute_shapes = reserve_shapes + reduce_shapes
    # num_reduce_dims = len(reduce_shapes)
    input = input.permute(*final_permute_shapes).contiguous()
    input = input.view(outer_dims, inner_dims).contiguous().detach().cpu().numpy()

    length = inner_dims

    if float_out_type == 0:
        output = np.zeros(input.shape[0], dtype=np.float16)
    else:
        output = np.zeros(input.shape[0], dtype=np.float32)

    sum_i(
        input,
        output,
        float_out_type,
        round_mode,
        fp16_exp_sub_value,
        neg_zero,
        length,
        0,
        outer_dims,
    )

    output = torch.from_numpy(output).to(dev)
    output = output.view(*target_shapes)
    return output


def sum_i(
    input,
    output,
    float_out_type,
    round_mode,
    fp16_exp_sub_value,
    neg_zero=False,
    length=None,
    offset=0,
    stride=1,
):
    if neg_zero in [0, False]:
        neg_zero = 0
    else:
        neg_zero = 1

    assert isinstance(stride, int) and stride > 0
    debug = 0

    exception = np.zeros(1, dtype=np.int16)
    exception_c = ctypes.c_void_p(exception.view(np.int16).ctypes.data)

    output_offset = input_offset = offset
    input_ptr = ctypes.c_void_p(input.view(np.int16).ctypes.data)
    if float_out_type == 0:
        output_ptr = ctypes.c_void_p(output.view(np.int16).ctypes.data)
        fp16_sum.fp16_sum_fp16_wrapper(
            input_ptr,
            ctypes.c_int(length),
            output_ptr,
            round_mode,
            fp16_exp_sub_value,
            neg_zero,
            exception_c,
            ctypes.c_int64(input_offset),
            ctypes.c_int64(stride),
            ctypes.c_int64(output_offset),
            debug,
        )
    else:
        output_ptr = ctypes.c_void_p(output.view(np.int32).ctypes.data)
        fp16_sum.fp16_sum_fp32_wrapper(
            input_ptr,
            ctypes.c_int(length),
            output_ptr,
            round_mode,
            neg_zero,
            exception_c,
            ctypes.c_int64(input_offset),
            ctypes.c_int64(stride),
            ctypes.c_int64(output_offset),
            debug,
        )


def dot(
    input1,
    input2,
    dim,
    keepdim=True,
    to_fp32=False,
    fp16_exp_sub_value=0,
    round_mode=0,
    is_lut=0,
    enable_openmp=1,
    debug=0,
):
    if ip is None:
        raise RuntimeError("module of VP dot not available")
    if not input1.is_contiguous():
        input1 = input1.contiguous()
    if not input2.is_contiguous():
        input2 = input2.contiguous()
    assert isinstance(input1, torch.Tensor) and input1.dtype == torch.float16, "invalid input1"
    assert isinstance(input2, torch.Tensor) and input2.dtype == torch.float16, "invalid input2"
    assert input1.shape == input2.shape, "invalid length of inputs"

    float_out_type = 0 if to_fp32 in [0, False] else 1
    assert isinstance(round_mode, int) and isinstance(fp16_exp_sub_value, int) and isinstance(float_out_type, int)
    assert round_mode in [0, 1, 2, 3, 4]
    assert float_out_type in [0, 1]
    assert fp16_exp_sub_value >= 0 and fp16_exp_sub_value <= 64
    assert keepdim in [0, 1, True, False]
    if input1.shape != input2.shape:
        raise RuntimeError("dot requires exactly the same shape of the 2 input tensors")
    if isinstance(dim, int):
        dim = [dim]
    reduce_dim = set()
    for d in dim:
        if d < -input1.ndim or d >= input1.ndim:
            raise Exception(
                "Range error! Valid dim should be [%d, %d], but found %d" % (-input1.ndim, input1.ndim - 1),
                d,
            )
        if d < 0:
            d += input1.ndim
        reduce_dim.add(d)
    target_shapes = [input1.shape[i] for i in range(input1.ndim)]
    reserve_shapes = []
    reduce_shapes = []
    outer_dims = 1
    inner_dims = 1
    for i in range(input1.ndim):
        if i in reduce_dim:
            target_shapes[i] = 1
            reduce_shapes.append(i)
            inner_dims *= input1.shape[i]
        else:
            outer_dims *= input1.shape[i]
            reserve_shapes.append(i)
    final_permute_shapes = reserve_shapes + reduce_shapes
    # num_reduce_dims = len(reduce_shapes)
    input = input1.permute(*final_permute_shapes).contiguous()
    input1_ = input1.view(outer_dims, inner_dims).contiguous().cpu().numpy()

    input2 = input2.permute(*final_permute_shapes).contiguous()
    input2_ = input2.view(outer_dims, inner_dims).contiguous().cpu().numpy()

    input1_c = ctypes.c_void_p(input1_.view(np.int16).ctypes.data)
    input2_c = ctypes.c_void_p(input2_.view(np.int16).ctypes.data)
    length = inner_dims

    if float_out_type == 0:
        output = np.zeros(input1_.shape[0], dtype=np.float16)
        output_c = ctypes.c_void_p(output.ctypes.data)
    else:
        output = np.zeros(input1_.shape[0], dtype=np.float32)
        output_c = ctypes.c_void_p(output.ctypes.data)

    dot_i(
        input1_c,
        input2_c,
        output_c,
        float_out_type,
        round_mode,
        fp16_exp_sub_value,
        length,
        0,
        outer_dims,
        is_lut=is_lut,
        enable_openmp=enable_openmp,
        debug=debug,
    )

    output = torch.from_numpy(output).to(input.device)
    if keepdim:
        output = output.view(*target_shapes)
    else:
        output = output.view(*reserve_shapes)
    return output


def dot_i(
    input1_c,
    input2_c,
    output_c,
    float_out_type,
    round_mode,
    fp16_exp_sub_value,
    length=None,
    offset=0,
    stride=1,
    is_lut=0,
    enable_openmp=1,
    debug=0,
):
    assert isinstance(stride, int) and stride > 0

    if length is None:
        length = int(input1_c.size)

    exception = np.zeros(1, dtype=np.int8)
    exception_c = ctypes.c_void_p(exception.view(np.int8).ctypes.data)

    input_offset = output_offset = offset

    ip.fp16_inner_product_wrapper(
        input1_c,
        input2_c,
        ctypes.c_int(length),
        output_c,
        ctypes.c_uint8(round_mode),
        ctypes.c_uint8(float_out_type),
        ctypes.c_uint8(fp16_exp_sub_value),
        exception_c,
        ctypes.c_int64(input_offset),
        ctypes.c_int64(stride),
        ctypes.c_int64(output_offset),
        ctypes.c_uint8(is_lut),
        ctypes.c_uint8(enable_openmp),
        ctypes.c_uint8(debug),
    )
