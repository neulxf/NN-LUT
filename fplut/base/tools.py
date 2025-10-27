import inspect
import os

import numpy as np
import torch


def tohex(val, nbits=None, bytes=2):
    if isinstance(val, torch.Tensor) and val.numel() == 1:
        val = val.item()

    if nbits is None:
        nbits = bytes * 8

    if bytes == 1:
        # return hex((val + (1 << nbits)) % (1 << nbits))
        return "0x{:02X}".format((val + (1 << nbits)) % (1 << nbits))
    elif bytes == 2:
        return "0x{:04X}".format((val + (1 << nbits)) % (1 << nbits))
    elif bytes == 4:
        return "0x{:08X}".format((val + (1 << nbits)) % (1 << nbits))
    elif bytes == 8:
        return "0x{:016X}".format((val + (1 << nbits)) % (1 << nbits))
    else:
        raise RuntimeError("no support byte length {}".format(bytes))


def hex_string(tensor, bytes=2, nbits=None, separator=", ", line_limit=None):
    string = ""
    if isinstance(tensor, torch.Tensor):
        if tensor.dtype in [torch.float16]:
            tensor = tensor.view(torch.int16)

    if isinstance(tensor, torch.Tensor) and tensor.numel() == 1:
        string = string + "{}".format(tohex(tensor.item(), bytes=bytes, nbits=nbits))
        return string

    for i, val in enumerate(tensor):
        if i == len(tensor) - 1:
            string = string + "{}".format(tohex(val, bytes=bytes, nbits=nbits))
        else:
            string = string + "{}{}".format(tohex(val, bytes=bytes, nbits=nbits), separator)

        if line_limit is not None and isinstance(line_limit, int) and line_limit >= 1:
            if (i % line_limit) == (line_limit - 1):
                string = string + "\n"
    return string


def dec_string(tensor, separator=", ", line_limit=None):
    string = ""
    if isinstance(tensor, torch.Tensor):
        if tensor.dtype in [torch.float16]:
            tensor = tensor.view(torch.int16)
        tensor = tensor.reshape(-1)

    if isinstance(tensor, torch.Tensor) and tensor.numel() == 1:
        string = string + "{}".format(tensor.item())
        return string

    for i, val in enumerate(tensor):
        if isinstance(val, torch.Tensor):
            val = val.item()

        if i == len(tensor) - 1:
            string = string + "{}".format(val)
        else:
            string = string + "{}{}".format(val, separator)

        if line_limit is not None and isinstance(line_limit, int) and line_limit >= 1:
            if (i % line_limit) == (line_limit - 1):
                string = string + "\n"
    return string


def save_buffer(x, name, is_hex=False, bytes=2, nbits=None):
    if isinstance(x, np.ndarray):
        x = x.tolist()
    elif isinstance(x, torch.Tensor):
        x = x.detach().reshape(-1).cpu().numpy().tolist()

    print("saving data into {}".format(name))
    fo = open(name, "w")
    for i, val in enumerate(x):
        fo.write(f"{val}\n")
    fo.close()

    if is_hex:
        hex_name = name.replace(".txt", ".hex.txt")
        print("saving data into {}".format(hex_name))
        fo = open(hex_name, "w")
        for i, val in enumerate(x):
            fo.write("{}\n".format(tohex(val, bytes=bytes, nbits=nbits)))
        fo.close()


def load_buffer(name, is_hex=False):
    if not os.path.isfile(name):
        return None

    fo = open(name, "r")
    lines = fo.readlines()
    fo.close()

    x = []
    for line in lines:
        if is_hex:
            integer = int(line, 16)
        else:
            integer = int(line)
        x.append(integer)

    if len(x) == 0:
        return None

    return x


def comparison(baseline, value, strict=True):
    if baseline is None or value is None:
        return
    if type(baseline) in (list, tuple):
        assert type(value) in (list, tuple)
        assert len(baseline) == len(value)
        for b, v in zip(baseline, value):
            comparison(b, v, strict)
    if isinstance(baseline, torch.Tensor):
        if baseline.dtype == value.dtype:
            if baseline.dtype in [torch.float16]:
                baseline = baseline.view(torch.int16)
                value = value.view(torch.int16)
            elif baseline.dtype in [torch.float32]:
                baseline = baseline.view(torch.int32)
                value = value.view(torch.int32)
            elif baseline.dtype in [torch.float64]:
                baseline = baseline.view(torch.int64)
                value = value.view(torch.int64)

            differ = baseline != value
            if strict:
                assert differ.sum().item() == 0, "value does not match baseline, {} in {} mismatch".format(
                    differ.sum().item(), differ.numel()
                )
    # else:
    #     raise RuntimeError("Unknown result type")


def init_tensor_with_fix_val(tensor: torch.Tensor):
    shape = tensor.shape
    size = np.prod((shape))
    val = range(size)

    val = np.array(val, dtype=np.int64)
    if tensor.element_size() == 1:
        val = val.astype(dtype=np.int8)
    elif tensor.element_size() == 2:
        val = val.astype(dtype=np.int16)
    elif tensor.element_size() == 4:
        val = val.astype(dtype=np.int32)
    elif tensor.element_size() == 8:
        val = val.astype(dtype=np.int64)
    else:
        raise RuntimeError("element_size: {} not support".format(tensor.element_size()))

    val = torch.from_numpy(val).to(tensor.device).reshape(shape).view(tensor.dtype)
    return val


def unit_test(func, throw_error=False):
    if throw_error:
        func()
    else:
        try:
            func()
        except Exception as e:
            string = str(e)
            if hasattr(func, "__name__"):
                string = f"{func.__name__} {string}"
            for i, stack in enumerate(inspect.stack()):
                if stack[3] == "unit_test":
                    string = f"[{inspect.stack()[i+1][3]}]: {string}"
                    break
            print(string)


def check_should_equal(tensor, val):
    # assert len(tensor) == len(val)
    if tensor.element_size() == 1:
        tensor = tensor.view(torch.int8)
    elif tensor.element_size() == 2:
        tensor = tensor.view(torch.int16)
    elif tensor.element_size() == 4:
        tensor = tensor.view(torch.int32)
    elif tensor.element_size() == 8:
        tensor = tensor.view(torch.int64)
    else:
        raise RuntimeError("element_size: {} not support".format(tensor.element_size()))

    if isinstance(val, torch.Tensor):
        if val.element_size() == 1:
            val = val.view(torch.int8)
        elif val.element_size() == 2:
            val = val.view(torch.int16)
        elif val.element_size() == 4:
            val = val.view(torch.int32)
        elif val.element_size() == 8:
            val = val.view(torch.int64)
    else:
        if tensor.element_size() == 1:
            val = np.array(val, dtype=np.int8)
        elif tensor.element_size() == 2:
            val = np.array(val, dtype=np.int16)
        elif tensor.element_size() == 4:
            val = np.array(val, dtype=np.int32)
        elif tensor.element_size() == 8:
            val = np.array(val, dtype=np.int64)
        val = torch.from_numpy(val).to(tensor.device).reshape(tensor.shape).view(tensor.dtype)

    assert (val != tensor).sum() == 0, "not pass"
