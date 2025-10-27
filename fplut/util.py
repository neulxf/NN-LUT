def generate_all_fp16_values():
    all_values = []
    for exp in range(0, 32):  # 指数有32种可能 (0到31)
        if exp == 31:  # 排除指数为31的情况，这代表特殊值
            continue
        for frac in range(0, 1024):  # 尾数有1024种可能 (0到1023)
            for sign in range(0, 2):  # 符号位，0代表正数，1代表负数
                # # 构建二进制数值
                # binary_value = (sign << 15) | (exp << 10) | fracå
                # 将整数值转为fp16浮点数
                if exp == 0:
                    fp16 = (-1) ** sign * 2 ** (-14) * (frac / 1024)
                else:
                    fp16 = (-1) ** sign * 2 ** (exp - 15) * (1 + frac / 1024)
                all_values.append(fp16)
    return all_values
