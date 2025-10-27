from typing import List

import numpy as np
import torch
from torch import nn


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


def find_and_sort_medians(start, end, medians_stack):
    if start > end:
        return  # 递归终止条件

    # 计算当前段的中位数
    median = (start + end) // 2

    # 先递归处理左侧段
    find_and_sort_medians(start, median - 1, medians_stack)
    # 再递归处理右侧段，但确保不重复处理中位数
    find_and_sort_medians(median + 1, end, medians_stack)

    # 在递归返回后，添加当前中位数到栈顶
    medians_stack.append(median)


class NewTableV2(nn.Module):
    def __init__(
        self,
        func,
        cut_points: List,
        table_size=259,
        num_points=32,
        y_min=-65504,
        y_max=65504,
        device="cpu",
    ) -> None:
        super().__init__()
        self.func = func
        self.cut_points = torch.tensor(cut_points, dtype=torch.float32).to(device)
        assert len(self.cut_points) == 11, "cut_points must be 11 points"
        self.num_tables = len(self.cut_points) - 1
        self.table_size = table_size
        self.num_points = num_points
        self.y_min = y_min
        self.y_max = y_max
        self.device = device
        self.create_table()

    def fp32_to_fp16_floor(self, x: torch.Tensor):
        int_tensor = x.view(torch.int32)

        # 提取指数位和mantissa位
        sign_mask = 0x80000000
        exp_mask = 0x7F800000
        mantissa_mask = 0x007FFFFF

        sign_bit = int_tensor & sign_mask
        exp = int_tensor & exp_mask
        mantissa = int_tensor & mantissa_mask

        mantissa = mantissa >> 8 << 8
        result = torch.zeros_like(int_tensor, dtype=torch.int32)
        result = result | sign_bit
        result = result | exp
        result = result | mantissa
        result = result.view(torch.float32)
        return result

    def fp32_to_fp16_ceil(self, x: torch.Tensor):
        int_tensor = x.view(torch.int32)

        # 提取指数位和 mantissa 位
        sign_mask = 0x80000000
        exp_mask = 0x7F800000
        mantissa_mask = 0x007FFFFF

        sign_bit = int_tensor & sign_mask
        exp = int_tensor & exp_mask
        mantissa = int_tensor & mantissa_mask

        # 判断是否需要进位
        if mantissa & 0x1FFF != 0:  # 检查低 13 位是否不为 0
            mantissa = (mantissa >> 13) + 1  # 进位操作
        else:
            mantissa = mantissa >> 13

        mantissa = mantissa << 13
        result = torch.zeros_like(int_tensor, dtype=torch.int32)
        result = result | sign_bit
        result = result | exp
        result = result | mantissa
        result = result.view(torch.float32)
        return result

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

    def create_table(self):
        # 生成插值表
        self.table = torch.zeros(self.table_size, dtype=torch.float16).to(self.device)
        self.index = torch.zeros(self.table_size, dtype=torch.float16).to(self.device)
        # self.index2 = torch.zeros(self.table_size, dtype=torch.float32).to(self.device)

        self.mul_scale = torch.zeros(len(self.cut_points) - 1, dtype=torch.float16).to(self.device)  # 乘子系数
        self.mul_scale[0] = 1 / (self.cut_points[1] - self.cut_points[0])  # .clip(-65504, 65504).half()
        # self.cut_points[0] = (self.cut_points[1] - (1 / self.mul_scale[0])).clip(-65504, 65504).half().float()

        for i in range(1, len(self.cut_points[:-2])):
            start = self.cut_points[i]
            end = self.cut_points[i + 1]
            self.mul_scale[i] = self.num_points / (end - start)
        self.mul_scale[-1] = 1 / (self.cut_points[-1] - self.cut_points[-2])
        print(self.mul_scale)

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

    def create_table_bak(self):
        # 生成插值表
        self.table = torch.zeros(self.table_size, dtype=torch.float16).to(self.device)
        self.index1index = torch.zeros(self.table_size, dtype=torch.float16).to(self.device)

        self.mul_scale = torch.zeros(len(self.cut_points) - 1, dtype=torch.float16).to(self.device)  # 乘子系数

        self.cut_points[0] = (
            (self.cut_points[1] - (1 / self.fp32_to_fp16_ceil((1 / (self.cut_points[1] - self.cut_points[0])))))
            .clip(-65504, 65504)
            .half()
            .float()
        )
        self.mul_scale[0] = self.fp32_to_fp16_ceil((1 / (self.cut_points[1] - self.cut_points[0]))).half()
        # 计算乘系数

        for i in range(1, len(self.cut_points[:-2])):
            start = self.cut_points[i]
            end = self.cut_points[i + 1]

            end = (
                ((self.num_points / self.fp32_to_fp16_floor(self.num_points / (end - start))) + start)
                .clip(-65504, 65504)
                .half()
                .float()
            )
            self.cut_points[i + 1] = end
            self.mul_scale[i] = self.fp32_to_fp16_floor((self.num_points / (end - start))).half()

        self.cut_points[-1] = (
            ((1 / self.fp32_to_fp16_ceil(1 / (self.cut_points[-1] - self.cut_points[-2]))) + self.cut_points[-2])
            .clamp_(-65504, 65504)
            .half()
            .float()
        )
        self.mul_scale[-1] = self.fp32_to_fp16_ceil(1 / (self.cut_points[-1] - self.cut_points[-2]))

        self.index[0] = self.cut_points[0]
        self.cut_points[1] = self.cut_points[0] + 1 / self.mul_scale[0]

        for i in range(1, len(self.cut_points) - 2):
            start = self.cut_points[i]
            end = self.cut_points[i + 1] = self.num_points / self.mul_scale[i] + self.cut_points[i]

            x = torch.linspace(start, end, self.num_points + 1).to(self.device)

            if i != len(self.cut_points) - 3:
                self.index[(i - 1) * self.num_points + 1 : i * self.num_points + 1] = x[:-1].to(torch.float16)
            else:
                self.index[(i - 1) * self.num_points + 1 : -1] = x.to(torch.float16)

        self.cut_points[-1] = self.cut_points[-2] + 1 / self.mul_scale[-1]
        self.index[-1] = self.cut_points[-1]
        self.table = self.func(self.index).clamp(self.y_min, self.y_max).half()

    def forward(self, x):
        # 根据切分点，找到x所在的区间
        cut_indices = (torch.bucketize(x, self.cut_points, right=True).clamp(max=self.num_tables) - 1).clip(0)

        temp = self.fp32_to_fp16_floor((x - self.cut_points[cut_indices]) * self.mul_scale[cut_indices])
        # temp = ((x - self.cut_points[cut_indices]) * self.mul_scale[cut_indices])

        index = temp.floor().to(torch.int32)
        # index[index == 1] = 0
        mask_last_table = (cut_indices == 9) * (index == 1)
        index = torch.where(mask_last_table, torch.zeros_like(index), index)

        decimal = temp - index

        indices = torch.zeros_like(cut_indices, dtype=torch.int64)

        indices[cut_indices == 0] = (0 + index[cut_indices == 0]).long()
        indices[cut_indices >= 1] = (
            1 + (cut_indices[cut_indices >= 1] - 1) * self.num_points + index[cut_indices >= 1]
        ).long()

        # 计算x的插值
        y = self.table[indices.long()] + (self.table[indices.long() + 1] - self.table[indices.long()]) * decimal
        return y


class LUT(nn.Module):
    def __init__(self, func, cut_points: List, table_size=257, min=-65504, max=65504, device="cpu") -> None:
        super().__init__()
        self.func = func
        self.cut_points = cut_points
        self.table_size = table_size
        self.min = min
        self.max = max
        self.num_points = (table_size - 1) // (len(cut_points) + 1)
        self.device = device
        self.creat_table()

    def creat_table(self):
        # 生成插值表
        self.table = torch.zeros(self.table_size, dtype=torch.float32).to(self.device)
        self.index = torch.zeros(self.table_size, dtype=torch.float16).to(self.device)
        self.all_points = all_points = [self.min] + self.cut_points + [self.max]
        for i in range(len(all_points) - 1):
            start = all_points[i]
            end = all_points[i + 1]
            x = torch.linspace(start, end, self.num_points + 1).to(self.device)

            if i != len(all_points) - 2:
                self.index[i * self.num_points : (i + 1) * self.num_points] = x[:-1].to(torch.float16)
                y = self.func(x).clamp(-65504, 65504)
                y = y.view(torch.int32) >> 8 << 8
                y = y.view(torch.float32)

                self.table[i * self.num_points : (i + 1) * self.num_points] = y[:-1]
            else:
                self.index[i * self.num_points :] = x.to(torch.float16)
                y = self.func(x).clamp(-65504, 65504)
                y = y.view(torch.int32) >> 8 << 8
                y = y.view(torch.float32)
                self.table[i * self.num_points :] = y

    def forward(self, x):
        # x = x.clamp(self.min, self.max)

        # 根据切分点，找到x所在的区间
        indices = torch.bucketize(x, self.index, right=True).clip(1, self.table_size - 1)

        # 计算x到左右两个切分点的距离
        interval = self.index[indices] - self.index[indices - 1]
        interval[interval == 0] = 1e-5
        m1 = (x - self.index[indices - 1]) / interval
        m2 = 1 - m1

        # 计算x的插值
        y = self.table[indices - 1] * m2.half() + self.table[indices] * m1.half()
        return y.half()


class LUT_B(nn.Module):
    def __init__(self, func, table_size=32, start=-65504, end=65504, device="cpu") -> None:
        super().__init__()
        self.func = func
        self.table_size = table_size
        self.start = start
        self.end = end
        self.device = device
        self.creat_table()

    def creat_table(self):
        # 生成插值表
        self.table = torch.zeros(self.table_size + 1, dtype=torch.float16).to(self.device)
        self.index = torch.zeros(self.table_size + 1, dtype=torch.float16).to(self.device)
        x = torch.linspace(self.start, self.end, self.table_size + 1).to(self.device)

        self.index = x
        self.table = self.func(x).clamp(-65504, 65504).to(torch.float16)

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
        # 根据切分点，找到x所在的区间
        indices = torch.bucketize(x, self.index, right=True).clip(1, self.table_size - 1)

        self.mul_scale = self.fp32_to_fp16_floor(1 / (self.index[indices] - self.index[indices - 1]))

        # 计算x到左右两个切分点的距离
        m1 = self.fp32_to_fp16_floor((x.float() - self.index[indices - 1])).float() * self.mul_scale

        # 计算x的插值
        y = self.fp32_to_fp16_floor(
            self.table[indices - 1]
            + self.fp32_to_fp16_floor(
                self.fp32_to_fp16_floor(self.table[indices].float() - self.table[indices - 1]) * m1.float()
            ).float()
        )
        return y


from tqdm import tqdm


def get_best_luts(func, x_tensors, device="cpu"):
    # 初始化中位数栈
    medians_stack = []

    # 调用函数，范围从0到256
    find_and_sort_medians(1, 255, medians_stack)

    # 因为中位数是按相反顺序入栈的，我们需要反转栈以获得正确的顺序
    medians_stack.reverse()

    eps = 1e-3

    def get_nonlinear_cut_points(func, x_tensors):
        # 迭代优化法
        def update_nexts(best_points_indexes, i, table_size, x):
            interval = (len(x) - best_points_indexes[i]) // (table_size - 1 - i)
            for j in range(1, table_size - 1 - i):
                best_points_indexes[i + j] = best_points_indexes[i] + j * interval

        def select_best_points(func, x: torch.Tensor):
            # 表的大小
            table_size = 257

            # 间隔 248
            interval = len(x) // (table_size - 1)

            # 初始化257个点
            best_points = torch.zeros(table_size, dtype=torch.float16).to(device)

            best_points_indexes = [0] * table_size

            # 首尾固定
            best_points[0] = x[0]
            best_points[-1] = x[-1]

            best_points_indexes[0] = 0
            best_points_indexes[-1] = len(x) - 1

            # 根据间隔初始化切分点
            for i in range(1, table_size - 1):
                best_points[i] = x[i * interval]
                best_points_indexes[i] = i * interval

            float_out = func(x.float()).clip(-65504, 65504)
            eat_list = [0] * len(best_points_indexes)

            diffs = list()
            progress_bar = tqdm(medians_stack)
            progress_bar.set_description("for select_best_points")
            # for i in tqdm(medians_stack):
            for i in progress_bar:
                if eat_list[i] == 1:
                    continue
                diff_max = torch.inf
                pre_index = best_points_indexes[i - 1]
                best_index = None  # 初始化 best_index

                step = 1
                next_index = best_points_indexes[i + step]
                j = pre_index + 1

                right_eat = 0
                left_eat = 0

                while j < next_index:
                    # 计算[i - 1, i + 1]之间的误差，取最小值
                    # 左侧的插值计算得到的 和 浮点的 误差
                    if float_out[pre_index] == float_out[j]:
                        diff_left = float_out[pre_index] - float_out[j]  # 0
                    else:
                        m1 = (x.float()[pre_index:j] - x.float()[pre_index]) / (x.float()[j] - x.float()[pre_index])
                        # m2 = 1 - m1
                        left_lut_out = (
                            (float_out[pre_index] + m1 * (float_out[j] - float_out[pre_index]))
                            .clip(-65504, 65504)
                            .float()
                        )
                        left_out = float_out[pre_index:j]
                        diff_left = torch.abs((left_lut_out - left_out)).mean()

                    # 右侧的插值计算得到的 和 浮点的 误差
                    if float_out[j] == float_out[next_index]:
                        diff_right = float_out[j] - float_out[next_index]
                    else:
                        m1 = (x.float()[j:next_index] - x.float()[j]) / (x.float()[next_index] - x.float()[j]).double()
                        # m2 = 1 - m1
                        right_lut_out = (
                            (float_out[j] + m1 * (float_out[next_index] - float_out[j])).clip(-65504, 65504).float()
                        )
                        right_out = float_out[j:next_index]
                        diff_right = torch.abs((right_lut_out - right_out)).mean()

                    diff = diff_left + diff_right  # 从最左侧开始找 有可能不是最优的

                    if diff <= diff_max:
                        # if diff <= diff_max:
                        diff_max = diff
                        best_index = j  # 更新 best_index
                    j += 1

                    # if j == len(x) - 1:
                    #     break

                    if j == (next_index - 1):
                        # 确保 best_index 已经被赋值
                        if best_index is None:
                            best_index = j  # 如果没有找到更好的点，使用当前点

                        if best_index == next_index - 1:
                            if i + step > table_size - 1:
                                continue
                            else:
                                step += 1
                                next_index = best_points_indexes[i + step]
                                eat_list[i + step] = 1
                            right_eat += 1

                        if best_index == pre_index + 1:
                            if i - step < 1:
                                continue
                            else:
                                step += 1
                                next_index = pre_index + 1
                                pre_index = best_points_indexes[i - step]
                                eat_list[i - step] = 1
                                j = pre_index + 1
                            left_eat += 1

                # 确保 best_index 已经被赋值
                if best_index is None:
                    best_index = pre_index + 1  # 如果没有找到更好的点，使用前一个点的下一个位置

                diffs.append(diff_max)
                best_points_indexes[i] = best_index

                # if j == len(x) - 1:
                #     break

                # update_nexts(best_points_indexes, i, table_size, x)
            # print(i)
            for i, eat_if in enumerate(eat_list):
                if eat_if:
                    best_points_indexes[i] = -1
            progress_bar.close()
            best_points_indexes = sorted(best_points_indexes)
            best_points_indexes = sorted(list(set(best_points_indexes)))
            best_points_indexes.pop(0)
            return best_points_indexes, diffs

        cuts, diffs = select_best_points(func, x_tensors)
        return cuts, diffs

    cut_points, diffs = get_nonlinear_cut_points(func, x_tensors)
    cut_points = sorted(cut_points)
    # print(cut_points)

    # for index, i in enumerate(cut_points):
    #     if i>15506 and i < 20000:
    #         cut_points[index] = 358
    # cut_points = sorted( list(set(cut_points)) )

    new_diffs = []
    for i in range(len(cut_points) - 1):
        start = x_tensors[cut_points[i]].item()
        end = x_tensors[cut_points[i + 1]].item()
        lut = LUT_B(func, table_size=32, start=start, end=end, device=device)
        float_out = func(x_tensors.float())[cut_points[i] : cut_points[i + 1]].clamp(-65504, 65504)
        lut_out = lut(x_tensors.float())[cut_points[i] : cut_points[i + 1]].clamp(-65504, 65504)
        diff = ((float_out - lut_out).abs() / (float_out.abs() + eps)).mean().item()
        new_diffs.append(diff)

    # base = torch.sum(torch.tensor(new_diffs))
    iter_steps = len(cut_points) - 11
    pbar = tqdm(total=iter_steps)
    while len(cut_points) > 11:
        per_table_size = max(256 // (len(cut_points) - 2), 1)
        conbine_diffs = []

        # medians_stack = []
        # find_and_sort_medians(1, len(cut_points) - 1, medians_stack)
        # medians_stack.reverse()

        for i in range(1, len(cut_points) - 1):
            start = x_tensors[cut_points[i - 1]].item()
            end = x_tensors[cut_points[i + 1]].item()
            if start == x_tensors.min() or end == x_tensors.max():
                lut = LUT_B(func, table_size=1, start=start, end=end, device=device)
            else:
                lut = LUT_B(func, table_size=per_table_size, start=start, end=end, device=device)
            float_out = func(x_tensors.float())[cut_points[i - 1] : cut_points[i + 1]].clamp(-65504, 65504)
            lut_out = lut(x_tensors.float())[cut_points[i - 1] : cut_points[i + 1]].clamp(-65504, 65504)
            diff = ((float_out - lut_out).abs() / (float_out.abs() + eps)).mean().item()
            conbine_diffs.append(diff)
        indx = torch.tensor(conbine_diffs).argmin().item()
        cut_points.pop(indx + 1)
        new_diffs.pop(indx + 1)
        new_diffs[indx] = conbine_diffs[indx]
        # base = torch.sum(torch.tensor(new_diffs))
        pbar.update()

    best_cut_points = [0] * 11
    best_cut_points[0] = x_tensors[cut_points[0]].cpu().item()
    best_cut_points[-1] = x_tensors[cut_points[-1]].cpu().item()
    best_cut_points[1:-1] = x_tensors[cut_points[1:-1]].tolist()

    # print(best_cut_points)
    # best_lut = NewTableV2(func, best_cut_points, table_size=259, device=device)
    # i_data = x_tensors.clone()
    # i_data = i_data.clamp(best_lut.cut_points[0], best_lut.cut_points[-1])
    # float_out = func(i_data.float()).clamp(-65504, 65504)
    # lut_out = best_lut(i_data).clamp(-65504, 65504)
    # diff = ((float_out - lut_out).abs() / (torch.maximum(lut_out.abs(), float_out.abs()) + eps)).mean().item()
    return x_tensors[cut_points].tolist(), None, None


def plot_points(x, y, show_points=None, func_name="act"):
    import matplotlib.pyplot as plt

    # 创建图形和轴
    fig, ax = plt.subplots()

    # 绘制曲线
    ax.plot(x, y, label=func_name.split("_")[0])

    # 特定点：比如正弦函数的最大值和最小值
    if show_points is not None:
        for point in show_points:
            ax.plot(*point, "ro")  # 红色的圆点

            # 在这些点绘制对应的坐标线
            # 最大值点的垂直和水平线
            ax.axvline(x=point[0], color="r", linestyle="--", linewidth=0.5)
            ax.axhline(y=point[1], color="r", linestyle="--", linewidth=0.5)

            # 标注坐标值
            ax.annotate(
                f"({point[0]:.2f}, {point[1]:.2f})", xy=point, textcoords="offset points", xytext=(10, -10), fontsize=4
            )
            ax.annotate(
                f"({point[0]:.2f}, {point[1]:.2f})", xy=point, textcoords="offset points", xytext=(10, -10), fontsize=4
            )

    # 设置图例
    ax.legend()

    # 展示图形
    fig.savefig(f"{func_name}.png", dpi=300)


def divide(x):
    return 1 / x


import torch.nn.functional as F
from torch import nn


class Swish(nn.Module):
    def __init__(self, beta=1.0) -> None:
        super().__init__()
        self.beta = beta
        self.sigmoid = nn.Sigmoid()
        self.__name__ = f"swish"

    def forward(self, x):
        out = x * self.sigmoid(self.beta * x)
        return out


class InverseSigmoid(nn.Module):
    def __init__(self, eps=1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.__name__ = f"inversesigmoid_{eps}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=self.eps, max=1)
        x2 = (1 - x).clamp(min=self.eps, max=1)
        return torch.log(x1 / x2)


class PowM:
    def __init__(self, m) -> None:
        self.m = float(m)
        self.__name__ = f"pow_{m}"

    def __call__(self, x):
        return x**self.m


def hardswish(x):
    return x * F.relu6(x + 3, inplace=True) / 6


# 处理周期函数的输入 取mod
def mod_data(x, cycle=None, positive_only=False):
    if cycle is None:
        pi_int_2 = 2 * torch.pi  # torch.tensor(torch.pi, dtype=torch.float16).view(torch.int16)
        mod_data = x % pi_int_2
        if not positive_only:
            mod_data = mod_data - torch.pi
        mod_data = mod_data.sort(descending=False)[0]
        return mod_data
    else:
        mod_data = x % cycle
        mod_data = mod_data.sort(descending=False)[0]
        return mod_data


def main():
    func_list = [
        nn.LeakyReLU(0.1),
        torch.exp,
        torch.log,
        divide,
        F.silu,
        F.sigmoid,
        Swish(1.702),
        InverseSigmoid(),
        PowM(2),
        hardswish,
        F.gelu,
        F.tanh,
        F.mish,
        F.elu,
        F.softplus,
        torch.sin,
        torch.cos,
        F.relu,
    ]
    # func_list = [ PowM(2), hardswish, F.gelu, F.tanh, F.mish, F.elu, F.softplus, torch.sin, torch.cos,]
    # 生成所有fp16值（除了NaN和无穷大）
    fp16_values = generate_all_fp16_values()

    fp16_values = torch.tensor(fp16_values, dtype=torch.float16).sort(descending=False)[0]
    print("Number of fp16 values generated (excluding NaN and inf):", len(fp16_values))

    device = "cuda:1"

    func_names = list()
    re_diff = list()

    import pandas as pd

    for func in func_list:
        data = fp16_values.clone()
        if hasattr(func, "__name__"):
            if func.__name__ in ["log", "divide"]:
                data = data[data > 0]
            if "inversesigmoid" in func.__name__:
                data = data[data.abs() < 1]
            if func.__name__ in ["exp"]:
                data = data[data <= 0]

            if func.__name__ in ["sin", "cos"]:
                data = mod_data(data)

            name = func.__name__
        else:
            name = "leakyrelu"

        if name != "relu":
            continue

        func_names.append(name)
        print(name)

        best_cut_points, best_lut, diff = get_best_luts(func, data.to(device), device=device)
        re_diff.append(diff)
        print(diff)

        best_cut_points = torch.tensor(best_cut_points, device=device)

        y_best_cut_points = best_lut(best_cut_points)

        points = torch.stack((best_cut_points, y_best_cut_points), 1).tolist()

        plot_points(
            best_lut.index.cpu().numpy(), best_lut.table.cpu().numpy(), show_points=None, func_name=name + "_all"
        )
        start = 1
        end = -1

        t_start = 1
        t_end = -1
        if "divide" in name:
            start = 3
            end = -4
            t_start = 1 + 32 * 2
            t_end = -1 - 32 * 3

        if name == "log":
            end = -3
            t_end = -1 - 32 * 2
        if "pow_2" in name:
            start = 2
            end = -2
            t_start = 1 + 32 * 1
            t_end = -1 - 32 * 1
        plot_points(
            best_lut.index.cpu().numpy()[t_start:t_end],
            best_lut.table.cpu().numpy()[t_start:t_end],
            points[start:end],
            func_name=name + "_cut",
        )
        print(best_cut_points)
    df = pd.DataFrame({"func": func_names, "re diff": re_diff})
    df.to_csv("lut_errors.csv", index=False)


if __name__ == "__main__":
    main()
