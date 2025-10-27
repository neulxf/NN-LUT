import torch
from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple, List, Any, Union

# 分发器,首先需要根据函数名和性质，生成对应的函数唯一名,然后根据根据性质进行合适的参数生成,如限定最大值/最小值, 生成对应的点数

# 为不同的函数采取不同的方式生成，我该怎么设计这个架构？
# 有一个函数/方法 从cache中读取缓存
from ...base.xh2_fp_lut import NewTable
from ._generate_split_points_v1 import generate_cut_points as generate_cut_points_v1
from ._generate_split_points_v2 import generate_cut_points_triton as generate_cut_points_v2, if_can_v2
from ._cached_cut_points_v1 import cut_points_dict as cut_points_dict_v1
from ._cached_cut_points_v2 import cut_points_dict as cut_points_dict_v2


@dataclass
class SplitConfig:
    func_name: str
    unique_name: str = None
    func: Callable = None
    func_propertys: Tuple = None
    x_range: Tuple[float, float] = None
    num_points: int = None
    metric: str = "rel_l1"
    left_edge: float = None
    right_edge: float = None


class SplitPointStrategy:
    func_name: str
    unique_name: str
    func: Callable
    func_propertys: Tuple[Any, Any]
    x_range: Tuple[float, float]
    num_points: int
    metric: str
    left_edge: float
    right_edge: float

    def __init__(self, split_config: SplitConfig):
        func_name, unique_name, func, func_propertys, x_range, num_points, metric, left_edge, right_edge = (
            split_config.func_name,
            split_config.unique_name,
            split_config.func,
            split_config.func_propertys,
            split_config.x_range,
            split_config.num_points,
            split_config.metric,
            split_config.left_edge,
            split_config.right_edge,
        )
        assert metric.lower() in ["l2", "l1", "rel_l1"], "only support l2,l1,rel_l1"
        if unique_name is None:
            unique_name = func_name
        self.func_name = func_name
        self.unique_name = unique_name
        self.func = func
        self.func_propertys = func_propertys
        self.x_range = x_range
        self.num_points = num_points
        self.metric = metric.lower()
        self.left_edge = left_edge
        self.right_edge = right_edge

    def _process_cache(self, unique_name: str) -> List[float]:
        if unique_name in cut_points_dict_v2:
            return cut_points_dict_v2[unique_name]
        # if unique_name in cut_points_dict_v1:
        #     return cut_points_dict_v1[unique_name]
        return None

    def _generate_x_values(self):
        func = self.func
        min_val, max_val = self.x_range if self.x_range is not None else (-65504, 65504)
        num_samples = self.num_points if self.num_points is not None else None

        x = torch.tensor(range(65536), dtype=torch.uint16).view(torch.float16)
        mask = x.isnan() | x.isinf()
        x = x[~mask]
        x = x[(x <= max_val) & (x >= min_val)]
        if func is not None:
            y = func(x)
            mask = y.isnan() | y.isinf()
            x = x[~mask]
        x = x.sort()[0]
        if num_samples is not None:
            indices = torch.linspace(0, x.numel() - 1, num_samples).round_().int().unique()
            x = x[indices]
        x = torch.cat([torch.tensor([0], dtype=torch.float16), x], dim=0).unique().sort()[0].contiguous()
        return x

    def _generate_split_points_v2(self):
        x_values = self._generate_x_values()
        if self.func_propertys is None or len(self.func_propertys) == 0:
            func_arg1, func_arg2 = None, None
        elif len(self.func_propertys) == 1:
            func_arg1, func_arg2 = self.func_propertys[0], None
        else:
            func_arg1, func_arg2 = self.func_propertys
        return generate_cut_points_v2(
            self.func,
            x_values,
            func_name=self.func_name,
            left_edge=self.left_edge,
            right_edge=self.right_edge,
            max_search_points=self.num_points,
            func_arg1=func_arg1,
            func_arg2=func_arg2,
        )[1]

    def _generate_split_points_v1(self, x_values: torch.Tensor = None):
        return generate_cut_points_v1(
            self.func, self.func_name, torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def generate_split_points(self) -> List[float]:
        # 1. if cached, return cached
        cached_cut_points = self._process_cache(self.unique_name)
        if cached_cut_points is not None:
            return cached_cut_points

        # 2. if not cached, generate
        if if_can_v2(self.func_name):
            split_points = self._generate_split_points_v2()
            cut_points_dict_v2[self.unique_name] = split_points
        else:
            split_points = self._generate_split_points_v1()
            cut_points_dict_v1[self.unique_name] = split_points
        return split_points


def generate_lut_table_v2(split_config: Union[SplitConfig,Dict]):
    assert isinstance(split_config, (SplitConfig,Dict))
    if isinstance(split_config,Dict):
        split_config = SplitConfig(**split_config)
    split_points = SplitPointStrategy(split_config).generate_split_points()
    return NewTable(func=split_config.func,cut_points=split_points)

