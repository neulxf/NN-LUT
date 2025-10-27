
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..base.xh2_fp_lut import HXTable, NewTable, cut_points_dict
from ._lut_fast_imp import lut_fast

def verbose_lut(table: NewTable) -> Tuple[List[float], List[float], List[float]]:
    cut_points = table.cut_points.cpu().detach().numpy()
    values = table.table.cpu().detach().numpy().tolist()
    assert len(values) == 259
    scale = table.mul_scale.cpu().detach().numpy()
    return cut_points.tolist(), values, scale.tolist()

def lut_v2_fast(
    x: torch.Tensor,
    lut_cut_points: Sequence[float],
    lut_table: Sequence[float],
    lut_scale: Sequence[float],
) -> Tensor:
    cut_points = torch.tensor(lut_cut_points, device=x.device, dtype=torch.float32)
    values = torch.tensor(lut_table, device=x.device, dtype=torch.float32)
    scales = torch.tensor(lut_scale, device=x.device, dtype=torch.float32)
    return lut_fast(x, cut_points, values, scales)
