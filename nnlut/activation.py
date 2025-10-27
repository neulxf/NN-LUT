import torch
from torch import nn


def lut_forward_fast(x, s, t, d):
    dtype = x.dtype
    x = x.to(s.dtype)
    cut_indices = (
        torch.bucketize(x, d, right=True).clamp(
            max=s.shape[0]
        )
        - 1
    ).clip(0)
    y = s[cut_indices] * x + t[cut_indices]
    return y.to(dtype)

class Nnlut(nn.Module):
    def __init__(self, d, s, t):
        super().__init__()
        self.d = nn.Parameter(d, requires_grad=False)
        self.s = nn.Parameter(s, requires_grad=False)
        self.t = nn.Parameter(t, requires_grad=False)

    def forward(self, x):
        return lut_forward_fast(x, self.s, self.t, self.d)


class NnlutSoftmax(nn.Module):
    def __init__(self, d, s, t, axis=-1):
        super().__init__()
        self.axis = axis
        self.op_class = "NnlutSoftmax"
        self.nnlut = Nnlut(d, s, t)

    def forward(self, x):
        dtype = x.dtype
        re_x = x - x.max(dim=self.axis, keepdim=True)[0]
        exp_out = self.nnlut(re_x)
        # exp_out[re_x < -400] = 0
        exp_sum = exp_out.sum(dim=self.axis, keepdim=True, dtype=torch.float16).float()
        y = exp_out / exp_sum
        return y.to(dtype)
