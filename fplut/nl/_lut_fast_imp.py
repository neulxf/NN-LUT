import torch, torch.nn as nn, torch.nn.functional as F
import triton, triton.language as tl

@triton.autotune(
    configs=[
        # triton.Config({"BLOCK_SIZE": 8}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
    ],
    key=["n_x"],
)
@triton.jit
def _kernel_lut_fast(
    x_ptr,
    o_ptr,
    cut_points_ptr,
    mul_ptr,
    table_ptr,
    pre_indices_ptr,  # New pointer for pre-computed start indices
    interval_lengths_ptr,  # New pointer for interval lengths
    n_cut_points: tl.constexpr,  # 11
    n_x: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized and vectorized lookup table kernel.
    This version removes the inner loop by first finding the interval for each x
    and then performing a single gather and interpolation.
    """
    # --- 1. Block and mask setup ---
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    x_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = x_offsets < n_x

    # --- 2. Load input data ---
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)

    # --- 3. Handle edge cases (out of bounds) ---
    # Load the first and last cut points and corresponding table values
    first_cut_point = tl.load(cut_points_ptr).to(tl.float32)
    last_cut_point = tl.load(cut_points_ptr + n_cut_points - 1).to(tl.float32)
    first_table_val = tl.load(table_ptr).to(tl.float32)
    # The last table value is at index 258 for the described structure
    last_table_val = tl.load(table_ptr + 258).to(tl.float32)

    # --- 4. Find the correct interval index without looping ---
    # This is the core optimization. We unroll the search for the 10 intervals.
    # We compare x against each cut point to find its interval index.
    # Note: `n_cut_points` is 11, so indices are 0-10. Intervals are 0-9.
    interval_idx = 0
    # Unrolled search from right to left is efficient
    interval_idx = tl.where((x >= tl.load(cut_points_ptr + 9)) & (interval_idx == 0), 9, interval_idx)
    interval_idx = tl.where((x >= tl.load(cut_points_ptr + 8)) & (interval_idx == 0), 8, interval_idx)
    interval_idx = tl.where((x >= tl.load(cut_points_ptr + 7)) & (interval_idx == 0), 7, interval_idx)
    interval_idx = tl.where((x >= tl.load(cut_points_ptr + 6)) & (interval_idx == 0), 6, interval_idx)
    interval_idx = tl.where((x >= tl.load(cut_points_ptr + 5)) & (interval_idx == 0), 5, interval_idx)
    interval_idx = tl.where((x >= tl.load(cut_points_ptr + 4)) & (interval_idx == 0), 4, interval_idx)
    interval_idx = tl.where((x >= tl.load(cut_points_ptr + 3)) & (interval_idx == 0), 3, interval_idx)
    interval_idx = tl.where((x >= tl.load(cut_points_ptr + 2)) & (interval_idx == 0), 2, interval_idx)
    interval_idx = tl.where((x >= tl.load(cut_points_ptr + 1)) & (interval_idx == 0), 1, interval_idx)
    # Final check for the first interval is implicitly handled by initialization to 0

    # --- 5. Gather data for the identified interval ---
    # Load the start of the interval, the multiplier, and the table offset
    # using the calculated interval_idx. This is a "gather" operation.
    in_bounds_mask = (x >= first_cut_point) & (x < last_cut_point) & mask

    cut_points_i = tl.load(cut_points_ptr + interval_idx, mask=in_bounds_mask).to(tl.float32)
    mul_values_i = tl.load(mul_ptr + interval_idx, mask=in_bounds_mask).to(tl.float32)
    pre_idx = tl.load(pre_indices_ptr + interval_idx, mask=in_bounds_mask).to(tl.int16)
    interval_len = tl.load(interval_lengths_ptr + interval_idx, mask=in_bounds_mask).to(tl.int16)

    # --- 6. Perform linear interpolation ---
    # This logic is the same as before, but now runs only once per element.
    minus_start = x - cut_points_i
    position = (minus_start * mul_values_i).to(tl.float32)

    # Clamp idxs to be within the valid range for the sub-table
    position = position.to(tl.float32)
    m1 = (position - tl.floor(position)).to(tl.float32)

    idxs = tl.floor(position).to(tl.int16)
    idxs = tl.maximum(0, idxs)
    idxs = tl.minimum(idxs, interval_len).to(tl.int16)
    idxs_plus = tl.minimum(idxs + 1, interval_len).to(tl.int16)

    # Gather the two table values for interpolation
    v1 = tl.load(table_ptr + pre_idx + idxs, mask=in_bounds_mask, other=0.0).to(tl.float32)
    v2 = tl.load(table_ptr + pre_idx + idxs_plus, mask=in_bounds_mask, other=0.0).to(tl.float32)

    # Calculate the interpolated output
    # m2 = 1.0 - m1
    # output = (m1 * v2 + m2 * v1).to(tl.float16)
    right_sub_left = v2 - v1
    output = (right_sub_left.to(tl.float32) * m1.to(tl.float32) + v1.to(tl.float32)).to(tl.float32)

    # --- 7. Combine results and store ---
    # Start with the interpolated values for in-bounds elements
    # Apply the out-of-bounds values using tl.where
    output = tl.where(x <= first_cut_point, first_table_val, output)
    output = tl.where(x >= last_cut_point, last_table_val, output)

    if False:
        tl.device_print("m1", m1 * 1024)
        tl.device_print("position", position * 1024)
        tl.device_print("idxs", idxs)
        tl.device_print("idxs_plus", idxs_plus)
        tl.device_print("m1", m1 * 1024)
        tl.device_print("v1", v1 * 1024)
        tl.device_print("v2", v2 * 1024)
        tl.device_print("right_left", right_sub_left * 1024)

    # Store the final result
    tl.store(o_ptr + x_offsets, output, mask=mask)


def lut_fast_triton(x, cut_points, values, scales):
    """
    Optimized Python wrapper for the fast lookup table using Triton.
    """
    torch.cuda.set_device(x.device) 
    assert cut_points.dim() == 1 and cut_points.size(0) == 11, "cut_points must be 1D and have 11 elements"
    assert values.dim() == 1 and values.size(0) == 259, "table_values must be 1D and have 259 elements"
    # --- Pre-computation on Host ---
    # Based on the original kernel's logic: 0, 1, 1+32, 1+32*2, ..., 257
    pre_indices = torch.tensor(
        [0] + [1 + 32 * (i - 1) for i in range(1, 9)] + [257], device=x.device, dtype=torch.int32
    )
    # Based on the original kernel's logic: 1 for ends, 32 for middle.
    interval_lengths = torch.tensor([1] + [32] * 8 + [1], device=x.device, dtype=torch.int32)
    # Prepare tensors
    x = x.contiguous().float()
    output = torch.empty_like(x)
    n_x = x.numel()
    grid = lambda meta: (triton.cdiv(n_x, meta["BLOCK_SIZE"]),)
    # Launch the optimized kernel
    _kernel_lut_fast[grid](
        x_ptr=x,
        o_ptr=output,
        cut_points_ptr=cut_points,
        mul_ptr=scales,
        table_ptr=values,
        pre_indices_ptr=pre_indices,
        interval_lengths_ptr=interval_lengths,
        n_cut_points=cut_points.size(0),
        n_x=n_x,
    )
    return output


def lut_fast_torch(x: torch.Tensor, cut_points: torch.Tensor, values: torch.Tensor, scales: torch.Tensor):
    # assert (
    #     (x.dtype == torch.float16)
    #     and (cut_points.dtype == torch.float16)
    #     and (values.dtype == torch.float16)
    #     and (scales.dtype == torch.float16)
    # ), "x, cut_points, table_values, mul_scales must be float16"
    pre_indices = torch.tensor(
        [0] + [1 + 32 * (i - 1) for i in range(1, 9)] + [257], device=x.device, dtype=torch.int16
    )
    interval_lengths = torch.tensor([1] + [32] * 8 + [1], device=x.device, dtype=torch.int16)
    # x = x.clip(min=cut_points[0],max=cut_points[-1])
    # right means [,)
    cut_indices = torch.bucketize(x, cut_points, right=True).sub_(1).clip_(0, scales.numel() - 1)
    pre_indices = pre_indices[cut_indices]

    interval_lengths = interval_lengths[cut_indices]
    idxs_f = (x - cut_points[cut_indices]).mul_(scales[cut_indices])
    idxs = (
        (idxs_f.floor())
        .clip_(
            0,
        )
        .clamp_max_(interval_lengths)
        .to(torch.int16)
    )
    idxs_plus = (idxs + 1).clip_(0).clamp_max_(interval_lengths).to(torch.int16)

    m1 = (idxs_f - idxs).clip_(0, 1).to(torch.float32)

    idxs = idxs + pre_indices
    idxs_plus = idxs_plus + pre_indices

    y1 = values[idxs.int()]
    y2 = values[idxs_plus.int()]
    right_left = y2 - y1
    y = (right_left.float() * m1 + y1.to(torch.float32))
    y[x <= cut_points[0]] = values[0]
    y[x >= cut_points[-1]] = values[-1]
    return y


def _generate_fp16_values(func=None, num_samples=None, min_val=-65504, max_val=65504):  # 不包括正负0
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


def lut_fast(x, cut_points, values, scales):
    if x.is_cuda:
        return lut_fast_triton(x, cut_points, values, scales)
    else:
        return lut_fast_torch(x, cut_points, values, scales)


# Example usage and test function
def test_lut_fast():
    """Test the lookup table implementation"""
    import torch
    from fplut.base.xh2_fp_lut import NewTable, cut_points_dict
    from fplut.nl.base import verbose_lut

    torch.manual_seed(0)

    table = NewTable(F.silu, cut_points_dict["silu"], device="cuda")
    cut_points, table_values, scale = verbose_lut(table)

    # Create test data
    x = torch.randn(100000, device="cuda", dtype=torch.float16)
    x[0] = -65504
    x[1] = 65504
    x = torch.tensor([-2.64453125], dtype=torch.float16, device="cuda")
    x = _generate_fp16_values(F.silu).to(x.device)
    y0 = F.silu(x)
    # y1 = table(x, use_gpu_lut=True)
    x = x.float()

    # Run Triton implementations
    output_triton = lut_fast_triton(
        x,
        torch.tensor(cut_points, device="cuda", dtype=torch.float32),
        torch.tensor(table_values, device="cuda", dtype=torch.float32),
        torch.tensor(scale, device="cuda", dtype=torch.float32),
    )

    output_torch = lut_fast_torch(
        x,
        torch.tensor(cut_points, device="cuda", dtype=torch.float32),
        torch.tensor(table_values, device="cuda", dtype=torch.float32),
        torch.tensor(scale, device="cuda", dtype=torch.float32),
    )

    print(torch.allclose(output_triton, output_torch, atol=1e-4))


if __name__ == "__main__":
    test_lut_fast()
