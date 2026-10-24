"""MPS overlays for 3-D adaptive pooling.

torch 2.14.1 has no MPS kernel for any of these four operators, so every call
raises NotImplementedError on an MPS tensor:

    _adaptive_avg_pool3d            adaptive_max_pool3d
    _adaptive_avg_pool3d_backward   adaptive_max_pool3d_backward

Semantics match the CPU kernel exactly:

* bins are ATen's rule, unchanged for nondivisible sizes --
  ``start_i = floor(i * in / out)``, ``end_i = ceil((i + 1) * in / out)``;
* indices are the flat index into the input's three spatial dims,
  ``d * H * W + h * W + w``;
* on a tie the first maximum in (d, h, w) scan order wins, and a bin holding a
  NaN returns NaN with the *last* NaN in that order (ATen's
  ``val > max || isnan(val)`` update rule);
* half and bfloat16 are summed in float32 and rounded once, so the value is
  within one rounding of the exact mean. The CPU average of those dtypes
  accumulates in the narrow type, which drifts away from the exact mean as the
  bin widens: measured 1.4e-2 relative for half on 40^3 bins and up to 7e-1 for
  bfloat16 on integer-valued 34x13x30 bins, where this package stays at 2e-3.
  Half results therefore agree with CPU to CPU's own error, not bit for bit,
  and this package is never the less accurate of the two;
* in float32 the fused kernels sum in the CPU kernel's order and match it bit
  for bit; the decomposition sums per axis, which differs from CPU by up to a
  few 1e-7 relative on wide bins;
* a NaN reaches only the bins that contain it, in the value and in the
  gradient.

Metal kernels (``torch.mps.compile_shader``, compiled on first use) carry
every shape a model reaches; the aten decomposition is the fallback for
outputs far wider than the input, where one input element sits under more than
``_FANIN_LIMIT`` bins:

* the average forward runs one thread per output bin while a box has fewer
  than ``_AVG_NARROW_LIMIT`` elements, and one threadgroup of cooperating
  lanes per bin above that -- lanes split the box into lines and reduce in
  threadgroup memory, with several bins' worth of partials added by a second
  pass when the bin count alone cannot fill the device;
* the average backward runs one thread per four consecutive inputs, walking
  the bins that cover them. Bin bounds and reciprocal bin volumes come from a
  per-shape table, so the inner loop holds no integer division;
* max pooling keeps one thread per output bin while the box is under
  ``_WINDOW_LIMIT`` elements.

Gradients do not come from these bodies: the autograd key of each operator has
its own formula, which calls the matching ``*_backward`` operator, and that
call lands on the overlay registered for it.

Importing this module registers nothing and compiles nothing. ``OVERLAYS`` is
data for the caller: ``Library("aten", "FRAGMENT").impl(name, fn, "MPS")``.
"""

from functools import lru_cache

import torch

# Largest box (elements in one bin) the single-thread-per-bin walks cover: the
# average above this hands a bin to a whole threadgroup, which keeps 64^3
# pooling at or below 0.05 ms where the single-thread walk takes 0.06 ms at a
# box of 512 and 18 ms at 274625 (measured/adaptive_pool_ops.json,
# kernel_crossing). Max pooling keeps the single-thread walk, whose limit is
# where the decomposition overtakes it.
_AVG_NARROW_LIMIT = 512
_WINDOW_LIMIT = 1000
# Largest fan-in (bins covering one input element) the fused backward walks
# before the decomposition spreads the work instead.
_FANIN_LIMIT = 8**3
_FLAT_LIMIT = 2**31

# Ceiling on lanes per threadgroup, lanes across a box line and groups per bin
# on the wide average path, with the elements per lane that decide both: fewer
# lanes than this spend more on the reduction than on the sum, more groups than
# this leave every lane with too little to do.
_WIDE_GROUP = 256
_WIDE_LANES = 32
_WIDE_TREAT = 8
_WIDE_SLICES = 512

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

kernel void max_pool_f32(
    device const float* x, device float* values, device int* indices,
    constant uint& rows, constant uint& D, constant uint& H, constant uint& W,
    constant uint& OD, constant uint& OH, constant uint& OW,
    uint gid [[thread_position_in_grid]]) {
    if (gid >= rows * OD * OH * OW) return;
    uint ow = gid % OW;
    uint t = gid / OW;
    uint oh = t % OH;
    t /= OH;
    uint od = t % OD;
    uint row = t / OD;
    uint d0 = (od * D) / OD, d1 = ((od + 1) * D + OD - 1) / OD;
    uint h0 = (oh * H) / OH, h1 = ((oh + 1) * H + OH - 1) / OH;
    uint w0 = (ow * W) / OW, w1 = ((ow + 1) * W + OW - 1) / OW;
    device const float* base = x + (size_t)row * D * H * W;
    float best = -INFINITY;
    uint at = (d0 * H + h0) * W + w0;
    for (uint d = d0; d < d1; ++d) {
        device const float* plane = base + (size_t)d * H * W;
        for (uint h = h0; h < h1; ++h) {
            device const float* line = plane + (size_t)h * W;
            for (uint w = w0; w < w1; ++w) {
                float v = line[w];
                if (v > best || v != v) {
                    best = v;
                    at = (d * H + h) * W + w;
                }
            }
        }
    }
    values[gid] = best;
    indices[gid] = int(at);
}

kernel void max_pool_f16(
    device const half* x, device half* values, device int* indices,
    constant uint& rows, constant uint& D, constant uint& H, constant uint& W,
    constant uint& OD, constant uint& OH, constant uint& OW,
    uint gid [[thread_position_in_grid]]) {
    if (gid >= rows * OD * OH * OW) return;
    uint ow = gid % OW;
    uint t = gid / OW;
    uint oh = t % OH;
    t /= OH;
    uint od = t % OD;
    uint row = t / OD;
    uint d0 = (od * D) / OD, d1 = ((od + 1) * D + OD - 1) / OD;
    uint h0 = (oh * H) / OH, h1 = ((oh + 1) * H + OH - 1) / OH;
    uint w0 = (ow * W) / OW, w1 = ((ow + 1) * W + OW - 1) / OW;
    device const half* base = x + (size_t)row * D * H * W;
    float best = -INFINITY;
    uint at = (d0 * H + h0) * W + w0;
    for (uint d = d0; d < d1; ++d) {
        device const half* plane = base + (size_t)d * H * W;
        for (uint h = h0; h < h1; ++h) {
            device const half* line = plane + (size_t)h * W;
            for (uint w = w0; w < w1; ++w) {
                float v = float(line[w]);
                if (v > best || v != v) {
                    best = v;
                    at = (d * H + h) * W + w;
                }
            }
        }
    }
    values[gid] = (half)best;
    indices[gid] = int(at);
}

kernel void avg_pool_f32(
    device const float* x, device float* y,
    constant uint& rows, constant uint& D, constant uint& H, constant uint& W,
    constant uint& OD, constant uint& OH, constant uint& OW,
    uint gid [[thread_position_in_grid]]) {
    if (gid >= rows * OD * OH * OW) return;
    uint ow = gid % OW;
    uint t = gid / OW;
    uint oh = t % OH;
    t /= OH;
    uint od = t % OD;
    uint row = t / OD;
    uint d0 = (od * D) / OD, d1 = ((od + 1) * D + OD - 1) / OD;
    uint h0 = (oh * H) / OH, h1 = ((oh + 1) * H + OH - 1) / OH;
    uint w0 = (ow * W) / OW, w1 = ((ow + 1) * W + OW - 1) / OW;
    device const float* base = x + (size_t)row * D * H * W;
    float sum = 0.0f;
    for (uint d = d0; d < d1; ++d) {
        device const float* plane = base + (size_t)d * H * W;
        for (uint h = h0; h < h1; ++h) {
            device const float* line = plane + (size_t)h * W;
            for (uint w = w0; w < w1; ++w) sum += line[w];
        }
    }
    y[gid] = sum / float((d1 - d0) * (h1 - h0) * (w1 - w0));
}

kernel void avg_pool_f16(
    device const half* x, device half* y,
    constant uint& rows, constant uint& D, constant uint& H, constant uint& W,
    constant uint& OD, constant uint& OH, constant uint& OW,
    uint gid [[thread_position_in_grid]]) {
    if (gid >= rows * OD * OH * OW) return;
    uint ow = gid % OW;
    uint t = gid / OW;
    uint oh = t % OH;
    t /= OH;
    uint od = t % OD;
    uint row = t / OD;
    uint d0 = (od * D) / OD, d1 = ((od + 1) * D + OD - 1) / OD;
    uint h0 = (oh * H) / OH, h1 = ((oh + 1) * H + OH - 1) / OH;
    uint w0 = (ow * W) / OW, w1 = ((ow + 1) * W + OW - 1) / OW;
    device const half* base = x + (size_t)row * D * H * W;
    float sum = 0.0f;
    for (uint d = d0; d < d1; ++d) {
        device const half* plane = base + (size_t)d * H * W;
        for (uint h = h0; h < h1; ++h) {
            device const half* line = plane + (size_t)h * W;
            for (uint w = w0; w < w1; ++w) sum += float(line[w]);
        }
    }
    y[gid] = (half)(sum / float((d1 - d0) * (h1 - h0) * (w1 - w0)));
}

// One thread per four consecutive inputs on the last axis. The covering-bin
// bounds and the reciprocal bin volumes are read from a per-shape table, so
// the inner loop holds no integer division.
kernel void avg_pool_backward_f32(
    device const float* g, device float4* dx, device const int* bounds,
    device const float* recip, constant uint& rows, constant uint& D,
    constant uint& H, constant uint& W, constant uint& OD,
    constant uint& OH, constant uint& OW,
    uint gid [[thread_position_in_grid]]) {
    uint W4 = W >> 2;
    if (gid >= rows * D * H * W4) return;
    uint w4 = gid % W4;
    uint t = gid / W4;
    uint h = t % H;
    t /= H;
    uint d = t % D;
    uint row = t / D;
    uint w = w4 << 2;
    int d0 = bounds[d], d1 = bounds[D + d];
    int h0 = bounds[2 * D + h], h1 = bounds[2 * D + H + h];
    device const int* low = bounds + 2 * D + 2 * H + w;
    device const int* high = low + W;
    int lo = min(min(low[0], low[1]), min(low[2], low[3]));
    int hi = max(max(high[0], high[1]), max(high[2], high[3]));
    device const float* base = g + (size_t)row * OD * OH * OW;
    float4 acc = 0.0f;
    for (int od = d0; od <= d1; ++od) {
        float id = recip[od];
        device const float* plane = base + (size_t)od * OH * OW;
        for (int oh = h0; oh <= h1; ++oh) {
            float idh = id * recip[OD + oh];
            device const float* line = plane + (size_t)oh * OW;
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            for (int ow = lo; ow <= hi; ++ow) {
                float v = line[ow] * (idh * recip[OD + OH + ow]);
                s0 += (ow >= low[0] && ow <= high[0]) ? v : 0.0f;
                s1 += (ow >= low[1] && ow <= high[1]) ? v : 0.0f;
                s2 += (ow >= low[2] && ow <= high[2]) ? v : 0.0f;
                s3 += (ow >= low[3] && ow <= high[3]) ? v : 0.0f;
            }
            acc.x += s0; acc.y += s1; acc.z += s2; acc.w += s3;
        }
    }
    dx[gid] = acc;
}

kernel void avg_pool_backward_f16(
    device const half* g, device half4* dx, device const int* bounds,
    device const float* recip, constant uint& rows, constant uint& D,
    constant uint& H, constant uint& W, constant uint& OD,
    constant uint& OH, constant uint& OW,
    uint gid [[thread_position_in_grid]]) {
    uint W4 = W >> 2;
    if (gid >= rows * D * H * W4) return;
    uint w4 = gid % W4;
    uint t = gid / W4;
    uint h = t % H;
    t /= H;
    uint d = t % D;
    uint row = t / D;
    uint w = w4 << 2;
    int d0 = bounds[d], d1 = bounds[D + d];
    int h0 = bounds[2 * D + h], h1 = bounds[2 * D + H + h];
    device const int* low = bounds + 2 * D + 2 * H + w;
    device const int* high = low + W;
    int lo = min(min(low[0], low[1]), min(low[2], low[3]));
    int hi = max(max(high[0], high[1]), max(high[2], high[3]));
    device const half* base = g + (size_t)row * OD * OH * OW;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (int od = d0; od <= d1; ++od) {
        float id = recip[od];
        device const half* plane = base + (size_t)od * OH * OW;
        for (int oh = h0; oh <= h1; ++oh) {
            float idh = id * recip[OD + oh];
            device const half* line = plane + (size_t)oh * OW;
            for (int ow = lo; ow <= hi; ++ow) {
                float v = float(line[ow]) * (idh * recip[OD + OH + ow]);
                s0 += (ow >= low[0] && ow <= high[0]) ? v : 0.0f;
                s1 += (ow >= low[1] && ow <= high[1]) ? v : 0.0f;
                s2 += (ow >= low[2] && ow <= high[2]) ? v : 0.0f;
                s3 += (ow >= low[3] && ow <= high[3]) ? v : 0.0f;
            }
        }
    }
    dx[gid] = half4((half)s0, (half)s1, (half)s2, (half)s3);
}

// Last axis not a multiple of four: one thread per input. Same tables, same
// walk; only the four-lane grouping is gone.
kernel void avg_pool_backward_scalar_f32(
    device const float* g, device float* dx, device const int* bounds,
    device const float* recip, constant uint& rows, constant uint& D,
    constant uint& H, constant uint& W, constant uint& OD,
    constant uint& OH, constant uint& OW,
    uint gid [[thread_position_in_grid]]) {
    if (gid >= rows * D * H * W) return;
    uint w = gid % W;
    uint t = gid / W;
    uint h = t % H;
    t /= H;
    uint d = t % D;
    uint row = t / D;
    int d0 = bounds[d], d1 = bounds[D + d];
    int h0 = bounds[2 * D + h], h1 = bounds[2 * D + H + h];
    int w0 = bounds[2 * D + 2 * H + w], w1 = bounds[2 * D + 2 * H + W + w];
    device const float* base = g + (size_t)row * OD * OH * OW;
    float acc = 0.0f;
    for (int od = d0; od <= d1; ++od) {
        float id = recip[od];
        device const float* plane = base + (size_t)od * OH * OW;
        for (int oh = h0; oh <= h1; ++oh) {
            float idh = id * recip[OD + oh];
            device const float* line = plane + (size_t)oh * OW;
            for (int ow = w0; ow <= w1; ++ow) {
                acc += line[ow] * (idh * recip[OD + OH + ow]);
            }
        }
    }
    dx[gid] = acc;
}

kernel void avg_pool_backward_scalar_f16(
    device const half* g, device half* dx, device const int* bounds,
    device const float* recip, constant uint& rows, constant uint& D,
    constant uint& H, constant uint& W, constant uint& OD,
    constant uint& OH, constant uint& OW,
    uint gid [[thread_position_in_grid]]) {
    if (gid >= rows * D * H * W) return;
    uint w = gid % W;
    uint t = gid / W;
    uint h = t % H;
    t /= H;
    uint d = t % D;
    uint row = t / D;
    int d0 = bounds[d], d1 = bounds[D + d];
    int h0 = bounds[2 * D + h], h1 = bounds[2 * D + H + h];
    int w0 = bounds[2 * D + 2 * H + w], w1 = bounds[2 * D + 2 * H + W + w];
    device const half* base = g + (size_t)row * OD * OH * OW;
    float acc = 0.0f;
    for (int od = d0; od <= d1; ++od) {
        float id = recip[od];
        device const half* plane = base + (size_t)od * OH * OW;
        for (int oh = h0; oh <= h1; ++oh) {
            float idh = id * recip[OD + oh];
            device const half* line = plane + (size_t)oh * OW;
            for (int ow = w0; ow <= w1; ++ow) {
                acc += float(line[ow]) * (idh * recip[OD + OH + ow]);
            }
        }
    }
    dx[gid] = (half)acc;
}

// Wide bins: one threadgroup per bin, lanes splitting the box into lines that
// end at the same last-axis offset, so every lane reads the same four-byte
// alignment. A second kernel adds the partials when several threadgroups
// share a bin.
kernel void avg_pool_wide_f32(
    device const float* x, device float* y, device float* partial,
    constant uint& rows, constant uint& D, constant uint& H, constant uint& W,
    constant uint& OD, constant uint& OH, constant uint& OW,
    constant uint& slices, constant uint& lanes,
    uint tid [[thread_position_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],
    uint width [[threads_per_threadgroup]]) {
    uint out_id = slices == 1 ? group : group / slices;
    uint slice = slices == 1 ? 0 : group - out_id * slices;
    uint ow = out_id % OW;
    uint t = out_id / OW;
    uint oh = t % OH;
    t /= OH;
    uint od = t % OD;
    uint row = t / OD;
    uint d0 = (od * D) / OD, d1 = ((od + 1) * D + OD - 1) / OD;
    uint h0 = (oh * H) / OH, h1 = ((oh + 1) * H + OH - 1) / OH;
    uint w0 = (ow * W) / OW, w1 = ((ow + 1) * W + OW - 1) / OW;
    uint rd = d1 - d0, rh = h1 - h0, rw = w1 - w0;
    uint lines = rd * rh;
    uint per = (lines + slices - 1) / slices;
    uint first = min(slice * per, lines);
    uint last = min(first + per, lines);
    // Lane ``tid % lanes`` takes element ``w0 + tid % lanes`` of its line: with
    // ``slices == 1`` the first lane lands on ``w0``, a multiple of four for an
    // aligned base, so every lane reads at the same alignment offset.
    uint lane = tid & (lanes - 1);
    uint crew = width >> __builtin_ctz(lanes);
    device const float* base = x + (size_t)row * D * H * W;
    float acc = 0.0f;
    for (uint r = first + (tid >> __builtin_ctz(lanes)); r < last; r += crew) {
        uint hi = r % rh;
        uint di = r / rh;
        device const float* line = base + (size_t)(d0 + di) * H * W
            + (size_t)(h0 + hi) * W + w0;
        for (uint i = lane; i < rw; i += lanes) acc += line[i];
    }
    threadgroup float scratch[1024];
    scratch[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = width >> 1; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float v = scratch[0] / float(rd * rh * rw);
        if (slices == 1) y[out_id] = v;
        else partial[out_id * slices + slice] = v;
    }
}

kernel void avg_pool_wide_f16(
    device const half* x, device half* y, device half* partial,
    constant uint& rows, constant uint& D, constant uint& H, constant uint& W,
    constant uint& OD, constant uint& OH, constant uint& OW,
    constant uint& slices, constant uint& lanes,
    uint tid [[thread_position_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],
    uint width [[threads_per_threadgroup]]) {
    uint out_id = slices == 1 ? group : group / slices;
    uint slice = slices == 1 ? 0 : group - out_id * slices;
    uint ow = out_id % OW;
    uint t = out_id / OW;
    uint oh = t % OH;
    t /= OH;
    uint od = t % OD;
    uint row = t / OD;
    uint d0 = (od * D) / OD, d1 = ((od + 1) * D + OD - 1) / OD;
    uint h0 = (oh * H) / OH, h1 = ((oh + 1) * H + OH - 1) / OH;
    uint w0 = (ow * W) / OW, w1 = ((ow + 1) * W + OW - 1) / OW;
    uint rd = d1 - d0, rh = h1 - h0, rw = w1 - w0;
    uint lines = rd * rh;
    uint per = (lines + slices - 1) / slices;
    uint first = min(slice * per, lines);
    uint last = min(first + per, lines);
    uint lane = tid & (lanes - 1);
    uint crew = width >> __builtin_ctz(lanes);
    device const half* base = x + (size_t)row * D * H * W;
    float acc = 0.0f;
    for (uint r = first + (tid >> __builtin_ctz(lanes)); r < last; r += crew) {
        uint hi = r % rh;
        uint di = r / rh;
        device const half* line = base + (size_t)(d0 + di) * H * W
            + (size_t)(h0 + hi) * W + w0;
        for (uint i = lane; i < rw; i += lanes) acc += float(line[i]);
    }

    threadgroup float scratch[1024];
    scratch[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = width >> 1; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float v = scratch[0] / float(rd * rh * rw);
        if (slices == 1) y[out_id] = (half)v;
        else partial[out_id * slices + slice] = (half)v;
    }
}

kernel void avg_pool_wide_add_f32(
    device const float* partial, device float* y, constant uint& slices,
    uint gid [[thread_position_in_grid]]) {
    float acc = 0.0f;
    for (uint i = 0; i < slices; ++i) acc += partial[gid * slices + i];
    y[gid] = acc;
}

kernel void avg_pool_wide_add_f16(
    device const half* partial, device half* y, constant uint& slices,
    uint gid [[thread_position_in_grid]]) {
    float acc = 0.0f;
    for (uint i = 0; i < slices; ++i) acc += float(partial[gid * slices + i]);
    y[gid] = (half)acc;
}
"""

_PLANS = {}
_PLAN_LIMIT = 128
_SENTINEL = 2**31 - 1


@lru_cache(maxsize=1)
def _library():
    """Compiled on first use; importing this module compiles nothing."""
    return torch.mps.compile_shader(_SOURCE)


def _bins(in_size, out_size, device):
    index = torch.arange(out_size, device=device)
    start = index * in_size // out_size
    end = ((index + 1) * in_size + out_size - 1) // out_size
    return start, end


def _axis(in_size, out_size, device):
    start, end = _bins(in_size, out_size, device)
    width = int((end - start).max())
    offsets = torch.arange(width, device=device)
    window = (start[:, None] + offsets[None, :]).clamp_max((end - 1)[:, None])
    valid = (start[:, None] + offsets[None, :]) < end[:, None]
    return start, end, window, valid


def _cover(in_size, out_size, device):
    """For each input position, the bins that contain it, padded with valid."""
    position = torch.arange(in_size, device=device)
    low = position * out_size // in_size
    high = ((position + 1) * out_size + in_size - 1) // in_size - 1
    width = int((high - low).max()) + 1
    offsets = torch.arange(width, device=device)
    index = (low[:, None] + offsets[None, :]).clamp_max(high[:, None])
    valid = (low[:, None] + offsets[None, :]) <= high[:, None]
    return index, valid


def _plan(in_shape, out_shape, device):
    key = (str(device), tuple(in_shape), tuple(out_shape))
    entry = _PLANS.get(key)
    if entry is None:
        if len(_PLANS) >= _PLAN_LIMIT:
            _PLANS.clear()
        entry = (
            [_axis(i, o, device) for i, o in zip(in_shape, out_shape)],
            [_cover(i, o, device) for i, o in zip(in_shape, out_shape)],
            _covers(in_shape, out_shape, device),
        )
        _PLANS[key] = entry
    return entry


def _covers(in_shape, out_shape, device):
    """Bounds and reciprocal bin volumes the fused backward reads.

    ``bounds`` holds, for every input position of each axis, the first and last
    output bin covering it; ``recip`` the reciprocal volume of each output bin,
    flattened over the three axes. Both are summed over the axes the gradient
    reads, which is what keeps divisions out of the kernel's inner loop. The
    bound of each axis is a pair -- lower, then upper -- and the shader walks
    the six blocks by the axis sizes it is given.
    """
    lower, upper, recip = [], [], []
    for i, o in zip(in_shape, out_shape):
        position = torch.arange(i, device=device)
        lower += [
            (position * o // i).to(torch.int32),
            (((position + 1) * o + i - 1) // i - 1).to(torch.int32),
        ]
        index = torch.arange(o, device=device)
        count = ((index + 1) * i + o - 1) // o - index * i // o
        recip.append(torch.reciprocal(count.to(torch.float32)))
    return torch.cat(lower).contiguous(), torch.cat(recip).contiguous()


def _window_volume(in_shape, out_shape):
    """Upper bound on the elements in one bin."""
    volume = 1
    for i, o in zip(in_shape, out_shape):
        volume *= (i + o - 1) // o + 1
    return volume


def _fan_in(in_shape, out_shape):
    """Upper bound on the bins that can cover one input element."""
    volume = 1
    for i, o in zip(in_shape, out_shape):
        volume *= (o + i - 1) // i + 1
    return volume


def _flat_ok(in_shape):
    volume = 1
    for size in in_shape:
        volume *= size
    return volume < _FLAT_LIMIT


def _check(out_shape):
    for size in out_shape:
        if size <= 0:
            raise RuntimeError(
                "adaptive_pool3d(): output size must be greater than zero, "
                f"got {tuple(out_shape)}"
            )


def _check_spatial(self):
    if any(size == 0 for size in self.shape[-3:]):
        raise RuntimeError(
            "adaptive_pool3d(): Expected input to have non-zero size for non-batch "
            f"dimensions, but got {tuple(self.shape)}"
        )


def _fused_ready(tensor, in_shape, out_shape, measure, bound):
    return (
        tensor.device.type == "mps"
        and tensor.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and _flat_ok(in_shape)
        and measure(in_shape, out_shape) <= bound
    )


def _working(tensor):
    """float32 for bfloat16, which has no pointer in the shaders."""
    if tensor.dtype == torch.bfloat16:
        return tensor.float()
    return tensor


def _contiguous(tensor):
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def _fused_max(x, out_shape):
    work = _contiguous(_working(x))
    shape = tuple(x.shape[:-3]) + tuple(out_shape)
    values = torch.empty(shape, dtype=work.dtype, device=work.device)
    indices = torch.empty(shape, dtype=torch.int32, device=work.device)
    rows, D, H, W = _dims(work)
    total = values.numel()
    name = "max_pool_f32" if work.dtype == torch.float32 else "max_pool_f16"
    getattr(_library(), name)(
        work, values, indices, rows, D, H, W, *out_shape,
        threads=(total + 31) // 32 * 32, group_size=32,
    )
    return values.to(x.dtype), indices.to(torch.int64)


def _fused_avg(x, out_shape):
    work = _contiguous(_working(x))
    shape = tuple(x.shape[:-3]) + tuple(out_shape)
    out = torch.empty(shape, dtype=work.dtype, device=work.device)
    rows, D, H, W = _dims(work)
    name = "avg_pool_f32" if work.dtype == torch.float32 else "avg_pool_f16"
    getattr(_library(), name)(
        work, out, rows, D, H, W, *out_shape,
        threads=(out.numel() + 31) // 32 * 32, group_size=32,
    )
    return out.to(x.dtype)


def _wide_shape(in_shape, out_shape):
    """Lanes across a line, threadgroup width and groups per bin.

    The width follows the box: ``_WIDE_TREAT`` elements per lane is the point
    where the threadgroup reduction stops paying for itself, and the group is
    capped so a small box does not launch a thread per element. A bin wider
    than that still keeps ``_WIDE_TREAT`` elements per lane by giving the bin
    several threadgroups.
    """
    # A bin's last axis holds at most ceil(in / out) + 1 elements; lanes never
    # exceed that, so no lane of the group idles inside the line loop.
    line = max((i + o - 1) // o + 1 for i, o in zip(in_shape, out_shape))
    lanes = 1 << (min(_WIDE_LANES, line).bit_length() - 1)
    box = 1
    for i, o in zip(in_shape, out_shape):
        box *= (i + o - 1) // o
    group = 1 << (max(32, min(_WIDE_GROUP, box // _WIDE_TREAT)).bit_length() - 1)
    group = max(group, lanes)
    return lanes, group, max(1, min(_WIDE_SLICES, box // (group * _WIDE_TREAT)))


def _fused_avg_wide(x, out_shape):
    work = _contiguous(_working(x))
    shape = tuple(x.shape[:-3]) + tuple(out_shape)
    out = torch.empty(shape, dtype=work.dtype, device=work.device)
    rows, D, H, W = _dims(work)
    lanes, group, slices = _wide_shape((D, H, W), out_shape)
    partial = (
        torch.empty(out.numel() * slices, dtype=work.dtype, device=work.device)
        if slices > 1
        else out
    )
    stem = "avg_pool_wide_f32" if work.dtype == torch.float32 else "avg_pool_wide_f16"
    getattr(_library(), stem)(
        work, out, partial, rows, D, H, W, *out_shape, slices, lanes,
        threads=out.numel() * slices * group, group_size=group,
    )
    if slices > 1:
        add = "avg_pool_wide_add_f32" if work.dtype == torch.float32 else "avg_pool_wide_add_f16"
        getattr(_library(), add)(
            partial, out, slices,
            threads=(out.numel() + 31) // 32 * 32, group_size=32,
        )
    return out.to(x.dtype)


def _fused_avg_backward(grad, in_shape):
    work = _contiguous(_working(grad))
    out_shape = tuple(grad.shape[-3:])
    out = torch.empty(
        tuple(grad.shape[:-3]) + tuple(in_shape),
        dtype=work.dtype,
        device=work.device,
    )
    rows = 1
    for size in grad.shape[:-3]:
        rows *= size
    D, H, W = in_shape
    bounds, recip = _plan(in_shape, out_shape, work.device)[2]
    group = 4 if W % 4 == 0 else 1
    stem = "avg_pool_backward_" if group == 4 else "avg_pool_backward_scalar_"
    wide = "f32" if work.dtype == torch.float32 else "f16"
    getattr(_library(), stem + wide)(
        work, out, bounds, recip, rows, D, H, W, *out_shape,
        threads=(rows * D * H * (W // group) + 31) // 32 * 32, group_size=32,
    )
    return out.to(grad.dtype)


def _dims(tensor):
    rows = 1
    for size in tensor.shape[:-3]:
        rows *= size
    return rows, tensor.shape[-3], tensor.shape[-2], tensor.shape[-1]


def _pair_view(window, rank, dim):
    shape = [1] * rank
    shape[dim], shape[dim + 1] = window.shape
    return window.reshape(shape)


def _gather(tensor, dim, index):
    count, width = index.shape
    flat = torch.index_select(tensor, dim, index.reshape(-1))
    return flat.unflatten(dim, (count, width))


def _write(out, value):
    if out.shape != value.shape:
        out.resize_(value.shape)
    out.copy_(value)
    return out


# --------------------------------------------------------------------------
# average decomposition: window sum forward, covering-bin sum backward
# --------------------------------------------------------------------------


def _bin_sizes(plan):
    sizes = [
        (plan[axis][1] - plan[axis][0]).reshape(
            [-1 if i == axis else 1 for i in range(3)]
        )
        for axis in range(3)
    ]
    return sizes[0] * sizes[1] * sizes[2]


def _avg_pool3d_general(work, in_shape, out_shape, plan):
    """Sum each bin over the window and divide once by the bin volume."""
    axes = sorted(range(3), key=lambda a: in_shape[a] / out_shape[a], reverse=True)
    pooled = work
    for axis in axes:
        if in_shape[axis] != out_shape[axis]:
            dim = pooled.dim() - 3 + axis
            _, _, window, valid = plan[axis]
            gathered = _gather(pooled, dim, window)
            if window.shape[1] > 1:
                gathered = torch.where(
                    _pair_view(valid, gathered.dim(), dim),
                    gathered,
                    torch.zeros((), dtype=gathered.dtype, device=gathered.device),
                )
            pooled = gathered.sum(dim + 1)
    return pooled / _bin_sizes(plan).to(pooled.dtype)


def _avg_pool3d_backward_general(work, in_shape, out_shape, plan, cover):
    """Sum, for each input element, the bins that cover it."""
    axes = sorted(range(3), key=lambda a: in_shape[a] / out_shape[a])
    spread = work
    for axis in axes:
        if in_shape[axis] != out_shape[axis]:
            dim = spread.dim() - 3 + axis
            start, end, _, _ = plan[axis]
            index, valid = cover[axis]
            gathered = _gather(spread, dim, index)
            counts = (end - start).to(spread.dtype)[index]
            spread = torch.where(
                _pair_view(valid, gathered.dim(), dim),
                gathered / _pair_view(counts, gathered.dim(), dim),
                torch.zeros((), dtype=gathered.dtype, device=gathered.device),
            ).sum(dim + 1)
    return spread


# --------------------------------------------------------------------------
# max decomposition: window chain carrying the flat index of the winning slot
# --------------------------------------------------------------------------


def _pick(values, keys, nan_keys, dim):
    """Reduce the window axis to the value and the key of the winning slot.

    ATen's rule is ``val > max || isnan(val)``: the first maximum wins, and
    once a NaN is seen only later NaNs replace it.
    """
    top = values.amax(dim, keepdim=True)
    best = top.squeeze(dim)
    numeric = torch.where(values == top, keys, _SENTINEL).amin(dim)
    last = torch.where(values != values, nan_keys, -1).amax(dim)
    return best, torch.where(best != best, last, numeric), last


def _max_pool3d_general(x, in_shape, out_shape, plan):
    rank = x.dim()
    values, keys, nan_keys = x, None, None
    stride = 1
    for axis in (2, 1, 0):
        dim = rank - 3 + axis
        window = plan[axis][2]
        identity = window.shape[0] == values.shape[dim]
        window_key = _pair_view(window, rank + 1, dim).to(torch.int32)
        if identity:
            gathered = values.unsqueeze(dim + 1)
        else:
            gathered = _gather(values, dim, window)
        if keys is None:
            keys = nan_keys = window_key
        else:
            keys = window_key * stride + (
                keys.unsqueeze(dim + 1) if identity else _gather(keys, dim, window)
            )
            nan_keys = window_key * stride + (
                nan_keys.unsqueeze(dim + 1)
                if identity
                else _gather(nan_keys, dim, window)
            )
        values, keys, nan_keys = _pick(gathered, keys, nan_keys, dim + 1)
        stride *= in_shape[axis]
    return values, keys.to(torch.int64)


def _identity_indices(tensor):
    shape = tensor.shape
    spatial = shape[-1] * shape[-2] * shape[-3]
    index = torch.arange(spatial, device=tensor.device)
    view = index.reshape((1,) * (len(shape) - 3) + tuple(shape[-3:]))
    return view.expand(shape).contiguous()


def _adaptive_avg_pool3d(self, output_size):
    out_shape = [int(size) for size in output_size]
    _check(out_shape)
    _check_spatial(self)
    in_shape = list(self.shape[-3:])
    if in_shape == out_shape:
        return self.clone()
    if self.numel() == 0:
        return self.clone().reshape(tuple(self.shape[:-3]) + tuple(out_shape))
    if out_shape == [1, 1, 1]:
        work = _working(self)
        rows = 1
        for size in work.shape[:-3]:
            rows *= size
        flat = work.reshape(rows, -1)
        pooled = (flat.sum(dim=1) / flat.shape[1]).reshape(
            tuple(work.shape[:-3]) + (1, 1, 1)
        )
        return pooled.to(self.dtype)
    if _fused_ready(self, in_shape, out_shape, _fan_in, _FANIN_LIMIT):
        if _window_volume(in_shape, out_shape) > _AVG_NARROW_LIMIT:
            return _fused_avg_wide(self, out_shape)
        return _fused_avg(self, out_shape)
    plan = _plan(in_shape, out_shape, self.device)[0]
    work = _working(self)
    return _avg_pool3d_general(work, in_shape, out_shape, plan).to(self.dtype)


def _adaptive_avg_pool3d_out(self, output_size, *, out):
    return _write(out, _adaptive_avg_pool3d(self, output_size))


def _adaptive_avg_pool3d_backward(grad_output, self):
    in_shape = list(self.shape[-3:])
    out_shape = list(grad_output.shape[-3:])
    if in_shape == out_shape:
        return grad_output.clone()
    if _fused_ready(grad_output, in_shape, out_shape, _fan_in, _FANIN_LIMIT):
        return _fused_avg_backward(grad_output, in_shape)
    plan, cover, _ = _plan(in_shape, out_shape, grad_output.device)
    work = _working(grad_output)
    spread = _avg_pool3d_backward_general(work, in_shape, out_shape, plan, cover)
    return spread.to(grad_output.dtype)


def _adaptive_avg_pool3d_backward_out(grad_output, self, *, out):
    return _write(out, _adaptive_avg_pool3d_backward(grad_output, self))


def _adaptive_avg_pool3d_backward_grad_input(grad_output, self, *, grad_input):
    return _write(grad_input, _adaptive_avg_pool3d_backward(grad_output, self))


def _adaptive_max_pool3d(self, output_size):
    out_shape = [int(size) for size in output_size]
    _check(out_shape)
    _check_spatial(self)
    in_shape = list(self.shape[-3:])
    if in_shape == out_shape:
        return self.clone(), _identity_indices(self)
    if _fused_ready(self, in_shape, out_shape, _window_volume, _WINDOW_LIMIT):
        return _fused_max(self, out_shape)
    plan = _plan(in_shape, out_shape, self.device)[0]
    return _max_pool3d_general(self, in_shape, out_shape, plan)


def _adaptive_max_pool3d_out(self, output_size, *, out, indices):
    values, picked = _adaptive_max_pool3d(self, output_size)
    return _write(out, values), _write(indices, picked)


def _adaptive_max_pool3d_backward(grad_output, self, indices):
    in_shape = list(self.shape[-3:])
    rows = 1
    for size in grad_output.shape[:-3]:
        rows *= size
    if rows == 0:
        return torch.zeros(tuple(grad_output.shape[:-3]) + tuple(in_shape), dtype=grad_output.dtype, device=grad_output.device)
    spread = torch.zeros(
        (rows, in_shape[0] * in_shape[1] * in_shape[2]),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    spread.scatter_add_(1, indices.reshape(rows, -1), grad_output.reshape(rows, -1))
    return spread.reshape(tuple(grad_output.shape[:-3]) + tuple(in_shape))


def _adaptive_max_pool3d_backward_grad_input(grad_output, self, indices, *, grad_input):
    return _write(grad_input, _adaptive_max_pool3d_backward(grad_output, self, indices))


OVERLAYS = (
    ("_adaptive_avg_pool3d", _adaptive_avg_pool3d),
    ("_adaptive_avg_pool3d.out", _adaptive_avg_pool3d_out),
    ("_adaptive_avg_pool3d_backward", _adaptive_avg_pool3d_backward),
    ("_adaptive_avg_pool3d_backward.out", _adaptive_avg_pool3d_backward_out),
    ("adaptive_avg_pool3d_backward.grad_input", _adaptive_avg_pool3d_backward_grad_input),
    ("adaptive_max_pool3d", _adaptive_max_pool3d),
    ("adaptive_max_pool3d.out", _adaptive_max_pool3d_out),
    ("adaptive_max_pool3d_backward", _adaptive_max_pool3d_backward),
    ("adaptive_max_pool3d_backward.grad_input", _adaptive_max_pool3d_backward_grad_input),
)
