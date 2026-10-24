"""MPS overlays for aten's antialiased Lanczos 2-D resampler.

torch 2.14.1 ships ``aten::_upsample_lanczos2d_aa`` and its backward for CPU
only; on MPS both raise ``NotImplementedError``. This module fills the two
missing MPS kernels.

Semantics are copied from ``aten/src/ATen/native/cpu/UpSampleKernel.cpp``
(``HelperInterpLanczos``, ``_compute_indices_min_size_weights_aa`` and
``upsample_separable_Nd_backward_aa``) rather than approximated:

* The filter is Lanczos-3, ``sinc(x) * sinc(x/3)`` for ``|x| < 3`` with
  ``sinc(x) = sin(pi x) / (pi x)``. The op name says "lanczos2", the taps do
  not: upstream's ``interp_size`` is 6, i.e. three lobes per side.
* Per output index, ``center = scale * (i + 0.5)``,
  ``support = 3 * scale`` when ``scale >= 1`` else ``3``. The window is
  ``[center - support + 0.5, center + support + 0.5)`` clipped to the input, and
  the taps inside it are normalized to sum to 1, so image borders see a
  renormalized truncated filter rather than zero padding.
* ``scale`` is ``area_pixel_compute_scale``: ``(in - 1) / (out - 1)`` with
  ``align_corners`` (0 when ``out == 1``), else ``1 / scale_factor`` when a
  positive scale factor was supplied, else ``in / out``.
* The forward runs as two 1-D passes, horizontal then vertical, accumulating in
  float32. An axis whose size is unchanged is skipped entirely, which is why
  the identity plan (one tap, weight 1) is used instead of the near-identity
  weights the filter itself would produce.
* The backward is the transpose of that scatter: every input pixel sums
  ``wy * wx * grad_output`` over the output pixels whose window covers it (no
  atomics, so results are deterministic), and it does not skip unchanged axes.
* Upstream's *CPU forward* is wrong when the output width is 1 and the height
  changes: TensorIterator collapses the height axis, so every output row takes
  row 0's weights (the same defect hits ``_upsample_bilinear2d_aa`` and
  ``_upsample_bicubic2d_aa``). The CPU backward, the native MPS antialiased
  kernels and the math above all give per-row results, so this module computes
  the per-row result; ``benchmarks/verify_lanczos_ops.py`` checks that case
  against the adjoint of the CPU backward instead of the CPU forward.

All stages are Metal kernels compiled through ``torch.mps.compile_shader``: a
per-axis plan (window, normalized weights), a horizontal and a vertical pass, a
threadgroup tile for inputs whose intermediate would exceed
``_INTERMEDIATE_LIMIT``, and a fused 2-D pass for windows too tall for a tile.
The tile keeps the intermediate rows of one ``(_TILE_ROWS, _TILE_COLUMNS)``
block in threadgroup memory, so the forward reads the input once and writes the
output once however large the map is. At 8x64x64x64 -> 128x128 that removes the
16 MiB scratch round trip: measured against the two-pass form alternating in one
loop, the kernels take about a third to a quarter of the time and run at the
machine's copy bandwidth, where the two-pass form sits 3x to 4x above it.
The backward is the transpose of the same two passes, which costs ``btaps``
products per pixel per pass instead of the ``btaps_w * btaps_h`` of the fused
form, at the price of one scratch buffer.

Every arithmetic op, including the filter's ``sin``, runs on the MPS device;
plans are cached per geometry (``_axis_plan``, ``_transposed_plan``) because a
model resizes the same maps on every step and one plan build costs more than a
small resample; the caches hold 128 geometries and the report in
``benchmarks/verify_lanczos_ops.py`` records their hit rate and what a cold
resize costs on top of a warm call. float32 only: the MPS backend cannot hold
float64, and the upstream kernel does not implement Half/BFloat16.
"""

import ctypes
import functools
import math

import torch

CHECKED_TORCH = "2.14.1"

#: Lanczos-3: half-width of the filter in input pixels at scale 1.
_SUPPORT = 3.0

#: Output columns one threadgroup tile of the fused forward resamples at once.
_TILE_COLUMNS = 64

#: Intermediate rows a tile holds in threadgroup memory.
_TILE_ROWS = 48

#: Threads per tile.
_TILE_THREADS = 256

#: Most output rows one tile covers, however small the scale.
_TILE_MAX_ROWS = 32

#: Runtime cap on the intermediate rows a tile may require; 0 disables the tile.
_TILE_ROW_CAP = _TILE_ROWS

#: Bytes of scratch a downsampled height may allocate in the two-pass form before
#: the tile takes over; an upsampled height always takes the tile, which measured
#: 1.5x ahead at a 1 MiB buffer and 2.4x at 16 MiB. Measured in
#: ``benchmarks/verify_lanczos_ops.py``: 1024x1024 -> 512x512 (6 MiB) puts the
#: tile 1.3x ahead and 2048x2048 -> 1024x1024 (24 MiB) 1.4x, while 1024x1024 ->
#: 256x256 (3 MiB, a 4x downscale) keeps the two-pass form 1.35x ahead.
_INTERMEDIATE_LIMIT = 4 << 20

#: Scratch buffer limit for the separable backward. The two gather passes avoid
#: the O(btaps_w * btaps_h) double loop of the fused backward, saving 2.8x+ at
#: feature map sizes.
_BACKWARD_INTERMEDIATE_LIMIT = 256 << 20

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// UpSampleKernel.cpp HelperInterpLanczos::sinc_filter / aa_filter.
static inline float sinc_filter(float x) {
    if (x == 0.0f) {
        return 1.0f;
    }
    x *= 3.14159265358979323846f;
    return sin(x) / x;
}

static inline float lanczos_filter(float x) {
    float a = fabs(x);
    if (a < 3.0f) {
        return sinc_filter(a) * sinc_filter(a / 3.0f);
    }
    return 0.0f;
}

// One thread per output index: the clipped tap window and its normalized
// weights, exactly the layout _compute_index_ranges_weights produces.
kernel void axis_plan(
    device float* wts [[buffer(0)]],
    device int* mins [[buffer(1)]],
    device int* sizes [[buffer(2)]],
    constant int& out_size [[buffer(3)]],
    constant int& in_size [[buffer(4)]],
    constant int& taps [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    constant float& support [[buffer(7)]],
    constant float& invscale [[buffer(8)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= uint(out_size)) {
        return;
    }
    float center = scale * (float(i) + 0.5f);
    int lo = max(int(center - support + 0.5f), 0);
    int size = min(int(center + support + 0.5f), in_size) - lo;
    size = clamp(size, 0, taps);
    device float* row = wts + i * taps;
    float total = 0.0f;
    for (int j = 0; j < size; ++j) {
        float w = lanczos_filter((float(j + lo) - center + 0.5f) * invscale);
        row[j] = w;
        total += w;
    }
    if (total != 0.0f) {
        for (int j = 0; j < size; ++j) {
            row[j] /= total;
        }
    }
    for (int j = size; j < taps; ++j) {
        row[j] = 0.0f;
    }
    mins[i] = lo;
    sizes[i] = size;
}

// Transpose of axis_plan: for every input index, the output indices whose
// window covers it and the weight each of them gives it. Forward and backward
// therefore share one weight table, and a tap outside the window is 0.
kernel void backward_plan(
    device const float* wts [[buffer(0)]],
    device const int* mins [[buffer(1)]],
    device const int* sizes [[buffer(2)]],
    device float* twts [[buffer(3)]],
    device int* tids [[buffer(4)]],
    constant int& out_size [[buffer(5)]],
    constant int& in_size [[buffer(6)]],
    constant int& taps [[buffer(7)]],
    constant int& btaps [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    constant float& support [[buffer(10)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= uint(in_size)) {
        return;
    }
    // First candidate: one below the open support interval, so float rounding
    // in the cast can never drop a contributing index.
    int lo = 0;
    if (scale > 0.0f) {
        lo = int(floor((float(i) + 0.5f - support) / scale - 0.5f)) - 1;
    }
    for (int t = 0; t < btaps; ++t) {
        int o = lo + t;
        float w = 0.0f;
        int id = 0;
        if (o >= 0 && o < out_size) {
            int k = i - mins[o];
            if (k >= 0 && k < sizes[o]) {
                w = wts[o * taps + k];
                id = o;
            }
        }
        twts[i * btaps + t] = w;
        tids[i * btaps + t] = id;
    }
}

// Horizontal pass of the separable resample: one thread per output pixel row.
kernel void resample_width(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    device const float* wx [[buffer(2)]],
    device const int* xmin [[buffer(3)]],
    device const int* xsize [[buffer(4)]],
    constant int& chan [[buffer(5)]],
    constant int& planes [[buffer(6)]],
    constant int& ih [[buffer(7)]],
    constant int& ow [[buffer(8)]],
    constant int& taps_w [[buffer(9)]],
    constant int& ps0 [[buffer(10)]],
    constant int& ps1 [[buffer(11)]],
    constant int& sh [[buffer(12)]],
    constant int& sw [[buffer(13)]],
    constant int& pd0 [[buffer(14)]],
    constant int& pd1 [[buffer(15)]],
    constant int& dh [[buffer(16)]],
    constant int& dw [[buffer(17)]],
    uint gid [[thread_position_in_grid]]) {
    long total = (long)planes * ih * ow;
    if ((long)gid >= total) {
        return;
    }
    int ox = gid % ow;
    int rest = gid / ow;
    int y = rest % ih;
    int p = rest / ih;
    int n = p / chan, c = p - n * chan;
    device const float* row = src + n * (long)ps0 + c * (long)ps1 + y * (long)sh;
    device float* drow = dst + n * (long)pd0 + c * (long)pd1 + y * (long)dh;

    int xs = xmin[ox], xz = xsize[ox];
    device const float* wrow = wx + ox * taps_w;
    float sum = 0.0f;
    if (sw == 1) {
        for (int x = 0; x < xz; ++x) {
            sum += row[xs + x] * wrow[x];
        }
    } else {
        for (int x = 0; x < xz; ++x) {
            sum += row[(xs + x) * sw] * wrow[x];
        }
    }
    drow[ox * dw] = sum;
}

// Vertical pass: one thread per output pixel, summing over the row window.
kernel void resample_height(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    device const float* wy [[buffer(2)]],
    device const int* ymin [[buffer(3)]],
    device const int* ysize [[buffer(4)]],
    constant int& chan [[buffer(5)]],
    constant int& planes [[buffer(6)]],
    constant int& oh [[buffer(7)]],
    constant int& ow [[buffer(8)]],
    constant int& taps_h [[buffer(9)]],
    constant int& ps0 [[buffer(10)]],
    constant int& ps1 [[buffer(11)]],
    constant int& sh [[buffer(12)]],
    constant int& sw [[buffer(13)]],
    constant int& pd0 [[buffer(14)]],
    constant int& pd1 [[buffer(15)]],
    constant int& dh [[buffer(16)]],
    constant int& dw [[buffer(17)]],
    uint gid [[thread_position_in_grid]]) {
    long total = (long)planes * oh * ow;
    if ((long)gid >= total) {
        return;
    }
    int ox = gid % ow;
    int rest = gid / ow;
    int oy = rest % oh;
    int p = rest / oh;
    int n = p / chan, c = p - n * chan;
    device const float* plane = src + n * (long)ps0 + c * (long)ps1 + ox * (long)sw;
    device float* dplane = dst + n * (long)pd0 + c * (long)pd1;

    int ys = ymin[oy], yz = ysize[oy];
    device const float* wcol = wy + oy * taps_h;
    float acc = 0.0f;
    for (int y = 0; y < yz; ++y) {
        acc += plane[(ys + y) * (long)sh] * wcol[y];
    }
    dplane[oy * (long)dh + ox * (long)dw] = acc;
}

// Both axes at once: costs xsize * ysize products per output instead of
// xsize + ysize, but needs no intermediate, so it is the fallback when the
// two-pass intermediate would not fit in the budget.
kernel void resample_forward(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    device const float* wx [[buffer(2)]],
    device const float* wy [[buffer(3)]],
    device const int* xmin [[buffer(4)]],
    device const int* ymin [[buffer(5)]],
    device const int* xsize [[buffer(6)]],
    device const int* ysize [[buffer(7)]],
    constant int& chan [[buffer(8)]],
    constant int& planes [[buffer(9)]],
    constant int& oh [[buffer(10)]],
    constant int& ow [[buffer(11)]],
    constant int& taps_w [[buffer(12)]],
    constant int& taps_h [[buffer(13)]],
    constant int& ps0 [[buffer(14)]],
    constant int& ps1 [[buffer(15)]],
    constant int& sh [[buffer(16)]],
    constant int& sw [[buffer(17)]],
    constant int& pd0 [[buffer(18)]],
    constant int& pd1 [[buffer(19)]],
    constant int& dh [[buffer(20)]],
    constant int& dw [[buffer(21)]],
    uint gid [[thread_position_in_grid]]) {
    long total = (long)planes * oh * ow;
    if ((long)gid >= total) {
        return;
    }
    int ox = gid % ow;
    int rest = gid / ow;
    int oy = rest % oh;
    int p = rest / oh;
    int n = p / chan, c = p - n * chan;
    device const float* plane = src + n * (long)ps0 + c * (long)ps1;
    device float* dplane = dst + n * (long)pd0 + c * (long)pd1;

    int xs = xmin[ox], xz = xsize[ox];
    int ys = ymin[oy], yz = ysize[oy];
    device const float* wrow = wx + ox * taps_w;
    device const float* wcol = wy + oy * taps_h;

    float acc = 0.0f;
    for (int y = 0; y < yz; ++y) {
        device const float* row = plane + (ys + y) * (long)sh + xs * (long)sw;
        float sum = 0.0f;
        if (sw == 1) {
            for (int x = 0; x < xz; ++x) {
                sum += row[x] * wrow[x];
            }
        } else {
            for (int x = 0; x < xz; ++x) {
                sum += row[x * sw] * wrow[x];
            }
        }
        acc += sum * wcol[y];
    }
    dplane[oy * (long)dh + ox * (long)dw] = acc;
}

kernel void resample_backward(
    device const float* gout [[buffer(0)]],
    device float* gin [[buffer(1)]],
    device const float* wxt [[buffer(2)]],
    device const float* wyt [[buffer(3)]],
    device const int* xidx [[buffer(4)]],
    device const int* yidx [[buffer(5)]],
    constant int& chan [[buffer(6)]],
    constant int& planes [[buffer(7)]],
    constant int& ih [[buffer(8)]],
    constant int& iw [[buffer(9)]],
    constant int& taps_w [[buffer(10)]],
    constant int& taps_h [[buffer(11)]],
    constant int& gs0 [[buffer(12)]],
    constant int& gs1 [[buffer(13)]],
    constant int& gh [[buffer(14)]],
    constant int& gw [[buffer(15)]],
    constant int& di0 [[buffer(16)]],
    constant int& di1 [[buffer(17)]],
    constant int& dh [[buffer(18)]],
    constant int& dw [[buffer(19)]],
    uint gid [[thread_position_in_grid]]) {
    long total = (long)planes * ih * iw;
    if ((long)gid >= total) {
        return;
    }
    int iw_i = gid % iw;
    int rest = gid / iw;
    int ih_i = rest % ih;
    int p = rest / ih;
    int n = p / chan, c = p - n * chan;
    device const float* gplane = gout + n * (long)gs0 + c * (long)gs1;
    device float* dplane = gin + n * (long)di0 + c * (long)di1;

    device const int* xrow = xidx + iw_i * taps_w;
    device const float* wxrow = wxt + iw_i * taps_w;
    device const int* yrow = yidx + ih_i * taps_h;
    device const float* wyrow = wyt + ih_i * taps_h;

    float acc = 0.0f;
    for (int ty = 0; ty < taps_h; ++ty) {
        float wy = wyrow[ty];
        if (wy == 0.0f) {
            continue;
        }
        device const float* row = gplane + yrow[ty] * (long)gh;
        float sum = 0.0f;
        if (gw == 1) {
            for (int tx = 0; tx < taps_w; ++tx) {
                sum += row[xrow[tx]] * wxrow[tx];
            }
        } else {
            for (int tx = 0; tx < taps_w; ++tx) {
                sum += row[xrow[tx] * gw] * wxrow[tx];
            }
        }
        acc += sum * wy;
    }
    dplane[ih_i * (long)dh + iw_i * (long)dw] = acc;
}

// Constant-sized threadgroup tile: the horizontal pass writes the intermediate
// rows this tile needs into threadgroup memory and the vertical pass reads them
// back, so the intermediate never reaches DRAM. Costs ``span * TILE_W``
// horizontal products per ``th * TILE_W`` outputs instead of one per output,
// where ``span`` is the union of the tile's row windows.
constant int TILE_W = $TILE_COLUMNS;
constant int TILE_H = $TILE_ROWS;
constant int TILE_T = $TILE_THREADS;

kernel void resample_tile(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    device const float* wx [[buffer(2)]],
    device const int* xmin [[buffer(3)]],
    device const int* xsize [[buffer(4)]],
    device const float* wy [[buffer(5)]],
    device const int* ymin [[buffer(6)]],
    device const int* ysize [[buffer(7)]],
    constant int& chan [[buffer(8)]],
    constant int& oh [[buffer(9)]],
    constant int& ow [[buffer(10)]],
    constant int& taps_w [[buffer(11)]],
    constant int& taps_h [[buffer(12)]],
    constant int& ps0 [[buffer(13)]],
    constant int& ps1 [[buffer(14)]],
    constant int& sh_row [[buffer(15)]],
    constant int& sw_col [[buffer(16)]],
    constant int& pd0 [[buffer(17)]],
    constant int& pd1 [[buffer(18)]],
    constant int& dh [[buffer(19)]],
    constant int& dw [[buffer(20)]],
    constant int& th [[buffer(21)]],
    constant int& tiles_x [[buffer(22)]],
    constant int& tiles_y [[buffer(23)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
    // Declared as float4 so the vertical pass can read whole blocks.
    threadgroup float4 mid4[TILE_H * TILE_W / 4];
    threadgroup float* mid = (threadgroup float*)mid4;

    int per_plane = tiles_y * tiles_x;
    int plane = int(tgid) / per_plane;
    int rem = int(tgid) - plane * per_plane;
    int ti = rem / tiles_x, tj = rem - ti * tiles_x;
    int oy0 = ti * th, ox0 = tj * TILE_W;
    int n = plane / chan, c = plane - n * chan;
    device const float* sp = src + n * (long)ps0 + c * (long)ps1;
    device float* dp = dst + n * (long)pd0 + c * (long)pd1;

    int oy_last = min(oy0 + th, oh) - 1;
    int lo = ymin[oy0];
    int span = ymin[oy_last] + ysize[oy_last] - lo;

    for (int idx = int(tid); idx < span * TILE_W; idx += TILE_T) {
        int r = idx / TILE_W, cc = idx - r * TILE_W;
        int ox = ox0 + cc;
        float sum = 0.0f;
        if (ox < ow) {
            device const float* row = sp + (long)(lo + r) * sh_row;
            int xs = xmin[ox], xz = xsize[ox];
            device const float* wrow = wx + ox * taps_w;
            if (sw_col == 1) {
                for (int x = 0; x < xz; ++x) {
                    sum += row[xs + x] * wrow[x];
    }
            } else {
                for (int x = 0; x < xz; ++x) {
                    sum += row[(xs + x) * sw_col] * wrow[x];
}
            }
        }
        mid[r * TILE_W + cc] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int blocks = TILE_W / 4;
    for (int idx = int(tid); idx < th * blocks; idx += TILE_T) {
        int rr = idx / blocks, blk = idx - rr * blocks;
        int oy = oy0 + rr;
        if (oy >= oh) {
            break;
        }
        int cc = blk * 4;
        int ox = ox0 + cc;
        if (ox >= ow) {
            continue;
        }
        int ys = ymin[oy] - lo, yz = ysize[oy];
        device const float* wcol = wy + oy * taps_h;
        if (ox + 3 < ow) {
            threadgroup const float* base = mid + ys * TILE_W + cc;
            float4 acc = float4(0.0f);
            for (int y = 0; y < yz; ++y) {
                acc += *(threadgroup const float4*)(base + y * TILE_W) * wcol[y];
            }
            device float* d = dp + oy * (long)dh + ox * dw;
            if (dw == 1 && (dh & 3) == 0) {
                *(device float4*)d = acc;
            } else {
                d[0] = acc.x;
                d[dw] = acc.y;
                d[2 * dw] = acc.z;
                d[3 * dw] = acc.w;
            }
        } else {
            for (int k = 0; ox + k < ow; ++k) {
                float acc = 0.0f;
                for (int y = 0; y < yz; ++y) {
                    acc += mid[(ys + y) * TILE_W + cc + k] * wcol[y];
                }
                dp[oy * (long)dh + (ox + k) * dw] = acc;
            }
        }
    }
}

// One pass of the transposed resample: ``dst[p, r, c]`` sums the source rows
// ``ids[r][:]`` at column ``c``. Half of the backward's separable form, and the
// exact transpose of ``resample_height``.
kernel void gather_rows(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    device const float* wts [[buffer(2)]],
    device const int* ids [[buffer(3)]],
    constant int& chan [[buffer(4)]],
    constant int& planes [[buffer(5)]],
    constant int& rows [[buffer(6)]],
    constant int& cols [[buffer(7)]],
    constant int& taps [[buffer(8)]],
    constant int& ss0 [[buffer(9)]],
    constant int& ss1 [[buffer(10)]],
    constant int& srow [[buffer(11)]],
    constant int& scol [[buffer(12)]],
    constant int& ds0 [[buffer(13)]],
    constant int& ds1 [[buffer(14)]],
    constant int& drow [[buffer(15)]],
    constant int& dcol [[buffer(16)]],
    uint gid [[thread_position_in_grid]]) {
    long total = (long)planes * rows * cols;
    if ((long)gid >= total) {
        return;
    }
    int col = int(gid) % cols;
    int rest = int(gid) / cols;
    int r = rest % rows;
    int p = rest / rows;
    int n = p / chan, c = p - n * chan;
    device const float* sp = src + n * (long)ss0 + c * (long)ss1;
    device float* dp = dst + n * (long)ds0 + c * (long)ds1;
    device const int* idr = ids + r * taps;
    device const float* wtr = wts + r * taps;
    float acc = 0.0f;
    for (int t = 0; t < taps; ++t) {
        float w = wtr[t];
        if (w != 0.0f) {
            acc += sp[idr[t] * (long)srow + col * (long)scol] * w;
        }
    }
    dp[r * (long)drow + col * (long)dcol] = acc;
}

// The other half: ``dst[p, r, c]`` sums column ``c``'s source indices within
// row ``r``, the exact transpose of ``resample_width``.
kernel void gather_cols(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    device const float* wts [[buffer(2)]],
    device const int* ids [[buffer(3)]],
    constant int& chan [[buffer(4)]],
    constant int& planes [[buffer(5)]],
    constant int& rows [[buffer(6)]],
    constant int& cols [[buffer(7)]],
    constant int& taps [[buffer(8)]],
    constant int& ss0 [[buffer(9)]],
    constant int& ss1 [[buffer(10)]],
    constant int& srow [[buffer(11)]],
    constant int& scol [[buffer(12)]],
    constant int& ds0 [[buffer(13)]],
    constant int& ds1 [[buffer(14)]],
    constant int& drow [[buffer(15)]],
    constant int& dcol [[buffer(16)]],
    uint gid [[thread_position_in_grid]]) {
    long total = (long)planes * rows * cols;
    if ((long)gid >= total) {
        return;
    }
    int col = int(gid) % cols;
    int rest = int(gid) / cols;
    int r = rest % rows;
    int p = rest / rows;
    int n = p / chan, c = p - n * chan;
    device const float* sp = src + n * (long)ss0 + c * (long)ss1;
    device float* dp = dst + n * (long)ds0 + c * (long)ds1;
    device const int* idc = ids + col * taps;
    device const float* wtc = wts + col * taps;
    device const float* srow_p = sp + r * (long)srow;
    float acc = 0.0f;
    for (int t = 0; t < taps; ++t) {
        float w = wtc[t];
        if (w != 0.0f) {
            acc += srow_p[idc[t] * (long)scol] * w;
        }
    }
    dp[r * (long)drow + col * (long)dcol] = acc;
}
"""


@functools.lru_cache(maxsize=1)
def _library():
    source = (
        _SOURCE.replace("$TILE_COLUMNS", str(_TILE_COLUMNS))
        .replace("$TILE_ROWS", str(_TILE_ROWS))
        .replace("$TILE_THREADS", str(_TILE_THREADS))
    )
    return torch.mps.compile_shader(source)


def _f32(value):
    """Round a Python float to float32 the way ``static_cast<float>`` does."""
    return ctypes.c_float(value).value


def _axis_scale(in_size, out_size, align_corners, scale):
    if align_corners:
        if out_size > 1:
            return _f32(_f32(in_size - 1) / _f32(out_size - 1))
        return 0.0
    if scale is not None and scale > 0:
        return _f32(1.0 / scale)
    return _f32(_f32(in_size) / _f32(out_size))


@functools.lru_cache(maxsize=128)
def _axis_plan(in_size, out_size, align_corners, scale, identity):
    """Weights, window start and window size for one axis.

    ``identity`` is the forward's skip of an unchanged axis: one tap at the same
    index with weight 1, which reproduces upstream's ``copy_`` branch exactly.
    A plan depends only on the geometry, so it is cached: models resize the
    same maps every step, and one plan build costs more than a small resample.
    """
    device = torch.device("mps")
    if identity:
        mins = torch.arange(out_size, dtype=torch.int32, device=device)
        sizes = torch.ones(out_size, dtype=torch.int32, device=device)
        wts = torch.ones((out_size, 1), dtype=torch.float32, device=device)
        return wts, mins, sizes, 1, 1.0, _SUPPORT

    scale_f = _axis_scale(in_size, out_size, align_corners, scale)
    support = _f32(_SUPPORT * scale_f) if scale_f >= 1.0 else _SUPPORT
    invscale = _f32(1.0 / scale_f) if scale_f >= 1.0 and scale_f > 0.0 else 1.0
    taps = int(math.ceil(support)) * 2 + 1
    wts = torch.empty((out_size, taps), dtype=torch.float32, device=device)
    mins = torch.empty(out_size, dtype=torch.int32, device=device)
    sizes = torch.empty(out_size, dtype=torch.int32, device=device)
    _library().axis_plan(
        wts, mins, sizes, out_size, in_size, taps, scale_f, support, invscale,
        threads=out_size, group_size=32,
    )
    return wts, mins, sizes, taps, scale_f, support


@functools.lru_cache(maxsize=128)
def _transposed_plan(in_size, out_size, align_corners, scale):
    """Backward tables: for each input index, the output indices whose window
    covers it and the weight each gives it."""
    wts, mins, sizes, taps, scale_f, support = _axis_plan(
        in_size, out_size, align_corners, scale, False)
    if scale_f > 0.0:
        btaps = int(math.ceil(2.0 * support / scale_f)) + 3
    else:
        # align_corners with a single input row: every output sits on that row.
        btaps = out_size
    twts = torch.empty((in_size, btaps), dtype=torch.float32, device=wts.device)
    tids = torch.empty((in_size, btaps), dtype=torch.int32, device=wts.device)
    _library().backward_plan(
        wts, mins, sizes, twts, tids,
        out_size, in_size, taps, btaps, scale_f, support,
        threads=in_size, group_size=32,
    )
    return twts, tids, btaps


def _check_float32(tensor, name):
    if tensor.device.type != "mps":
        raise ValueError(f"{name} must be on the mps device, got {tensor.device}")
    if tensor.dtype != torch.float32:
        raise NotImplementedError(
            f"aten::_upsample_lanczos2d_aa supports float32 on MPS, got {tensor.dtype}"
        )


def _suggest_memory_format(tensor):
    """``Tensor::suggest_memory_format``: channels-last only when the tensor
    actually uses that layout, as opposed to also being contiguous."""
    if (
        tensor.dim() == 4
        and tensor.is_contiguous(memory_format=torch.channels_last)
        and not tensor.is_contiguous()
    ):
        return torch.channels_last
    return torch.contiguous_format


def _strides(tensor):
    return tuple(tensor.stride(i) for i in range(4))


def _launch_width(src, dst, plan, ih, ow, n, c):
    wx, xmin, xsize, taps, _, _ = plan
    _library().resample_width(
        src, dst, wx, xmin, xsize,
        c, n * c, ih, ow, taps, *_strides(src), *_strides(dst),
        threads=n * c * ih * ow, group_size=256,
    )


def _launch_height(src, dst, plan, oh, ow, n, c):
    wy, ymin, ysize, taps, _, _ = plan
    _library().resample_height(
        src, dst, wy, ymin, ysize,
        c, n * c, oh, ow, taps, *_strides(src), *_strides(dst),
        threads=n * c * oh * ow, group_size=256,
    )


def _tile_rows(scale, support):
    """Output rows one tile covers, or 0 when its window cannot fit.

    A tile of ``th`` output rows holds the union of their row windows, at most
    ``th * scale + 2 * support + 2`` intermediate rows. Rows are the scarce
    resource, so the tile is as tall as the window allows.
    """
    room = min(_TILE_ROWS, _TILE_ROW_CAP) - 2.0 * support - 2.0
    if room < 0.0:
        return 0
    if scale <= 0.0:
        return _TILE_MAX_ROWS
    return max(0, min(int(room / scale), _TILE_MAX_ROWS))


def _launch_tile(src, dst, wplan, hplan, th, oh, ow, n, c):
    wx, xmin, xsize, taps_w, _, _ = wplan
    wy, ymin, ysize, taps_h, _, _ = hplan
    tiles_x = -(-ow // _TILE_COLUMNS)
    tiles_y = -(-oh // th)
    _library().resample_tile(
        src, dst, wx, xmin, xsize, wy, ymin, ysize,
        c, oh, ow, taps_w, taps_h, *_strides(src), *_strides(dst),
        th, tiles_x, tiles_y,
        threads=n * c * tiles_x * tiles_y * _TILE_THREADS,
        group_size=_TILE_THREADS,
    )


def _launch_gather_rows(src, dst, plan, rows, cols, n, c):
    wts, ids, taps = plan
    _library().gather_rows(
        src, dst, wts, ids, c, n * c, rows, cols, taps,
        *_strides(src), *_strides(dst),
        threads=n * c * rows * cols, group_size=256,
    )


def _launch_gather_cols(src, dst, plan, rows, cols, n, c):
    wts, ids, taps = plan
    _library().gather_cols(
        src, dst, wts, ids, c, n * c, rows, cols, taps,
        *_strides(src), *_strides(dst),
        threads=n * c * rows * cols, group_size=256,
    )


def _forward_stage(n, c, ih, iw, oh, ow, height_plan, element_size):
    """Which forward kernels this geometry takes.

    ``tile`` keeps the intermediate rows in threadgroup memory and needs no
    scratch at all. On an upsampled height it spans ``(th * scale + 8) / th``
    intermediate rows per output row whatever the map size, and it measured
    ahead of the two-pass form from the smallest maps up (1.5x at a 1 MiB
    buffer, 2.4x at 16 MiB). On a downsampled height it spans ``scale``
    intermediate rows per output row, so it redoes the horizontal work ``scale``
    times: there the two-pass form, which spreads one thread per output pixel,
    keeps the small buffers and the tile takes over at
    ``_INTERMEDIATE_LIMIT``. ``fused`` is the last resort, for windows too tall
    for a tile and a buffer past the limit.
    """
    if n * c == 0:
        return "empty"
    if iw == ow:
        return "height"
    if ih == oh:
        return "width"
    scratch = n * c * ih * ow * element_size
    if (height_plan[4] <= 1.0 or scratch > _INTERMEDIATE_LIMIT) and _tile_rows(
            height_plan[4], height_plan[5]):
        return "tile"
    if scratch <= _INTERMEDIATE_LIMIT:
        return "two_pass"
    return "fused"


def _backward_stage(planes, ih, iw, oh, ow, element_size):
    """Each pass of the separable backward costs ``btaps`` products per pixel;
    the fused form costs ``btaps_w * btaps_h`` but needs no scratch."""
    if planes == 0:
        return "empty"
    if planes * min(ih * ow, oh * iw) * element_size <= _BACKWARD_INTERMEDIATE_LIMIT:
        return "separable"
    return "fused"


def _upsample_lanczos2d_aa(self, output_size, align_corners, scales_h=None, scales_w=None):
    _check_float32(self, "input")
    if self.dim() != 4:
        raise ValueError(f"Non-empty 4D data tensor expected but got {self.dim()}D")
    if len(output_size) != 2:
        raise ValueError(f"expected output_size of 2 elements, got {len(output_size)}")
    oh, ow = int(output_size[0]), int(output_size[1])
    n, c, ih, iw = self.shape
    if min(ih, iw, oh, ow) <= 0:
        raise ValueError(
            f"Input and output sizes should be greater than 0, got input {(ih, iw)} "
            f"and output {(oh, ow)}"
        )

    out = torch.empty(
        (n, c, oh, ow), dtype=self.dtype, device=self.device,
        memory_format=_suggest_memory_format(self),
    )
    if ih == oh and iw == ow:
        return out.copy_(self)

    width_plan = _axis_plan(iw, ow, align_corners, scales_w, iw == ow)
    height_plan = _axis_plan(ih, oh, align_corners, scales_h, ih == oh)
    stage = _forward_stage(
        n, c, ih, iw, oh, ow, height_plan, self.element_size())
    if stage == "empty":
        return out
    if stage == "height":
        _launch_height(self, out, height_plan, oh, ow, n, c)
    elif stage == "width":
        _launch_width(self, out, width_plan, ih, ow, n, c)
    elif stage == "two_pass":
        temp = torch.empty(
            (n, c, ih, ow), dtype=self.dtype, device=self.device,
            memory_format=torch.contiguous_format,
        )
        _launch_width(self, temp, width_plan, ih, ow, n, c)
        _launch_height(temp, out, height_plan, oh, ow, n, c)
        del temp
    elif stage == "tile":
        _launch_tile(
            self, out, width_plan, height_plan,
            _tile_rows(height_plan[4], height_plan[5]), oh, ow, n, c,
        )
    else:
        wx, xmin, xsize, taps_w, _, _ = width_plan
        wy, ymin, ysize, taps_h, _, _ = height_plan
        ps0, ps1, sh, sw = (
            self.stride(0), self.stride(1), self.stride(2), self.stride(3))
        pd0, pd1, dh, dw = (
            out.stride(0), out.stride(1), out.stride(2), out.stride(3))
        _library().resample_forward(
            self, out, wx, wy, xmin, ymin, xsize, ysize,
            c, n * c, oh, ow, taps_w, taps_h,
            ps0, ps1, sh, sw, pd0, pd1, dh, dw,
            threads=n * c * oh * ow, group_size=256,
        )
    return out


def _upsample_lanczos2d_aa_backward(
    grad_output, output_size, input_size, align_corners, scales_h=None, scales_w=None
):
    _check_float32(grad_output, "grad_output")
    if grad_output.dim() != 4:
        raise ValueError(
            f"Expected grad_output to be a tensor of dimension 4 but got: "
            f"dimension {grad_output.dim()}"
        )
    if len(output_size) != 2:
        raise ValueError(f"expected output_size of 2 elements, got {len(output_size)}")
    if len(input_size) != 4:
        raise ValueError(f"expected input_size of 4 elements, got {len(input_size)}")
    n, c, ih, iw = (int(s) for s in input_size)
    oh, ow = int(output_size[0]), int(output_size[1])
    if min(ih, iw, oh, ow) <= 0:
        raise ValueError(
            f"Input and output sizes should be greater than 0, got input {(ih, iw)} "
            f"and output {(oh, ow)}"
        )
    expected = (n, c, oh, ow)
    if tuple(grad_output.shape) != expected:
        raise ValueError(
            f"Expected grad_output to have the same shape as output; output.size() = "
            f"{expected} but got grad_output.size() = {tuple(grad_output.shape)}"
        )

    # Upstream's meta allocates grad_input contiguous, whatever grad_output's layout.
    grad_input = torch.empty((n, c, ih, iw), dtype=grad_output.dtype, device=grad_output.device)
    wxt, xidx, btaps_w = _transposed_plan(iw, ow, align_corners, scales_w)
    wyt, yidx, btaps_h = _transposed_plan(ih, oh, align_corners, scales_h)
    planes = n * c

    gs0, gs1, gh, gw = (
        grad_output.stride(0), grad_output.stride(1),
        grad_output.stride(2), grad_output.stride(3),
    )
    di0, di1, dh, dw = (
        grad_input.stride(0), grad_input.stride(1),
        grad_input.stride(2), grad_input.stride(3),
    )
    threads = planes * ih * iw

    stage = _backward_stage(
        planes, ih, iw, oh, ow, grad_output.element_size())
    if stage == "empty":
        return grad_input
    if stage == "separable":
        # Transpose of the forward's two passes, smaller intermediate first.
        rows_first = ih * ow <= oh * iw
        temp = torch.empty(
            (n, c) + ((ih, ow) if rows_first else (oh, iw)),
            dtype=grad_output.dtype, device=grad_output.device,
        )
        if rows_first:
            _launch_gather_rows(grad_output, temp, (wyt, yidx, btaps_h), ih, ow, n, c)
            _launch_gather_cols(temp, grad_input, (wxt, xidx, btaps_w), ih, iw, n, c)
        else:
            _launch_gather_cols(grad_output, temp, (wxt, xidx, btaps_w), oh, iw, n, c)
            _launch_gather_rows(temp, grad_input, (wyt, yidx, btaps_h), ih, iw, n, c)
        return grad_input

    _library().resample_backward(
        grad_output, grad_input, wxt, wyt, xidx, yidx,
        c, planes, ih, iw, btaps_w, btaps_h,
        gs0, gs1, gh, gw, di0, di1, dh, dw,
        threads=threads, group_size=256,
    )
    return grad_input



OVERLAYS = (
    ("_upsample_lanczos2d_aa", _upsample_lanczos2d_aa),
    ("_upsample_lanczos2d_aa_backward", _upsample_lanczos2d_aa_backward),
)
