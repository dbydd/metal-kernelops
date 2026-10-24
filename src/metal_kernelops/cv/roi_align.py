"""Differentiable RoIAlign on MPS, with a Metal kernel and a torch fallback.

``torchvision.ops.roi_align`` has no MPS kernel in torchvision 0.29.1 and raises
on an MPS input. This module supplies the same operator.

Two paths share one definition, transcribed from torchvision's pure-Python
implementation (``torchvision/ops/roi_align.py``, BSD-3-Clause), itself a
transcription of the CUDA kernel:

* float32 on MPS takes Metal kernels. The forward gives one thread a
  ``(roi, channel block, bin)`` and accumulates the bin's bilinear taps in
  registers, so the ``(K, C, PH, PW, IY, IX)`` intermediate the composed form
  needs is never allocated. The adaptive grid, whose sample count depends on the
  box, has its own kernel and still never materializes the masked
  ``(K, C, PH, PW, H, W)`` tensor. Each kernel is compiled per
  ``(ratio, PH, PW, C, block)`` so every index split and the sample loops are
  compile-time constants; the compile is lazy and the Metal compiler caches it
  on disk.
* the backward gathers instead of scattering: one thread owns one input element
  and sums the sampling-grid cells that can reach it, found by inverting the
  evenly spaced sample position rather than walking the grid. One writer per
  element means the result is bitwise reproducible, which the atomic scatter
  torchvision's CUDA kernel uses is not.
* everything else (CPU, half and bfloat16 inputs, degenerate shapes) takes the
  composed torch path. It is exact and differentiable but allocates the
  intermediate.

Args match torchvision::

    roi_align(input, boxes, output_size, spatial_scale=1.0,
              sampling_ratio=-1, aligned=False)

``input`` is ``(N, C, H, W)`` and ``boxes`` is ``(K, 5)`` with a batch index in
the first column, or a list of ``(L, 4)`` tensors. The result is
``(K, C, output_size[0], output_size[1])``.

Box coordinates are not differentiated, matching torchvision, whose kernel
treats them as constants. Gradients flow to ``input`` only.
"""

from functools import lru_cache

import torch

__all__ = ["roi_align"]

_GROUP = 128

_BLOCKS = (4, 2, 1)
"""Forward channel blocking candidates, widest first."""

_BACKWARD_BLOCKS = (32, 16, 8, 4, 2, 1)
"""The backward walks every box of its image once per thread, so it wants the
widest block the register file still takes."""

_MIN_THREADS = 1 << 16
"""Below this, narrowing the channel block buys parallelism instead."""

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// One thread per (roi, channel block, bin). The bin's bilinear taps are
// accumulated in registers for $BLOCK channels at once, so the box geometry and
// the sample coordinates are computed once per $BLOCK * ratio^2 taps and no
// (K, C, PH, PW, IY, IX) tensor is allocated. Ratio, pooled size, channel count
// and block size are compiled in: every index split becomes a constant multiply
// and both loops unroll.
kernel void roi_align_f32(
    device const float* x, device const float* rois, device float* out,
    constant uint& height, constant uint& width,
    constant float& scale, constant float& offset,
    constant uint& total,
    uint gid [[thread_position_in_grid]]) {
    if (gid >= total) return;
    const uint kC = $CH, kPH = $PH, kPW = $PW, kR = $RATIO, kCB = $BLOCK;
    const uint bins = kPH * kPW;
    uint bin = gid % bins;
    uint rest = gid / bins;
    uint cb = rest % (kC / kCB);
    uint k = rest / (kC / kCB);
    uint ow = bin % kPW;
    uint oh = bin / kPW;

    device const float* roi = rois + k * 5u;
    uint batch = uint(roi[0]);
    float start_w = roi[1] * scale - offset;
    float start_h = roi[2] * scale - offset;
    float roi_w = roi[3] * scale - offset - start_w;
    float roi_h = roi[4] * scale - offset - start_h;
    if (offset == 0.0f) {
        roi_w = max(roi_w, 1.0f);
        roi_h = max(roi_h, 1.0f);
    }
    float bin_w = roi_w / float(kPW);
    float bin_h = roi_h / float(kPH);
    float step_w = bin_w / float(kR);
    float step_h = bin_h / float(kR);
    float origin_w = start_w + float(ow) * bin_w;
    float origin_h = start_h + float(oh) * bin_h;

    const size_t stride = (size_t)height * width;
    device const float* plane = x + ((size_t)batch * kC + cb * kCB) * stride;
    const int hmax = int(height) - 1;
    const int wmax = int(width) - 1;
    float acc[$BLOCK];
    #pragma unroll
    for (uint j = 0; j < kCB; ++j) acc[j] = 0.0f;
    #pragma unroll
    for (uint iy = 0; iy < kR; ++iy) {
        float y = origin_h + (float(iy) + 0.5f) * step_h;
        // torchvision masks a sample outside [-1, size) instead of clamping it,
        // so a box that leaves the map contributes zeros for that tap.
        if (y < -1.0f || y > float(height)) continue;
        float yc = max(y, 0.0f);
        int y0 = int(yc);
        if (y0 >= hmax) { y0 = hmax; yc = float(y0); }
        float ly = yc - float(y0);
        float hy = 1.0f - ly;
        int y1 = min(y0 + 1, hmax);
        device const float* row0 = plane + (size_t)y0 * width;
        device const float* row1 = plane + (size_t)y1 * width;
        #pragma unroll
        for (uint ix = 0; ix < kR; ++ix) {
            float xs = origin_w + (float(ix) + 0.5f) * step_w;
            if (xs < -1.0f || xs > float(width)) continue;
            float xc = max(xs, 0.0f);
            int x0 = int(xc);
            if (x0 >= wmax) { x0 = wmax; xc = float(x0); }
            float lx = xc - float(x0);
            float hx = 1.0f - lx;
            int x1 = min(x0 + 1, wmax);
            #pragma unroll
            for (uint j = 0; j < kCB; ++j) {
                device const float* r0 = row0 + (size_t)j * stride;
                device const float* r1 = row1 + (size_t)j * stride;
                acc[j] += hy * hx * r0[x0] + hy * lx * r0[x1]
                        + ly * hx * r1[x0] + ly * lx * r1[x1];
            }
        }
    }
    const uint obase = rest * (kCB * bins) + bin;
    #pragma unroll
    for (uint j = 0; j < kCB; ++j) out[obase + j * bins] = acc[j] / float(kR * kR);
}

// Adaptive grid: the sample count depends on the box, so the same thread walks
// ceil(roi/bin) samples per axis and divides by that product. No mask tensor and
// no (K, C, PH, PW, H, W) intermediate, unlike the composed form.
kernel void roi_align_adaptive_f32(
    device const float* x, device const float* rois, device float* out,
    constant uint& height, constant uint& width,
    constant float& scale, constant float& offset,
    constant uint& total,
    uint gid [[thread_position_in_grid]]) {
    if (gid >= total) return;
    const uint kC = $CH, kPH = $PH, kPW = $PW, kCB = $BLOCK;
    const uint bins = kPH * kPW;
    uint bin = gid % bins;
    uint rest = gid / bins;
    uint cb = rest % (kC / kCB);
    uint k = rest / (kC / kCB);
    uint ow = bin % kPW;
    uint oh = bin / kPW;

    device const float* roi = rois + k * 5u;
    uint batch = uint(roi[0]);
    float start_w = roi[1] * scale - offset;
    float start_h = roi[2] * scale - offset;
    float roi_w = roi[3] * scale - offset - start_w;
    float roi_h = roi[4] * scale - offset - start_h;
    if (offset == 0.0f) {
        roi_w = max(roi_w, 1.0f);
        roi_h = max(roi_h, 1.0f);
    }
    float bin_w = roi_w / float(kPW);
    float bin_h = roi_h / float(kPH);
    const uint obase = rest * (kCB * bins) + bin;
    if (bin_w <= 0.0f || bin_h <= 0.0f) {
        #pragma unroll
        for (uint j = 0; j < kCB; ++j) out[obase + j * bins] = 0.0f;
        return;
    }
    int gh = int(ceil(bin_h));
    int gw = int(ceil(bin_w));
    if (gh < 1 || gw < 1) {
        #pragma unroll
        for (uint j = 0; j < kCB; ++j) out[obase + j * bins] = 0.0f;
        return;
    }
    float step_w = bin_w / float(gw);
    float step_h = bin_h / float(gh);
    float origin_w = start_w + float(ow) * bin_w;
    float origin_h = start_h + float(oh) * bin_h;

    const size_t stride = (size_t)height * width;
    device const float* plane = x + ((size_t)batch * kC + cb * kCB) * stride;
    const int hmax = int(height) - 1;
    const int wmax = int(width) - 1;
    float acc[$BLOCK];
    #pragma unroll
    for (uint j = 0; j < kCB; ++j) acc[j] = 0.0f;
    for (int iy = 0; iy < gh; ++iy) {
        float y = origin_h + (float(iy) + 0.5f) * step_h;
        if (y < -1.0f || y > float(height)) continue;
        float yc = max(y, 0.0f);
        int y0 = int(yc);
        if (y0 >= hmax) { y0 = hmax; yc = float(y0); }
        float ly = yc - float(y0);
        float hy = 1.0f - ly;
        int y1 = min(y0 + 1, hmax);
        device const float* row0 = plane + (size_t)y0 * width;
        device const float* row1 = plane + (size_t)y1 * width;
        for (int ix = 0; ix < gw; ++ix) {
            float xs = origin_w + (float(ix) + 0.5f) * step_w;
            if (xs < -1.0f || xs > float(width)) continue;
            float xc = max(xs, 0.0f);
            int x0 = int(xc);
            if (x0 >= wmax) { x0 = wmax; xc = float(x0); }
            float lx = xc - float(x0);
            float hx = 1.0f - lx;
            int x1 = min(x0 + 1, wmax);
            #pragma unroll
            for (uint j = 0; j < kCB; ++j) {
                device const float* r0 = row0 + (size_t)j * stride;
                device const float* r1 = row1 + (size_t)j * stride;
                acc[j] += hy * hx * r0[x0] + hy * lx * r0[x1]
                        + ly * hx * r1[x0] + ly * lx * r1[x1];
            }
        }
    }
    #pragma unroll
    for (uint j = 0; j < kCB; ++j) out[obase + j * bins] = acc[j] / float(gh * gw);
}
"""

_BACKWARD_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// One thread per (image, channel block, row, column) of the input: the gradient
// is gathered, never scattered, so each output element has exactly one writer
// and the result does not depend on thread scheduling. Only the boxes of the
// thread's own image are scanned, and only the sampling grid cells that can
// reach the thread's pixel are visited: the samples are evenly spaced along each
// axis, so the cells worth testing are found by inverting the sample position
// instead of walking the whole grid.
kernel void roi_align_backward_f32(
    device const float* grad_out, device const float* rois,
    device const uint* order, device const uint* batch_offsets,
    device float* grad_in,
    constant uint& height, constant uint& width,
    constant float& scale, constant float& offset,
    constant uint& total,
    uint gid [[thread_position_in_grid]]) {
    if (gid >= total) return;
    const uint kCH = $CH, kPH = $PH, kPW = $PW, kCB = $BLOCK;
    const uint x = gid % width;
    uint rest = gid / width;
    uint y = rest % height;
    rest /= height;
    uint cblk = rest % (kCH / kCB);
    uint b = rest / (kCH / kCB);

    float fx = float(x), fy = float(y);
    float acc[$BLOCK];
    #pragma unroll
    for (uint j = 0; j < kCB; ++j) acc[j] = 0.0f;

    const uint bins = kPH * kPW;
    for (uint i = batch_offsets[b]; i < batch_offsets[b + 1]; ++i) {
        uint k = order[i];
        device const float* roi = rois + k * 5u;
        float start_w = roi[1] * scale - offset;
        float start_h = roi[2] * scale - offset;
        float roi_w = roi[3] * scale - offset - start_w;
        float roi_h = roi[4] * scale - offset - start_h;
        if (offset == 0.0f) {
            roi_w = max(roi_w, 1.0f);
            roi_h = max(roi_h, 1.0f);
        }
        float bin_w = roi_w / float(kPW);
        float bin_h = roi_h / float(kPH);
        // No sample of a box reaches further than one pixel past its bounds.
        if (fy < start_h - 1.0f || fy > start_h + roi_h + 1.0f
            || fx < start_w - 1.0f || fx > start_w + roi_w + 1.0f) continue;
        int gh = $RATIO, gw = $RATIO;
        if (gh <= 0) {
            gh = int(ceil(bin_h));
            gw = int(ceil(bin_w));
        }
        if (gh < 1 || gw < 1) continue;
        float step_w = bin_w / float(gw);
        float step_h = bin_h / float(gh);
        if (step_w <= 0.0f || step_h <= 0.0f) continue;

        // Rows whose bilinear taps can include y: y_s = start_h + (s + 0.5) *
        // step_h over the grid, so the useful s range is [y-1, y+1) inverted.
        int n_s = int(kPH) * gh;
        int sy_lo = 0, sy_hi = n_s - 1;
        float a = (fy - 1.0f - start_h) / step_h - 0.5f;
        float c = (fy + 1.0f - start_h) / step_h - 0.5f;
        sy_lo = max(0, min(sy_hi, int(floor(a)) - 2));
        sy_hi = max(sy_lo, min(sy_hi, int(ceil(c)) + 2));
        int n_t = int(kPW) * gw;
        int sx_lo = 0, sx_hi = n_t - 1;
        float d = (fx - 1.0f - start_w) / step_w - 0.5f;
        float e = (fx + 1.0f - start_w) / step_w - 0.5f;
        sx_lo = max(0, min(sx_hi, int(floor(d)) - 2));
        sx_hi = max(sx_lo, min(sx_hi, int(ceil(e)) + 2));

        device const float* g = grad_out + (size_t)k * kCH * bins;
        for (int oh = sy_lo / gh; oh <= sy_hi / gh; ++oh) {
            int iy_lo = (oh == sy_lo / gh) ? sy_lo - oh * gh : 0;
            int iy_hi = (oh == sy_hi / gh) ? sy_hi - oh * gh : gh - 1;
            for (int iy = iy_lo; iy <= iy_hi; ++iy) {
                float ys = start_h + float(oh) * bin_h + (float(iy) + 0.5f) * step_h;
                if (ys < -1.0f || ys > float(height)) continue;
                float yc = max(ys, 0.0f);
                int y0 = int(yc);
                if (y0 >= int(height) - 1) { y0 = int(height) - 1; yc = float(y0); }
                int y1 = min(y0 + 1, int(height) - 1);
                float ly = yc - float(y0);
                float wy = (y0 == int(y) ? 1.0f - ly : 0.0f) + (y1 == int(y) ? ly : 0.0f);
                if (wy == 0.0f) continue;
                for (int ow = sx_lo / gw; ow <= sx_hi / gw; ++ow) {
                    int ix_lo = (ow == sx_lo / gw) ? sx_lo - ow * gw : 0;
                    int ix_hi = (ow == sx_hi / gw) ? sx_hi - ow * gw : gw - 1;
                    for (int ix = ix_lo; ix <= ix_hi; ++ix) {
                        float xs = start_w + float(ow) * bin_w
                                 + (float(ix) + 0.5f) * step_w;
                        if (xs < -1.0f || xs > float(width)) continue;
                        float xc = max(xs, 0.0f);
                        int x0 = int(xc);
                        if (x0 >= int(width) - 1) { x0 = int(width) - 1; xc = float(x0); }
                        int x1 = min(x0 + 1, int(width) - 1);
                        float lx = xc - float(x0);
                        float wx = (x0 == int(x) ? 1.0f - lx : 0.0f)
                                 + (x1 == int(x) ? lx : 0.0f);
                        if (wx == 0.0f) continue;
                        float w = wy * wx / float(gh * gw);
                        device const float* gb = g + oh * kPW + ow;
                        uint gid_c = (uint)cblk * kCB;
                        #pragma unroll
                        for (uint j = 0; j < kCB; ++j) {
                            acc[j] += w * gb[(gid_c + j) * bins];
                        }
                    }
                }
            }
        }
    }

    const uint base = ((b * (kCH / kCB) + cblk) * height + y) * width + x;
    const size_t plane = (size_t)height * width;
    #pragma unroll
    for (uint j = 0; j < kCB; ++j) {
        grad_in[base + (size_t)j * plane] = acc[j];
    }
}
"""


@lru_cache(maxsize=16)
def _library(ratio, pooled_height, pooled_width, channels, block):
    """Compiled on first use; importing this module compiles nothing."""
    source = (
        _SOURCE.replace("$RATIO", str(ratio))
        .replace("$PH", str(pooled_height))
        .replace("$PW", str(pooled_width))
        .replace("$CH", str(channels))
        .replace("$BLOCK", str(block))
    )
    return torch.mps.compile_shader(source)


@lru_cache(maxsize=16)
def _backward_library(ratio, pooled_height, pooled_width, channels, block):
    """Compiled on first use; the forward's key and this one are kept apart."""
    source = (
        _BACKWARD_SOURCE.replace("$RATIO", str(ratio))
        .replace("$PH", str(pooled_height))
        .replace("$PW", str(pooled_width))
        .replace("$CH", str(channels))
        .replace("$BLOCK", str(block))
    )
    return torch.mps.compile_shader(source)


def _pair(value):
    if isinstance(value, int):
        return value, value
    return int(value[0]), int(value[1])


def _block(channels, units, extent, candidates=_BLOCKS):
    """Widest channel block that still fills the GPU: ``units * extent`` threads
    per channel block must stay above the launch-latency threshold."""
    for block in candidates:
        if channels % block == 0 and units * extent * (channels // block) >= _MIN_THREADS:
            return block
    return 1


def _convert_boxes(boxes):
    if isinstance(boxes, torch.Tensor):
        return boxes
    return torch.cat(
        [
            torch.cat(
                [torch.full((box.shape[0], 1), i, dtype=box.dtype, device=box.device), box],
                dim=1,
            )
            for i, box in enumerate(boxes)
        ]
    )


def _bilinear(input, roi_batch, y, x, ymask, xmask, compute):
    _, channels, height, width = input.shape
    valid_y = (y >= -1.0) & (y <= height)
    valid_x = (x >= -1.0) & (x <= width)
    y = y.clamp(min=0)
    x = x.clamp(min=0)
    y_low = y.int()
    x_low = x.int()
    y_high = torch.where(y_low >= height - 1, height - 1, y_low + 1)
    y_low = torch.where(y_low >= height - 1, height - 1, y_low)
    y = torch.where(y_low >= height - 1, y, y.to(compute))
    x_high = torch.where(x_low >= width - 1, width - 1, x_low + 1)
    x_low = torch.where(x_low >= width - 1, width - 1, x_low)
    x = torch.where(x_low >= width - 1, x, x.to(compute))

    ly = y - y_low
    lx = x - x_low
    hy = 1.0 - ly
    hx = 1.0 - lx
    batch = roi_batch[:, None, None, None, None, None]
    channel = torch.arange(channels, device=input.device)[None, :, None, None, None, None]

    def sample(yy, xx):
        if ymask is not None:
            yy = torch.where(ymask[:, None, :], yy, 0)
            xx = torch.where(xmask[:, None, :], xx, 0)
        return input[
            batch,
            channel,
            yy[:, None, :, None, :, None],
            xx[:, None, None, :, None, :],
        ]

    def weights(a, b):
        return a[:, None, :, None, :, None] * b[:, None, None, :, None, :]

    value = (
        weights(hy, hx) * sample(y_low, x_low)
        + weights(hy, lx) * sample(y_low, x_high)
        + weights(ly, hx) * sample(y_high, x_low)
        + weights(ly, lx) * sample(y_high, x_high)
    )
    valid = valid_y[:, None, :, None, :, None] & valid_x[:, None, None, :, None, :]
    return torch.where(valid, value, torch.zeros_like(value))


def _composed(input, rois, pooled_height, pooled_width, spatial_scale, sampling_ratio, aligned):
    """The fallback: torchvision's vectorized form, materializing (K,C,PH,PW,IY,IX).

    Half and bfloat16 inputs are interpolated in float32 and rounded back once at
    the end, which is what torchvision's kernel does; interpolating in the
    storage type instead costs up to a percent of the result.
    """
    dtype = input.dtype
    compute = torch.promote_types(dtype, torch.float32)
    height, width = input.shape[2:]
    ph = torch.arange(pooled_height, device=input.device)
    pw = torch.arange(pooled_width, device=input.device)
    batch = rois[:, 0].int()
    offset = 0.5 if aligned else 0.0
    start_w = rois[:, 1] * spatial_scale - offset
    start_h = rois[:, 2] * spatial_scale - offset
    end_w = rois[:, 3] * spatial_scale - offset
    end_h = rois[:, 4] * spatial_scale - offset
    roi_w = end_w - start_w
    roi_h = end_h - start_h
    if not aligned:
        roi_w = roi_w.clamp(min=1.0)
        roi_h = roi_h.clamp(min=1.0)
    bin_w = roi_w / pooled_width
    bin_h = roi_h / pooled_height
    exact = sampling_ratio > 0

    if exact:
        iy = torch.arange(sampling_ratio, device=input.device)
        ix = torch.arange(sampling_ratio, device=input.device)
        count = sampling_ratio * sampling_ratio
        grid_h = grid_w = sampling_ratio
        ymask = xmask = None
    else:
        grid_h = torch.ceil(roi_h / pooled_height)
        grid_w = torch.ceil(roi_w / pooled_width)
        count = torch.clamp(grid_h * grid_w, min=1)
        iy = torch.arange(height, device=input.device)
        ix = torch.arange(width, device=input.device)
        ymask = iy[None, :] < grid_h[:, None]
        xmask = ix[None, :] < grid_w[:, None]

    def column(value):
        return value[:, None, None]

    y = (
        column(start_h)
        + ph[None, :, None] * column(bin_h)
        + (iy[None, None, :] + 0.5).to(compute) * column(bin_h / grid_h)
    )
    x = (
        column(start_w)
        + pw[None, :, None] * column(bin_w)
        + (ix[None, None, :] + 0.5).to(compute) * column(bin_w / grid_w)
    )
    value = _bilinear(input, batch, y, x, ymask, xmask, compute)
    if not exact:
        value = torch.where(ymask[:, None, None, None, :, None], value, torch.zeros_like(value))
        value = torch.where(xmask[:, None, None, None, None, :], value, torch.zeros_like(value))
    value = value.sum((-1, -2))
    if isinstance(count, torch.Tensor):
        value = value / count[:, None, None, None]
    else:
        value = value / count
    return value.to(dtype)


def roi_align(input, boxes, output_size, spatial_scale=1.0, sampling_ratio=-1, aligned=False):
    if input.dim() != 4:
        raise ValueError(f"input must have shape (N,C,H,W), got {tuple(input.shape)}")
    if input.device.type == "mps" and input.dtype == torch.float64:
        raise TypeError("MPS does not support float64")
    output_size = _pair(output_size)
    rois = _convert_boxes(boxes).detach()
    if rois.dim() != 2 or rois.shape[1] != 5:
        raise ValueError(f"boxes must have shape (K,5), got {tuple(rois.shape)}")
    if output_size[0] <= 0 or output_size[1] <= 0:
        raise ValueError(f"output_size must be positive, got {output_size}")
    pooled_height, pooled_width = output_size
    _, channels, height, width = input.shape

    if (
        input.device.type == "mps"
        and input.dtype == torch.float32
        and rois.dtype == torch.float32
        and channels > 0
        and rois.shape[0] > 0
        and height > 0
        and width > 0
        and input.is_contiguous()
        and rois.is_contiguous()
    ):
        total = rois.shape[0] * channels * pooled_height * pooled_width
        if total < 1 << 31:
            return _metal(
                input, rois, pooled_height, pooled_width, sampling_ratio,
                spatial_scale, aligned,
            )

    return _composed(input, rois, pooled_height, pooled_width, spatial_scale, sampling_ratio, aligned)


class _RoIAlign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, rois, pooled_height, pooled_width, ratio, scale, offset):
        channels, height, width = input.shape[1:]
        output = torch.empty(
            rois.shape[0], channels, pooled_height, pooled_width,
            dtype=input.dtype, device=input.device,
        )
        bins = pooled_height * pooled_width
        block = _block(channels, rois.shape[0], bins)
        total = output.numel() // block
        if total:
            library = _library(abs(ratio), pooled_height, pooled_width, channels, block)
            name = "roi_align_f32" if ratio > 0 else "roi_align_adaptive_f32"
            getattr(library, name)(
                input, rois, output, height, width, float(scale), float(offset), total,
                threads=total, group_size=_GROUP,
            )
        ctx.save_for_backward(input, rois)
        ctx.params = (pooled_height, pooled_width, ratio, scale, offset)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, rois = ctx.saved_tensors
        pooled_height, pooled_width, ratio, scale, offset = ctx.params
        order, offsets = _batch_order(rois, input.shape[0])
        grad_input = _roi_align_backward(
            grad_output, input, rois, order, offsets, pooled_height,
            pooled_width, ratio, scale, offset,
        )
        return grad_input, None, None, None, None, None, None


def _metal(input, rois, pooled_height, pooled_width, ratio, scale, aligned):
    return _RoIAlign.apply(
        input, rois.contiguous(), pooled_height, pooled_width, int(ratio),
        float(scale), 0.5 if aligned else 0.0,
    )


def _batch_order(rois, batch_size):
    """RoI indices sorted by batch, plus the per-batch bucket boundaries.

    Only the backward needs this: a thread walks one image's boxes, so they have
    to be contiguous. The sort is stable, so the order is a function of the input
    alone and the accumulation order is reproducible.
    """
    batch = rois[:, 0].long()
    order = torch.argsort(batch, stable=True).to(torch.int32)
    offsets = torch.zeros(batch_size + 1, dtype=torch.int32, device=rois.device)
    offsets[1:] = torch.cumsum(torch.bincount(batch, minlength=batch_size), 0).to(torch.int32)
    return order, offsets


def _roi_align_backward(grad_output, input, rois, order, offsets, pooled_height,
                        pooled_width, ratio, scale, offset):
    """Gather the gradient: one thread per input element, one writer each.

    Boxes enter the sum in their stored order and each thread adds them in that
    order, so the result is independent of scheduling and of the batch grouping.
    """
    batch_size, channels, height, width = input.shape
    grad_input = torch.empty_like(input)
    block = _block(
        channels, batch_size, height * width, _BACKWARD_BLOCKS
    )
    total = grad_input.numel() // block
    if total == 0:
        return grad_input
    library = _backward_library(ratio, pooled_height, pooled_width, channels, block)
    library.roi_align_backward_f32(
        grad_output.contiguous(), rois, order, offsets, grad_input,
        height, width, float(scale), float(offset), total,
        threads=total, group_size=_GROUP,
    )
    return grad_input
