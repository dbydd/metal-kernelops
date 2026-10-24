"""MPS overlays for the transformer attention fast paths.

Six ``aten`` operators behind ``nn.MultiheadAttention`` and
``nn.TransformerEncoderLayer`` raise ``NotImplementedError`` on the MPS
dispatch key in torch 2.14.1:

    _masked_softmax, _masked_softmax_backward, _transform_bias_rescale_qkv,
    _native_multi_head_attention, _transformer_encoder_layer_fwd,
    _fused_sdp_choice

Every body below is written in aten ops MPS already runs and follows the CPU
kernel it replaces: ``aten/src/ATen/native/SoftMax.cpp`` for the two masked
softmax kernels, ``aten/src/ATen/native/transformers/attention.cpp`` for the
projection and the multi-head attention, ``.../transformers/transformer.cpp``
for the encoder layer. Mask layouts, ``mask_type`` values, error messages,
dtypes, strides of the projections and the empty-batch result follow that
reference; where the CPU kernel is undefined (an op without an autograd rule)
the overlay is undefined in the same place, because the overlay replaces only
the backend kernel and autograd still sits above the dispatch key.

One attention pipeline serves ``_native_multi_head_attention`` and, through it,
``_transformer_encoder_layer_fwd``:

* weights requested: explicit ``q @ k^T``, masked softmax, ``@ v``;
* no mask, no weights: ``F.scaled_dot_product_attention`` with ``scale=1.0``.
  ``q`` already carries ``1/sqrt(dim_per_head)`` from the projection, so the
  scaling is not applied twice. On MPS this is PyTorch's fused
  scaled-dot-product kernel, measured 1.4-2.6x faster than the explicit chain
  at 128-512 tokens with 64-wide heads (``attention_core`` in
  ``measured/attention_ops.json``);
* mask, no weights: the same fused kernel with the mask inverted, then a fill
  of every row the mask blocks completely. ``_masked_softmax`` returns NaN for
  an all-masked row while the fused kernel returns zero; the fill puts the
  reference value back.

Four of the six operators carry a Metal kernel of their own, and each one
folds the passes the aten composition needed around it:

* ``_masked_softmax`` and ``_masked_softmax_backward``: one pass per row, a
  simdgroup per row, no full-size intermediate, and the compact ``(L, L)`` and
  ``(B, L)`` masks read through zero strides instead of being expanded;
* ``_transform_bias_rescale_qkv``: one pass over the packed projection writing
  the three ``(batch, heads, length, head_dim)`` outputs;
* ``_native_multi_head_attention``: the split, the scores, the mask, the
  softmax and the context in one dispatch, straight into the layout the output
  projection reads;
* ``_transformer_encoder_layer_fwd``: two kernels around torch's matmuls, one
  per stage, each absorbing the residual, the bias and the activation that
  would otherwise be separate launches.

Every layout, device or dtype a kernel does not take falls back to the aten
composition, which is the definition of the arithmetic the kernels reproduce.

``_fused_sdp_choice`` reports the backend MPS dispatch reaches:
``SDPBackend::math``. ``aten::scaled_dot_product_attention`` has no MPS
kernel and ``_fused_sdp_choice_stub`` has no MPS registration, so the
composite in ``attention.cpp`` falls through to
``_scaled_dot_product_attention_math_for_mps``.

Importing this module compiles nothing and registers nothing. ``OVERLAYS`` is
the table the accelerator installs.
"""

import math
from functools import lru_cache

import torch
import torch.nn.functional as F


def _wrap_dim(dim, ndim):
    """``maybe_wrap_dim``: negative dims count from the end, 0-d tensors take -1 and 0."""
    if ndim == 0:
        if dim not in (0, -1):
            raise IndexError(
                f"Dimension out of range (expected to be in range of [-1, 0], but got {dim})"
            )
        return 0
    if dim < -ndim or dim >= ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], "
            f"but got {dim})"
        )
    return dim + ndim if dim < 0 else dim


def _mask_for_scores(shape, mask, mask_type, dim):
    """Validate ``mask`` against a score tensor of ``shape``, as the CPU kernel does.

    Types 0 and 1 are the two compact layouts and only exist for a 2-D mask over
    a 4-D score tensor; anything else is checked for an exact shape match. The
    compact layouts stay unexpanded here because ``masked_fill`` and the fused
    kernel broadcast them the same way.
    """
    if mask.dtype != torch.bool:
        raise RuntimeError("Mask should be a boolean tensor")
    kind = mask_type
    if mask.dim() != 2 or len(shape) != 4:
        kind = 2
    if kind == 2:
        if tuple(shape) != tuple(mask.shape):
            raise RuntimeError("For mask_type == 2 mask shape should match input shape")
    elif kind == 1:
        if shape[0] != mask.shape[0] or shape[2] != mask.shape[1]:
            raise RuntimeError("For mask_type == 1 mask shape should be (B, L)")
        if dim is None or dim != len(shape) - 1 or shape[2] == mask.shape[1]:
            mask = mask.reshape(mask.shape[0], 1, 1, mask.shape[1])
    elif kind == 0:
        if shape[2] != mask.shape[0] or shape[2] != mask.shape[1]:
            raise RuntimeError("For mask_type == 0 mask shape should be (L, L)")
        if dim is None or dim != len(shape) - 1:
            mask = mask.reshape(1, 1, mask.shape[0], mask.shape[1])
    if kind is None:
        raise RuntimeError("Mask Type should be defined")
    if kind not in (0, 1, 2):
        raise RuntimeError(
            "Mask Type should be 0 (src_mask) or 1 (src_key_padding_mask), or 2 (default_mask)"
        )
    return mask


_MS_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

static inline float ms_warp_max(float v) {
    for (ushort off = 16u; off > 0u; off >>= 1) v = fmax(v, simd_shuffle_xor(v, off));
    return v;
}

// One simdgroup (warp) per row of the softmax axis, four warps to a threadgroup
// of 128 threads. A lane holds up to $HELD values in registers, covering a row
// of up to 32 * $HELD = 512 elements, and every reduction is intra-warp, so the
// pass needs no threadgroup memory and no barrier.
//
// ``d1``/``d2`` and the four strides describe the mask view of the scores: an
// expanded ``(L, L)`` or ``(B, L)`` mask has a stride of zero on the dimensions
// it broadcasts, and that is what keeps the compact layouts compact instead of
// materializing them. A fully contiguous mask takes the linear branch.
template <typename T>
static inline void ms_forward_row(
    device const T* x, device const uchar* mask, device T* dst,
    constant uint& n, constant uint& rows,
    constant uint& d1, constant uint& d2,
    constant ulong& s0, constant ulong& s1, constant ulong& s2, constant ulong& scol,
    constant uint& is_contiguous_mask,
    uint tg, uint lid)
{
    uint lane = lid & 31u;
    uint warp = lid >> 5u;
    uint row = tg * 4u + warp;
    if (row >= rows) return;

    ulong mbase;
    ulong mcol;
    if (is_contiguous_mask != 0u) {
        mbase = (ulong)row * n;
        mcol = 1u;
    } else {
        uint i2 = row % d2;
        uint rest = row / d2;
        uint i1 = rest % d1;
        uint i0 = rest / d1;
        mbase = (ulong)i0 * s0 + (ulong)i1 * s1 + (ulong)i2 * s2;
        mcol = scol;
    }
    device const T* xr = x + (ulong)row * n;
    device T* dr = dst + (ulong)row * n;

    float held[$HELD];
    uchar keep[$HELD];
    float peak = -INFINITY;
    uint dropped = 0u;
    uint c = lane;
    for (uint k = 0u; k < $HELD; ++k, c += 32u) {
        if (c < n) {
            bool cut = mask[mbase + (ulong)c * mcol] != 0u;
            keep[k] = cut ? (uchar)0u : (uchar)1u;
            dropped += cut ? 1u : 0u;
            held[k] = cut ? 0.0f : float(xr[c]);
            if (!cut) peak = fmax(peak, held[k]);
        }
    }
    peak = ms_warp_max(peak);
    // An all-blocked row has no normalizer: the CPU kernel divides by an empty
    // sum and writes NaN, so this writes NaN too.
    if (simd_sum(dropped) >= n) {
        T nan = (T)as_type<float>(0x7fc00000u);
        c = lane;
        for (uint k = 0u; k < $HELD; ++k, c += 32u) {
            if (c < n) dr[c] = nan;
        }
        return;
    }
    float total = 0.0f;
    c = lane;
    for (uint k = 0u; k < $HELD; ++k, c += 32u) {
        if (c < n) {
            float e = keep[k] ? exp(held[k] - peak) : 0.0f;
            held[k] = e;
            total += e;
        }
    }
    total = simd_sum(total);
    float inv = 1.0f / total;
    c = lane;
    for (uint k = 0u; k < $HELD; ++k, c += 32u) {
        if (c < n) dr[c] = (T)(held[k] * inv);
    }
}

template <typename T>
static inline void ms_backward_row(
    device const T* grad, device const T* out, device const uchar* mask,
    device T* dst, constant uint& n, constant uint& rows,
    uint tg, uint lid)
{
    uint lane = lid & 31u;
    uint warp = lid >> 5u;
    uint row = tg * 4u + warp;
    if (row >= rows) return;

    ulong base = (ulong)row * n;
    float g[$HELD];
    float o[$HELD];
    uchar keep[$HELD];
    float total = 0.0f;
    uint c = lane;
    for (uint k = 0u; k < $HELD; ++k, c += 32u) {
        if (c < n) {
            bool cut = mask[base + c] != 0u;
            keep[k] = cut ? (uchar)0u : (uchar)1u;
            g[k] = float(grad[base + c]);
            o[k] = float(out[base + c]);
            if (!cut) total += g[k] * o[k];
        }
    }
    total = simd_sum(total);
    c = lane;
    for (uint k = 0u; k < $HELD; ++k, c += 32u) {
        if (c < n) {
            dst[base + c] = keep[k] ? (T)(o[k] * (g[k] - total)) : (T)0.0f;
        }
    }
}

#define MS_KERNELS(suffix, T) \
kernel void ms_forward_##suffix( \
    device const T* x, device const uchar* mask, device T* dst, \
    constant uint& n, constant uint& rows, \
    constant uint& d1, constant uint& d2, \
    constant ulong& s0, constant ulong& s1, constant ulong& s2, constant ulong& scol, \
    constant uint& is_contiguous_mask, \
    uint tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]) { \
    ms_forward_row<T>(x, mask, dst, n, rows, d1, d2, s0, s1, s2, scol, \
                      is_contiguous_mask, tg, lid); \
} \
kernel void ms_backward_##suffix( \
    device const T* grad, device const T* out, device const uchar* mask, device T* dst, \
    constant uint& n, constant uint& rows, \
    uint tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]) { \
    ms_backward_row<T>(grad, out, mask, dst, n, rows, tg, lid); \
}

MS_KERNELS(f32, float)
MS_KERNELS(f16, half)
MS_KERNELS(bf16, bfloat)
"""
"""One row per simdgroup; the mask stays in whatever layout it arrives in."""

_MS_TYPES = {
    torch.float32: ("ms_forward_f32", "ms_backward_f32"),
    torch.float16: ("ms_forward_f16", "ms_backward_f16"),
    torch.bfloat16: ("ms_forward_bf16", "ms_backward_bf16"),
}

_MS_HELD_MAX = 16
"""Values one lane holds at most: 32 lanes per warp cover rows up to 512 wide."""

_MS_WARPS_PER_GROUP = 4
"""Rows per threadgroup of 128 threads."""


@lru_cache(maxsize=8)
def _ms_library(held):
    """One shader per register count, compiled on first use; importing compiles nothing."""
    return torch.mps.compile_shader(_MS_SOURCE.replace("$HELD", str(held)))


def _ms_held(n):
    """Values per lane: the smallest power of two covering ``n`` in 32 lanes."""
    held = 1
    while held * 32 < n:
        held <<= 1
    return held


def _ms_forward_plan(self, mask, dim):
    """``(rows, n, threads, held, view, strides, d1, d2, linear)``, or None for the composite.

    Only the last axis takes the kernel: the row a simdgroup reduces has to be
    contiguous in memory, and the mask has to describe that same row. The
    register count follows the row length rather than a fixed maximum, because a
    lane holding more than the row needs costs more than it saves.
    """
    if self.device.type != "mps" or self.dtype not in _MS_TYPES:
        return None
    if dim != self.dim() - 1 or not self.is_contiguous():
        return None
    n = self.shape[-1]
    if n == 0 or self.numel() == 0 or n > 32 * _MS_HELD_MAX:
        return None
    try:
        view = mask.expand(self.shape)
    except RuntimeError:
        return None  # the composite raises the broadcast error the CPU kernel raises
    rows = self.numel() // n
    linear = 1 if (view.is_contiguous() and tuple(view.shape) == tuple(self.shape)) else 0
    strides = (n, 0, 0, 1)
    d1 = d2 = 1
    if not linear:
        if view.dim() != 4:
            return None
        if min(view.stride()) < 0:
            return None
        strides = view.stride()
        d1, d2 = self.shape[1], self.shape[2]
    threads = -(-rows // _MS_WARPS_PER_GROUP) * 128
    return rows, n, threads, _ms_held(n), view, strides, d1, d2, linear


def _ms_backward_plan(grad_output, output, mask, dim):
    """``(rows, n, threads, held)`` for the fused pass, or None for the composite.

    The backward mirrors the forward's layout: four rows per threadgroup, up to
    512 elements per row. The aten contract makes the mask the same shape as the
    gradient, so the strided forms the forward carries cannot arise here.
    """
    dtype = grad_output.dtype
    if grad_output.device.type != "mps" or dtype not in _MS_TYPES:
        return None
    if output.dtype != dtype or tuple(output.shape) != tuple(grad_output.shape):
        return None
    if dim != grad_output.dim() - 1 or grad_output.dim() == 0:
        return None
    n = grad_output.shape[-1]
    if n == 0 or grad_output.numel() == 0 or n > 32 * _MS_HELD_MAX:
        return None
    if not (grad_output.is_contiguous() and output.is_contiguous() and mask.is_contiguous()):
        return None
    rows = grad_output.numel() // n
    threads = -(-rows // _MS_WARPS_PER_GROUP) * 128
    return rows, n, threads, _ms_held(n)


def _masked_softmax(self, mask, dim=None, mask_type=None):
    """``softmax(masked_fill(self, mask, -inf), dim)``: True marks a dropped element.

    On the last axis the whole expression is one Metal kernel reading the scores
    and the mask view, with no full-size intermediate, and an all-blocked row is
    written as NaN there too. Every other layout keeps the aten composite.
    """
    mask = _mask_for_scores(self.shape, mask, mask_type, dim)
    dim = _wrap_dim(self.dim() - 1 if dim is None else dim, self.dim())
    if self.dim() == 0:
        # The CPU kernel views a 0-d input as length 1 before walking it.
        flat = self.reshape(1).masked_fill(mask.reshape(1), float("-inf"))
        return torch.softmax(flat, 0).reshape(self.shape)
    plan = _ms_forward_plan(self, mask, dim)
    if plan is not None:
        rows, n, threads, held, view, strides, d1, d2, linear = plan
        out = torch.empty_like(self)
        kernel = getattr(_ms_library(held), _MS_TYPES[self.dtype][0])
        kernel(
            self, view, out, n, rows, d1, d2, *strides, linear,
            threads=threads, group_size=128,
        )
        return out
    return torch.softmax(self.masked_fill(mask, float("-inf")), dim)


def _masked_softmax_backward(grad_output, output, mask, dim=None):
    """Softmax backward with dropped elements forced to zero.

    The reduction skips dropped elements rather than multiplying them by zero,
    which keeps a row whose entire mask is True on the CPU kernel's NaN path.
    One Metal kernel does the row reduction and the output in a single pass over
    the three inputs; other layouts keep the aten composite.
    """
    if grad_output.shape != mask.shape:
        raise RuntimeError("Mask shape should match grad shape")
    if mask.dtype != torch.bool:
        raise RuntimeError("Mask should be a boolean tensor")
    dim = _wrap_dim(output.dim() - 1 if dim is None else dim, grad_output.dim())
    if grad_output.dim() == 0:
        grad_output = grad_output.reshape(1)
        output = output.reshape(1)
        mask = mask.reshape(1)
    plan = _ms_backward_plan(grad_output, output, mask, dim)
    if plan is not None:
        rows, n, threads, held = plan
        grad_input = torch.empty_like(grad_output)
        kernel = getattr(_ms_library(held), _MS_TYPES[grad_output.dtype][1])
        kernel(
            grad_output, output, mask, grad_input, n, rows,
            threads=threads, group_size=128,
        )
        return grad_input
    total = (grad_output * output).masked_fill(mask, 0.0).sum(dim, keepdim=True)
    return (output * (grad_output - total)).masked_fill(mask, 0.0)


_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// One thread per four head-dim elements, or per element in the scalar kernels.
// The grid is (elements of one batch-head slice, batch * head), so every lane
// of a threadgroup shares the batch-head index and the per-lane integer work is
// one division of the slice index by the head width. A lane reads its three
// qkv values and three biases and writes its three results: one pass over qkv,
// one over q, k and v, already in the layout the caller asks for.
//
// ``scale`` is 1 / sqrt(head_dim) rounded to float32, computed by the caller.
// Over head widths 1..64 and 96/128 that multiply reproduces the CPU kernel bit
// for bit except at head widths 6 and 7, where the CPU result follows neither
// the reciprocal product nor the quotient and one float32 ulp separates them.

kernel void split4_f32(
    device const float* qkv, device const float* bias,
    device float* q, device float* k, device float* v,
    constant uint& H, constant uint& L, constant uint& D4, constant float& scale,
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= L * D4) return;
    uint d4 = gid.x % D4;
    uint l = gid.x / D4;
    uint h = gid.y % H;
    uint b = gid.y / H;
    uint D = D4 * 4u;
    uint stride = H * D;
    uint hd = h * D + d4 * 4u;
    size_t src = (size_t)(b * L + l) * 3u * stride + hd;
    size_t dst = ((size_t)gid.y * L + l) * D + d4 * 4u;
    *(device float4*)(q + dst) = (*(device const float4*)(qkv + src)
                                  + *(device const float4*)(bias + hd)) * scale;
    *(device float4*)(k + dst) = *(device const float4*)(qkv + src + stride)
                                 + *(device const float4*)(bias + hd + stride);
    *(device float4*)(v + dst) = *(device const float4*)(qkv + src + 2u * stride)
                                 + *(device const float4*)(bias + hd + 2u * stride);
}

// The half kernels convert to float, add, rescale and round once. Adding and
// rescaling in half precision rounds twice and lands one ulp off, and that ulp
// grows through a softmax.

kernel void split4_f16(
    device const half* qkv, device const half* bias,
    device half* q, device half* k, device half* v,
    constant uint& H, constant uint& L, constant uint& D4, constant float& scale,
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= L * D4) return;
    uint d4 = gid.x % D4;
    uint l = gid.x / D4;
    uint h = gid.y % H;
    uint b = gid.y / H;
    uint D = D4 * 4u;
    uint stride = H * D;
    uint hd = h * D + d4 * 4u;
    size_t src = (size_t)(b * L + l) * 3u * stride + hd;
    size_t dst = ((size_t)gid.y * L + l) * D + d4 * 4u;
    float4 a = float4(*(device const half4*)(qkv + src));
    float4 c = float4(*(device const half4*)(qkv + src + stride));
    float4 e = float4(*(device const half4*)(qkv + src + 2u * stride));
    float4 bq = float4(*(device const half4*)(bias + hd));
    float4 bk = float4(*(device const half4*)(bias + hd + stride));
    float4 bv = float4(*(device const half4*)(bias + hd + 2u * stride));
    *(device half4*)(q + dst) = half4((a + bq) * scale);
    *(device half4*)(k + dst) = half4(c + bk);
    *(device half4*)(v + dst) = half4(e + bv);
}

kernel void split1_f32(
    device const float* qkv, device const float* bias,
    device float* q, device float* k, device float* v,
    constant uint& H, constant uint& L, constant uint& D, constant float& scale,
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= L * D) return;
    uint d = gid.x % D;
    uint l = gid.x / D;
    uint h = gid.y % H;
    uint b = gid.y / H;
    uint stride = H * D;
    uint hd = h * D + d;
    size_t src = (size_t)(b * L + l) * 3u * stride + hd;
    size_t dst = ((size_t)gid.y * L + l) * D + d;
    q[dst] = (qkv[src] + bias[hd]) * scale;
    k[dst] = qkv[src + stride] + bias[hd + stride];
    v[dst] = qkv[src + 2u * stride] + bias[hd + 2u * stride];
}

kernel void split1_f16(
    device const half* qkv, device const half* bias,
    device half* q, device half* k, device half* v,
    constant uint& H, constant uint& L, constant uint& D, constant float& scale,
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= L * D) return;
    uint d = gid.x % D;
    uint l = gid.x / D;
    uint h = gid.y % H;
    uint b = gid.y / H;
    uint stride = H * D;
    uint hd = h * D + d;
    size_t src = (size_t)(b * L + l) * 3u * stride + hd;
    size_t dst = ((size_t)gid.y * L + l) * D + d;
    q[dst] = half((float(qkv[src]) + float(bias[hd])) * scale);
    k[dst] = half(float(qkv[src + stride]) + float(bias[hd + stride]));
    v[dst] = half(float(qkv[src + 2u * stride]) + float(bias[hd + 2u * stride]));
}
"""

_SPLIT = {
    (4, torch.float32): "split4_f32",
    (4, torch.float16): "split4_f16",
    (1, torch.float32): "split1_f32",
    (1, torch.float16): "split1_f16",
}


@lru_cache(maxsize=1)
def _library():
    """Compiled on first use; importing this module compiles nothing."""
    return torch.mps.compile_shader(_SOURCE)


def _split_fused(qkv, bias, num_heads, head_dim):
    """One launch: ``qkv`` and the bias read once, ``q``, ``k`` and ``v`` written directly."""
    batch, length, _ = qkv.shape
    shape = (batch, num_heads, length, head_dim)
    q = torch.empty(shape, dtype=qkv.dtype, device=qkv.device)
    k = torch.empty(shape, dtype=qkv.dtype, device=qkv.device)
    v = torch.empty(shape, dtype=qkv.dtype, device=qkv.device)
    width = 4 if head_dim % 4 == 0 else 1
    name = _SPLIT[width, qkv.dtype]
    getattr(_library(), name)(
        qkv, bias, q, k, v, num_heads, length, head_dim // width,
        1.0 / math.sqrt(head_dim),
        threads=(length * head_dim // width, batch * num_heads), group_size=(32, 1),
    )
    return q, k, v


def _transform_bias_rescale_qkv(qkv, qkv_bias, num_heads):
    """Split ``(batch, length, 3 * embed)`` into ``q``, ``k``, ``v`` of ``(batch, heads, length, head_dim)``.

    ``q`` is rescaled by ``1/sqrt(head_dim)`` so the score matmul needs no
    scaling; ``k`` and ``v`` only take the bias.

    ``q``'s bias add and rescale run in float32 for the half dtypes and round
    once, which is what the CPU kernel produces; doing both in half precision
    rounds twice and lands one ulp off, and that ulp grows through a softmax.

    On the device one kernel does the whole split: it reads ``qkv`` and the bias
    once and writes the three ``(batch, heads, length, head_dim)`` outputs, in
    place of four elementwise passes (add ``q``, rescale ``q`` in place, add
    ``k``, add ``v``) that each walk a permuted view of the input.
    ``transform_bias_rescale_qkv`` in ``measured/attention_ops.json`` times both
    against each other and against a ``copy_`` of the same bytes: with the queue
    kept full the kernel is several times the four passes, and one call is
    dominated by the launch and the device round trip either way.

    Anything the shader does not take -- another device, bfloat16, a
    non-contiguous input, more elements than the grid counts in 32 bits -- falls
    back to the decomposition below, which produces the same values.
    """
    batch, length, triple = qkv.shape
    embed = triple // 3
    if embed % num_heads:
        raise RuntimeError(f"embedding dimension {embed} is not divisible by {num_heads} heads")
    if triple % 3:
        raise RuntimeError(f"expected 3 * num_embedding to be divisible by 3, got {triple}")
    head_dim = embed // num_heads
    bias = qkv_bias.to(qkv.dtype).reshape(3, num_heads, 1, head_dim)
    if (
        qkv.device.type == "mps"
        and qkv.dtype in (torch.float32, torch.float16)
        and qkv.is_contiguous()
        and bias.is_contiguous()
        and 0 < qkv.numel() < 2**31
        and length * head_dim < 2**31
    ):
        return _split_fused(qkv, bias, num_heads, head_dim)
    packed = qkv.reshape(batch, length, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    wide = torch.float32 if qkv.dtype in (torch.float16, torch.bfloat16) else qkv.dtype
    # Add into a contiguous buffer and rescale in place: one pass, and the
    # result keeps the CPU kernel's (batch, heads, length, head_dim) layout.
    q = torch.add(packed[0].to(wide), bias[0].to(wide)).contiguous()
    q.mul_(1.0 / math.sqrt(head_dim))
    return (
        q.to(qkv.dtype),
        torch.add(packed[1], bias[1]).contiguous(),
        torch.add(packed[2], bias[2]).contiguous(),
    )


def _attention(q, k, v, mask, mask_type, need_weights, average_attn_weights, num_heads):
    """The shared pipeline: ``q``, ``k``, ``v`` are ``(batch, heads, length, head_dim)``.

    Returns ``(context, weights)`` with ``weights`` undefined when not requested.
    """
    if need_weights:
        scores = torch.matmul(q, k.transpose(-2, -1))
        if mask is None:
            weights = torch.softmax(scores, -1)
        else:
            weights = _masked_softmax(scores, mask, None, mask_type)
        context = torch.matmul(weights, v)
        if average_attn_weights:
            weights = weights.sum(1) / num_heads
        return context, weights

    if mask is None:
        return F.scaled_dot_product_attention(q, k, v, scale=1.0), None

    shape = (q.shape[0], q.shape[1], q.shape[2], k.shape[2])
    mask = _mask_for_scores(shape, mask, mask_type, None)
    context = F.scaled_dot_product_attention(q, k, v, attn_mask=~mask, scale=1.0)
    blocked = mask.all(-1, keepdim=True)
    blocked = blocked.reshape((1,) * (4 - blocked.dim()) + blocked.shape)
    return context.masked_fill(blocked, float("nan")), None


def _project_qkv(query, key, value, embed_dim, qkv_weight):
    """``(batch, length, 3 * embed)`` from one matmul for self-attention, three otherwise."""
    if query.data_ptr() == key.data_ptr() == value.data_ptr():
        return torch.matmul(query, qkv_weight.t())
    wq, wk, wv = qkv_weight.split(embed_dim, 0)
    return torch.cat(
        (
            torch.matmul(query, wq.t()),
            torch.matmul(key, wk.t()),
            torch.matmul(value, wv.t()),
        ),
        dim=2,
    )


_MHA_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

#define DIM $DIM
#define ROWS $ROWS
#define SPAN $SPAN
#define NJS (SPAN / 32)
#define ACCS ((DIM + 31) / 32)
#define TD $T

// One threadgroup owns ``ROWS`` query rows of one (batch, head) pair, and one
// simdgroup owns a single row: lane ``j`` holds the score against key ``j``, so
// the row's maximum and its normalizer are simd reductions and the score matrix
// never reaches memory. Phase 3 turns those weights into the context and writes
// it straight into the (batch, length, embed) layout the output projection
// reads, which is what removes the per-call transposes from the operator.
kernel void nmha_core(
    device const TD* qkv [[buffer(0)]],
    device const TD* bias [[buffer(1)]],
    device const uchar* mask [[buffer(2)]],
    device TD* weights [[buffer(3)]],
    device TD* out [[buffer(4)]],
    constant uint& L [[buffer(5)]],
    constant uint& E [[buffer(6)]],
    constant uint& H [[buffer(7)]],
    constant uint& tiles [[buffer(8)]],
    constant uint& mask_kind [[buffer(9)]],
    constant uint& want_weights [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    threadgroup float P[ROWS * SPAN];
    threadgroup float LSUM[ROWS];

    const float QNAN = as_type<float>(0x7fc00000u);
    const uint lane = lid & 31u;
    const uint row = lid >> 5;
    const uint LI = uint(L);
    const long stride = 3L * long(E);
    const uint b = (tgid / tiles) / H;
    const uint h = (tgid / tiles) - b * H;
    const uint qr = (tgid % tiles) * ROWS + row;
    const uint qc = min(qr, LI - 1u);

    device const TD* qrow = qkv + (long(b) * long(LI) + long(qc)) * stride + long(h) * DIM;
    device const TD* kbase = qkv + long(b) * long(LI) * stride + long(E) + long(h) * DIM;
    device const TD* vbase = qkv + long(b) * long(LI) * stride + 2L * long(E) + long(h) * DIM;
    device const TD* qbias = bias + long(h) * DIM;
    device const TD* kbias = bias + long(E) + long(h) * DIM;
    device const TD* vbias = bias + 2L * long(E) + long(h) * DIM;

    long mbase = 0;
    if (mask_kind == 0u) {
        mbase = long(qc) * long(LI);
    } else if (mask_kind == 1u) {
        mbase = long(b) * long(LI);
    } else if (mask_kind == 2u) {
        mbase = (long(b) * long(H) + long(h)) * long(LI) * long(LI) + long(qc) * long(LI);
    }

    // Scores straight from the packed projection: the split, the bias add, the
    // 1/sqrt(head_dim) rescale and the mask all fold into this dot product.
    float localmax = -INFINITY;
    for (int t = 0; t < NJS; ++t) {
        int j = int(lane) + 32 * t;
        float s = -INFINITY;
        if (j < int(LI)) {
            device const TD* krow = kbase + long(j) * stride;
            float acc = 0.0f;
            for (int d = 0; d < DIM; ++d) {
                float qv = float(TD((float(qrow[d]) + float(qbias[d])) * scale));
                float kv = float(TD(float(krow[d]) + float(kbias[d])));
                acc = fma(qv, kv, acc);
            }
            // ATen's half path stores the scores, the exponentials and the
            // normalizer in the element type before it uses them; half-float
            // arithmetic in this kernel has to round at the same places or the
            // softmax moves by a percent.
            s = float(TD(acc));
            if (mask_kind != 3u && mask[mbase + long(j)] != uchar(0)) {
                s = -INFINITY;
            }
        }
        P[row * SPAN + uint(j)] = s;
        localmax = fmax(localmax, s);
    }

    float m = simd_max(localmax);
    float localsum = 0.0f;
    for (int t = 0; t < NJS; ++t) {
        int j = int(lane) + 32 * t;
        float p = 0.0f;
        if (j < int(LI) && m > -INFINITY) {
            p = float(TD(exp(P[row * SPAN + uint(j)] - m)));
        }
        P[row * SPAN + uint(j)] = p;
        localsum += p;
    }
    float lsum = float(TD(simd_sum(localsum)));
    if (lane == 0u) {
        LSUM[row] = lsum;
    }

    // An all-blocked row has no normalizer: the CPU kernel divides by an empty
    // sum, and every value of that row is NaN.
    if (want_weights != 0u && qr < LI) {
        float inv = lsum > 0.0f ? 1.0f / lsum : QNAN;
        device TD* wrow = weights + (long(b) * long(H) + long(h)) * long(LI) * long(LI)
                          + long(qr) * long(LI);
        for (int t = 0; t < NJS; ++t) {
            int j = int(lane) + 32 * t;
            if (j < int(LI)) {
                wrow[j] = TD(P[row * SPAN + uint(j)] * inv);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device TD* orow_base =
        out + (long(b) * long(LI) + long((tgid % tiles) * ROWS)) * long(E);
    for (int k = 0; k < ACCS; ++k) {
        int w = int(lid) + k * (ROWS * 32);
        if (w < ROWS * DIM) {
            int r = w / DIM;
            int d = w - r * DIM;
            uint orow = (tgid % tiles) * ROWS + uint(r);
            if (orow < LI) {
                float acc = 0.0f;
                for (int j = 0; j < int(LI); ++j) {
                    float vv = float(TD(float(vbase[long(j) * stride + d]) + float(vbias[d])));
                    acc = fma(float(TD(P[uint(r) * SPAN + uint(j)])), vv, acc);
                }
                float ls = LSUM[r];
                float inv = ls > 0.0f ? 1.0f / ls : QNAN;
                orow_base[long(r) * long(E) + long(h) * DIM + long(d)] = TD(acc * inv);
            }
        }
    }
}
"""
"""The fused attention core: projection split, scores, mask, softmax and context.

``$DIM`` is the head width, ``$ROWS`` the query rows per threadgroup, ``$SPAN``
the row length rounded up to a whole number of simdgroups, and ``$T`` the
element type. One dispatch replaces the split, the score matmul, the masked
softmax, the weighted-value matmul, the merge transpose and the all-blocked-row
fill, and writes the result in the layout the output projection consumes.
"""

_MHA_TYPES = {
    torch.float32: "float",
    torch.float16: "half",
    torch.bfloat16: "bfloat",
}

_MHA_PLANE = 16384
"""Threadgroup memory the score plane may take, in bytes (the budget is 32 KiB)."""

_MHA_ROWS = 8
"""Query rows per threadgroup; one simdgroup each."""

_MHA_KERNELS = {}
"""Compiled kernels, keyed by element type and shape. Import compiles nothing."""


def _mha_plan(dtype, head_dim, length):
    """``(kernel, rows, span)`` for this shape, or None when the kernel cannot run.

    Rows per threadgroup fall as the row grows so the score plane stays inside
    ``_MHA_PLANE``; a length past 4096 needs more than that and takes the aten
    path instead. The dtype table is the set of element types the shader has a
    form for.
    """
    name = _MHA_TYPES.get(dtype)
    if name is None:
        return None
    span = (length + 31) // 32 * 32
    rows = min(_MHA_ROWS, _MHA_PLANE // (span * 4))
    if rows < 1:
        return None
    key = (name, head_dim, rows, span)
    entry = _MHA_KERNELS.get(key)
    if entry is None:
        source = (
            _MHA_SOURCE.replace("$DIM", str(head_dim))
            .replace("$ROWS", str(rows))
            .replace("$SPAN", str(span))
            .replace("$T", name)
        )
        library = torch.mps.compile_shader(source)
        entry = (library, getattr(library, "nmha_core"))
        _MHA_KERNELS[key] = entry
    return entry[1], rows, span


_MHA_SPARE = {}


def _mha_spare(dtype, device):
    """A one-element buffer standing in for a kernel argument this call ignores."""
    key = (dtype, device)
    spare = _MHA_SPARE.get(key)
    if spare is None:
        spare = torch.empty(1, dtype=dtype, device=device)
        _MHA_SPARE[key] = spare
    return spare


def _native_multi_head_attention(
    query,
    key,
    value,
    embed_dim,
    num_head,
    qkv_weight,
    qkv_bias,
    proj_weight,
    proj_bias,
    mask=None,
    need_weights=True,
    average_attn_weights=True,
    mask_type=None,
):
    """Self-, cross- or encoder-decoder attention over a packed qkv projection."""
    if query.dim() != 3:
        raise RuntimeError(f"expected 3-D `query`, got {query.dim()}-D tensor")
    if query.shape[2] != embed_dim:
        raise RuntimeError(
            f"passed-in embed_dim {embed_dim} didn't match last dim of query {query.shape[2]}"
        )
    if key.dim() != 3:
        raise RuntimeError(f"expected 3-D `key`, got {key.dim()}-D tensor")
    if value.dim() != 3:
        raise RuntimeError(f"expected 3-D `value`, got {value.dim()}-D tensor")
    if query.shape != key.shape or key.shape != value.shape:
        raise RuntimeError("expected `query`/`key`/`value` shapes to match")
    if qkv_weight.dim() != 2:
        raise RuntimeError(f"expected 2-D `qkv_weight`, got {qkv_weight.dim()}-D tensor")
    if qkv_weight.shape[0] != 3 * embed_dim:
        raise RuntimeError("expected `qkv_weight` first dim to be 3x embed_dim")
    if qkv_weight.shape[1] != embed_dim:
        raise RuntimeError("expected `qkv_weight` second dim to be embed_Dim")
    if qkv_bias.dim() != 1:
        raise RuntimeError(f"expected 1-D `qkv_bias`, got {qkv_bias.dim()}-D tensor")
    if qkv_bias.shape[0] != 3 * embed_dim:
        raise RuntimeError("expected `qkv_bias` first dim and first dim of query to be equal")
    if embed_dim % num_head:
        raise RuntimeError("`embed_dim` must divide evenly by `num_heads`")

    qkv = _project_qkv(query, key, value, embed_dim, qkv_weight)
    if qkv.numel() == 0:
        return torch.empty_like(query), None
    if mask is not None and mask.dtype != torch.bool:
        mask = mask.to(torch.bool)

    batch, length, _ = qkv.shape
    head_dim = embed_dim // num_head
    plan = _mha_plan(qkv.dtype, head_dim, length)
    if plan is None:
        # A row past the score plane's budget: the aten composition still holds.
        q, k, v = _transform_bias_rescale_qkv(qkv, qkv_bias, num_head)
        context, weights = _attention(
            q, k, v, mask, mask_type, need_weights, average_attn_weights, num_head
        )
        merged = context.permute(0, 2, 1, 3).reshape(batch, length, embed_dim)
        return F.linear(merged, proj_weight, proj_bias), weights

    kernel, rows, _ = plan
    if mask is None:
        kind, mask_arg = 3, _mha_spare(torch.uint8, qkv.device)
    else:
        # The compact layouts only exist for a 2-D mask over 4-D scores, which is
        # the rule ``_mask_for_scores`` applies below.
        kind = 2 if mask.dim() != 2 else mask_type
        mask = _mask_for_scores((batch, num_head, length, length), mask, mask_type, None)
        mask_arg = mask.contiguous().view(torch.uint8)
    bias = qkv_bias.to(qkv.dtype)
    if not bias.is_contiguous():
        bias = bias.contiguous()

    merged = torch.empty(batch, length, embed_dim, dtype=qkv.dtype, device=qkv.device)
    if need_weights:
        weights = torch.empty(
            batch, num_head, length, length, dtype=qkv.dtype, device=qkv.device
        )
    else:
        weights = _mha_spare(qkv.dtype, qkv.device)
    tiles = (length + rows - 1) // rows
    kernel(
        qkv,
        bias,
        mask_arg,
        weights,
        merged,
        length,
        embed_dim,
        num_head,
        tiles,
        kind,
        1 if need_weights else 0,
        1.0 / math.sqrt(head_dim),
        threads=batch * num_head * tiles * rows * 32,
        group_size=rows * 32,
    )
    out = F.linear(merged, proj_weight, proj_bias)
    if need_weights:
        if average_attn_weights:
            weights = weights.sum(1) / num_head
        return out, weights
    return out, None


_ENCODER_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// One launch per stage of a TransformerEncoderLayer. Each kernel fuses the
// elementwise work that the composed path gives its own launch: the layer norm
// absorbs the residual add in front of it, and each projection absorbs its
// bias, the q/k/v split with the 1/sqrt(head_dim) scale, GELU/ReLU, or the
// residual add behind it.
//
// Arithmetic keeps the composed path's rounding points where a half-precision
// result can move: the projection is rounded to the storage dtype before the
// bias joins it (``matmul`` then ``add`` there), and the q bias is added and
// scaled in float32, which is what ``_transform_bias_rescale_qkv`` does.

template <typename T>
static inline float enc_load(const device T* p, long i) {
    return static_cast<float>(p[i]);
}

template <typename T>
static inline void enc_store(device T* p, long i, float v) {
    p[i] = static_cast<T>(v);
}

// One threadgroup per row: mean, then the centered second moment, then the
// affine write. ``DEPTH`` values per thread cover the row without a second read.
// The residual add rounds to the storage dtype before the reduction, because
// the composed path adds two dtype tensors and normalizes the rounded sum.
// ``pre``, when not null, receives that added value, which is what the residual
// behind the feed-forward layer adds back, so the add needs no pass of its own.
template <typename T, uint DEPTH>
static inline void enc_norm_body(
    device const T* x, device const T* add, device const T* w, device const T* b,
    device T* y, device T* pre, long base, constant uint& cols, constant float& eps,
    constant uint& has_add, constant uint& has_pre, uint lid, uint tgs,
    threadgroup float* scratch)
{
    float v[DEPTH];
    uint n = 0;
    float sum = 0.0f;
    for (uint i = lid; i < cols; i += tgs, ++n) {
        float xv = enc_load(x, base + i);
        if (has_add != 0u) {
            xv = static_cast<float>(static_cast<T>(xv + enc_load(add, base + i)));
        }
        if (has_pre != 0u) enc_store(pre, base + i, xv);
        v[n] = xv;
        sum += xv;
    }
    uint lane = lid & 31u;
    uint warp = lid >> 5u;
    uint warps = (tgs + 31u) >> 5u;
    sum = simd_sum(sum);
    if (lane == 0u) scratch[warp] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint j = 0; j < warps; ++j) total += scratch[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = total / static_cast<float>(cols);
    float sq = 0.0f;
    for (uint j = 0; j < n; ++j) {
        float d = v[j] - mean;
        sq += d * d;
    }
    sq = simd_sum(sq);
    if (lane == 0u) scratch[warp] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = 0.0f;
    for (uint j = 0; j < warps; ++j) total += scratch[j];
    float rstd = rsqrt(total / static_cast<float>(cols) + eps);
    n = 0;
    for (uint i = lid; i < cols; i += tgs, ++n) {
        enc_store(y, base + i, (v[n] - mean) * rstd * enc_load(w, i) + enc_load(b, i));
    }
}

// Keys per staged tile: 2048 floats, 8 KiB, whatever the head width.
#define ENC_KC(D) ((2048u / (D)) > 8u ? (2048u / (D)) : 8u)

// Attention over the packed ``(rows, 3 * embed)`` projection, one threadgroup
// per (batch, head) and one thread per query row.
//
// ``q``, ``k`` and ``v`` are columns of the projection, so the split that
// ``_transform_bias_rescale_qkv`` materialises never exists, and the context is
// written straight into the ``(rows, embed)`` layout the output projection
// reads, so the ``permute`` + ``reshape`` copy never exists either. The bias is
// added here in the same order and at the same rounding points as that
// function: q in float32 then scaled and rounded, k and v added in the storage
// dtype. The online softmax keeps the running maximum, the running normaliser
// and the accumulator in registers; a row the mask blocks completely ends at
// ``0 / 0``, which is the NaN ``_masked_softmax`` produces.
template <typename T, uint D>
static inline void enc_attn_body(
    device const T* qkv, device const T* bias, device const uchar* mask,
    device T* ctx, constant uint& length, constant uint& embed,
    constant uint& stride_mb, constant uint& stride_mh, constant uint& stride_ml,
    constant uint& stride_mj, constant float& scale, constant uint& has_mask,
    threadgroup float* Ks, threadgroup float* Vs, uint bh, uint lid, uint tgs)
{
    constexpr uint KC = ENC_KC(D);
    uint heads = embed / D;
    uint b = bh / heads;
    uint h = bh - b * heads;
    uint col = h * D;
    device const T* kbias = bias + embed + col;
    device const T* vbias = bias + 2u * embed + col;

    // Every thread walks the same number of query blocks so that the two
    // barriers per key chunk are reached by the whole threadgroup; a thread
    // whose row is past the end still stages tiles and skips its arithmetic.
    // Its query row is clamped, so no launch reads past the projection.
    uint blocks = (length + tgs - 1u) / tgs;
    for (uint bi = 0; bi < blocks; ++bi) {
        uint l = bi * tgs + lid;
        bool active = l < length;
        long row = (long)b * length + (long)min(l, length - 1u);
        device const T* qrow = qkv + row * (3 * (long)embed) + col;
        float q[D];
        for (uint d = 0; d < D; ++d) {
            float qv = (enc_load(qrow, d) + enc_load(bias, col + d)) * scale;
            q[d] = static_cast<float>(static_cast<T>(qv));
        }

        float m = -INFINITY;
        float lsum = 0.0f;
        float acc[D];
        for (uint d = 0; d < D; ++d) acc[d] = 0.0f;

        for (uint k0 = 0; k0 < length; k0 += KC) {
            uint nk = length - k0 < KC ? length - k0 : KC;
            for (uint i = lid; i < nk * D; i += tgs) {
                uint j = i / D;
                uint d = i - j * D;
                long krow = (long)b * length + k0 + j;
                Ks[i] = static_cast<float>(static_cast<T>(
                    enc_load(qkv, krow * (3 * (long)embed) + embed + col + d)
                    + enc_load(kbias, d)));
                Vs[i] = static_cast<float>(static_cast<T>(
                    enc_load(qkv, krow * (3 * (long)embed) + 2u * embed + col + d)
                    + enc_load(vbias, d)));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (active) {
                for (uint j = 0; j < nk; ++j) {
                    if (has_mask != 0u) {
                        long mi = (long)b * stride_mb + (long)h * stride_mh
                                + (long)l * stride_ml + (long)(k0 + j) * stride_mj;
                        if (mask[mi] != 0u) continue;
                    }
                    float s = 0.0f;
                    for (uint d = 0; d < D; ++d) s += q[d] * Ks[j * D + d];
                    s = static_cast<float>(static_cast<T>(s));
                    float m_new = max(m, s);
                    float corr = exp(m - m_new);
                    float p = exp(s - m_new);
                    lsum = lsum * corr + p;
                    float ph = static_cast<float>(static_cast<T>(p));
                    for (uint d = 0; d < D; ++d) {
                        acc[d] = acc[d] * corr + ph * Vs[j * D + d];
                    }
                    m = m_new;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (active) {
            device T* dst = ctx + row * (long)embed + col;
            for (uint d = 0; d < D; ++d) dst[d] = static_cast<T>(acc[d] / lsum);
        }
    }
}

#define DECLARE_ENC_NORM(NAME, T, D)                                                          \
kernel void NAME(device const T* x, device const T* add, device const T* w,                    \
                 device const T* b, device T* y, device T* pre, constant uint& cols,            \
                 constant float& eps, constant uint& has_add, constant uint& has_pre,           \
                 uint lid [[thread_position_in_threadgroup]],                                  \
                 uint tg [[threadgroup_position_in_grid]],                                     \
                 uint tgs [[threads_per_threadgroup]]) {                                       \
    threadgroup float scratch[32];                                                             \
    enc_norm_body<T, D>(x, add, w, b, y, pre, (long)tg * (long)cols, cols, eps, has_add,        \
                        has_pre, lid, tgs, scratch);                                            \
}

#define DECLARE_ENC_ATTN(NAME, T, D)                                                           \
kernel void NAME(device const T* qkv, device const T* bias, device const uchar* mask,           \
                 device T* ctx, constant uint& length, constant uint& embed,                    \
                 constant uint& stride_mb, constant uint& stride_mh, constant uint& stride_ml,   \
                 constant uint& stride_mj, constant float& scale, constant uint& has_mask,       \
                 uint lid [[thread_position_in_threadgroup]],                                   \
                 uint tg [[threadgroup_position_in_grid]],                                      \
                 uint tgs [[threads_per_threadgroup]]) {                                        \
    threadgroup float Ks[ENC_KC(D) * D];                                                        \
    threadgroup float Vs[ENC_KC(D) * D];                                                        \
    enc_attn_body<T, D>(qkv, bias, mask, ctx, length, embed, stride_mb, stride_mh, stride_ml,   \
                        stride_mj, scale, has_mask, Ks, Vs, tg, lid, tgs);                       \
}

DECLARE_ENC_NORM(enc_norm_f32_d1, float, 1)
DECLARE_ENC_NORM(enc_norm_f32_d2, float, 2)
DECLARE_ENC_NORM(enc_norm_f32_d4, float, 4)
DECLARE_ENC_NORM(enc_norm_f32_d8, float, 8)
DECLARE_ENC_NORM(enc_norm_f32_d16, float, 16)
DECLARE_ENC_NORM(enc_norm_f16_d1, half, 1)
DECLARE_ENC_NORM(enc_norm_f16_d2, half, 2)
DECLARE_ENC_NORM(enc_norm_f16_d4, half, 4)
DECLARE_ENC_NORM(enc_norm_f16_d8, half, 8)
DECLARE_ENC_NORM(enc_norm_f16_d16, half, 16)
DECLARE_ENC_ATTN(enc_attn_f32_d8, float, 8)
DECLARE_ENC_ATTN(enc_attn_f32_d16, float, 16)
DECLARE_ENC_ATTN(enc_attn_f32_d32, float, 32)
DECLARE_ENC_ATTN(enc_attn_f32_d64, float, 64)
DECLARE_ENC_ATTN(enc_attn_f32_d128, float, 128)
DECLARE_ENC_ATTN(enc_attn_f16_d8, half, 8)
DECLARE_ENC_ATTN(enc_attn_f16_d16, half, 16)
DECLARE_ENC_ATTN(enc_attn_f16_d32, half, 32)
DECLARE_ENC_ATTN(enc_attn_f16_d64, half, 64)
DECLARE_ENC_ATTN(enc_attn_f16_d128, half, 128)
"""
"""The encoder layer's two stages: the layer norm and the fused attention."""

_ENCODER_SUFFIX = {torch.float32: "f32", torch.float16: "f16"}
_ENCODER_MAX_DEPTH = 16
_ENCODER_HEAD_DIMS = (8, 16, 32, 64, 128)
"""Per-head widths the attention kernel is compiled for."""


@lru_cache(maxsize=1)
def _encoder_library():
    """Compiled on first use; importing this module compiles nothing."""
    return torch.mps.compile_shader(_ENCODER_SOURCE)


def _encoder_norm_layout(cols):
    """Threadgroup width and per-thread depth for one normalization row."""
    width = 32
    while width < cols and width < 1024:
        width *= 2
    depth = 1
    while depth * width < cols:
        depth *= 2
    return (width, depth) if depth <= _ENCODER_MAX_DEPTH else None


def _encoder_norm(y, x, add, weight, bias, eps, pre=None):
    """``layer_norm(x + add)`` into ``y``; ``pre`` also receives ``x + add``.

    ``x``, ``y``, ``add`` and ``pre`` are 2-D views of ``(rows, cols)``.
    """
    rows, cols = x.shape
    width, depth = _encoder_norm_layout(cols)
    kernel = getattr(_encoder_library(), f"enc_norm_{_ENCODER_SUFFIX[x.dtype]}_d{depth}")
    kernel(
        x,
        x if add is None else add,
        weight,
        bias,
        y,
        y if pre is None else pre,
        cols,
        float(eps),
        0 if add is None else 1,
        0 if pre is None else 1,
        threads=rows * width,
        group_size=width,
    )


def _encoder_attention(ctx, qkv, bias, mask, batch, length, embed, head_dim):
    """Scaled dot-product attention over the packed projection, into ``ctx``.

    ``qkv`` is the projection without its bias: the kernel adds the bias the way
    ``_transform_bias_rescale_qkv`` does, and a mask dimension of extent one
    broadcasts, which is a stride of zero.
    """
    heads = embed // head_dim
    if mask is None:
        strides = (0, 0, 0, 0)
        mask_tensor = qkv
    else:
        strides = tuple(0 if size == 1 else step for size, step in zip(mask.shape, mask.stride()))
        mask_tensor = mask.view(torch.uint8)
    kernel = getattr(_encoder_library(), f"enc_attn_{_ENCODER_SUFFIX[ctx.dtype]}_d{head_dim}")
    width = min(1024, max(32, -(-length // 32) * 32))
    kernel(
        qkv,
        bias,
        mask_tensor,
        ctx,
        length,
        embed,
        strides[0],
        strides[1],
        strides[2],
        strides[3],
        float(1.0 / math.sqrt(head_dim)),
        0 if mask is None else 1,
        threads=batch * heads * width,
        group_size=width,
    )


def _transformer_encoder_layer_fwd(
    src,
    embed_dim,
    num_heads,
    qkv_weight,
    qkv_bias,
    proj_weight,
    proj_bias,
    use_gelu,
    norm_first,
    eps,
    norm_weight_1,
    norm_bias_1,
    norm_weight_2,
    norm_bias_2,
    ffn_weight_1,
    ffn_bias_1,
    ffn_weight_2,
    ffn_bias_2,
    mask=None,
    mask_type=None,
):
    """One ``TransformerEncoderLayer`` pass with the weights passed in flat.

    An MPS float32 or float16 batch runs the fused pass: seven kernel-bearing
    launches of which two are this module's kernels, against the composed path's
    29. Every other device, dtype and shape takes the composed path, which is
    also the definition of the arithmetic the kernels reproduce.
    """
    if src.numel() == 0:
        return src.clone()
    fused = _encoder_metal(
        src, embed_dim, num_heads, qkv_weight, qkv_bias, proj_weight, proj_bias,
        use_gelu, norm_first, eps, norm_weight_1, norm_bias_1, norm_weight_2,
        norm_bias_2, ffn_weight_1, ffn_bias_1, ffn_weight_2, ffn_bias_2, mask, mask_type,
    )
    if fused is not None:
        return fused
    return _encoder_composed(
        src, embed_dim, num_heads, qkv_weight, qkv_bias, proj_weight, proj_bias,
        use_gelu, norm_first, eps, norm_weight_1, norm_bias_1, norm_weight_2,
        norm_bias_2, ffn_weight_1, ffn_bias_1, ffn_weight_2, ffn_bias_2, mask, mask_type,
    )


def _encoder_composed(
    src,
    embed_dim,
    num_heads,
    qkv_weight,
    qkv_bias,
    proj_weight,
    proj_bias,
    use_gelu,
    norm_first,
    eps,
    norm_weight_1,
    norm_bias_1,
    norm_weight_2,
    norm_bias_2,
    ffn_weight_1,
    ffn_bias_1,
    ffn_weight_2,
    ffn_bias_2,
    mask=None,
    mask_type=None,
):
    """The layer as the aten steps it was made of: every device and dtype."""
    norm_shape = (embed_dim,)
    x = src
    if norm_first:
        x = F.layer_norm(x, norm_shape, norm_weight_1, norm_bias_1, eps)
    x = _native_multi_head_attention(
        x,
        x,
        x,
        embed_dim,
        num_heads,
        qkv_weight,
        qkv_bias,
        proj_weight,
        proj_bias,
        mask,
        False,
        True,
        mask_type,
    )[0]
    x = x + src
    if not norm_first:
        x = F.layer_norm(x, norm_shape, norm_weight_1, norm_bias_1, eps)

    residual = x
    if norm_first:
        x = F.layer_norm(x, norm_shape, norm_weight_2, norm_bias_2, eps)
    x = F.linear(x, ffn_weight_1, ffn_bias_1)
    x = F.gelu(x) if use_gelu else F.relu(x)
    x = F.linear(x, ffn_weight_2, ffn_bias_2)
    x = x + residual
    if not norm_first:
        x = F.layer_norm(x, norm_shape, norm_weight_2, norm_bias_2, eps)
    return x


def _encoder_metal(
    src,
    embed_dim,
    num_heads,
    qkv_weight,
    qkv_bias,
    proj_weight,
    proj_bias,
    use_gelu,
    norm_first,
    eps,
    norm_weight_1,
    norm_bias_1,
    norm_weight_2,
    norm_bias_2,
    ffn_weight_1,
    ffn_bias_1,
    ffn_weight_2,
    ffn_bias_2,
    mask=None,
    mask_type=None,
):
    """The same layer in seven kernel-bearing launches, or None when it cannot run.

    Two Metal passes cover the whole attention block: one writes the layer norm
    (plus the residual add in front of it, plus the value that add produces,
    which the feed-forward residual needs), and one does the q/k/v bias and
    scale, the softmax and the context, writing the ``(rows, embed)`` layout the
    output projection reads. Against the composed path that removes the
    ``(rows, 3 * embed)`` split pass, the materialised scores and their softmax,
    and the ``permute`` + ``reshape`` copy; the projections stay torch matmuls,
    which beat a hand-written tiled kernel at these shapes by 1.4-1.6x
    (``4x128x128``, measured interleaved).

    The second layer norm fuses into the residual add before it, and in the
    pre-norm order the first one writes the residual too, so each norm costs one
    launch rather than two.
    """
    if src.device.type != "mps" or src.dtype not in _ENCODER_SUFFIX:
        return None
    if src.dim() != 3 or src.shape[2] != embed_dim or embed_dim == 0:
        return None
    head_dim = embed_dim // num_heads if num_heads else 0
    if not num_heads or embed_dim % num_heads or head_dim not in _ENCODER_HEAD_DIMS:
        return None
    rows = src.shape[0] * src.shape[1]
    if rows == 0 or _encoder_norm_layout(embed_dim) is None:
        return None
    for tensor in (
        qkv_weight, qkv_bias, proj_weight, proj_bias, norm_weight_1, norm_bias_1,
        norm_weight_2, norm_bias_2, ffn_weight_1, ffn_bias_1, ffn_weight_2, ffn_bias_2,
    ):
        if tensor.dtype != src.dtype or tensor.device != src.device:
            return None
    if qkv_weight.shape != (3 * embed_dim, embed_dim):
        return None
    if proj_weight.shape != (embed_dim, embed_dim):
        return None
    if ffn_weight_1.dim() != 2 or ffn_weight_1.shape[1] != embed_dim or not ffn_weight_1.shape[0]:
        return None
    if ffn_weight_2.shape != (embed_dim, ffn_weight_1.shape[0]):
        return None
    if qkv_bias.shape != (3 * embed_dim,) or proj_bias.shape != (embed_dim,):
        return None
    if ffn_bias_1.shape != (ffn_weight_1.shape[0],) or ffn_bias_2.shape != (embed_dim,):
        return None
    for tensor in (norm_weight_1, norm_bias_1, norm_weight_2, norm_bias_2):
        if tensor.shape != (embed_dim,):
            return None
    flat = src.reshape(rows, embed_dim)
    x = flat if flat.is_contiguous() else flat.contiguous()

    batch = src.shape[0]
    length = src.shape[1]
    if mask is not None:
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)
        mask = _mask_for_scores((batch, num_heads, length, length), mask, mask_type, None)
        if not mask.is_contiguous():
            mask = mask.contiguous()

    # Stage 1: the pre-norm, only in the order that consumes it.
    if norm_first:
        normed = torch.empty_like(x)
        _encoder_norm(normed, x, None, norm_weight_1, norm_bias_1, eps)
        projected_from = normed
    else:
        projected_from = x

    # Stage 2: the projection without its bias -- the attention kernel adds it
    # at the rounding points ``_transform_bias_rescale_qkv`` uses -- then
    # attention over it.
    qkv = torch.matmul(projected_from, qkv_weight.t())
    context = torch.empty_like(x)
    _encoder_attention(context, qkv, qkv_bias, mask, batch, length, embed_dim, head_dim)

    # Stage 3: out projection, into the residual the layer norm in front of it
    # adds -- or, post-norm, into the add that norm performs itself.
    attention_out = F.linear(context, proj_weight, proj_bias)

    # Stage 4: the feed-forward block.
    if norm_first:
        # y = attention_out + src; norm2 into normed; the FFN residual is y.
        y = torch.empty_like(x)
        _encoder_norm(normed, attention_out, x, norm_weight_2, norm_bias_2, eps, pre=y)
        hidden = F.linear(normed, ffn_weight_1, ffn_bias_1)
        act = F.gelu(hidden) if use_gelu else F.relu(hidden)
        return (F.linear(act, ffn_weight_2, ffn_bias_2) + y).reshape(src.shape)

    # norm1(attention_out + src) is the FFN residual in the post-norm order.
    normed = torch.empty_like(x)
    _encoder_norm(normed, attention_out, x, norm_weight_1, norm_bias_1, eps)
    hidden = F.linear(normed, ffn_weight_1, ffn_bias_1)
    act = F.gelu(hidden) if use_gelu else F.relu(hidden)
    second = F.linear(act, ffn_weight_2, ffn_bias_2) + normed
    out = torch.empty_like(x)
    _encoder_norm(out, second, None, norm_weight_2, norm_bias_2, eps)
    return out.reshape(src.shape)


def _fused_sdp_choice(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
    enable_gqa=False,
):
    """The backend ``aten::scaled_dot_product_attention`` reaches on MPS: math."""
    return int(torch.nn.attention.SDPBackend.MATH)


OVERLAYS = (
    ("_masked_softmax", _masked_softmax),
    ("_masked_softmax_backward", _masked_softmax_backward),
    ("_transform_bias_rescale_qkv", _transform_bias_rescale_qkv),
    ("_native_multi_head_attention", _native_multi_head_attention),
    ("_transformer_encoder_layer_fwd", _transformer_encoder_layer_fwd),
    ("_fused_sdp_choice", _fused_sdp_choice),
)
"""``(aten overload, MPS kernel)`` pairs for the accelerator to install."""
