"""Gated RMSNorm over the feature axis: one fused Metal pass, analytic backward.

This is not an ``aten`` operator. It is the normalization that Mamba-2 and the
gated linear-attention models run between the convolution and the output
projection, so it is exposed as a Python function with autograd and registered on
no dispatch key.

The operation, matching ``mamba_ssm.ops.triton.layernorm_gated.rms_norm_ref``:

    gate = silu(z) if z is not None else 1
    x'   = x * gate            if norm_before_gate is False
    r    = rsqrt(mean(x'^2 over the group) + eps)
    y    = (x' * r * weight + bias) * gate   if norm_before_gate is True

with the mean taken over the last ``group_size`` elements, or over the whole
feature axis when ``group_size`` is None. Averaging is biased (divide by the
group width), as in the reference. Every intermediate is float32 and the result
is cast back to ``x.dtype``, so a float16 input gets a float32 reduction.

Design
------
One threadgroup owns one ``(row, group)`` block. When the group fits in
registers -- ``DENSE_GROUP`` = 16384 elements, at most 16 of them in each thread
-- its values sit in registers between the reduction and the write: ``x`` and
``z`` cross the memory bus once and ``y`` once, where the eager chain (square,
mean, scale, weight, bias, silu, gate) reads and writes the feature axis a dozen
times. The reduction is a ``simd_sum`` per warp plus 32 floats of threadgroup
memory, so a group is normalized without a second kernel.

A wider group cannot be held in registers, so it streams: the same threadgroup
walks the group twice, once for the mean square and once for the write,
re-reading ``x`` and ``z`` only from what the first pass left in cache. The
forward is one kernel at any width.

Backward is analytic and fused the same way. ``dx`` and ``dz`` need one
threadgroup-wide reduction per row (``c1 = sum(xhat * dy * w) / group_size``);
``dw`` and ``db`` are reductions across rows, so each threadgroup keeps one
partial column block per thread in registers while it walks a run of rows and
writes a single partial afterwards, and the partials are summed in torch.
Wider than the register set, the same two quantities come from two kernels: one
threadgroup per row-group for ``c1``, then one thread per feature column, which
walks its row block and writes a ``dw``/``db`` partial -- a column is a register
whatever the group width.

Layout: float32, float16 and bfloat16 on MPS take the Metal path at any group
width; everything else (CPU, float64, empty input) takes the composed-torch
path, which is the same arithmetic in the same order and is what the reference
is written against. This MSL has no bfloat16 type, so those tensors are read as
the top 16 bits of a float32 and written back rounded to nearest even, which is
what torch's own cast does.

The Metal backward is first-order only (``once_differentiable``), as in the
Triton original: it writes into fresh buffers rather than composing an op the
engine could differentiate again.
"""

from functools import lru_cache

import torch
from torch.autograd.function import once_differentiable

# The numbers are the shader's; keep them in step with the GATE_* defines below.
_GATE_NONE = 0
_GATE_BEFORE = 1
"""``norm_before_gate=False``: multiply by silu(z), then normalize and scale."""
_GATE_AFTER = 2
"""``norm_before_gate=True``: normalize, scale, then multiply by silu(z)."""

DENSE_GROUP = 16384
"""Widest group the register-cached kernels take; one thread holds at most 16 values."""

_STREAM_THREADS = 256
"""Threadgroup width of the kernels that walk a group instead of holding it."""

_APPLY_THREADS = 256
"""Threadgroup width of the per-column gradient kernel; it never shares registers."""

_MAX_ELEMENTS_PER_THREAD = 16
_MAX_THREADS = 1024
_MAX_PARTIAL_BYTES = 8 << 20
"""Cap on one gradient-partial buffer, so a wide feature axis cannot blow up ``dw``."""

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

#define GATE_NONE 0u
#define GATE_BEFORE 1u
#define GATE_AFTER 2u

/* One element per access. bfloat16 has no type in this MSL, so it is read as the
   top 16 bits of its float32 and written back rounded to nearest even -- the
   rounding torch's own cast applies -- rather than truncated. */
static inline float load_as_float(const device float* p, uint i) {
    return p[i];
}
static inline float load_as_float(const device half* p, uint i) {
    return static_cast<float>(p[i]);
}
static inline float load_as_float(const device ushort* p, uint i) {
    return as_type<float>(((uint)p[i]) << 16);
}
static inline uint bf16_bits(float v) {
    uint bits = as_type<uint>(v);
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return bits >> 16;
}
static inline void store_from_float(device float* p, uint i, float v) {
    p[i] = v;
}
static inline void store_from_float(device half* p, uint i, float v) {
    p[i] = static_cast<half>(v);
}
static inline void store_from_float(device ushort* p, uint i, float v) {
    p[i] = static_cast<ushort>(bf16_bits(v));
}

static inline void silu_pair(float z, thread float& silu, thread float& sigmoid) {
    sigmoid = 1.0f / (1.0f + exp(-z));
    silu = z * sigmoid;
}
static inline float silu_value(float z) {
    float silu, sigmoid;
    silu_pair(z, silu, sigmoid);
    return silu;
}
/* d silu(z) / dz, the factor the gate contributes to dx and dz. */
static inline float silu_slope(float z) {
    float silu, sigmoid;
    silu_pair(z, silu, sigmoid);
    return sigmoid * (1.0f + z * (1.0f - sigmoid));
}

/* One threadgroup-wide sum: simd_sum inside a warp, one scratch slot per warp,
   then every thread replays the slots. The caller consumes the result before
   the slots are written again, so one barrier per call is enough. */
static inline float tg_sum(float v, threadgroup float* scratch, uint lid, uint tgs) {
    v = simd_sum(v);
    if (lid % 32u == 0u) scratch[lid / 32u] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint j = 0; j < (tgs + 31u) / 32u; ++j) total += scratch[j];
    return total;
}

/* tg_sum for a caller that reuses the scratch slots on the next iteration. */
static inline float tg_sum_repeating(float v, threadgroup float* scratch, uint lid, uint tgs) {
    float total = tg_sum(v, scratch, lid, tgs);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

template <typename T, uint E>
static inline void fwd_impl(
    const device T* x, const device T* z, const device float* w, const device float* b,
    device T* y, device float* rstd,
    constant uint& d, constant uint& gs, constant uint& ngroups, constant float& eps,
    constant uint& has_bias, constant uint& gate,
    threadgroup float* scratch, uint lid, uint tg, uint tgs)
{
    uint group = tg % ngroups;
    uint row = tg / ngroups;
    uint base = row * d + group * gs;
    float v[E];
    uint n = 0;
    float sumsq = 0.0f;
    for (uint i = lid; i < gs; i += tgs, ++n) {
        float xv = load_as_float(x, base + i);
        if (gate == GATE_BEFORE) xv *= silu_value(load_as_float(z, base + i));
        v[n] = xv;
        sumsq += xv * xv;
    }
    float r = rsqrt(tg_sum(sumsq, scratch, lid, tgs) / float(gs) + eps);
    if (lid == 0u) rstd[row * ngroups + group] = r;
    n = 0;
    for (uint i = lid; i < gs; i += tgs, ++n) {
        float o = v[n] * r * w[group * gs + i];
        if (has_bias == 1u) o += b[group * gs + i];
        if (gate == GATE_AFTER) o *= silu_value(load_as_float(z, base + i));
        store_from_float(y, base + i, o);
    }
}

template <typename T, uint E>
static inline void bwd_impl(
    const device T* x, const device T* z, const device float* w, const device float* b,
    const device T* dy, const device float* rstd, device T* dx, device T* dz,
    device float* dwp, device float* dbp,
    constant uint& d, constant uint& gs, constant uint& ngroups, constant uint& rows,
    constant uint& rpp, constant uint& has_bias, constant uint& gate,
    threadgroup float* scratch, uint lid, uint tg, uint tgs)
{
    uint block = tg / ngroups;
    uint group = tg % ngroups;
    float dw_acc[E];
    float db_acc[E];
    for (uint k = 0; k < E; ++k) {
        dw_acc[k] = 0.0f;
        db_acc[k] = 0.0f;
    }
    uint row0 = block * rpp;
    for (uint r = 0; r < rpp; ++r) {
        uint row = row0 + r;
        if (row >= rows) break;
        uint base = row * d + group * gs;
        float rstdv = rstd[row * ngroups + group];
        float v[E];
        float gdy[E];
        uint n = 0;
        float c1 = 0.0f;
        for (uint i = lid; i < gs; i += tgs, ++n) {
            float xv = load_as_float(x, base + i);
            if (gate == GATE_BEFORE) xv *= silu_value(load_as_float(z, base + i));
            float xhat = xv * rstdv;
            float gv = load_as_float(dy, base + i);
            if (gate == GATE_AFTER) gv *= silu_value(load_as_float(z, base + i));
            v[n] = xv;
            gdy[n] = gv;
            c1 += xhat * gv * w[group * gs + i];
            dw_acc[n] += gv * xhat;
            if (has_bias == 1u) db_acc[n] += gv;
        }
        c1 = tg_sum_repeating(c1, scratch, lid, tgs) / float(gs);
        n = 0;
        for (uint i = lid; i < gs; i += tgs, ++n) {
            float xhat = v[n] * rstdv;
            float dxv = (gdy[n] * w[group * gs + i] - xhat * c1) * rstdv;
            if (gate == GATE_BEFORE) {
                float xraw = load_as_float(x, base + i);
                float zv = load_as_float(z, base + i);
                store_from_float(dz, base + i, dxv * xraw * silu_slope(zv));
                dxv *= silu_value(zv);
            } else if (gate == GATE_AFTER) {
                float zv = load_as_float(z, base + i);
                float pre = xhat * w[group * gs + i];
                if (has_bias == 1u) pre += b[group * gs + i];
                store_from_float(dz, base + i, load_as_float(dy, base + i) * pre * silu_slope(zv));
            }
            store_from_float(dx, base + i, dxv);
        }
    }
    uint n = 0;
    for (uint i = lid; i < gs; i += tgs, ++n) {
        dwp[block * d + group * gs + i] = dw_acc[n];
        if (has_bias == 1u) dbp[block * d + group * gs + i] = db_acc[n];
    }
}

/* Groups past the register set: the same threadgroup walks the group once for the
   mean square and once for the write. */
template <typename T>
static inline void fwd_stream_impl(
    const device T* x, const device T* z, const device float* w, const device float* b,
    device T* y, device float* rstd,
    constant uint& d, constant uint& gs, constant uint& ngroups, constant float& eps,
    constant uint& has_bias, constant uint& gate,
    threadgroup float* scratch, uint lid, uint tg, uint tgs)
{
    uint group = tg % ngroups;
    uint row = tg / ngroups;
    uint base = row * d + group * gs;
    float sumsq = 0.0f;
    for (uint i = lid; i < gs; i += tgs) {
        float xv = load_as_float(x, base + i);
        if (gate == GATE_BEFORE) xv *= silu_value(load_as_float(z, base + i));
        sumsq += xv * xv;
    }
    float r = rsqrt(tg_sum(sumsq, scratch, lid, tgs) / float(gs) + eps);
    if (lid == 0u) rstd[row * ngroups + group] = r;
    for (uint i = lid; i < gs; i += tgs) {
        float xv = load_as_float(x, base + i);
        if (gate == GATE_BEFORE) xv *= silu_value(load_as_float(z, base + i));
        float o = xv * r * w[group * gs + i];
        if (has_bias == 1u) o += b[group * gs + i];
        if (gate == GATE_AFTER) o *= silu_value(load_as_float(z, base + i));
        store_from_float(y, base + i, o);
    }
}

/* The one whole-group quantity the backward needs beyond rstd: c1 for one
   (row, group), written for the per-column kernel that follows. */
template <typename T>
static inline void bwd_c1_impl(
    const device T* x, const device T* z, const device float* w, const device T* dy,
    const device float* rstd, device float* c1,
    constant uint& d, constant uint& gs, constant uint& ngroups, constant uint& gate,
    threadgroup float* scratch, uint lid, uint tg, uint tgs)
{
    uint group = tg % ngroups;
    uint row = tg / ngroups;
    uint base = row * d + group * gs;
    float rstdv = rstd[row * ngroups + group];
    float acc = 0.0f;
    for (uint i = lid; i < gs; i += tgs) {
        float xv = load_as_float(x, base + i);
        if (gate == GATE_BEFORE) xv *= silu_value(load_as_float(z, base + i));
        float gv = load_as_float(dy, base + i);
        if (gate == GATE_AFTER) gv *= silu_value(load_as_float(z, base + i));
        acc += xv * rstdv * gv * w[group * gs + i];
    }
    float total = tg_sum(acc, scratch, lid, tgs);
    if (lid == 0u) c1[row * ngroups + group] = total / float(gs);
}

/* One thread owns one feature column of one row block: it accumulates dw and db
   across the block's rows in registers and writes one partial, so the column
   width never reaches the register budget whatever the group width is. */
template <typename T>
static inline void bwd_apply_impl(
    const device T* x, const device T* z, const device float* w, const device float* b,
    const device T* dy, const device float* rstd, const device float* c1,
    device T* dx, device T* dz, device float* dwp, device float* dbp,
    constant uint& d, constant uint& gs, constant uint& ngroups, constant uint& rows,
    constant uint& rpp, constant uint& blocks, constant uint& has_bias,
    constant uint& gate, uint gid)
{
    uint block = gid / d;
    if (block >= blocks) return;
    uint f = gid - block * d;
    uint group = f / gs;
    float wf = w[f];
    float bf = has_bias == 1u ? b[f] : 0.0f;
    float dw = 0.0f;
    float db = 0.0f;
    uint row0 = block * rpp;
    for (uint r = 0; r < rpp; ++r) {
        uint row = row0 + r;
        if (row >= rows) break;
        uint idx = row * d + f;
        float rstdv = rstd[row * ngroups + group];
        float c1v = c1[row * ngroups + group];
        float xraw = load_as_float(x, idx);
        float zv = 0.0f;
        float xv = xraw;
        if (gate == GATE_BEFORE) {
            zv = load_as_float(z, idx);
            xv *= silu_value(zv);
        }
        float xhat = xv * rstdv;
        float dyv = load_as_float(dy, idx);
        float gv = dyv;
        if (gate == GATE_AFTER) {
            zv = load_as_float(z, idx);
            gv *= silu_value(zv);
        }
        float dxv = (gv * wf - xhat * c1v) * rstdv;
        if (gate == GATE_BEFORE) {
            store_from_float(dz, idx, dxv * xraw * silu_slope(zv));
            dxv *= silu_value(zv);
        } else if (gate == GATE_AFTER) {
            float pre = xhat * wf;
            if (has_bias == 1u) pre += bf;
            store_from_float(dz, idx, dyv * pre * silu_slope(zv));
        }
        store_from_float(dx, idx, dxv);
        dw += gv * xhat;
        if (has_bias == 1u) db += gv;
    }
    dwp[block * d + f] = dw;
    if (has_bias == 1u) dbp[block * d + f] = db;
}

#define DECLARE_FWD(NAME, T, E)                                                            \
kernel void NAME(device const T* x, device const T* z, device const float* w,                    \
                 device const float* b, device T* y, device float* rstd,                         \
                 constant uint& d, constant uint& gs, constant uint& ngroups,                    \
                 constant float& eps, constant uint& has_bias, constant uint& gate,              \
                 uint lid [[thread_position_in_threadgroup]],                                    \
                 uint tg [[threadgroup_position_in_grid]],                                       \
                 uint tgs [[threads_per_threadgroup]]) {                                         \
    threadgroup float scratch[32];                                                               \
    fwd_impl<T, E>(x, z, w, b, y, rstd, d, gs, ngroups, eps, has_bias, gate, scratch, lid, tg,   \
                   tgs);                                                                          \
}

#define DECLARE_BWD(NAME, T, E)                                                            \
kernel void NAME(device const T* x, device const T* z, device const float* w,                    \
                 device const float* b, device const T* dy, device const float* rstd,             \
                 device T* dx, device T* dz, device float* dwp, device float* dbp,               \
                 constant uint& d, constant uint& gs, constant uint& ngroups,                    \
                 constant uint& rows, constant uint& rpp, constant uint& has_bias,               \
                 constant uint& gate,                                                            \
                 uint lid [[thread_position_in_threadgroup]],                                    \
                 uint tg [[threadgroup_position_in_grid]],                                       \
                 uint tgs [[threads_per_threadgroup]]) {                                         \
    threadgroup float scratch[32];                                                               \
    bwd_impl<T, E>(x, z, w, b, dy, rstd, dx, dz, dwp, dbp, d, gs, ngroups, rows, rpp, has_bias,  \
                   gate, scratch, lid, tg, tgs);                                                 \
}

#define DECLARE_FWD_STREAM(NAME, T)                                                         \
kernel void NAME(device const T* x, device const T* z, device const float* w,                    \
                 device const float* b, device T* y, device float* rstd,                         \
                 constant uint& d, constant uint& gs, constant uint& ngroups,                    \
                 constant float& eps, constant uint& has_bias, constant uint& gate,              \
                 uint lid [[thread_position_in_threadgroup]],                                    \
                 uint tg [[threadgroup_position_in_grid]],                                       \
                 uint tgs [[threads_per_threadgroup]]) {                                         \
    threadgroup float scratch[32];                                                               \
    fwd_stream_impl<T>(x, z, w, b, y, rstd, d, gs, ngroups, eps, has_bias, gate, scratch, lid,   \
                       tg, tgs);                                                                  \
}

#define DECLARE_BWD_C1(NAME, T)                                                            \
kernel void NAME(device const T* x, device const T* z, device const float* w,                    \
                 device const T* dy, device const float* rstd, device float* c1,                 \
                 constant uint& d, constant uint& gs, constant uint& ngroups,                     \
                 constant uint& gate,                                                            \
                 uint lid [[thread_position_in_threadgroup]],                                    \
                 uint tg [[threadgroup_position_in_grid]],                                       \
                 uint tgs [[threads_per_threadgroup]]) {                                         \
    threadgroup float scratch[32];                                                               \
    bwd_c1_impl<T>(x, z, w, dy, rstd, c1, d, gs, ngroups, gate, scratch, lid, tg, tgs);           \
}

#define DECLARE_BWD_APPLY(NAME, T)                                                         \
kernel void NAME(device const T* x, device const T* z, device const float* w,                    \
                 device const float* b, device const T* dy, device const float* rstd,             \
                 device const float* c1, device T* dx, device T* dz, device float* dwp,           \
                 device float* dbp,                                                              \
                 constant uint& d, constant uint& gs, constant uint& ngroups,                     \
                 constant uint& rows, constant uint& rpp, constant uint& blocks,                  \
                 constant uint& has_bias, constant uint& gate,                                     \
                 uint gid [[thread_position_in_grid]]) {                                          \
    bwd_apply_impl<T>(x, z, w, b, dy, rstd, c1, dx, dz, dwp, dbp, d, gs, ngroups, rows, rpp,      \
                      blocks, has_bias, gate, gid);                                               \
}

DECLARE_FWD(fwd_f32_e2, float, 2)
DECLARE_FWD(fwd_f32_e4, float, 4)
DECLARE_FWD(fwd_f32_e8, float, 8)
DECLARE_FWD(fwd_f32_e16, float, 16)
DECLARE_FWD(fwd_f16_e2, half, 2)
DECLARE_FWD(fwd_f16_e4, half, 4)
DECLARE_FWD(fwd_f16_e8, half, 8)
DECLARE_FWD(fwd_f16_e16, half, 16)
DECLARE_FWD(fwd_bf16_e2, ushort, 2)
DECLARE_FWD(fwd_bf16_e4, ushort, 4)
DECLARE_FWD(fwd_bf16_e8, ushort, 8)
DECLARE_FWD(fwd_bf16_e16, ushort, 16)
DECLARE_BWD(bwd_f32_e2, float, 2)
DECLARE_BWD(bwd_f32_e4, float, 4)
DECLARE_BWD(bwd_f32_e8, float, 8)
DECLARE_BWD(bwd_f32_e16, float, 16)
DECLARE_BWD(bwd_f16_e2, half, 2)
DECLARE_BWD(bwd_f16_e4, half, 4)
DECLARE_BWD(bwd_f16_e8, half, 8)
DECLARE_BWD(bwd_f16_e16, half, 16)
DECLARE_BWD(bwd_bf16_e2, ushort, 2)
DECLARE_BWD(bwd_bf16_e4, ushort, 4)
DECLARE_BWD(bwd_bf16_e8, ushort, 8)
DECLARE_BWD(bwd_bf16_e16, ushort, 16)
DECLARE_FWD_STREAM(fwd_stream_f32, float)
DECLARE_FWD_STREAM(fwd_stream_f16, half)
DECLARE_FWD_STREAM(fwd_stream_bf16, ushort)
DECLARE_BWD_C1(bwd_c1_f32, float)
DECLARE_BWD_C1(bwd_c1_f16, half)
DECLARE_BWD_C1(bwd_c1_bf16, ushort)
DECLARE_BWD_APPLY(bwd_apply_f32, float)
DECLARE_BWD_APPLY(bwd_apply_f16, half)
DECLARE_BWD_APPLY(bwd_apply_bf16, ushort)
"""

_SUFFIX = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}


@lru_cache(maxsize=1)
def _library():
    return torch.mps.compile_shader(_SOURCE)


def _threads(group_size):
    """Threadgroup width for a group; ``_depth`` turns it into registers.

    Measured on M-series: a power-of-two threadgroup near a quarter of the group
    width reaches the same bandwidth as half or an eighth of it, while one value
    per thread -- a threadgroup as wide as the group -- costs a third of the
    bandwidth for a group of 512 and three quarters for a group of 1024. The
    width never drops below a value per 16 elements, which is what bounds the
    register array.
    """
    width = max(32, group_size // 4)
    width = 1 << (width.bit_length() - 1)
    width = min(width, _MAX_THREADS)
    width = max(width, -(-group_size // _MAX_ELEMENTS_PER_THREAD))
    width = -(-width // 32) * 32
    return min(width, _MAX_THREADS)


def _depth(group_size, threads):
    """Register depth: at least 2, a power of two, at most the template set."""
    depth = 2
    while depth < -(-group_size // threads):
        depth *= 2
    return min(depth, _MAX_ELEMENTS_PER_THREAD)


def _row_blocks(rows, ngroups, features):
    """Row blocks for the backward: how many partial ``dw`` blocks to keep.

    Each block owns a contiguous run of rows and accumulates ``dw``/``db`` in
    registers, so more blocks mean more parallelism and a bigger partial buffer
    to reduce at the end. Measured on 1024x5120, 4096x768 and 8192x2560, the cost
    is flat within 5% from 192 to 1024 blocks and 4096x768 loses 70% at 64, so
    256 threadgroups sits in the middle of the plateau and the byte cap bounds
    the reduction it costs.
    """
    blocks = max(1, min(rows, 256 // ngroups))
    blocks = min(blocks, max(1, _MAX_PARTIAL_BYTES // (4 * features)))
    return blocks, -(-rows // blocks)


def _dense_kernel(direction, dtype, group_size):
    """The register-cached entry point for one direction, and its threadgroup."""
    threads = _threads(group_size)
    name = f"{direction}_{_SUFFIX[dtype]}_e{_depth(group_size, threads)}"
    return getattr(_library(), name), threads


def _stream_kernel(direction, dtype):
    """The streaming entry point for one direction; it walks the group."""
    return getattr(_library(), f"{direction}_{_SUFFIX[dtype]}")


def _metal_applies(x, weight, bias, z):
    if x.device.type != "mps" or x.dtype not in _SUFFIX:
        return False
    if x.numel() == 0 or x.shape[-1] == 0:
        return False
    if weight.device != x.device or (bias is not None and bias.device != x.device):
        return False
    if z is not None and (z.dtype != x.dtype or z.shape != x.shape):
        return False
    return weight.shape == (x.shape[-1],) and (bias is None or bias.shape == weight.shape)


def _composed(x, weight, bias, z, eps, group_size, norm_before_gate):
    """The arithmetic of the reference, in torch ops, on whatever device x is on."""
    features = x.shape[-1]
    # float64 in, float64 through, so the CPU path can serve as a reference for
    # the float32 kernels; everything else accumulates in float32.
    work = torch.float64 if x.dtype == torch.float64 else torch.float32
    xf = x.to(work)
    zf = z.to(work) if z is not None else None
    weightf = weight.to(work)
    biasf = bias.to(work) if bias is not None else None
    if zf is not None and not norm_before_gate:
        xf = xf * torch.nn.functional.silu(zf)
    if group_size == features:
        rstd = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
        out = xf * rstd * weightf
    else:
        grouped = xf.reshape(xf.shape[:-1] + (features // group_size, group_size))
        rstd = torch.rsqrt(grouped.square().mean(-1, keepdim=True) + eps)
        out = (grouped * rstd).reshape(xf.shape) * weightf
    if biasf is not None:
        out = out + biasf
    if zf is not None and norm_before_gate:
        out = out * torch.nn.functional.silu(zf)
    return out.to(x.dtype)


class _GatedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, z, eps, group_size, norm_before_gate):
        shape = x.shape
        features = shape[-1]
        # contiguous() is a no-op on a contiguous tensor; reshape alone returns a
        # strided view when x already has two dims, which the kernels cannot read.
        grouped_x = x.reshape(-1, features).contiguous()
        grouped_z = z.reshape(-1, features).contiguous() if z is not None else None
        weight_f32 = weight.float().contiguous()
        bias_f32 = bias.float().contiguous() if bias is not None else None
        rows = grouped_x.shape[0]
        ngroups = features // group_size
        if grouped_z is None:
            gate = _GATE_NONE
        else:
            gate = _GATE_AFTER if norm_before_gate else _GATE_BEFORE
        out = torch.empty_like(grouped_x)
        rstd = torch.empty(rows * ngroups, dtype=torch.float32, device=grouped_x.device)
        # The gate tensor doubles as a dummy of x's storage when there is none:
        # the shader's gate enum, not the pointer, decides whether it is read.
        gate_src = grouped_z if grouped_z is not None else grouped_x
        bias_arg = bias_f32 if bias_f32 is not None else weight_f32
        has_bias = 0 if bias_f32 is None else 1
        if group_size > DENSE_GROUP:
            kernel, threads = _stream_kernel("fwd_stream", x.dtype), _STREAM_THREADS
        else:
            kernel, threads = _dense_kernel("fwd", x.dtype, group_size)
        kernel(grouped_x, gate_src, weight_f32, bias_arg, out, rstd,
               features, group_size, ngroups, float(eps), has_bias, gate,
               threads=rows * ngroups * threads, group_size=threads)
        ctx.save_for_backward(grouped_x, grouped_z, weight_f32, bias_f32, rstd)
        ctx.shape = shape
        ctx.group_size = group_size
        ctx.ngroups = ngroups
        ctx.gate = gate
        ctx.weight_dtype = weight.dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.has_bias = bias is not None
        return out.reshape(shape)

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        grouped_x, grouped_z, weight_f32, bias_f32, rstd = ctx.saved_tensors
        rows, features = grouped_x.shape
        group_size, ngroups, gate = ctx.group_size, ctx.ngroups, ctx.gate
        dy = dy.reshape(rows, features).contiguous()
        dx = torch.empty_like(grouped_x)
        dz = torch.empty_like(grouped_z) if grouped_z is not None else None
        blocks, rows_per_block = _row_blocks(rows, ngroups, features)
        dwp = torch.empty((blocks, features), dtype=torch.float32, device=grouped_x.device)
        dbp = torch.empty_like(dwp) if bias_f32 is not None else dwp
        gate_src = grouped_z if grouped_z is not None else grouped_x
        bias_arg = bias_f32 if bias_f32 is not None else weight_f32
        has_bias = 0 if bias_f32 is None else 1
        dz_arg = dz if dz is not None else dx
        if group_size > DENSE_GROUP:
            c1 = torch.empty(rows * ngroups, dtype=torch.float32, device=grouped_x.device)
            _stream_kernel("bwd_c1", grouped_x.dtype)(
                grouped_x, gate_src, weight_f32, dy, rstd, c1,
                features, group_size, ngroups, gate,
                threads=rows * ngroups * _STREAM_THREADS, group_size=_STREAM_THREADS)
            # One thread owns one feature column of one row block, so the grid is
            # the whole column set of every block.
            _stream_kernel("bwd_apply", grouped_x.dtype)(
                grouped_x, gate_src, weight_f32, bias_arg, dy, rstd, c1,
                dx, dz_arg, dwp, dbp,
                features, group_size, ngroups, rows, rows_per_block, blocks,
                has_bias, gate,
                threads=blocks * features, group_size=_APPLY_THREADS)
        else:
            kernel, threads = _dense_kernel("bwd", grouped_x.dtype, group_size)
            kernel(grouped_x, gate_src, weight_f32, bias_arg, dy, rstd, dx, dz_arg, dwp, dbp,
                   features, group_size, ngroups, rows, rows_per_block,
                   has_bias, gate,
                   threads=blocks * ngroups * threads, group_size=threads)
        dw = dwp.sum(0).to(ctx.weight_dtype) if ctx.needs_input_grad[1] else None
        db = (dbp.sum(0).to(ctx.bias_dtype)
              if ctx.has_bias and ctx.needs_input_grad[2] else None)
        return (dx.reshape(ctx.shape), dw, db,
                dz.reshape(ctx.shape) if dz is not None else None, None, None, None)


def rmsnorm_gated(x, weight, bias=None, z=None, eps=1e-5, group_size=None,
                  norm_before_gate=True):
    """Gated RMSNorm over the last axis, as in ``mamba_ssm``'s ``RMSNorm``.

    Args:
        x: ``(..., features)`` input. float32, float16 and bfloat16 on MPS use
            the fused Metal kernels at any group width; only CPU and float64
            take the composed path.
        weight: ``(features,)`` scale.
        bias: ``(features,)`` shift, added before the gate.
        z: ``(features,)``-shaped gate, same shape as ``x``; ``silu(z)`` either
            multiplies the normalized output (``norm_before_gate=True``) or the
            input before normalization.
        eps: added to the mean square before ``rsqrt``.
        group_size: elements per normalization group; None means the whole
            feature axis. Must divide ``features``.
        norm_before_gate: whether the gate is applied after normalization.

    Returns:
        ``(..., features)`` in ``x.dtype``.
    """
    if x.dim() < 1:
        raise ValueError("rmsnorm_gated needs a feature axis")
    features = x.shape[-1]
    if features == 0:
        # Nothing to average over; the reference returns an empty tensor too.
        return x.clone()
    group = features if group_size is None else int(group_size)
    if group <= 0 or features % group:
        raise ValueError(f"group_size {group} does not divide {features}")
    if weight.shape != (features,):
        raise ValueError(f"weight must be ({features},), got {tuple(weight.shape)}")
    if bias is not None and bias.shape != (features,):
        raise ValueError(f"bias must be ({features},), got {tuple(bias.shape)}")
    if z is not None and z.shape != x.shape:
        raise ValueError(f"z must match x {tuple(x.shape)}, got {tuple(z.shape)}")
    if _metal_applies(x, weight, bias, z):
        return _GatedRMSNorm.apply(x, weight, bias, z, float(eps), group,
                                   bool(norm_before_gate))
    return _composed(x, weight, bias, z, eps, group, norm_before_gate)


def rmsnorm_gated_reference(x, weight, bias=None, z=None, eps=1e-5, group_size=None,
                            norm_before_gate=True):
    """``rms_norm_ref`` from mamba's ``layernorm_gated.py``, evaluated in float64.

    This is the specification the kernels are checked against, not a code path of
    :func:`rmsnorm_gated`: it keeps the reference's exact statement order (gate
    before the mean, scale, shift, gate after) and its biased mean.

    Runs on the CPU, because float64 has no MPS kernels; the result comes back as
    a CPU tensor of ``x.dtype``.
    """
    out_dtype = x.dtype
    # Two steps: `mps.to("cpu", torch.float64)` does not raise on torch 2.14.1,
    # it returns zeros, because the dtype has no MPS kernel.
    xf = _as_double(x)
    weightf = _as_double(weight)
    biasf = _as_double(bias) if bias is not None else None
    zf = _as_double(z) if z is not None else None
    features = xf.shape[-1]
    if features == 0:
        return torch.empty(x.shape, dtype=out_dtype)
    group = features if group_size is None else group_size
    if zf is not None and not norm_before_gate:
        xf = xf * torch.nn.functional.silu(zf)
    grouped = xf.reshape(xf.shape[:-1] + (features // group, group))
    rstd = torch.rsqrt(grouped.square().mean(-1, keepdim=True) + eps)
    out = (grouped * rstd).reshape(xf.shape) * weightf
    if biasf is not None:
        out = out + biasf
    if zf is not None and norm_before_gate:
        out = out * torch.nn.functional.silu(zf)
    return out.to(out_dtype)


def _as_double(tensor):
    return tensor.to("cpu").to(torch.float64)
