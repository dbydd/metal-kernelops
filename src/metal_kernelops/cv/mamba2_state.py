"""Mamba-2 selective state update on Metal: one fused step of the decode recurrence.

The operator is ``selective_state_update`` from
``mamba_ssm.ops.triton.selective_state_update`` (state-spaces/mamba): the single
step a Mamba-2 layer runs per token when generating, as opposed to the chunked
scan it runs over a prompt.  One step is

    delta = softplus(dt + dt_bias)            (dt_softplus=True)
    a     = exp(A_h * delta)
    s'    = a * s + delta * x * B[g]          (g = h // (nheads / ngroups))
    y     = sum_n s' * C[g] + D * x
    y     = y * silu(z)                       (z given)

Layouts (no head axis folding; the official function squeezes absent axes, this
one requires them):

    state   (batch, nheads, dim, dstate)
    x       (batch, nheads, dim)
    dt      (batch, nheads) or (batch, nheads, dim)
    A       (nheads,)                scalar per head
    B, C    (batch, ngroups, dstate)  nheads % ngroups == 0
    D       (nheads,) or (nheads, dim)
    z       (batch, nheads, dim)
    dt_bias (nheads,)

Two departures from the official kernel are deliberate.  ``A`` is one scalar per
head instead of a ``(nheads, dim, dstate)`` matrix, which is what Mamba-2 uses
(``A = -exp(A_log)``, one value per head); and the new state is written to a
fresh tensor unless ``out_state`` names a destination.  The official Triton
kernel overwrites ``state`` unconditionally, which makes it unusable when the
caller still needs the old state, e.g. for a backward pass.

Kernel
------
Both passes map one row -- one ``(batch, head, dim)`` element -- onto ``lanes``
threads and pack ``rtg`` rows into one threadgroup.  ``lanes`` is the largest
power of two not above ``min(dstate, 32)``, so a small ``dstate`` (16, 8, 1)
fills the threadgroup with several rows instead of leaving lanes idle; ``rtg``
is the largest power of two up to 8 that divides ``dim``, so a threadgroup never
spans two heads.  Lane ``l`` of a row owns the four consecutive ``dstate``
elements at ``4l``, so a simdgroup issues one 16-byte request per thread and the
four loads of a row arrive as one 128-byte line instead of four strided rounds;
``dstate`` below four falls back to scalar loads.  The per-row reduction is a
``simd_shuffle_xor`` butterfly over ``lanes`` (the hardware ``simd_sum`` over a
full simdgroup), which stays inside the row because both numbers are powers of
two.

Forward is one dispatch.  Each lane reads its slice of the state, ``B`` and ``C``
once, writes its slice of the new state once, and the row reduces ``s' * C``;
lane 0 applies ``D * x``, the ``silu`` gate and stores one ``y``.  With no input
requiring grad the public entry point calls the kernel directly instead of going
through ``autograd.Function``, which is the path a decode loop takes.

Backward is two dispatches, and every gradient has exactly one writer, so it is
bit-deterministic.  The first kernel produces the row-local adjoints
(``dstate``, ``dx``, ``ddt``, ``dz``) and the per-row partials of ``dA`` and
``dD``; ``dB`` and ``dC``, whose reduction axis is the row axis, are folded
across the ``rtg`` rows of the threadgroup -- through threadgroup memory when
``rtg > 1`` -- and written as one partial row per threadgroup.  The second kernel
is a single dispatch of four flat jobs that collapse the partials: per head
(``dA``, ``ddt_bias``, a rank-1 ``dD``), per ``(head, dim)`` (a rank-2 ``dD``),
per ``(batch, group, dstate)`` (``dB``, ``dC``) and per ``(batch, head)`` (a
per-head ``dt``).  Its threadgroup is four simdgroups: the head jobs take one
head per simdgroup, the wide jobs take 128 outputs, and the ``dB``/``dC`` job --
the only long chain in it -- splits its partial rows four ways and meets in
threadgroup memory.  At decode shapes that workspace is a few hundred KB, against
the atomics it replaces, which serialised thousands of adds per address.

Precision
---------
float32 computes in float32.  float16 and bfloat16 round once, on the store; the
whole recurrence runs in float32 internally.  Backward always computes in
float32 and casts the gradients back to the dtype of the input they belong to,
so a float16 step gets a float32-accurate gradient.  ``A``, ``D`` and
``dt_bias`` are read as float32 whatever their storage dtype.

Importing this module compiles no shader and allocates nothing: each library is
built on the first call that needs it and cached.
"""

from functools import lru_cache

import torch
from torch.autograd.function import once_differentiable

CHECKED_TORCH = "2.14.1"
"""Torch release the module was written and measured against."""

_DTYPE_NAMES = {
    torch.float32: ("f32", "float", "float4"),
    torch.float16: ("f16", "half", "half4"),
    torch.bfloat16: ("bf16", "bfloat", "bfloat4"),
}

_MAX_RTG = 8
"""Most rows one threadgroup folds together; also the shared-memory row count."""

# Option bits.  The backward kernel passes 21 tensors plus 7 scalars, and Metal
# gives a kernel 31 buffers, so the bits share one word instead of one each.
_DT_PER_ROW = 1
_HAS_BIAS = 2
_HAS_D = 4
_D_PER_HEAD = 8
_HAS_Z = 16
_SOFTPLUS = 32
_HAS_DY = 64
_HAS_DNEW = 128

_FLAGS = {
    "$DT_PER_ROW": _DT_PER_ROW,
    "$HAS_BIAS": _HAS_BIAS,
    "$HAS_D": _HAS_D,
    "$D_PER_HEAD": _D_PER_HEAD,
    "$HAS_Z": _HAS_Z,
    "$SOFTPLUS": _SOFTPLUS,
    "$HAS_DY": _HAS_DY,
    "$HAS_DNEW": _HAS_DNEW,
}


def _layout(dstate, dim):
    """``(lanes, rtg)``: threads per row, and rows per threadgroup."""
    lanes = 1
    while lanes * 2 <= dstate and lanes < 32:
        lanes *= 2
    rtg = 1
    while rtg * 2 <= _MAX_RTG and dim % (rtg * 2) == 0:
        rtg *= 2
    return lanes, rtg


def _substitute(source, extra):
    merged = {**_FLAGS, **extra}
    # Longest first: "$HAS_D" is a prefix of "$HAS_DY" and "$HAS_DNEW".
    for name in sorted(merged, key=len, reverse=True):
        source = source.replace(name, str(merged[name]))
    return source


_HELPERS = r"""
#include <metal_stdlib>
using namespace metal;

inline float loadv(device const float* p, long i) { return p[i]; }
inline float loadv(device const half* p, long i) { return float(p[i]); }
inline float loadv(device const bfloat* p, long i) { return float(p[i]); }
inline void storev(device float* p, long i, float v) { p[i] = v; }
inline void storev(device half* p, long i, float v) { p[i] = half(v); }
inline void storev(device bfloat* p, long i, float v) { p[i] = bfloat(v); }

/* Sum over the `width` lanes of one row.  Both the width and the lane offsets
 * are powers of two, so the butterfly never leaves the row, and every lane of
 * the row ends up with the total.  A full simdgroup uses the hardware
 * instruction; the shorter widths unroll to a handful of shuffles, because a
 * loop over a runtime width costs more than the four-element body it reduces. */
inline float block_sum(float v, uint width) {
    if (width == 32) return simd_sum(v);
    if (width >= 16) v += simd_shuffle_xor(v, 8);
    if (width >= 8) v += simd_shuffle_xor(v, 4);
    if (width >= 4) v += simd_shuffle_xor(v, 2);
    if (width >= 2) v += simd_shuffle_xor(v, 1);
    return v;
}

/* log(1 + exp(v)) to within an ulp of the float32 result.  Metal has no log1p,
 * and log(1 + exp(v)) loses most of its relative accuracy for small v, which is
 * exactly the range a delta of 1e-3 lives in.  Around zero the series
 * log2 + v/2 + v^2/8 - v^4/192 + v^6/2880 is used instead. */
inline float softplusv(float v) {
    if (v > 20.0f) return v;
    if (fabs(v) < 0.25f) {
        float v2 = v * v;
        return 0.6931471805599453f + 0.5f * v +
               v2 * (0.125f + v2 * (-0.005208333333333333f + v2 * 0.0003472222222222222f));
    }
    return log(1.0f + exp(v));
}

inline float sigmoidv(float v) { return 1.0f / (1.0f + exp(-v)); }

inline bool flagged(uint flags, uint bit) { return (flags & bit) != 0u; }
"""

_FORWARD = r"""
kernel void state_update_$S(
    device const $T* state, device const $T* x, device const $T* dt,
    device const float* dt_bias, device const float* A,
    device const $T* Bv, device const $T* Cv, device const float* Dv,
    device const $T* zv, device $T* new_state, device $T* y, device float* delta,
    constant uint& nheads, constant uint& dim, constant uint& dstate_size,
    constant uint& groups, constant uint& lanes, constant uint& rtg,
    constant uint& dt_per_row, constant uint& has_bias,
    constant uint& has_d, constant uint& d_per_head,
    constant uint& has_z, constant uint& softplus,
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint rin = tid / lanes;
    uint lane = tid - rin * lanes;
    uint row = tg * rtg + rin;
    uint per_head = nheads * dim;
    uint b = row / per_head;
    uint rest = row - b * per_head;
    uint h = rest / dim;
    uint p = rest - h * dim;
    uint g = h / (nheads / groups);
    uint dt_at = dt_per_row ? row : b * nheads + h;

    /* softplus and the decay are one value per row, so lane 0 evaluates them and
     * the row reads the result back: otherwise all `lanes` threads run the same
     * exp and log, which at 17 us a step is a measurable share of the kernel. */
    float d = 0.0f, a = 0.0f;
    if (lane == 0) {
        float raw = loadv(dt, dt_at);
        if (has_bias) raw += dt_bias[h];
        d = softplus ? softplusv(raw) : raw;
        a = exp(A[h] * d);
    }
    uint head = (rin * lanes) & 31;
    float xv = loadv(x, row);
    /* simd_shuffle, not simd_broadcast: several rows can share one simdgroup, so
     * the source lane differs between threads. */
    d = simd_shuffle(d, head);
    a = simd_shuffle(a, head);
    long bbase = (long) ((b * groups + g) * dstate_size);
    long sbase = (long) row * dstate_size;
    float dv = has_d ? (d_per_head ? Dv[h * dim + p] : Dv[h]) : 0.0f;

    float acc = 0.0f;
    /* Each thread walks its row from lane 0 of the row up, in float4 where it
     * can: one 16-byte request per thread keeps the warp's accesses coalesced
     * and needs neither a second read of B/C nor a shuffle. */
    for (uint q = lane * 4; q < dstate_size; q += lanes * 4) {
        if (q + 3 < dstate_size && dstate_size % 4 == 0) {
            $T4 sv = *((device const $T4*) (state + sbase + q));
            $T4 bqv = *((device const $T4*) (Bv + bbase + q));
            $T4 cqv = *((device const $T4*) (Cv + bbase + q));
            float4 nv = float4(a * float(sv.x) + (d * float(bqv.x)) * xv,
                               a * float(sv.y) + (d * float(bqv.y)) * xv,
                               a * float(sv.z) + (d * float(bqv.z)) * xv,
                               a * float(sv.w) + (d * float(bqv.w)) * xv);
            *((device $T4*) (new_state + sbase + q)) =
                $T4($T(nv.x), $T(nv.y), $T(nv.z), $T(nv.w));
            acc += nv.x * float(cqv.x) + nv.y * float(cqv.y)
                 + nv.z * float(cqv.z) + nv.w * float(cqv.w);
        } else {
            for (uint k = 0; k < 4; ++k) {
                uint n = q + k;
                if (n < dstate_size) {
                    float nv = a * loadv(state, sbase + n) + (d * loadv(Bv, bbase + n)) * xv;
                    storev(new_state, sbase + n, nv);
                    acc += nv * loadv(Cv, bbase + n);
                }
            }
        }
    }
    float total = block_sum(acc, lanes);
    if (lane == 0) {
        delta[row] = d;
        float out = total + (has_d ? xv * dv : 0.0f);
        if (has_z) {
            float zv1 = loadv(zv, row);
            out *= sigmoidv(zv1) * zv1;
        }
        storev(y, row, out);
    }
}
"""

_BACKWARD = r"""
kernel void state_update_backward(
    device const float* state, device const float* x, device const float* dt,
    device const float* dt_bias, device const float* A,
    device const float* Bv, device const float* Cv, device const float* Dv,
    device const float* zv, device const float* new_state, device const float* delta,
    device const float* dy, device const float* dnew,
    device float* dstate_out, device float* dx, device float* ddt, device float* dz,
    device float* dA_row, device float* dD_row,
    device float* dB_part, device float* dC_part,
    constant uint& nheads, constant uint& dim, constant uint& dstate_size,
    constant uint& groups, constant uint& lanes, constant uint& rtg,
    constant uint& flags,
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float shb[$MAX_RTG][128];
    threadgroup float shc[$MAX_RTG][128];

    uint rin = tid / lanes;
    uint lane = tid - rin * lanes;
    uint row = tg * rtg + rin;
    uint per_head = nheads * dim;
    uint b = row / per_head;
    uint rest = row - b * per_head;
    uint h = rest / dim;
    uint p = rest - h * dim;
    uint g = h / (nheads / groups);
    uint dt_at = flagged(flags, $DT_PER_ROW) ? row : b * nheads + h;

    /* The decay and the gate are one value per row: lane 0 evaluates them and the
     * row reads them back, so the exp and the sigmoid are not run `lanes` times. */
    float d = delta[row];
    float a = 0.0f, sig = 1.0f;
    if (lane == 0) {
        a = exp(A[h] * d);
        if (flagged(flags, $HAS_Z)) sig = sigmoidv(zv[row]);
    }
    uint head = (rin * lanes) & 31;
    a = simd_shuffle(a, head);
    sig = simd_shuffle(sig, head);
    float raw = dt[dt_at];
    if (flagged(flags, $HAS_BIAS)) raw += dt_bias[h];
    float xv = x[row];
    float dv = flagged(flags, $HAS_D)
                   ? (flagged(flags, $D_PER_HEAD) ? Dv[h * dim + p] : Dv[h]) : 0.0f;
    float dyv = flagged(flags, $HAS_DY) ? dy[row] : 0.0f;
    float zval = flagged(flags, $HAS_Z) ? zv[row] : 0.0f;
    float gate = flagged(flags, $HAS_Z) ? zval * sig : 1.0f;
    float dy_pre = dyv * gate;

    long sbase = (long) row * dstate_size;
    long bbase = (long) ((b * groups + g) * dstate_size);
    long partbase = (long) tg * dstate_size;
    bool fold = rtg > 1;
    float dxa = 0.0f, dda = 0.0f, daa = 0.0f, ya = 0.0f;

    /* Four consecutive elements per lane per step, as in the forward: one
     * 16-byte request per thread, and a quarter of the loop iterations. */
    bool wide = (dstate_size % 4) == 0;
    for (uint q = lane * 4; q < dstate_size; q += lanes * 4) {
        float dbv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float dcv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (wide) {
            float4 sn = *((device const float4*) (state + sbase + q));
            float4 sv = *((device const float4*) (new_state + sbase + q));
            float4 bvn = *((device const float4*) (Bv + bbase + q));
            float4 cv = *((device const float4*) (Cv + bbase + q));
            float4 dn = flagged(flags, $HAS_DNEW)
                            ? *((device const float4*) (dnew + sbase + q))
                            : float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 qv = float4(dn.x + dy_pre * cv.x, dn.y + dy_pre * cv.y,
                               dn.z + dy_pre * cv.z, dn.w + dy_pre * cv.w);
            *((device float4*) (dstate_out + sbase + q)) =
                float4(qv.x * a, qv.y * a, qv.z * a, qv.w * a);
            dbv[0] = qv.x * (d * xv);
            dbv[1] = qv.y * (d * xv);
            dbv[2] = qv.z * (d * xv);
            dbv[3] = qv.w * (d * xv);
            dcv[0] = dy_pre * sv.x;
            dcv[1] = dy_pre * sv.y;
            dcv[2] = dy_pre * sv.z;
            dcv[3] = dy_pre * sv.w;
            dxa += qv.x * (d * bvn.x) + qv.y * (d * bvn.y)
                 + qv.z * (d * bvn.z) + qv.w * (d * bvn.w);
            dda += qv.x * (xv * bvn.x) + qv.y * (xv * bvn.y)
                 + qv.z * (xv * bvn.z) + qv.w * (xv * bvn.w);
            daa += qv.x * sn.x + qv.y * sn.y + qv.z * sn.z + qv.w * sn.w;
            ya += sv.x * cv.x + sv.y * cv.y + sv.z * cv.z + sv.w * cv.w;
        } else {
            for (uint k = 0; k < 4; ++k) {
                uint n = q + k;
                if (n < dstate_size) {
                    float cv = Cv[bbase + n];
                    float qv = (flagged(flags, $HAS_DNEW) ? dnew[sbase + n] : 0.0f)
                             + dy_pre * cv;
                    float sn = state[sbase + n];
                    float sv = new_state[sbase + n];
                    float bvn = Bv[bbase + n];
                    dstate_out[sbase + n] = qv * a;
                    dbv[k] = qv * (d * xv);
                    dcv[k] = dy_pre * sv;
                    dxa += qv * (d * bvn);
                    dda += qv * (xv * bvn);
                    daa += qv * sn;
                    ya += sv * cv;
                }
            }
        }
        uint slot = lane * 4;
        if (fold) {
            for (uint k = 0; k < 4; ++k) {
                shb[rin][slot + k] = dbv[k];
                shc[rin][slot + k] = dcv[k];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (rin == 0) {
                for (uint k = 0; k < 4; ++k) {
                    uint n = q + k;
                    if (n < dstate_size) {
                        float sb = shb[0][slot + k], sc = shc[0][slot + k];
                        for (uint r = 1; r < rtg; ++r) {
                            sb += shb[r][slot + k];
                            sc += shc[r][slot + k];
                        }
                        dB_part[partbase + n] = sb;
                        dC_part[partbase + n] = sc;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            for (uint k = 0; k < 4; ++k) {
                uint n = q + k;
                if (n < dstate_size) {
                    dB_part[partbase + n] = dbv[k];
                    dC_part[partbase + n] = dcv[k];
                }
            }
        }
    }

    float dxs = block_sum(dxa, lanes);
    float dds = block_sum(dda, lanes);
    float das = block_sum(daa, lanes);
    float ys = block_sum(ya, lanes);
    if (lane == 0) {
        dx[row] = dxs + dy_pre * dv;
        dA_row[row] = das * a * d;
        dD_row[row] = dy_pre * xv;
        /* d enters through the drive (s' = a s + d x B) and through the decay
         * (a = exp(A d), so da/dd = a A). */
        float dd = dds + das * a * A[h];
        if (flagged(flags, $SOFTPLUS)) dd *= raw > 20.0f ? 1.0f : sigmoidv(raw);
        ddt[row] = dd;
        if (flagged(flags, $HAS_Z)) {
            float ypre = ys + (flagged(flags, $HAS_D) ? xv * dv : 0.0f);
            dz[row] = dyv * ypre * sig * (1.0f + zval * (1.0f - sig));
        }
    }
}
"""

_REDUCE = r"""
kernel void state_update_reduce(
    device const float* dA_row, device const float* ddt, device const float* dD_row,
    device const float* dB_part, device const float* dC_part,
    device float* dA, device float* ddt_bias, device float* dD_full,
    device float* dD_head, device float* dB, device float* dC, device float* ddt_head,
    constant uint& nheads, constant uint& dim, constant uint& dstate_size,
    constant uint& groups, constant uint& batch, constant uint& rtg,
    constant uint& flags,
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[2][4][32];

    /* Four simdgroups per threadgroup: job C, the only long chain here, splits
     * its partial rows four ways, and the other jobs cover four outputs per
     * simdgroup instead of leaving three of them idle. */
    uint sub = tid / 32;
    uint lane = tid - sub * 32;
    bool head_d = flagged(flags, $HAS_D) && !flagged(flags, $D_PER_HEAD);
    uint n_a = (nheads + 3) / 4;
    uint n_c = (batch * groups * dstate_size + 31) / 32;
    uint n_b = head_d || !flagged(flags, $HAS_D) ? 0 : (nheads * dim + 127) / 128;
    uint n_d = flagged(flags, $DT_PER_ROW) ? 0 : (batch * nheads + 127) / 128;

    if (tg < n_a) {
        /* Per head: dA, ddt_bias, and a rank-1 dD.  Rows of one head are
         * (b, p) pairs, so the lane walks p with a stride of 32. */
        uint h = tg * 4 + sub;
        if (h < nheads) {
            uint total = batch * dim;
            float sa = 0.0f, st = 0.0f, sd = 0.0f;
            for (uint j = lane; j < total; j += 32) {
                uint b = j / dim;
                uint row = (b * nheads + h) * dim + (j - b * dim);
                sa += dA_row[row];
                st += ddt[row];
                if (head_d) sd += dD_row[row];
            }
            sa = simd_sum(sa);
            st = simd_sum(st);
            sd = simd_sum(sd);
            if (lane == 0) {
                dA[h] = sa;
                if (flagged(flags, $HAS_BIAS)) ddt_bias[h] = st;
                if (head_d) dD_head[h] = sd;
            }
        }
    } else if (tg < n_a + n_c) {
        /* Per (batch, group, dstate): sum the threadgroup partials of that
         * group.  Their partial rows are contiguous, because rows are ordered
         * (batch, head, dim) and a group owns a contiguous run of heads.  The
         * four simdgroups take every fourth partial row and meet in shared. */
        uint total = batch * groups * dstate_size;
        uint v = (tg - n_a) * 32 + lane;
        uint at = min(v, total - 1);
        uint b = at / (groups * dstate_size);
        uint gn = at - b * (groups * dstate_size);
        uint g = gn / dstate_size;
        uint n = gn - g * dstate_size;
        uint count = (nheads / groups) * dim / rtg;
        uint first = (b * nheads * dim + g * (nheads / groups) * dim) / rtg;
        float sb0 = 0.0f, sc0 = 0.0f, sb1 = 0.0f, sc1 = 0.0f;
        float sb2 = 0.0f, sc2 = 0.0f, sb3 = 0.0f, sc3 = 0.0f;
        uint c = sub;
        for (; c + 12 < count; c += 16) {
            uint i0 = (first + c) * dstate_size + n;
            uint i1 = i0 + 4 * dstate_size;
            uint i2 = i1 + 4 * dstate_size;
            uint i3 = i2 + 4 * dstate_size;
            sb0 += dB_part[i0]; sc0 += dC_part[i0];
            sb1 += dB_part[i1]; sc1 += dC_part[i1];
            sb2 += dB_part[i2]; sc2 += dC_part[i2];
            sb3 += dB_part[i3]; sc3 += dC_part[i3];
        }
        for (; c < count; c += 4) {
            uint i = (first + c) * dstate_size + n;
            sb0 += dB_part[i]; sc0 += dC_part[i];
        }
        sh[0][sub][lane] = (sb0 + sb1) + (sb2 + sb3);
        sh[1][sub][lane] = (sc0 + sc1) + (sc2 + sc3);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sub == 0 && v < total) {
            dB[v] = (sh[0][0][lane] + sh[0][1][lane]) + (sh[0][2][lane] + sh[0][3][lane]);
            dC[v] = (sh[1][0][lane] + sh[1][1][lane]) + (sh[1][2][lane] + sh[1][3][lane]);
        }
    } else if (tg < n_a + n_c + n_b) {
        uint v = (tg - n_a - n_c) * 128 + tid;
        if (v < nheads * dim) {
            uint h = v / dim;
            uint p = v - h * dim;
            float s = 0.0f;
            for (uint b = 0; b < batch; ++b) s += dD_row[(b * nheads + h) * dim + p];
            dD_full[v] = s;
        }
    } else {
        uint v = (tg - n_a - n_c - n_b) * 128 + tid;
        if (v < batch * nheads) {
            uint b = v / nheads;
            uint h = v - b * nheads;
            float s = 0.0f;
            for (uint p = 0; p < dim; ++p) s += ddt[(b * nheads + h) * dim + p];
            ddt_head[v] = s;
        }
    }
}
"""



@lru_cache(maxsize=3)
def _forward_library(dtype):
    suffix, storage, vector = _DTYPE_NAMES[dtype]
    source = _substitute(_HELPERS + _FORWARD,
                         {"$T4": vector, "$T": storage, "$S": suffix, "$MAX_RTG": _MAX_RTG})
    return torch.mps.compile_shader(source)


@lru_cache(maxsize=1)
def _backward_library():
    return torch.mps.compile_shader(
        _substitute(_HELPERS + _BACKWARD + _REDUCE, {"$MAX_RTG": _MAX_RTG}))


def _wide(tensor):
    """float32 view of an activation tensor; the backward always computes there."""
    if tensor is None or tensor.dtype == torch.float32:
        return tensor
    return tensor.float()


def _narrow(tensor, dtype):
    return tensor if dtype == torch.float32 else tensor.to(dtype)


def _check(state, x, dt, A, B, C, D, z, dt_bias, out_state):
    if state.dim() != 4:
        raise ValueError(f"state must be (batch, nheads, dim, dstate), got {tuple(state.shape)}")
    batch, nheads, dim, dstate = state.shape
    if dstate < 1:
        raise ValueError("state must have a non-empty last dimension")
    expected = (batch, nheads, dim)
    for name, tensor in (("x", x), ("z", z)):
        if tensor is not None and tuple(tensor.shape) != expected:
            raise ValueError(f"{name} must be {expected}, got {tuple(tensor.shape)}")
    if dt.shape not in ((batch, nheads), expected):
        raise ValueError(f"dt must be {(batch, nheads)} or {expected}, got {tuple(dt.shape)}")
    if tuple(A.shape) != (nheads,):
        raise ValueError(f"A must be {(nheads,)}, got {tuple(A.shape)}")
    groups = B.shape[1]
    if tuple(B.shape) != (batch, groups, dstate) or tuple(C.shape) != tuple(B.shape):
        raise ValueError(
            f"B and C must be {(batch, groups, dstate)}, got {tuple(B.shape)}, {tuple(C.shape)}"
        )
    if nheads % groups:
        raise ValueError(f"nheads ({nheads}) must be divisible by ngroups ({groups})")
    if D is not None and tuple(D.shape) not in ((nheads,), (nheads, dim)):
        raise ValueError(f"D must be {(nheads,)} or {(nheads, dim)}, got {tuple(D.shape)}")
    if dt_bias is not None and tuple(dt_bias.shape) != (nheads,):
        raise ValueError(f"dt_bias must be {(nheads,)}, got {tuple(dt_bias.shape)}")
    if out_state is not None:
        if out_state.device != state.device:
            raise ValueError("out_state must be on the same device as state")
        if tuple(out_state.shape) != tuple(state.shape) or out_state.dtype != state.dtype:
            raise ValueError("out_state must match state in shape and dtype")
        if not out_state.is_contiguous():
            raise ValueError("out_state must be contiguous")
        if out_state.requires_grad:
            raise ValueError("out_state is written in place and cannot require grad")


def _launch(state, x, dt, A, Bv, Cv, D, z, bias, softplus, out_state):
    """Run the forward kernel and return ``(y, new_state, delta)``."""
    nheads, dim, dstate = state.shape[1], state.shape[2], state.shape[3]
    groups = Bv.shape[1]
    rows = state.shape[0] * nheads * dim
    lanes, rtg = _layout(dstate, dim)
    new_state = out_state if out_state is not None else torch.empty_like(state)
    delta = torch.empty((state.shape[0], nheads, dim), dtype=torch.float32,
                        device=state.device)
    y = torch.empty((state.shape[0], nheads, dim), dtype=state.dtype, device=state.device)

    state = state if state.is_contiguous() else state.contiguous()
    x = x if x.is_contiguous() else x.contiguous()
    dt = dt if dt.is_contiguous() else dt.contiguous()
    Bv = Bv if Bv.is_contiguous() else Bv.contiguous()
    Cv = Cv if Cv.is_contiguous() else Cv.contiguous()
    if D is not None:
        D = D if D.is_contiguous() else D.contiguous()
    if z is not None:
        z = z if z.is_contiguous() else z.contiguous()
    if bias is not None:
        bias = bias if bias.is_contiguous() else bias.contiguous()

    if rows:
        suffix = _DTYPE_NAMES[state.dtype][0]
        kernel = getattr(_forward_library(state.dtype), "state_update_" + suffix)
        kernel(
            state, x, dt, bias if bias is not None else A, A,
            Bv, Cv, D if D is not None else A, z if z is not None else x,
            new_state, y, delta,
            nheads, dim, dstate, groups, lanes, rtg,
            int(dt.dim() == 3), int(bias is not None),
            int(D is not None), int(D is not None and D.dim() == 2),
            int(z is not None), int(bool(softplus)),
            threads=rows * lanes, group_size=rtg * lanes,
        )
    return y, new_state, delta


class _StateUpdate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, x, dt, A, Bv, Cv, D, z, bias, softplus, out_state):
        y, new_state, delta = _launch(state, x, dt, A, Bv, Cv, D, z, bias,
                                      softplus, out_state)
        ctx.softplus = bool(softplus)
        ctx.dt_per_row = dt.dim() == 3
        # An unused output should reach backward as None, not as a zero tensor the
        # size of the state.
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(state, x, dt, A, Bv, Cv, D, z, bias, new_state, delta)
        return y, new_state if out_state is None else new_state.detach()

    @staticmethod
    @once_differentiable
    def backward(ctx, dy, dnew):
        state, x, dt, A, Bv, Cv, D, z, bias, new_state, delta = ctx.saved_tensors
        softplus, dt_per_row = ctx.softplus, ctx.dt_per_row
        batch, nheads, dim, dstate = state.shape
        groups = Bv.shape[1]
        rows = batch * nheads * dim
        lanes, rtg = _layout(dstate, dim)

        w_state, w_x, w_dt = _wide(state), _wide(x), _wide(dt)
        w_new, w_delta = _wide(new_state), delta
        w_Bv, w_Cv = _wide(Bv), _wide(Cv)
        w_D = _wide(D)
        w_z = _wide(z)
        w_bias = _wide(bias)
        w_dy = None if dy is None else _wide(dy).contiguous()
        w_dnew = None if dnew is None else _wide(dnew).contiguous()

        f32 = torch.float32
        device = state.device
        grad_state = torch.empty_like(w_state)
        dx = torch.empty_like(w_x)
        ddt = torch.empty((batch, nheads, dim), dtype=f32, device=device)
        dz = torch.empty_like(w_x) if z is not None else None
        dA_row = torch.empty(rows, dtype=f32, device=device)
        dD_row = torch.empty(rows, dtype=f32, device=device)
        parts = rows // rtg
        dB_part = torch.empty((parts, dstate), dtype=f32, device=device)
        dC_part = torch.empty((parts, dstate), dtype=f32, device=device)
        dBv = torch.empty((batch, groups, dstate), dtype=f32, device=device)
        dCv = torch.empty((batch, groups, dstate), dtype=f32, device=device)
        dA = torch.empty(nheads, dtype=f32, device=device)
        ddt_bias = torch.empty(nheads, dtype=f32, device=device) if bias is not None else None
        if D is None:
            dD = dD_full = dD_head = None
        elif D.dim() == 1:
            dD = torch.empty(nheads, dtype=f32, device=device)
            dD_full, dD_head = None, dD
        else:
            dD = torch.empty((nheads, dim), dtype=f32, device=device)
            dD_full, dD_head = dD, None
        ddt_head = (torch.empty((batch, nheads), dtype=f32, device=device)
                    if not dt_per_row else None)

        flags = (
            (_DT_PER_ROW if dt_per_row else 0)
            | (_HAS_BIAS if bias is not None else 0)
            | (_HAS_D if D is not None else 0)
            | (_D_PER_HEAD if D is not None and D.dim() == 2 else 0)
            | (_HAS_Z if z is not None else 0)
            | (_SOFTPLUS if softplus else 0)
            | (_HAS_DY if w_dy is not None else 0)
            | (_HAS_DNEW if w_dnew is not None else 0)
        )

        if rows:
            unused = dA
            library = _backward_library()
            library.state_update_backward(
                w_state, w_x, w_dt, w_bias if w_bias is not None else A, A,
                w_Bv, w_Cv, w_D if w_D is not None else A, w_z if w_z is not None else w_x,
                w_new, w_delta,
                w_dy if w_dy is not None else unused, w_dnew if w_dnew is not None else unused,
                grad_state, dx, ddt, dz if dz is not None else dx,
                dA_row, dD_row,
                dB_part, dC_part,
                nheads, dim, dstate, groups, lanes, rtg,
                flags,
                threads=rows * lanes, group_size=rtg * lanes,
            )
            n_a = (nheads + 3) // 4
            n_c = (batch * groups * dstate + 31) // 32
            n_b = (nheads * dim + 127) // 128 if dD_full is not None else 0
            n_d = 0 if dt_per_row else (batch * nheads + 127) // 128
            total = n_a + n_c + n_b + n_d
            library.state_update_reduce(
                dA_row, ddt, dD_row, dB_part, dC_part,
                dA, ddt_bias if ddt_bias is not None else dA,
                dD_full if dD_full is not None else dA,
                dD_head if dD_head is not None else dA,
                dBv, dCv,
                ddt_head if ddt_head is not None else dA,
                nheads, dim, dstate, groups, batch, rtg, flags,
                threads=total * 128, group_size=128,
            )
        else:
            for tensor in (dA, ddt_bias, dD):
                if tensor is not None:
                    tensor.zero_()

        return (
            _narrow(grad_state, state.dtype),
            _narrow(dx, x.dtype),
            _narrow(ddt if dt_per_row else ddt_head, dt.dtype),
            _narrow(dA, A.dtype),
            _narrow(dBv, Bv.dtype),
            _narrow(dCv, Cv.dtype),
            None if D is None else _narrow(dD, D.dtype),
            None if z is None else _narrow(dz, z.dtype),
            None if bias is None else _narrow(ddt_bias, bias.dtype),
            None,
            None,
        )


def mamba2_state_update(state, x, dt, A, B, C, D=None, z=None, dt_bias=None,
                        dt_softplus=False, out_state=None):
    """One Mamba-2 selective state update step on MPS.

    Args:
        state: ``(batch, nheads, dim, dstate)`` float32/float16/bfloat16, not modified.
        x: ``(batch, nheads, dim)``.
        dt: ``(batch, nheads)`` or ``(batch, nheads, dim)``.
        A: ``(nheads,)`` per-head log decay, normally ``-exp(A_log)``.
        B, C: ``(batch, ngroups, dstate)``; head ``h`` reads group ``h // (nheads // ngroups)``.
        D: ``(nheads,)`` or ``(nheads, dim)`` skip connection.
        z: ``(batch, nheads, dim)`` gate, applied as ``y * silu(z)``.
        dt_bias: ``(nheads,)`` added to ``dt`` before the optional softplus.
        dt_softplus: apply softplus to ``dt + dt_bias`` (torch's threshold-20 rule).
        out_state: optional destination for the new state, written in place.  It must
            not require grad; the differentiable path is the default one.

    Returns:
        ``(y, updated_state)`` with ``y`` of shape ``(batch, nheads, dim)``.  Without
        ``out_state`` the input ``state`` is untouched and ``updated_state`` is a
        fresh tensor; with it, ``updated_state`` shares storage with ``out_state``.
    """
    if state.device.type != "mps":
        raise ValueError("mamba2_state_update runs on the mps device")
    if state.dtype not in _DTYPE_NAMES:
        raise ValueError(f"state must be float32, float16 or bfloat16, got {state.dtype}")
    _check(state, x, dt, A, B, C, D, z, dt_bias, out_state)

    dtype = state.dtype
    if x.dtype != dtype:
        x = x.to(dtype)
    if dt.dtype != dtype:
        dt = dt.to(dtype)
    if B.dtype != dtype:
        B = B.to(dtype)
    if C.dtype != dtype:
        C = C.to(dtype)
    if z is not None and z.dtype != dtype:
        z = z.to(dtype)
    if A.dtype != torch.float32:
        A = A.float()
    if D is not None and D.dtype != torch.float32:
        D = D.float()
    if dt_bias is not None and dt_bias.dtype != torch.float32:
        dt_bias = dt_bias.float()

    needs_grad = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad
        for t in (state, x, dt, A, B, C, D, z, dt_bias)
    )
    if not needs_grad:
        y, new_state, _ = _launch(state, x, dt, A, B, C, D, z, dt_bias,
                                  bool(dt_softplus), out_state)
        return y, new_state

    if (out_state is not None and state.requires_grad
            and out_state.untyped_storage().data_ptr() == state.untyped_storage().data_ptr()):
        # The backward needs the entering state, which an aliasing destination overwrites.
        state = state.clone()
    return _StateUpdate.apply(state, x, dt, A, B, C, D, z, dt_bias,
                              bool(dt_softplus), out_state)


def mamba2_state_update_reference(state, x, dt, A, B, C, D=None, z=None,
                                  dt_bias=None, dt_softplus=False):
    """Step the recurrence in torch ops, the definition the kernels are checked against.

    Works on any device and dtype, including float64 on the CPU, which is what the
    verifier compares against.  Leaves ``state`` untouched and returns
    ``(y, new_state)``.
    """
    if dt.dim() == 2:
        dt = dt.unsqueeze(-1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(-1)
    if dt_bias is not None:
        dt = dt + dt_bias[None, :, None]
    if dt_softplus:
        dt = torch.nn.functional.softplus(dt)
    nheads, groups = state.shape[1], B.shape[1]
    ratio = nheads // groups
    B = B.repeat_interleave(ratio, dim=1)
    C = C.repeat_interleave(ratio, dim=1)
    decay = torch.exp(A[None, :, None, None] * dt[..., None])
    new_state = decay * state + (dt * x)[..., None] * B[:, :, None, :]
    y = (new_state * C[:, :, None, :]).sum(-1)
    if D is not None:
        y = y + x * D
    if z is not None:
        y = y * torch.nn.functional.silu(z)
    return y, new_state
