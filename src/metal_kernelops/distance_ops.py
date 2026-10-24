"""``aten::_pdist_forward`` / ``aten::_pdist_backward`` on MPS.

``torch.nn.functional.pdist`` is a composite: it validates its arguments and
calls ``aten::_pdist_forward``, whose backward formula calls
``aten::_pdist_backward``.  Neither overload has an MPS kernel, so ``pdist`` on
an MPS tensor raises ``NotImplementedError`` today.  This module supplies both
as Metal kernels; installing it makes ordinary ``F.pdist(x, p)`` work on MPS
with nothing changed at the call site.

Shape of the work
-----------------
Pair ``(i, j)``, ``i < j``, occupies ``out[i*(n-1) - i*(i-1)/2 + (j-i-1)]``,
which is the row-major upper triangle of the distance matrix — the layout
``pdist`` documents and returns.

Neither kernel materialises the ``n x n x d`` difference tensor.  Peak temporary
memory is the ``O(n^2)`` output plus a few scalar loads per pair.

The cost of the naive mapping is that a partner row is read once per
``(destination row, partner)`` pair, so ``n^2 d`` words cross the cache for an
input of ``n d`` words.  Both kernels instead give one threadgroup or thread a
set of output elements and make each partner it reads serve all of them.

Forward, for wide rows: one threadgroup owns ``TILE_SGS * FWD_RPS`` destination
rows, one simdgroup per ``FWD_RPS`` of them, and streams the partners through a
threadgroup tile holding up to ``TILE_F4`` float4 columns.  A partner row is
loaded from device memory once per threadgroup instead of once per pair, so the
device traffic falls by the number of destination rows in the group — 16 at
``TILE_SGS = 8``, ``FWD_RPS = 2``.  The row each simdgroup is measuring stays in
registers; lanes stride the columns and ``simd_sum`` finishes the reduction,
which keeps the accumulation order of the untiled kernel.

Backward: one thread owns one destination row and ``d / 4 / tpr`` float4 columns
and accumulates its own gradient in registers, sweeping every partner.  That is
the loop nest inverted, so the destination row is read once instead of once per
partner.  ``BWD_UNROLL`` independent accumulators break the serial add chain,
and the per-pair scalars (``grad / dist`` for ``p <= 2``, ``dist**(p-1)`` for the
general branch) are hoisted out of the column loop: without that the general
branch spent two ``pow`` calls per element instead of one per pair.

Rows too narrow for a simdgroup to fill take the serial kernel, and the untiled
wide kernels stay for the shapes the tiled forward does not cover (rows below
``TILE_MIN_ROWS``, more than ``TILE_MAX_VEC`` float4 per lane, and the general
``p`` branch, whose ``pow`` cost the tile cannot repay).

Semantics
---------
* Forward: ``p == 0`` counts non-zero components of the difference, ``p == 1``
  sums absolute differences, ``p == 2`` is the Euclidean norm, ``p == inf`` the
  Chebyshev maximum, anything else ``(sum |diff|^p) ** (1/p)``.  ``-0.0`` is
  ``0``. Identical rows therefore produce exactly ``0`` for every mode.
* Backward reproduces the ATen kernel, including its use of the *supplied*
  ``pdist`` values rather than a recomputation and its split at ``p == 2``:
  ``p == 0`` gives zeros, ``p == 1`` gives ``grad * sign(diff)``, ``p == 2``
  gives ``grad * diff / dist`` masked at ``dist == 0``, ``p == inf`` gives
  ``grad * (diff / dist) * (|diff| == dist)`` — ``grad * sign(diff)`` at the
  tied coordinates, but the quotient also carries a NaN or infinite distance
  through the zero mask as ATen does — ``p > 2`` gives
  ``diff * |diff|**(p-2) * grad / dist**(p-1)`` masked at ``dist == 0``, and
  ``p < 2`` gives ``sign(diff) * |diff|**(p-1) * grad / dist**(p-1)`` masked at
  ``dist == 0`` and, for ``p < 1``, at ``diff == 0``.  A zero distance
  therefore contributes no gradient, matching the CPU reference on duplicate
  rows instead of producing NaN.
* ATen orders the branches differently in its two kernels: the forward tests
  ``isinf`` before the general formula, the backward tests ``p < 2`` first.  A
  negative ``p`` therefore takes the general branch forward and the ``p < 2``
  quotient backward, and ``p == -inf`` is a maximum forward but the ``p < 2``
  quotient backward.  ``pdist`` rejects negative ``p``, so only a direct
  ``aten::_pdist_*`` call sees that.  ``d == 0`` returns zeros forward, as
  ATen's ``fill_(0)`` does.  A direct backward call whose ``grad_output`` or
  ``pdist`` does not hold ``n*(n-1)/2`` values raises instead of reading out of
  bounds.
* NaN propagation follows the same branches as ATen: a NaN difference makes the
  pair NaN for ``p == 0`` (ATen's ``min(ceil(|diff|), 1)`` sum), ``p == inf``
  (the maximum is taken on the IEEE bit pattern, so a NaN outranks every
  finite value) and every general ``p``, and it poisons ``sign(diff)`` in the
  ``p == 1`` and ``p == inf`` gradients even where the mask is zero, exactly as
  ATen's product of mask and ``sign`` does.
* ``float32`` only.  ATen's ``pdist`` kernels are float/double only and raise
  ``NotImplementedError`` for half and bfloat16; MPS has no float64, so raising
  for every dtype but float32 keeps the same contract rather than inventing a
  half-precision accuracy promise.  ``pdist``'s own dimension, dtype and
  ``p >= 0`` checks run before this kernel is reached.
* Python reductions are replaced by ``simd_sum`` / ``simd_max`` over 32 lanes,
  so results match ATen to float32 rounding rather than bit-for-bit.  Over the
  shapes in ``benchmarks/verify_distance_ops.py`` the packed forward stays
  within 2.7e-7 relative of the CPU kernel (37 of 112 cases bit-exact) and the
  gradient norm within 6.9e-7.  Against a float64 evaluation the MPS result is
  the closer of the two once ``d`` is large (4.6e-8 against 6.1e-7 at 16x65536),
  because the CPU kernel accumulates in float32 while ``simd_sum`` reduces
  pairwise.  ATen finishes the general branch with ``pow(double(1/p), ...)``;
  this kernel uses float32, which is a one-ulp difference.
* The backward hoists ``grad / dist`` out of the column loop for ``p <= 2`` and
  ``dist**(p-1)`` for the general branch, so a term is computed as
  ``(grad/dist) * diff`` rather than ``grad * diff / dist``, and
  ``BWD_UNROLL`` accumulators are summed at the end, which reorders the partner
  sweep by groups of four.  Both are float32 reassociations: the gradient
  differs from ATen in the last ulp or two, and the measured gradient norm moved
  from 7.8e-7 to 6.9e-7.  They are worth it — three of every four divisions and
  half the ``pow`` calls per pair disappear, and the partner loop stops
  serialising on one accumulator, which is what makes the backward 2.4x faster
  at ``p == 2``.
* Contiguity: ATen's ``_pdist_forward`` asserts a contiguous input; this
  implementation copies instead, so a direct ``aten::_pdist_forward`` call on a
  strided tensor returns the right answer rather than raising.  Public
  ``pdist`` makes the input contiguous first, so only that misuse differs.
* ``_pdist_forward.out``, ``_pdist_backward.out`` and the second-order
  ``_pdist_backward_backward`` are not covered.

Importing this module compiles nothing and registers nothing; the shader is
compiled on the first MPS call, and :data:`OVERLAYS` is what
``metal_kernelops.accelerator`` installs.
"""

import math
from functools import lru_cache

import torch

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

#define BWD_UNROLL 4

// Flat pair index -> (i, j) with i < j, row-major upper triangle.  A float
// estimate of the triangular inverse lands within one row of the answer; the
// two exact integer corrections that follow make it exact.
inline void pair_ij(uint64_t k, uint n, thread uint& i, thread uint& j) {
    float fn = float(2u * n - 1u);
    float disc = fn * fn - 8.0f * float(k);
    uint ii = uint(max(fn - sqrt(max(disc, 0.0f)), 0.0f) * 0.5f);
    if (ii >= n) ii = n - 1;
    uint64_t base = uint64_t(ii) * uint64_t(n - 1) - uint64_t(ii) * uint64_t(ii - 1) / 2;
    while (base > k) {
        --ii;
        base = uint64_t(ii) * uint64_t(n - 1) - uint64_t(ii) * uint64_t(ii - 1) / 2;
    }
    while (base + uint64_t(n - 1 - ii) <= k) {
        ++ii;
        base = uint64_t(ii) * uint64_t(n - 1) - uint64_t(ii) * uint64_t(ii - 1) / 2;
    }
    i = ii;
    j = ii + 1u + uint(k - base);
}

// MODE: 0 count, 1 L1, 2 L2, 3 inf, 4 general with p > 2, 5 general with p < 2.
// The general split only affects the backward; both compute the same forward.
template <uint MODE>
inline float pair_accum(device const float* a, device const float* b,
                        uint d, uint dv, float p, uint lane, uint step) {
    float acc = 0.0f;
    uint accu = 0u;   // MODE 3 accumulates |diff| as bits so NaN outranks inf
    if (dv != 0u) {
        device const float4* a4 = (device const float4*)a;
        device const float4* b4 = (device const float4*)b;
        uint nvec = dv >> 2;
        for (uint c = lane; c < nvec; c += step) {
            float4 df = a4[c] - b4[c];
            if (MODE == 0) {
                acc += (df.x != 0.0f ? (isnan(df.x) ? df.x : 1.0f) : 0.0f)
                     + (df.y != 0.0f ? (isnan(df.y) ? df.y : 1.0f) : 0.0f)
                     + (df.z != 0.0f ? (isnan(df.z) ? df.z : 1.0f) : 0.0f)
                     + (df.w != 0.0f ? (isnan(df.w) ? df.w : 1.0f) : 0.0f);
            } else if (MODE == 1) {
                acc += fabs(df.x) + fabs(df.y) + fabs(df.z) + fabs(df.w);
            } else if (MODE == 2) {
                acc += df.x * df.x + df.y * df.y + df.z * df.z + df.w * df.w;
            } else if (MODE == 3) {
                accu = max(accu, as_type<uint>(fabs(df.x)));
                accu = max(accu, as_type<uint>(fabs(df.y)));
                accu = max(accu, as_type<uint>(fabs(df.z)));
                accu = max(accu, as_type<uint>(fabs(df.w)));
            } else {
                acc += pow(fabs(df.x), p) + pow(fabs(df.y), p)
                     + pow(fabs(df.z), p) + pow(fabs(df.w), p);
            }
        }
    }
    for (uint c = dv + lane; c < d; c += step) {
        float diff = a[c] - b[c];
        if (MODE == 0) acc += (diff != 0.0f ? (isnan(diff) ? diff : 1.0f) : 0.0f);
        else if (MODE == 1) acc += fabs(diff);
        else if (MODE == 2) acc += diff * diff;
        else if (MODE == 3) accu = max(accu, as_type<uint>(fabs(diff)));
        else acc += pow(fabs(diff), p);
    }
    return MODE == 3 ? as_type<float>(accu) : acc;
}

template <uint MODE>
inline float pair_simd(device const float* a, device const float* b,
                       uint d, uint dv, float p, uint lane, uint step) {
    float acc = pair_accum<MODE>(a, b, d, dv, p, lane, step);
    if (MODE == 3) return as_type<float>(simd_max(as_type<uint>(acc)));
    acc = simd_sum(acc);
    if (MODE == 2) return sqrt(acc);
    if (MODE == 4 || MODE == 5) return pow(acc, 1.0f / p);
    return acc;
}

// One simdgroup per pair, per_sg consecutive pairs per simdgroup.  Used once a
// row is wide enough to keep all 32 lanes busy.
kernel void pdist_forward(
    device const float* x,
    device float* out,
    constant uint& n,
    constant uint& d,
    constant float& p,
    constant uint& mode,
    constant uint64_t& npairs,
    constant uint& per_sg,
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint sgsize [[threads_per_simdgroup]]) {
    uint64_t k = (uint64_t(gid) * uint64_t(lsize / sgsize) + uint64_t(sg)) * uint64_t(per_sg);
    if (k >= npairs) return;
    uint i, j;
    pair_ij(k, n, i, j);
    uint dv = (d & 3u) == 0u ? d : 0u;
    for (uint t = 0; t < per_sg && k < npairs; ++t) {
        device const float* a = x + uint64_t(i) * uint64_t(d);
        device const float* b = x + uint64_t(j) * uint64_t(d);
        float acc;
        switch (mode) {
            case 0: acc = pair_simd<0>(a, b, d, dv, p, lane, sgsize); break;
            case 1: acc = pair_simd<1>(a, b, d, dv, p, lane, sgsize); break;
            case 2: acc = pair_simd<2>(a, b, d, dv, p, lane, sgsize); break;
            case 3: acc = pair_simd<3>(a, b, d, dv, p, lane, sgsize); break;
            default: acc = pair_simd<4>(a, b, d, dv, p, lane, sgsize); break;
        }
        if (lane == 0) out[k] = acc;
        ++k;
        ++j;
        if (j == n) { ++i; j = i + 1u; }
    }
}

// One thread per pair for rows narrower than a simdgroup, where the version
// above would idle most lanes.
kernel void pdist_forward_serial(
    device const float* x,
    device float* out,
    constant uint& n,
    constant uint& d,
    constant float& p,
    constant uint& mode,
    constant uint64_t& npairs,
    constant uint& per_thread,
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]) {
    uint64_t k = (uint64_t(gid) * uint64_t(lsize) + uint64_t(tid)) * uint64_t(per_thread);
    if (k >= npairs) return;
    uint i, j;
    pair_ij(k, n, i, j);
    uint dv = (d & 3u) == 0u ? d : 0u;
    for (uint t = 0; t < per_thread && k < npairs; ++t) {
        device const float* a = x + uint64_t(i) * uint64_t(d);
        device const float* b = x + uint64_t(j) * uint64_t(d);
        float acc;
        switch (mode) {
            case 0: acc = pair_accum<0>(a, b, d, dv, p, 0, 1); break;
            case 1: acc = pair_accum<1>(a, b, d, dv, p, 0, 1); break;
            case 2: acc = pair_accum<2>(a, b, d, dv, p, 0, 1); break;
            case 3: acc = pair_accum<3>(a, b, d, dv, p, 0, 1); break;
            default: acc = pair_accum<4>(a, b, d, dv, p, 0, 1); break;
        }
        if (mode == 2) acc = sqrt(acc);
        else if (mode == 4 || mode == 5) acc = pow(acc, 1.0f / p);
        out[k] = acc;
        ++k;
        ++j;
        if (j == n) { ++i; j = i + 1u; }
    }
}

// sign(diff) with ATen's meaning: 0 at zero, NaN for NaN, +/-1 elsewhere.
inline float diff_sign(float diff) {
    if (isnan(diff)) return diff;
    return diff > 0.0f ? 1.0f : (diff < 0.0f ? -1.0f : 0.0f);
}

// One pair's contribution to one column, with the per-pair scalars already
// reduced to a single factor where that is exact for the mode.
template <uint MODE>
inline float bwd_term(float g, float gd, float diff, float dval, float p, float dp) {
    if (MODE == 0) return 0.0f;
    if (MODE == 1) return g * diff_sign(diff);
    if (MODE == 2) return dval != 0.0f ? gd * diff : 0.0f;
    if (MODE == 3) {
        if (dval == 0.0f) return 0.0f;
        float mask = fabs(diff) == dval ? 1.0f : 0.0f;
        return gd * diff * mask;
    }
    if (MODE == 4) {
        if (dval == 0.0f) return 0.0f;
        return diff * pow(fabs(diff), p - 2.0f) * g / dp;
    }
    if (dval == 0.0f || (diff == 0.0f && p < 1.0f)) return 0.0f;
    return diff_sign(diff) * pow(fabs(diff), p - 1.0f) * g / dp;
}

template <uint MODE>
inline float4 bwd_term4(float g, float gd, float4 diff, float dval, float p, float dp) {
    return float4(bwd_term<MODE>(g, gd, diff.x, dval, p, dp),
                  bwd_term<MODE>(g, gd, diff.y, dval, p, dp),
                  bwd_term<MODE>(g, gd, diff.z, dval, p, dp),
                  bwd_term<MODE>(g, gd, diff.w, dval, p, dp));
}

// One thread owns one destination row and one float4 column; BWD_UNROLL accumulators
// break the serial add chain across partners.
template <uint MODE>
inline void bwd_row(device const float* grad, device const float* x,
                    device const float* dist, device float* gx,
                    uint i, uint n, uint d, float p, uint col, uint step, uint dv) {
    uint64_t base_i = uint64_t(i) * uint64_t(n - 1) - uint64_t(i) * uint64_t(i - 1) / 2;
    uint64_t row = uint64_t(i) * uint64_t(d);
    uint nvec = dv >> 2;
    for (uint c = col; c < nvec; c += step) {
        float4 xi = ((device const float4*)(x + row))[c];
        float4 acc[BWD_UNROLL];
        #pragma unroll
        for (uint u = 0; u < BWD_UNROLL; ++u) acc[u] = float4(0.0f);

        uint64_t idx = base_i;
        uint j = i + 1u;
        // A running row pointer: the address no longer costs a 64-bit multiply
        // per partner.
        device const float4* jrow = (device const float4*)(x + uint64_t(j) * uint64_t(d)) + c;
        for (; j + BWD_UNROLL <= n; j += BWD_UNROLL, jrow += uint64_t(BWD_UNROLL) * uint64_t(d) / 4u) {
            #pragma unroll
            for (uint u = 0; u < BWD_UNROLL; ++u) {
                uint64_t at = idx + u;
                float g = grad[at];
                float dval = dist[at];
                float gd = (MODE == 2 || MODE == 3) && dval != 0.0f ? g / dval : g;
                float dp = (MODE == 4 || MODE == 5) ? pow(dval, p - 1.0f) : 0.0f;
                acc[u] += bwd_term4<MODE>(g, gd, xi - jrow[uint64_t(u) * uint64_t(d) / 4u], dval, p, dp);
            }
            idx += BWD_UNROLL;
        }
        for (; j < n; ++j, ++idx, jrow += uint64_t(d) / 4u) {
            float g = grad[idx];
            float dval = dist[idx];
            float gd = (MODE == 2 || MODE == 3) && dval != 0.0f ? g / dval : g;
            float dp = (MODE == 4 || MODE == 5) ? pow(dval, p - 1.0f) : 0.0f;
            acc[0] += bwd_term4<MODE>(g, gd, xi - jrow[0], dval, p, dp);
        }

        int64_t idxb = int64_t(base_i) - int64_t(n - i);
        int64_t step_n = int64_t(1) - int64_t(n);
        uint jj = i;
        device const float4* krow =
            (device const float4*)(x + uint64_t(i > 0u ? i - 1u : 0u) * uint64_t(d)) + c;
        for (; jj >= BWD_UNROLL; jj -= BWD_UNROLL) {
            // The pair index for partner jj - 1 - u, written out in closed form
            // so the BWD_UNROLL loads do not depend on each other.
            #pragma unroll
            for (uint u = 0; u < BWD_UNROLL; ++u) {
                int64_t at = idxb + int64_t(u) * step_n + int64_t(u) * int64_t(jj - 1u)
                           - int64_t(u * (u - 1u) / 2u);
                float g = grad[at];
                float dval = dist[at];
                float gd = (MODE == 2 || MODE == 3) && dval != 0.0f ? g / dval : g;
                float dp = (MODE == 4 || MODE == 5) ? pow(dval, p - 1.0f) : 0.0f;
                acc[u] += bwd_term4<MODE>(g, gd, xi - krow[-int64_t(u) * int64_t(d) / 4], dval, p, dp);
            }
            idxb += int64_t(BWD_UNROLL) * step_n + int64_t(BWD_UNROLL) * int64_t(jj - 1u)
                  - int64_t(BWD_UNROLL * (BWD_UNROLL - 1u) / 2u);
            krow -= int64_t(BWD_UNROLL) * int64_t(d) / 4;
        }
        for (; jj > 0u; --jj) {
            float g = grad[idxb];
            float dval = dist[idxb];
            float gd = (MODE == 2 || MODE == 3) && dval != 0.0f ? g / dval : g;
            float dp = (MODE == 4 || MODE == 5) ? pow(dval, p - 1.0f) : 0.0f;
            acc[0] += bwd_term4<MODE>(g, gd, xi - krow[0], dval, p, dp);
            idxb += int64_t(1) - int64_t(n) + int64_t(jj - 1u);
            krow -= int64_t(d) / 4;
        }

        float4 total = acc[0];
        #pragma unroll
        for (uint u = 1; u < BWD_UNROLL; ++u) total += acc[u];
        ((device float4*)(gx + row))[c] = total;
    }
    for (uint c = dv + col; c < d; c += step) {
        float xi = x[row + c];
        float acc = 0.0f;
        uint64_t idx = base_i;
        for (uint j = i + 1u; j < n; ++j, ++idx) {
            float g = grad[idx];
            float dval = dist[idx];
            float gd = (MODE == 2 || MODE == 3) && dval != 0.0f ? g / dval : g;
            float dp = (MODE == 4 || MODE == 5) ? pow(dval, p - 1.0f) : 0.0f;
            acc += bwd_term<MODE>(g, gd, xi - x[uint64_t(j) * uint64_t(d) + c], dval, p, dp);
        }
        int64_t idxb = int64_t(base_i) - int64_t(n - i);
        for (uint j = i; j-- > 0;) {
            float g = grad[idxb];
            float dval = dist[idxb];
            float gd = (MODE == 2 || MODE == 3) && dval != 0.0f ? g / dval : g;
            float dp = (MODE == 4 || MODE == 5) ? pow(dval, p - 1.0f) : 0.0f;
            acc += bwd_term<MODE>(g, gd, xi - x[uint64_t(j) * uint64_t(d) + c], dval, p, dp);
            idxb += int64_t(1) - int64_t(n) + int64_t(j);
        }
        gx[row + c] = acc;
    }
}

kernel void pdist_backward(
    device const float* grad, device const float* x, device const float* dist,
    device float* gx, constant uint& n, constant uint& d, constant float& p,
    constant uint& mode, constant uint& dv, constant uint& tpr, constant uint& rows,
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint row_in_group = tid / tpr;
    uint col = tid - row_in_group * tpr;
    uint i = gid * rows + row_in_group;
    if (i >= n) return;
    switch (mode) {
        case 0: bwd_row<0>(grad, x, dist, gx, i, n, d, p, col, tpr, dv); break;
        case 1: bwd_row<1>(grad, x, dist, gx, i, n, d, p, col, tpr, dv); break;
        case 2: bwd_row<2>(grad, x, dist, gx, i, n, d, p, col, tpr, dv); break;
        case 3: bwd_row<3>(grad, x, dist, gx, i, n, d, p, col, tpr, dv); break;
        case 4: bwd_row<4>(grad, x, dist, gx, i, n, d, p, col, tpr, dv); break;
        default: bwd_row<5>(grad, x, dist, gx, i, n, d, p, col, tpr, dv); break;
    }
}

"""

_TILE_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

#define CPL $CPL
#define RPS $RPS
#define SGS $SGS
#define TILE_F4 $TILE_F4

// Each simdgroup loads its own slice of the tile rows, so no integer division
// is needed to recover (row, column) from a flat index.
inline void tile_load(device const float* x, threadgroup float4* tile,
                      uint j0, uint n, uint d, uint nvec, uint bj,
                      uint sg, uint lane) {
    uint rows_per_sg = (bj + SGS - 1u) / SGS;
    uint jj0 = sg * rows_per_sg;
    uint jj1 = min(jj0 + rows_per_sg, bj);
    for (uint jj = jj0; jj < jj1; ++jj) {
        uint j = j0 + jj;
        if (j >= n) break;
        device const float4* src = (device const float4*)(x + uint64_t(j) * uint64_t(d));
        threadgroup float4* dst = tile + jj * nvec;
        for (uint c = lane; c < nvec; c += 32u) dst[c] = src[c];
    }
}

template <uint MODE>
inline void fwd_impl(device const float* x, device float* out,
                     threadgroup float4* tile,
                     uint n, uint d, float p, uint nvec, uint bj,
                     uint gid, uint tid) {
    uint lane = tid & 31u;
    uint sg = tid >> 5;
    // Only a threadgroup-uniform exit is legal here: simdgroups of a live
    // threadgroup may be staggered past n and must still reach the barriers.
    uint rowblk = gid * (SGS * RPS);
    if (rowblk >= n) return;
    uint i0 = rowblk + sg * RPS;

    float4 xi[RPS * CPL];
    uint64_t base_i[RPS];
    #pragma unroll
    for (uint r = 0; r < RPS; ++r) {
        uint i = i0 + r;
        bool live = i < n;
        device const float* xi_row = x + uint64_t(i) * uint64_t(d);
        #pragma unroll
        for (uint u = 0; u < CPL; ++u) {
            uint c = lane + u * 32u;
            xi[r * CPL + u] = (live && c < nvec) ? ((device const float4*)xi_row)[c] : float4(0.0f);
        }
        base_i[r] = uint64_t(i) * uint64_t(n - 1u) - uint64_t(i) * uint64_t(i - 1u) / 2u;
    }

    // A threadgroup covers rows [i0_min, i0_min + SGS * RPS); only partners
    // above each row belong to it, so the sweep starts at the tile holding the
    // smallest row + 1.
    uint ntiles = (n + bj - 1u) / bj;
    for (uint t = (rowblk + 1u) / bj; t < ntiles; ++t) {
        uint j0 = t * bj;
        tile_load(x, tile, j0, n, d, nvec, bj, sg, lane);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint jj_end = min(bj, n - j0);
        for (uint jj = 0; jj < jj_end; ++jj) {
            uint j = j0 + jj;
            threadgroup const float4* trow = tile + jj * nvec;
            float4 xj[CPL];
            #pragma unroll
            for (uint u = 0; u < CPL; ++u) {
                uint c = lane + u * 32u;
                xj[u] = (c < nvec) ? trow[c] : float4(0.0f);
            }
            #pragma unroll
            for (uint r = 0; r < RPS; ++r) {
                uint i = i0 + r;
                if (i >= n || j <= i) continue;
                float acc = 0.0f;
                uint accu = 0u;
                #pragma unroll
                for (uint u = 0; u < CPL; ++u) {
                    uint c = lane + u * 32u;
                    if (c < nvec) {
                        float4 df = xi[r * CPL + u] - xj[u];
                        if (MODE == 0) {
                            acc += (df.x != 0.0f ? (isnan(df.x) ? df.x : 1.0f) : 0.0f)
                                 + (df.y != 0.0f ? (isnan(df.y) ? df.y : 1.0f) : 0.0f)
                                 + (df.z != 0.0f ? (isnan(df.z) ? df.z : 1.0f) : 0.0f)
                                 + (df.w != 0.0f ? (isnan(df.w) ? df.w : 1.0f) : 0.0f);
                        } else if (MODE == 1) {
                            acc += fabs(df.x) + fabs(df.y) + fabs(df.z) + fabs(df.w);
                        } else if (MODE == 2) {
                            acc += df.x * df.x + df.y * df.y + df.z * df.z + df.w * df.w;
                        } else if (MODE == 3) {
                            accu = max(accu, as_type<uint>(fabs(df.x)));
                            accu = max(accu, as_type<uint>(fabs(df.y)));
                            accu = max(accu, as_type<uint>(fabs(df.z)));
                            accu = max(accu, as_type<uint>(fabs(df.w)));
                        } else {
                            acc += pow(fabs(df.x), p) + pow(fabs(df.y), p)
                                 + pow(fabs(df.z), p) + pow(fabs(df.w), p);
                        }
                    }
                }
                uint64_t at = base_i[r] + uint64_t(j - i - 1u);
                if (MODE == 3) {
                    float rmax = as_type<float>(simd_max(accu));
                    if (lane == 0) out[at] = rmax;
                } else {
                    acc = simd_sum(acc);
                    if (MODE == 2) acc = sqrt(acc);
                    else if (MODE == 4 || MODE == 5) acc = pow(acc, 1.0f / p);
                    if (lane == 0) out[at] = acc;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void pdist_forward_tile(
    device const float* x, device float* out,
    constant uint& n, constant uint& d, constant float& p, constant uint& mode,
    constant uint& nvec, constant uint& bj,
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float4 tile[TILE_F4];
    switch (mode) {
        case 0: fwd_impl<0>(x, out, tile, n, d, p, nvec, bj, gid, tid); break;
        case 1: fwd_impl<1>(x, out, tile, n, d, p, nvec, bj, gid, tid); break;
        case 2: fwd_impl<2>(x, out, tile, n, d, p, nvec, bj, gid, tid); break;
        case 3: fwd_impl<3>(x, out, tile, n, d, p, nvec, bj, gid, tid); break;
        default: fwd_impl<4>(x, out, tile, n, d, p, nvec, bj, gid, tid); break;
    }
}
"""
"""Tiled kernels, compiled per (float4 columns per lane, rows per simdgroup)."""

SERIAL_MAX_VEC = 16
"""Rows with at most this many float4 lanes per pair use the serial kernel."""

TILE_F4 = 2048
"""float4 slots in the threadgroup tile: 32 KiB, the largest the device allows."""

TILE_SGS = 8
"""Simdgroups per threadgroup in the tiled forward."""

FWD_RPS = 2
"""Destination rows per simdgroup in the tiled forward; the tile row is read once
and used RPS times."""

TILE_MAX_VEC = 8
"""Largest float4 slice per lane the tiled forward is instantiated for."""

TILE_MIN_ROWS = 2048
"""Rows below which the tiled forward stops paying: with only n / (SGS * RPS)
threadgroups the tile load is no longer amortised, and 1024 rows measures level
with the untiled kernel."""

PER_SG = 16
"""Pairs per simdgroup in the wide-row forward kernel."""

PER_GROUP = 8
"""Simdgroups per thread group in the wide-row forward kernel."""

PER_THREAD = 4
"""Pairs per thread in the serial forward kernel."""

BWD_MAX_TPR = 128
"""Cap on threads per row in the backward, four float lanes each."""


@lru_cache(maxsize=1)
def _library():
    return torch.mps.compile_shader(_SOURCE)


@lru_cache(maxsize=8)
def _tiled(cpl, rps):
    return torch.mps.compile_shader(
        _TILE_SOURCE.replace("$CPL", str(cpl)).replace("$RPS", str(rps))
                     .replace("$SGS", str(TILE_SGS)).replace("$TILE_F4", str(TILE_F4))
    )


def _tiled_forward_config(d):
    """Columns and partners per threadgroup, or None when the tiled forward does
    not apply: it reduces over every column inside one threadgroup, so a row must
    fit in a handful of float4 per lane."""
    nvec = d // 4
    if d % 4 or nvec == 0:
        return None
    cpl = 1
    while cpl * 32 < nvec:
        cpl *= 2
    if cpl > TILE_MAX_VEC:
        return None
    return nvec, cpl, max(1, TILE_F4 // nvec)


def _mode(p, backward=False):
    """ATen's kernel selector.  The two kernels test the branches in a different
    order: forward asks ``isinf`` before falling back to the general formula,
    backward asks ``p < 2`` first, so ``-inf`` is a maximum going forward and the
    ``p < 2`` quotient going backward."""
    if p == 0.0:
        return 0
    if p == 1.0:
        return 1
    if p == 2.0:
        return 2
    if backward:
        if p < 2.0:
            return 5
        return 3 if math.isinf(p) else 4
    if math.isinf(p):
        return 3
    return 4 if p > 2.0 else 5


def _check(self, kernel):
    if self.device.type != "mps":
        raise ValueError(f"{kernel} needs tensors on the mps device")
    if self.dtype != torch.float32:
        raise NotImplementedError(f"pdist is not implemented for {self.dtype} on MPS")


def _shape(self):
    return self.size(0), self.size(1)


def pdist_forward(self, p=2.0):
    """``aten::_pdist_forward`` on MPS: one Metal launch, no n x n x d tensor."""
    _check(self, "pdist")
    n, d = _shape(self)
    npairs = n * (n - 1) // 2
    out = torch.empty(npairs, dtype=self.dtype, device=self.device)
    if npairs == 0:
        return out
    if d == 0:
        return out.zero_()
    x = self.contiguous()
    mode = _mode(p)
    if d // 4 <= SERIAL_MAX_VEC:
        units = (npairs + PER_THREAD - 1) // PER_THREAD
        threads = 256
        groups = (units + threads - 1) // threads
        _library().pdist_forward_serial(
            x, out, n, d, float(p), mode, npairs, PER_THREAD,
            threads=groups * threads, group_size=threads,
        )
        return out
    tiled = _tiled_forward_config(d) if mode < 4 and n >= TILE_MIN_ROWS else None
    if tiled is not None:
        nvec, cpl, bj = tiled
        rows = TILE_SGS * FWD_RPS
        groups = (n + rows - 1) // rows
        _tiled(cpl, FWD_RPS).pdist_forward_tile(
            x, out, n, d, float(p), mode, nvec, bj,
            threads=groups * TILE_SGS * 32, group_size=TILE_SGS * 32,
        )
        return out
    units = (npairs + PER_SG - 1) // PER_SG
    groups = (units + PER_GROUP - 1) // PER_GROUP
    _library().pdist_forward(
        x, out, n, d, float(p), mode, npairs, PER_SG,
        threads=groups * 32 * PER_GROUP, group_size=32 * PER_GROUP,
    )
    return out


def pdist_backward(grad, self, p, pdist):
    """``aten::_pdist_backward`` on MPS, matching ATen's masking at zero distance."""
    _check(self, "pdist backward")
    n, d = _shape(self)
    npairs = n * (n - 1) // 2
    if grad.numel() != npairs or pdist.numel() != npairs:
        raise RuntimeError(
            f"_pdist_backward expects grad_output and pdist to hold {npairs} "
            f"values for a {n}-row input, got {grad.numel()} and {pdist.numel()}"
        )
    x = self.contiguous()
    gx = torch.empty_like(x)
    if n < 2 or d == 0:
        gx.zero_()
        return gx
    mode = _mode(p, backward=True)
    if mode == 0:
        gx.zero_()
        return gx
    dv = d if d % 4 == 0 else 0
    nvec = dv // 4
    tpr = min(BWD_MAX_TPR, max(4, nvec)) if nvec else 32
    rows = max(1, 256 // tpr)
    groups = (n + rows - 1) // rows
    _library().pdist_backward(
        grad.contiguous(), x, pdist.contiguous(), gx, n, d, float(p), mode, dv, tpr, rows,
        threads=groups * tpr * rows, group_size=tpr * rows,
    )
    return gx


OVERLAYS = (
    ("_pdist_forward", pdist_forward),
    ("_pdist_backward", pdist_backward),
)
