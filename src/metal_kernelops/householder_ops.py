"""MPS kernels for the Householder pairs of ``torch.geqrf``.

Two operators share one kernel: ``aten::ormqr`` (apply the implicit product to
another matrix) and ``aten::linalg_householder_product`` (form the leading
columns of the product itself, which is the same application with an identity
target). ``torch.orgqr`` has no kernel of its own and is decomposed over
``linalg_householder_product``, so it inherits the faster path as well.

``torch.geqrf`` returns the reflector storage ``input`` (rows below the diagonal
hold the reflector vectors) and the scalar factors ``tau``. ``ormqr`` applies
``Q``, or its transpose, to ``other`` without ever forming ``Q``:

    Q = H_1 H_2 ... H_k,   H_i = I - tau_i v_i v_i^T

with ``v_i`` unit at row ``i`` and ``input[i+1:, i]`` below it. ``left`` picks
the side of ``other`` the product acts on and ``transpose`` whether ``Q`` or
``Q^T`` is applied; both values reach the kernel as the axis of the target
matrix, the order the blocks are replayed in, and which triangle of the block
factor is used.

torch 2.14.1 has no MPS kernel for ``aten::ormqr`` and no composite fallback:
the call raises ``NotImplementedError``. ``linalg_householder_product`` does
have an MPS kernel, but it costs about half a second for a 256x256 input and
eight seconds for 512x512 on this machine, against 0.4 ms and 2 ms on the CPU;
the overlay replaces it.

Replaying the reflectors one at a time is a chain of ``k`` serial steps, and
each step re-reads one reflector column with a 4-byte stride, so the chain runs
at a few GB/s and pays a full memory latency plus two threadgroup barriers per
reflector. Instead the reflectors are grouped into blocks of ``_BLOCK`` and
each block is applied in its compact WY form:

    H_{i0} H_{i0+1} ... H_{i0+ib-1} = I - V T V^T      (forward product, ``T`` upper)
    H_{i0+ib-1} ... H_{i0+1} H_{i0} = I - V T' V^T     (reverse product, ``T'`` lower)

so one block costs one pass over ``V`` instead of ``ib`` dependent passes, and
each pass reads ``ib`` reflector entries of a row instead of one strided entry.
``T`` depends only on the reflectors, not on the target, so ``wy_factor``
builds every block factor in one launch — one threadgroup per block, two
threads per entry of ``V^T V`` (that scan is latency bound), then thread 0 runs
the ``ib x ib`` triangular recursion — and the update kernel then replays
``ceil(k / ib)`` blocks per target column. Block ``i`` of the chain is applied
before block ``i-1`` for a right-hand product and after it for a left-hand one;
``reverse`` carries that order and ``upper`` carries which triangle of ``T``
the product needs.

One threadgroup owns one column of the target. Up to ``m = 2048`` that column
lives in the registers of the threadgroup, one or two rows per thread, so the
block's reflector entries stay in registers between the two halves of the
update and the target never leaves the chip. Above that it is kept in
threadgroup memory up to ``m = 4096`` and in the output buffer beyond, with the
same block structure. The reflector storage is read in whichever of the two
dense layouts it arrives in — ``torch.geqrf`` hands back the column-major one,
which needs no copy.

``ormqr`` replays the reflectors without forming ``Q``; no QR factorisation is
performed anywhere here. The reference for verification is ``torch.ormqr`` on
the CPU, which is LAPACK ``dormqr``, plus ``torch.linalg.householder_product``
on the CPU for the second overlay.
"""

from functools import lru_cache

import torch

_MAX_SHARED = 4096
"""Largest ``m`` whose target column is kept in threadgroup memory (16 KB)."""

_GROUP = 256
"""Threads per target column when the column does not fit in registers."""

_REG_MAX = 2048
"""Largest ``m`` whose target column fits in the registers of one threadgroup."""

_GROUP_MAX = 1024
"""Threads per target column in the register variant: one row per thread up to
this many rows, two rows per thread above it."""

_BLOCK = 16
"""Reflectors per block. The update cost falls with a larger block and the
factorisation cost rises with it, since ``V^T V`` has ``ib (ib - 1) / 2``
entries per block; 16 is where the two meet on this machine."""

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// A preprocessor literal, not a `constant uint`: the block arrays need a
// compile-time bound or they land in local memory instead of registers.
#define IB @IB@u
// Target rows per thread in the register variant; 0 selects the variant that
// keeps the column in threadgroup or output memory instead.
#define ROWS @ROWS@u

@ACCESS@

// Compact WY factor of one block: ``T`` such that the block's product of
// reflectors is I - V T V^T, upper triangular for the forward product
// H_i0 ... H_i0+ib-1 and lower triangular for the reverse one.
inline void wy_block(
    device const float* in_b,
    device const float* tau_b,
    threadgroup float (*g0)[IB],
    threadgroup float (*g1)[IB],
    threadgroup float (*ts)[IB],
    device float* T,
    uint i0,
    uint kk,
    uint m,
    uint tid,
    uint tg,
    uint upper)
{
    for (uint i = tid; i < IB * IB; i += tg) ts[i / IB][i % IB] = 0.0f;
    // Entry (a, j) of V^T V for a < j. v_a is unit at row i0 + a, so the sum
    // starts at the unit of the later vector and covers the trailing rows.
    // Two threads share one entry, each taking every other row with four
    // accumulators: this scan is latency bound and needs loads in flight.
    uint p = tid >> 1, part = tid & 1;
    if (p < kk * (kk - 1) / 2) {
        uint a = 0, j = 0, rem = p;
        for (uint aa = 0; aa < kk - 1; ++aa) {
            uint cnt = kk - 1 - aa;
            if (rem < cnt) { a = aa; j = aa + 1 + rem; break; }
            rem -= cnt;
        }
        uint ra = i0 + a, rj = i0 + j;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        uint r = rj + 1 + part;
        for (; r + 6 < m; r += 8) {
            a0 += VC(r, ra) * VC(r, rj);
            a1 += VC(r + 2, ra) * VC(r + 2, rj);
            a2 += VC(r + 4, ra) * VC(r + 4, rj);
            a3 += VC(r + 6, ra) * VC(r + 6, rj);
        }
        for (; r < m; r += 2) a0 += VC(r, ra) * VC(r, rj);
        if (part == 0) g0[a][j] = VC(rj, ra) + a0 + a1 + a2 + a3;
        else g1[a][j] = a0 + a1 + a2 + a3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (uint jj = 0; jj < kk; ++jj) {
            float tj = tau_b[i0 + jj];
            for (uint a2 = 0; a2 < jj; ++a2) {
                float s = 0.0f;
                if (upper) {
                    for (uint q = a2; q < jj; ++q) {
                        s += ts[a2][q] * (g0[q][jj] + g1[q][jj]);
                    }
                    ts[a2][jj] = -tj * s;
                } else {
                    for (uint q = a2; q < jj; ++q) {
                        s += (g0[q][jj] + g1[q][jj]) * ts[q][a2];
                    }
                    ts[jj][a2] = -tj * s;
                }
            }
            ts[jj][jj] = tj;
        }
        for (uint i = 0; i < IB * IB; ++i) T[i] = ts[i / IB][i % IB];
    }
}

kernel void wy_factor(
    device const float* inp,
    device const float* tau,
    device float* tb,
    constant uint& m,
    constant uint& n,
    constant uint& k,
    constant uint& nb,
    constant uint& upper,
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg [[threads_per_threadgroup]])
{
    uint b = gid / nb, blk = gid % nb;
    uint i0 = blk * IB;
    threadgroup float g0[IB][IB];
    threadgroup float g1[IB][IB];
    threadgroup float ts[IB][IB];
    wy_block(inp + b * m * n, tau + b * k, g0, g1, ts, tb + (b * nb + blk) * (IB * IB),
             i0, min(IB, k - i0), m, tid, tg, upper);
}

kernel void ormqr_wy(
    device const float* inp,
    device const float* tb,
    device const float* other,
    device float* out,
    constant uint& m,
    constant uint& n,
    constant uint& k,
    constant uint& nrhs,
    constant uint& left,
    constant uint& reverse,
    constant uint& nb,
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint wid [[simdgroup_index_in_threadgroup]],
    uint nwarps [[simdgroups_per_threadgroup]])
{
    threadgroup float red[32][IB];
    threadgroup float yvec[IB];
    threadgroup float tvec[IB];
    @YS@
    uint b = gid / nrhs;
    uint c = gid % nrhs;
    uint boff = b * m * nrhs;
    uint in_base, out_base, in_stride, out_stride;
    if (left) {
        in_base = boff + c;      in_stride = nrhs;
        out_base = boff + c * m; out_stride = 1;
    } else {
        in_base = boff + c * m;  in_stride = 1;
        out_base = boff + c;     out_stride = nrhs;
    }
    device const float* in_b = inp + b * m * n;
    for (uint r = tid; r < m; r += tg)
        YS(r) = other[in_base + r * in_stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // A thread keeps the rows of one residue class for every block, so a row
    // never moves between threads and the blocks need no barrier between them.
    // left applies T from the left (T[j][a]), right from the right (T[a][j]).
    uint sa = left ? 1 : IB;
    uint sb = left ? IB : 1;
    for (uint step = 0; step < nb; ++step) {
        uint blk = reverse ? (nb - 1 - step) : step;
        uint i0 = blk * IB;
        uint kk = min(IB, k - i0);
        device const float* T = tb + (b * nb + blk) * (IB * IB);
        uint r0 = i0 - (i0 % tg) + tid;
        if (r0 < i0) r0 += tg;
        // Rows below the block's diagonal are the bulk and read the reflector
        // entries straight; the ib rows that carry the implicit unit need the
        // masked form, and taking them in a separate arm keeps the bulk free of
        // per-element branches, which otherwise serialize the loads.
        uint rdiag = i0 + kk;
        float acc[IB];
        #pragma unroll
        for (uint j = 0; j < IB; ++j) acc[j] = 0.0f;
        for (uint r = r0; r < m; r += tg) {
            float y = YS(r);
            if (r >= rdiag) {
                #pragma unroll
                for (uint j = 0; j < IB; ++j) {
                    if (j < kk) acc[j] += VJ(r, j, i0) * y;
                }
            } else {
                #pragma unroll
                for (uint j = 0; j < IB; ++j) {
                    if (j < kk) {
                        float x = (r > i0 + j) ? VJ(r, j, i0) : ((r == i0 + j) ? 1.0f : 0.0f);
                        acc[j] += x * y;
                    }
                }
            }
        }
        #pragma unroll
        for (uint j = 0; j < IB; ++j) {
            if (j < kk) {
                acc[j] = simd_sum(acc[j]);
                if (lane == 0) red[wid][j] = acc[j];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (wid == 0 && lane < kk) {
            float yv = 0.0f;
            for (uint w = 0; w < nwarps; ++w) yv += red[w][lane];
            yvec[lane] = yv;
        }
        if (wid == 0) {
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (lane < kk) {
                float t = 0.0f;
                #pragma unroll
                for (uint a = 0; a < IB; ++a) {
                    if (a < kk) t += T[a * sa + lane * sb] * yvec[a];
                }
                tvec[lane] = t;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = r0; r < m; r += tg) {
            if (r >= rdiag) {
                float s = 0.0f;
                #pragma unroll
                for (uint j = 0; j < IB; ++j) {
                    if (j < kk) s += VJ(r, j, i0) * tvec[j];
                }
                YS(r) -= s;
            } else {
                float s = 0.0f;
                #pragma unroll
                for (uint j = 0; j < IB; ++j) {
                    if (j < kk) {
                        float x = (r > i0 + j) ? VJ(r, j, i0) : ((r == i0 + j) ? 1.0f : 0.0f);
                        s += x * tvec[j];
                    }
                }
                YS(r) -= s;
            }
        }
    }
    @STORE@
}

kernel void ormqr_wy_reg(
    device const float* inp,
    device const float* tb,
    device const float* other,
    device float* out,
    constant uint& m,
    constant uint& n,
    constant uint& k,
    constant uint& nrhs,
    constant uint& left,
    constant uint& reverse,
    constant uint& nb,
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint wid [[simdgroup_index_in_threadgroup]],
    uint nwarps [[simdgroups_per_threadgroup]])
{
    threadgroup float red[32][IB];
    threadgroup float yvec[IB];
    threadgroup float tvec[IB];
    uint b = gid / nrhs;
    uint c = gid % nrhs;
    uint boff = b * m * nrhs;
    uint in_base, out_base, in_stride, out_stride;
    if (left) {
        in_base = boff + c;      in_stride = nrhs;
        out_base = boff + c * m; out_stride = 1;
    } else {
        in_base = boff + c * m;  in_stride = 1;
        out_base = boff + c;     out_stride = nrhs;
    }
    device const float* in_b = inp + b * m * n;
    // ROWS is at most 2, so the target column and the block of reflectors both
    // live in registers: one pass over the reflectors per block instead of two.
    float ysv[ROWS];
    #pragma unroll
    for (uint q = 0; q < ROWS; ++q) {
        uint row = tid + q * tg;
        ysv[q] = (row < m) ? other[in_base + row * in_stride] : 0.0f;
    }
    // left applies T from the left (T[j][a]), right from the right (T[a][j]).
    uint sa = left ? 1 : IB;
    uint sb = left ? IB : 1;
    uint clast = k - 1;
    for (uint step = 0; step < nb; ++step) {
        uint blk = reverse ? (nb - 1 - step) : step;
        uint i0 = blk * IB;
        uint kk = min(IB, k - i0);
        device const float* T = tb + (b * nb + blk) * (IB * IB);
        float acc[IB];
        float x[ROWS * IB];
        #pragma unroll
        for (uint j = 0; j < IB; ++j) acc[j] = 0.0f;
        #pragma unroll
        for (uint q = 0; q < ROWS; ++q) {
            uint row = tid + q * tg;
            float y = ysv[q];
            // Idle threads past the last row still run the loads, so the row is
            // clamped; their y is zero and their result is never stored.
            uint rr = min(row, m - 1);
            if (row >= i0 + kk) {
                #pragma unroll
                for (uint j = 0; j < IB; ++j) {
                    // Past the last reflector the entry is irrelevant: its row
                    // of T is zero, so the clamped column contributes nothing.
                    float v = VC(rr, min(i0 + j, clast));
                    x[q * IB + j] = v;
                    acc[j] += v * y;
                }
            } else {
                #pragma unroll
                for (uint j = 0; j < IB; ++j) {
                    float v = (row > i0 + j) ? VC(rr, min(i0 + j, clast))
                                            : ((row == i0 + j) ? 1.0f : 0.0f);
                    x[q * IB + j] = v;
                    acc[j] += v * y;
                }
            }
        }
        #pragma unroll
        for (uint j = 0; j < IB; ++j) {
            acc[j] = simd_sum(acc[j]);
            if (lane == 0) red[wid][j] = acc[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // The update runs at full block width, so the entries past the last
        // reflector of a partial block are written as zero.
        if (wid == 0 && lane < kk) {
            float yv = 0.0f;
            for (uint w = 0; w < nwarps; ++w) yv += red[w][lane];
            yvec[lane] = yv;
        }
        if (wid == 0) {
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (lane < IB) {
                float t = 0.0f;
                if (lane < kk) {
                    #pragma unroll
                    for (uint a = 0; a < IB; ++a) {
                        if (a < kk) t += T[a * sa + lane * sb] * yvec[a];
                    }
                }
                tvec[lane] = t;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint q = 0; q < ROWS; ++q) {
            float s = 0.0f;
            #pragma unroll
            for (uint j = 0; j < IB; ++j) s += x[q * IB + j] * tvec[j];
            // A thread past the last row owns no column entry. It still runs
            // the loads so the block stays branch-free, so its value has to be
            // pinned back to zero or it feeds every later dot.
            ysv[q] = (tid + q * tg < m) ? ysv[q] - s : 0.0f;
        }
    }
    #pragma unroll
    for (uint q = 0; q < ROWS; ++q) {
        uint row = tid + q * tg;
        if (row < m) out[out_base + row * out_stride] = ysv[q];
    }
}
"""


def _source(shared, column_major, rows, ib):
    """Materialise the kernel source for one storage, layout and block shape."""
    if shared:
        ys = f"threadgroup float ysbuf[{_MAX_SHARED}];\n#define YS(r) ysbuf[(r)]"
        store = (
            "for (uint r = tid; r < m; r += tg)\n"
            "        out[out_base + r * out_stride] = ysbuf[r];"
        )
    else:
        ys = "#define YS(r) out[out_base + (r) * out_stride]"
        store = ""
    if column_major:
        access = (
            "#define VJ(r, j, i0) in_b[(i0 + (j)) * m + (r)]\n"
            "#define VC(r, c) in_b[(c) * m + (r)]"
        )
    else:
        access = (
            "#define VJ(r, j, i0) in_b[(r) * n + i0 + (j)]\n"
            "#define VC(r, c) in_b[(r) * n + (c)]"
        )
    source = _SOURCE
    if rows == 0:
        # The register variant needs a compile-time row count; drop it when it
        # cannot be used so the kernel it belongs to is not even compiled.
        cut = source.index("kernel void ormqr_wy_reg(")
        source = source[:cut]
    return (
        source.replace("@IB@", str(ib))
        .replace("@ROWS@", str(rows))
        .replace("@ACCESS@", access)
        .replace("@YS@", ys)
        .replace("@STORE@", store)
    )


@lru_cache(maxsize=None)
def _library(shared, column_major, rows, ib):
    """Compile the shader on first use. Importing this module compiles nothing."""
    return torch.mps.compile_shader(_source(shared, column_major, rows, ib))


def _check(input, tau, other, left):
    """The argument contract ``aten::ormqr`` enforces on the CPU."""
    if input.ndim < 2:
        raise RuntimeError("torch.ormqr: input must have at least 2 dimensions.")
    if other.ndim < 2:
        raise RuntimeError("torch.ormqr: other must have at least 2 dimensions.")
    side = -2 if left else -1
    if input.ndim - tau.ndim != 1:
        raise RuntimeError(
            "torch.ormqr: Expected tau to have one dimension less than input, "
            f"but got tau.ndim equal to {tau.ndim} and input.ndim is equal to {input.ndim}"
        )
    if other.shape[side] != input.shape[-2]:
        raise RuntimeError(
            f"torch.ormqr: other.shape[{side}] must be equal to input.shape[-2]"
        )
    expected = min(other.shape[side], input.shape[-1])
    if tau.shape[-1] != expected:
        raise RuntimeError(
            f"torch.ormqr: tau.shape[-1] must be equal to "
            f"min(other.shape[{side}], input.shape[-1])"
        )
    if input.ndim != other.ndim:
        raise RuntimeError(
            "torch.ormqr: Expected other to have the same number of dimensions "
            f"as input, but got other.ndim equal to {other.ndim} and "
            f"input.ndim is equal to {input.ndim}"
        )
    if input.shape[:-2] != tau.shape[:-1]:
        raise RuntimeError(
            "torch.ormqr: Expected batch dimensions of tau to be equal to "
            f"input.shape[:-2], but got {tuple(tau.shape[:-1])}"
        )
    if input.shape[:-2] != other.shape[:-2]:
        raise RuntimeError(
            "torch.ormqr: Expected batch dimensions of other to be equal to "
            f"input.shape[:-2], but got {tuple(other.shape[:-2])}"
        )
    if tau.dtype != input.dtype:
        raise RuntimeError(
            "torch.ormqr: Expected input and tau to have the same dtype, "
            f"but input has dtype {input.dtype} and tau has dtype {tau.dtype}"
        )
    if other.dtype != input.dtype:
        raise RuntimeError(
            "torch.ormqr: Expected input and other to have the same dtype, "
            f"but input has dtype {input.dtype} and other has dtype {other.dtype}"
        )
    if tau.device != input.device:
        raise RuntimeError("Expected all tensors to be on the same device, but found at least two devices")
    if other.device != input.device:
        raise RuntimeError("Expected all tensors to be on the same device, but found at least two devices")
    if input.dtype != torch.float32:
        raise NotImplementedError(
            f"ormqr_mps: the Metal kernel handles float32 only, got {input.dtype}"
        )


def _check_product(input, tau):
    """The argument contract ``torch.linalg.householder_product`` enforces on the CPU."""
    name = "torch.linalg.householder_product"
    if input.ndim < 2:
        raise RuntimeError(f"{name}: input must have at least 2 dimensions.")
    if input.shape[-2] < input.shape[-1]:
        raise RuntimeError(
            f"{name}: input.shape[-2] must be greater than or equal to input.shape[-1]"
        )
    if input.ndim - tau.ndim != 1:
        raise RuntimeError(
            f"{name}: Expected tau to have one dimension less than input, but got "
            f"tau.ndim equal to {tau.ndim} and input.ndim is equal to {input.ndim}"
        )
    if tau.shape[-1] > input.shape[-1]:
        raise RuntimeError(
            f"{name}: input.shape[-1] must be greater than or equal to tau.shape[-1]"
        )
    if input.shape[:-2] != tau.shape[:-1]:
        raise RuntimeError(
            f"{name}: Expected batch dimensions of tau to be equal to input.shape[:-2], "
            f"but got {list(tau.shape[:-1])}"
        )
    if tau.dtype != input.dtype:
        raise RuntimeError(
            f"{name}: tau dtype {str(tau.dtype).replace('torch.', '').capitalize()} "
            f"does not match input dtype {str(input.dtype).replace('torch.', '').capitalize()}"
        )
    if tau.device != input.device:
        raise RuntimeError(
            "Expected all tensors to be on the same device, but found at least two devices"
        )
    if input.dtype != torch.float32:
        raise NotImplementedError(
            f"householder_product_mps: the Metal kernel handles float32 only, got {input.dtype}"
        )


def ormqr_mps(input, tau, other, left=True, transpose=False):
    """Apply ``Q`` from ``torch.geqrf(input, tau)`` to ``other`` on the MPS device.

    Mirrors ``torch.ormqr``: ``left=True`` needs ``other`` of shape
    ``(*, m, nrhs)`` and applies ``Q`` (``transpose=False``) or ``Q^T``;
    ``left=False`` needs ``other`` of shape ``(*, nrhs, m)`` and multiplies it
    by the same ``m x m`` factor from the right.

    Returns a tensor with the shape of ``other`` and the column-major layout the
    CPU kernel produces.
    """
    if input.device.type != "mps":
        raise ValueError("ormqr_mps needs tensors on the mps device")
    _check(input, tau, other, left)

    m, n = input.shape[-2], input.shape[-1]
    k = tau.shape[-1]
    nrhs = other.shape[-1] if left else other.shape[-2]
    batch = input.numel() // (m * n) if m and n else 0
    out = _apply(input, tau, other, left, transpose, batch, m, n, k, nrhs)
    return out.view(other.shape)


def householder_product_mps(input, tau):
    """First ``input.shape[-1]`` columns of ``Q = H_1 ... H_k`` on the MPS device.

    Same contract as ``torch.linalg.householder_product``: ``input`` is
    ``(*, m, n)`` with ``m >= n``, ``tau`` is ``(*, k)`` with ``k <= n``, and the
    result is ``(*, m, n)``. Columns past the reflectors are ``Q`` applied to the
    corresponding basis vectors, which is what this computes: the target is the
    ``m x n`` identity and the kernel applies the same reflector chain ``ormqr``
    uses.
    """
    if input.device.type != "mps":
        raise ValueError("householder_product_mps needs tensors on the mps device")
    _check_product(input, tau)

    m, n = input.shape[-2], input.shape[-1]
    k = tau.shape[-1]
    batch = input.numel() // (m * n) if m and n else 0
    eye = torch.eye(m, n, dtype=input.dtype, device=input.device)
    other = eye.unsqueeze(0) if batch == 1 else eye.expand(batch, m, n).contiguous()
    out = _apply(input, tau, other, True, False, batch, m, n, k, n)
    return out.view(input.shape)


def _prepare(input, tau, other):
    """Bring the operands into a layout the kernel reads without a copy.

    ``torch.geqrf`` returns the reflectors column-major, which the kernel reads
    natively, so the common case copies nothing.
    """
    column_major = input.transpose(-1, -2).is_contiguous()
    if not column_major and not input.is_contiguous():
        input = input.contiguous()
    if not tau.is_contiguous():
        tau = tau.contiguous()
    if not other.is_contiguous():
        other = other.contiguous()
    return input, tau, other, column_major


def _apply(input, tau, other, left, transpose, batch, m, n, k, nrhs):
    """Dispatch the Metal kernels and return the column-major result, unviewed."""
    flat = (batch, m, nrhs) if left else (batch, nrhs, m)
    out = torch.empty_strided(
        flat, _strides(flat, m, nrhs, left), dtype=other.dtype, device=other.device
    )
    if batch == 0 or k == 0 or nrhs == 0:
        out.copy_(other.reshape(flat))
        return out
    inp, tau, src, column_major = _prepare(input, tau, other)
    rows, ib, group = _shape(m)
    blocks = -(-k // ib)
    shared = rows == 0 and m <= _MAX_SHARED
    kernel = _library(shared, column_major, rows, ib)
    # One T per reflector block serves every target column of this call.
    factors = torch.empty(
        batch * blocks * ib * ib, dtype=torch.float32, device=other.device
    )
    kernel.wy_factor(
        inp, tau, factors, m, n, k, blocks, int(not transpose),
        threads=batch * blocks * (ib * ib), group_size=ib * ib,
    )
    apply = kernel.ormqr_wy_reg if rows else kernel.ormqr_wy
    apply(
        inp, factors, src, out, m, n, k, nrhs, int(left), int(left != transpose),
        blocks, threads=batch * nrhs * group, group_size=group,
    )
    return out


def _shape(m):
    """Kernel variant for this row count: (rows per thread, block, threadgroup).

    ``rows == 0`` selects the variant that keeps the target column in
    threadgroup or output memory, which is the only option once the column no
    longer fits in the registers of one threadgroup.
    """
    if m <= _REG_MAX:
        group = min(_GROUP_MAX, max(32, -(-m // 32) * 32))
        return -(-m // group), _BLOCK, group
    return 0, _BLOCK, _GROUP


def _strides(shape, m, nrhs, left):
    """Strides the CPU kernel produces: batch dims row-major, matrix column-major."""
    strides = []
    step = m * nrhs
    for size in reversed(shape[:-2]):
        strides.append(step)
        step *= size
    strides.reverse()
    strides += [1, m] if left else [1, nrhs]
    return tuple(strides)


def _ormqr_out(self, tau, other, left=True, transpose=False, out=None):
    result = ormqr_mps(self, tau, other, left, transpose)
    out.copy_(result)
    return out


OVERLAYS = (
    ("ormqr", ormqr_mps),
    ("ormqr.out", _ormqr_out),
    ("linalg_householder_product", householder_product_mps),
)
