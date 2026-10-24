"""MPS overlays for the general (non-symmetric) eigendecomposition.

``aten::linalg_eig`` and ``aten::_linalg_eigvals`` raise ``NotImplementedError``
on the MPS device. This module fills both with a Metal kernel.

The kernel is one threadgroup per matrix. It runs the whole decomposition in a
single dispatch:

1. Householder reduction to Hessenberg form, accumulating the orthogonal
   factor ``Z``. One threadgroup reduction per step: the reflector norm is
   reduced, and ``|v|^2`` follows in closed form from it.
2. Implicit shifted QR iteration on the Hessenberg matrix with complex
   Wilkinson shifts, accumulating ``Z``. Each sweep first scans up from the
   bottom for the nearest negligible subdiagonal and runs the step on that
   window only, so work shrinks as eigenvalues converge; a negligible bottom
   subdiagonal deflates one eigenvalue per pass. Each rotation is one barrier
   interval: the row update, the column update and the ``Z`` update are issued
   together, and the single thread that owns the 2x2 corner applies the left
   and right rotations there. The sweep budget is bounded and a matrix that
   does not converge makes the call raise.
3. Eigenvectors of the triangular factor by back-substitution, one thread per
   column and no barrier between steps (a column never reads another). Columns
   are normalised here, and optionally the inverse of the resulting upper
   triangular factor is formed.

Because the columns of ``Y`` leave the kernel with unit norm and ``Z`` is
unitary, the host forms ``V = Z @ Y`` with no normalisation pass, and
``V^-1 = Y^-1 Z^H``.

Everything is evaluated in ``complex64`` because MPS has no ``float64``. A real
``float32`` input is widened to ``complex64``; the result is ``complex64`` as
the operator specifies.

Autograd: PyTorch's registered formula for ``linalg_eig`` calls
``_linalg_solve_ex``, which lowers to ``linalg_lu_factor_ex`` and fails on
complex MPS tensors ("MPS doesn't support complex types"). The same formula,
reproduced in ``_EigBackward``, runs in complex ops MPS does support, so the
autograd path is registered separately on ``AutogradMPS`` and no other device
is touched.

Torch version: validated against 2.14.1.
"""

from functools import lru_cache

import torch

CHECKED_TORCH = "2.14.1"

#: Deflation tolerance, relative to the adjacent diagonal entries.
_QR_TOL = 1e-6

#: Sweeps allowed per row before the call fails to converge, plus a constant.
_SWEEPS_PER_ROW = 30
_SWEEPS_FLOOR = 100

#: Largest supported matrix order. The kernel keeps two ``N``-element complex
#: threadgroup arrays (16 bytes per row of the order) plus the reduction
#: scratch, which has to stay inside the 32 KiB threadgroup memory the device
#: guarantees.
_MAX_ORDER = 1024

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

constant uint N = $N;
constant uint T = $T;

inline float2 cmul(float2 a, float2 b) { return float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
inline float2 cconj(float2 a) { return float2(a.x, -a.y); }
inline float cabs2(float2 a) { return a.x*a.x + a.y*a.y; }
inline float2 cdiv(float2 a, float2 b) {
    float d = cabs2(b);
    return float2(a.x*b.x + a.y*b.y, a.y*b.x - a.x*b.y) / d;
}
inline float2 csqrt2(float2 a) {
    float r = sqrt(cabs2(a));
    float re = sqrt(max(0.5f*(r + a.x), 0.0f));
    float im = sqrt(max(0.5f*(r - a.x), 0.0f));
    return float2(re, a.y < 0.0f ? -im : im);
}

// Sum across the threadgroup, broadcast in red[0]. Reduce inside the simdgroup
// first, so only one value per simdgroup crosses threadgroup memory.
inline float tg_sum(threadgroup float* red, uint tid, uint lane, uint sg, uint nsg, float val) {
    float s = simd_sum(val);
    if (lane == 0) red[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = red[0];
        for (uint i = 1; i < nsg; ++i) total += red[i];
        red[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return red[0];
}

inline float tg_max(threadgroup float* red, uint tid, uint lane, uint sg, uint nsg, float val) {
    float s = simd_max(val);
    if (lane == 0) red[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float top = red[0];
        for (uint i = 1; i < nsg; ++i) top = max(top, red[i]);
        red[0] = top;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return red[0];
}

kernel void $KERNEL(
    device float2* A,        // (N, N) complex, overwritten with the Schur factor
    device float2* Z,        // (N, N) complex, overwritten with the accumulated unitary
    device float2* Y,        // (N, N) complex eigenvectors of the Schur factor
    device float2* Yinv,     // (N, N) complex inverse of Y, written when do_inv
    device int* stats,       // per matrix: {did not converge, sweeps, rotations}
    constant float& tol,
    constant uint& max_sweeps,
    constant uint& do_inv,
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]]) {

    threadgroup float2 sh_v[N];
    threadgroup float2 sh_t[N];
    threadgroup float red[T];

    device float2* h = A + (ulong)tg * N * N;
    device float2* z = Z + (ulong)tg * N * N;
    device float2* y = Y + (ulong)tg * N * N;

    // Z starts as the identity; every element is written here so the caller can
    // hand over uninitialised memory.
    for (uint i = tid; i < N * N; i += T) z[i] = (i % (N + 1) == 0) ? float2(1.0f, 0.0f) : float2(0.0f, 0.0f);
    if (tid == 0) { stats[3 * tg] = 0; stats[3 * tg + 1] = 0; stats[3 * tg + 2] = 0; }

    // ---- Hessenberg reduction, H <- (I - beta v v^H) H (I - beta v v^H) ----
    for (uint k = 0; k + 2 < N; ++k) {
        float loc = 0.0f;
        for (uint i = k + 1 + tid; i < N; i += T) loc += cabs2(h[i * N + k]);
        float nrm = sqrt(tg_sum(red, tid, lane, sg, nsg, loc));
        float2 x0 = h[(k + 1) * N + k];
        float ax0 = length(x0);
        float2 ph = (ax0 > 0.0f) ? x0 / ax0 : float2(1.0f, 0.0f);
        float2 alpha = -ph * nrm;
        // v = x - alpha e1, so |v|^2 = 2 nrm (nrm + |x_1|): no second reduction.
        float vtv = 2.0f * nrm * (nrm + ax0);
        float beta = (vtv > 0.0f) ? 2.0f / vtv : 0.0f;
        for (uint i = k + 1 + tid; i < N; i += T) {
            float2 xi = h[i * N + k];
            if (i == k + 1) xi -= alpha;
            sh_v[i - (k + 1)] = xi;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = k + tid; j < N; j += T) {
            // The sum walks down a column (stride N), so four accumulators keep
            // four loads in flight instead of stalling on each one.
            float2 s0 = float2(0.0f, 0.0f), s1 = s0, s2 = s0, s3 = s0;
            uint i = k + 1;
            for (; i + 3 < N; i += 4) {
                s0 += cmul(cconj(sh_v[i - k - 1]), h[i * N + j]);
                s1 += cmul(cconj(sh_v[i - k]), h[(i + 1) * N + j]);
                s2 += cmul(cconj(sh_v[i - k + 1]), h[(i + 2) * N + j]);
                s3 += cmul(cconj(sh_v[i - k + 2]), h[(i + 3) * N + j]);
            }
            for (; i < N; ++i) s0 += cmul(cconj(sh_v[i - k - 1]), h[i * N + j]);
            sh_t[j] = beta * ((s0 + s1) + (s2 + s3));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = k + 1 + tid; i < N; i += T) {
            float2 vi = sh_v[i - k - 1];
            for (uint j = k; j < N; ++j) h[i * N + j] -= cmul(vi, sh_t[j]);
        }
        threadgroup_barrier(mem_flags::mem_device);
        for (uint i = tid; i < N; i += T) {
            float2 su = float2(0.0f, 0.0f), sz = float2(0.0f, 0.0f);
            for (uint j = k + 1; j < N; ++j) {
                su += cmul(h[i * N + j], sh_v[j - k - 1]);
                sz += cmul(z[i * N + j], sh_v[j - k - 1]);
            }
            su *= beta;
            sz *= beta;
            for (uint j = k + 1; j < N; ++j) {
                float2 vc = cconj(sh_v[j - k - 1]);
                h[i * N + j] -= cmul(su, vc);
                z[i * N + j] -= cmul(sz, vc);
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }

    // ---- shifted QR iteration to triangular form ----
    uint m = N;
    uint sweeps = 0;
    uint rotations = 0;
    uint stall = 0;
    while (m > 1 && sweeps < max_sweeps) {
        threadgroup_barrier(mem_flags::mem_device);
        // Nearest split: the highest negligible subdiagonal. The step then runs
        // on the block it separates from the converged part above, and the same
        // test at the bottom deflates one eigenvalue. One parallel pass plus one
        // reduction, because a serial walk costs a dependent load per row.
        float split = 0.0f;
        for (uint i = 1 + tid; i < m; i += T) {
            float2 sb = h[i * N + (i - 1)];
            float sc = length(h[(i - 1) * N + (i - 1)]) + length(h[i * N + i]);
            if (length(sb) <= tol * max(sc, 1e-30f)) split = (float)i;
        }
        uint l = (uint)tg_max(red, tid, lane, sg, nsg, split);
        if (l == m - 1) {
            if (tid == 0) h[(m - 1) * N + (m - 2)] = float2(0.0f, 0.0f);
            threadgroup_barrier(mem_flags::mem_device);
            m -= 1;
            stall = 0;
            continue;
        }
        float2 a = h[(m - 2) * N + (m - 2)];
        float2 b = h[(m - 2) * N + (m - 1)];
        float2 c = h[(m - 1) * N + (m - 2)];
        float2 d = h[(m - 1) * N + (m - 1)];
        float2 mu;
        if (stall >= 20) {
            // Exceptional shift: breaks cycles on matrices with repeated roots.
            float mag = length(c);
            mu = d + float2(0.6f * mag, 0.8f * mag);
            stall = 0;
        } else {
            float2 tr = 0.5f * (a + d);
            float2 q = 0.5f * (a - d);
            float2 sq = csqrt2(cmul(q, q) + cmul(b, c));
            float2 mu1 = tr + sq;
            float2 mu2 = tr - sq;
            mu = (cabs2(mu1 - d) < cabs2(mu2 - d)) ? mu1 : mu2;
        }
        stall += 1;
        rotations += m - 1 - l;

        for (uint k = l; k + 1 < m; ++k) {
            threadgroup_barrier(mem_flags::mem_device);
            float2 x, yv;
            if (k == l) {
                x = h[l * N + l] - mu;
                yv = h[(l + 1) * N + l];
            } else {
                x = h[k * N + (k - 1)];
                yv = h[(k + 1) * N + (k - 1)];
            }
            float r = sqrt(cabs2(x) + cabs2(yv));
            float cc;
            float2 ss;
            if (r == 0.0f) {
                cc = 1.0f;
                ss = float2(0.0f, 0.0f);
            } else {
                float ax = length(x);
                cc = ax / r;
                float2 p = (ax > 0.0f) ? x / ax : float2(1.0f, 0.0f);
                ss = cmul(p, cconj(yv)) / r;
            }
            // One barrier interval per rotation. Row update, column update and
            // Z update all read the pre-rotation matrix, so they share one pass:
            // the three sets of loads issue together and only their stores wait
            // on the rotation. Thread 0 owns the 2x2 corner the first two share.
            uint lo = (k > l) ? k - 1 : l;
            uint hi = min(k + 2, m - 1);
            for (uint i = tid; i < N; i += T) {
                bool row = (i >= lo && i != k && i != k + 1);
                bool col = (i <= hi && i != k && i != k + 1);
                float2 r0 = float2(0.0f, 0.0f), r1 = float2(0.0f, 0.0f);
                float2 c0 = float2(0.0f, 0.0f), c1 = float2(0.0f, 0.0f);
                float2 a00 = float2(0.0f, 0.0f), a01 = a00, a10 = a00, a11 = a00;
                float2 z0 = z[i * N + k];
                float2 z1 = z[i * N + k + 1];
                if (row) { r0 = h[k * N + i]; r1 = h[(k + 1) * N + i]; }
                if (col) { c0 = h[i * N + k]; c1 = h[i * N + k + 1]; }
                if (i == k) { a00 = h[k * N + k]; a01 = h[k * N + k + 1];
                              a10 = h[(k + 1) * N + k]; a11 = h[(k + 1) * N + k + 1]; }
                if (row) {
                    h[k * N + i] = cc * r0 + cmul(ss, r1);
                    h[(k + 1) * N + i] = cmul(-cconj(ss), r0) + cc * r1;
                }
                if (col) {
                    h[i * N + k] = cc * c0 + cmul(cconj(ss), c1);
                    h[i * N + k + 1] = cmul(-ss, c0) + cc * c1;
                }
                if (i == k) {
                    // The two rows first, then the two columns: this thread owns
                    // the 2x2 block both updates would otherwise race on.
                    float2 b00 = cc * a00 + cmul(ss, a10);
                    float2 b01 = cc * a01 + cmul(ss, a11);
                    float2 b10 = cmul(-cconj(ss), a00) + cc * a10;
                    float2 b11 = cmul(-cconj(ss), a01) + cc * a11;
                    h[k * N + k] = cc * b00 + cmul(cconj(ss), b01);
                    h[k * N + k + 1] = cmul(-ss, b00) + cc * b01;
                    h[(k + 1) * N + k] = cc * b10 + cmul(cconj(ss), b11);
                    h[(k + 1) * N + k + 1] = cmul(-ss, b10) + cc * b11;
                }
                z[i * N + k] = cc * z0 + cmul(cconj(ss), z1);
                z[i * N + k + 1] = cmul(-ss, z0) + cc * z1;
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
        sweeps += 1;
    }
    // Chased-out bulges leave rounding dust below the subdiagonal. One pass at
    // the end is enough: each rotation consumes the previous one's bulge, and
    // the triangular solve above reads the upper triangle only.
    for (uint i = tid; i < N * N; i += T) {
        uint rowidx = i / N, colidx = i % N;
        if (colidx + 1 < rowidx) h[i] = float2(0.0f, 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
        stats[3 * tg] = (m > 1) ? 1 : 0;
        stats[3 * tg + 1] = (int)sweeps;
        stats[3 * tg + 2] = (int)rotations;
    }

    float locmax = 0.0f;
    for (uint i = tid; i < N * N; i += T) locmax = max(locmax, length(h[i]));
    float tiny = 1e-6f * max(tg_max(red, tid, lane, sg, nsg, locmax), 1.0f);

    // ---- eigenvectors of the triangular factor ----
    // Column j of Y only ever reads column j, so there is no barrier between
    // the rows: each thread walks its own columns top-down.
    for (uint j = tid; j < N; j += T) {
        for (int i = (int)N - 1; i >= 0; --i) {
            float2 val = float2(0.0f, 0.0f);
            if (j == (uint)i) {
                val = float2(1.0f, 0.0f);
            } else if (j > (uint)i) {
                // y is read down a column and h across a row: two accumulators
                // hide the strided half.
                float2 s0 = float2(0.0f, 0.0f), s1 = s0;
                uint k = i + 1;
                for (; k + 1 <= j; k += 2) {
                    s0 += cmul(h[i * N + k], y[k * N + j]);
                    s1 += cmul(h[i * N + k + 1], y[(k + 1) * N + j]);
                }
                for (; k <= j; ++k) s0 += cmul(h[i * N + k], y[k * N + j]);
                float2 s = s0 + s1;
                float2 den = h[i * N + i] - h[j * N + j];
                // Repeated eigenvalues leave the eigenvector undefined; a
                // relative perturbation mirrors LAPACK's handling of the
                // defective case instead of dividing by zero.
                if (cabs2(den) < tiny * tiny) den += float2(tiny, 0.0f);
                val = -cdiv(s, den);
            }
            y[i * N + j] = val;
        }
        // Unit columns: Z is unitary, so V = Z Y needs no host normalisation.
        float nrm = 0.0f;
        for (uint i = 0; i < N; ++i) nrm += cabs2(y[i * N + j]);
        float inv = (nrm > 0.0f) ? rsqrt(nrm) : 1.0f;
        for (uint i = 0; i < N; ++i) y[i * N + j] = y[i * N + j] * inv;
    }

    // ---- inverse of the upper triangular Y ----
    if (do_inv) {
        // Unlike the triangular solve above, row i of Y is read across columns,
        // so the normalised columns of every thread must be visible first.
        threadgroup_barrier(mem_flags::mem_device);
        device float2* yi = Yinv + (ulong)tg * N * N;
        for (uint j = tid; j < N; j += T) {
            for (uint i = j + 1; i < N; ++i) yi[i * N + j] = float2(0.0f, 0.0f);
            for (int i = (int)j; i >= 0; --i) {
                float2 s = (i == (int)j) ? float2(1.0f, 0.0f) : float2(0.0f, 0.0f);
                for (uint k = i + 1; k <= j; ++k) s -= cmul(y[i * N + k], yi[k * N + j]);
                // Column normalisation made the diagonal non-unit.
                yi[i * N + j] = cdiv(s, y[i * N + i]);
            }
        }
    }
}
"""


@lru_cache(maxsize=16)
def _kernel(n, threads):
    """Compiled ``eig_decompose`` for order ``n`` and threadgroup width ``threads``.

    The kernel name carries ``n`` and ``threads``. Torch resolves a compiled
    shader library by function name, and two libraries that define the same name
    in one process do not reliably run their own body, so a process that
    decomposes several matrix orders would mix them up.
    """
    name = f"eig_decompose_{n}_{threads}"
    source = (
        _SOURCE.replace("$N", str(n))
        .replace("$T", str(threads))
        .replace("$KERNEL", name)
    )
    return getattr(torch.mps.compile_shader(source), name)


def _threads_for(n):
    """Threadgroup width: wide enough to spread the row/column updates."""
    for width in (32, 64, 128, 256):
        if n <= width:
            return width
    return 256


def _check_input(A):
    if A.dim() < 2:
        raise RuntimeError(
            f"linalg.eig: Expected a tensor with at least 2 dimensions, got {A.dim()}"
        )
    if A.shape[-1] != A.shape[-2]:
        raise RuntimeError(
            "linalg.eig: Expected a square matrix, "
            f"but got a matrix of size {tuple(A.shape[-2:])}"
        )
    if A.dtype not in (torch.float32, torch.complex64):
        raise TypeError(
            "linalg.eig on MPS needs float32 or complex64 (MPS has no float64), "
            f"got {A.dtype}"
        )
    if A.shape[-1] > _MAX_ORDER:
        raise RuntimeError(
            f"linalg.eig on MPS supports matrices up to order {_MAX_ORDER}, "
            f"got {A.shape[-1]}"
        )


def _decompose(A, need_inverse, need_vectors=True):
    """Run the Metal decomposition.

    Args:
        A: (..., n, n) float32 or complex64 on MPS.
        need_inverse: also return ``V^{-1}``.
        need_vectors: form ``V = Z @ Y``. When false the product is skipped and
            ``None`` is returned in its place.

    Returns:
        (w, V[, W]) with complex64 ``w`` of shape ``(..., n)``, ``V`` of shape
        ``(..., n, n)`` and ``W = V^{-1}`` of the same shape.
    """
    _check_input(A)
    batch_shape = A.shape[:-2]
    n = A.shape[-1]
    out_shape = batch_shape + (n,)
    if n == 0:
        empty = A.new_empty(out_shape, dtype=torch.complex64)
        vectors = A.new_empty(batch_shape + (n, n), dtype=torch.complex64)
        return (empty, vectors, vectors.clone()) if need_inverse else (empty, vectors)

    batch = 1
    for size in batch_shape:
        batch *= size
    if batch == 0:
        empty = A.new_empty(out_shape, dtype=torch.complex64)
        vectors = A.new_empty(batch_shape + (n, n), dtype=torch.complex64)
        return (empty, vectors, vectors.clone()) if need_inverse else (empty, vectors)
    flat = A.reshape(batch, n, n)
    work = flat.to(torch.complex64)
    if work.data_ptr() == flat.data_ptr():
        # Complex input: ``to`` is a no-op and the kernel writes in place.
        work = work.clone()
    work = work.contiguous()
    # The kernel writes every element of these three, so they start empty:
    # ``torch.empty`` is an allocator call, not a dispatch.
    unitary = torch.empty(batch, n, n, dtype=torch.complex64, device=A.device)
    vectors_y = torch.empty(batch, n, n, dtype=torch.complex64, device=A.device)
    stats = torch.empty(batch, 3, dtype=torch.int32, device=A.device)
    threads = _threads_for(n)
    max_sweeps = _SWEEPS_PER_ROW * n + _SWEEPS_FLOOR

    work_real = torch.view_as_real(work)
    unitary_real = torch.view_as_real(unitary)
    y_real = torch.view_as_real(vectors_y)
    if need_inverse:
        inverse_y = torch.empty(batch, n, n, dtype=torch.complex64, device=A.device)
        inverse_real = torch.view_as_real(inverse_y)
    else:
        inverse_real = y_real
    _kernel(n, threads)(
        work_real,
        unitary_real,
        y_real,
        inverse_real,
        stats,
        _QR_TOL,
        max_sweeps,
        1 if need_inverse else 0,
        threads=batch * threads,
        group_size=threads,
    )

    schur = torch.view_as_complex(work_real)
    unitary = torch.view_as_complex(unitary_real)
    vectors_y = torch.view_as_complex(y_real)
    values = torch.diagonal(schur, dim1=-2, dim2=-1).reshape(out_shape)
    # The kernel leaves Y with unit columns, so no normalisation pass is due.
    vectors = (unitary @ vectors_y).reshape(batch_shape + (n, n)) if need_vectors else None
    # Queued after the matmul, so this one sync waits for the whole chain.
    failed = bool(torch.any(stats[:, 0]))
    if failed:
        raise RuntimeError(
            "linalg.eig: The algorithm failed to converge because the input "
            "matrix is ill-conditioned or has too many repeated eigenvalues"
        )
    if not need_inverse:
        return values, vectors
    inverse_y = torch.view_as_complex(inverse_real)
    # V = Z Y with Z unitary and the columns of Y of unit norm, so V^-1 = Y^-1 Z^H.
    inverse = (inverse_y @ unitary.conj().transpose(-1, -2)).reshape(batch_shape + (n, n))
    return values, vectors, inverse


def _eig(self):
    values, vectors = _decompose(self, need_inverse=False)
    return values, vectors


def _eigvals(self):
    values, _ = _decompose(self, need_inverse=False, need_vectors=False)
    return values


def _eig_out(self, *, eigenvalues, eigenvectors):
    values, vectors = _decompose(self, need_inverse=False)
    eigenvalues.resize_(values.shape).copy_(values)
    eigenvectors.resize_(vectors.shape).copy_(vectors)
    return eigenvalues, eigenvectors


def _eigvals_out(self, *, out):
    values, _ = _decompose(self, need_inverse=False, need_vectors=False)
    out.resize_(values.shape).copy_(values)
    return out


class _EigBackward:
    """PyTorch's ``linalg_eig_backward`` in ops that MPS supports.

    The registered C++ formula is

        gA = V^{-H} (diag(gL) + (V^H gV - V^H V diag(Re diag(V^H gV))) / E*) V^H

    with ``E*_ij = conj(L_i - L_j)`` off the diagonal and 1 on it. It is
    evaluated here with matmuls and ``W = V^{-1}``, which the forward kernel
    computes.
    """

    @staticmethod
    def _grad(gw, gV, w, V, W, input_is_complex):
        vh = V.conj().transpose(-1, -2)
        wh = W.conj().transpose(-1, -2)
        if gV is None:
            grad = wh @ (gw.unsqueeze(-1) * vh)
        else:
            vhgv = vh @ gV
            diagonal = torch.diagonal(vhgv, dim1=-2, dim2=-1)
            if V.is_complex():
                imaginary = diagonal.imag
                if not torch.allclose(
                    imaginary, torch.zeros_like(imaginary), rtol=1e-2, atol=1e-2
                ):
                    raise RuntimeError(
                        "linalg_eig_backward: The eigenvectors in the complex "
                        "case are specified up to multiplication by e^{i phi}. "
                        "The specified loss function depends on this quantity, "
                        "so it is ill-defined."
                    )
            projected = vhgv - vh @ (V * diagonal.real.unsqueeze(-2))
            econj = w.conj().unsqueeze(-2) - w.conj().unsqueeze(-1)
            econj.diagonal(dim1=-2, dim2=-1).fill_(1.0)
            mixed = projected / econj
            if gw is not None:
                mixed.diagonal(dim1=-2, dim2=-1).copy_(gw)
            grad = wh @ (mixed @ vh)
        if not input_is_complex:
            grad = grad.real
        return grad


class _Eig(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        values, vectors, inverse = _decompose(A, need_inverse=True)
        ctx.save_for_backward(values, vectors, inverse)
        ctx.input_is_complex = A.is_complex()
        return values, vectors

    @staticmethod
    def backward(ctx, grad_values, grad_vectors):
        values, vectors, inverse = ctx.saved_tensors
        grad = _EigBackward._grad(
            grad_values, grad_vectors, values, vectors, inverse, ctx.input_is_complex
        )
        return grad


class _EigVals(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        values, vectors, inverse = _decompose(A, need_inverse=True)
        ctx.save_for_backward(values, vectors, inverse)
        ctx.input_is_complex = A.is_complex()
        return values

    @staticmethod
    def backward(ctx, grad_values):
        values, vectors, inverse = ctx.saved_tensors
        # No eigenvector gradient exists for eigvals; the formula collapses to
        # V^{-H} diag(gL) V^H.
        return _EigBackward._grad(
            grad_values, None, values, vectors, inverse, ctx.input_is_complex
        )


def _eig_autograd(self):
    return _Eig.apply(self)


def _eigvals_autograd(self):
    return _EigVals.apply(self)


#: ``(aten overload name, callable)`` pairs for the MPS dispatch key.
OVERLAYS = (
    ("linalg_eig", _eig),
    ("linalg_eig.out", _eig_out),
    ("_linalg_eigvals", _eigvals),
    ("linalg_eigvals.out", _eigvals_out),
)

#: ``(aten overload name, callable)`` pairs for ``AutogradMPS``. Registering
#: these keeps the built-in formula for every other device.
AUTOGRAD_OVERLAYS = (
    ("linalg_eig", _eig_autograd),
    ("_linalg_eigvals", _eigvals_autograd),
)
