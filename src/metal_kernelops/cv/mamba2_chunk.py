"""Mamba-2 chunked SSD scan on MPS: an exact torch composition, no CUDA/Triton.

The function computed is ``mamba_chunk_scan_combined`` from
``mamba_ssm/ops/triton/ssd_combined.py``, i.e. the SSD recurrence of
``mamba_ssm/modules/ssd_minimal.py`` with the discretization and the optional
pieces applied in the same order as the Triton forward:

    dt_t  = softplus(dt_t + dt_bias)                      (dt_softplus)
    a_t   = exp(dt_t * A)                                 A negative, per head
    s_t   = a_t * s_{t-1} + dt_t * x_t (x) B_t            s (batch, head, d_head, d_state)
    y_t   = <C_t, s_t> + D * x_t                          D (heads,) or (heads, d_head)
    y_t   = y_t * silu(z_t)                               z gate
    final = s_{L-1}

Algorithm
---------
The sequence is cut into chunks of ``chunk_size`` and every chunk is handled
with matrix products instead of a per-token loop, following the block
factorization of the SSD paper:

1. ``dA`` is accumulated per chunk: ``dcs[l] = sum_{k<=l} dt_k A``.
2. Chunk states ``S_c = sum_l dt_l exp(dcs[last] - dcs[l]) x_l (x) B_l`` — one
   grouped matrix product over all chunks at once.
3. The chunk-to-chunk transfer is a causal ``(#chunks, #chunks)`` decay matrix
   applied to ``S``, plus the decayed initial state: the closed form of
   ``state_passing_ref``. It gives the state entering every chunk and the state
   after the last one.
4. Intra-chunk output ``y = tril(exp(dcs_l - dcs_s) <C_l, B_s> dt_s) x`` plus the
   inter-chunk term ``exp(dcs_l) <C_l, entering state>``, both matrix products
   per group of heads, so grouped-query ``B``/``C`` are shared without
   materializing the repeat.

Every step is a batched ``torch.matmul`` composition, so the autograd graph is
the exact derivative of the recurrence and every differentiable input gets a
gradient. Nothing here registers on a dispatch key; the module is a plain Python
entry point.

Layout contract (``mamba_chunk_scan_combined``)
-----------------------------------------------
* ``x``     ``(batch, length, nheads, d_head)``
* ``dt``    ``(batch, length, nheads)``
* ``A``     ``(nheads,)``, negative
* ``B``, ``C`` ``(batch, length, ngroups, d_state)``, ``nheads % ngroups == 0``
* ``D``     ``(nheads,)`` or ``(nheads, d_head)``
* ``z``     ``(batch, length, nheads, d_head)``
* ``dt_bias`` ``(nheads,)``
* ``initial_states`` ``(batch, nheads, d_head, d_state)``, the
  ``mamba_chunk_scan_combined`` order. The transposed ``(batch, nheads, d_state,
  d_head)`` spelling is accepted when ``d_state != d_head``; when the two are
  equal the canonical ``(batch, nheads, d_head, d_state)`` reading is used. A
  leading singleton chunk axis (the ``ssd_minimal`` form) is dropped.
* returns ``(y, final_state)`` with ``y`` ``(batch, length, nheads, d_head)`` in
  ``x.dtype`` and ``final_state`` ``(batch, nheads, d_head, d_state)``.

``length`` need not be a multiple of ``chunk_size``: ``dt`` is zero-extended for
the tail chunk, which contributes no input and unit decay, exactly like the
Triton kernel's ``chunk_size_limit`` masking. The returned ``y`` is sliced back
to ``length``.

Dtype
-----
``dt``, ``A``, the cumulative decay, the chunk states and the decay matrices are
computed in float32 whatever the input dtype, exactly as the Triton kernels keep
``dA_cumsum``, ``chunk_state`` and ``state_passing`` in float32. The score
matrices and the head-state products run in ``x.dtype`` when it is
float16/bfloat16 (MPS matrix multiplication supports both). ``y`` and
``final_state`` come back in the input dtype. Measured against the float64
recurrence: 2e-7 relative in float32, 6e-4 in float16, 5e-3 in bfloat16.

Memory
------
Peak is dominated by the score matrix ``nheads * length * chunk_size`` and the
per-chunk state ``batch * nheads * length * d_state``, both float32 and linear in
``length``.

Cost
----
The chunk-to-chunk transfer is a ``(#chunks, #chunks)`` product, so its time and
memory grow quadratically in ``length / chunk_size`` while everything else grows
linearly. Raising ``chunk_size`` is the knob for very long sequences; 32 is the
measured optimum for model-like shapes because the score matrix grows with it.

Measured
--------
Torch 2.14.1 on an M5 Max, float32, forward only, best of 15 warmed and
synchronized runs (see ``measured/mamba2_chunk.json``):

=======================  =====  ========  ===============  ========
shape                    chunk  scan      step recurrence  speed-up
=======================  =====  ========  ===============  ========
1x2048, H24 P64 N128     32     2.15 ms   96.8 ms          45x
4x1024, H32 P64 N128 G8  32     6.52 ms   153.1 ms         23x
1x4096, H64 P64 N64      32     8.04 ms   224.5 ms         28x
=======================  =====  ========  ===============  ========

No Metal kernel is used. A fused ``compile_shader`` kernel for the largest
elementwise step (``exp(dcs_l - dcs_s) * <C_l, B_s> * dt_s``, five passes in the
composition) measured 0.53 ms against 1.31 ms on the 4x1024 shape, of which
0.28 ms is the ``expf`` itself: 12% of the forward, and it would need a second
hand-written backward for the gradient path. The composed matrix products and
transposes around it are already at memory bandwidth, so the fusion is not
included.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F

__all__ = ["mamba2_chunk_scan", "mamba2_chunk_scan_reference"]

_HALF = (torch.float16, torch.bfloat16)

_DTYPE_NAMES = {
    torch.float32: ("f32", "float"),
    torch.float16: ("f16", "half"),
    torch.bfloat16: ("bf16", "bfloat"),
}

# Option bits, shared by the kernels below.
_HAS_Z = 1
_HAS_D = 2
_D_PER_HEAD = 4
_HAS_INIT = 8
_SOFTPLUS = 16
_HAS_BIAS = 32

_MAX_CHUNK = 64
"""Largest chunk whose L x L scores still fit in threadgroup memory."""

_MAX_THREADS = 1024
_TG_BUDGET = 28 * 1024
"""Threadgroup bytes the fused path plans for; the hardware limit is 32 KiB."""

_HELPERS = r"""
#include <metal_stdlib>
using namespace metal;

inline float loadv(device const float* p, long i) { return p[i]; }
inline float loadv(device const half* p, long i) { return float(p[i]); }
inline float loadv(device const bfloat* p, long i) { return float(p[i]); }
inline void storev(device float* p, long i, float v) { p[i] = v; }
inline void storev(device half* p, long i, float v) { p[i] = half(v); }
inline void storev(device bfloat* p, long i, float v) { p[i] = bfloat(v); }
inline float4 load4(device const float* p, long i) { return *(device const float4*)(p + i); }
inline float4 load4(device const half* p, long i) {
    return float4(float(p[i]), float(p[i + 1]), float(p[i + 2]), float(p[i + 3]));
}
inline float4 load4(device const bfloat* p, long i) {
    return float4(float(p[i]), float(p[i + 1]), float(p[i + 2]), float(p[i + 3]));
}
inline void store4(device float* p, long i, float4 v) { *(device float4*)(p + i) = v; }
inline void store4(device half* p, long i, float4 v) {
    p[i] = half(v.x); p[i + 1] = half(v.y); p[i + 2] = half(v.z); p[i + 3] = half(v.w);
}
inline void store4(device bfloat* p, long i, float4 v) {
    p[i] = bfloat(v.x); p[i + 1] = bfloat(v.y);
    p[i + 2] = bfloat(v.z); p[i + 3] = bfloat(v.w);
}
inline float4 silu4(float4 v) { return v / (1.0f + exp(-v)); }

/* log(1 + exp(v)) to within an ulp of the float32 result; the series around zero
 * keeps the relative accuracy that log(1 + exp(v)) loses for the small deltas a
 * softplus'd dt lives in.  Same branches as torch.nn.functional.softplus. */
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

_CB = r"""
/* cb[group, chunk][l, s] = <C[l], B[s]> for one group of heads.  It does not
 * depend on the head, so one dispatch per (batch, group, chunk) computes it once
 * for every head that shares the group.  B is transposed into threadgroup memory
 * so the inner loop reads it without bank conflicts. */
kernel void mamba2_cb(
    device const $T* Bv [[buffer(0)]],
    device const $T* Cv [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    constant uint& groups [[buffer(4)]],
    constant uint& nchunks [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint L = $L;
    const uint N = $N;
    threadgroup float bt[$L * $N];

    const uint gc = tgid;
    const uint c = gc % nchunks;
    const uint bg = gc / nchunks;
    const uint b = bg / groups;
    const uint g = bg - b * groups;
    const uint tok0 = c * L;
    const uint lc = min(L, length - tok0);
    const long gsN = (long) groups * N;
    const long base = ((long) b * length * groups + g) * N + (long) tok0 * gsN;

    for (uint t = tid; t < $L * $N; t += $THREADS) {
        uint l = t / N;
        uint n = t - l * N;
        bt[n * $L + l] = (l < lc) ? loadv(Bv, base + (long) l * gsN + n) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = tid; t < $L * $L; t += $THREADS) {
        uint l = t / $L;
        uint s = t - l * $L;
        float dot = 0.0f;
        if (s <= l && l < lc) {
            long cbase = base + (long) l * gsN;
            for (uint n = 0; n < N; ++n) {
                dot += loadv(Cv, cbase + n) * bt[n * $L + s];
            }
        }
        out[(long) gc * $L * $L + t] = dot;
    }
}
"""

_SCAN = r"""
/* One threadgroup per (batch, head, chunk).  With d the time step and dcs its
 * cumulative decay inside the chunk, the chunk's output is
 *
 *   scores[l][s] = exp(dcs[l] - dcs[s]) d[s] cb[l][s]        (s <= l, else 0)
 *   y[l, p] = sum_{s<=l} scores[l][s] x[s, p]                intra chunk
 *           + exp(dcs[l]) sum_n C[l, n] entering[n, p]       carried state
 *           + D[p] x[l, p]                                   skip
 *   y *= silu(z)                                             gate
 *
 * The masked decay, the dt scaling, the score product and the accumulation over
 * s all happen in registers: `scores` is the only threadgroup buffer, and the
 * `<C[l], entering[., p]>` sum is a per-thread reduction over n, so no output
 * element is ever materialised anywhere but its final store.
 *
 * Thread t owns head-dim column p = t % P and one block of LPB chunk rows, so
 * every lane of a warp walks the same (l, n) and reads C and the scores as
 * broadcasts. */
kernel void mamba2_scan(
    device const $T* xv [[buffer(0)]],
    device const $T* Cv [[buffer(1)]],
    device const float* dv [[buffer(2)]],
    device const float* dcv [[buffer(3)]],
    device const float* entv [[buffer(4)]],
    device const float* Dv [[buffer(5)]],
    device const $T* zv [[buffer(6)]],
    device const float* cbv [[buffer(7)]],
    device $T* yv [[buffer(8)]],
    constant uint& length [[buffer(9)]],
    constant uint& nheads [[buffer(10)]],
    constant uint& dstate [[buffer(11)]],
    constant uint& groups [[buffer(12)]],
    constant uint& nchunks [[buffer(13)]],
    constant uint& flags [[buffer(14)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint L = $L;
    const uint P = $PP;
    const uint P4 = P / 4;
    threadgroup float scores[$L * $L];
    threadgroup float csh[$L * 128];

    const uint hs = tgid / nchunks;
    const uint c = tgid - hs * nchunks;
    const uint b = hs / nheads;
    const uint h = hs - b * nheads;
    const uint g = h / (nheads / groups);
    const uint p4 = tid % P4;
    const uint p = p4 * 4;
    const uint lbase = (tid / P4) * $LPB;
    const uint tok0 = c * L;
    const uint lc = min(L, length - tok0);

    const long gsN = (long) groups * dstate;
    const long dbase = ((long) hs * nchunks + c) * L;
    const long cb_off = ((long) (b * groups + g) * nchunks + c) * (long) L * L;
    const long cbase = ((long) b * length * groups + g) * dstate + (long) tok0 * gsN;
    const long ebase = ((long) hs * nchunks + c) * (long) dstate * P + p;
    const long xbase = ((long) b * length * nheads + h) * P + p;
    const long xstride = (long) nheads * P;
    const float4 skip = flagged(flags, $HAS_D)
                            ? (flagged(flags, $D_PER_HEAD)
                                   ? load4(Dv, (long) h * P + p)
                                   : float4(Dv[h]))
                            : float4(0.0f);

    for (uint t = tid; t < L * dstate; t += $THREADS) {
        const uint l = t / dstate, n = t - l * dstate;
        csh[l * 128 + n] = (l < lc) ? loadv(Cv, cbase + (long) l * gsN + n) : 0.0f;
    }
    for (uint t = tid; t < L * L; t += $THREADS) {
        const uint l = t / L, s = t - l * L;
        float v = 0.0f;
        if (s <= l && l < lc) {
            v = exp(dcv[dbase + l] - dcv[dbase + s]) * dv[dbase + s] * cbv[cb_off + t];
        }
        scores[t] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 acc[$LPB];
    for (uint j = 0; j < $LPB; ++j) acc[j] = float4(0.0f);

    /* carried state: y += exp(dcs[l]) sum_n C[l, n] entering[n, p..p+3] */
    for (uint n = 0; n < dstate; ++n) {
        const float4 sv = *(device const float4*)(entv + ebase + (long) n * P);
        for (uint j = 0; j < $LPB; ++j) {
            const uint l = lbase + j;
            if (l < lc) acc[j] += csh[l * 128 + n] * sv;
        }
    }
    for (uint j = 0; j < $LPB; ++j) {
        const uint l = lbase + j;
        if (l < L) acc[j] *= exp(dcv[dbase + l]);
    }

    /* intra chunk: y += sum_{s<=l} scores[l][s] x[s, p..p+3] */
    for (uint s = 0; s < lc; ++s) {
        const float4 xs = load4(xv, xbase + (long) (tok0 + s) * xstride);
        for (uint j = 0; j < $LPB; ++j) {
            if (s <= lbase + j) acc[j] += scores[(lbase + j) * L + s] * xs;
        }
    }

    for (uint j = 0; j < $LPB; ++j) {
        const uint l = lbase + j;
        if (l >= lc) continue;
        const long at = xbase + (long) (tok0 + l) * xstride;
        float4 v = acc[j] + skip * load4(xv, at);
        if (flagged(flags, $HAS_Z)) {
            v = v * silu4(load4(zv, at));
        }
        store4(yv, at, v);
    }
}
"""


def _substitute(source, values):
    # Longest first: "$L" is a prefix of "$LPB" and "$LBG".
    for name in sorted(values, key=len, reverse=True):
        source = source.replace(name, str(values[name]))
    return source


def _plan(chunk, headdim):
    """Thread mapping for the output kernel, or ``None`` when it does not fit."""
    if headdim % 4 != 0:
        return None
    p4 = headdim // 4
    lbg = max(1, min(16, max(1, 256 // p4)))
    lpb = -(-chunk // lbg)
    threads = p4 * lbg
    tg_bytes = (chunk * chunk + chunk * 128) * 4
    if tg_bytes > _TG_BUDGET or lpb > 16 or threads > _MAX_THREADS:
        return None
    return dict(lbg=lbg, lpb=lpb, threads=threads)


@lru_cache(maxsize=32)
def _scan_library(chunk, headdim, lbg, lpb, threads, dtype):
    _, storage = _DTYPE_NAMES[dtype]
    return torch.mps.compile_shader(_substitute(_HELPERS + _SCAN, {
        "$T": storage, "$L": chunk, "$PP": headdim, "$LBG": lbg, "$LPB": lpb,
        "$THREADS": threads, "$HAS_Z": _HAS_Z, "$HAS_D": _HAS_D,
        "$D_PER_HEAD": _D_PER_HEAD,
    }))


@lru_cache(maxsize=16)
def _cb_library(chunk, dstate, threads, dtype):
    _, storage = _DTYPE_NAMES[dtype]
    return torch.mps.compile_shader(_substitute(_HELPERS + _CB, {
        "$T": storage, "$L": chunk, "$N": dstate, "$THREADS": threads,
    }))


_path = "none"
"""Which path the last :func:`mamba2_chunk_scan` call took."""


def last_path():
    """``"metal"`` or ``"composition"``: the path the last scan took."""
    return _path


def _check(x, dt, A, B, C, D, z, dt_bias, initial_states, chunk_size):
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if x.dim() != 4:
        raise ValueError(f"x must be (batch, length, nheads, d_head), got {tuple(x.shape)}")
    batch, length, nheads, headdim = x.shape
    if dt.shape != (batch, length, nheads):
        raise ValueError(f"dt must be {(batch, length, nheads)}, got {tuple(dt.shape)}")
    if A.shape != (nheads,):
        raise ValueError(f"A must be ({nheads},), got {tuple(A.shape)}")
    if B.dim() != 4 or C.shape != B.shape:
        raise ValueError(
            "B and C must both be (batch, length, ngroups, d_state), got "
            f"{tuple(B.shape)} and {tuple(C.shape)}"
        )
    if B.shape[:2] != (batch, length):
        raise ValueError(f"B/C must start with {(batch, length)}, got {tuple(B.shape)}")
    ngroups, dstate = B.shape[2], B.shape[3]
    if ngroups < 1 or nheads % ngroups:
        raise ValueError(f"nheads={nheads} must be a multiple of ngroups={ngroups}")
    if z is not None and z.shape != x.shape:
        raise ValueError(f"z must be {tuple(x.shape)}, got {tuple(z.shape)}")
    if D is not None and D.shape not in ((nheads,), (nheads, headdim)):
        raise ValueError(
            f"D must be ({nheads},) or ({nheads}, {headdim}), got {tuple(D.shape)}"
        )
    if dt_bias is not None and dt_bias.shape != (nheads,):
        raise ValueError(f"dt_bias must be ({nheads},), got {tuple(dt_bias.shape)}")
    if initial_states is not None:
        if initial_states.dim() == 5 and initial_states.shape[1] == 1:
            initial_states = initial_states[:, 0]
        if initial_states.shape != (batch, nheads, headdim, dstate):
            if (initial_states.shape == (batch, nheads, dstate, headdim)
                    and headdim != dstate):
                initial_states = initial_states.transpose(-1, -2)
            else:
                raise ValueError(
                    "initial_states must be (batch, nheads, d_head, d_state) = "
                    f"{(batch, nheads, headdim, dstate)}, got "
                    f"{tuple(initial_states.shape)}"
                )
    if x.dtype not in _HALF and x.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"x must be float32, float16, bfloat16 or float64, got {x.dtype}")
    for name, tensor in (("dt", dt), ("A", A), ("B", B), ("C", C)):
        if tensor.device != x.device:
            raise ValueError(f"{name} is on {tensor.device}, x is on {x.device}")
    return batch, length, nheads, headdim, ngroups, dstate, initial_states


def _discretize(dt, A, dt_bias, dt_softplus, chunk_size, length):
    """Per-chunk ``dt`` and ``dcs = cumsum(dt * A)``, float32 and padded."""
    batch, _, nheads = dt.shape
    chunk_size = min(chunk_size, length)
    nchunks = -(-length // chunk_size)
    padded = nchunks * chunk_size
    d = dt.transpose(1, 2).to(torch.float32)
    if dt_bias is not None:
        d = d + dt_bias.to(torch.float32).view(1, nheads, 1)
    if dt_softplus:
        d = F.softplus(d)
    if padded > length:
        d = F.pad(d, (0, padded - length))
    d = d.reshape(batch, nheads, nchunks, chunk_size)
    dcs = torch.cumsum(d * A.to(torch.float32).view(1, nheads, 1, 1), dim=-1)
    return d, dcs, nchunks, padded, chunk_size


def _chunk_states(x, B, d, dcs, nheads, ngroups, headdim, dstate, nchunks, chunk_size):
    """State each chunk leaves behind: ``(batch, nheads, chunk, d_state, d_head)``.

    The batch order of the block product is ``(batch, group, head, chunk)``, so
    the result reshapes into the ``(batch, nheads, chunk, d_state, d_head)``
    order for free, and that order is the one the chunk transfer consumes: its
    flat form ``(chunk, d_state * d_head)`` is already contiguous. Doing the
    product head-major instead would cost two full transposes of the state.
    """
    batch = x.shape[0]
    per_group = nheads // ngroups
    weight = d * torch.exp(dcs[..., -1:] - dcs)            # (B, H, C, L) fp32
    weight = weight.permute(0, 2, 3, 1).unsqueeze(-1)      # (B, C, L, H, 1)
    xw = x.view(batch, nchunks, chunk_size, nheads, headdim) * weight.to(x.dtype)
    xg = xw.view(batch, nchunks, chunk_size, ngroups, per_group, headdim).permute(
        0, 3, 4, 1, 2, 5
    )
    Bg = B.view(batch, nchunks, chunk_size, ngroups, dstate).permute(0, 3, 1, 2, 4)
    states = torch.matmul(Bg.transpose(-1, -2).unsqueeze(2), xg)   # (B, G, Hp, C, N, P)
    return states.float().reshape(batch, nheads, nchunks, dstate, headdim)


def _state_passing(states, dcs, initial, nchunks):
    """Causal chunk transfer, the closed form of ``state_passing_ref``."""
    batch, nheads, nchunks, dstate, headdim = states.shape
    device = states.device
    total = dcs[..., -1]                                        # (B, H, C)
    before = torch.cumsum(F.pad(total, (1, 0)), dim=-1)         # (B, H, C+1), q[c]=sum_{i<c}
    diff = before[..., :-1, None] - before[..., None, 1:]       # (B, H, C, C)
    mask = torch.ones(nchunks, nchunks, dtype=torch.bool, device=device).tril(-1)
    decay = torch.exp(diff.masked_fill(~mask, float("-inf")))
    flat = states.reshape(batch, nheads, nchunks, dstate * headdim)
    entering = torch.matmul(decay, flat)
    carry = torch.exp(before[..., :-1]).unsqueeze(-1)           # (B, H, C, 1)
    if initial is not None:
        entering = entering + carry * initial.transpose(-1, -2).reshape(
            batch, nheads, headdim * dstate
        ).unsqueeze(2)
    entering = entering.reshape(batch, nheads, nchunks, dstate, headdim)
    final = states[:, :, -1] + torch.exp(total[:, :, -1])[..., None, None] * entering[:, :, -1]
    return entering, final.transpose(-1, -2).contiguous()


def _chunk_outputs(x, B, C, d, dcs, entering, D, nheads, ngroups, headdim,
                   dstate, nchunks, chunk_size):
    batch = x.shape[0]
    per_group = nheads // ngroups
    lower = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=x.device).tril()
    decay = torch.exp(
        (dcs[..., :, None] - dcs[..., None, :]).masked_fill(~lower, float("-inf"))
    )                                                           # (B, H, C, L, L) fp32
    out_decay = torch.exp(dcs)                                  # (B, H, C, L) fp32
    parts = []
    for group in range(ngroups):
        heads = slice(group * per_group, (group + 1) * per_group)
        Cg = C[:, :, group].reshape(batch, nchunks, chunk_size, dstate).contiguous()
        Bg = B[:, :, group].reshape(batch, nchunks, chunk_size, dstate).contiguous()
        cb = torch.matmul(Cg, Bg.transpose(-1, -2)).float()      # (B, C, L, L) fp32
        scores = (decay[:, heads] * cb[:, None] * d[:, heads][..., None, :]).to(x.dtype)
        xg = x[:, :, heads].view(
            batch, nchunks, chunk_size, per_group, headdim
        ).permute(0, 3, 1, 2, 4)                                 # (B, Hp, C, L, P)
        out = torch.matmul(scores, xg)
        state = entering[:, heads].to(x.dtype)                   # (B, Hp, C, N, P)
        prev = torch.matmul(Cg[:, None].to(x.dtype), state)      # (B, Hp, C, L, P)
        prev = prev * out_decay[:, heads][..., None].to(x.dtype)
        parts.append((out + prev).permute(0, 2, 3, 1, 4))        # (B, C, L, Hp, P)
    y = parts[0] if ngroups == 1 else torch.cat(parts, dim=3)
    return y.reshape(batch, nchunks * chunk_size, nheads, headdim)


def _metal_plan(x, dt, A, B, C, D, z, dt_bias, initial_states,
                batch, length, nheads, headdim, ngroups, dstate, chunk_size):
    """Fused-path parameters, or ``None`` to use the composition."""
    if x.device.type != "mps" or x.dtype not in _DTYPE_NAMES:
        return None
    if B.dtype != x.dtype or C.dtype != x.dtype or dt.dtype not in _DTYPE_NAMES:
        return None
    if chunk_size > _MAX_CHUNK or headdim > 1024 or dstate > 4096:
        return None
    if torch.is_grad_enabled() and any(
        t is not None and t.requires_grad
        for t in (x, dt, A, B, C, D, z, dt_bias, initial_states)
    ):
        return None
    chunk = min(chunk_size, length)
    if chunk * dstate * 4 > _TG_BUDGET:                   # cb staging
        return None
    plan = _plan(chunk, headdim)
    if plan is None:
        return None
    plan.update(chunk=chunk, nchunks=-(-length // chunk),
                cb_threads=min(_MAX_THREADS, chunk * chunk))
    return plan


def _cast(tensor, fallback=None):
    """float32 contiguous copy, or ``fallback`` when ``tensor`` is ``None``."""
    if tensor is None:
        return fallback
    return tensor.contiguous() if tensor.dtype == torch.float32 else tensor.float().contiguous()


def _metal_forward(x, dt, A, B, C, D, z, dt_bias, dt_softplus, initial_states,
                   batch, length, nheads, headdim, ngroups, dstate, plan):
    """The chunked composition with its output stage fused into two dispatches.

    ``mamba2_cb`` builds the per-group ``<C[l], B[s]>`` block and ``mamba2_scan``
    turns it into the masked decay, the intra-chunk product, the carried-state
    term, the skip and the gate in one threadgroup per (batch, head, chunk).
    The discretization, the chunk states and the chunk-to-chunk transfer are the
    same torch matmuls as in the composition.
    """
    device, dtype = x.device, x.dtype
    chunk, nchunks = plan["chunk"], plan["nchunks"]
    d, dcs, nchunks, padded, chunk = _discretize(
        dt, A, dt_bias, dt_softplus, chunk, length
    )
    # ``d`` comes back as the transposed view ``dt`` was built from; a kernel
    # launch reads the storage raw, so both have to be dense before dispatch.
    d, dcs = d.contiguous(), dcs.contiguous()
    if padded > length:
        xp = F.pad(x, (0, 0, 0, 0, 0, padded - length))
        Bp = F.pad(B, (0, 0, 0, 0, 0, padded - length))
    else:
        xp, Bp = x, B
    states = _chunk_states(xp, Bp, d, dcs, nheads, ngroups, headdim, dstate,
                           nchunks, chunk)
    entering, final_state = _state_passing(states, dcs, initial_states, nchunks)

    cb = torch.empty((batch * ngroups * nchunks, chunk, chunk),
                     dtype=torch.float32, device=device)
    cb_threads = plan["cb_threads"]
    _cb_library(chunk, dstate, cb_threads, dtype).mamba2_cb(
        B, C, cb, length, ngroups, nchunks,
        threads=batch * ngroups * nchunks * cb_threads, group_size=cb_threads,
    )

    y = torch.empty_like(x)
    flags = (_HAS_D | (4 if D.dim() == 2 else 0) if D is not None else 0)
    flags |= _HAS_Z if z is not None else 0
    dummy = d
    threads = plan["threads"]
    kernel = _scan_library(chunk, headdim, plan["lbg"], plan["lpb"], threads, dtype)
    kernel.mamba2_scan(
        x, C, d, dcs, entering, _cast(D, dummy), z if z is not None else x, cb, y,
        length, nheads, dstate, ngroups, nchunks, flags,
        threads=batch * nheads * nchunks * threads, group_size=threads,
    )
    return y, final_state.to(dtype)


def mamba2_chunk_scan(x, dt, A, B, C, chunk_size=64, D=None, z=None,
                      dt_bias=None, dt_softplus=False, initial_states=None):
    """Exact Mamba-2 chunked SSD scan on any device torch runs.

    Args:
        x: ``(batch, length, nheads, d_head)``, float32/float16/bfloat16.
        dt: ``(batch, length, nheads)`` pre-activation time steps.
        A: ``(nheads,)`` negative state decay.
        B, C: ``(batch, length, ngroups, d_state)``; ``ngroups`` divides
            ``nheads`` and the heads of a group share both.
        chunk_size: chunk length for the block factorization. The result does
            not depend on it; memory and speed do.
        D: ``(nheads,)`` or ``(nheads, d_head)`` skip connection.
        z: ``(batch, length, nheads, d_head)`` gate, applied as ``y * silu(z)``.
        dt_bias: ``(nheads,)`` added to ``dt`` before the softplus.
        dt_softplus: apply ``softplus`` to ``dt + dt_bias``.
        initial_states: ``(batch, nheads, d_head, d_state)``.

    Returns:
        ``(y, final_state)``: ``y`` is ``(batch, length, nheads, d_head)`` and
        ``final_state`` is ``(batch, nheads, d_head, d_state)``, both in
        ``x.dtype``.
    """
    batch, length, nheads, headdim, ngroups, dstate, initial_states = _check(
        x, dt, A, B, C, D, z, dt_bias, initial_states, chunk_size
    )
    x, B, C = x.contiguous(), B.contiguous(), C.contiguous()
    global _path
    plan = _metal_plan(x, dt, A, B, C, D, z, dt_bias, initial_states,
                       batch, length, nheads, headdim, ngroups, dstate, chunk_size)
    if plan is not None:
        _path = "metal"
        return _metal_forward(x, dt, A, B, C, D, z, dt_bias, dt_softplus,
                              initial_states, batch, length, nheads, headdim,
                              ngroups, dstate, plan)
    _path = "composition"
    d, dcs, nchunks, padded, chunk_size = _discretize(
        dt, A, dt_bias, dt_softplus, chunk_size, length
    )
    if padded > length:
        xp = F.pad(x, (0, 0, 0, 0, 0, padded - length))
        Bp = F.pad(B, (0, 0, 0, 0, 0, padded - length))
        Cp = F.pad(C, (0, 0, 0, 0, 0, padded - length))
    else:
        xp, Bp, Cp = x, B, C
    states = _chunk_states(xp, Bp, d, dcs, nheads, ngroups, headdim, dstate, nchunks, chunk_size)
    entering, final_state = _state_passing(states, dcs, initial_states, nchunks)
    y = _chunk_outputs(
        xp, Bp, Cp, d, dcs, entering, D, nheads, ngroups, headdim, dstate,
        nchunks, chunk_size,
    )
    if D is not None:
        skip = D.to(x.dtype).view(1, 1, nheads, 1) if D.dim() == 1 else D.to(x.dtype).view(
            1, 1, nheads, headdim
        )
        y = y + xp * skip
    if z is not None:
        zp = F.pad(z, (0, 0, 0, 0, 0, padded - length)) if padded > length else z
        y = y * F.silu(zp)
    if padded > length:
        y = y[:, :length].contiguous()
    return y, final_state.to(x.dtype)


def mamba2_chunk_scan_reference(x, dt, A, B, C, chunk_size=64, D=None, z=None,
                                dt_bias=None, dt_softplus=False, initial_states=None,
                                dtype=torch.float64):
    """Step-by-step recurrence: the definition the chunked scan must reproduce.

    Runs on the tensors' own device but walks ``length`` one token at a time, so
    it is meant for CPU at small sizes. ``chunk_size`` is accepted for signature
    parity and ignored. Everything is evaluated in ``dtype`` (float64 by
    default); the returned tensors are cast back to the input dtype.
    """
    out_dtype = x.dtype
    x, dt, A, B, C = (t.to(dtype) for t in (x, dt, A, B, C))
    batch, length, nheads, headdim = x.shape
    ngroups, dstate = B.shape[2], B.shape[3]
    per_group = nheads // ngroups
    Bg = B.repeat_interleave(per_group, dim=2)
    Cg = C.repeat_interleave(per_group, dim=2)
    if dt_bias is not None:
        dt = dt + dt_bias.to(dtype).view(1, 1, nheads)
    if dt_softplus:
        dt = F.softplus(dt)
    state = torch.zeros(batch, nheads, headdim, dstate, dtype=dtype, device=x.device)
    if initial_states is not None:
        initial = initial_states.to(dtype)
        if initial.dim() == 5:
            initial = initial[:, 0]
        if initial.shape != state.shape and initial.shape == state.transpose(-1, -2).shape:
            initial = initial.transpose(-1, -2)
        state = initial.clone()
    steps = []
    for t in range(length):
        delta = dt[:, t]                                             # (B, H)
        state = state * torch.exp(delta * A)[..., None, None] \
            + (delta[..., None] * x[:, t])[..., :, None] * Bg[:, t][:, :, None, :]
        steps.append((state * Cg[:, t][:, :, None, :]).sum(-1))
    y = torch.stack(steps, dim=1)
    if D is not None:
        skip = D.to(dtype)
        y = y + x * (skip.view(1, 1, nheads, 1) if skip.dim() == 1
                     else skip.view(1, 1, nheads, headdim))
    if z is not None:
        y = y * F.silu(z.to(dtype))
    return y.to(out_dtype), state.to(out_dtype)
