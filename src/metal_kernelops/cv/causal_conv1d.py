"""Causal depthwise conv1d, the convolution every Mamba block runs first.

Two entry points, matching Dao-AILab/causal-conv1d:

``causal_conv1d_fn``
    the prefill pass over a whole sequence, ``(batch, channels, length)``

``causal_conv1d_update``
    the decode pass, one or a few tokens against a rolling conv state

The convolution is depthwise, ``groups == channels``, and causal: output ``t``
sees inputs ``t - (K-1) .. t`` only, so it is a plain left-padded convolution.
With ``z = cat(initial_state, x)`` (``z = cat(zeros(K-1), x)`` when there is no
state) and ``m = K - 1``::

    out[b, c, t] = act(bias[c] + sum_{j < K} weight[c, j] * z[b, c, t + j])
    final_state  = z[..., length : length + m]

``act`` is ``None`` or SiLU (``"silu"`` / ``"swish"``).

Layout and dtype contract
-------------------------
* ``x``: ``(batch, channels, length)``, any stride; it is made contiguous.
  ``weight``: ``(channels, 1, K)`` depthwise, as ``nn.Conv1d(..., groups=
  channels)`` stores it; upstream's ``(channels, K)`` is accepted too.
  ``bias``: ``(channels,)`` or ``None``.  ``initial_state``:
  ``(batch, channels, K-1)`` or ``None``.
* Arithmetic runs in ``weight.dtype`` and the result comes back in ``x.dtype``,
  which is what upstream's reference does when the two differ.
* ``causal_conv1d_update`` takes ``x`` of shape ``(batch, channels)`` or
  ``(batch, channels, q)`` with ``q >= 1``, and a state ``(batch, channels, s)``
  with ``s >= K-1`` whose last stride is 1: a slice of a larger cache is read
  and written in place through its strides.  It returns the rank it got.
* fp32 and fp16 use the Metal kernels; other dtypes and non-MPS devices go
  through the ATen composition, which is also the timing baseline.

Why a kernel at all
-------------------
``conv1d`` on MPS lowers to an MPSGraph convolution: at decode sizes its
dispatch cost dwarfs the arithmetic, and at prefill sizes it is 2-3x slower
than a fused tap loop.  The kernels here fuse the bias, the SiLU and the
final state, and keep the taps in registers; the backward is analytic and
runs as one fused ``dx`` + ``dweight`` pass instead of autograd through
conv1d.  ``benchmarks/verify_causal_conv1d.py`` holds the measurements.

There is no saved forward activation: the backward recomputes the
pre-activation from ``x`` while it already holds the tap window it needs for
``dweight``.  That trades a few flops for not writing and re-reading a
``(batch, channels, length)`` buffer.
"""

import torch
import torch.nn.functional as F

__all__ = ["causal_conv1d_fn", "causal_conv1d_ref", "causal_conv1d_update"]

_ACTIVATIONS = (None, "silu", "swish")
_TILE = 128
"""Threads per threadgroup in the fused backward, where the taps reduce."""
_TARGET_GROUPS = 8192
"""Threadgroups the backward aims for, so short rows do not underfill the GPU."""
_METAL_DTYPES = (torch.float32, torch.float16)


# --------------------------------------------------------------------------- #
# Metal sources.  ``$T`` becomes float/half, ``$K`` the weight width (the tap
# count is ``$K - 1``), ``$VEC`` the elements one thread owns, ``$CT`` the tap
# window measured in ``$VEC``-sized blocks either side of the block a thread
# owns, ``$NB``/``$ND`` the window lengths of the input and of ``dout``,
# ``$TILE$`` the threadgroup width and ``$NSIMD`` the SIMD groups in one.
#
# Both kernels give each thread one ``$VEC``-aligned block of one row.  The
# interior of a row reads its tap window with 16-byte vector loads; the blocks
# within ``$CT`` of either end fall back to scalar indexing, which is what
# carries the causal left padding, the initial state, and any length shorter
# than one window.
# --------------------------------------------------------------------------- #

_FORWARD = r"""
#include <metal_stdlib>
using namespace metal;

kernel void causal_conv1d_fwd(
    device const $T* x, device const $T* init, device const $T* w,
    device const $T* b, device $T* out, device $T* final_state,
    constant uint& C, constant uint& L, constant uint& blocks,
    constant uint& has_init, constant uint& has_bias,
    constant uint& silu, constant uint& write_final,
    uint gid [[thread_position_in_grid]]) {
    const uint W = ($K);
    const uint taps = W - 1;
    const uint V = ($VEC);
    uint row = gid / blocks;
    uint blk = gid - row * blocks;
    uint t = blk * V;
    uint c = row % C;
    device const $T* xr = x + (long)row * L;
    device const $T* ir = init + (long)row * taps;
    device $T* orow = out + (long)row * L;
    float wv[$K];
    #pragma unroll
    for (uint j = 0; j < W; ++j) wv[j] = float(w[c * W + j]);
    float bv = has_bias ? float(b[c]) : 0.0f;

    if (write_final && blk == 0) {
        device $T* fr = final_state + (long)row * taps;
        for (uint i = 0; i < taps; ++i) {
            uint p = L + i;
            fr[i] = (p < taps) ? (has_init ? ir[p] : ($T)0) : xr[p - taps];
        }
    }

    vec<$T, $VEC> ov;
    if (blk >= $CT) {
        float win[$NB * $VEC];
        #pragma unroll
        for (uint q = 0; q < $NB; ++q) {
            vec<$T, $VEC> v = *(device const vec<$T, $VEC>*)(xr + (long)(blk - $CT + q) * V);
            #pragma unroll
            for (uint i = 0; i < V; ++i) win[q * V + i] = float(v[i]);
        }
        const uint off = $CT * V - taps;
        float res[$VEC];
        #pragma unroll
        for (uint i = 0; i < V; ++i) res[i] = bv;
        #pragma unroll
        for (uint j = 0; j < W; ++j) {
            #pragma unroll
            for (uint i = 0; i < V; ++i) res[i] += wv[j] * win[off + i + j];
        }
        #pragma unroll
        for (uint i = 0; i < V; ++i) {
            float a = res[i];
            if (silu) a = a / (1.0f + exp(-a));
            ov[i] = ($T)a;
        }
    } else {
        #pragma unroll
        for (uint i = 0; i < V; ++i) {
            uint oi = t + i;
            float acc = bv;
            #pragma unroll
            for (uint j = 0; j < W; ++j) {
                uint p = oi + j;
                float v = 0.0f;
                if (p < taps) {
                    if (has_init) v = float(ir[p]);
                } else {
                    v = float(xr[p - taps]);
                }
                acc += wv[j] * v;
            }
            if (silu) acc = acc / (1.0f + exp(-acc));
            ov[i] = ($T)acc;
        }
    }
    *(device vec<$T, $VEC>*)(orow + t) = ov;
}
"""

_BACKWARD = r"""
#include <metal_stdlib>
using namespace metal;

// With g[t] = dout[t] * act'(pre[t]) over the output positions that exist:
//   dx[t]      = sum_j w[j] * g[t + (K-1) - j]
//   dstate[s]  = sum_j w[j] * g[s - j]          s < K-1
//   dweight[j] = sum_t g[t] * z[t + j]
// A row longer than one threadgroup's reach is walked by ``blk`` striding
// ``$TILE$`` blocks, so the shuffle tree runs once per threadgroup.  Each
// threadgroup writes one partial row per (batch, channel, chunk) and the
// finalize kernel folds those.  ``dstate`` is the one exception to the strides:
// it has the width of one tap window, so block 0 alone computes it in scalars.
kernel void causal_conv1d_bwd(
    device const $T* dout, device const $T* x, device const $T* init,
    device const $T* w, device const $T* b,
    device $T* dx, device $T* dstate, device float* partial,
    constant uint& C, constant uint& L, constant uint& blocks,
    constant uint& groups_per_row, constant uint& npc,
    constant uint& has_init, constant uint& has_bias,
    constant uint& silu, constant uint& write_dstate,
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint slane [[thread_index_in_simdgroup]],
    uint sid [[simdgroup_index_in_threadgroup]]) {
    const uint W = ($K);
    const uint taps = W - 1;
    const uint V = ($VEC);
    threadgroup float folded[$NSIMD * ($K + 1)];

    uint row = tgid / groups_per_row;
    uint chunk = tgid - row * groups_per_row;
    uint c = row % C;
    device const $T* xr = x + (long)row * L;
    device const $T* dr = dout + (long)row * L;
    device const $T* ir = init + (long)row * taps;
    float wv[$K];
    #pragma unroll
    for (uint j = 0; j < W; ++j) wv[j] = float(w[c * W + j]);
    float bv = has_bias ? float(b[c]) : 0.0f;
    float dw[$K + 1];
    #pragma unroll
    for (uint j = 0; j <= $K; ++j) dw[j] = 0.0f;

    for (uint blk = chunk * ($TILE$) + lid; blk < blocks; blk += groups_per_row * ($TILE$)) {
        uint t = blk * V;
        if (blk >= $CT && blk + $CT < blocks) {
            float wx[$NB * $VEC];
            #pragma unroll
            for (uint q = 0; q < $NB; ++q) {
                vec<$T, $VEC> v = *(device const vec<$T, $VEC>*)(xr + (long)(blk - $CT + q) * V);
                #pragma unroll
                for (uint i = 0; i < V; ++i) wx[q * V + i] = float(v[i]);
            }
            float wd[$ND * $VEC];
            #pragma unroll
            for (uint q = 0; q < $ND; ++q) {
                vec<$T, $VEC> v = *(device const vec<$T, $VEC>*)(dr + (long)(blk + q) * V);
                #pragma unroll
                for (uint i = 0; i < V; ++i) wd[q * V + i] = float(v[i]);
            }
            const uint ox = $CT * V - taps;
            float g[$VEC + $K - 1];
            #pragma unroll
            for (uint u = 0; u < $VEC + $K - 1; ++u) {
                float pre = bv;
                #pragma unroll
                for (uint j = 0; j < W; ++j) pre += wv[j] * wx[ox + u + j];
                float gv = wd[u];
                if (silu) {
                    float sg = 1.0f / (1.0f + exp(-pre));
                    gv *= sg * (1.0f + pre * (1.0f - sg));
                }
                g[u] = gv;
            }
            vec<$T, $VEC> ov;
            #pragma unroll
            for (uint i = 0; i < V; ++i) {
                float acc = 0.0f;
                #pragma unroll
                for (uint j = 0; j < W; ++j) acc += wv[j] * g[i + taps - j];
                ov[i] = ($T)acc;
            }
            *(device vec<$T, $VEC>*)(dx + (long)row * L + t) = ov;
            #pragma unroll
            for (uint i = 0; i < V; ++i) {
                dw[$K] += g[i];
                #pragma unroll
                for (uint j = 0; j < W; ++j) dw[j] += g[i] * wx[ox + i + j];
            }
        } else {
            auto tap = [&](uint p) -> float {
                return (p < taps) ? (has_init ? float(ir[p]) : 0.0f) : float(xr[p - taps]);
            };
            auto grad = [&](int p) -> float {
                if (p < 0 || p >= int(L)) return 0.0f;
                float pre = bv;
                #pragma unroll
                for (uint j = 0; j < W; ++j) pre += wv[j] * tap(uint(p) + j);
                float gv = float(dr[p]);
                if (silu) {
                    float sg = 1.0f / (1.0f + exp(-pre));
                    gv *= sg * (1.0f + pre * (1.0f - sg));
                }
                return gv;
            };
            vec<$T, $VEC> ov;
            #pragma unroll
            for (uint i = 0; i < V; ++i) {
                uint oi = t + i;
                float acc = 0.0f;
                #pragma unroll
                for (uint j = 0; j < W; ++j) acc += wv[j] * grad(int(oi) + int(taps) - int(j));
                ov[i] = ($T)acc;
                float gv = grad(int(oi));
                dw[$K] += gv;
                #pragma unroll
                for (uint j = 0; j < W; ++j) dw[j] += gv * tap(oi + j);
            }
            *(device vec<$T, $VEC>*)(dx + (long)row * L + t) = ov;
            if (write_dstate && blk == 0) {
                device $T* sr = dstate + (long)row * taps;
                for (uint s = 0; s < taps; ++s) {
                    float acc = 0.0f;
                    #pragma unroll
                    for (uint j = 0; j < W; ++j) {
                        int p = int(s) - int(j);
                        if (p >= 0) acc += wv[j] * grad(p);
                    }
                    sr[s] = ($T)acc;
                }
            }
        }
    }

    #pragma unroll
    for (uint j = 0; j <= $K; ++j) {
        float total = simd_sum(dw[j]);
        if (slane == 0) folded[sid * ($K + 1) + j] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        device float* base =
            partial + (long)(c * npc + (row / C) * groups_per_row + chunk) * ($K + 1);
        #pragma unroll
        for (uint j = 0; j <= $K; ++j) {
            float total = 0.0f;
            for (uint s = 0; s < $NSIMD; ++s) total += folded[s * ($K + 1) + j];
            base[j] = total;
        }
    }
}
"""

_FINALIZE = r"""
#include <metal_stdlib>
using namespace metal;

// One thread per (channel, column): fold the per-threadgroup partials of every
// batch row into dweight, and the last column into dbias.
kernel void causal_conv1d_dwfinal(
    device const float* partial, device $T* dweight, device $T* dbias,
    constant uint& npc, constant uint& write_weight, constant uint& write_bias,
    uint gid [[thread_position_in_grid]]) {
    const uint W = ($K);
    const uint columns = W + 1;
    uint c = gid / columns;
    uint j = gid - c * columns;
    device const float* base = partial + (long)c * npc * columns;
    float total = 0.0f;
    for (uint b = 0; b < npc; ++b) total += base[(long)b * columns + j];
    if (j < W) {
        if (write_weight) dweight[c * W + j] = ($T)total;
    } else if (write_bias) {
        dbias[c] = ($T)total;
    }
}
"""

_UPDATE = r"""
#include <metal_stdlib>
using namespace metal;

// One thread per (batch, channel) row.  Rows are contiguous in x, but the state
// may be a slice of a larger cache, so the state is addressed by its strides.
// The window is loaded once and shifted in registers; the copies that leave it
// either side are exact-size, which keeps the overlapping read of x down to the
// ``L < S`` case.
kernel void causal_conv1d_update(
    device $T* state,
    device const $T* x, device const $T* w, device const $T* b, device $T* y,
    constant uint& C, constant uint& L, constant uint& S, constant uint& rows,
    constant uint& state_row, constant uint& state_ch,
    constant uint& has_bias, constant uint& silu,
    uint gid [[thread_position_in_grid]]) {
    if (gid >= rows) return;
    const uint W = ($K);
    const uint taps = W - 1;
    uint c = gid % C;
    device $T* st = state + (long)(gid / C) * state_row + (long)c * state_ch;
    device const $T* xr = x + (long)gid * L;
    device const $T* wr = w + c * W;
    device $T* yr = y + (long)gid * L;
    float wv[$K];
    #pragma unroll
    for (uint j = 0; j < W; ++j) wv[j] = float(wr[j]);
    float bv = has_bias ? float(b[c]) : 0.0f;
    // z = cat(state, x) with the stale columns of a state wider than the tap
    // window skipped, so output t reads z[S - taps + t .. + W).  The weights are
    // hoisted because ``y`` may alias ``w`` as far as the compiler can prove.
    for (uint t = 0; t < L; ++t) {
        float acc = bv;
        #pragma unroll
        for (uint j = 0; j < W; ++j) {
            uint p = S - taps + t + j;
            acc += wv[j] * (p < S ? float(st[p]) : float(xr[p - S]));
        }
        if (silu) acc = acc / (1.0f + exp(-acc));
        yr[t] = ($T)acc;
    }
    // Ascending: state[i] only reads state[i + L], which is still the old value.
    for (uint i = 0; i < S; ++i) {
        uint src = i + L;
        st[i] = src < S ? st[src] : xr[src - S];
    }
}
"""

_shaders = {}


def _shader(source, dtype, width, vector=1, tile=_TILE):
    """Compile once per (kernel, dtype, weight width, block width, group width)."""
    key = (source, dtype, width, vector, tile)
    library = _shaders.get(key)
    if library is None:
        taps = width - 1
        window = -(-taps // vector)
        text = (
            source.replace("$TILE$", str(tile))
            .replace("$NSIMD", str(tile // 32))
            .replace("$VEC", str(vector))
            .replace("$CT", str(window))
            .replace("$NB", str(2 * window + 1 if source is _BACKWARD else window + 1))
            .replace("$ND", str(window + 1))
            .replace("$T", "float" if dtype == torch.float32 else "half")
            .replace("$K", str(width))
        )
        library = torch.mps.compile_shader(text)
        _shaders[key] = library
    return library


def _vector(tensors, length, wide):
    """Elements per thread: ``wide``, or one where a row cannot carry that.

    The forward asks for 16 bytes and the backward for four elements: the
    backward's window is the tap count either side of the block, so widening it
    reads proportionally more waste and measures 2x slower at eight, while the
    forward's window is only the taps at one end.
    """
    if length % wide or any(t.data_ptr() % 16 for t in tensors):
        return 1
    return wide


# --------------------------------------------------------------------------- #
# Argument checks and the canonical weight layout.
# --------------------------------------------------------------------------- #

def _as_depthwise(weight):
    if weight.dim() == 3:
        if weight.shape[1] != 1:
            raise ValueError(
                "weight must be (channels, 1, K) for a depthwise convolution, "
                f"got {tuple(weight.shape)}"
            )
        weight = weight[:, 0, :]
    elif weight.dim() != 2:
        raise ValueError(
            f"weight must be (channels, 1, K) or (channels, K), got {tuple(weight.shape)}"
        )
    if weight.shape[1] < 1:
        raise ValueError("the kernel width must be at least 1")
    return weight


def _check_activation(activation):
    if activation not in _ACTIVATIONS:
        raise NotImplementedError(f"activation must be one of {_ACTIVATIONS}, got {activation!r}")


def _metal_ok(x, weight):
    return x.device.type == "mps" and x.dtype in _METAL_DTYPES and weight.dtype == x.dtype


def _group(threads):
    return 32 if threads < 32 * _TILE else _TILE


def _grads(x, weight, initial_state):
    return torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in (x, weight, initial_state)
    )


# --------------------------------------------------------------------------- #
# ATen composition: the portability path, and the baseline the kernels beat.
# --------------------------------------------------------------------------- #

def _forward_aten(x, weight, bias, initial_state, activation):
    channels, width = weight.shape
    taps = width - 1
    length = x.shape[-1]
    if initial_state is not None:
        joined = torch.cat([initial_state, x], dim=-1)
        out = F.conv1d(joined, weight.unsqueeze(1), bias, groups=channels)
    else:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=taps, groups=channels)[..., :length]
    return F.silu(out) if activation is not None else out


def _final_state(x, initial_state, taps):
    """The last ``taps`` columns of ``cat(initial_state, x)``, as a fresh tensor."""
    length = x.shape[-1]
    if length >= taps:
        return x[..., length - taps:].contiguous()
    if initial_state is not None:
        head = initial_state[..., length:]
    else:
        head = x.new_zeros(x.shape[:-1] + (taps - length,))
    return torch.cat([head, x], dim=-1)


def causal_conv1d_ref(x, weight, bias=None, activation=None, initial_state=None,
                      return_final_state=False):
    """Reference definition: explicit taps, no convolution, differentiable.

    Same contract as :func:`causal_conv1d_fn`, valid on any device and dtype
    including float64.  It is the oracle in ``benchmarks/verify_causal_conv1d.py``.
    """
    weight = _as_depthwise(weight)
    _check_activation(activation)
    channels, width = weight.shape
    if x.dim() != 3 or x.shape[1] != channels:
        raise ValueError(f"x must be (batch, {channels}, length), got {tuple(x.shape)}")
    taps = width - 1
    length = x.shape[-1]
    out_dtype = x.dtype
    compute = weight.dtype
    work = x.to(compute)
    if initial_state is not None:
        work = torch.cat([initial_state.to(compute), work], dim=-1)
    else:
        work = F.pad(work, (taps, 0))
    out = None
    for j in range(width):
        term = work[..., j:j + length] * weight[:, j].view(1, channels, 1)
        out = term if out is None else out + term
    if bias is not None:
        out = out + bias.to(compute).view(1, channels, 1)
    final = work[..., length:].to(out_dtype) if return_final_state else None
    out = out.to(out_dtype)
    if activation is not None:
        out = F.silu(out)
    return (out, final) if return_final_state else out


# --------------------------------------------------------------------------- #
# Metal forward with an analytic backward.
# --------------------------------------------------------------------------- #

def _forward_metal(x, weight, bias, initial_state, activation, out, final, write_final):
    """One dispatch over the whole sequence; ``final`` is only touched if asked."""
    batch, channels, length = x.shape
    width = weight.shape[1]
    vector = _vector((x,), length, 16 // x.element_size())
    blocks = length // vector
    threads = batch * channels * blocks
    _shader(_FORWARD, x.dtype, width, vector).causal_conv1d_fwd(
        x, initial_state if initial_state is not None else x, weight,
        bias if bias is not None else x, out, final,
        channels, length, blocks,
        1 if initial_state is not None else 0,
        1 if bias is not None else 0,
        1 if activation is not None else 0,
        1 if write_final else 0,
        threads=threads, group_size=_group(threads),
    )


class _CausalConv1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, initial_state, activation):
        out = torch.empty_like(x)
        _forward_metal(x, weight, bias, initial_state, activation, out, out, False)
        ctx.save_for_backward(x, weight, bias, initial_state)
        ctx.activation = activation
        return out

    @staticmethod
    def backward(ctx, dout):
        x, weight, bias, initial_state = ctx.saved_tensors
        needs = ctx.needs_input_grad
        batch, channels, length = x.shape
        width = weight.shape[1]
        taps = width - 1
        dout = dout.contiguous()
        vector = _vector((x, dout), length, 4)
        blocks = length // vector
        groups = min(-(-blocks // _TILE), max(1, _TARGET_GROUPS // (batch * channels)))
        npc = batch * groups
        partial = torch.empty(channels * npc * (width + 1), dtype=torch.float32, device=x.device)
        dx = torch.empty_like(x)
        dstate = (
            torch.empty((batch, channels, taps), dtype=x.dtype, device=x.device)
            if initial_state is not None and needs[3]
            else dx
        )
        _shader(_BACKWARD, x.dtype, width, vector).causal_conv1d_bwd(
            dout, x, initial_state if initial_state is not None else x,
            weight, bias if bias is not None else x,
            dx, dstate, partial,
            channels, length, blocks, groups, npc,
            1 if initial_state is not None else 0,
            1 if bias is not None else 0,
            1 if ctx.activation is not None else 0,
            1 if dstate is not dx else 0,
            threads=batch * channels * groups * _TILE, group_size=_TILE,
        )

        dweight = dbias = None
        wants_bias = needs[2] and bias is not None
        if needs[1] or wants_bias:
            if needs[1]:
                dweight = torch.empty((channels, width), dtype=x.dtype, device=x.device)
            if wants_bias:
                dbias = torch.empty((channels,), dtype=x.dtype, device=x.device)
            _shader(_FINALIZE, x.dtype, width).causal_conv1d_dwfinal(
                partial,
                dweight if dweight is not None else dx,
                dbias if dbias is not None else dx,
                npc,
                1 if dweight is not None else 0, 1 if dbias is not None else 0,
                threads=channels * (width + 1), group_size=64,
            )
        return dx, dweight, dbias, dstate if dstate is not dx else None, None


def causal_conv1d_fn(x, weight, bias=None, activation=None, initial_state=None,
                     return_final_state=False):
    """Causal depthwise conv1d over a whole sequence.

    Args:
        x: ``(batch, channels, length)``.
        weight: ``(channels, 1, K)`` depthwise, or upstream's ``(channels, K)``.
        bias: ``(channels,)`` or ``None``.
        activation: ``None``, ``"silu"`` or ``"swish"``.
        initial_state: ``(batch, channels, K-1)`` entering the sequence.
        return_final_state: also return the state leaving the sequence.

    Returns:
        ``(batch, channels, length)`` in ``x.dtype``, or ``(out, final_state)``
        with ``final_state`` of shape ``(batch, channels, K-1)``.
    """
    weight = _as_depthwise(weight)
    _check_activation(activation)
    if x.dim() != 3:
        raise ValueError(f"x must be (batch, channels, length), got {tuple(x.shape)}")
    channels, width = weight.shape
    if x.shape[1] != channels:
        raise ValueError(f"x has {x.shape[1]} channels, weight has {channels}")
    if bias is not None and (bias.dim() != 1 or bias.shape[0] != channels):
        raise ValueError(f"bias must be ({channels},), got {tuple(bias.shape)}")
    if initial_state is not None:
        expected = (x.shape[0], channels, width - 1)
        if tuple(initial_state.shape) != expected:
            raise ValueError(
                f"initial_state must be {expected}, got {tuple(initial_state.shape)}"
            )

    compute = weight.dtype
    work = x.to(compute).contiguous()
    init = initial_state.to(compute).contiguous() if initial_state is not None else None
    shift = bias.to(compute).contiguous() if bias is not None else None

    # The final state is a slice of ``cat(initial_state, x)``, so it has to stay
    # in the autograd graph whenever a gradient can flow through it; with no
    # gradient in play the forward kernel writes it out instead of a copy.
    fused_final = return_final_state and not _grads(work, weight, init)

    # An empty batch or an empty sequence has no convolution to run, and neither
    # conv1d nor the kernels accept a zero-length input.
    if work.numel() == 0:
        out = work.new_empty(work.shape)
    elif _metal_ok(work, weight) and fused_final:
        out = torch.empty_like(work)
        final = work.new_empty(work.shape[:2] + (width - 1,))
        _forward_metal(work, weight.contiguous(), shift, init, activation, out, final, True)
        return out.to(x.dtype), final.to(x.dtype)
    elif _metal_ok(work, weight):
        out = _CausalConv1d.apply(work, weight.contiguous(), shift, init, activation)
    else:
        out = _forward_aten(work, weight, shift, init, activation)
    out = out.to(x.dtype)

    if return_final_state:
        return out, _final_state(work, init, width - 1).to(x.dtype)
    return out


# --------------------------------------------------------------------------- #
# Decode: one or a few tokens against the rolling state.
# --------------------------------------------------------------------------- #

def _update_aten(x, state, weight, bias, activation):
    channels = weight.shape[0]
    length = x.shape[-1]
    compute = weight.dtype
    joined = torch.cat([state.to(compute), x.to(compute)], dim=-1)
    shift = bias.to(compute) if bias is not None else None
    out = F.conv1d(joined, weight.unsqueeze(1), shift, groups=channels)
    state.copy_(joined[..., -state.shape[-1]:])
    out = out[..., -length:]
    if activation is not None:
        out = F.silu(out)
    return out.to(x.dtype)


def causal_conv1d_update(x, state, weight, bias=None, activation=None):
    """Advance the causal convolution by ``q`` tokens, updating ``state`` in place.

    Args:
        x: ``(batch, channels)`` or ``(batch, channels, q)``.
        state: ``(batch, channels, s)``, ``s >= K-1``, last stride 1, overwritten
            in place with ``cat(state, x)[..., -s:]``.
        weight: ``(channels, 1, K)`` or ``(channels, K)``.
        bias: ``(channels,)`` or ``None``.
        activation: ``None``, ``"silu"`` or ``"swish"``.

    Returns:
        ``(batch, channels)`` or ``(batch, channels, q)``, matching the rank of
        ``x``.  This is the decode path: it carries no autograd.
    """
    weight = _as_depthwise(weight)
    _check_activation(activation)
    squeezed = x.dim() == 2
    if squeezed:
        x = x.unsqueeze(-1)
    if x.dim() != 3:
        raise ValueError(
            f"x must be (batch, channels) or (batch, channels, q), got {tuple(x.shape)}"
        )
    channels, width = weight.shape
    taps = width - 1
    batch, xchannels, length = x.shape
    if xchannels != channels:
        raise ValueError(f"x has {xchannels} channels, weight has {channels}")
    if state.dim() != 3 or state.shape[0] != batch or state.shape[1] != channels:
        raise ValueError(
            f"state must be ({batch}, {channels}, s) with s >= {taps}, "
            f"got {tuple(state.shape)}"
        )
    span = state.shape[-1]
    if span < taps:
        raise ValueError(f"state must hold at least K-1 = {taps} columns, got {span}")
    if state.stride(-1) != 1:
        raise ValueError(f"state must have stride 1 in its last dimension, got {state.stride()}")
    if bias is not None and (bias.dim() != 1 or bias.shape[0] != channels):
        raise ValueError(f"bias must be ({channels},), got {tuple(bias.shape)}")
    if x.requires_grad or weight.requires_grad or state.requires_grad:
        raise RuntimeError("causal_conv1d_update is the decode path and carries no autograd")

    work = x.contiguous()
    if work.numel() == 0:
        out = work.new_empty(work.shape)
        return out.squeeze(-1) if squeezed else out
    usable = (
        _metal_ok(work, weight)
        and state.dtype == work.dtype
        and (bias is None or bias.dtype == work.dtype)
    )
    if usable:
        out = torch.empty_like(work)
        rows = batch * channels
        _shader(_UPDATE, work.dtype, width).causal_conv1d_update(
            state, work, weight.contiguous(), bias if bias is not None else work, out,
            channels, length, span, rows,
            state.stride(0), state.stride(1),
            1 if bias is not None else 0,
            1 if activation is not None else 0,
            threads=rows, group_size=_group(rows),
        )
    else:
        out = _update_aten(work, state, weight, bias, activation)
    return out.squeeze(-1) if squeezed else out
