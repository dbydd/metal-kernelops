"""Four-path 2D selective scan (VMamba's ``ss2d``) on Metal.

What it computes
----------------
A 2D map is traversed along four paths; each path is scanned with the same
selective scan a 1D Mamba layer uses, and the four results are added at every
pixel. With ``x`` of shape ``(batch, channels, height, width)`` and
``L = height * width``, the four traversal orders are, in path order:

======  ==========================  ==========================================
path    index of pixel ``(h, w)``   VMamba's label
======  ==========================  ==========================================
0       ``h * W + w``               TL -> BR, row-major
1       ``w * H + h``               TR -> BL, column-major
2       ``L - 1 - (h * W + w)``     BR -> TL, path 0 reversed
3       ``L - 1 - (w * H + h)``     BL -> TR, path 1 reversed
======  ==========================  ==========================================

Path 1 is the map written transposed and then read row-major: it sweeps down
every column, left to right, still starting at ``(0, 0)`` and ending at
``(H-1, W-1)``. That column-by-column sweep is the one VMamba labels
``TR -> BL``. The reverse paths are the exact reverses of paths 0 and 1.

How it runs
-----------
One Metal kernel walks all four paths over the ``(height, width)`` grid. A
threadgroup holds one path, one block of ``32 / states`` channels and one chunk,
with one state per lane, so a lane carries the recurrence for its own state and
the warp reduces the output dot product across states. The path is a cursor over
``h`` and ``w`` that wraps at the grid edge, so no permuted copy of ``u``,
``dt``, ``B`` or ``C`` is ever built. The four paths reach a given pixel in four
different step slots, so each path writes its own accumulator slice at the pixel
it stands on, and one pass adds the four slices -- which is where the previous
implementation's merge permutation used to run. The recurrence, and the
``CHUNK``-step chunking with its chunk-entry summary and prefix chain, are the
same as :mod:`metal_kernelops.scan`.

Forward is five launches: transpose ``B`` and ``C`` to state-minor, chunk
summaries, the prefix chain (``scan``'s ``boundaries``), the scan, the merge. At
the measured shapes it moves 4 MB, so the launches rather than the arithmetic
set the time.

The backward pass reuses the three kernels of :mod:`metal_kernelops.scan`
(``reverse_summaries``, ``reverse_boundaries``, ``gradients``) on the stacked
layout, because ``dB`` and ``dC`` reduce over the channels of a group and that
layout is what exposes them to a threadgroup reduction instead of atomics. The
reuse is why the backward still builds the stacked inputs and the step-ordered
``dy``; only its permutation and merge are fused, into ``merge_paths``.

Layout contract
---------------
``u``, ``dt``  ``(batch, channels, height, width)``
``A``          ``(channels, states)``, negative real decay
``B``, ``C``   ``(batch, groups, states, height, width)``
output         ``(batch, channels, height, width)``, contiguous

``groups`` must divide ``channels``; the ``channels / groups`` consecutive
channels of one group share that group's ``B`` and ``C``. A ``groups`` that does
not divide ``channels`` would leave channels with no group, so it is rejected
rather than silently dropped. ``states`` must be at most 32, one per lane. A
zero extent in any of ``height``, ``width``, ``channels``, ``states`` or
``batch`` produces the mathematically empty result, and gradients of the empty
result are zero rather than an autograd error.

The fused path needs float32 on the ``mps`` device and does not require
contiguous inputs. Entry on another device or dtype is rejected --
:func:`ss2d_scan_reference` is the CPU reference for every dtype.

Autograd
--------
One ``backward`` produces ``du``, ``dt``, ``dA``, ``dB``, ``dC``. ``dt``, ``A``,
``B`` and ``C`` are shared by the four paths, and their gradients accumulate the
four path contributions, matching a loop over paths that reuses the same
parameters.

Per-path parameters
-------------------
VMamba projects its own ``dt``, ``B`` and ``C`` per path before scanning. That
case is not this signature: it applies one ``dt``, ``A``, ``B``, ``C`` to all
four paths. To scan per-path parameters with four times the channels and
groups, that is exactly the stacked layout the backward pass uses; call
:func:`metal_kernelops.scan.selective_scan` directly.
"""

from functools import lru_cache

import torch
from torch.autograd.function import once_differentiable

from ..scan import CHUNK, _library as _scan_library, _pick_block

__all__ = ["path_orders", "ss2d_scan", "ss2d_scan_reference"]

MERGE_MODES = ("sum", "mean")
"""How the four path outputs combine: elementwise sum, or their mean."""

MAX_STATES = 32
"""One state per lane, so the widest state axis a warp can carry is 32."""

EXP2 = False
"""
Route the recurrence's ``exp(delta * A)`` through ``exp2(x * log2e)``.

Kept off: the two forms agree to every recorded digit and their timings swap
places between runs, because Metal's ``exp`` is itself the hardware exponential
scaled by log2(e). ``exp`` also matches what the reused backward spells.
Measured in ``measured/path_scan.json``.
"""


def _check_merge(merge):
    if merge not in MERGE_MODES:
        raise ValueError(f"merge must be one of {MERGE_MODES}, got {merge!r}")


def _shape_contract(u, dt, A, B, C, where):
    """Validate the shared shape contract, returning the six dimension sizes."""
    if u.dim() != 4:
        raise ValueError(
            f"{where}: u must be (batch, channels, height, width), got {tuple(u.shape)}"
        )
    batch, channels, height, width = u.shape
    if dt.shape != u.shape:
        raise ValueError(f"{where}: dt {tuple(dt.shape)} != u {tuple(u.shape)}")
    if A.dim() != 2:
        raise ValueError(f"{where}: A must be (channels, states), got {tuple(A.shape)}")
    if A.shape[0] != channels:
        raise ValueError(f"{where}: A has {A.shape[0]} rows for {channels} channels")
    if B.dim() != 5:
        raise ValueError(
            f"{where}: B must be (batch, groups, states, height, width), "
            f"got {tuple(B.shape)}"
        )
    if C.shape != B.shape:
        raise ValueError(f"{where}: C {tuple(C.shape)} != B {tuple(B.shape)}")
    if B.shape[0] != batch:
        raise ValueError(f"{where}: B batch {B.shape[0]} != u batch {batch}")
    if B.shape[2] != A.shape[1]:
        raise ValueError(f"{where}: B has {B.shape[2]} states, A has {A.shape[1]}")
    if tuple(B.shape[3:]) != (height, width):
        raise ValueError(f"{where}: B spatial {tuple(B.shape[3:])} != {(height, width)}")
    groups = B.shape[1]
    if channels and groups < 1:
        raise ValueError(f"{where}: {channels} channels need at least one group")
    if groups and channels % groups:
        raise ValueError(
            f"{where}: {channels} channels do not divide into {groups} groups; "
            f"a channel would map to no group"
        )
    return batch, channels, groups, A.shape[1], height, width


def path_orders(height, width, device=None):
    """Spatial index order of each of the four scan paths, as a list of tensors.

    Entry ``p[i]`` of path ``k`` is the flat ``h * width + w`` index of the pixel
    at step ``i``. Paths 2 and 3 are the reverses of paths 0 and 1.
    """
    row_major = torch.arange(height * width, device=device).reshape(height, width)
    flat = row_major.reshape(-1)
    column_major = row_major.transpose(0, 1).reshape(-1)
    return [flat, column_major, flat.flip(0), column_major.flip(0)]


def _stack_spatial(x):
    """(B, C, H, W) -> (B, 4C, H*W) holding the four traversal orders."""
    batch, channels, height, width = x.shape
    length = height * width
    stacked = x.new_empty(batch, 4 * channels, length)
    stacked[:, :channels] = x.reshape(batch, channels, length)
    stacked[:, channels : 2 * channels] = x.transpose(2, 3).reshape(
        batch, channels, length
    )
    stacked[:, 2 * channels :] = torch.flip(stacked[:, : 2 * channels], dims=[-1])
    return stacked


def _stack_groups(x):
    """(B, G, N, H, W) -> (B, 4G, N, H*W) holding the four traversal orders."""
    batch, groups, states, height, width = x.shape
    length = height * width
    stacked = x.new_empty(batch, 4 * groups, states, length)
    stacked[:, :groups] = x.reshape(batch, groups, states, length)
    stacked[:, groups : 2 * groups] = x.transpose(3, 4).reshape(
        batch, groups, states, length
    )
    stacked[:, 2 * groups :] = torch.flip(stacked[:, : 2 * groups], dims=[-1])
    return stacked


def _zero_result(shape, tensors):
    """A zero result that still carries the gradient graph of ``tensors``."""
    zero = tensors[0].sum() * 0
    for tensor in tensors[1:]:
        zero = zero + tensor.sum() * 0
    return tensors[0].new_zeros(shape) + zero


@lru_cache(maxsize=16)
def _library(lanes, exp2):
    """Shader library for one lane width and exponential form, compiled on use."""
    source = (
        _SOURCE.replace("$SP", str(lanes))
        .replace("$EXP2", "1" if exp2 else "0")
        .replace("$CHUNK", str(CHUNK))
    )
    return torch.mps.compile_shader(source)


_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

constant uint chunk_size = $CHUNK;
constant uint sp = $SP;
constant uint cpw = 32 / sp;

// exp(delta * A), the recurrence's only transcendental. $EXP2 spells it as the
// hardware exponential scaled by log2(e), which costs one extra multiply.
static inline float exp_of(float z) {
#if $EXP2
    return exp2(z * 1.4426950408889634f);
#else
    return exp(z);
#endif
}

// A path's position, as an inner index that runs out first and an outer index
// that advances when it wraps. Row-major paths run over ``w`` then ``h``,
// column-major paths over ``h`` then ``w``; both reverses walk the same lattice
// the other way. One pair of branches therefore covers all four paths.
struct Walk {
    uint inner, outer;
};

// Index of the pixel a walk stands on, and the step mapping back from a pixel.
static inline uint pixel_of(Walk q, uint width, bool column) {
    return column ? q.inner * width + q.outer : q.outer * width + q.inner;
}

static inline Walk walk_start(uint path, uint step, uint length, uint height,
                              uint width) {
    uint extent = (path & 1) ? height : width;
    uint s = (path < 2) ? step : length - 1 - step;
    Walk q;
    q.inner = s % extent;
    q.outer = s / extent;
    return q;
}

static inline void walk_next(thread Walk& q, uint extent, bool forward) {
    if (forward) {
        if (++q.inner == extent) {
            q.inner = 0;
            ++q.outer;
        }
    } else if (q.inner == 0) {
        q.inner = extent - 1;
        --q.outer;
    } else {
        --q.inner;
    }
}

// B and C arrive as (rows, states, length) and every kernel here wants the
// states of one pixel next to each other, so this one pass transposes both.
kernel void transpose_states(
    device const float* in_b, device float* out_b,
    device const float* in_c, device float* out_c,
    constant uint& rows, constant uint& length, constant uint& states,
    uint gid [[thread_position_in_grid]]) {
    uint pixel = gid % length, row = gid / length;
    if (row >= rows) return;
    ulong source = (ulong)row * states * length + pixel;
    ulong target = ((ulong)row * length + pixel) * states;
    for (uint n = 0; n < states; ++n) {
        out_b[target + n] = in_b[source + n * length];
        out_c[target + n] = in_c[source + n * length];
    }
}

// Chunk summaries per (batch, path, channel, chunk): the affine map the chunk
// applies to the state entering it. Lane ``n`` carries state ``n``.
kernel void path_summaries(
    device const float* u, device const float* dt, device const float* A,
    device const float* B, device float* starts, device float* decays,
    constant uint& channels, constant uint& groups, constant uint& states,
    constant uint& height, constant uint& width, constant uint& chunks,
    uint gid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    uint k = gid % chunks, rest = gid / chunks;
    uint path = rest % 4; rest /= 4;
    uint d = (rest % ((channels + cpw - 1) / cpw)) * cpw + lane / sp;
    uint b = rest / ((channels + cpw - 1) / cpw);
    uint n = lane % sp;
    bool active = d < channels && n < states;
    uint length = height * width;
    uint start = k * chunk_size, end = min(start + chunk_size, length);
    float av = active ? A[d * states + n] : 0.0f;
    device const float* U = u + (ulong)(b * channels + d) * length;
    device const float* DT = dt + (ulong)(b * channels + d) * length;
    device const float* BP = B + (ulong)(b * groups + d / (channels / groups)) *
                                states * length;
    Walk walk = walk_start(path, start, length, height, width);
    uint extent = (path & 1) ? height : width;
    bool column = (path & 1) != 0, forward = path < 2;
    float drive = 0.0f, log_decay = 0.0f;
    for (uint t = start; t < end; ++t) {
        uint pixel = pixel_of(walk, width, column);
        float delta = DT[pixel], a = exp_of(delta * av);
        drive = a * drive + delta * U[pixel] * BP[(ulong)pixel * states + n];
        log_decay += delta * av;
        walk_next(walk, extent, forward);
    }
    ulong at = ((ulong)((b * 4 + path) * channels + d) * chunks + k) * states + n;
    if (active) {
        starts[at] = drive;
        decays[at] = exp_of(log_decay);
    }
}

// The scan itself: lane ``n`` holds state ``n`` of ``cpw`` channels at once, the
// four paths run in lockstep over this chunk, and each path writes its own
// accumulator slice at the pixel it stands on.
kernel void path_scan(
    device const float* u, device const float* dt, device const float* A,
    device const float* B, device const float* C, device const float* starts,
    device float* y,
    constant uint& channels, constant uint& groups, constant uint& states,
    constant uint& height, constant uint& width, constant uint& chunks,
    uint gid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    uint k = gid % chunks, rest = gid / chunks;
    uint path = rest % 4; rest /= 4;
    uint blocks = (channels + cpw - 1) / cpw;
    uint d = (rest % blocks) * cpw + lane / sp;
    uint b = rest / blocks;
    uint n = lane % sp;
    uint length = height * width;
    uint start = k * chunk_size, end = min(start + chunk_size, length);
    bool active = d < channels && n < states;
    float av = active ? A[d * states + n] : 0.0f;
    device const float* U = u + (ulong)(b * channels + d) * length;
    device const float* DT = dt + (ulong)(b * channels + d) * length;
    device const float* BP = B + (ulong)(b * groups + d / (channels / groups)) *
                                states * length;
    device const float* CP = C + (ulong)(b * groups + d / (channels / groups)) *
                                states * length;
    ulong row = (ulong)((b * 4 + path) * channels + d);
    float x = active ? starts[(row * chunks + k) * states + n] : 0.0f;
    Walk walk = walk_start(path, start, length, height, width);
    uint extent = (path & 1) ? height : width;
    bool column = (path & 1) != 0, forward = path < 2;
    for (uint t = start; t < end; ++t) {
        uint pixel = pixel_of(walk, width, column);
        float yt = 0.0f;
        if (active) {
            float delta = DT[pixel];
            ulong at = (ulong)pixel * states + n;
            x = exp_of(delta * av) * x + delta * U[pixel] * BP[at];
            yt = x * CP[at];
        }
        _Pragma("unroll")
        for (int stride = sp / 2; stride >= 1; stride >>= 1) {
            yt += simd_shuffle_xor(yt, (ushort)stride);
        }
        if (n == 0 && d < channels) y[row * length + pixel] = yt;
        walk_next(walk, extent, forward);
    }
}

// VMamba's cross_merge order, (0 + 2) + (1 + 3). The scan writes each path at
// the pixel it stands on, so the forward is a plain four-way add; the backward's
// gradients come back in path-step order, so it passes ``permute`` to gather
// them back. One kernel serves both.
static inline ulong at(ulong base, uint stride, uint path, uint index) {
    return base + (ulong)path * stride + index;
}

kernel void merge_paths(
    device const float* in, device float* out,
    constant uint& rows, constant uint& length, constant uint& elements,
    constant uint& height, constant uint& width,
    constant float& scale, constant uint& permute,
    uint gid [[thread_position_in_grid]]) {
    if (gid >= elements) return;
    uint total = rows * length;
    uint pixel = gid % length, rest = gid / length;
    uint row = rest % rows, b = rest / rows;
    uint indices[4] = {pixel, pixel, pixel, pixel};
    if (permute) {
        uint h = pixel / width, w = pixel - h * width;
        uint column = w * height + h;
        indices[1] = column;
        indices[2] = length - 1 - pixel;
        indices[3] = length - 1 - column;
    }
    ulong base = (ulong)b * 4 * total + (ulong)row * length;
    float summed = (in[at(base, total, 0, indices[0])] +
                    in[at(base, total, 2, indices[2])]) +
                   (in[at(base, total, 1, indices[1])] +
                    in[at(base, total, 3, indices[3])]);
    out[(ulong)b * total + (ulong)row * length + pixel] = summed * scale;
}
"""


def _lane_width(states):
    """Lanes one channel's state axis occupies: the next power of two, up to 32.

    Every state needs its own lane, so this is never narrower than ``states``:
    the remaining lanes of the warp carry further channels instead. At
    ``states = 16`` that is two channels per warp, and the sweep in
    ``measured/path_scan.json`` shows that beats one channel per warp.
    """
    width = 1
    while width < states:
        width <<= 1
    return min(width, MAX_STATES)


def _grid(elements):
    """Thread count covering ``elements`` in whole 64-wide threadgroups."""
    return (elements + 63) // 64 * 64


class _PathScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, dt, A, B, C, scale):
        batch, channels, height, width = u.shape
        u, dt, A = u.contiguous(), dt.contiguous(), A.contiguous()
        groups, states = B.shape[1], A.shape[1]
        length = height * width
        chunks = (length + CHUNK - 1) // CHUNK
        lanes = _lane_width(states)
        per_warp = 32 // lanes
        blocks = (channels + per_warp - 1) // per_warp
        library = _library(lanes, EXP2)
        flat = (batch, groups, length, states)
        flat_b, flat_c = u.new_empty(flat), u.new_empty(flat)
        library.transpose_states(
            B.reshape(batch, groups, states, length), flat_b,
            C.reshape(batch, groups, states, length), flat_c,
            batch * groups, length, states,
            threads=_grid(batch * groups * length), group_size=64,
        )
        starts = u.new_empty(batch, 4, channels, chunks, states)
        decays = torch.empty_like(starts)
        groups_span = batch * 4 * blocks * chunks
        library.path_summaries(
            u, dt, A, flat_b, starts, decays,
            channels, groups, states, height, width, chunks,
            threads=groups_span * 32, group_size=32,
        )
        _scan_library(_pick_block(channels // groups)).boundaries(
            starts, decays, chunks, states,
            threads=batch * 4 * channels * 32, group_size=32,
        )
        folded = u.new_empty(batch, 4, channels, length)
        library.path_scan(
            u, dt, A, flat_b, flat_c, starts, folded,
            channels, groups, states, height, width, chunks,
            threads=groups_span * 32, group_size=32,
        )
        out = u.new_empty(batch, channels, length)
        library.merge_paths(
            folded, out, channels, length, batch * channels * length,
            height, width, scale, 0,
            threads=_grid(batch * channels * length), group_size=64,
        )
        ctx.scale = scale
        ctx.save_for_backward(u, dt, A, B, C, starts, decays)
        return out.reshape(batch, channels, height, width)

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        u, dt, A, B, C, starts, decays = ctx.saved_tensors
        batch, channels, height, width = u.shape
        groups, states = B.shape[1], A.shape[1]
        dim, folded_groups = 4 * channels, 4 * groups
        length = height * width
        chunks = starts.shape[3]
        # The backward reuses scan.py's three kernels, which walk the stacked
        # layout and reduce over the channels of a group: dB and dC need that
        # reduction, and it is cheaper to give the whole pass the stacked inputs
        # than to write a second backward. Only the permutation and merge are
        # fused here, in merge_paths.
        library = _library(_lane_width(states), EXP2)
        stacked_u = _stack_spatial(u)
        stacked_dt = _stack_spatial(dt)
        stacked_a = A.repeat(4, 1)
        stacked_b = _stack_groups(B).transpose(-1, -2).contiguous()
        stacked_c = _stack_groups(C).transpose(-1, -2).contiguous()
        stacked_dy = _stack_spatial(dy * ctx.scale)
        adjoints = torch.empty_like(starts)
        du, ddt = u.new_empty(batch, dim, length), u.new_empty(batch, dim, length)
        dA = torch.empty_like(starts)
        dB, dC = torch.zeros_like(stacked_b), torch.zeros_like(stacked_c)
        block = _pick_block(channels // groups)
        scan = _scan_library(block)
        shape = (dim, length, folded_groups, states, chunks)
        paths = {"threads": batch * dim * chunks * 32, "group_size": 32}
        scan.reverse_summaries(
            stacked_dt, stacked_a, stacked_c, stacked_dy, adjoints,
            *shape, **paths,
        )
        scan.reverse_boundaries(
            adjoints, decays, chunks, states,
            threads=batch * dim * 32, group_size=32,
        )
        scan.gradients(
            stacked_u, stacked_dt, stacked_a, stacked_b, stacked_c, starts,
            adjoints, stacked_dy, du, ddt, dA, dB, dC, *shape, dim // block,
            threads=batch * (dim // block) * chunks * 32 * block,
            group_size=32 * block,
        )
        # Every gradient comes back in path-step order, so one permuting fold per
        # tensor turns the four path slices into per-pixel sums.
        def fold(source, rows):
            out = u.new_empty(batch, rows, length)
            library.merge_paths(
                source, out, rows, length, batch * rows * length,
                height, width, 1.0, 1,
                threads=_grid(batch * rows * length), group_size=64,
            )
            return out

        grad_u = fold(du, channels)
        grad_dt = fold(ddt, channels)
        grad_b = fold(dB.transpose(-1, -2).contiguous().reshape(
            batch, 4, groups * states, length), groups * states)
        grad_c = fold(dC.transpose(-1, -2).contiguous().reshape(
            batch, 4, groups * states, length), groups * states)
        return (
            grad_u.view(batch, channels, height, width),
            grad_dt.view(batch, channels, height, width),
            dA.view(batch, dim, chunks, states).sum(dim=(0, 2))
              .view(4, channels, states).sum(0),
            grad_b.view(batch, groups, states, height, width),
            grad_c.view(batch, groups, states, height, width),
            None,
        )


def ss2d_scan(u, dt, A, B, C, merge="sum"):
    """Four-path 2D selective scan, one fused Metal pass. float32, mps only.

    Args:
        u, dt: (batch, channels, height, width)
        A: (channels, states), negative
        B, C: (batch, groups, states, height, width)
        merge: ``"sum"`` adds the four path outputs, ``"mean"`` divides by four.

    Returns:
        (batch, channels, height, width) contiguous output.
    """
    _check_merge(merge)
    batch, channels, groups, states, height, width = _shape_contract(
        u, dt, A, B, C, "ss2d_scan"
    )
    for name, tensor in (("u", u), ("dt", dt), ("A", A), ("B", B), ("C", C)):
        if tensor.device.type != "mps":
            raise ValueError(f"ss2d_scan runs Metal kernels, {name} is on {tensor.device}")
        if tensor.dtype != torch.float32:
            raise ValueError(f"ss2d_scan is float32 only, {name} is {tensor.dtype}")

    if states > MAX_STATES:
        raise ValueError(
            f"ss2d_scan holds one state per lane, states must be at most "
            f"{MAX_STATES}, got {states}"
        )
    if u.numel() == 0 or B.numel() == 0:
        return _zero_result((batch, channels, height, width), (u, dt, A, B, C))

    return _PathScan.apply(u, dt, A, B, C, 0.25 if merge == "mean" else 1.0)


def _scan_reference(sequence_u, sequence_dt, A, sequence_b, sequence_c):
    """Step-by-step selective scan on a 1D sequence, written functionally.

    ``metal_kernelops.reference.selective_scan`` overwrites its running state in
    place, which autograd cannot differentiate; this one rebuilds the state each
    step so the reference carries gradients.
    """
    batch, channels, length = sequence_u.shape
    states = A.shape[1]
    groups = sequence_b.shape[1]
    per_group = channels // groups if groups else 0
    state = sequence_u.new_zeros(batch, channels, states)
    outputs = []
    for step in range(length):
        delta = sequence_dt[:, :, step]
        b_step = sequence_b[:, :, :, step].repeat_interleave(per_group, dim=1)
        c_step = sequence_c[:, :, :, step].repeat_interleave(per_group, dim=1)
        state = torch.exp(delta.unsqueeze(-1) * A) * state + (
            delta * sequence_u[:, :, step]
        ).unsqueeze(-1) * b_step
        outputs.append((state * c_step).sum(-1))
    if length == 0:
        return _zero_result(
            (batch, channels, 0),
            (sequence_u, sequence_dt, A, sequence_b, sequence_c),
        )
    return torch.stack(outputs, dim=-1)


def ss2d_scan_reference(u, dt, A, B, C, merge="sum"):
    """CPU reference for :func:`ss2d_scan`, in pure torch on any device/dtype.

    Builds each path from explicit index arithmetic rather than transposed
    views, so it also checks the fused implementation's permutations.
    """
    _check_merge(merge)
    batch, channels, groups, states, height, width = _shape_contract(
        u, dt, A, B, C, "ss2d_scan_reference"
    )
    length = height * width
    sequences_u = u.reshape(batch, channels, length)
    sequences_dt = dt.reshape(batch, channels, length)
    sequences_b = B.reshape(batch, groups, states, length)
    sequences_c = C.reshape(batch, groups, states, length)

    restored = []
    for order in path_orders(height, width, u.device):
        scan = _scan_reference(
            sequences_u.index_select(-1, order),
            sequences_dt.index_select(-1, order),
            A,
            sequences_b.index_select(-1, order),
            sequences_c.index_select(-1, order),
        )
        restored.append(
            sequences_u.new_zeros(batch, channels, length).index_copy(-1, order, scan)
        )
    total = (restored[0] + restored[2]) + (restored[1] + restored[3])
    if merge == "mean":
        total = total * 0.25
    return total.reshape(batch, channels, height, width)
