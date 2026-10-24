"""MPS dispatch-key overlays for ``segment_reduce`` and ``_segment_reduce_backward``.

Both overloads raise ``NotImplementedError`` on MPS in torch 2.14.1. Both are
registered, so autograd's MPS node for ``segment_reduce`` reaches the backward
overload on the same device and gradients flow with no CPU round trip.

Layout
------
``data`` is viewed as ``(rows, axis_len, inner)``: ``rows`` folds every
dimension before ``axis``, ``inner`` every dimension after it. ``lengths`` and
``offsets`` are per-row descriptors of shape ``rows_shape + (segments,)`` and
``rows_shape + (segments + 1,)``. Either descriptor becomes one int64
``(rows, segments + 1)`` boundary table on the device.

Two paths, one contract
-----------------------
The kernel path runs when ``data`` is contiguous and one threadgroup per segment
keeps the GPU busy, which is every case whose descriptors passed validation.
One threadgroup owns one ``(row, segment)`` output: it reads ``bounds[segment]``
and ``bounds[segment + 1]``, strides the axis itself and reduces in registers,
so there is no position-to-segment map, no scatter, no cumulative sum, no
scratch row and no per-column ``index_add_``. Segments narrower than 32 outputs
(the per-row case, ``inner == 1``) are reduced lane-parallel with one SIMD group
per segment; wider ones give each thread a set of ``inner`` columns so no
cross-lane reduction is needed at all. The backward kernel mirrors that split
and needs no second map either: the tie count, the product of the non-zero
elements and the zero count are accumulated in the same walk that writes the
gradient, and the positions outside every segment -- only a prefix or a suffix,
because monotone boundaries leave no gaps -- are zeroed by the first and last
segment of each row.

``scan_bounds`` is the only other dispatch on the kernel path and the only host
read the operator makes: one thread per row walks the descriptor, writes the
clamped boundary table and leaves a verdict in a two-word flag. The strict
forward launches the reduction before reading that flag back, so its one round
trip covers the kernel instead of preceding it.

Everything else takes the aten fallback: a ``rows x axis_len`` int32
position-to-segment map built with a scatter of one marker per segment start
followed by a cumulative sum, then one ``index_add_``/``index_reduce_`` over the
positions. That covers non-contiguous input and shapes too small or too
column-heavy to fill the GPU from one threadgroup per segment -- one thread
walking more than ``_MAX_ROUNDS`` elements. Positions no segment covers are
directed at one scratch row appended to the accumulator and dropped afterwards,
so ``data`` is never masked elementwise.

Semantics matched against the CPU kernel
----------------------------------------
* ``lengths`` means segment lengths and wins when both descriptors are given;
  ``axis`` must equal ``lengths.dim() - 1``; segment counts may differ per row.
* The ``axis`` check, the ``indices`` rejection, negative lengths, unsorted
  offsets and the "rows sum to the axis length" check reproduce the CPU errors.
  ``unsafe=True`` skips only the last of these.
* Empty segments: sum 0, prod 1, min +inf, max -inf, mean 0/0 = nan; each
  becomes ``initial`` when it is given. For ``mean`` the divisor is the element
  count, except that a given ``initial`` floors it at one.
* ``max``/``min`` split the gradient evenly across positions equal to the
  output, as on CPU. ``prod`` treats zeros exactly: one zero in a segment takes
  the product of the rest, two or more zero every gradient in that segment.
* Singleton leading dimensions of ``lengths``/``offsets`` broadcast; any other
  mismatch raises instead of returning zero rows the way the CPU kernel does.

Known delta: ``max``/``min`` are bit-exact against the CPU kernel. Sums, products
and means differ only in association -- the kernel reduces lanes in parallel and
keeps fp16/bf16 accumulators in float32 until the final narrowing -- so those
match to a tolerance rather than bit for bit.
"""

from functools import lru_cache

import torch

CHECKED_TORCH = "2.14.1"

_MODES = {
    "sum": "sum",
    "mean": "mean",
    "prod": "prod",
    "max": "amax",
    "amax": "amax",
    "min": "amin",
    "amin": "amin",
}

_IDENTITY = {
    "sum": 0.0,
    "mean": 0.0,
    "prod": 1.0,
    "amax": float("-inf"),
    "amin": float("inf"),
}

_AXIS = "segment_reduce(): Expected axis to be the last dimension of {} but got {}."
_RANK = (
    "Expected data.dim() >= {}_value.dim() to be true, but got false.  "
    "(Could this error message be improved?  If so, please report an "
    "enhancement request to PyTorch.)"
)
_NEGATIVE_SEGMENT = (
    "Expected segment_length >= 0 to be true, but got false.  "
    "(Could this error message be improved?  If so, please report an "
    "enhancement request to PyTorch.)"
)
_SUM = (
    "segment_reduce(): Expected all rows of lengths along axis to sum to "
    "data.size(lengths.dim()-1) when !unsafe."
)
_REDUCE = "reduce argument must be either sum, prod, mean, amax or amin, got {}"
_INDICES = "segment_reduce(): indices based reduction is not supported yet."
_MISSING = "segment_reduce(): Either lengths or offsets must be defined."
_LEAD = "segment_reduce(): Expected {} leading dimensions {} to match data leading dimensions {}."
_DTYPE = "segment_reduce(): MPS kernel not implemented for '{}'"

_FLAG_SUM = 1
"""``scan_bounds`` error code: ``lengths`` do not sum to the axis length."""
_FLAG_BAD = 2
"""``scan_bounds`` error code: a negative length or unsorted offsets."""

_SPEC_DTYPES = {torch.int32: 0, torch.int64: 1}
"""Descriptor dtypes ``scan_bounds`` reads in place; anything else is widened."""


def _rows_inner(shape, axis):
    rows = 1
    for size in shape[:axis]:
        rows *= size
    inner = 1
    for size in shape[axis + 1 :]:
        inner *= size
    return rows, inner


def _layout(data, lengths, offsets, axis, unsafe, strict=True):
    """Normalize a segment descriptor to ``(bounds, rows, segments, axis_len, inner, axis, exact)``.

    ``bounds`` is an int64 ``(rows, segments + 1)`` boundary table on ``data``'s
    device, clamped into ``[0, axis_len]`` and never aliasing caller storage;
    one ``scan_bounds`` dispatch builds it and validates it at the same time.
    ``exact`` reports that every position on the axis belongs to a segment, so
    the position map needs no masking. ``strict`` validation costs the only
    host read the operator makes; the backward pass turns it off the way the
    CPU kernel does and then reads nothing back.
    """
    kind = "lengths" if lengths is not None else "offsets"
    spec = lengths if lengths is not None else offsets
    if spec is None:
        raise RuntimeError(_MISSING)
    if data.dim() < spec.dim():
        raise RuntimeError(_RANK.format(kind))
    given, axis = axis, axis + data.dim() if axis < 0 else axis
    if axis < 0 or axis >= data.dim():
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[-{data.dim()}, {data.dim() - 1}], but got {given})"
        )
    if spec.dim() < 1 or axis != spec.dim() - 1:
        raise RuntimeError(_AXIS.format(kind, axis))
    if spec.dtype.is_floating_point or spec.dtype.is_complex:
        raise NotImplementedError(_DTYPE.format(spec.dtype))

    shape = data.shape
    axis_len = shape[axis]
    rows, inner = _rows_inner(shape, axis)
    width = spec.shape[-1]
    segments = width - (1 if kind == "offsets" else 0)
    if segments < 0:
        raise RuntimeError("segment_reduce(): offsets must have at least one element.")

    if spec.device != data.device:
        spec = spec.to(data.device)
    lead = tuple(shape[:axis])
    spec_lead = tuple(spec.shape[:-1])
    if spec_lead != lead:
        broadcastable = len(spec_lead) == len(lead) and all(
            want == got or got == 1 for got, want in zip(spec_lead, lead)
        )
        if not broadcastable:
            raise RuntimeError(_LEAD.format(kind, spec_lead, lead))
        spec = spec.expand(lead + (width,))
    spec = spec.reshape(rows, width)
    if spec.dtype not in _SPEC_DTYPES:
        spec = spec.to(torch.int64)

    device = data.device
    if segments == 0:
        bounds = torch.zeros((rows, 1), dtype=torch.int64, device=device)
        return bounds, rows, 0, axis_len, inner, axis, None
    bounds = torch.empty((rows, segments + 1), dtype=torch.int64, device=device)
    flag = torch.zeros(2, dtype=torch.int32, device=device)
    _library(data.dtype).scan_bounds(
        spec, bounds, flag, rows, segments, axis_len,
        1 if kind == "offsets" else 0, 1 if strict else 0, 1 if unsafe else 0,
        _SPEC_DTYPES[spec.dtype], threads=rows,
    )
    return bounds, rows, segments, axis_len, inner, axis, flag if strict else None


def _settle(flag, kind):
    """Wait for ``scan_bounds`` and raise what the CPU kernel would; returns ``exact``."""
    if flag is None:
        return False
    uncovered, code = flag.tolist()
    if code == _FLAG_BAD:
        raise RuntimeError(
            "lengths contains negative value!" if kind == "lengths" else _NEGATIVE_SEGMENT
        )
    if code == _FLAG_SUM:
        raise RuntimeError(_SUM)
    return uncovered == 0


def _positions(bounds, rows, segments, axis_len, exact):
    """Map each ``(row, position)`` to ``row * segments + segment``; uncovered positions get ``rows * segments``."""
    device = bounds.device
    starts = bounds[:, :segments].to(torch.int32)
    inside = (starts < axis_len).to(torch.int32)
    starts.clamp_(0, axis_len - 1)
    rank = torch.zeros((rows, axis_len), dtype=torch.int32, device=device)
    rank.scatter_add_(1, starts, inside)
    torch.cumsum(rank, 1, out=rank)
    row_offset = torch.arange(
        -1, rows * segments - 1, segments, dtype=torch.int32, device=device
    ).unsqueeze(1)
    rank.add_(row_offset)
    if not exact:
        position = torch.arange(axis_len, dtype=torch.int32, device=device)
        end = bounds[:, segments : segments + 1].to(torch.int32)
        rank.masked_fill_((rank <= row_offset) | (position >= end), rows * segments)
    return rank.reshape(-1)


def _accumulate(source, target, rows, segments, inner, mode, dtype, initial, device):
    """Reduce ``source`` (positions x inner) into segments; returns ``(rows * segments, inner)``."""
    fill = _IDENTITY[mode] if initial is None else float(initial)
    acc = torch.full((rows * segments + 1, inner), fill, dtype=dtype, device=device)
    if target is not None:
        if mode == "sum" or mode == "mean":
            acc.index_add_(0, target, source)
        else:
            acc.index_reduce_(0, target, source, mode)
    return acc[: rows * segments]


_DTYPE_NAMES = {
    torch.float32: "float",
    torch.float16: "half",
    torch.bfloat16: "bfloat",
}

_MODE_IDS = {"sum": 0, "mean": 1, "prod": 2, "amax": 3, "amin": 4}

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

#define MODE_SUM 0u
#define MODE_MEAN 1u
#define MODE_PROD 2u
#define MODE_MAX 3u
#define MODE_MIN 4u

/* The boundary table, its validation and the uncovered-row count in one
 * dispatch, and the only host read the operator makes.
 *
 * One thread per row walks the descriptor: a single cumulative sum beats a
 * parallel scan at any segment count this operator sees. Each row clamps its
 * boundaries into [0, axis_len] as it writes them, so nothing downstream needs
 * a second pass, and reports two facts into `flag`: whether the row leaves part
 * of the axis uncovered (counted in slot 0) and whether it must raise (code in
 * slot 1, the largest code across rows wins).
 */
kernel void scan_bounds(
    device const int* spec, device long* bounds, device int* flag,
    constant uint& rows, constant uint& segments, constant uint& axis_len,
    constant uint& is_offsets, constant uint& strict, constant uint& unsafe,
    constant uint& spec_i64,
    uint gid [[thread_position_in_grid]]) {
    if (gid >= rows) return;
    device const int* s32 = spec + (long) gid * (segments + is_offsets);
    device const long* s64 = (device const long*) spec + (long) gid * (segments + is_offsets);
    device long* b = bounds + (long) gid * (segments + 1);
    long limit = (long) axis_len;
    long running = 0, previous = 0;
    bool bad = false;
    if (is_offsets) {
        for (uint k = 0; k <= segments; ++k) {
            long value = spec_i64 ? s64[k] : (long) s32[k];
            bad = bad || (k > 0 && value < previous);
            previous = value;
            b[k] = clamp(value, 0L, limit);
        }
    } else {
        b[0] = 0;
        for (uint k = 0; k < segments; ++k) {
            long value = spec_i64 ? s64[k] : (long) s32[k];
            bad = bad || value < 0;
            running += value;
            b[k + 1] = clamp(running, 0L, limit);
        }
    }
    /* The codes ascend in the order the CPU kernel raises them, so the largest
     * code any row reports is the error to raise. */
    int code = 0;
    if (strict && !unsafe && !is_offsets && running != limit) {
        code = $FLAG_SUM;
    }
    if (strict && bad) {
        code = $FLAG_BAD;
    }
    if (code) {
        atomic_fetch_max_explicit((device atomic_int*) (flag + 1), code,
                                  memory_order_relaxed);
    } else if (b[0] != 0 || b[segments] != limit) {
        atomic_fetch_add_explicit((device atomic_int*) flag, 1, memory_order_relaxed);
    }
}

inline float sload(device const float* p, long i) { return p[i]; }
inline float sload(device const half* p, long i) { return float(p[i]); }
inline float sload(device const bfloat* p, long i) { return float(p[i]); }
inline void sstore(device float* p, long i, float v) { p[i] = v; }
inline void sstore(device half* p, long i, float v) { p[i] = half(v); }
inline void sstore(device bfloat* p, long i, float v) { p[i] = bfloat(v); }

/* Start value of an empty segment, and the value a partial reduction folds in
 * before it has seen an element. */
inline float seg_identity(uint mode) {
    if (mode == MODE_PROD) return 1.0f;
    if (mode == MODE_MAX) return -INFINITY;
    if (mode == MODE_MIN) return INFINITY;
    return 0.0f;
}

/* ATen amax/amin propagate a NaN; fmax/fmin drop it, so the flag is separate. */
inline void seg_step(uint mode, thread float& acc, thread bool& nan, float v) {
    if (mode == MODE_PROD) {
        acc *= v;
    } else if (mode == MODE_MAX) {
        nan = nan || isnan(v);
        acc = fmax(acc, v);
    } else if (mode == MODE_MIN) {
        nan = nan || isnan(v);
        acc = fmin(acc, v);
    } else {
        acc += v;
    }
}

inline float seg_finish(uint mode, float acc, bool nan, uint count,
                        uint has_initial, float initial) {
    if (mode == MODE_SUM) return has_initial ? initial + acc : acc;
    if (mode == MODE_PROD) return has_initial ? initial * acc : acc;
    if (mode == MODE_MEAN) {
        if (count == 0) return has_initial ? initial : NAN;
        return (has_initial ? initial + acc : acc) / float(count);
    }
    if (nan) return NAN;
    if (has_initial) acc = (mode == MODE_MAX) ? fmax(acc, initial) : fmin(acc, initial);
    return acc;
}

inline float lane_reduce(uint mode, float v) {
    if (mode == MODE_PROD) return simd_product(v);
    if (mode == MODE_MAX) return simd_max(v);
    if (mode == MODE_MIN) return simd_min(v);
    return simd_sum(v);
}

/* The dashes left of bounds[0] and right of bounds[segments] belong to no
 * segment. Monotone boundaries leave no gaps between segments, so those two
 * ranges are all a row has to zero, and the first and last segment of the row
 * do it. A segment covers a contiguous (position, inner) block, so both ranges
 * are flat runs of axis_len * inner elements. */
inline void seg_zero_edges(device $T* dx, long offset, long axis_len, uint inner,
                           uint start, uint end, uint segments, uint seg,
                           uint tid, uint lanes) {
    long lead = (long) start * inner;
    long trail = (long) axis_len * inner;
    if (seg == 0) {
        for (long i = tid; i < lead; i += lanes) sstore(dx, offset + i, 0.0f);
    }
    if (seg + 1 == segments) {
        for (long i = (long) end * inner + tid; i < trail; i += lanes) {
            sstore(dx, offset + i, 0.0f);
        }
    }
}

/* Wide path: one threadgroup per segment, one thread per run of inner columns,
 * so nothing is reduced across lanes. */
kernel void seg_cols(
    device const $T* x, device const long* bounds, device $T* out,
    constant uint& segments, constant uint& axis_len, constant uint& inner,
    constant uint& mode, constant uint& has_initial, constant float& initial,
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    uint row = gid / segments;
    uint seg = gid - row * segments;
    long at = (long) row * (segments + 1) + seg;
    uint start = uint(bounds[at]), end = uint(bounds[at + 1]);
    uint count = end > start ? end - start : 0u;
    long offset = (long) row * axis_len * inner;
    long cell = (long) gid * inner;
    for (uint j = tid; j < inner; j += lanes) {
        float acc = seg_identity(mode);
        bool nan = false;
        for (uint p = start; p < end; ++p) {
            seg_step(mode, acc, nan, sload(x, offset + (long) p * inner + j));
        }
        sstore(out, cell + j, seg_finish(mode, acc, nan, count, has_initial, initial));
    }
}

/* Narrow path: inner < 32, so one SIMD group per segment, lanes along the axis
 * and one column at a time. */
kernel void seg_rows(
    device const $T* x, device const long* bounds, device $T* out,
    constant uint& segments, constant uint& axis_len, constant uint& inner,
    constant uint& mode, constant uint& has_initial, constant float& initial,
    uint gid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    uint row = gid / segments;
    uint seg = gid - row * segments;
    long at = (long) row * (segments + 1) + seg;
    uint start = uint(bounds[at]), end = uint(bounds[at + 1]);
    uint count = end > start ? end - start : 0u;
    long offset = (long) row * axis_len * inner;
    long cell = (long) gid * inner;
    for (uint j = 0; j < inner; ++j) {
        float acc = seg_identity(mode);
        bool nan = false;
        for (uint p = start + lane; p < end; p += 32) {
            seg_step(mode, acc, nan, sload(x, offset + (long) p * inner + j));
        }
        float total = lane_reduce(mode, acc);
        bool any_nan = simd_max(nan ? 1.0f : 0.0f) > 0.0f;
        if (lane == 0) {
            sstore(out, cell + j, seg_finish(mode, total, any_nan, count, has_initial, initial));
        }
    }
}

kernel void seg_cols_backward(
    device const $T* x, device const long* bounds, device const $T* grad,
    device const $T* out, device $T* dx,
    constant uint& segments, constant uint& axis_len, constant uint& inner,
    constant uint& mode, constant uint& has_initial, constant float& initial,
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    uint row = gid / segments;
    uint seg = gid - row * segments;
    long at = (long) row * (segments + 1) + seg;
    uint start = uint(bounds[at]), end = uint(bounds[at + 1]);
    uint count = end > start ? end - start : 0u;
    long offset = (long) row * axis_len * inner;
    long cell = (long) gid * inner;
    seg_zero_edges(dx, offset, axis_len, inner, start, end, segments, seg, tid, lanes);
    /* mean divides by the element count, floored at one when initial is set. */
    float divisor = has_initial ? fmax(float(count), 1.0f) : float(count);
    for (uint j = tid; j < inner; j += lanes) {
        float g = sload(grad, cell + j);
        long column = offset + j;
        if (mode == MODE_SUM) {
            for (uint p = start; p < end; ++p) sstore(dx, column + (long) p * inner, g);
        } else if (mode == MODE_MEAN) {
            float d = g / divisor;
            for (uint p = start; p < end; ++p) sstore(dx, column + (long) p * inner, d);
        } else if (mode == MODE_PROD) {
            /* one zero in the segment takes the product of the rest; two or
             * more zero every gradient in the segment. */
            float selected = sload(out, cell + j);
            float rest = has_initial ? initial : 1.0f;
            uint zeros = 0;
            for (uint p = start; p < end; ++p) {
                float v = sload(x, column + (long) p * inner);
                if (v == 0.0f) ++zeros; else rest *= v;
            }
            for (uint p = start; p < end; ++p) {
                float v = sload(x, column + (long) p * inner);
                float d = v == 0.0f ? (zeros == 1 ? g * rest : 0.0f) : g * selected / v;
                sstore(dx, column + (long) p * inner, d);
            }
        } else {
            /* max/min split the gradient evenly across the positions equal to
             * the output. */
            float selected = sload(out, cell + j);
            uint ties = 0;
            for (uint p = start; p < end; ++p) {
                ties += sload(x, column + (long) p * inner) == selected ? 1u : 0u;
            }
            float d = ties ? g / float(ties) : 0.0f;
            for (uint p = start; p < end; ++p) {
                bool hit = sload(x, column + (long) p * inner) == selected;
                sstore(dx, column + (long) p * inner, hit ? d : 0.0f);
            }
        }
    }
}

kernel void seg_rows_backward(
    device const $T* x, device const long* bounds, device const $T* grad,
    device const $T* out, device $T* dx,
    constant uint& segments, constant uint& axis_len, constant uint& inner,
    constant uint& mode, constant uint& has_initial, constant float& initial,
    uint gid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    uint row = gid / segments;
    uint seg = gid - row * segments;
    long at = (long) row * (segments + 1) + seg;
    uint start = uint(bounds[at]), end = uint(bounds[at + 1]);
    uint count = end > start ? end - start : 0u;
    long offset = (long) row * axis_len * inner;
    long cell = (long) gid * inner;
    seg_zero_edges(dx, offset, axis_len, inner, start, end, segments, seg, lane, 32);
    float divisor = has_initial ? fmax(float(count), 1.0f) : float(count);
    for (uint j = 0; j < inner; ++j) {
        float g = sload(grad, cell + j);
        long column = offset + j;
        if (mode == MODE_SUM || mode == MODE_MEAN) {
            float d = mode == MODE_SUM ? g : g / divisor;
            for (uint p = start + lane; p < end; p += 32) {
                sstore(dx, column + (long) p * inner, d);
            }
        } else if (mode == MODE_PROD) {
            float selected = sload(out, cell + j);
            float rest = has_initial ? initial : 1.0f;
            uint mine = 0;
            for (uint p = start + lane; p < end; p += 32) {
                float v = sload(x, column + (long) p * inner);
                if (v == 0.0f) ++mine; else rest *= v;
            }
            uint zeros = uint(simd_sum(float(mine)));
            float product = simd_product(rest);
            for (uint p = start + lane; p < end; p += 32) {
                float v = sload(x, column + (long) p * inner);
                float d = v == 0.0f ? (zeros == 1 ? g * product : 0.0f) : g * selected / v;
                sstore(dx, column + (long) p * inner, d);
            }
        } else {
            float selected = sload(out, cell + j);
            uint mine = 0;
            for (uint p = start + lane; p < end; p += 32) {
                mine += sload(x, column + (long) p * inner) == selected ? 1u : 0u;
            }
            uint ties = uint(simd_sum(float(mine)));
            float d = ties ? g / float(ties) : 0.0f;
            for (uint p = start + lane; p < end; p += 32) {
                bool hit = sload(x, column + (long) p * inner) == selected;
                sstore(dx, column + (long) p * inner, hit ? d : 0.0f);
            }
        }
    }
}
"""

_MIN_GROUPS = 8
"""Fewer threadgroups than this leave the GPU idle; the aten fallback wins."""

_MAX_ROUNDS = 512
"""Elements one thread may walk before the aten reduction is the better split."""

_NARROW = 32
"""``inner`` below this reduces lane-parallel along the axis instead of by column."""

_PATHS = {"kernel": 0, "aten": 0}


@lru_cache(maxsize=4)
def _library(dtype):
    """Compiled on first use, one library per dtype; importing compiles nothing."""
    return torch.mps.compile_shader(
        _SOURCE.replace("$T", _DTYPE_NAMES[dtype])
        .replace("$FLAG_SUM", str(_FLAG_SUM))
        .replace("$FLAG_BAD", str(_FLAG_BAD))
    )


def _lanes(inner):
    """Threads per threadgroup: one per column, in whole SIMD groups, up to 256.

    Interleaved in one batch on the 4096x768 embedding-bag shape, where a
    threadgroup is one segment of eight positions, 32 and 128 threads ranged
    over 0.03-0.10 ms between runs while 256 and 512 held at 0.062-0.065 ms
    against a 0.048-0.070 ms copy of the same bytes; 256 was the faster of the
    two stable settings in both runs.
    """
    if inner < _NARROW:
        return 32
    return min(256, max(32, inner - inner % 32))


def _kernel_fits(rows, segments, axis_len, inner):
    """Whether one threadgroup per segment keeps every lane busy."""
    groups = rows * segments
    if groups < _MIN_GROUPS:
        return False
    lanes = groups * _lanes(inner)
    return rows * axis_len * inner <= lanes * _MAX_ROUNDS


def _kernel_ready(data, rows, segments, axis_len, inner):
    """Layout, dtype and occupancy test shared by the two directions."""
    return (
        data.is_contiguous()
        and data.dtype in _DTYPE_NAMES
        and inner > 0
        and _kernel_fits(rows, segments, axis_len, inner)
    )


def _forward_kernel(data, bounds, out, rows, segments, axis_len, inner, mode, initial):
    library = _library(data.dtype)
    args = (
        data, bounds, out, segments, axis_len, inner,
        _MODE_IDS[mode], 0 if initial is None else 1,
        0.0 if initial is None else float(initial),
    )
    groups = rows * segments
    lanes = _lanes(inner)
    if inner < _NARROW:
        library.seg_rows(*args, threads=groups * 32, group_size=32)
    else:
        library.seg_cols(*args, threads=groups * lanes, group_size=lanes)
    _PATHS["kernel"] += 1


def _backward_kernel(grad, output, data, bounds, rows, segments, axis_len, inner,
                     mode, initial):
    library = _library(data.dtype)
    dx = torch.empty_like(data)
    args = (
        data, bounds, grad, output, dx, segments, axis_len, inner,
        _MODE_IDS[mode], 0 if initial is None else 1,
        0.0 if initial is None else float(initial),
    )
    groups = rows * segments
    lanes = _lanes(inner)
    if inner < _NARROW:
        library.seg_rows_backward(*args, threads=groups * 32, group_size=32)
    else:
        library.seg_cols_backward(*args, threads=groups * lanes, group_size=lanes)
    _PATHS["kernel"] += 1
    return dx


def _segment_reduce(
    data,
    reduce,
    lengths=None,
    indices=None,
    offsets=None,
    axis=0,
    unsafe=False,
    initial=None,
):
    if indices is not None:
        raise RuntimeError(_INDICES)
    mode = _MODES.get(reduce)
    if mode is None:
        raise RuntimeError(_REDUCE.format(reduce))
    if not data.dtype.is_floating_point:
        raise NotImplementedError(_DTYPE.format(data.dtype))

    kind = "lengths" if lengths is not None else "offsets"
    bounds, rows, segments, axis_len, inner, axis, flag = _layout(
        data, lengths, offsets, axis, unsafe
    )
    shape = list(data.shape)
    shape[axis] = segments
    if segments == 0:
        return torch.empty(shape, dtype=data.dtype, device=data.device)
    if _kernel_ready(data, rows, segments, axis_len, inner):
        # The reduction runs before the descriptor verdict is read back, so the
        # one host wait this operator makes covers the kernel instead of
        # preceding it. Clamped bounds keep an invalid descriptor inside the
        # buffer, and the result is dropped when the verdict is a raise.
        out = torch.empty(shape, dtype=data.dtype, device=data.device)
        _forward_kernel(data, bounds, out, rows, segments, axis_len, inner, mode, initial)
        _settle(flag, kind)
        return out
    _PATHS["aten"] += 1
    exact = _settle(flag, kind)

    positions = rows * axis_len
    source = data.reshape(positions, inner)
    target = _positions(bounds, rows, segments, axis_len, exact) if positions else None
    acc = _accumulate(
        source, target, rows, segments, inner, mode, data.dtype, initial, data.device
    )
    if mode == "mean":
        counts = (bounds[:, 1:] - bounds[:, :-1]).to(data.dtype)
        if initial is not None:
            counts.clamp_min_(1)
        acc = acc.reshape(rows, segments, inner) / counts.unsqueeze(-1)
    return acc.reshape(shape)


def _segment_reduce_backward(
    grad, output, data, reduce, lengths=None, offsets=None, axis=0, initial=None
):
    mode = _MODES.get(reduce)
    if mode is None:
        raise RuntimeError(_REDUCE.format(reduce))
    if not data.dtype.is_floating_point:
        raise NotImplementedError(_DTYPE.format(data.dtype))
    if lengths is None and offsets is None:
        raise RuntimeError(_MISSING)
    if grad.dtype != data.dtype:
        raise RuntimeError(f"expected scalar type {data.dtype} but found {grad.dtype}")
    if output.dtype != data.dtype:
        raise RuntimeError(f"expected scalar type {data.dtype} but found {output.dtype}")

    bounds, rows, segments, axis_len, inner, axis, flag = _layout(
        data, lengths, offsets, axis, True, strict=False
    )
    exact = _settle(flag, "lengths")
    if segments == 0 or axis_len == 0 or rows == 0:
        return torch.zeros_like(data)
    if _kernel_ready(data, rows, segments, axis_len, inner):
        return _backward_kernel(
            grad.contiguous(), output.contiguous(), data, bounds,
            rows, segments, axis_len, inner, mode, initial,
        )
    _PATHS["aten"] += 1

    positions = rows * axis_len
    target = _positions(bounds, rows, segments, axis_len, exact)
    tail = rows * segments
    if exact:
        lookup = target
        upstream = grad.reshape(tail, inner).index_select(0, target)
    else:
        lookup = target.clamp_max(tail - 1)
        upstream = grad.reshape(tail, inner).index_select(0, lookup)
        upstream.masked_fill_((target == tail).unsqueeze(1), 0)
    elements = data.reshape(positions, inner)

    if mode == "sum":
        result = upstream
    elif mode == "mean":
        counts = (bounds[:, 1:] - bounds[:, :-1]).to(data.dtype)
        divisor = torch.cat(
            (counts.clamp_min(1).reshape(-1), counts.new_ones(1)), 0
        ).index_select(0, lookup)
        result = upstream / divisor.unsqueeze(1)
    elif mode == "amax" or mode == "amin":
        selected = output.reshape(tail, inner).index_select(0, lookup)
        mask = (elements == selected).to(data.dtype)
        ties = torch.zeros((tail + 1, inner), dtype=data.dtype, device=data.device)
        ties.index_add_(0, target, mask)
        result = upstream * mask / ties.clamp_min(1).index_select(0, lookup)
    else:
        selected = output.reshape(tail, inner).index_select(0, lookup)
        zero = elements == 0
        zeros = torch.zeros((tail + 1, inner), dtype=data.dtype, device=data.device)
        zeros.index_add_(0, target, zero.to(data.dtype))
        nonzero = torch.where(zero, torch.ones_like(elements), elements)
        rest = torch.full(
            (tail + 1, inner),
            1.0 if initial is None else float(initial),
            dtype=data.dtype,
            device=data.device,
        )
        rest.index_reduce_(0, target, nonzero, "prod")
        isolated_zero = zeros.index_select(0, lookup) == 1
        result = torch.where(
            zero,
            torch.where(
                isolated_zero,
                upstream * rest.index_select(0, lookup),
                torch.zeros_like(elements),
            ),
            upstream * selected / nonzero,
        )
    return result.reshape(data.shape)


OVERLAYS = (
    ("segment_reduce", _segment_reduce),
    ("_segment_reduce_backward", _segment_reduce_backward),
)
