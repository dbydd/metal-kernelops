"""MPS overlays for ``aten::mode`` and ``aten::mode.values``.

Both overloads are missing on the MPS dispatch key in torch 2.14.1: ``mode`` has
a CPU/CUDA/XPU-only kernel and ``mode.values`` is a composite that calls the same
``mode_out`` and therefore rejects an MPS tensor with "mode only supports CPU,
CUDA and XPU device type".  Neither has a Metal kernel to replace.

Two paths, chosen by ``_plan`` from a ``(rows, row length, dtype)`` rule measured
on this machine, compute the same result:

* a Metal kernel, one dispatch for the whole block, covering row lengths up to
  ``n = 256``.  Two kernel forms split the work, and which one a block takes is
  the measured part of the rule (``_plan``):
  - ``reg``: one thread per row, the row staged once into a fixed-size
    per-thread array that the count scan then reads without threadgroup memory
    and without a barrier.  It needs one thread per row, so it is the only form
    that scales to tall blocks; it takes blocks of 2048 or more rows up to 64
    elements.  A second instantiation with a 16-slot array takes every row of 16
    elements or less, where the 64-slot one would walk slots that cannot hold
    data;
  - ``lane``: one threadgroup of 64 lanes per row, the row staged in threadgroup
    memory and the candidate positions split across the lanes, which cuts the
    count scan's latency from ``n**2`` to ``n**2 / 64`` and costs a prefix sum
    and a tournament of barriers per row.  It takes rows of 65 to 256 elements,
    and rows of 17 to 64 in blocks too small for the thread count to be the
    limit, which is where the extra parallelism wins over the extra barriers.
    Its cost grows with ``rows * n**2`` and the sort pipeline's with
    ``rows * n``, so it hands the widest blocks back: above ``_LANE_MAX_WORK``
    the sort path is at least as fast and takes over;
* the MPS aten sort pipeline described below for rows longer than 256 elements
  and for the widest blocks, where a row's ``n**2`` count scan loses to
  ``n log n``.

The sort path reproduces the CPU kernel's semantics in one sort per row and no
Python loop over elements or rows:

* take the row order the CPU kernel's sort produces (stable ascending; see
  ``_sorted`` for the NaN case),
* mark the positions where the sorted value changes (``!=``, so every NaN is its
  own run exactly as the CPU kernel's ``elements[i].first != elements[i+1].first``
  treats it),
* ``position - run_start[position]`` grows by one along a run, so its first
  maximum is the CPU kernel's winning run: strictly greater frequency scanning
  the row's segments in order and, inside a segment, ascending values; ties to
  the earliest segment and the smallest value.  The winner is reported at the
  last element of its run, which a stable sort leaves holding the value's last
  occurrence.  A NaN is its own run of length one (``NaN != NaN`` starts a new
  run), so it can win a tie against one-occurrence numbers only from an earlier
  segment.

The kernels implement the same scan without sorting: segment ``s`` holds the
elements after the ``s``-th NaN and before the ``(s+1)-th`` -- the segment id of
each element is the prefix sum of NaNs before it -- and a run is a maximal
equal-value group inside one segment.  The winner is the longest run, ties to
the earliest segment and then the smallest value, which is the order the CPU
kernel scans; it is reported at the run's last element in original order, i.e.
the last occurrence of the value in its segment.  A NaN is a run of length one
because ``NaN == NaN`` is false.

Both kernel forms take the same short cut on the rows that dominate real inputs:
when the row holds no NaN it is a single segment, so the count loop drops the
segment comparison and reads only the values.  That branch is per row and
uniform across the threadgroup, and it is worth about 1.2x-1.6x on the shapes
measured here.

Tie detail, measured rather than assumed: the CPU kernel sorts ``(value, index)``
pairs with an unstable ``std::sort`` whose comparator compares values only, so
the index it reports for a value that occurs more than once is an artefact of
the standard library's introsort once the row is long enough for the partition
pass to reorder equal keys.  ``benchmarks/verify_mode.py`` measures where the two
agree; what the overlay guarantees on its own is: the value with the highest
frequency, ties to the earliest segment and then the smallest value in it,
reported at the last occurrence of that value inside its segment (NaN rows: see
``_sorted``), and an index that always points at the value it reports.  CPU
parity of the value additionally holds on every row
without NaN and on every NaN row whose length keeps it inside the CPU sort's
reproducible regime; parity of the reported index holds on the same rows and
whenever the winning value occurs once (every row with distinct entries, hence
any continuous-valued row).  Above row length 23 a NaN row's CPU value and index
are not reproducible at all, so the report lists those divergences instead of
counting them as passes.

Import registers nothing and compiles nothing; the shader library is built on the
first call that takes the kernel path.  ``accelerator.py`` is the only installer.
"""

from functools import lru_cache

import torch

__all__ = ["OVERLAYS"]


_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// Value order of the CPU kernel's sort: ascending with every NaN above every
// number; -0.0 == 0.0 holds, as it does for that kernel's run test.
static inline bool is_nan(float v) { return isnan(v); }
static inline bool is_nan(half v) { return isnan(v); }
static inline bool is_nan(bfloat v) { return isnan(float(v)); }
static inline bool is_nan(char v) { return false; }
static inline bool is_nan(uchar v) { return false; }
static inline bool is_nan(short v) { return false; }
static inline bool is_nan(int v) { return false; }
static inline bool is_nan(long v) { return false; }

static inline bool lt(float a, float b) {
    if (isnan(a)) return false;
    if (isnan(b)) return true;
    return a < b;
}
static inline bool lt(half a, half b) {
    if (isnan(a)) return false;
    if (isnan(b)) return true;
    return a < b;
}
static inline bool lt(bfloat a, bfloat b) { return lt(float(a), float(b)); }
static inline bool lt(char a, char b) { return a < b; }
static inline bool lt(uchar a, uchar b) { return a < b; }
static inline bool lt(short a, short b) { return a < b; }
static inline bool lt(int a, int b) { return a < b; }
static inline bool lt(long a, long b) { return a < b; }

// Whether candidate A beats candidate B: the longest run first, then the
// earliest segment, then the smallest value, then -- for one value counted
// more than once -- the last occurrence in the row.  A strict total order
// over candidates, because the index is unique, so the cross-lane tournament
// in the lane kernel finds the same winner whatever the reduction order.
template <typename T>
static inline bool beats(int ca, int sa, T va, int ia, int cb, int sb, T vb, int ib) {
    if (ca != cb) return ca > cb;
    if (sa != sb) return sa < sb;
    if (lt(va, vb)) return true;
    if (lt(vb, va)) return false;
    return ia > ib;
}

// Register-resident row form: one thread per row.  The entire row and its
// segment prefix are loaded once into fixed-size register arrays; the O(n**2)
// count scan then executes purely from registers with zero threadgroup memory
// and zero barriers.  This achieves full core occupancy (1024+ threads/core).
template <typename T, uint MAXN>
static inline void mode_reg_impl(device const T* base, uint n, device T* values,
                                 device long* indices, uint r) {
    T v[MAXN];
    ushort s[MAXN];
    ushort prefix = 0;
    for (uint j = 0; j < MAXN; ++j) {
        if (j < n) {
            v[j] = base[j];
            s[j] = prefix;
            prefix += is_nan(v[j]) ? 1 : 0;
        }
    }
    int bc = -1, bs = 0, bi = 0;
    T bv = v[0];
    // Without a NaN the row is one segment, so the count drops the segment
    // test, which is the load in the inner loop of every other row here.
    bool plain = prefix == 0;
    for (uint i = 0; i < n; ++i) {
        int si = (int)s[i];
        T vi = v[i];
        int count = is_nan(vi) ? 1 : 0;
        if (!is_nan(vi) && plain) {
            for (uint j = 0; j < MAXN; ++j) {
                if (j < n && v[j] == vi) ++count;
            }
        } else if (!is_nan(vi)) {
            for (uint j = 0; j < MAXN; ++j) {
                if (j < n && (int)s[j] == si && v[j] == vi) ++count;
            }
        }
        if (beats(count, si, vi, (int)i, bc, bs, bv, bi)) {
            bc = count;
            bs = si;
            bi = (int)i;
            bv = vi;
        }
    }
    values[r] = bv;
    indices[r] = (long)bi;
}

// Lane-parallel form: one threadgroup of LANES lanes per row.  The row is staged
// into threadgroup memory and the candidate positions are split across lanes
// (stride LANES).  A parallel prefix sum builds the segment ids, each lane
// computes its local winner, and a binary tournament over shared memory reduces
// to the global winner.
template <typename T, uint LANES, uint MAXN>
static inline void mode_lane_impl(device const T* rows, device T* values, device long* indices,
                                  uint count, uint n, uint gid, uint tid,
                                  threadgroup T* tile, threadgroup ushort* seg,
                                  threadgroup ushort* blocks, threadgroup uint* nan_total,
                                  threadgroup int* rc, threadgroup int* rs,
                                  threadgroup T* rv, threadgroup int* ri) {
    if (gid >= count) return;
    device const T* src = rows + (ulong)gid * n;
    for (uint j = tid; j < n; j += LANES) tile[j] = src[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint per = (n + LANES - 1) / LANES;
    uint lo = tid * per;
    uint hi = min(lo + per, n);
    ushort run = 0;
    for (uint j = lo; j < hi; ++j) {
        seg[j] = run;
        run += is_nan(tile[j]) ? 1 : 0;
    }
    blocks[tid] = run;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        ushort acc = 0;
        for (uint t = 0; t < LANES; ++t) {
            ushort val = blocks[t];
            blocks[t] = acc;
            acc += val;
        }
        nan_total[0] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ushort offset = blocks[tid];
    for (uint j = lo; j < hi; ++j) seg[j] += offset;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int bc = -1, bs = 0, bi = 0;
    T bv = tile[0];
    // Without a NaN the row is one segment, so the count drops the segment
    // test, which is the load in the inner loop of every other row here.
    bool plain = nan_total[0] == 0;
    for (uint i = tid; i < n; i += LANES) {
        int si = (int)seg[i];
        T vi = tile[i];
        int count = is_nan(vi) ? 1 : 0;
        if (!is_nan(vi) && plain) {
            for (uint j = 0; j < n; ++j) {
                if (tile[j] == vi) ++count;
            }
        } else if (!is_nan(vi)) {
            for (uint j = 0; j < n; ++j) {
                if ((int)seg[j] == si && tile[j] == vi) ++count;
            }
        }
        if (beats(count, si, vi, (int)i, bc, bs, bv, bi)) {
            bc = count;
            bs = si;
            bi = (int)i;
            bv = vi;
        }
    }
    rc[tid] = bc;
    rs[tid] = bs;
    rv[tid] = bv;
    ri[tid] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = LANES / 2; stride >= 1; stride >>= 1) {
        if (tid < stride) {
            uint k = tid + stride;
            if (beats(rc[k], rs[k], rv[k], ri[k], rc[tid], rs[tid], rv[tid], ri[tid])) {
                rc[tid] = rc[k];
                rs[tid] = rs[k];
                rv[tid] = rv[k];
                ri[tid] = ri[k];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        values[gid] = rv[0];
        indices[gid] = (long)ri[0];
    }
}

#define REG(name, type, maxn)                                                                 \
kernel void name(device const type* rows, device type* values, device long* indices,          \
                 constant uint& count, constant uint& n, uint r [[thread_position_in_grid]]) { \
    if (r < count) mode_reg_impl<type, maxn>(rows + (ulong)r * n, n, values, indices, r);     \
}

#define LANES(name, type, lanes, maxn)                                                        \
kernel void name(device const type* rows, device type* values, device long* indices,          \
                 constant uint& count, constant uint& n,                                      \
                 uint gid [[threadgroup_position_in_grid]],                                   \
                 uint tid [[thread_index_in_threadgroup]]) {                                  \
    threadgroup type tile[maxn];                                                              \
    threadgroup ushort seg[maxn];                                                             \
    threadgroup ushort blocks[lanes];                                                         \
    threadgroup uint nan_total[1];                                                            \
    threadgroup int rc[lanes];                                                                \
    threadgroup int rs[lanes];                                                                \
    threadgroup type rv[lanes];                                                               \
    threadgroup int ri[lanes];                                                                \
    mode_lane_impl<type, lanes, maxn>(rows, values, indices, count, n, gid, tid, tile,        \
                                      seg, blocks, nan_total, rc, rs, rv, ri);                \
}

REG(mode_reg_f32, float, 64)
REG(mode_reg_f16, half, 64)
REG(mode_reg_bf16, bfloat, 64)
REG(mode_reg_i8, char, 64)
REG(mode_reg_u8, uchar, 64)
REG(mode_reg_i16, short, 64)
REG(mode_reg_i32, int, 64)
REG(mode_reg_i64, long, 64)

// The same form with a 16-slot row.  A block of rows this short would otherwise
// walk 64 slots per candidate and hold 64 values plus 64 segment ids in
// registers per thread, which costs both the wasted compares and the occupancy
// the tall narrow blocks need.
REG(mode_reg16_f32, float, 16)
REG(mode_reg16_f16, half, 16)
REG(mode_reg16_bf16, bfloat, 16)
REG(mode_reg16_i8, char, 16)
REG(mode_reg16_u8, uchar, 16)
REG(mode_reg16_i16, short, 16)
REG(mode_reg16_i32, int, 16)
REG(mode_reg16_i64, long, 16)

LANES(mode_lane_f32, float, 64, 256)
LANES(mode_lane_f16, half, 64, 256)
LANES(mode_lane_bf16, bfloat, 64, 256)
LANES(mode_lane_i8, char, 64, 256)
LANES(mode_lane_u8, uchar, 64, 256)
LANES(mode_lane_i16, short, 64, 256)
LANES(mode_lane_i32, int, 64, 256)
LANES(mode_lane_i64, long, 64, 256)
"""

# Row lengths each form covers, and the row count where the two kernel forms
# cross over.  All of it was measured on this machine by timing the forms
# against each other and against the sort path at fixed (rows, n), one dispatch
# each, minimum of interleaved rounds so a shared box's load cancels.  The
# numbers behind each boundary:
#
#   n <= 16                      reg16  at this length the 64-slot form's extra
#                                       compares and register pressure buy nothing
#   17 <= n <= 64, rows <  2048   lane   1000x64: 43 us lane vs 86 us reg
#   17 <= n <= 64, rows >= 2048   reg    4096x64: 92 us reg vs 140 us lane;
#                                        16384x64: 202 us vs 270 us, because one
#                                        thread per row stops paying for the lane
#                                        form's barriers once the rows fill the GPU
#   65 <= n <= 256                lane   256x256: 275 us vs 896 us sort path;
#                                        4096x256: 1651 us vs 2341 us
#   n > 256                       sort   a row's n**2 scan loses to n log n
_REG16_MAX_N = 16
_REG_MAX_N = 64
_LANE_MAX_N = 256
_REG_GROUP = 64
_LANE_GROUP = 64
# For 17 <= n <= 64, `lane` saturates the GPU better when rows < 2048, while
# `reg` wins on larger row counts where occupancy dominates.
_LANE_ROW_THRESHOLD = 2048
# ``count * n * n`` above which the lane form stops paying: its cost grows with
# that product, the sort pipeline's with ``count * n``, and the two met between
# 2e7 and 3e7 on this machine.  Sort time over lane time measured at the
# boundary: 1.15-2.2 around 1.7e7, 0.6-1.1 around 3.4e7, 0.4-0.9 around 6.7e7.
# Capping at the break-even means the kernel is not chosen for a block where it
# can lose, so blocks of more than about 370 rows of 256 elements keep the sort
# path they had before the kernel existed.
_LANE_MAX_WORK = 24_000_000

_REG16_NAMES = {
    torch.float32: "mode_reg16_f32",
    torch.float16: "mode_reg16_f16",
    torch.bfloat16: "mode_reg16_bf16",
    torch.int8: "mode_reg16_i8",
    torch.uint8: "mode_reg16_u8",
    torch.int16: "mode_reg16_i16",
    torch.int32: "mode_reg16_i32",
    torch.int64: "mode_reg16_i64",
    torch.bool: "mode_reg16_u8",
}

_REG_NAMES = {
    torch.float32: "mode_reg_f32",
    torch.float16: "mode_reg_f16",
    torch.bfloat16: "mode_reg_bf16",
    torch.int8: "mode_reg_i8",
    torch.uint8: "mode_reg_u8",
    torch.int16: "mode_reg_i16",
    torch.int32: "mode_reg_i32",
    torch.int64: "mode_reg_i64",
    torch.bool: "mode_reg_u8",
}

_LANE_NAMES = {
    torch.float32: "mode_lane_f32",
    torch.float16: "mode_lane_f16",
    torch.bfloat16: "mode_lane_bf16",
    torch.int8: "mode_lane_i8",
    torch.uint8: "mode_lane_u8",
    torch.int16: "mode_lane_i16",
    torch.int32: "mode_lane_i32",
    torch.int64: "mode_lane_i64",
    torch.bool: "mode_lane_u8",
}


@lru_cache(maxsize=1)
def _shader():
    return torch.mps.compile_shader(_SOURCE)


@lru_cache(maxsize=16)
def _reg_kernel(dtype):
    name = _REG_NAMES.get(dtype)
    return getattr(_shader(), name) if name is not None else None


@lru_cache(maxsize=16)
def _reg16_kernel(dtype):
    name = _REG16_NAMES.get(dtype)
    return getattr(_shader(), name) if name is not None else None


@lru_cache(maxsize=16)
def _lane_kernel(dtype):
    name = _LANE_NAMES.get(dtype)
    return getattr(_shader(), name) if name is not None else None


def _plan(rows):
    """``(kernel, threads, group_size)`` for one ``(rows, n)`` block, or None.

    No host read of the data and no per-row Python: the choice is a function of
    the shape, and one dispatch covers the whole block.  The row-length and row
    count boundaries are the measured ones documented at the constants above.
    """
    if not rows.is_contiguous():
        return None
    count, n = rows.shape
    if n <= _REG16_MAX_N:
        kernel = _reg16_kernel(rows.dtype)
        if kernel is not None:
            return kernel, count, _REG_GROUP
    elif n <= _REG_MAX_N:
        if count < _LANE_ROW_THRESHOLD:
            kernel = _lane_kernel(rows.dtype)
            if kernel is not None:
                return kernel, count * _LANE_GROUP, _LANE_GROUP
        kernel = _reg_kernel(rows.dtype)
        if kernel is not None:
            return kernel, count, _REG_GROUP
    elif n <= _LANE_MAX_N:
        if count * n * n <= _LANE_MAX_WORK:
            kernel = _lane_kernel(rows.dtype)
            if kernel is not None:
                return kernel, count * _LANE_GROUP, _LANE_GROUP
    return None


def _resize(out, shape):
    if tuple(out.shape) != tuple(shape):
        out.resize_(shape)


_TYPE_NAMES = {
    torch.float32: "Float",
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.int8: "Char",
    torch.uint8: "Byte",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.bool: "Bool",
    torch.complex64: "ComplexFloat",
}


def _type_name(dtype):
    return _TYPE_NAMES.get(dtype, str(dtype).replace("torch.", ""))


def _check_outputs(self, values, indices):
    if values.device != self.device:
        raise RuntimeError(
            f"expected device '{self.device}' but got '{values.device}' for values output"
        )
    if indices.device != self.device:
        raise RuntimeError(
            f"expected device '{self.device}' but got '{indices.device}' for indices output"
        )
    if values.dtype != self.dtype:
        raise RuntimeError(
            f"expected scalar type '{_type_name(self.dtype)}' but got "
            f"'{_type_name(values.dtype)}' for values output"
        )
    if indices.dtype != torch.long:
        raise RuntimeError(
            f"expected scalar type 'Long' but got '{_type_name(indices.dtype)}' "
            "for indices output"
        )


def _wrap_dim(self, dim):
    ndim = self.dim()
    if ndim == 0:
        if dim not in (0, -1):
            raise IndexError(
                "Dimension out of range (expected to be in range of [-1, 0], "
                f"but got {dim})"
            )
        return 0
    if not -ndim <= dim < ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-ndim}, "
            f"{ndim - 1}], but got {dim})"
        )
    return dim + ndim if dim < 0 else dim


def _mode_rows(self, dim):
    """``(rows, n)`` view of ``self`` with ``dim`` moved last."""
    n = self.shape[dim]
    return self.movedim(dim, -1).reshape(-1, n)


def _sorted(rows):
    """Row values in the order the CPU kernel's sort produces, plus that order.

    Without NaN, ascending stable order.  With NaN the CPU comparator is not a
    strict weak order -- NaN compares neither less nor greater than anything --
    so what ``std::sort`` does to the row is unspecified; on the machine this was
    measured on it is reproducible only up to row length 23, where the sort is
    insertion sort and leaves each NaN in place: the row reads
    ``[sorted segment] NaN [sorted segment] NaN ...``.  Longer rows go through
    introsort's partition pass, which does move elements across NaNs, and from
    there neither the NaN placement nor the reported value or index is specified
    or reproducible.  The reproducible regime is what this function computes,
    unconditionally, on the GPU: a stable sort on the value followed by a stable
    sort on a segment id that is ``2 * (NaNs before the element)`` for numbers
    and ``2 * (NaNs before) + 1`` for the NaN itself.  There is no
    ``isnan(...).any()`` host read choosing between the two orders: a
    device-to-CPU sync costs about as much as the whole mode computation here, so
    float rows always pay the second argsort and integer rows never do.
    """
    if rows.is_floating_point():
        nan = torch.isnan(rows)
        seen = torch.cumsum(nan.to(torch.int32), -1)
        segment = torch.where(nan, 2 * seen - 1, 2 * seen)
        by_value = torch.argsort(rows, dim=-1, stable=True)
        by_segment = torch.argsort(segment.gather(-1, by_value), dim=-1, stable=True)
        order = by_value.gather(-1, by_segment)
        return rows.gather(-1, order), order
    return torch.sort(rows, dim=-1, stable=True)


def _mode_raw(rows):
    """CPU ``mode_kernel_impl`` semantics for one ``(rows, n)`` float/int block.

    ``i - run_start[i]`` orders positions by the length of the run that ends at
    ``i``, so its first maximum is the CPU kernel's winner: strictly greater
    frequency scanning ascending values, ties to the smallest value, at the last
    element of its run.  (The constant ``+ 1`` of the length is left out because
    it does not move the maximum.)
    """
    sorted_values, order = _sorted(rows)
    n = rows.shape[-1]
    pos = torch.arange(n, dtype=torch.int32, device=rows.device)
    changed = sorted_values[..., 1:] != sorted_values[..., :-1]
    starts_at = torch.where(changed, pos[1:], 0)
    start = torch.constant_pad_nd(torch.cummax(starts_at, -1).values, (1, 0), 0)
    best = (pos - start).argmax(-1, keepdim=True)
    return sorted_values.gather(-1, best), order.gather(-1, best)


def _mode_block(rows):
    """``(values, indices)`` for one ``(rows, n)`` block, shaped ``(rows,)``."""
    plan = _plan(rows)
    if plan is None:
        values, indices = _mode_raw(rows)
        return values.squeeze(-1), indices.squeeze(-1)
    kernel, threads, group = plan
    count, n = rows.shape
    values = torch.empty(count, dtype=rows.dtype, device=rows.device)
    indices = torch.empty(count, dtype=torch.long, device=rows.device)
    kernel(rows, values, indices, count, n, threads=threads, group_size=group)
    return values, indices


def _mode_result(self, dim, keepdim):
    """``(values, indices)`` for ``self``, freshly allocated."""
    if self.is_complex():
        raise NotImplementedError(
            f'"mode_mps" not implemented for \'{_type_name(self.dtype)}\''
        )
    dim = _wrap_dim(self, dim)

    if self.numel() == 0:
        if self.shape[dim] == 0:
            raise IndexError(
                f"mode(): Expected reduction dim {dim} to have non-zero size."
            )
        sizes = list(self.shape)
        if keepdim:
            sizes[dim] = 1
        else:
            del sizes[dim]
        return (
            torch.empty(sizes, dtype=self.dtype, device=self.device),
            torch.empty(sizes, dtype=torch.long, device=self.device),
        )

    if self.dim() == 0:
        return self.clone(), torch.zeros((), dtype=torch.long, device=self.device)

    out_shape = list(self.shape)
    if keepdim:
        out_shape[dim] = 1
    else:
        del out_shape[dim]
    mode_values, mode_indices = _mode_block(_mode_rows(self, dim))
    return mode_values.reshape(out_shape), mode_indices.reshape(out_shape)


def mode_mps(self, dim=-1, keepdim=False):
    return _mode_result(self, dim, keepdim)


def mode_values_mps(self, dim=-1, keepdim=False, values=None, indices=None):
    _check_outputs(self, values, indices)
    result = _mode_result(self, dim, keepdim)
    _resize(values, result[0].shape)
    _resize(indices, result[1].shape)
    values.copy_(result[0])
    indices.copy_(result[1])
    return values, indices


OVERLAYS = (
    ("mode", mode_mps),
    ("mode.values", mode_values_mps),
)
