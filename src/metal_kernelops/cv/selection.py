"""NSA key-block selection: the top-k blocks of a score row, with a total order.

What this is
------------
Native Sparse Attention scores every candidate key block of a query against the
query's head group, sums that into one importance value per block, and keeps the
``k`` blocks with the largest value.  ``fla/ops/nsa/parallel.py:
parallel_nsa_kernel_topk`` is that step upstream: it accumulates ``b_i = sum
over the group's heads of exp(score - lse)``, merges pairs of block chunks with
a bitonic network, and stores the first ``k`` indices per query position.  The
bitonic merge has no defined order among equal scores, so two runs may select
different blocks from a tie; here the tie is decided.

This module implements the selection primitive alone.  Building the scores is
the caller's job: it needs the query, the compressed keys, the softmax
denominator, and NSA's rule that block 0 and the last two blocks are always
candidates.  What lands here is a float tensor with one row per query head and
one column per candidate block.

Contract
--------
``k_select(scores, k, dim=-1, sorted=True)`` returns int64 indices shaped like
``scores`` with ``k`` in place of the selected axis.  Selection is a total,
deterministic order:

1. larger score first;
2. equal scores: smaller index first;
3. NaN scores last, among themselves by smaller index first.

``sorted=True`` returns that order.  ``sorted=False`` returns the same set in
ascending index order, which is the order a block mask or a CSR row wants and
is also deterministic, unlike ``torch.topk(sorted=False)`` whose order is
whatever the kernel happened to produce.

``k_select_values`` returns the selected scores instead of their positions.
Selection itself carries no gradient — the indices are integers and
``k_select`` builds no graph — while ``k_select_values`` is ``torch.gather``
over the selection, so the gradient reaches exactly the selected entries and is
zero elsewhere.  Patch the block scores through it and the rest of NSA trains
as usual.

Devices and dtypes
------------------
float32, float16 and bfloat16 on MPS go through one Metal kernel, in one of three
forms, provided ``k <= 256``.  The half types are widened to float32 first,
which is exact, so the order and the ties are the ones the original values have.
Any other device, any other floating dtype, or ``k > 256`` takes the composed
path: two stable argsorts that implement the same order in ordinary aten ops.
That path is the reference used for testing, and it is what runs on CPU.

The forms trade per-element work against what they keep per row:

* small ``k``: a threadgroup per row, each lane holding a top-k window in
  registers, the windows merged through threadgroup memory;
* small ``k`` with many rows: the same window, one thread per row, no merge;
* ``k >= 24``: one simdgroup per row and 32-bit order keys — the row is turned
  into keys whose unsigned order is the contract's order, the k-th key is found
  by histogramming eight bits at a time, and the winners are placed by rank.  It
  keeps nothing per-k in registers, which is what lets it carry k into the
  hundreds where the window forms spill.

The cutoffs between them are measured, not derived: the k-limit probe in
``measured/k_select.json`` reports the crossover of every form against the sort
path at both a few-row and a many-row shape, and the probe is what the two
constants in this module are set from.

Layout: the selected axis may be any axis.  The kernel wants it last, so any
other axis costs one ``movedim`` copy, and the kernel needs a contiguous float32
buffer, so a non-contiguous input or a half input costs one conversion copy.
Beyond its output and those conversions the kernel path allocates nothing; the
sort path allocates the masks and permutations it sorts.  No module import
compiles a shader or allocates a tensor.
"""

import operator
from functools import lru_cache

import torch

__all__ = ["k_select", "k_select_values", "reference_k_select"]

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

constant uint selected = $K;
constant uint max_lanes = $LANES;

// The whole contract is this comparison: score descending, index ascending,
// NaN after every non-NaN value.
static inline bool takes_priority(float a, uint ia, float b, uint ib) {
    bool na = isnan(a), nb = isnan(b);
    if (na != nb) return nb;
    if (na) return ia < ib;
    if (a != b) return a > b;
    return ia < ib;
}

// One element against the current window.  An element that does not beat the
// current k-th best fails the two-instruction test above, so on a random row
// the unrolled shift almost never runs.
#define SCAN_STEP(v, i)                                                    \
    if (takes_priority(v, i, value[selected - 1], index[selected - 1])) {  \
        uint pos = selected - 1;                                           \
        bool placed = false;                                               \
        _Pragma("unroll")                                                  \
        for (uint j = 0; j < selected; ++j) {                              \
            bool hit = takes_priority(v, i, value[j], index[j]);           \
            if (hit && !placed) { pos = j; placed = true; }                \
        }                                                                  \
        _Pragma("unroll")                                                  \
        for (int j = (int)selected - 1; j >= 1; --j) {                     \
            if ((uint)j >= pos) {                                          \
                value[j] = ((uint)j == pos) ? v : value[j - 1];            \
                index[j] = ((uint)j == pos) ? i : index[j - 1];            \
            }                                                              \
        }                                                                  \
        if (pos == 0) { value[0] = v; index[0] = i; }                      \
    }

// A NaN sentinel at the largest index is beaten by every real candidate,
// including a NaN at a lower index, so the window fills from the left.
#define WINDOW_INIT                                                        \
    _Pragma("unroll")                                                      \
    for (uint j = 0; j < selected; ++j) {                                  \
        value[j] = NAN;                                                    \
        index[j] = 0xFFFFFFFFu;                                            \
    }

#if $ASCENDING
// sorted=False: the same set by ascending index, insertion sort over k.
#define Emit(dst)                                                          \
    for (uint t = 1; t < selected; ++t) {                                  \
        uint key = index[t];                                               \
        uint j = t;                                                        \
        while (j > 0 && index[j - 1] > key) { index[j] = index[j - 1]; --j; } \
        index[j] = key;                                                    \
    }                                                                      \
    for (uint j = 0; j < selected; ++j) dst[j] = (long)index[j];
#else
#define Emit(dst)                                                          \
    for (uint j = 0; j < selected; ++j) dst[j] = (long)index[j];
#endif

// One thread per row: no shared state, no barrier, and neighbouring threads
// read separate rows, so every thread's loads are its own cache lines.  At
// many rows that spreads the traffic instead of concentrating it, which is why
// this form scales where the threadgroup form stops scaling.
kernel void select_topk_tall(device const float* scores, device long* out,
                             constant uint& n, constant uint& rows,
                             uint row [[thread_position_in_grid]]) {
    float value[selected];
    uint index[selected];
    if (row >= rows) return;
    device const float* base = scores + (ulong)row * n;
    WINDOW_INIT
    for (uint i = 0; i < n; ++i) {
        float v = base[i];
        SCAN_STEP(v, i)
    }
    device long* dst = out + (ulong)row * selected;
    Emit(dst)
}

// One threadgroup per row: each lane walks every ``lanes``-th element, so a
// lane's loads coalesce with its neighbours' and the group keeps that many
// loads in flight.  A lane keeps the window for the elements it walked in
// registers; the windows then merge pairwise in shared memory down to one.
#define MERGE_KERNEL(name, lanes)                                          \
kernel void name(device const float* scores, device long* out,             \
                 constant uint& n, constant uint& rows,                    \
                 uint row [[threadgroup_position_in_grid]],                \
                 uint lane [[thread_index_in_threadgroup]]) {               \
    threadgroup float share_v[max_lanes * selected];                       \
    threadgroup uint share_i[max_lanes * selected];                        \
    float value[selected];                                                 \
    uint index[selected];                                                  \
    if (row >= rows) return;                                               \
    device const float* base = scores + (ulong)row * n;                    \
    WINDOW_INIT                                                            \
    for (uint i = lane; i < n; i += (lanes)) {                             \
        float v = base[i];                                                 \
        SCAN_STEP(v, i)                                                    \
    }                                                                      \
    _Pragma("unroll")                                                      \
    for (uint j = 0; j < selected; ++j) {                                  \
        share_v[lane * selected + j] = value[j];                           \
        share_i[lane * selected + j] = index[j];                           \
    }                                                                      \
    threadgroup_barrier(mem_flags::mem_threadgroup);                       \
    for (uint stride = (lanes) >> 1; stride >= 1; stride >>= 1) {          \
        if (lane < stride) {                                               \
            uint a = lane * selected, b = (lane + stride) * selected;      \
            float mv[selected];                                            \
            uint mi[selected];                                             \
            uint ia = 0, ib = 0;                                           \
            _Pragma("unroll")                                              \
            for (uint j = 0; j < selected; ++j) {                          \
                bool take_a = ia < selected && (ib >= selected ||          \
                    takes_priority(share_v[a + ia], share_i[a + ia],       \
                                   share_v[b + ib], share_i[b + ib]));     \
                if (take_a) {                                              \
                    mv[j] = share_v[a + ia];                               \
                    mi[j] = share_i[a + ia];                               \
                    ++ia;                                                  \
                } else {                                                   \
                    mv[j] = share_v[b + ib];                               \
                    mi[j] = share_i[b + ib];                               \
                    ++ib;                                                  \
                }                                                          \
            }                                                              \
            _Pragma("unroll")                                              \
            for (uint j = 0; j < selected; ++j) {                          \
                share_v[a + j] = mv[j];                                    \
                share_i[a + j] = mi[j];                                    \
            }                                                              \
        }                                                                  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                   \
    }                                                                      \
    if (lane == 0) {                                                       \
        _Pragma("unroll")                                                  \
        for (uint j = 0; j < selected; ++j) index[j] = share_i[j];         \
        device long* dst = out + (ulong)row * selected;                    \
        Emit(dst)                                                          \
    }                                                                      \
}

MERGE_KERNEL(select_topk_l16, 16)
MERGE_KERNEL(select_topk_l32, 32)
MERGE_KERNEL(select_topk_l64, 64)

// The same contract as a single unsigned key: taking the sign bit apart makes
// float bits monotone, and a NaN becomes 0, which is below every finite key
// (the smallest is -inf's 0x007FFFFF).  -0.0 and +0.0 must tie, so they
// canonicalise to one bit pattern.
constant uint k_padded = $PADDED;      // k rounded up to a power of two
constant uint k_threads = $THREADS;
constant uint k_bins_per_thread = 256 / k_threads;

static inline uint order_key(float v) {
    uint bits = as_type<uint>(v);
    if (isnan(v)) return 0u;
    if (v == 0.0f) return 0x80000000u;
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Score in the high half, inverted index in the low half: unsigned order on the
// packed key is the contract's order, and it is strict because indices are.
static inline ulong pack_key(uint key, uint index) {
    return ((ulong)key << 32) | (ulong)(~index);
}

static inline uint scan_up(uint value, uint lane) {
    for (uint d = 1; d < 32; d <<= 1) {
        uint up = simd_shuffle_up(value, d);
        if (lane >= d) value += up;
    }
    return value;
}

// Exclusive prefix sum of one value per thread, in thread order.  ``totals`` is
// the caller's scratch: it must be its own array because the group total is
// read after this returns.  ``groups`` is the threadgroup's simdgroup count.
static inline uint group_prefix(uint value, threadgroup uint* totals,
                                uint tid, uint lane, uint groups) {
    uint incl = scan_up(value, lane);
    if (lane == 31) totals[tid >> 5] = incl;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint own = tid < groups ? totals[tid] : 0u;   // every simdgroup runs the
    uint scanned = scan_up(own, lane);            // shuffle, only group 0 counts
    if (tid < groups) totals[tid] = scanned - own;
    if (tid == groups - 1) totals[groups] = scanned;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return totals[tid >> 5] + incl - value;
}

// Large-k form: one threadgroup per row, 32-bit order keys, and the k-th key
// found by histogramming eight bits at a time.  The window forms keep k scores
// and k indices per lane in registers and merge them through threadgroup
// memory, so cost and register pressure both grow with k until the window
// spills — at k = 56 the merge form is already three times the sort path.  This
// form holds nothing per-k in registers: the only thing that grows with k is a
// k-entry candidate buffer, and the per-element work is a shift, a compare and
// one histogram increment.
kernel void select_radix(device const float* scores, device long* out,
                         constant uint& n, constant uint& rows,
                         uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint hist[256];
    threadgroup uint tot_hist[16];
    threadgroup uint tot_gt[16];
    threadgroup uint tot_eq[16];
    threadgroup uint chosen[2];
    threadgroup ulong cand[k_padded];
#if $ASCENDING
    threadgroup uint member[k_padded];
#endif

    if (row >= rows) return;
    device const float* base = scores + (ulong)row * n;
    uint lane = tid & 31u;

    // Four digits of eight bits, most significant first.  ``mask`` and
    // ``prefix`` hold the digits already decided; ``need`` is how many of the
    // top entries are still to be drawn from the surviving group.
    uint mask = 0u, prefix = 0u, need = selected;
    for (uint pass = 0; pass < 4; ++pass) {
        for (uint b = tid; b < 256; b += k_threads)
            atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint shift = 24 - 8 * pass;
        for (uint i = tid; i < n; i += k_threads) {
            uint key = order_key(base[i]);
            if ((key & mask) == prefix)
                atomic_fetch_add_explicit(&hist[(key >> shift) & 0xFFu], 1u,
                                          memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Thread t owns the highest bins not yet claimed by the threads below
        // it, so the exclusive prefix over that reversed order counts the keys
        // in the bins above it, and exactly one bin satisfies
        // ``above < need <= above + own``.
        uint first = 255 - tid * k_bins_per_thread;
        uint own = 0;
        for (uint j = 0; j < k_bins_per_thread; ++j)
            own += atomic_load_explicit(&hist[first - j], memory_order_relaxed);
        uint above = group_prefix(own, tot_hist, tid, lane, k_threads >> 5);
        for (uint j = 0; j < k_bins_per_thread; ++j) {
            uint bin = atomic_load_explicit(&hist[first - j], memory_order_relaxed);
            if (above < need && need <= above + bin) {
                chosen[0] = first - j;
                chosen[1] = need - above;
            }
            above += bin;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix |= chosen[0] << shift;
        need = chosen[1];
        mask = (mask >> 8) | 0xFF000000u;
    }
    // ``prefix`` is the exact k-th key now, and ``need`` — at least 1, at most
    // the count of that key — is how many exact ties the selection still takes.

    // Chunk t owns the contiguous range [t*per, (t+1)*per), so walking chunks in
    // thread order walks the row in index order, which is what decides the ties.
    uint per = (n + k_threads - 1) / k_threads;
    uint lo = tid * per;
    uint hi = min(n, lo + per);
    uint ngt = 0, neq = 0;
    for (uint i = lo; i < hi; ++i) {
        uint key = order_key(base[i]);
        ngt += key > prefix ? 1u : 0u;
        neq += key == prefix ? 1u : 0u;
    }
    uint at_gt = group_prefix(ngt, tot_gt, tid, lane, k_threads >> 5);
    uint total_gt = tot_gt[k_threads >> 5];
    uint at_eq = group_prefix(neq, tot_eq, tid, lane, k_threads >> 5);
    uint take_eq = selected - total_gt;      // at least 1 by construction
    uint w = at_gt, e = at_eq;
    for (uint i = lo; i < hi; ++i) {
        uint key = order_key(base[i]);
        if (key > prefix) {
            cand[w++] = pack_key(key, i);
        } else if (key == prefix) {
            if (e < take_eq) cand[total_gt + e] = pack_key(key, i);
            ++e;
        }
    }
    for (uint c = selected + tid; c < k_padded; c += k_threads) cand[c] = 0ul;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The packed key is a strict total order, so a candidate's rank among the
    // candidates is its output slot: k compares per candidate in one pass,
    // against log2(k)^2/2 barriers for a bitonic sort of the same buffer.
    for (uint c = tid; c < k_padded; c += k_threads) {
        ulong mine = cand[c];
        uint rank = 0;
        for (uint j = 0; j < k_padded; ++j) {
            ulong other = cand[j];
            rank += other > mine ? 1u : (other == mine && j < c ? 1u : 0u);
        }
#if $ASCENDING
        member[c] = rank < selected ? (uint)(~mine) : 0xFFFFFFFFu;
#else
        if (rank < selected) out[(ulong)row * selected + rank] = (long)(uint)(~mine);
#endif
    }
#if $ASCENDING
    // sorted=False: the same k entries, in ascending index order.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint c = tid; c < k_padded; c += k_threads) {
        uint mine = member[c];
        if (mine == 0xFFFFFFFFu) continue;
        uint rank = 0;
        for (uint j = 0; j < k_padded; ++j) {
            uint other = member[j];
            rank += other != 0xFFFFFFFFu && (other < mine || (other == mine && j < c))
                    ? 1u : 0u;
        }
        out[(ulong)row * selected + rank] = (long)mine;
    }
#endif
}
"""

# The radix form holds nothing per-k in registers, so it carries the whole range
# the kernel covers.  Above it the two-argsort path wins again: the candidate
# rank pass is O(k^2) per row and the candidate buffer is O(k), so the margin
# erodes with k until the sort path takes back over somewhere above 384.
# Measured: k-limit probe in measured/k_select.json.
_KERNEL_MAX_K = 256

# The radix form takes over from the window forms at this k.  Measured across
# nine shapes: below it the window still fits in registers and wins by up to a
# quarter (k = 8), at it the two are within noise, and from 20 up the radix form
# wins on every shape probed, by 10% at 20 and by 3x at 48 where the window's
# merge tree has doubled.  Measured crossover in measured/k_select.json.
_RADIX_MIN_K = 16

# Threadgroup size of the radix form.  Each thread owns 256/k_threads histogram
# bins and walks the row in ``k_threads`` strides; a narrower group puts more
# rows in flight per core, which is what the many-short-row shapes want.
_RADIX_THREADS = 32

# Threadgroup memory the device gives one threadgroup; every per-row buffer is
# sized from it, and beyond it the pipeline fails to build rather than running
# slow, so the k each form can hold follows from it.
_THREADGROUP_BYTES = 32768


def _lanes_max_k(lanes):
    """Largest k the merge form holds: two arrays of k per lane, 8 B a slot."""
    return _THREADGROUP_BYTES // (8 * lanes)


def _radix_max_k():
    """Largest k whose candidate buffer fits beside the histogram and the scan."""
    return _THREADGROUP_BYTES // 16      # k_padded <= 2k ulongs, plus 2 KB of scratch


# Most lanes one row is split across; sizes the threadgroup buffer in the shader.
_MAX_LANES = 64

_KERNEL_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


@lru_cache(maxsize=32)
def _library(k, ascending, threads=None):
    """Shader library for one ``k``, output order and radix width.

    Compiles on first use only; ``threads`` only reaches the radix form, so the
    window forms share the library of the width they happen to be compiled with.
    """
    padded = 1 << max(k - 1, 0).bit_length()
    source = (_SOURCE.replace("$K", str(k)).replace("$LANES", str(_MAX_LANES))
              .replace("$PADDED", str(padded))
              .replace("$THREADS", str(threads or _RADIX_THREADS))
              .replace("$ASCENDING", "1" if ascending else "0"))
    return torch.mps.compile_shader(source)


def _plan(k, width, rows):
    """Kernel name and threadgroup size for this shape, or None for the sort path.

    Two window forms and one radix form.  The window forms do one compare per
    element and differ in occupancy: the merge form keeps ``lanes`` loads in
    flight and coalesces them, and pays for that with a shared-memory merge per
    row, so it owns every shape with few enough rows to leave cores idle.  Past
    8192 rows the tall form's one thread per row keeps more of them busy.

    From ``_RADIX_MIN_K`` the radix form owns every shape: its per-element work
    does not depend on k at all, so there is no occupancy crossover left to
    choose between — it beats the window forms at both ends of the shape range,
    which is what the k-limit probe measured.  Both cutoffs are measured there.
    """
    if k > _KERNEL_MAX_K:
        return None
    if k >= _RADIX_MIN_K:
        return "select_radix", _RADIX_THREADS
    if rows >= 8192:
        return "select_topk_tall", 32
    lanes = 64 if width >= 1024 else 32 if width >= 256 else 16
    return f"select_topk_l{lanes}", lanes


def _check(scores, k, dim):
    if scores.dim() == 0:
        raise ValueError("scores must have at least one dimension")
    if isinstance(k, bool):
        raise TypeError("k must be an integer, got bool")
    try:
        k = operator.index(k)
    except TypeError:
        raise TypeError(f"k must be an integer, got {type(k).__name__}") from None
    dim = dim + scores.dim() if dim < 0 else dim
    if not 0 <= dim < scores.dim():
        raise IndexError(f"dim {dim} is out of range for {scores.dim()} dimensions")
    width = scores.shape[dim]
    if not 0 <= k <= width:
        raise ValueError(f"k must satisfy 0 <= k <= {width}, got {k}")
    if not scores.is_floating_point():
        raise TypeError(f"scores must be floating point, got {scores.dtype}")
    return dim, k, width


def _empty_like_selection(scores, dim):
    shape = list(scores.shape)
    shape[dim] = 0
    return torch.empty(shape, dtype=torch.long, device=scores.device)


def k_select(scores, k, dim=-1, sorted=True):
    """Indices of the ``k`` best blocks along ``dim``, ties broken by lower index.

    Args:
        scores: (..., blocks) float scores; NaNs rank after every finite value.
        k: number of blocks to keep, ``0 <= k <= scores.shape[dim]``.
        dim: axis the blocks live on.
        sorted: True for descending score order, False for ascending index order.

    Returns:
        int64 indices, ``scores.shape`` with ``k`` in place of ``dim``.
    """
    dim, k, width = _check(scores, k, dim)
    if k == 0:
        return _empty_like_selection(scores, dim)
    if scores.device.type == "mps" and scores.dtype in _KERNEL_DTYPES:
        rows = scores.numel() // width
        plan = _plan(k, width, rows)
        if plan is not None:
            return _kernel_select(scores, k, dim, sorted, plan)
    return _sort_select(scores, k, dim, sorted)


def k_select_values(scores, k, dim=-1, sorted=True):
    """The selected scores, differentiable at exactly the selected entries."""
    return torch.gather(scores, dim, k_select(scores, k, dim, sorted))


def reference_k_select(scores, k, dim=-1, sorted=True):
    """The same selection in plain aten ops: CPU, any device, any floating width."""
    dim, k, _ = _check(scores, k, dim)
    if k == 0:
        return _empty_like_selection(scores, dim)
    return _sort_select(scores, k, dim, sorted)


def _sort_select(scores, k, dim, sorted):
    """Stable argsorts implement the order exactly, including the NaN tail.

    NaN cannot be a sort key: ``argsort`` puts it last ascending and first
    descending, which would place a NaN row above every finite score.  Masking
    them to -inf ties them with real -inf, so a second stable sort on the NaN
    flag lifts the real -inf entries back above the NaNs while keeping every
    within-group order from the first sort.
    """
    nan = torch.isnan(scores)
    primary = torch.where(nan, torch.full_like(scores, float("-inf")), scores)
    order = torch.argsort(primary, dim=dim, descending=True, stable=True)
    nan_after = torch.gather(nan, dim, order)
    order = torch.gather(order, dim, torch.argsort(nan_after, dim=dim, stable=True))
    order = order.narrow(dim, 0, k)
    if not sorted:
        order = order.sort(dim=dim).values
    return order


def _kernel_select(scores, k, dim, sorted, plan):
    rows_tensor = scores if dim == scores.dim() - 1 else scores.movedim(dim, -1)
    rows_tensor = rows_tensor.contiguous()
    if rows_tensor.dtype is not torch.float32:
        rows_tensor = rows_tensor.float()
    width = rows_tensor.shape[-1]
    rows = rows_tensor.numel() // width
    out = torch.empty(rows * k, dtype=torch.long, device=scores.device)
    if rows:
        name, group = plan
        threads_per_row = 1 if name == "select_topk_tall" else group
        kernel = getattr(_library(k, not sorted,
                              group if name == "select_radix" else None), name)
        kernel(rows_tensor, out, width, rows,
               threads=rows * threads_per_row, group_size=group)
    selected = out.view(*rows_tensor.shape[:-1], k)
    return selected if dim == scores.dim() - 1 else selected.movedim(-1, dim)
