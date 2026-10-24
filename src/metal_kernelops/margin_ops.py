"""MPS overlays for ``multi_margin_loss`` and ``multilabel_margin_loss``.

torch 2.14.1 has no MPS kernel for any of the four aten overloads, so
``nn.MultiMarginLoss`` and ``nn.MultiLabelMarginLoss`` raise
``NotImplementedError`` on an mps tensor. This module fills them with the
native equations from ``aten/src/ATen/native/LossMultiMargin.cpp`` and
``LossMultiLabelMargin.cpp`` at git ``5c488690``.

Two native behaviours are easy to get wrong and are reproduced on purpose:

* ``multi_margin_loss`` scales every positive term by ``weight[target]``, not
  by ``weight[d]`` (LossMultiMargin.cpp, ``multi_margin_inner_sum_cpu``).
* ``multilabel_margin_loss`` ignores target entries at and after the first
  negative one, and its ``is_target`` mask is a float tensor with one column
  per class.

Every dtype the fused path takes (float32, float16, bfloat16) has its own
kernel set generated from one templated body: loads widen to float32, the whole
reduction runs in float32, and each store narrows once. bfloat16 is read and
written as raw 16-bit words and converted here, because this Metal toolchain
rejects ``bfloat16_t``. ``multilabel_margin_loss`` walks the classes once and
weighs every label inside that walk instead of walking them once per label; its
backward counts how many classes each label beats and how many labels each
class beats, so no pass reads back what it wrote.

Narrow rows (at most 32 classes) run one thread per row, wider rows one
threadgroup per row. When a reduction is asked for and the batch fits a single
threadgroup, that one launch produces the reduced scalar; larger batches keep
the per-row partials plus a reduce pass. Importing this module compiles nothing
and registers nothing; ``OVERLAYS`` is data for the accelerator.

Target entries are assumed to be in range, as the CPU kernels require.
Out-of-range entries are bounds-checked by the fused kernels and contribute
nothing; the CPU op raises for them instead.
"""

from functools import lru_cache

import torch

REDUCTION_NONE = 0
REDUCTION_MEAN = 1
REDUCTION_SUM = 2

_MAX_LABEL_CHUNK = 64
"""Target slots folded into one three-dimensional temporary."""

_CHUNK_ELEMENTS = 1 << 22
"""Cap on ``nframe * chunk * dim`` elements per temporary, 16 MiB at float32."""

_REDUCE_GROUP = 128
"""Threads in the single threadgroup that reduces a batch to a scalar."""

_ROW_DIM = 32
"""Rows at most this wide run one thread per row."""

_TOTAL_ELEMENTS = 1 << 12
"""Largest ``nframe * dim`` the single-launch reduction takes, 16 KiB at float32.

One threadgroup reads the whole batch at about 7 GB/s against 34 GB/s for the
row-parallel path, and a second launch costs about 2 us of enqueue time, so the
two cross near 4k elements. The single launch also needs narrow rows, because
it reuses the serial row scan.
"""

_TOTAL_MIN_ROWS = 8
"""Fewer rows than this leave the single reducing threadgroup mostly idle."""

_TOTAL_GROUP = 256
"""Most threads in the threadgroup that reduces a whole batch."""

_SUFFIX = {torch.float32: "", torch.float16: "_h", torch.bfloat16: "_b"}
"""Kernel name suffix per fused dtype."""

_TRACE = None
"""When this is a list, every fused launch appends the name it dispatched."""


def _group_for(dim):
    """Threads per row: never much more than the row has classes."""
    if dim <= 32:
        return 32
    if dim <= 64:
        return 64
    return 128


def _row_per_thread(dim):
    """Narrow rows are cheaper as one serial row per thread than as a grid."""
    return dim <= _ROW_DIM


def _label_group(dim):
    """The multilabel kernels measured fastest with one or two warps per row."""
    return 32 if dim <= 32 else 64


def _total_group(nframe):
    """One row per thread, clamped to the size of a single threadgroup."""
    group = 32
    while group < nframe and group < _TOTAL_GROUP:
        group *= 2
    return group


def _multi_margin_matrix(self, target):
    """Shape-check per LossMulti.h and return the 2-D working view."""
    ndims = self.dim()
    if (
        ndims > 2
        or (ndims == 2 and self.size(1) == 0)
        or (ndims == 1 and self.size(0) == 0)
    ):
        raise RuntimeError(
            "Expected non-empty vector or matrix with optional 0-dim batch size, "
            f"but got: {list(self.shape)}"
        )
    if ndims <= 1:
        nframe = 1
        dim = 1 if ndims == 0 else self.size(0)
        x = self.reshape(1, dim)
    else:
        nframe, dim = self.shape
        x = self
    if target.dim() > 1 or target.numel() != nframe:
        raise RuntimeError(
            "multi_margin_loss: target tensor should be 1-D with size equal to "
            f"the number of input samples (batch size). Expected target size "
            f"[{nframe}], but got {list(target.shape)}. Input has shape "
            f"{list(self.shape)}."
        )
    if target.dtype != torch.int64:
        raise RuntimeError(
            "expected scalar type Long but found "
            f"{str(target.dtype).split('.')[-1].capitalize()}"
        )
    return x, target.reshape(nframe), nframe, dim


def _check_weight(weight, dim):
    if weight is None:
        return
    if weight.dim() > 1 or weight.numel() != dim:
        raise RuntimeError(
            f"inconsistent weight size, expected {dim} but got {list(weight.shape)}"
        )


def _multi_margin_weight(weight, dim, x, t):
    if weight is None:
        return None
    w = weight.reshape(dim)
    if w.dtype != x.dtype:
        w = w.to(x.dtype)
    return w.gather(0, t).unsqueeze(1)


def _multi_margin_column_mask(x, t, dim):
    columns = torch.arange(dim, device=x.device)
    return columns.unsqueeze(0) == t.unsqueeze(1)


def _multi_margin_inner(x, t, margin, p):
    target_value = x.gather(1, t.unsqueeze(1))
    z = ((margin - target_value) + x).clamp_min(0)
    return z if p == 1 else z * z


def _check_p(p):
    if p != 1 and p != 2:
        raise ValueError("only p == 1 and p == 2 supported")


def _composed_multi_margin(x, t, p, margin, weight, reduction):
    term = _multi_margin_inner(x, t, margin, p)
    weight_target = _multi_margin_weight(weight, x.size(1), x, t)
    if weight_target is not None:
        term = term * weight_target
    term = term.masked_fill(_multi_margin_column_mask(x, t, x.size(1)), 0)
    wide = x.dtype in (torch.float16, torch.bfloat16)
    sample = term.sum(1).float() / x.size(1) if wide else term.sum(1) / x.size(1)
    if reduction == REDUCTION_NONE:
        return sample.to(x.dtype)
    total = sample.sum()
    if reduction == REDUCTION_MEAN:
        total = total / x.size(0)
    return total.to(x.dtype)


def multi_margin_loss(self, target, p=1, margin=1, weight=None, reduction=REDUCTION_MEAN):
    _check_p(p)
    x, t, _, dim = _multi_margin_matrix(self, target)
    scalar_target = target.dim() == 0
    _check_weight(weight, dim)
    if x.numel() == 0:
        if reduction == REDUCTION_NONE and not scalar_target:
            return x.new_empty((0,))
        return x.new_zeros(())
    if _metal_ready(self, weight):
        result = _metal_multi_margin(x, t, p, margin, weight, reduction)
    else:
        result = _composed_multi_margin(x, t, p, margin, weight, reduction)
    if reduction == REDUCTION_NONE and scalar_target:
        return result.reshape(())
    return result


def multi_margin_loss_backward(
    grad_output, self, target, p, margin, weight=None, reduction=REDUCTION_MEAN
):
    _check_p(p)
    x, t, nframe, dim = _multi_margin_matrix(self, target)
    _check_weight(weight, dim)
    if x.numel() == 0:
        return x.new_zeros(self.shape)
    if _metal_ready(self, weight):
        grad_input = _metal_multi_margin_backward(
            grad_output, x, t, p, margin, weight, reduction
        )
    else:
        scale = 1.0 / (nframe * dim) if reduction == REDUCTION_MEAN else 1.0 / dim
        target_value = x.gather(1, t.unsqueeze(1))
        z = ((margin - target_value) + x).clamp_min(0)
        if p == 1:
            term = (z > 0).to(x.dtype) * scale
        else:
            term = z * (2.0 * scale)
        weight_target = _multi_margin_weight(weight, dim, x, t)
        if weight_target is not None:
            term = term * weight_target
        term = term.masked_fill(_multi_margin_column_mask(x, t, dim), 0)
        row = term.sum(1)
        grad_input = term.scatter(1, t.unsqueeze(1), -row.unsqueeze(1))
        grad_output = grad_output.to(x.dtype)
        if reduction != REDUCTION_NONE or grad_output.dim() == 0:
            grad_input = grad_input * grad_output
        else:
            grad_input = grad_input * grad_output.reshape(nframe, 1)
    return grad_input.reshape(self.shape)


def _multilabel_matrix(self, target):
    ndims = self.dim()
    if (
        ndims > 2
        or (ndims == 2 and self.size(1) == 0)
        or (ndims == 1 and self.size(0) == 0)
    ):
        raise RuntimeError(
            "Expected non-empty vector or matrix with optional 0-dim batch size, "
            f"but got: {list(self.shape)}"
        )
    if ndims <= 1:
        nframe = 1
        dim = 1 if ndims == 0 else self.size(0)
        x = self.reshape(1, dim)
        if target.dim() > 1 or target.numel() != dim:
            raise RuntimeError(
                f"inconsistent target size: {list(target.shape)} for input of "
                f"size: {list(self.shape)}"
            )
        t = target.reshape(1, dim)
    else:
        nframe, dim = self.shape
        x = self
        if target.dim() != 2 or target.shape[0] != nframe or target.shape[1] != dim:
            raise RuntimeError(
                f"inconsistent target size: {list(target.shape)} for input of "
                f"size: {list(self.shape)}"
            )
        t = target
    if t.dtype != torch.int64:
        raise RuntimeError(
            "expected scalar type Long but found "
            f"{str(t.dtype).split('.')[-1].capitalize()}"
        )
    return x, t, nframe, dim


def _label_chunk(nframe, dim):
    return max(1, min(_MAX_LABEL_CHUNK, _CHUNK_ELEMENTS // max(1, nframe * dim)))


def _target_prefix(t, dim):
    """True up to the first negative target entry, matching the native break."""
    columns = torch.arange(dim, device=t.device)
    end = torch.where(t < 0, columns, dim).amin(-1, keepdim=True)
    return columns < end


def _composed_multilabel_target(t, dim, dtype):
    active = _target_prefix(t, dim)
    nframe = t.size(0)
    padded = torch.zeros(nframe, dim + 1, device=t.device, dtype=dtype)
    padded.scatter_(1, torch.where(active, t, dim), active.to(dtype))
    return active, padded[:, :dim]


def _composed_multilabel_forward(x, t, active, is_target, reduction, scalar_target):
    nframe, dim = x.shape
    keep = (1 - is_target).unsqueeze(1)
    wide = x.dtype in (torch.float16, torch.bfloat16)
    acc = torch.zeros(nframe, dtype=torch.float32 if wide else x.dtype, device=x.device)
    chunk = _label_chunk(nframe, dim)
    for start in range(0, dim, chunk):
        slots = t[:, start : start + chunk]
        live = active[:, start : start + chunk].to(x.dtype).unsqueeze(-1)
        value = x.unsqueeze(1).expand(nframe, slots.size(1), dim)
        value = value.gather(2, slots.clamp_min(0).unsqueeze(-1))
        z = (1.0 - value) + x.unsqueeze(1)
        z.clamp_min_(0)
        z.mul_(keep).mul_(live)
        acc.add_(z.sum((1, 2)).to(acc.dtype))
    if reduction != REDUCTION_NONE or scalar_target:
        total = acc.sum() / dim
        if reduction == REDUCTION_MEAN:
            total = total / nframe
        return total.to(x.dtype)
    return (acc / dim).to(x.dtype)


def _composed_multilabel_backward(x, t, active, is_target, grad_output, reduction):
    nframe, dim = x.shape
    keep = (1 - is_target).to(x.dtype).unsqueeze(1)
    scale = 1.0 / (nframe * dim) if reduction == REDUCTION_MEAN else 1.0 / dim
    grad_input = x.new_zeros(nframe, dim)
    chunk = _label_chunk(nframe, dim)
    for start in range(0, dim, chunk):
        slots = t[:, start : start + chunk]
        live = active[:, start : start + chunk].to(x.dtype).unsqueeze(-1)
        index = slots.clamp_min(0)
        value = x.unsqueeze(1).expand(nframe, slots.size(1), dim)
        value = value.gather(2, index.unsqueeze(-1))
        positive = (1.0 - value) + x.unsqueeze(1)
        positive = (positive > 0).to(x.dtype)
        positive.mul_(keep).mul_(live)
        grad_input.add_(positive.sum(1) * scale)
        grad_input.scatter_add_(1, index, positive.sum(2).mul_(-scale))
    grad_output = grad_output.to(x.dtype)
    if reduction != REDUCTION_NONE or grad_output.dim() == 0:
        return grad_input * grad_output
    return grad_input * grad_output.reshape(nframe, 1)


def multilabel_margin_loss_forward(self, target, reduction=REDUCTION_MEAN):
    x, t, _, dim = _multilabel_matrix(self, target)
    scalar_target = target.dim() <= 1
    if x.numel() == 0:
        empty = reduction == REDUCTION_NONE and not scalar_target
        output = x.new_empty((0,)) if empty else x.new_zeros(())
        return output, x.new_zeros(target.shape)
    if _metal_ready(self, None):
        output, is_target = _metal_multilabel(x, t, reduction)
    else:
        active, is_target = _composed_multilabel_target(t, dim, x.dtype)
        output = _composed_multilabel_forward(
            x, t, active, is_target, reduction, scalar_target
        )
    if reduction == REDUCTION_NONE and scalar_target:
        output = output.reshape(())
    return output, is_target.reshape(target.shape)


def multilabel_margin_loss_backward(
    grad_output, self, target, reduction, is_target
):
    x, t, nframe, dim = _multilabel_matrix(self, target)
    if is_target.numel() != target.numel():
        raise RuntimeError(
            f"inconsistent is_target size: {list(is_target.shape)} for target of "
            f"size: {list(target.shape)}"
        )
    if x.numel() == 0:
        return x.new_zeros(self.shape)
    mask = is_target.reshape(nframe, dim)
    if _metal_ready(self, None) and mask.dtype == x.dtype:
        grad_input = _metal_multilabel_backward(
            grad_output, x, t, mask.contiguous(), reduction
        )
    else:
        active = _target_prefix(t, dim)
        grad_input = _composed_multilabel_backward(
            x, t, active, mask.to(x.dtype), grad_output, reduction
        )
    return grad_input.reshape(self.shape)


@lru_cache(maxsize=1)
def _library():
    return torch.mps.compile_shader(_SOURCE)


_KERNELS = {}


def _kernel(name, dtype):
    """The compiled entry point for ``name`` at ``dtype``, resolved once.

    A plain dict, not ``lru_cache``: this lookup sits between the aten call and
    the launch, where a microsecond of wrapper overhead is a visible share of a
    small batch.
    """
    key = (name, dtype)
    kernel = _KERNELS.get(key)
    if kernel is None:
        kernel = _KERNELS[key] = getattr(_library(), name + _SUFFIX[dtype])
    return kernel


def _metal_ready(self, weight):
    return (
        self.device.type == "mps"
        and self.dim() <= 2
        and self.dtype in _SUFFIX
        and (
            weight is None
            or (weight.dtype == self.dtype and weight.device.type == "mps")
        )
    )


def _metal_multi_margin(x, t, p, margin, weight, reduction):
    nframe, dim = x.shape
    x = x.contiguous()
    t = t.contiguous()
    weight_buffer = x if weight is None else weight.contiguous()
    has_weight = 0 if weight is None else 1
    if (
        reduction != REDUCTION_NONE
        and dim <= _ROW_DIM
        and nframe * dim <= _TOTAL_ELEMENTS
    ):
        group = _total_group(nframe)
        total = torch.empty((), device=x.device, dtype=x.dtype)
        kernel = _kernel("mm_total", x.dtype)
        if _TRACE is not None:
            _TRACE.append("mm_total" + _SUFFIX[x.dtype])
        kernel(
            x, t, weight_buffer, total, nframe, dim, float(margin), int(p), has_weight,
            1 if reduction == REDUCTION_MEAN else 0,
            threads=group, group_size=group,
        )
        return total
    partial = torch.empty(nframe, device=x.device, dtype=x.dtype)
    group = _group_for(dim)
    margin = float(margin)
    p = int(p)
    if _row_per_thread(dim):
        kernel = _kernel("mm_forward_rows", x.dtype)
        if _TRACE is not None:
            _TRACE.append("mm_forward_rows" + _SUFFIX[x.dtype])
        kernel(x, t, weight_buffer, partial, nframe, dim, margin, p, has_weight,
               threads=nframe)
    else:
        kernel = _kernel("mm_forward", x.dtype)
        if _TRACE is not None:
            _TRACE.append("mm_forward" + _SUFFIX[x.dtype])
        kernel(x, t, weight_buffer, partial, nframe, dim, margin, p, has_weight,
               threads=nframe * group, group_size=group)
    if reduction == REDUCTION_NONE:
        return partial
    return _reduce(partial, reduction == REDUCTION_MEAN)


def _metal_multi_margin_backward(grad_output, x, t, p, margin, weight, reduction):
    nframe, dim = x.shape
    group = _group_for(dim)
    x = x.contiguous()
    t = t.contiguous()
    grad_output = grad_output.to(x.dtype).contiguous()
    weight_buffer = x if weight is None else weight.contiguous()
    per_row = reduction == REDUCTION_NONE and grad_output.dim() > 0
    grad_input = torch.empty_like(x)
    margin = float(margin)
    p = int(p)
    has_weight = 0 if weight is None else 1
    scale = float(1.0 / (nframe * dim) if reduction == REDUCTION_MEAN else 1.0 / dim)
    per_row = 1 if per_row else 0
    if _row_per_thread(dim):
        kernel = _kernel("mm_backward_rows", x.dtype)
        if _TRACE is not None:
            _TRACE.append("mm_backward_rows" + _SUFFIX[x.dtype])
        kernel(x, t, weight_buffer, grad_output, grad_input, nframe, dim, margin,
               p, has_weight, scale, per_row, threads=nframe)
    else:
        kernel = _kernel("mm_backward", x.dtype)
        if _TRACE is not None:
            _TRACE.append("mm_backward" + _SUFFIX[x.dtype])
        kernel(x, t, weight_buffer, grad_output, grad_input, nframe, dim, margin,
               p, has_weight, scale, per_row,
               threads=nframe * group, group_size=group)
    return grad_input


def _metal_multilabel(x, t, reduction):
    nframe, dim = x.shape
    x = x.contiguous()
    t = t.contiguous()
    is_target = torch.empty(nframe, dim, device=x.device, dtype=x.dtype)
    if (
        reduction != REDUCTION_NONE
        and nframe >= _TOTAL_MIN_ROWS
        and nframe * dim <= _TOTAL_ELEMENTS
    ):
        group = _total_group(nframe)
        total = torch.empty((), device=x.device, dtype=x.dtype)
        kernel = _kernel("ml_total", x.dtype)
        if _TRACE is not None:
            _TRACE.append("ml_total" + _SUFFIX[x.dtype])
        kernel(
            x, t, is_target, total, nframe, dim,
            1 if reduction == REDUCTION_MEAN else 0,
            threads=group, group_size=group,
        )
        return total, is_target
    partial = torch.empty(nframe, device=x.device, dtype=x.dtype)
    group = _label_group(dim)
    kernel = _kernel("ml_forward", x.dtype)
    if _TRACE is not None:
        _TRACE.append("ml_forward" + _SUFFIX[x.dtype])
    kernel(x, t, is_target, partial, nframe, dim,
           threads=nframe * group, group_size=group)
    if reduction == REDUCTION_NONE:
        return partial, is_target
    total = _reduce(partial, reduction == REDUCTION_MEAN)
    return total, is_target


def _metal_multilabel_backward(grad_output, x, t, is_target, reduction):
    nframe, dim = x.shape
    group = _label_group(dim)
    x = x.contiguous()
    t = t.contiguous()
    grad_output = grad_output.to(x.dtype).contiguous()
    per_row = reduction == REDUCTION_NONE and grad_output.dim() > 0
    grad_input = torch.empty_like(x)
    kernel = _kernel("ml_backward", x.dtype)
    if _TRACE is not None:
        _TRACE.append("ml_backward" + _SUFFIX[x.dtype])
    kernel(
        x, t, is_target, grad_output, grad_input, nframe, dim,
        float(1.0 / (nframe * dim) if reduction == REDUCTION_MEAN else 1.0 / dim),
        1 if per_row else 0,
        threads=nframe * group, group_size=group,
    )
    return grad_input


def _reduce(partial, divides):
    output = torch.empty((), device=partial.device, dtype=partial.dtype)
    kernel = _kernel("row_reduce", partial.dtype)
    if _TRACE is not None:
        _TRACE.append("row_reduce" + _SUFFIX[partial.dtype])
    kernel(partial, output, partial.numel(), 1 if divides else 0,
           threads=_REDUCE_GROUP, group_size=_REDUCE_GROUP)
    return output


def _store(out, result):
    if out.shape != result.shape:
        out.resize_(result.shape)
    out.copy_(result)
    return out


def multi_margin_loss_out(
    self, target, p=1, margin=1, weight=None, reduction=REDUCTION_MEAN, *, out
):
    return _store(out, multi_margin_loss(self, target, p, margin, weight, reduction))


def multi_margin_loss_backward_out(
    grad_output, self, target, p, margin, weight=None,
    reduction=REDUCTION_MEAN, *, grad_input
):
    return _store(
        grad_input,
        multi_margin_loss_backward(
            grad_output, self, target, p, margin, weight, reduction
        ),
    )


def multilabel_margin_loss_forward_out(
    self, target, reduction=REDUCTION_MEAN, *, output, is_target
):
    result, mask = multilabel_margin_loss_forward(self, target, reduction)
    _store(output, result)
    _store(is_target, mask)
    return output, is_target


def multilabel_margin_loss_backward_out(
    grad_output, self, target, reduction, is_target, *, grad_input
):
    return _store(
        grad_input,
        multilabel_margin_loss_backward(
            grad_output, self, target, reduction, is_target
        ),
    )


_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// Widening loads and narrowing stores: every kernel accumulates in float32 and
// rounds once, at the store. bfloat16 has no type in this Metal toolchain, so
// it travels as a 16-bit word and is converted by hand.
inline float ld(device const float* p, uint i) { return p[i]; }
inline float ld(device const half* p, uint i) { return float(p[i]); }
inline float ld(device const ushort* p, uint i) { return as_type<float>(uint(p[i]) << 16); }

inline void st(device float* p, uint i, float v) { p[i] = v; }
inline void st(device half* p, uint i, float v) { p[i] = half(v); }
inline void st(device ushort* p, uint i, float v) {
    uint bits = as_type<uint>(v);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    p[i] = ushort(bits >> 16);
}

inline void barrier_group() {
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
}

inline float group_sum(threadgroup float* shared, uint tid, uint group) {
    barrier_group();
    for (uint off = group / 2; off > 0; off >>= 1) {
        if (tid < off) shared[tid] += shared[tid + off];
        barrier_group();
    }
    return shared[0];
}

inline float margin_term(float gap, uint p, uint has_weight, float w) {
    if (gap <= 0.0f) return 0.0f;
    float h = (p == 1) ? gap : gap * gap;
    return has_weight ? h * w : h;
}

template <typename T>
inline float mm_row_serial(device const T* x, uint base, uint dim, uint ti,
                           float margin, uint p, uint has_weight, float w) {
    float x_target = ld(x, base + ti);
    float sum = 0.0f;
    for (uint d = 0; d < dim; ++d) {
        if (d == ti) continue;
        sum += margin_term(margin - x_target + ld(x, base + d), p, has_weight, w);
    }
    return sum;
}

template <typename T>
inline float mm_row_strided(device const T* x, uint base, uint dim, uint ti,
                            float margin, uint p, uint has_weight, float w,
                            uint tid, uint group) {
    float x_target = ld(x, base + ti);
    float sum = 0.0f;
    for (uint d = tid; d < dim; d += group) {
        if (d == ti) continue;
        sum += margin_term(margin - x_target + ld(x, base + d), p, has_weight, w);
    }
    return sum;
}

template <typename T>
inline void mm_forward_impl(device const T* x, device const long* target,
                            device const T* weight, device T* partial, uint dim,
                            float margin, uint p, uint has_weight, uint row,
                            uint tid, uint group, threadgroup float* shared) {
    uint base = row * dim;
    uint ti = uint(target[row]);
    float sum = 0.0f;
    if (ti < dim) {
        float w = has_weight ? ld(weight, ti) : 1.0f;
        sum = mm_row_strided(x, base, dim, ti, margin, p, has_weight, w, tid, group);
    }
    shared[tid] = sum;
    float total = group_sum(shared, tid, group);
    if (tid == 0) st(partial, row, total / float(dim));
}

template <typename T>
inline void mm_forward_rows_impl(device const T* x, device const long* target,
                                 device const T* weight, device T* partial,
                                 uint nframe, uint dim, float margin, uint p,
                                 uint has_weight, uint row) {
    if (row >= nframe) return;
    uint ti = uint(target[row]);
    if (ti >= dim) {
        st(partial, row, 0.0f);
        return;
    }
    float w = has_weight ? ld(weight, ti) : 1.0f;
    float sum = mm_row_serial(x, row * dim, dim, ti, margin, p, has_weight, w);
    st(partial, row, sum / float(dim));
}

template <typename T>
inline void mm_total_impl(device const T* x, device const long* target,
                          device const T* weight, device T* out, uint nframe,
                          uint dim, float margin, uint p, uint has_weight,
                          uint divides, uint tid, uint group,
                          threadgroup float* shared) {
    float acc = 0.0f;
    for (uint row = tid; row < nframe; row += group) {
        uint ti = uint(target[row]);
        if (ti >= dim) continue;
        float w = has_weight ? ld(weight, ti) : 1.0f;
        acc += mm_row_serial(x, row * dim, dim, ti, margin, p, has_weight, w)
             / float(dim);
    }
    shared[tid] = acc;
    float total = group_sum(shared, tid, group);
    if (tid == 0) st(out, 0, divides ? total / float(nframe) : total);
}

template <typename T>
inline void mm_backward_impl(device const T* x, device const long* target,
                             device const T* weight, device const T* go,
                             device T* grad, uint dim, float margin, uint p,
                             uint has_weight, float base_scale, uint per_row,
                             uint row, uint tid, uint group,
                             threadgroup float* shared) {
    uint base = row * dim;
    uint ti = uint(target[row]);
    float go_value = per_row ? ld(go, row) : ld(go, 0);
    float acc = 0.0f;
    if (ti < dim) {
        float x_target = ld(x, base + ti);
        float w = has_weight ? ld(weight, ti) : 1.0f;
        for (uint d = tid; d < dim; d += group) {
            if (d == ti) continue;
            float z = margin - x_target + ld(x, base + d);
            float h = 0.0f;
            if (z > 0.0f) {
                h = (p == 1) ? base_scale : 2.0f * base_scale * z;
                if (has_weight) h *= w;
                acc -= h;
            }
            st(grad, base + d, h * go_value);
        }
    } else {
        for (uint d = tid; d < dim; d += group) st(grad, base + d, 0.0f);
    }
    shared[tid] = acc;
    float total = group_sum(shared, tid, group);
    if (tid == 0 && ti < dim) st(grad, base + ti, total * go_value);
}

template <typename T>
inline void mm_backward_rows_impl(device const T* x, device const long* target,
                                  device const T* weight, device const T* go,
                                  device T* grad, uint nframe, uint dim,
                                  float margin, uint p, uint has_weight,
                                  float base_scale, uint per_row, uint row) {
    if (row >= nframe) return;
    uint base = row * dim;
    uint ti = uint(target[row]);
    if (ti >= dim) {
        for (uint d = 0; d < dim; ++d) st(grad, base + d, 0.0f);
        return;
    }
    float x_target = ld(x, base + ti);
    float w = has_weight ? ld(weight, ti) : 1.0f;
    float go_value = per_row ? ld(go, row) : ld(go, 0);
    float acc = 0.0f;
    for (uint d = 0; d < dim; ++d) {
        if (d == ti) continue;
        float z = margin - x_target + ld(x, base + d);
        float h = 0.0f;
        if (z > 0.0f) {
            h = (p == 1) ? base_scale : 2.0f * base_scale * z;
            if (has_weight) h *= w;
            acc -= h;
        }
        st(grad, base + d, h * go_value);
    }
    st(grad, base + ti, acc * go_value);
}

inline uint label_end(device const long* target, uint base, uint dim) {
    for (uint k = 0; k < dim; ++k) {
        if (target[base + k] < 0) return k;
    }
    return dim;
}

#define ML_STAGED 32

// One threadgroup per row. The label loop runs per class instead of per label,
// so each x[d] is read once however many labels the row carries; the label
// values themselves are staged in threadgroup memory so the inner loop reads
// no target entry per class. Past ML_STAGED labels the rest are read directly.
template <typename T>
inline void ml_forward_impl(device const T* x, device const long* target,
                            device T* is_target, device T* partial, uint dim,
                            uint row, uint tid, uint group,
                            threadgroup float* shared, threadgroup float* staged,
                            threadgroup uint* end_slot) {
    uint base = row * dim;
    if (tid == 0) *end_slot = label_end(target, base, dim);
    barrier_group();
    uint end = *end_slot;
    for (uint d = tid; d < dim; d += group) st(is_target, base + d, 0.0f);
    uint staged_count = min(end, uint(ML_STAGED));
    if (tid < staged_count) {
        uint ti = uint(target[base + tid]);
        // Infinity makes a staged out-of-range entry contribute nothing.
        staged[tid] = ti < dim ? ld(x, base + ti) : INFINITY;
    }
    barrier_group();
    for (uint k = tid; k < end; k += group) {
        uint ti = uint(target[base + k]);
        if (ti < dim) st(is_target, base + ti, 1.0f);
    }
    barrier_group();
    float acc = 0.0f;
    for (uint d = tid; d < dim; d += group) {
        if (ld(is_target, base + d) != 0.0f) continue;
        float xd = ld(x, base + d);
        float local = 0.0f;
        for (uint k = 0; k < staged_count; ++k) {
            float gap = 1.0f - staged[k] + xd;
            if (gap > 0.0f) local += gap;
        }
        for (uint k = staged_count; k < end; ++k) {
            uint ti = uint(target[base + k]);
            if (ti >= dim) continue;
            float gap = 1.0f - ld(x, base + ti) + xd;
            if (gap > 0.0f) local += gap;
        }
        acc += local;
    }
    shared[tid] = acc;
    float total = group_sum(shared, tid, group);
    if (tid == 0) st(partial, row, total / float(dim));
}

// One threadgroup for the whole batch: each thread walks whole rows, so the
// is_target mask and the reduced scalar come out of a single launch.
template <typename T>
inline void ml_total_impl(device const T* x, device const long* target,
                          device T* is_target, device T* out, uint nframe,
                          uint dim, uint divides, uint tid, uint group,
                          threadgroup float* shared) {
    float acc = 0.0f;
    for (uint row = tid; row < nframe; row += group) {
        uint base = row * dim;
        uint end = label_end(target, base, dim);
        for (uint d = 0; d < dim; ++d) st(is_target, base + d, 0.0f);
        for (uint k = 0; k < end; ++k) {
            uint ti = uint(target[base + k]);
            if (ti < dim) st(is_target, base + ti, 1.0f);
        }
        float rowsum = 0.0f;
        for (uint d = 0; d < dim; ++d) {
            if (ld(is_target, base + d) != 0.0f) continue;
            float xd = ld(x, base + d);
            float local = 0.0f;
            for (uint k = 0; k < end; ++k) {
                uint ti = uint(target[base + k]);
                if (ti >= dim) continue;
                float gap = 1.0f - ld(x, base + ti) + xd;
                if (gap > 0.0f) local += gap;
            }
            rowsum += local;
        }
        acc += rowsum / float(dim);
    }
    shared[tid] = acc;
    float total = group_sum(shared, tid, group);
    if (tid == 0) st(out, 0, divides ? total / float(nframe) : total);
}

// Two counted passes, one write per class: the class that beats a label gets
// the count of classes it beats, every other class gets the count of labels it
// beats. Nothing reads back what it wrote, so the row is touched once for the
// target slots and once for the rest.
template <typename T>
inline void ml_backward_impl(device const T* x, device const long* target,
                             device const T* is_target, device const T* go,
                             device T* grad, uint dim, float base_scale,
                             uint per_row, uint row, uint tid, uint group,
                             threadgroup float* shared, threadgroup float* staged,
                             threadgroup uint* end_slot) {
    uint base = row * dim;
    if (tid == 0) *end_slot = label_end(target, base, dim);
    barrier_group();
    uint end = *end_slot;
    float go_value = per_row ? ld(go, row) : ld(go, 0);
    float scale = base_scale * go_value;
    uint staged_count = min(end, uint(ML_STAGED));
    if (tid < staged_count) {
        uint ti = uint(target[base + tid]);
        staged[tid] = ti < dim ? ld(x, base + ti) : INFINITY;
    }
    if (tid == 0) {
        for (uint k = 0; k < end; ++k) {
            uint ti = uint(target[base + k]);
            if (ti < dim) st(grad, base + ti, 0.0f);
        }
    }
    barrier_group();
    // Per label: how many classes it beats, subtracted from its own slot.
    for (uint k = 0; k < end; ++k) {
        uint ti = uint(target[base + k]);
        if (ti >= dim) continue;
        float x_target = ld(x, base + ti);
        float local = 0.0f;
        for (uint d = tid; d < dim; d += group) {
            if (ld(is_target, base + d) == 0.0f) {
                if (1.0f - x_target + ld(x, base + d) > 0.0f) local += 1.0f;
            }
        }
        shared[tid] = local;
        float total = group_sum(shared, tid, group);
        if (tid == 0) st(grad, base + ti, ld(grad, base + ti) - total * scale);
        barrier_group();
    }
    // Per class: how many labels it beats.
    for (uint d = tid; d < dim; d += group) {
        if (ld(is_target, base + d) != 0.0f) continue;
        float xd = ld(x, base + d);
        float count = 0.0f;
        for (uint k = 0; k < staged_count; ++k) {
            if (1.0f - staged[k] + xd > 0.0f) count += 1.0f;
        }
        for (uint k = staged_count; k < end; ++k) {
            uint ti = uint(target[base + k]);
            if (ti >= dim) continue;
            if (1.0f - ld(x, base + ti) + xd > 0.0f) count += 1.0f;
        }
        st(grad, base + d, count * scale);
    }
}

#define MM_PARAMS(T) \
    device const T* x, device const long* target, device const T* weight, \
    device T* partial, constant uint& nframe, constant uint& dim, \
    constant float& margin, constant uint& p, constant uint& has_weight

#define MM_FWD_KERNEL(NAME, T) \
kernel void NAME(MM_PARAMS(T), \
                 uint row [[threadgroup_position_in_grid]], \
                 uint tid [[thread_index_in_threadgroup]], \
                 uint group [[threads_per_threadgroup]]) { \
    threadgroup float shared[256]; \
    mm_forward_impl(x, target, weight, partial, dim, margin, p, has_weight, \
                    row, tid, group, shared); \
}

#define MM_FWD_ROWS_KERNEL(NAME, T) \
kernel void NAME(MM_PARAMS(T), uint row [[thread_position_in_grid]]) { \
    mm_forward_rows_impl(x, target, weight, partial, nframe, dim, margin, p, \
                         has_weight, row); \
}

#define MM_TOTAL_KERNEL(NAME, T) \
kernel void NAME(device const T* x, device const long* target, \
                 device const T* weight, device T* out, constant uint& nframe, \
                 constant uint& dim, constant float& margin, constant uint& p, \
                 constant uint& has_weight, constant uint& divides, \
                 uint tid [[thread_index_in_threadgroup]], \
                 uint group [[threads_per_threadgroup]]) { \
    threadgroup float shared[256]; \
    mm_total_impl(x, target, weight, out, nframe, dim, margin, p, has_weight, \
                  divides, tid, group, shared); \
}

#define MM_BWD_PARAMS(T) \
    device const T* x, device const long* target, device const T* weight, \
    device const T* go, device T* grad, constant uint& nframe, \
    constant uint& dim, constant float& margin, constant uint& p, \
    constant uint& has_weight, constant float& base_scale, \
    constant uint& per_row

#define MM_BWD_KERNEL(NAME, T) \
kernel void NAME(MM_BWD_PARAMS(T), \
                 uint row [[threadgroup_position_in_grid]], \
                 uint tid [[thread_index_in_threadgroup]], \
                 uint group [[threads_per_threadgroup]]) { \
    threadgroup float shared[256]; \
    mm_backward_impl(x, target, weight, go, grad, dim, margin, p, has_weight, \
                     base_scale, per_row, row, tid, group, shared); \
}

#define MM_BWD_ROWS_KERNEL(NAME, T) \
kernel void NAME(MM_BWD_PARAMS(T), uint row [[thread_position_in_grid]]) { \
    mm_backward_rows_impl(x, target, weight, go, grad, nframe, dim, margin, p, \
                          has_weight, base_scale, per_row, row); \
}

#define ML_PARAMS(T) \
    device const T* x, device const long* target, device T* is_target, \
    device T* partial, constant uint& nframe, constant uint& dim

#define ML_FWD_KERNEL(NAME, T) \
kernel void NAME(ML_PARAMS(T), \
                 uint row [[threadgroup_position_in_grid]], \
                 uint tid [[thread_index_in_threadgroup]], \
                 uint group [[threads_per_threadgroup]]) { \
    threadgroup float shared[256]; \
    threadgroup float staged[ML_STAGED]; \
    threadgroup uint end_slot; \
    ml_forward_impl(x, target, is_target, partial, dim, row, tid, group, \
                    shared, staged, &end_slot); \
}

#define ML_TOTAL_KERNEL(NAME, T) \
kernel void NAME(device const T* x, device const long* target, \
                 device T* is_target, device T* out, constant uint& nframe, \
                 constant uint& dim, constant uint& divides, \
                 uint tid [[thread_index_in_threadgroup]], \
                 uint group [[threads_per_threadgroup]]) { \
    threadgroup float shared[256]; \
    ml_total_impl(x, target, is_target, out, nframe, dim, divides, tid, group, \
                  shared); \
}

#define ML_BWD_KERNEL(NAME, T) \
kernel void NAME(device const T* x, device const long* target, \
                 device const T* is_target, device const T* go, device T* grad, \
                 constant uint& nframe, constant uint& dim, \
                 constant float& base_scale, constant uint& per_row, \
                 uint row [[threadgroup_position_in_grid]], \
                 uint tid [[thread_index_in_threadgroup]], \
                 uint group [[threads_per_threadgroup]]) { \
    threadgroup float shared[256]; \
    threadgroup float staged[ML_STAGED]; \
    threadgroup uint end_slot; \
    ml_backward_impl(x, target, is_target, go, grad, dim, base_scale, per_row, \
                     row, tid, group, shared, staged, &end_slot); \
}

#define ROW_REDUCE_KERNEL(NAME, T) \
kernel void NAME(device const T* partial, device T* output, \
                 constant uint& count, constant uint& divides, \
                 uint tid [[thread_index_in_threadgroup]], \
                 uint group [[threads_per_threadgroup]]) { \
    threadgroup float shared[256]; \
    float local = 0.0f; \
    for (uint i = tid; i < count; i += group) local += ld(partial, i); \
    shared[tid] = local; \
    float total = group_sum(shared, tid, group); \
    if (tid == 0) st(output, 0, divides ? total / float(count) : total); \
}

MM_FWD_KERNEL(mm_forward, float)
MM_FWD_KERNEL(mm_forward_h, half)
MM_FWD_KERNEL(mm_forward_b, ushort)

MM_FWD_ROWS_KERNEL(mm_forward_rows, float)
MM_FWD_ROWS_KERNEL(mm_forward_rows_h, half)
MM_FWD_ROWS_KERNEL(mm_forward_rows_b, ushort)

MM_TOTAL_KERNEL(mm_total, float)
MM_TOTAL_KERNEL(mm_total_h, half)
MM_TOTAL_KERNEL(mm_total_b, ushort)

MM_BWD_KERNEL(mm_backward, float)
MM_BWD_KERNEL(mm_backward_h, half)
MM_BWD_KERNEL(mm_backward_b, ushort)

MM_BWD_ROWS_KERNEL(mm_backward_rows, float)
MM_BWD_ROWS_KERNEL(mm_backward_rows_h, half)
MM_BWD_ROWS_KERNEL(mm_backward_rows_b, ushort)

ML_FWD_KERNEL(ml_forward, float)
ML_FWD_KERNEL(ml_forward_h, half)
ML_FWD_KERNEL(ml_forward_b, ushort)

ML_TOTAL_KERNEL(ml_total, float)
ML_TOTAL_KERNEL(ml_total_h, half)
ML_TOTAL_KERNEL(ml_total_b, ushort)

ML_BWD_KERNEL(ml_backward, float)
ML_BWD_KERNEL(ml_backward_h, half)
ML_BWD_KERNEL(ml_backward_b, ushort)

ROW_REDUCE_KERNEL(row_reduce, float)
ROW_REDUCE_KERNEL(row_reduce_h, half)
ROW_REDUCE_KERNEL(row_reduce_b, ushort)
"""

OVERLAYS = (
    ("multi_margin_loss", multi_margin_loss),
    ("multi_margin_loss.out", multi_margin_loss_out),
    ("multi_margin_loss_backward", multi_margin_loss_backward),
    ("multi_margin_loss_backward.grad_input", multi_margin_loss_backward_out),
    ("multilabel_margin_loss_forward", multilabel_margin_loss_forward),
    ("multilabel_margin_loss_forward.output", multilabel_margin_loss_forward_out),
    ("multilabel_margin_loss_backward", multilabel_margin_loss_backward),
    ("multilabel_margin_loss_backward.grad_input", multilabel_margin_loss_backward_out),
)
