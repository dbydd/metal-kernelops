"""``aten::special_legendre_polynomial_p`` on the MPS dispatch key.

The operator has a CPU kernel and no MPS kernel in torch 2.14.1: every call on
an MPS tensor raises ``NotImplementedError``. This module supplies a Metal
kernel and the overlay table :data:`OVERLAYS` for it. Importing it registers
nothing and compiles nothing; the shader is compiled on the first call.

Algorithm
---------
Ported from ``legendre_polynomial_p_forward`` in
``aten/src/ATen/native/Math.h``, PyTorch, BSD-3-Clause (Copyright (c)
2016-present, Facebook, Inc; see the PyTorch ``LICENSE``). The three-term
recurrence is unchanged::

    P_0 = 1, P_1 = x
    P_{k+1} = ((2k + 1) * x * P_k - k * P_{k-1}) / (k + 1)

with the upstream guards: a negative degree returns 0, ``|x| == 1`` returns
``(-1)**n``, and the loop stops early once the iterate turns NaN.

Bit-exactness
-------------
The CPU kernel is compiled so that the recurrence's subtraction contracts into
a fused multiply-add: ``num = fma((2k+1) * x, q, -(k * p))``. Writing that
fusion explicitly here is what makes the MPS result bit-identical to the CPU
result; a plain ``a * b - c`` in MSL contracts differently and differs in the
last bit on roughly 40% of inputs. Verified for degrees 0 to 100 over dense
samples in ``benchmarks/verify_legendre.py``.

Semantics preserved from upstream
---------------------------------
* The degree is cast to ``int64`` per element, so a non-integer degree
  truncates toward zero (``2.7`` is ``P_2``) and a degree in ``(-1, 0]`` is
  ``P_0``.
* A NaN degree casts to 0 on both CPU (arm64) and Apple GPUs, so it yields 1.
* ``+inf`` and any degree at or above about ``1e12`` do not terminate, exactly
  as upstream: the recurrence runs ``n`` times and ``|x| <= 1`` never turns
  NaN. ``-inf`` casts to ``INT64_MIN`` and returns 0.
* ``x`` and ``n`` broadcast; the recurrence runs per element, so one call may
  mix degrees.
* float32 only. float16 and bfloat16 raise, as they do on the CPU kernel;
  float64 cannot exist on MPS. Integer and bool inputs promote to float32;
  complex inputs and complex scalar operands raise as they do on the CPU.
* The operator is not differentiable upstream (``requires_grad`` is False and
  there is no ``grad_fn``), so the overlay returns a plain tensor.
* ``.out`` follows the structured CPU kernel: the output is resized to the
  broadcast shape, the result is cast into the output dtype, and a cast that
  cannot hold the result raises the same ``RuntimeError`` as TensorIterator.

Performance
-----------
One dispatch, one pass: a thread evaluates four elements' recurrences at once
in registers. The vector width is the whole low-degree win, because those
calls are pure memory traffic and a ``float4`` load and store move 16 bytes
each instead of 4. It is also why the 32-bit loop counter matters: a 64-bit
one costs several instructions per step on Apple GPUs, and a degree past
``2**23`` falls back to the upstream 64-bit loop, where the run is
astronomically long anyway.

Measured on this machine, interleaved, against the one-element-per-thread
kernel this replaces: a 4M-element call is 4.5x faster at degree 4 and 5.3x at
degree 32, and a 4096-element call drops from 9.2 us to 6.9 us. Splitting the
work over eight ``float4`` chains per thread, to give the recurrence more
independent divisions in flight, was measured too and is slower at every
degree and size tested, by 1.4x to 5x: the copy-in and the eight-way
register pressure cost more than the divisions they expose.

A scalar operand arrives as a kernel constant rather than as a one-element
device tensor, because building that tensor costs more than the launch. A
one-element tensor operand is read once and splatted by the kernel; any other
operand that does not already hold one value per output element is
materialized once by ``expand().contiguous()``. No call reads operand values
on the host.
"""

import torch

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// Degrees at or below this use the 32-bit loop. `2 * k + 1` and `k + 1` stay
// exactly representable in float below 2**24, so the fast path rounds exactly
// as the 64-bit loop it replaces.
#define LEGENDRE_FAST_MAX 8388608

// A plain (long) conversion is 0 for NaN and out-of-range values on Apple
// GPUs, where the arm64 CPU saturates. Saturate explicitly so a NaN degree is
// 0 and -inf is INT64_MIN, as on the CPU.
inline long legendre_degree(float nv) {
    if (isnan(nv)) {
        return 0;
    } else if (nv >= 9223372036854775808.0f) {
        return 9223372036854775807L;
    } else if (nv < -9223372036854775808.0f) {
        return -9223372036854775807L - 1;
    }
    return (long)nv;
}

// One element, upstream semantics. `fma(t, q, -(k * p))` is the contraction
// the CPU kernel is built with; without it the last bit differs.
inline float legendre_one(float xv, long degree) {
    if (degree < 0) {
        return 0.0f;
    }
    if (fabs(xv) == 1.0f) {
        return (xv > 0.0f || degree % 2 == 0) ? 1.0f : -1.0f;
    }
    if (degree == 0) {
        return 1.0f;
    }
    if (degree == 1) {
        return xv;
    }
    float p = 1.0f, q = xv, r = xv;
    if (degree <= LEGENDRE_FAST_MAX) {
        const int d = (int)degree;
        for (int k = 1; k < d && !isnan(q); ++k) {
            float t = (float)(k + k + 1) * xv;
            r = fma(t, q, -((float)k * p)) / (float)(k + 1);
            p = q;
            q = r;
        }
        return r;
    }
    for (long k = 1; k < degree && !isnan(q); ++k) {
        float t = (float)(k + k + 1) * xv;
        r = fma(t, q, -((float)k * p)) / (float)(k + 1);
        p = q;
        q = r;
    }
    return r;
}

// The degree shared by every component of `v`, or -1 when they differ, when
// any is negative, or when any leaves the 32-bit loop.
inline int legendre_common_degree(float4 v) {
    const long d = legendre_degree(v.x);
    if (legendre_degree(v.y) != d || legendre_degree(v.z) != d ||
        legendre_degree(v.w) != d || d < 0 || d > LEGENDRE_FAST_MAX) {
        return -1;
    }
    return (int)d;
}

// The `|x| == 1` short-circuit is not a recurrence step, so a group holding one
// takes the scalar path.
inline bool legendre_has_unit(float4 a, float4 b) {
    return fabs(a.x) == 1.0f || fabs(a.y) == 1.0f || fabs(a.z) == 1.0f ||
           fabs(a.w) == 1.0f || fabs(b.x) == 1.0f || fabs(b.y) == 1.0f ||
           fabs(b.z) == 1.0f || fabs(b.w) == 1.0f;
}

// An operand count of 0 selects the host scalar in `x_value` / `n_value`; a
// count of 1 reads element 0 and broadcasts it. Every full operand must be
// 16-byte aligned, which is the caller's job.
kernel void legendre8(
    device const float* x, device const float* n, device float* out,
    constant uint& count, constant uint& x_count, constant uint& n_count,
    constant float& x_value, constant float& n_value,
    uint i [[thread_position_in_grid]],
    uint stride [[threads_per_grid]]) {
    // One element per thread, a float4 at a time. This is the path a low
    // degree takes, where the call is memory-bound and the vector width is all
    // that matters; it costs one recurrence in flight, which costs nothing
    // while the loop is short.
    const uint groups = count >> 2;
    const uint tail = count & 3;
    const uint me = i;
    device const float4* x4 = (device const float4*)x;
    device const float4* n4 = (device const float4*)n;
    device float4* out4 = (device float4*)out;
    for (; i < groups; i += stride) {
        float4 xa;
        if (x_count == 0) {
            xa = float4(x_value);
        } else if (x_count == 1) {
            xa = float4(x[0]);
        } else {
            xa = x4[i];
        }
        float4 na;
        if (n_count == 0) {
            na = float4(n_value);
        } else if (n_count == 1) {
            na = float4(n[0]);
        } else {
            na = n4[i];
        }
        const int d = legendre_common_degree(na);
        if (d >= 2 && !legendre_has_unit(xa, xa)) {
            float4 pa = float4(1.0f), qa = xa, ra = xa;
            for (int k = 1; k < d; ++k) {
                const float t = (float)(k + k + 1);
                ra = fma(t * xa, qa, -(float)k * pa) / (float)(k + 1);
                pa = qa;
                qa = ra;
            }
            out4[i] = ra;
        } else {
            out4[i] = float4(
                legendre_one(xa.x, legendre_degree(na.x)),
                legendre_one(xa.y, legendre_degree(na.y)),
                legendre_one(xa.z, legendre_degree(na.z)),
                legendre_one(xa.w, legendre_degree(na.w)));
        }
    }
    // One thread takes the one to three elements that do not fill a vector.
    if (tail != 0 && me == 0) {
        for (uint j = count - tail; j < count; ++j) {
            float xv = x_count == 0 ? x_value : x[x_count == 1 ? 0 : j];
            float nv = n_count == 0 ? n_value : n[n_count == 1 ? 0 : j];
            out[j] = legendre_one(xv, legendre_degree(nv));
        }
    }
}

// One element per thread, for operands the vector kernel cannot take: a
// pointer that is not 16-byte aligned, which a sliced view can produce.
kernel void legendre1(
    device const float* x, device const float* n, device float* out,
    constant uint& count, constant uint& x_count, constant uint& n_count,
    constant float& x_value, constant float& n_value,
    uint i [[thread_position_in_grid]],
    uint stride [[threads_per_grid]]) {
    for (; i < count; i += stride) {
        float xv = x_count == 0 ? x_value : x[x_count == 1 ? 0 : i];
        float nv = n_count == 0 ? n_value : n[n_count == 1 ? 0 : i];
        out[i] = legendre_one(xv, legendre_degree(nv));
    }
}
"""

_MAX_THREADS = 1 << 20
"""Threads per dispatch. Each thread strides, so this bounds launch cost, not work."""

_GROUP = 64
"""Threads per threadgroup. 64 and 128 measured within noise of each other."""

_CPP_NAME = {
    torch.bool: "Bool",
    torch.uint8: "Byte",
    torch.int8: "Char",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.float32: "Float",
    torch.float64: "Double",
    torch.complex64: "ComplexFloat",
    torch.complex128: "ComplexDouble",
}


_VECTOR = None
"""``(kernel8, kernel1)`` once the shader is compiled; ``None`` before."""


def _kernels():
    """The two kernel entry points, compiled on first use."""
    global _VECTOR
    if _VECTOR is None:
        module = torch.mps.compile_shader(_SOURCE)
        _VECTOR = (module.legendre8, module.legendre1)
    return _VECTOR


def _result_dtype(x, n, x_value=0.0, n_value=0.0):
    """The dtype the CPU kernel would compute in, for an MPS-legal input pair.

    A ``None`` operand is a host scalar, which upstream treats as weak: it takes
    the dtype of the tensor operand instead of promoting it.
    """
    if x is None or n is None:
        scalar, tensor = (x_value, n) if x is None else (n_value, x)
        if isinstance(scalar, complex):
            dtype = tensor.dtype if tensor.dtype.is_complex else torch.complex64
        else:
            dtype = tensor.dtype
    else:
        dtype = torch.result_type(x, n)
    if dtype.is_complex:
        raise NotImplementedError(
            f'"legendre_polynomial_p_mps" not implemented for \'{_CPP_NAME[dtype]}\''
        )
    if dtype.is_floating_point and dtype.itemsize < 4:
        raise NotImplementedError(
            f'"legendre_polynomial_p_mps" not implemented for \'{_CPP_NAME[dtype]}\''
        )
    return dtype if dtype.is_floating_point else torch.float32


def _can_hold(dtype, target):
    """Whether TensorIterator's safe-cast check lets `dtype` land in `target`."""
    return target == dtype or target.is_floating_point or target.is_complex


def _operand(tensor, dtype, shape, count, filler):
    """`(tensor, count)` for the kernel, without copying when it is already there.

    The count is what the kernel indexes by: 0 selects the host scalar, 1 reads
    element 0 and broadcasts it, and the element count means one value per
    output element. `filler` is passed in place of an absent operand, which the
    kernel never reads.
    """
    if tensor is None:
        return filler, 0
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    numel = tensor.numel()
    if numel == count:
        return (tensor if tensor.is_contiguous() else tensor.contiguous().view(-1)), count
    if numel == 1:
        return tensor, 1
    return tensor.expand(shape).contiguous().view(-1), count


def _demand_device(*tensors):
    tensors = [tensor for tensor in tensors if tensor is not None]
    for tensor in tensors:
        if tensor.device.type != "mps":
            others = sorted({t.device for t in tensors if t.device.type != "mps"})
            raise RuntimeError(
                "Expected all tensors to be on the same device, but found at "
                f"least two devices, mps:0 and {others[0]}!"
            )


def _fill(x, n, out, x_value=0.0, n_value=0.0):
    """Write the polynomial of `x` at degree `n` into the contiguous `out`.

    A ``None`` operand is taken from the host scalar ``x_value``/``n_value``.
    """
    count = out.numel()
    if count == 0:
        return
    dtype = out.dtype
    shape = out.shape
    flat = out.view(-1)
    xv, x_count = _operand(x, dtype, shape, count, flat)
    nv, n_count = _operand(n, dtype, shape, count, flat)
    threads = -(-min(count, _MAX_THREADS) // _GROUP) * _GROUP
    vector, scalar = _kernels()
    args = (xv, nv, flat, count, x_count, n_count, float(x_value), float(n_value))
    if xv.data_ptr() % 16 or nv.data_ptr() % 16 or flat.data_ptr() % 16:
        # The vector kernel reads and writes `float4`, so every full operand
        # needs a 16-byte-aligned base, which a slice can break.
        scalar(*args, threads=threads, group_size=_GROUP)
        return
    vector(*args, threads=threads, group_size=_GROUP)


def _shares_memory(a, b):
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def _shape(x, n):
    if x is None:
        return n.shape
    if n is None or x.shape == n.shape:
        return x.shape
    return torch.broadcast_shapes(x.shape, n.shape)


def _device(x, n):
    return (x if n is None else n).device


def _run(x, n, x_value=0.0, n_value=0.0):
    _demand_device(x, n)
    dtype = _result_dtype(x, n, x_value, n_value)
    out = torch.empty(_shape(x, n), dtype=dtype, device=_device(x, n))
    _fill(x, n, out, x_value, n_value)
    return out


def _run_out(x, n, out, x_value=0.0, n_value=0.0):
    _demand_device(x, n, out)
    dtype = _result_dtype(x, n, x_value, n_value)
    if not _can_hold(dtype, out.dtype):
        raise RuntimeError(
            f"result type {_CPP_NAME[dtype]} can't be cast to the desired "
            f"output type {_CPP_NAME[out.dtype]}"
        )
    shape = _shape(x, n)
    if out.shape != shape:
        out.resize_(shape)
    if (
        out.dtype == dtype
        and out.is_contiguous()
        and not any(_shares_memory(out, t) for t in (x, n) if t is not None)
    ):
        _fill(x, n, out, x_value, n_value)
        return out
    filled = torch.empty(shape, dtype=dtype, device=out.device)
    _fill(x, n, filled, x_value, n_value)
    out.copy_(filled)
    return out


def polynomial(x, n):
    """``aten::special_legendre_polynomial_p(Tensor x, Tensor n)``."""
    return _run(x, n)


def polynomial_out(x, n, out):
    """``aten::special_legendre_polynomial_p.out(Tensor x, Tensor n, *, Tensor out)``."""
    return _run_out(x, n, out)


def x_scalar(x, n):
    """``aten::special_legendre_polynomial_p.x_scalar(Scalar x, Tensor n)``."""
    return _run(None, n, x_value=x)


def x_scalar_out(x, n, out):
    """``aten::special_legendre_polynomial_p.x_scalar_out(Scalar x, Tensor n, *, Tensor out)``."""
    return _run_out(None, n, out, x_value=x)


def n_scalar(x, n):
    """``aten::special_legendre_polynomial_p.n_scalar(Tensor x, Scalar n)``."""
    return _run(x, None, n_value=n)


def n_scalar_out(x, n, out):
    """``aten::special_legendre_polynomial_p.n_scalar_out(Tensor x, Scalar n, *, Tensor out)``."""
    return _run_out(x, None, out, n_value=n)


OVERLAYS = (
    ("special_legendre_polynomial_p", polynomial),
    ("special_legendre_polynomial_p.out", polynomial_out),
    ("special_legendre_polynomial_p.x_scalar", x_scalar),
    ("special_legendre_polynomial_p.x_scalar_out", x_scalar_out),
    ("special_legendre_polynomial_p.n_scalar", n_scalar),
    ("special_legendre_polynomial_p.n_scalar_out", n_scalar_out),
)
"""``(aten overload, kernel)`` for ``Library("aten", "FRAGMENT").impl(name, fn, "MPS")``."""
