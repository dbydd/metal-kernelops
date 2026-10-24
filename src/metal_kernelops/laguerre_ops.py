"""MPS overlay for ``aten::special_laguerre_polynomial_l``.

Six aten overloads are covered, all of which raise ``NotImplementedError`` on
MPS in torch 2.14.1::

    special_laguerre_polynomial_l(Tensor x, Tensor n)
    special_laguerre_polynomial_l.out(Tensor x, Tensor n, *, Tensor(a!) out)
    special_laguerre_polynomial_l.x_scalar(Scalar x, Tensor n)
    special_laguerre_polynomial_l.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out)
    special_laguerre_polynomial_l.n_scalar(Tensor x, Scalar n)
    special_laguerre_polynomial_l.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out)

Importing this module compiles nothing and registers nothing. ``OVERLAYS``
holds ``(overload name, callable)`` pairs for whoever installs them on the MPS
dispatch key.

Numerics
--------
ATen's CPU kernel (``laguerre_polynomial_l_forward`` in
``ATen/native/Math.h``, PyTorch BSD-3-Clause) evaluates

    L_0 = 1,  L_1 = 1 - x,  (k+1) L_{k+1} = (2k + 1 - x) L_k - k L_{k-1}

with an early ``L = 1`` for ``|x| == 0``. PyTorch's CPU build contracts
``s * q - k * p`` into a single fma; ``fmaf(s, q, -(k * p))`` reproduces the CPU
result bit for bit at every degree tested, while the unfused form is off by up
to ~1000 ulp at degree 8. The Metal kernel therefore calls ``fma`` explicitly
and keeps the rounding of ``s`` and ``k * p`` separate, which is what makes the
MPS result bit-identical to CPU for float32.

The ``/ (k + 1)`` step stays a correctly rounded quotient, but not as a hardware
divide: ``k + 1`` takes one of ``table_size`` integer values, so the kernel
multiplies by a correctly rounded reciprocal from a table and recovers the exact
quotient with one fma correction (Markstein's algorithm: ``q0 = num * r``,
``e = fma(-q0, d, num)``, ``q = fma(e, r, q0)``). A float divide expands to a
reciprocal approximation plus several refinement steps, and the recurrence
serialises on it, so the correction trades that sequence for three full-rate
operations with an identical result. The residual is dropped when the numerator
is not finite, where the quotient is the numerator's own infinity or nan and the
residual would be ``inf - inf``. Degrees above ``table_size`` keep the hardware
divide.

Degree semantics, matching the CPU kernel:

- ``n`` is truncated toward zero before use, so ``2.9 -> 2``, ``-0.5 -> 0``
  (hence ``L = 1``), ``-2.9 -> -2`` (hence ``L = 0``).
- ``n < 0`` gives 0, and that test comes before the ``|x| == 0`` shortcut.
- ``x == 0`` gives 1 for every ``n >= 0``, without touching the recurrence;
  ``L_n(0) = 1`` holds exactly.
- ``x = ±inf`` and ``x = nan`` propagate through the recurrence exactly as on
  CPU (``n = 1`` yields ``∓inf``, ``n >= 3`` yields nan from ``-inf + inf``).

Two deviations, both at inputs where the CPU path has no defined answer:

- A non-finite degree on CPU is undefined behaviour: ``static_cast<int64_t>``
  saturates and the loop runs ~9.2e18 iterations, so the process hangs. Here
  NaN maps to 0, ``-inf`` to a negative degree (result 0) and ``+inf`` to the
  degree cap.
- ``MAX_DEGREE`` caps the loop, so a stray infinite or 2**60 degree cannot hang
  the GPU the way the CPU kernel hangs on a non-finite degree. A degree at or
  above the cap returns nan. That is a real deviation only for ``x > 0``, where
  ``L_n(x)`` stays bounded instead of overflowing, so a finite answer at
  ``n >= 2**24`` becomes nan here.

The CPU and CUDA kernels have no Half or BFloat16 implementation
(``NotImplementedError``). This overlay computes in float32 and stores the
promoted dtype, so fp16/bf16 models get the operator at all.

Performance
-----------
Four elements per thread through one vector load and store, with the four
recurrences interleaved so one lane's divide latency hides behind the other
three's arithmetic. Degree 0 never reads ``x``, because the answer cannot depend
on it, so that case moves half the bytes. Measured on an M5 Max against the
previous one-element-per-thread kernel, both in one interleaved batch, the
minimum of 30 rounds: at 16.7M elements, degree 0 goes 0.306 ms -> 0.135 ms
(2.3x, 496 GB/s moved, equal to the device's write-only floor of 476 GB/s),
degree 8 goes 1.61x, degree 32 2.6x and degree 128 2.0x (8.51 ms -> 4.24 ms,
31.6 GB/s); at 4.19M elements the same degrees gain 1.6x, 1.4x, 2.0x and 2.0x.
Degrees 1 to 4 at 16.7M elements are unchanged because both kernels already run
at the copy floor (0.306 ms for 134 MB, 438 GB/s): that case is memory bound and
no rewrite of the arithmetic can move it.

Autograd
--------
``special_laguerre_polynomial_l`` carries no autograd formula in torch 2.14.1:
on CPU ``x.requires_grad = True`` still yields a result with no ``grad_fn``.
Registration here only supplies the MPS kernel, so that behaviour is preserved.
"""

import math
from functools import lru_cache

import torch

MAX_DEGREE = 1 << 24
"""Largest degree the kernel evaluates. At or above it the result is nan."""

_NO_DEGREE = -(1 << 31)
"""Sentinel telling the kernel to read a degree per element."""

_THREADS = 256

_LANES = 4
"""Elements per thread in the vectorised path; the shader is written for four."""

_TABLE = 4096
"""Count of tabulated ``1 / (k + 1)``; higher degrees keep the hardware divide.

The table is read sequentially, one entry per iteration, so 16 KiB costs no more
than 4 KiB: at degree 2000 it is 1.37x the hardware divide, and at degree 128 a
4096-entry and a 1024-entry table measure the same within run-to-run noise.
"""

_TYPES = {
    torch.float32: "float",
    torch.float16: "half",
    torch.bfloat16: "bfloat",
}

_VECTORS = {
    "float": "float4",
    "half": "half4",
    "bfloat": "bfloat4",
}

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

constant int max_degree = $MAXD;
constant int no_degree = $NODEG;
constant int table_size = $TABSZ;

static inline int laguerre_degree(
    device const float* degree_buffer,
    int degree_scalar,
    uint index,
    uint single) {
    if (degree_scalar != no_degree) {
        return degree_scalar;
    }
    float raw = degree_buffer[single != 0 ? 0 : index];
    if (isnan(raw)) return 0;
    if (raw >= 2147483648.0f) return max_degree;
    if (raw <= -2147483648.0f) return -1;
    return int(raw);
}

static inline float laguerre_value(float xv, int degree, device const float* recip) {
    if (degree < 0) {
        return 0.0f;
    }
    if (xv == 0.0f || degree == 0) {
        return 1.0f;
    }
    if (degree >= max_degree) {
        return NAN;
    }
    float t = 1.0f - xv;
    float p = 1.0f;
    float q = t;
    float r = q;
    float kf = 1.0f;
    float d = 2.0f;
    int k = 1;
    int limit = degree <= table_size ? degree : table_size + 1;
    for (; k < limit; ++k) {
        float num = fma(kf + kf + t, q, -(kf * p));
        float iv = recip[k - 1];
        float q0 = num * iv;
        float e = fma(-q0, d, num);
        e = isfinite(num) ? e : 0.0f;
        r = fma(e, iv, q0);
        p = q;
        q = r;
        kf += 1.0f;
        d += 1.0f;
    }
    for (; k < degree; ++k) {
        r = fma(kf + kf + t, q, -(kf * p)) / d;
        p = q;
        q = r;
        kf += 1.0f;
        d += 1.0f;
    }
    return r;
}

kernel void laguerre(
    device const $T* x,
    device $T* out,
    device const float* degree_buffer,
    device const float* recip,
    constant int& degree_scalar,
    constant uint& count,
    constant uint& x_single,
    constant uint& degree_single,
    constant uint& vectorized,
    uint gid [[thread_position_in_grid]]) {

    if (vectorized != 0) {
        uint i = gid * 4;
        if (i + 4 <= count) {
            if (degree_scalar == 0) {
                *(device $V*)(out + i) = $V($T(1.0f), $T(1.0f), $T(1.0f), $T(1.0f));
                return;
            }
            $V raw;
            if (x_single != 0) {
                float v = float(x[0]);
                raw = $V($T(v), $T(v), $T(v), $T(v));
            } else {
                raw = *(device const $V*)(x + i);
            }
            float xv[4] = {float(raw.x), float(raw.y), float(raw.z), float(raw.w)};
            int d0 = laguerre_degree(degree_buffer, degree_scalar, i, degree_single);
            int d1 = laguerre_degree(degree_buffer, degree_scalar, i + 1, degree_single);
            int d2 = laguerre_degree(degree_buffer, degree_scalar, i + 2, degree_single);
            int d3 = laguerre_degree(degree_buffer, degree_scalar, i + 3, degree_single);
            float r0 = laguerre_value(xv[0], d0, recip);
            float r1 = laguerre_value(xv[1], d1, recip);
            float r2 = laguerre_value(xv[2], d2, recip);
            float r3 = laguerre_value(xv[3], d3, recip);
            *(device $V*)(out + i) = $V($T(r0), $T(r1), $T(r2), $T(r3));
            return;
        }
        if (degree_scalar == 0) {
            for (uint j = i; j < count; ++j) out[j] = $T(1.0f);
            return;
        }
        for (uint j = i; j < count; ++j) {
            out[j] = $T(laguerre_value(
                float(x[x_single != 0 ? 0 : j]),
                laguerre_degree(degree_buffer, degree_scalar, j, degree_single),
                recip));
        }
        return;
    }
    if (gid >= count) return;
    if (degree_scalar == 0) {
        out[gid] = $T(1.0f);
        return;
    }
    out[gid] = $T(laguerre_value(
        float(x[x_single != 0 ? 0 : gid]),
        laguerre_degree(degree_buffer, degree_scalar, gid, degree_single),
        recip));
}
"""


@lru_cache(maxsize=4)
def _library(type_name):
    source = (
        _SOURCE.replace("$TABSZ", str(_TABLE))
        .replace("$MAXD", str(MAX_DEGREE))
        .replace("$NODEG", str(_NO_DEGREE))
        .replace("$V", _VECTORS[type_name])
        .replace("$T", type_name)
    )

    return torch.mps.compile_shader(source)


_dummies = {}

_DTYPE_LABELS = {
    torch.float32: "Float",
    torch.float64: "Double",
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.int64: "Long",
    torch.int32: "Int",
    torch.int16: "Short",
    torch.int8: "Char",
    torch.uint8: "Byte",
    torch.bool: "Bool",
}


def _dummy(device):
    """One float32 element, for the unused degree pointer."""
    tensor = _dummies.get(device)
    if tensor is None:
        tensor = _dummies[device] = torch.zeros(1, dtype=torch.float32, device=device)
    return tensor


_recips = {}


def _reciprocals(device):
    """Correctly rounded ``1 / (k + 1)`` for the Markstein correction."""
    tensor = _recips.get(device)
    if tensor is None:
        # Built on the CPU so the table is IEEE-correctly-rounded whatever the
        # MPS reciprocal approximation does.
        tensor = _recips[device] = torch.reciprocal(
            torch.arange(2, _TABLE + 2, dtype=torch.float32)
        ).to(device)
    return tensor


def _result_dtype(x, n):
    if isinstance(n, torch.Tensor):
        dtype = torch.promote_types(x.dtype, n.dtype)
    else:
        dtype = x.dtype if x.dtype.is_floating_point else torch.get_default_dtype()
    if not dtype.is_floating_point:
        dtype = torch.get_default_dtype()
    return dtype


def _uniform_degree(n):
    """Python scalar degree -> the integer the CPU cast produces, clamped."""
    if isinstance(n, float):
        if math.isnan(n):
            return 0
        if math.isinf(n):
            return MAX_DEGREE if n > 0 else -1
        n = int(n)
    else:
        n = int(n)
    return -1 if n < 0 else min(n, MAX_DEGREE)


def _apply(x, n, out):
    if x.device.type != "mps":
        raise RuntimeError(f"laguerre overlay needs MPS tensors, got {x.device}")
    if isinstance(n, torch.Tensor):
        if n.device.type != "mps":
            raise RuntimeError(
                f"expected x and n on the same device, got {x.device} and {n.device}"
            )
        shape = torch.broadcast_shapes(x.shape, n.shape)
    else:
        shape = x.shape

    dtype = _result_dtype(x, n)
    if out is not None:
        if out.device.type != "mps":
            raise RuntimeError(f"laguerre overlay needs MPS tensors, got out on {out.device}")
        if not torch.can_cast(dtype, out.dtype):
            raise RuntimeError(
                f"result type {_DTYPE_LABELS[dtype]} can't be cast to the desired output "
                f"type {_DTYPE_LABELS[out.dtype]}"
            )
    if 0 in shape:
        result = torch.empty(shape, dtype=dtype, device=x.device)
        if out is None:
            return result
        out.resize_(shape)
        out.copy_(result)
        return out

    count = 1
    for size in shape:
        count *= size

    direct = (
        out is not None
        and out.device.type == "mps"
        and out.dtype == dtype
        and out.shape == shape
        and out.is_contiguous()
    )
    target = out if direct else torch.empty(shape, dtype=dtype, device=x.device)

    x_single = x.numel() == 1
    xb = x.to(dtype=dtype) if x.dtype != dtype else x
    if x_single:
        xb = xb.contiguous()
    else:
        if xb.shape != shape:
            xb = xb.expand(shape)
        if not xb.is_contiguous():
            xb = xb.contiguous()

    if isinstance(n, torch.Tensor):
        nb = n if n.dtype == torch.float32 else n.to(torch.float32)
        if not nb.is_contiguous():
            nb = nb.contiguous()
        if n.numel() > 1:
            if nb.shape != shape:
                nb = nb.expand(shape)
            if not nb.is_contiguous():
                nb = nb.contiguous()
            degree, degree_single = _NO_DEGREE, 0
        else:
            degree, degree_single = _NO_DEGREE, 1
    else:
        nb = _dummy(x.device)
        degree, degree_single = _uniform_degree(n), 0

    library = _library(_TYPES[dtype])
    recip = _reciprocals(x.device)
    align = dtype.itemsize * _LANES
    vectorized = (
        count >= _LANES
        and (x_single or xb.data_ptr() % align == 0)
        and target.data_ptr() % align == 0
    )
    library.laguerre(
        xb,
        target,
        nb,
        recip,
        degree,
        count,
        1 if x_single else 0,
        degree_single,
        1 if vectorized else 0,
        threads=(count + _LANES - 1) // _LANES if vectorized else count,
        group_size=_THREADS,
    )

    if out is None or direct:
        return target
    out.resize_(shape)
    out.copy_(target)
    return out


def laguerre(x, n):
    return _apply(x, n, None)


def laguerre_out(x, n, out):
    return _apply(x, n, out)


def _scalar_tensor(value, like):
    dtype = like.dtype if like.dtype.is_floating_point else torch.get_default_dtype()
    return torch.tensor(value, dtype=dtype, device=like.device)


def laguerre_x_scalar(x, n):
    return _apply(_scalar_tensor(x, n), n, None)


def laguerre_x_scalar_out(x, n, out):
    return _apply(_scalar_tensor(x, n), n, out)


def laguerre_n_scalar(x, n):
    return _apply(x, n, None)


def laguerre_n_scalar_out(x, n, out):
    return _apply(x, n, out)


OVERLAYS = (
    ("special_laguerre_polynomial_l", laguerre),
    ("special_laguerre_polynomial_l.out", laguerre_out),
    ("special_laguerre_polynomial_l.x_scalar", laguerre_x_scalar),
    ("special_laguerre_polynomial_l.x_scalar_out", laguerre_x_scalar_out),
    ("special_laguerre_polynomial_l.n_scalar", laguerre_n_scalar),
    ("special_laguerre_polynomial_l.n_scalar_out", laguerre_n_scalar_out),
)
