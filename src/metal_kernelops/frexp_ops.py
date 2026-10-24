"""MPS ``frexp`` overlay: one Metal pass over the raw float bits.

``aten::frexp.Tensor`` is a composite that reaches ``aten::frexp.Tensor_out``,
and neither has an MPS kernel in torch 2.14.1: ``torch.frexp`` on an MPS tensor
raises ``NotImplementedError``. This module registers both overloads for MPS.

The kernel is exact by construction. It reads the sign, exponent field and
fraction of the input, and writes:

* normal: mantissa has exponent field ``bias - 1`` and the same fraction, so
  ``|mantissa|`` lands in ``[0.5, 1)``; exponent is ``ef - bias + 1``;
* subnormal: the fraction is shifted left until its top bit is set, giving the
  same ``[0.5, 1)`` mantissa and ``exponent = 1 - bias - shift``;
* zero: mantissa is a signed zero, exponent 0;
* inf / NaN: mantissa is the input unchanged (payload preserved), exponent 0.

That matches the CPU kernel bit for bit on float32, signed zeros included. On
float16 and bfloat16 the CPU kernel converts through float32 and so
canonicalises NaN payloads; this kernel returns the input NaN bits instead.
Either way the result is a NaN, which is all the operator promises. The
exponent is always ``int32``.

Gradients are not computed here. ``frexp`` has a generated autograd rule above
the backend key (``FrexpBackward0``, gradient ``grad / 2**exponent``), so an
overlay that only replaces the forward inherits it unchanged. That rule is
bit-identical to CPU everywhere except where Metal's flush-to-zero meets
``exp2(127)`` and ``exp2(-127)``: exponent 127 yields 0 instead of the
subnormal ``2**-127``, and exponent ``<= -127`` yields ``inf`` instead of a
finite quotient. In float32 that needs ``|x| >= 2**126`` or a subnormal input
below ``2**-126``; float16 never reaches it.

Importing this module compiles nothing and registers nothing; ``OVERLAYS`` is
the table the accelerator installs, and each shader compiles on first call.

Traffic is fixed by the operator contract: ``n`` input elements read, ``n``
mantissa elements written, ``n`` int32 exponents written. That is 12 bytes per
element for float32 and 8 for the 16-bit dtypes, so the floor is the write
bandwidth, not the arithmetic. ``frexp_bits4`` issues those accesses four
elements at a time (``uint4`` / ``ushort4`` loads, matching stores, ``int4``
exponent stores) instead of one per thread. Packing the mantissa and exponent
into a single interleaved buffer would halve the number of store instructions
but not the bytes, and it would break the ``out=`` contract that the two
outputs are separate tensors, so the shader keeps them separate.
"""

from functools import lru_cache

import torch

# dtype -> (mantissa bits, exponent bits, bias)
_PARAMS = {
    torch.float32: (23, 8, 127),
    torch.float16: (10, 5, 15),
    torch.bfloat16: (7, 8, 127),
}


def _raw_dtype(dtype):
    return torch.int32 if dtype == torch.float32 else torch.int16


_DTYPE_NAMES = {
    torch.float32: "float",
    torch.float16: "c10::Half",
    torch.bfloat16: "c10::BFloat16",
    torch.int64: "long long",
    torch.int32: "int",
    torch.int16: "short",
    torch.int8: "char",
    torch.bool: "bool",
}


def _dtype_name(dtype):
    return _DTYPE_NAMES.get(dtype, str(dtype))


# Bytes a single ``frexp_bits4`` access covers: uint4 for float32, ushort4 for
# the 16-bit dtypes.
_VECTOR_BYTES = {
    torch.float32: 16,
    torch.float16: 8,
    torch.bfloat16: 8,
}


_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

constant uint kMant = $MANT;
constant uint kExp = $EXP;
constant uint kMantMask = $MANT_MASK;
constant uint kExpMask = $EXP_MASK;
constant uint kHidden = $HIDDEN;
constant uint kSign = $SIGN;

struct split2 { uint m; int e; };

static inline split2 split_bits(uint bits) {
    uint sign = bits & kSign;
    uint ef = (bits >> kMant) & kExpMask;
    uint frac = bits & kMantMask;
    split2 r;
    if (ef == kExpMask) {
        r.m = bits;
        r.e = 0;
    } else if (ef == 0u) {
        if (frac == 0u) {
            r.m = sign;
            r.e = 0;
        } else {
            uint lz = clz(frac) - (32u - kMant);
            uint norm = frac << lz;
            r.m = sign | kHidden | ((norm << 1u) & kMantMask);
            r.e = $ONE_MINUS_BIAS - int(lz);
        }
    } else {
        r.m = sign | kHidden | frac;
        r.e = int(ef) - $BIAS + 1;
    }
    return r;
}

kernel void frexp_bits(
    device const $RAW* x, device $RAW* mantissa, device int* exponent,
    constant uint& n, uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    split2 r = split_bits(uint(x[i]));
    mantissa[i] = $RAW(r.m);
    exponent[i] = r.e;
}

kernel void frexp_bits4(
    device const $RAW* x, device $RAW* mantissa, device int* exponent,
    constant uint& n, uint i [[thread_position_in_grid]]) {
    uint base = i * 4u;
    if (base + 4u <= n) {
        $VEC v = *reinterpret_cast<device const $VEC*>(x + base);
        split2 a = split_bits(uint(v.x)), b = split_bits(uint(v.y)),
               c = split_bits(uint(v.z)), d = split_bits(uint(v.w));
        *reinterpret_cast<device $VEC*>(mantissa + base) =
            $VEC($RAW(a.m), $RAW(b.m), $RAW(c.m), $RAW(d.m));
        *reinterpret_cast<device int4*>(exponent + base) = int4(a.e, b.e, c.e, d.e);
    } else {
        for (uint j = 0; j < 4u; j++) {
            uint k = base + j;
            if (k >= n) break;
            split2 r = split_bits(uint(x[k]));
            mantissa[k] = $RAW(r.m);
            exponent[k] = r.e;
        }
    }
}
"""


@lru_cache(maxsize=8)
def _library(dtype):
    mant, exp, bias = _PARAMS[dtype]
    wide = dtype == torch.float32
    source = (
        _SOURCE.replace("$MANT_MASK", f"{(1 << mant) - 1}u")
        .replace("$EXP_MASK", f"{(1 << exp) - 1}u")
        .replace("$HIDDEN", f"{((bias - 1) << mant)}u")
        .replace("$SIGN", f"{1 << (mant + exp)}u")
        .replace("$ONE_MINUS_BIAS", str(1 - bias))
        .replace("$BIAS", str(bias))
        .replace("$MANT", str(mant))
        .replace("$EXP", str(exp))
        .replace("$RAW", "uint" if wide else "ushort")
        .replace("$VEC", "uint4" if wide else "ushort4")
    )
    return torch.mps.compile_shader(source)


def _check_dtype(dtype):
    if dtype not in _PARAMS:
        raise RuntimeError("torch.frexp() only supports floating-point dtypes")


def _run(x, mantissa, exponent):
    """Fill contiguous ``mantissa`` and ``exponent`` from contiguous ``x``."""
    n = x.numel()
    if n == 0:
        return
    raw = _raw_dtype(x.dtype)
    library = _library(x.dtype)
    x_raw, m_raw = x.view(raw), mantissa.view(raw)
    # frexp_bits4 reads four elements at once, so every buffer must start at an
    # address its vector load and store can name. Freshly allocated contiguous
    # tensors always qualify; a slice such as `torch.empty(8)[1:5]` does not.
    if n >= 4 and all(
        t.data_ptr() % _VECTOR_BYTES[x.dtype] == 0 for t in (x, mantissa, exponent)
    ):
        library.frexp_bits4(x_raw, m_raw, exponent, n, threads=(n + 3) // 4)
    else:
        library.frexp_bits(x_raw, m_raw, exponent, n, threads=n)


def frexp(self):
    """``aten::frexp.Tensor`` for MPS: returns ``(mantissa, exponent)``."""
    _check_dtype(self.dtype)
    mantissa = torch.empty_like(self, memory_format=torch.contiguous_format)
    exponent = torch.empty(self.shape, dtype=torch.int32, device=self.device)
    _run(self.contiguous(), mantissa, exponent)
    return mantissa, exponent


def frexp_out(self, *, mantissa, exponent):
    """``aten::frexp.Tensor_out`` for MPS, with the CPU kernel's checks."""
    dtype = self.dtype
    _check_dtype(dtype)
    if mantissa.dtype != dtype:
        raise RuntimeError(
            f"torch.frexp() expects mantissa to have dtype {_dtype_name(dtype)} "
            f"but got {_dtype_name(mantissa.dtype)}"
        )
    if exponent.dtype != torch.int32:
        raise RuntimeError(
            "torch.frexp() expects exponent to have int dtype "
            f"but got {_dtype_name(exponent.dtype)}"
        )
    if mantissa.device != self.device or exponent.device != self.device:
        raise RuntimeError(
            "Expected all tensors to be on the same device, but found at least "
            f"two devices, {self.device} and {mantissa.device}!"
        )
    if mantissa.shape != self.shape:
        mantissa.resize_(self.shape)
    if exponent.shape != self.shape:
        exponent.resize_(self.shape)

    shape = self.shape
    mant = mantissa if mantissa.is_contiguous() else torch.empty(
        shape, dtype=dtype, device=self.device
    )
    expo = exponent if exponent.is_contiguous() else torch.empty(
        shape, dtype=torch.int32, device=self.device
    )
    _run(self.contiguous(), mant, expo)
    if mant is not mantissa:
        mantissa.copy_(mant)
    if expo is not exponent:
        exponent.copy_(expo)
    return mantissa, exponent


OVERLAYS = (
    ("frexp.Tensor", frexp),
    ("frexp.Tensor_out", frexp_out),
)
