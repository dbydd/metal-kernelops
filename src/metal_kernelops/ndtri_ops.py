"""``aten::special_ndtri`` on MPS: the Cephes inverse normal CDF.

Algorithm
---------
This is a line-by-line port of ATen's ``calc_ndtri`` (``ATen/native/Math.h``),
itself derived from the Cephes Math Library. It is not an ``erfinv``
composition: ``sqrt(2) * erfinv(2p - 1)`` loses the tails at the exact
endpoints (``2p - 1`` rounds to -1 or 1) and the Cephes rational approximations
are the ones PyTorch's CPU kernel evaluates, so the two agree to a few ulp.
Center (``|p - 0.5| <= 0.365``), middle tail (``z < 8``) and far tail
(``8 <= z <= 64``) each use their own polynomial pair. The far-tail branch
covers every positive float32 (the smallest subnormal needs ``z = 14.1``) and
every float64; float32, float16 and bfloat16 pointers are read directly and all
arithmetic runs in float32.

The Metal source is compiled lazily on first use (``torch.mps.compile_shader``),
never at import. ``precise::log`` / ``precise::sqrt`` keep the runtime compiler's
fast-math defaults from degrading the tail branches.

Each thread evaluates four elements and moves them with a single 16-byte
``float4`` load and store (8-byte ``half4`` / ``ushort4`` for float16 and
bfloat16). The arithmetic is evaluated per element, so the polynomials,
branches and rounding are bit-identical to the one-element-per-thread kernel;
only the memory instructions and the grid size change. Vector accesses need 16
byte source and destination pointers, so a view whose storage offset breaks
that alignment falls back to the scalar kernel.

License
-------
``calc_ndtri`` and its coefficients are from the Cephes Math Library, via
PyTorch ``ATen/native/Math.h`` (note [3-Clause BSD License for the Cephes Math
Library]):

    Copyright (c) 2018, Steven Moshier
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the nor the names of its contributors may be used
      to endorse or promote products derived from this software without
      specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL Steven Moshier BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
    OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
    DAMAGE.
"""

from functools import lru_cache

import torch

_GROUP_SIZE = 512
"""Threads per threadgroup."""

_ELEMENTS_PER_THREAD = 4
"""Elements per thread in the vector kernels; also the vector width in elements."""

_KERNEL = {
    torch.float32: ("ndtri_f32", "ndtri_f32_scalar", 16),
    torch.float16: ("ndtri_f16", "ndtri_f16_scalar", 8),
    torch.bfloat16: ("ndtri_bf16", "ndtri_bf16_scalar", 8),
}
"""``dtype -> (vector kernel, scalar kernel, alignment in bytes)``."""


@lru_cache(maxsize=1)
def _library():
    return torch.mps.compile_shader(_SOURCE)


def _launch(source, destination):
    count = source.numel()
    vector, scalar, alignment = _KERNEL[source.dtype]
    if (source.data_ptr() % alignment
            or destination.data_ptr() % alignment):
        getattr(_library(), scalar)(source, destination, count,
                                    threads=count, group_size=_GROUP_SIZE)
        return
    getattr(_library(), vector)(source, destination, count,
                                threads=(count + _ELEMENTS_PER_THREAD - 1)
                                // _ELEMENTS_PER_THREAD,
                                group_size=_GROUP_SIZE)


def _ndtri(self):
    if self.device.type != "mps":
        raise ValueError(f"Metal kernels need tensors on the mps device, got {self.device}")
    if self.dtype not in _KERNEL:
        raise NotImplementedError(f'"ndtri_mps" not implemented for \'{self.dtype}\'')
    source = self.contiguous()
    out = torch.empty(self.shape, dtype=self.dtype, device=self.device)
    if source.numel():
        _launch(source, out)
    return out


def _ndtri_out(self, *, out):
    if self.device.type != "mps":
        raise ValueError(f"Metal kernels need tensors on the mps device, got {self.device}")
    if self.dtype not in _KERNEL:
        raise NotImplementedError(f'"ndtri_mps" not implemented for \'{self.dtype}\'')
    if out.device != self.device:
        raise RuntimeError(
            "Expected all tensors to be on the same device, but found at least two "
            f"devices, {self.device} and {out.device}!"
        )
    if not out.dtype.is_floating_point:
        raise RuntimeError(
            f"result type {self.dtype} can't be cast to the desired output type {out.dtype}")
    if out.shape != self.shape:
        out.resize_(self.shape)
    source = self.contiguous()
    count = source.numel()
    shared_storage = (source.untyped_storage().data_ptr()
                      == out.untyped_storage().data_ptr())
    direct = (out.dtype == self.dtype and out.is_contiguous()
              and not (shared_storage and source.data_ptr() != out.data_ptr()))
    if not count:
        return out
    if direct:
        _launch(source, out)
    else:
        result = torch.empty(self.shape, dtype=self.dtype, device=self.device)
        _launch(source, result)
        out.copy_(result)
    return out


OVERLAYS = (
    ("special_ndtri", _ndtri),
    ("special_ndtri.out", _ndtri_out),
)
"""``(aten overload, callable)`` rows for the accelerator's overlay table."""


_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

/* sqrt(2pi) */
constant float s2pi = 2.50662827463100050242E0f;

/* approximation for 0 <= |y - 0.5| <= 3/8 */
constant float P0[5] = {
    -5.99633501014107895267E1f,
    9.80010754185999661536E1f,
    -5.66762857469070293439E1f,
    1.39312609387279679503E1f,
    -1.23916583867381258016E0f,
};

constant float Q0[9] = {
    1.00000000000000000000E0f,
    1.95448858338141759834E0f,
    4.67627912898881538453E0f,
    8.63602421390890590575E1f,
    -2.25462687854119370527E2f,
    2.00260212380060660359E2f,
    -8.20372256168333339912E1f,
    1.59056225126211695515E1f,
    -1.18331621121330003142E0f,
};

/* Approximation for interval z = sqrt(-2 log y) between 2 and 8
 * i.e., y between exp(-2) = .135 and exp(-32) = 1.27e-14.
 */
constant float P1[9] = {
    4.05544892305962419923E0f,
    3.15251094599893866154E1f,
    5.71628192246421288162E1f,
    4.40805073893200834700E1f,
    1.46849561928858024014E1f,
    2.18663306850790267539E0f,
    -1.40256079171354495875E-1f,
    -3.50424626827848203418E-2f,
    -8.57456785154685413611E-4f,
};

constant float Q1[9] = {
    1.00000000000000000000E0f,
    1.57799883256466749731E1f,
    4.53907635128879210584E1f,
    4.13172038254672030440E1f,
    1.50425385692907503408E1f,
    2.50464946208309415979E0f,
    -1.42182922854787788574E-1f,
    -3.80806407691578277194E-2f,
    -9.33259480895457427372E-4f,
};

/* Approximation for interval z = sqrt(-2 log y) between 8 and 64
 * i.e., y between exp(-32) = 1.27e-14 and exp(-2048) = 3.67e-890.
 */
constant float P2[9] = {
    3.23774891776946035970E0f,
    6.91522889068984211695E0f,
    3.93881025292474443415E0f,
    1.33303460815807542389E0f,
    2.01485389549179081538E-1f,
    1.23716634817820021358E-2f,
    3.01581553508235416007E-4f,
    2.65806974686737550832E-6f,
    6.23974539184983293730E-9f,
};

constant float Q2[9] = {
    1.00000000000000000000E0f,
    6.02427039364742014255E0f,
    3.67983563856160859403E0f,
    1.37702099489081330271E0f,
    2.16236993594496635890E-1f,
    1.34204006088543189037E-2f,
    3.28014464682127739104E-4f,
    2.89247864745380683936E-6f,
    6.79019408009981274425E-9f,
};

inline float polevl(float x, constant float* coef, uint degree) {
    float ans = coef[0];
    for (uint i = 1; i <= degree; ++i) {
        ans = ans * x + coef[i];
    }
    return ans;
}

inline float ndtri_scalar(float y0) {
    if (y0 == 0.0f) {
        return -INFINITY;
    }
    if (y0 == 1.0f) {
        return INFINITY;
    }
    if (isnan(y0) || y0 < 0.0f || y0 > 1.0f) {
        return NAN;
    }

    bool code = true;
    float y = y0;
    if (y > 1.0f - 0.13533528323661269189f) { /* exp(-2) */
        y = 1.0f - y;
        code = false;
    }

    if (y > 0.13533528323661269189f) {
        y = y - 0.5f;
        float y2 = y * y;
        float x = y + y * (y2 * polevl(y2, P0, 4) / polevl(y2, Q0, 8));
        return x * s2pi;
    }

    float x = precise::sqrt(-2.0f * precise::log(y));
    float x0 = x - precise::log(x) / x;

    float z = 1.0f / x;
    float x1;
    if (x < 8.0f) { /* y > exp(-32) = 1.2664165549e-14 */
        x1 = z * polevl(z, P1, 8) / polevl(z, Q1, 8);
    } else {
        x1 = z * polevl(z, P2, 8) / polevl(z, Q2, 8);
    }
    x = x0 - x1;
    if (code) {
        x = -x;
    }
    return x;
}

/* Positive subnormals are flushed to zero by the Metal fast-math defaults, so
 * the input is read as raw bits and those inputs take the far-tail branch from
 * a normalized log: log(mantissa) - 149 log(2) for float32, and the equivalent
 * 24 / 133 exponent bias for float16 / bfloat16. Negative subnormals and
 * negative zero follow the CPU kernel: -inf for zero, NaN below it. */
inline float far_tail(float log_y) {
    float x = precise::sqrt(-2.0f * log_y);
    float x0 = x - precise::log(x) / x;
    float z = 1.0f / x;
    return x0 - z * polevl(z, P2, 8) / polevl(z, Q2, 8);
}

constant float ln2 = 6.93147180559945309417E-1f;

inline float ndtri_f32_bits(uint bits) {
    uint exponent = (bits >> 23) & 0xffu;
    if (exponent != 0u) {
        return ndtri_scalar(as_type<float>(bits));
    }
    uint mantissa = bits & 0x7fffffu;
    if (mantissa == 0u) {
        return -INFINITY;
    }
    if (bits >> 31 != 0u) {
        return NAN;
    }
    return -far_tail(precise::log(float(mantissa)) - 149.0f * ln2);
}

inline float ndtri_f16_bits(uint bits) {
    uint exponent = (bits >> 10) & 0x1fu;
    if (exponent != 0u) {
        return ndtri_scalar(float(as_type<half>(ushort(bits))));
    }
    uint mantissa = bits & 0x3ffu;
    if (mantissa == 0u) {
        return -INFINITY;
    }
    if (bits >> 15 != 0u) {
        return NAN;
    }
    return -far_tail(precise::log(float(mantissa)) - 24.0f * ln2);
}

inline float ndtri_bf16_bits(uint bits) {
    uint exponent = (bits >> 7) & 0xffu;
    if (exponent != 0u) {
        return ndtri_scalar(as_type<float>(bits << 16));
    }
    uint mantissa = bits & 0x7fu;
    if (mantissa == 0u) {
        return -INFINITY;
    }
    if (bits >> 15 != 0u) {
        return NAN;
    }
    return -far_tail(precise::log(float(mantissa)) - 133.0f * ln2);
}

kernel void ndtri_f32(device const float* x, device float* y,
                      constant uint& n,
                      uint i [[thread_position_in_grid]]) {
    uint base = i * 4u;
    if (base + 4u <= n) {
        uint4 bits = as_type<uint4>(
            *reinterpret_cast<device const float4*>(x + base));
        float4 r;
        r.x = ndtri_f32_bits(bits.x);
        r.y = ndtri_f32_bits(bits.y);
        r.z = ndtri_f32_bits(bits.z);
        r.w = ndtri_f32_bits(bits.w);
        *reinterpret_cast<device float4*>(y + base) = r;
    } else {
        for (uint k = base; k < n; ++k) {
            y[k] = ndtri_f32_bits(((device const uint*)x)[k]);
        }
    }
}

kernel void ndtri_f32_scalar(device const float* x, device float* y,
                             constant uint& n,
                             uint i [[thread_position_in_grid]]) {
    if (i < n) {
        y[i] = ndtri_f32_bits(((device const uint*)x)[i]);
    }
}

kernel void ndtri_f16(device const half* x, device half* y,
                      constant uint& n,
                      uint i [[thread_position_in_grid]]) {
    uint base = i * 4u;
    if (base + 4u <= n) {
        ushort4 bits = *reinterpret_cast<device const ushort4*>(x + base);
        half4 r;
        r.x = half(ndtri_f16_bits(bits.x));
        r.y = half(ndtri_f16_bits(bits.y));
        r.z = half(ndtri_f16_bits(bits.z));
        r.w = half(ndtri_f16_bits(bits.w));
        *reinterpret_cast<device half4*>(y + base) = r;
    } else {
        for (uint k = base; k < n; ++k) {
            y[k] = half(ndtri_f16_bits(((device const ushort*)x)[k]));
        }
    }
}

kernel void ndtri_f16_scalar(device const half* x, device half* y,
                             constant uint& n,
                             uint i [[thread_position_in_grid]]) {
    if (i < n) {
        y[i] = half(ndtri_f16_bits(((device const ushort*)x)[i]));
    }
}

kernel void ndtri_bf16(device const bfloat* x, device bfloat* y,
                       constant uint& n,
                       uint i [[thread_position_in_grid]]) {
    uint base = i * 4u;
    if (base + 4u <= n) {
        ushort4 bits = *reinterpret_cast<device const ushort4*>(x + base);
        ushort4 r;
        r.x = as_type<ushort>(bfloat(ndtri_bf16_bits(bits.x)));
        r.y = as_type<ushort>(bfloat(ndtri_bf16_bits(bits.y)));
        r.z = as_type<ushort>(bfloat(ndtri_bf16_bits(bits.z)));
        r.w = as_type<ushort>(bfloat(ndtri_bf16_bits(bits.w)));
        *reinterpret_cast<device ushort4*>(y + base) = r;
    } else {
        for (uint k = base; k < n; ++k) {
            y[k] = bfloat(ndtri_bf16_bits(((device const ushort*)x)[k]));
        }
    }
}

kernel void ndtri_bf16_scalar(device const bfloat* x, device bfloat* y,
                              constant uint& n,
                              uint i [[thread_position_in_grid]]) {
    if (i < n) {
        y[i] = bfloat(ndtri_bf16_bits(((device const ushort*)x)[i]));
    }
}
"""
