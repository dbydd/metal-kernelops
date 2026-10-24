"""MPS overlay for ``aten::special_airy_ai``.

Algorithm
---------
A direct float32 transcription of ATen's ``airy_ai_forward``
(``torch/include/ATen/native/Math.h``, torch 2.14.1), which is itself a
port of ``scipy/special/cephes/airy.c``. Three regimes, selected by the same
cut points as upstream:

* ``x < -2.09``: asymptotic oscillatory expansion in ``z = 3 / (2 (-x)^1.5)``
  with the AFN/AFD/AGN/AGD rational fits and phase
  ``t = 2(-x)^1.5/3 + pi/4``.
* ``2.09 <= x <= 103.892``: asymptotic decaying expansion in
  ``zeta = 2 x^1.5 / 3`` with the AN/AD fits. Above 8.3203353 the fit is the
  result; below it upstream keeps computing the power series afterwards and
  then discards it, so the dead series is skipped here.
* ``-2.09 <= x < 2.09``: the ascending power series for ``f`` and ``g``,
  combined as ``0.35502805... f - 0.25881940... g``.

``x >= 26.5`` returns ``0`` (the CPU float32 asymptotic denominator
overflows to ``+inf`` at ``x = 25.7709``, so all ``x >= 26.5`` are bit-exact
zeros), ``+-inf`` returns ``NaN``, ``NaN`` propagates.
The loop termination epsilon is ``FLT_EPSILON``, as for the float32 ATen
instantiation.

License
-------
The coefficients and the branch structure are from the Cephes Math Library
(Copyright (c) 2018, Steven Moshier, 3-clause BSD) as redistributed by SciPy
(SciPy Developers, 3-clause BSD); both are reproduced in PyTorch's ``NOTICE``.

Precision
---------
Metal's ``sin``/``cos``/``exp``/``sqrt`` are not the host libm, so results are
not bit-identical to the CPU kernel. ``benchmarks/verify_airy.py`` records the
measured agreement: the ascending-series branch is bit-exact, and the whole
finite domain stays within 9e-8 absolute and 22 float32 ulp (where
``|Ai| >= 1e-3``) of the CPU kernel on torch 2.14.1.

Semantics
---------
Only float32 is implemented: that is the dtype of the ATen kernel, and it is
the only floating input dtype MPS can host for this operator.
``out`` is honoured with the CPU layout and resize rules: a wrong-shaped
``out`` is resized, a strided ``out`` receives the values through its own
strides, and any dtype the CPU kernel rejects raises ``NotImplementedError``
with the same wording. The operator is not differentiable upstream, so the
overlay returns a plain tensor with no grad_fn.
"""

import torch

_SHADER = r"""
#include <metal_stdlib>
using namespace metal;

// ATen airy_ai_forward<float>, from Cephes via SciPy.
static inline float airy_ai(float x) {
    if (isinf(x)) {
        return NAN;
    }
    // Above 25.7709 the CPU float32 kernel returns +0.0 because the
    // asymptotic denominator overflows to +inf. Skipping the branch for
    // large x matches CPU bit-for-bit and removes the transcendental work.
    if (x >= 26.5f) {
        return 0.0f;
    }

    if (x < -2.09f) {
        float s = sqrt(-x);
        float q = -2.0f * x * s / 3.0f;
        float z = 1.0f / q;
        float zz = z * z;

        float afn = -1.31696323418331795333e-01f;
        afn = afn * zz - 6.26456544431912369773e-01f;
        afn = afn * zz - 6.93158036036933542233e-01f;
        afn = afn * zz - 2.79779981545119124951e-01f;
        afn = afn * zz - 4.91900132609500318020e-02f;
        afn = afn * zz - 4.06265923594885404393e-03f;
        afn = afn * zz - 1.59276496239262096340e-04f;
        afn = afn * zz - 2.77649108155232920844e-06f;
        afn = afn * zz - 1.67787698489114633780e-08f;

        float afd = 1.33560420706553243746e+01f;
        afd = afd * zz + 3.26825032795224613948e+01f;
        afd = afd * zz + 2.67367040941499554804e+01f;
        afd = afd * zz + 9.18707402907259625840e+00f;
        afd = afd * zz + 1.47529146771666414581e+00f;
        afd = afd * zz + 1.15687173795188044134e-01f;
        afd = afd * zz + 4.40291641615211203805e-03f;
        afd = afd * zz + 7.54720348287414296618e-05f;
        afd = afd * zz + 4.51850092970580378464e-07f;

        float agn = 1.97339932091685679179e-02f;
        agn = agn * zz + 3.91103029615688277255e-01f;
        agn = agn * zz + 1.06579897599595591108e+00f;
        agn = agn * zz + 9.39169229816650230044e-01f;
        agn = agn * zz + 3.51465656105547619242e-01f;
        agn = agn * zz + 6.33888919628925490927e-02f;
        agn = agn * zz + 5.85804113048388458567e-03f;
        agn = agn * zz + 2.82851600836737019778e-04f;
        agn = agn * zz + 6.98793669997260967291e-06f;
        agn = agn * zz + 8.11789239554389293311e-08f;
        agn = agn * zz + 3.41551784765923618484e-10f;

        float agd = 9.30892908077441974853e+00f;
        agd = agd * zz + 1.98352928718312140417e+01f;
        agd = agd * zz + 1.55646628932864612953e+01f;
        agd = agd * zz + 5.47686069422975497931e+00f;
        agd = agd * zz + 9.54293611618961883998e-01f;
        agd = agd * zz + 8.64580826352392193095e-02f;
        agd = agd * zz + 4.12656523824222607191e-03f;
        agd = agd * zz + 1.01259085116509135510e-04f;
        agd = agd * zz + 1.17166733214413521882e-06f;
        agd = agd * zz + 4.91834570062930015649e-09f;

        float t = q + 0.25f * 3.14159265358979323846f;
        return 5.64189583547756286948e-01f / sqrt(s) *
               (sin(t) * (1.0f + zz * afn / afd) - cos(t) * (z * agn / agd));
    }

    if (x >= 2.09f) {
        float zeta = 2.0f * x * sqrt(x) / 3.0f;
        float inv = 1.0f / zeta;

        float an = 3.46538101525629032477e-01f;
        an = an * inv + 1.20075952739645805542e+01f;
        an = an * inv + 7.62796053615234516538e+01f;
        an = an * inv + 1.68089224934630576269e+02f;
        an = an * inv + 1.59756391350164413639e+02f;
        an = an * inv + 7.05360906840444183113e+01f;
        an = an * inv + 1.40264691163389668864e+01f;
        an = an * inv + 9.99999999999999995305e-01f;

        float ad = 5.67594532638770212846e-01f;
        ad = ad * inv + 1.47562562584847203173e+01f;
        ad = ad * inv + 8.45138970141474626562e+01f;
        ad = ad * inv + 1.77318088145400459522e+02f;
        ad = ad * inv + 1.64234692871529701831e+02f;
        ad = ad * inv + 7.14778400825575695274e+01f;
        ad = ad * inv + 1.40959135607834029598e+01f;
        ad = ad * inv + 1.00000000000000000470e+00f;

        return 5.64189583547756286948e-01f * (an / ad) /
               (2.0f * sqrt(sqrt(x)) * exp(zeta));
    }

    float f = 1.0f;
    float g = x;
    float k = 1.0f;
    float m = 1.0f;
    float n = x;
    float t = 1.0f;
    float z = x * x * x;

    while (t > 1.1920928955078125e-07f) {
        m *= z;
        k += 1.0f;
        m /= k;
        n *= z;
        k += 1.0f;
        n /= k;
        m /= k;
        f += m;
        k += 1.0f;
        n /= k;
        g += n;
        t = abs(m / f);
    }

    return 0.355028053887817239260f * f - 0.258819403792806798405f * g;
}

kernel void airy_ai_kernel(device float* out,
                           constant float* x,
                           uint idx [[thread_position_in_grid]]) {
    out[idx] = airy_ai(x[idx]);
}
"""

_library = None

_ATEN_NAMES = {
    torch.float64: "Double",
    torch.float32: "Float",
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.complex64: "ComplexFloat",
    torch.complex128: "ComplexDouble",
    torch.int64: "Long",
    torch.int32: "Int",
    torch.int16: "Short",
    torch.int8: "Char",
    torch.uint8: "Byte",
    torch.bool: "Bool",
}

_FLOAT_OUT = (torch.float32, torch.float16, torch.bfloat16)


def _shader_library():
    global _library
    if _library is None:
        _library = torch.mps.compile_shader(_SHADER)
    return _library


def _unsupported(dtype):
    name = _ATEN_NAMES.get(dtype, str(dtype))
    return NotImplementedError(f'"airy_ai_mps" not implemented for \'{name}\'')


def airy_ai_out(x, out):
    """Write ``Ai(x)`` into ``out``; the MPS implementation of the out variant."""
    if x.dtype is not torch.float32:
        if x.is_floating_point() or x.is_complex():
            raise _unsupported(x.dtype)
        x = x.to(torch.float32)
    if out.dtype not in _FLOAT_OUT:
        raise RuntimeError(
            f"result type {_ATEN_NAMES.get(x.dtype, x.dtype)} can't be cast to the desired "
            f"output type {_ATEN_NAMES.get(out.dtype, out.dtype)}"
        )
    if out.shape != x.shape:
        out.resize_(x.shape)
    if not x.is_contiguous():
        x = x.contiguous()
    if out.numel() == 0:
        return out
    if out.dtype is torch.float32 and out.is_contiguous():
        _shader_library().airy_ai_kernel(out, x)
    else:
        tmp = torch.empty(x.shape, dtype=torch.float32, device=x.device)
        _shader_library().airy_ai_kernel(tmp, x)
        out.copy_(tmp)
    return out


def _special_airy_ai_out(x, *, out):
    return airy_ai_out(x, out)


OVERLAYS = (("special_airy_ai.out", _special_airy_ai_out),)
