"""MPS overlays for RReLU.

``aten::rrelu_with_noise`` has no MPS kernel in torch 2.14.1, so ``F.rrelu``,
``nn.RReLU`` and ``torch.rrelu`` reach the CPU-only kernel and raise
``NotImplementedError``. Three kernels are overlaid here, one per overload:
``rrelu_with_noise``, ``rrelu_with_noise_`` and ``rrelu_with_noise.out``.

Semantics copied from the CPU kernel:

* ``training=True`` -- ``noise`` is overwritten with the slope each element
  actually used: a uniform draw in ``[lower, upper)`` where ``self <= 0`` and
  ``1.0`` where ``self > 0``. The output is ``self * noise``.
* ``training=False`` -- ``noise`` is left untouched and the output is
  ``leaky_relu`` with ``negative_slope = (lower + upper) / 2``. torch computes
  that slope in double precision; rounding it to float32 first would differ in
  the last bit, and would show up for large inputs.
* ``noise`` is non-differentiable; the gradient is ``grad * noise`` in training
  and ``leaky_relu_backward`` otherwise. ``aten::rrelu_with_noise_backward`` is
  already a composite kernel that runs on MPS through ``mul`` and
  ``leaky_relu_backward``, so it gets no overlay here: a Python body would add
  a dispatch frame in front of the same single kernel.

The training path samples exactly one uniform tensor through MPS's own
``uniform_`` (so the MPS generator, seeded or explicit, drives it) and folds
the mask, the noise write and the output into one Metal kernel: two kernel
launches instead of the four an aten composition needs, and 1.3x-1.8x faster
across the sizes and dtypes recorded in ``measured/rrelu.json`` (1M to 16M
elements, float32/float16/bfloat16). Non-contiguous inputs and other dtypes
fall back to the aten composition, which is still entirely on the GPU.

The ``1.0`` at positive positions is not padding: the CPU kernel stores it there
and the backward multiplies the upstream gradient by ``noise`` unexamined, so a
positive element would otherwise be scaled by a random slope.

Importing this module neither registers nor compiles anything.
"""

from functools import lru_cache

import torch

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

kernel void rrelu_f32(device const float* x, device float* noise, device float* out,
                      constant uint& n, uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float v = x[i], u = noise[i];
    bool positive = v > 0.0f;
    noise[i] = positive ? 1.0f : u;
    out[i] = positive ? v : v * u;
}

kernel void rrelu_f16(device const half* x, device half* noise, device half* out,
                      constant uint& n, uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    half v = x[i], u = noise[i];
    bool positive = v > 0.0h;
    noise[i] = positive ? half(1.0) : u;
    out[i] = positive ? v : v * u;
}

kernel void rrelu_bf16(device const bfloat* x, device bfloat* noise, device bfloat* out,
                       constant uint& n, uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    bfloat v = x[i], u = noise[i];
    bool positive = v > bfloat(0.0);
    noise[i] = positive ? bfloat(1.0) : u;
    out[i] = positive ? v : v * u;
}

kernel void rrelu_inplace_f32(device float* x, device float* noise,
                              constant uint& n, uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float v = x[i], u = noise[i];
    bool positive = v > 0.0f;
    noise[i] = positive ? 1.0f : u;
    x[i] = positive ? v : v * u;
}

kernel void rrelu_inplace_f16(device half* x, device half* noise,
                              constant uint& n, uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    half v = x[i], u = noise[i];
    bool positive = v > 0.0h;
    noise[i] = positive ? half(1.0) : u;
    x[i] = positive ? v : v * u;
}

kernel void rrelu_inplace_bf16(device bfloat* x, device bfloat* noise,
                               constant uint& n, uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    bfloat v = x[i], u = noise[i];
    bool positive = v > bfloat(0.0);
    noise[i] = positive ? bfloat(1.0) : u;
    x[i] = positive ? v : v * u;
}
"""

_NAMES = {
    torch.float32: ("rrelu_f32", "rrelu_inplace_f32"),
    torch.float16: ("rrelu_f16", "rrelu_inplace_f16"),
    torch.bfloat16: ("rrelu_bf16", "rrelu_inplace_bf16"),
}

_THREADS_PER_GROUP = 256


@lru_cache(maxsize=1)
def _shader():
    return torch.mps.compile_shader(_SOURCE)


def _kernel(self, noise, inplace=False):
    """Fused Metal kernel for this call, or None when only the aten path applies."""
    names = _NAMES.get(self.dtype)
    if names is None:
        return None
    if not (self.is_contiguous() and noise.is_contiguous()):
        return None
    return getattr(_shader(), names[1 if inplace else 0])


def _slope(lower, upper):
    return (float(lower) + float(upper)) / 2


def _check(self, noise):
    if noise.shape != self.shape:
        raise RuntimeError(
            "noise tensor shape must match self tensor shape. Got "
            f"self.shape = {list(self.shape)} noise.shape = {list(noise.shape)}"
        )
    if noise.device != self.device or noise.dtype != self.dtype:
        raise RuntimeError(
            f"expected noise on {self.device} with dtype {self.dtype}, "
            f"got {noise.device} {noise.dtype}"
        )


def _launch(kernel, self, noise, out=None):
    n = self.numel()
    if not n:
        return
    if out is None:
        kernel(self, noise, n, threads=n, group_size=_THREADS_PER_GROUP)
    else:
        kernel(self, noise, out, n, threads=n, group_size=_THREADS_PER_GROUP)


def rrelu_with_noise(self, noise, lower=0.125, upper=1.0 / 3, training=False,
                     generator=None):
    """Overlay for ``aten::rrelu_with_noise`` on MPS."""
    _check(self, noise)
    if not training:
        return torch.ops.aten.leaky_relu(self, _slope(lower, upper))
    kernel = _kernel(self, noise)
    noise.uniform_(lower, upper, generator=generator)
    if kernel is None:
        noise.masked_fill_(self > 0, 1.0)
        return self * noise
    out = torch.empty_like(self)
    _launch(kernel, self, noise, out)
    return out


def rrelu_with_noise_(self, noise, lower=0.125, upper=1.0 / 3, training=False,
                      generator=None):
    """Overlay for ``aten::rrelu_with_noise_`` on MPS."""
    _check(self, noise)
    if not training:
        return torch.ops.aten.leaky_relu_(self, _slope(lower, upper))
    kernel = _kernel(self, noise, inplace=True)
    # The mask has to be read before the in-place write below.
    mask = None if kernel is not None else self > 0
    noise.uniform_(lower, upper, generator=generator)
    if kernel is None:
        noise.masked_fill_(mask, 1.0)
        return self.mul_(noise)
    _launch(kernel, self, noise)
    return self


def rrelu_with_noise_out(self, noise, lower=0.125, upper=1.0 / 3, training=False,
                         generator=None, *, out):
    """Overlay for ``aten::rrelu_with_noise.out`` on MPS."""
    _check(self, noise)
    if out.shape != self.shape:
        out.resize_(self.shape)
    if not training:
        return torch.ops.aten.leaky_relu.out(self, _slope(lower, upper), out=out)
    if out.data_ptr() == noise.data_ptr():
        return out.copy_(
            rrelu_with_noise(self, noise, lower, upper, True, generator)
        )
    kernel = _kernel(self, noise) if out.is_contiguous() and out.dtype == self.dtype else None
    noise.uniform_(lower, upper, generator=generator)
    if kernel is None:
        noise.masked_fill_(self > 0, 1.0)
        return torch.ops.aten.mul.out(self, noise, out=out)
    _launch(kernel, self, noise, out)
    return out


OVERLAYS = (
    ("rrelu_with_noise", rrelu_with_noise),
    ("rrelu_with_noise_", rrelu_with_noise_),
    ("rrelu_with_noise.out", rrelu_with_noise_out),
)
