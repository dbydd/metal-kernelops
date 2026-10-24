"""MPS overlays for ``aten::fractional_max_pool2d`` / ``aten::fractional_max_pool3d``.

Both operators are missing on MPS: ``.output`` and ``.grad_input`` reach
``MPSFallback.mm`` and raise, and the 3d backward has no composite either. The
forward is one Metal kernel that reproduces the CPU reference exactly:

* ``generate_intervals`` with the C++ arithmetic, including the round after
  every operation for half and bfloat16 (c10's operators compute in float and
  return the narrow type);
* window reduction with the CPU rule -- a NaN wins its window and the *last*
  NaN takes the index, otherwise the first maximum in row-major order;
* the plane-local flat index layout of the CPU kernel: ``(t*H + h)*W + w``.

The backward is ``grad_input.zero_()`` plus a scatter-add, which matches the
CPU loop including duplicate winning positions. The out spellings and the base
3d backward are all filled: the base 3d backward has no composite to reach the
out variant, while the other base spellings do.

Importing this module registers nothing and compiles nothing. The overlay table
is consumed by :mod:`metal_kernelops.accelerator`.
"""

import math

import torch

__all__ = ["OVERLAYS"]


def _narrow_round(value, dtype):
    """Round a python float the way ``static_cast<scalar_t>`` would."""
    return torch.tensor(value, dtype=dtype).item()


_DTYPES = {
    torch.float32: ("f32", lambda v: v),
    torch.float16: ("f16", lambda v: _narrow_round(v, torch.float16)),
    torch.bfloat16: ("bf16", lambda v: _narrow_round(v, torch.bfloat16)),
}

_HELPERS = """
inline float fmp_rbf16(float v) {
  uint u = as_type<uint>(v);
  u += 0x7FFFu + ((u >> 16) & 1u);
  u &= 0xFFFF0000u;
  return as_type<float>(u);
}
inline float load_f32(constant float* p, long i) { return p[i]; }
inline float load_f16(constant half* p, long i) { return (float)p[i]; }
inline float load_bf16(constant ushort* p, long i) { return as_type<float>(((uint)p[i]) << 16); }
inline void store_f32(device float* p, long i, float v) { p[i] = v; }
inline void store_f16(device half* p, long i, float v) { p[i] = (half)v; }
inline void store_bf16(device ushort* p, long i, float v) { p[i] = (ushort)(as_type<uint>(v) >> 16); }
inline float rnd_f32(float v) { return v; }
inline float rnd_f16(float v) { return (float)(half)v; }
inline float rnd_bf16(float v) { return fmp_rbf16(v); }
"""

_BODY = {
    2: """
  uint ovol = (uint)(outH * outW);
  uint plane = gid / ovol;
  uint rem = gid - plane * ovol;
  uint oh = rem / (uint)outW;
  uint ow = rem - oh * (uint)outW;
  float sW = LOAD(samples, plane * 2 + 0);
  float sH = LOAD(samples, plane * 2 + 1);
  int stH = oh == outH - 1 ? (int)(inH - kH)
      : (int)RND(RND(RND((float)oh) + sH) * alphaH) - (int)RND(sH * alphaH);
  int stW = ow == outW - 1 ? (int)(inW - kW)
      : (int)RND(RND(RND((float)ow) + sW) * alphaW) - (int)RND(sW * alphaW);
  long base = (long)plane * inH * inW;
  long best = (long)stH * inW + stW;
  long idx = best;
  float maxv = -INFINITY;
  for (int h = 0; h < kH; ++h) {
    for (int w = 0; w < kW; ++w) {
      float v = LOAD(x, base + idx);
      if (v > maxv || v != v) { maxv = v; best = idx; }
      ++idx;
    }
    idx += inW - kW;
  }
  STORE(out, gid, maxv);
  ind[gid] = best;
}
""",
    3: """
  uint ovol = (uint)(outT * outH * outW);
  uint plane = gid / ovol;
  uint rem = gid - plane * ovol;
  uint ohw = (uint)(outH * outW);
  uint ot = rem / ohw;
  uint r2 = rem - ot * ohw;
  uint oh = r2 / (uint)outW;
  uint ow = r2 - oh * (uint)outW;
  float sT = LOAD(samples, plane * 3 + 0);
  float sH = LOAD(samples, plane * 3 + 1);
  float sW = LOAD(samples, plane * 3 + 2);
  int stT = ot == outT - 1 ? (int)(inT - kT)
      : (int)RND(RND(RND((float)ot) + sT) * alphaT) - (int)RND(sT * alphaT);
  int stH = oh == outH - 1 ? (int)(inH - kH)
      : (int)RND(RND(RND((float)oh) + sH) * alphaH) - (int)RND(sH * alphaH);
  int stW = ow == outW - 1 ? (int)(inW - kW)
      : (int)RND(RND(RND((float)ow) + sW) * alphaW) - (int)RND(sW * alphaW);
  long base = (long)plane * inT * inH * inW;
  long best = ((long)stT * inH + stH) * inW + stW;
  long idx = best;
  float maxv = -INFINITY;
  for (int t = 0; t < kT; ++t) {
    for (int h = 0; h < kH; ++h) {
      for (int w = 0; w < kW; ++w) {
        float v = LOAD(x, base + idx);
        if (v > maxv || v != v) { maxv = v; best = idx; }
        ++idx;
      }
      idx += inW - kW;
    }
    idx += ((long)inH - kH) * inW;
  }
  STORE(out, gid, maxv);
  ind[gid] = best;
}
""",
}

_SIGNATURE = {
    2: (
        "constant long& inH, constant long& inW,\n"
        "    constant long& outH, constant long& outW,\n"
        "    constant long& kH, constant long& kW,\n"
        "    constant float& alphaH, constant float& alphaW,"
    ),
    3: (
        "constant long& inT, constant long& inH, constant long& inW,\n"
        "    constant long& outT, constant long& outH, constant long& outW,\n"
        "    constant long& kT, constant long& kH, constant long& kW,\n"
        "    constant float& alphaT, constant float& alphaH, constant float& alphaW,"
    ),
}


def _source():
    """One kernel per (dimension, dtype); helpers are shared and inlined."""
    kernels = [_HELPERS]
    for ndim in (2, 3):
        body = _BODY[ndim]
        for storage, suffix in (
            ("float", "f32"), ("half", "f16"), ("ushort", "bf16"),
        ):
            text = (
                body.replace("LOAD", f"load_{suffix}")
                .replace("STORE", f"store_{suffix}")
                .replace("RND", f"rnd_{suffix}")
            )
            kernels.append(
                f"kernel void fmp{ndim}d_{suffix}(\n"
                f"    device {storage}* out, device long* ind,\n"
                f"    constant {storage}* x, constant {storage}* samples,\n"
                f"    {_SIGNATURE[ndim]}\n"
                f"    uint gid [[thread_position_in_grid]]) {{\n{text}"
            )
    return "\n".join(kernels)


_LIB = None


def _library():
    global _LIB
    if _LIB is None:
        _LIB = torch.mps.compile_shader(_source())
    return _LIB


def _check_random_samples(input, random_samples, ndim):
    if input.dtype != random_samples.dtype:
        raise RuntimeError("Expect _random_samples to have the same dtype as input")
    if random_samples.dim() != 3:
        raise RuntimeError(
            f"Expect _random_samples to have 3 dimensions, got {random_samples.dim()}"
        )
    batch = 1 if input.dim() == ndim + 1 else input.shape[0]
    channel = input.shape[-ndim - 1]
    n, c, d = random_samples.shape
    if n < batch:
        raise RuntimeError("Expect _random_samples.size(0) no less then input batch size.")
    if c != channel:
        raise RuntimeError("Expect _random_samples.size(1) equals to input channel size.")
    if d != ndim:
        raise RuntimeError(f"Expect _random_samples.size(2) equals to {ndim}; got {d}.")


def _check_sizes(input, kernel_size, output_size, ndim):
    if input.dim() not in (ndim + 1, ndim + 2):
        raise RuntimeError(
            f"fractional_max_pool{ndim}d(): Expected {ndim + 1}D or {ndim + 2}D tensor, "
            f"but got: {tuple(input.shape)}"
        )
    spatial = input.shape[-ndim:]
    for name, value in (("kernel", kernel_size), ("output", output_size)):
        if len(value) != ndim:
            raise RuntimeError(
                f"fractional_max_pool{ndim}d: {name}_size must have {ndim} values"
            )
        if any(int(v) <= 0 for v in value):
            raise RuntimeError(
                f"fractional_max_pool{ndim}d(): {name} size should be greater than zero"
            )
    for i in range(ndim):
        size, pool, out = int(spatial[i]), int(kernel_size[i]), int(output_size[i])
        # 2d uses output + pool - 1 <= input, 3d uses <. Both are the bounds the
        # kernel relies on to stay inside the input.
        limit = size if ndim == 2 else size - 1
        if out + pool - 1 > limit:
            raise RuntimeError(
                f"fractional_max_pool{ndim}d(): output {out} with pool {pool} "
                f"too large relative to input {size}"
            )


def _fractional_pool(input, kernel_size, output_size, random_samples, ndim,
                     output, indices):
    _check_sizes(input, kernel_size, output_size, ndim)
    _check_random_samples(input, random_samples, ndim)
    if input.dtype not in _DTYPES:
        raise RuntimeError(
            f"fractional_max_pool{ndim}d is not implemented for {input.dtype} on MPS"
        )
    if output.numel() == 0:
        return output, indices
    suffix, round_alpha = _DTYPES[input.dtype]

    x = input.contiguous()
    spatial = tuple(int(s) for s in x.shape[-ndim:])
    planes = x.numel() // math.prod(spatial)
    samples = random_samples.contiguous().reshape(-1)[: planes * ndim].reshape(
        planes, ndim
    )
    kernel_size = tuple(int(k) for k in kernel_size)
    output_size = tuple(int(o) for o in output_size)
    alphas = tuple(
        0.0 if output_size[i] == 1
        else round_alpha((spatial[i] - kernel_size[i]) / (output_size[i] - 1))
        for i in range(ndim)
    )

    out, temp_out = output, None
    ind, temp_ind = indices, None
    if not out.is_contiguous():
        temp_out = torch.empty(out.shape, dtype=out.dtype, device=out.device)
        out = temp_out
    if not ind.is_contiguous():
        temp_ind = torch.empty(ind.shape, dtype=ind.dtype, device=ind.device)
        ind = temp_ind

    kernel = getattr(_library(), f"fmp{ndim}d_{suffix}")
    kernel(
        out, ind, x, samples,
        *spatial, *output_size, *kernel_size, *alphas,
        threads=[planes * math.prod(output_size)],
    )

    if temp_out is not None:
        output.copy_(temp_out)
    if temp_ind is not None:
        indices.copy_(temp_ind)
    return output, indices


def _fractional_pool_backward(grad_output, input, indices, grad_input, ndim):
    spatial = tuple(input.shape[-ndim:])
    planes = input.numel() // math.prod(spatial)
    grad_input.zero_()
    target = grad_input.reshape(planes, -1)
    target.scatter_add_(
        1,
        indices.contiguous().reshape(planes, -1),
        grad_output.contiguous().reshape(planes, -1),
    )
    if target.data_ptr() != grad_input.data_ptr():
        grad_input.copy_(target.reshape(grad_input.shape))
    return grad_input


def fractional_max_pool2d_out(self, kernel_size, output_size, random_samples, *,
                              output, indices):
    return _fractional_pool(
        self, kernel_size, output_size, random_samples, 2, output, indices
    )


def fractional_max_pool2d_backward(grad_output, self, kernel_size, output_size,
                                   indices, *, grad_input):
    return _fractional_pool_backward(grad_output, self, indices, grad_input, 2)


def fractional_max_pool3d_out(self, kernel_size, output_size, random_samples, *,
                              output, indices):
    return _fractional_pool(
        self, kernel_size, output_size, random_samples, 3, output, indices
    )


def fractional_max_pool3d_backward(grad_output, self, kernel_size, output_size,
                                   indices):
    # No composite exists for this overload, so this body is the whole kernel.
    expected = tuple(self.shape[:-3]) + tuple(int(o) for o in output_size)
    if tuple(grad_output.shape) != expected:
        raise RuntimeError("fractional_max_pool3d_backward(): gradOutput sizes unexpected")
    if tuple(indices.shape) != expected:
        raise RuntimeError("fractional_max_pool3d_backward(): indices sizes unexpected")
    grad_input = self.new_empty(self.shape)
    return _fractional_pool_backward(grad_output, self, indices, grad_input, 3)


def fractional_max_pool3d_backward_out(grad_output, self, kernel_size, output_size,
                                       indices, *, grad_input):
    return _fractional_pool_backward(grad_output, self, indices, grad_input, 3)


OVERLAYS = (
    ("fractional_max_pool2d.output", fractional_max_pool2d_out),
    ("fractional_max_pool2d_backward.grad_input", fractional_max_pool2d_backward),
    ("fractional_max_pool3d.output", fractional_max_pool3d_out),
    ("fractional_max_pool3d_backward", fractional_max_pool3d_backward),
    ("fractional_max_pool3d_backward.grad_input", fractional_max_pool3d_backward_out),
)
