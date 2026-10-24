"""Native aten decompositions for missing MPS schemas."""

import torch
import torch.nn.functional as F

_MAGNITUDE = {
    torch.float32: (torch.int32, 0x7FFFFFFF, 0x7F800000),
    torch.bfloat16: (torch.int16, 0x7FFF, 0x7F80),
}


def _check_overlap(output, *inputs):
    if any(torch._C._overlaps(output, value) for value in inputs):
        raise RuntimeError(
            "unsupported operation: some elements of the input tensor and the written-to "
            "tensor refer to a single memory location. Please clone() the tensor before "
            "performing the operation."
        )


def heaviside(self, values):
    if self.dtype != values.dtype:
        raise RuntimeError("heaviside is not yet implemented for tensors with different dtypes.")
    spec = _MAGNITUDE.get(self.dtype)
    if spec is not None:
        integer, mask, infinity = spec
        bits = self.view(integer)
        magnitude = bits & mask
        zero = torch.zeros((), dtype=self.dtype, device=self.device)
        one = torch.ones((), dtype=self.dtype, device=self.device)
        return torch.where(
            magnitude == 0, values,
            torch.where(bits < 0, zero, torch.where(magnitude > infinity, zero, one)),
        )
    return torch.where(self == 0, values, (self > 0).to(self.dtype))


def heaviside_out(self, values, *, out):
    _check_overlap(out, self, values)
    result = heaviside(self, values)
    if out.dtype != result.dtype:
        raise RuntimeError("heaviside: out dtype must match input dtype")
    out.resize_(result.shape)
    return out.copy_(result)


def heaviside_(self, values):
    _check_overlap(self, values)
    return self.copy_(heaviside(self, values))


def logspace_out(start, end, steps, base=10.0, *, out):
    if steps < 0:
        raise RuntimeError("number of steps must be non-negative")
    if out.dtype == torch.bool:
        raise NotImplementedError('"logspace_mps" not implemented for Bool')
    out.resize_((steps,))
    if steps == 0:
        return out
    if out.is_complex():
        real_endpoints = not (
            isinstance(start, complex)
            or isinstance(end, complex)
            or (torch.is_tensor(start) and start.is_complex())
            or (torch.is_tensor(end) and end.is_complex())
        )
        if real_endpoints:
            exponents = torch.linspace(
                start, end, steps, device=out.device
            ).to(torch.complex64)
        else:
            real_start = start.real if torch.is_tensor(start) else complex(start).real
            real_end = end.real if torch.is_tensor(end) else complex(end).real
            imaginary_start = start.imag if torch.is_tensor(start) else complex(start).imag
            imaginary_end = end.imag if torch.is_tensor(end) else complex(end).imag
            exponents = torch.complex(
                torch.linspace(real_start, real_end, steps, device=out.device),
                torch.linspace(imaginary_start, imaginary_end, steps, device=out.device),
            )
    elif not out.is_floating_point():
        if steps == 1:
            exponents = torch.tensor([float(start)], device=out.device)
        else:
            first, last = int(start), int(end)
            step = (last - first) / (steps - 1)
            exponents = torch.arange(steps, device=out.device) * step + first
    else:
        compute = torch.float32 if out.dtype in (torch.float16, torch.bfloat16) else out.dtype
        exponents = torch.linspace(start, end, steps, device=out.device, dtype=compute)
    return out.copy_(torch.pow(base, exponents))


def add_relu(self, other, alpha=1):
    promoted = torch.result_type(self, other)
    if not (promoted.is_floating_point or promoted.is_complex):
        alpha = int(alpha)
    return torch.add(self, other, alpha=alpha).relu_()


def add_relu_out(self, other, alpha=1, *, out):
    result = add_relu(self, other, alpha)
    if not torch.can_cast(result.dtype, out.dtype):
        raise RuntimeError(
            f"result type {result.dtype} can't be cast to the desired output type {out.dtype}"
        )
    out.resize_(result.shape)
    return out.copy_(result)


def add_relu_(self, other, alpha=1):
    promoted = torch.result_type(self, other)
    if not (promoted.is_floating_point or promoted.is_complex):
        alpha = int(alpha)
    return self.add_(other, alpha=alpha).relu_()


def addmm_activation(self, mat1, mat2, *, beta=1, alpha=1, use_gelu=False):
    result = torch.addmm(self, mat1, mat2, beta=beta, alpha=alpha)
    return F.gelu(result) if use_gelu else result.relu_()


def addmm_activation_out(self, mat1, mat2, *, beta=1, alpha=1, use_gelu=False, out):
    result = addmm_activation(self, mat1, mat2, beta=beta, alpha=alpha, use_gelu=use_gelu)
    if out.dtype != result.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {result.dtype}, but got {out.dtype} instead"
        )
    out.resize_(result.shape)
    return out.copy_(result)


def aminmax(self):
    return tuple(torch.aminmax(self))


def aminmax_dim(self, dim, keepdim=False):
    return tuple(torch.aminmax(self, dim=dim, keepdim=keepdim))


def aminmax_out(self, *, out0, out1):
    lo, hi = aminmax(self)
    if out0.dtype != lo.dtype or out1.dtype != hi.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {lo.dtype}, but got {out0.dtype} instead"
        )
    out0.resize_(lo.shape).copy_(lo)
    out1.resize_(hi.shape).copy_(hi)
    return out0, out1


def aminmax_dim_out(self, dim, keepdim=False, *, out0, out1):
    lo, hi = aminmax_dim(self, dim, keepdim)
    if out0.dtype != lo.dtype or out1.dtype != hi.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {lo.dtype}, but got {out0.dtype} instead"
        )
    out0.resize_(lo.shape).copy_(lo)
    out1.resize_(hi.shape).copy_(hi)
    return out0, out1


def batch_norm_update_stats(input, running_mean, running_var, momentum):
    if input.ndim < 2 or input.numel() == 0:
        raise RuntimeError("batch_norm_update_stats expects a non-empty input with at least 2 dimensions")
    for running in (running_mean, running_var):
        if running is None or running.dtype == input.dtype:
            continue
        if input.dtype in (torch.float16, torch.bfloat16) and running.dtype == torch.float32:
            continue
        if input.dtype == torch.float32:
            raise RuntimeError(
                "mixed dtype (CPU): expect parameter to have scalar type of Float"
            )
        raise RuntimeError("all inputs must share same datatype.")
    dims = (0,) + tuple(range(2, input.ndim))
    work = input.float() if input.dtype in (torch.float16, torch.bfloat16) else input
    variance, mean = torch.var_mean(work, dim=dims, correction=0)
    count = input.numel() // input.shape[1]
    unbiased = (
        variance * (count / (count - 1))
        if count > 1
        else torch.full_like(variance, float("nan"))
    )
    with torch.no_grad():
        for running, value in ((running_mean, mean), (running_var, unbiased)):
            if running is None:
                continue
            accumulator = (
                running.dtype
                if running.dtype in (torch.float32, torch.float64)
                else torch.float32
            )
            running.copy_(torch.lerp(running.to(accumulator), value.to(accumulator), momentum))
    if running_mean is not None:
        dtype = running_mean.dtype
    elif running_var is not None:
        dtype = running_var.dtype
    else:
        dtype = input.dtype
    return mean.to(dtype), variance.to(dtype)


def glu_jvp(glu, x, dx, dim):
    if not (glu.dtype == x.dtype == dx.dtype):
        raise RuntimeError(f"Found dtype {glu.dtype} but expected {x.dtype}")
    k = glu.shape[dim]
    b = x.narrow(dim, k, k)
    da, db = dx.narrow(dim, 0, k), dx.narrow(dim, k, k)
    s = b.sigmoid()
    return da * s + glu * (db - s * db)


def glu_jvp_out(glu, x, dx, dim, *, out):
    result = glu_jvp(glu, x, dx, dim)
    if out.dtype != result.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {result.dtype}, but got {out.dtype} instead"
        )
    out.resize_(result.shape)
    return out.copy_(result)


def glu_backward_jvp(grad_x, grad_glu, x, dgrad_glu, dx, dim):
    if not (grad_x.dtype == grad_glu.dtype == x.dtype == dgrad_glu.dtype == dx.dtype):
        raise RuntimeError(f"Found dtype {grad_glu.dtype} but expected {x.dtype}")
    k = grad_glu.shape[dim]
    if grad_x.shape[dim] != 2 * k:
        raise RuntimeError("grad_x must have twice the size of grad_glu on dim")
    a, b = x.narrow(dim, 0, k), x.narrow(dim, k, k)
    da, db = dx.narrow(dim, 0, k), dx.narrow(dim, k, k)
    ga = grad_x.narrow(dim, 0, k)
    s = b.sigmoid()
    y = a * s
    z = db - db * s
    dga = dgrad_glu * s + ga * z
    dgb = dga * (a - y) + ga * (da - da * s - y * z)
    return torch.cat((dga, dgb), dim=dim)


def glu_backward_jvp_out(grad_x, grad_glu, x, dgrad_glu, dx, dim, *, out):
    result = glu_backward_jvp(grad_x, grad_glu, x, dgrad_glu, dx, dim)
    if out.dtype != result.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {result.dtype}, but got {out.dtype} instead"
        )
    out.resize_(result.shape)
    return out.copy_(result)


OVERLAYS = (
    ("heaviside", heaviside), ("heaviside.out", heaviside_out), ("heaviside_", heaviside_),
    ("logspace.out", logspace_out),
    ("_add_relu.Tensor", add_relu), ("_add_relu.Scalar", add_relu),
    ("_add_relu.out", add_relu_out), ("_add_relu.Scalar_out", add_relu_out),
    ("_add_relu_.Tensor", add_relu_), ("_add_relu_.Scalar", add_relu_),
    ("_addmm_activation", addmm_activation), ("_addmm_activation.out", addmm_activation_out),
    ("_aminmax", aminmax), ("_aminmax.dim", aminmax_dim),
    ("_aminmax.out", aminmax_out), ("_aminmax.dim_out", aminmax_dim_out),
    ("batch_norm_update_stats", batch_norm_update_stats),
    ("glu_jvp", glu_jvp), ("glu_jvp.out", glu_jvp_out),
    ("glu_backward_jvp", glu_backward_jvp), ("glu_backward_jvp.out", glu_backward_jvp_out),
)
