"""MPS overlay for ``aten::linalg_matrix_sqrth``.

torch 2.14.1 has a CPU kernel for this operator and no MPS kernel, so
``torch.linalg.matrix_sqrth`` on an MPS tensor raises ``NotImplementedError``.
The CPU kernel reads the lower triangle, runs ``linalg_eigh``, clamps negative
eigenvalues to zero and rebuilds ``V diag(sqrt(w)) V^H``. MPS already has every
op it needs: ``eigh`` and the pointwise/matmul kernels. The overlay below is
that same sequence, which keeps the observed semantics identical:

* only the lower triangle (strict upper entries ignored) is read;
* negative eigenvalues are clamped to zero, so an indefinite input returns the
  square root of its positive part, as on CPU;
* the result is symmetric (Hermitian) and contiguous, with the input dtype;
* low-precision dtypes and non-floating inputs raise the CPU kernel's errors.

Autograd is unchanged: the generated ``linalg_matrix_sqrth`` formula builds its
backward graph from ``_linalg_eigh``, which is also present on MPS, so gradients
flow without a custom backward here.

This is decomposed coverage, not a universal GPU guarantee. torch 2.14.1's MPS
``eigh`` runs its Metal Jacobi kernel only while the full n x n matrix and
eigenvector buffer fit the 32 KiB threadgroup budget (n <= 64 for float32,
n <= 45 for complex64) and ``batch * n * n >= 12288``; outside that region the
built-in kernel copies to the CPU. The overlay removes the missing-op error and
keeps CPU semantics in both regions, but only the Metal region is GPU-native,
and its Jacobi tolerance is 1e-6 (forward and gradient accuracy ~1e-6..5e-5
instead of the ~1e-7 the CPU branch gives).

Importing this module registers nothing. ``OVERLAYS`` is the table the
accelerator installs on the MPS dispatch key.
"""

import torch

_LOW_PRECISION = {torch.float16: "Half", torch.bfloat16: "BFloat16"}
_OTHER_DTYPES = {
    torch.int8: "Char",
    torch.uint8: "Byte",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.bool: "Bool",
}


def matrix_sqrth(self):
    if self.dim() < 2:
        raise RuntimeError(
            "linalg.matrix_sqrth: The input tensor A must have at least 2 dimensions."
        )
    if self.shape[-2] != self.shape[-1]:
        raise RuntimeError(
            "linalg.matrix_sqrth: A must be batches of square matrices, but they are "
            f"{self.shape[-2]} by {self.shape[-1]} matrices"
        )
    if self.dtype in _LOW_PRECISION:
        raise RuntimeError(
            "linalg.matrix_sqrth: Low precision dtypes not supported. "
            f"Got {_LOW_PRECISION[self.dtype]}"
        )
    if not (self.dtype.is_floating_point or self.dtype.is_complex):
        name = _OTHER_DTYPES.get(self.dtype) or str(self.dtype).replace("torch.", "")
        raise RuntimeError(
            "linalg.matrix_sqrth: Expected a floating point or complex tensor as "
            f"input. Got {name}"
        )

    eigenvalues, eigenvectors = torch.linalg.eigh(self)
    eigenvalues = eigenvalues.clamp_min(0).sqrt()
    if eigenvectors.is_complex():
        eigenvalues = eigenvalues.to(eigenvectors.dtype)
    return (eigenvectors * eigenvalues.unsqueeze(-2)) @ eigenvectors.mH


OVERLAYS = (("linalg_matrix_sqrth", matrix_sqrth),)
