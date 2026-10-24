"""``aten::sspaddmm`` for sparse COO input on MPS.

``torch.sspaddmm(self, mat1, mat2, beta=1, alpha=1)`` is
``beta * self + alpha * (mat1 @ mat2)`` with ``self`` and ``mat1`` sparse COO and
``mat2`` strided. The CPU kernel is the contract:

* the result is COO with ``is_coalesced() == False``;
* its first ``self._nnz()`` entries are ``self``'s stored entries in storage order,
  duplicates included, scaled by ``beta``;
* the remaining entries are, for every row of the *coalesced* ``mat1`` that stores at
  least one entry, all ``mat2.size(1)`` columns in column order, scaled by ``alpha``.
  Structurally present zeros are kept, so the count is
  ``self._nnz() + rows(coalesced mat1) * mat2.size(1)``; a result with no stored entries
  is reported coalesced, which is what the CPU kernel does.

The product comes from the native SparseMPS ``mm``; its dense result is immediately
restricted to the occupied rows, so the returned COO never stores it.
``torch.unique_consecutive`` over the coalesced row index is the only data-dependent
shape in the call: it materialises the occupied-row count, and with it the output nnz,
on the host. That single scalar synchronisation is measured in
``benchmarks/verify_sspaddmm.py``.

MPS has no float64, so the dtype set is what the native sparse ``mm`` serves: float32,
float16, bfloat16, int8/16/32/64 and complex64. The CPU kernel rejects float16 and
bfloat16 (``"sspmm" not implemented``) and additionally serves float64 and complex128.
"""

import torch

_MATRICES = "sspaddmm: Argument #{}: matrices expected, got {}D tensor"


def _check(self, mat1, mat2):
    if not self.is_sparse:
        raise RuntimeError(
            f"sspaddmm: expected 'self' to have sparse layout, got 'self' with layout {self.layout}")
    if not mat1.is_sparse:
        raise RuntimeError(
            f"sspaddmm: expected 'mat1' to have sparse layout, got 'mat1' with layout {mat1.layout}")
    if mat2.layout != torch.strided:
        raise RuntimeError(
            f"sspaddmm: expected 'mat2' to have strided layout, got 'mat2' with layout {mat2.layout}")
    if self.sparse_dim() != 2:
        raise RuntimeError(_MATRICES.format(1, self.sparse_dim()))
    if mat1.sparse_dim() != 2:
        raise RuntimeError(_MATRICES.format(2, mat1.sparse_dim()))
    if mat2.dim() != 2:
        raise RuntimeError(_MATRICES.format(3, mat2.dim()))
    if self.size(0) != mat1.size(0):
        raise RuntimeError(
            f"sspaddmm: Argument #1: Expected dim 0 size {mat1.size(0)}, got {self.size(0)}")
    if self.size(1) != mat2.size(1):
        raise RuntimeError(
            f"sspaddmm: Argument #1: Expected dim 1 size {mat2.size(1)}, got {self.size(1)}")
    if mat1.size(1) != mat2.size(0):
        raise RuntimeError(
            f"sspaddmm: Argument #3: Expected dim 0 size {mat1.size(1)}, got {mat2.size(0)}")
    if self.dtype != mat1.dtype or self.dtype != mat2.dtype:
        raise RuntimeError(
            "sspaddmm: expected 'self', 'mat1' and 'mat2' to share a dtype, got "
            f"{self.dtype}, {mat1.dtype}, {mat2.dtype}")


def _combined(self, mat1, mat2, beta, alpha):
    columns = mat2.size(1)
    coalesced = mat1 if mat1.is_coalesced() else mat1.coalesce()
    rows = torch.unique_consecutive(coalesced._indices()[0])
    occupied = rows.size(0)
    owned = self._nnz()
    device = mat2.device
    indices = torch.empty((2, owned + occupied * columns), dtype=torch.int64, device=device)
    values = torch.empty(owned + occupied * columns, dtype=self.dtype, device=device)
    indices[:, :owned].copy_(self._indices())
    values[:owned].copy_(self._values()).mul_(beta)
    if coalesced._nnz() and occupied:
        # Only the occupied rows of the native sparse product reach the result.
        block = torch.mm(coalesced, mat2).index_select(0, rows)
        indices[0, owned:].view(occupied, columns).copy_(rows.unsqueeze(1).expand(occupied, columns))
        indices[1, owned:].view(occupied, columns).copy_(
            torch.arange(columns, dtype=torch.int64, device=device).unsqueeze(0).expand(occupied, columns))
        values[owned:].view(occupied, columns).copy_(block).mul_(alpha)
    # A tensor with no stored entries is coalesced by definition, and the CPU kernel
    # returns it that way.
    empty = owned + occupied * columns == 0
    return torch.sparse_coo_tensor(indices, values, self.size(), is_coalesced=empty)


def sspaddmm(self, mat1, mat2, *, beta=1, alpha=1):
    _check(self, mat1, mat2)
    return _combined(self, mat1, mat2, beta, alpha)


def sspaddmm_out(self, mat1, mat2, *, beta=1, alpha=1, out):
    _check(self, mat1, mat2)
    if not out.is_sparse:
        raise RuntimeError(
            f"sspaddmm: expected 'out' to have sparse layout, got 'out' with layout {out.layout}")
    if out.dtype != self.dtype:
        raise RuntimeError(
            f"sspaddmm: expected 'out' to have dtype {self.dtype}, got {out.dtype}")
    out.copy_(_combined(self, mat1, mat2, beta, alpha))
    return out


SPARSE_OVERLAYS = (("sspaddmm", sspaddmm), ("sspaddmm.out", sspaddmm_out))
