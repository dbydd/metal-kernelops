"""MPS overlays for ``aten::_spdiags``.

``torch.sparse.spdiags`` reaches ``aten::_spdiags``, which has a CPU kernel and
no MPS kernel in torch 2.14.1, so the call raises ``NotImplementedError``. This
module rebuilds the CPU result as a sparse COO tensor on MPS: the sparse
structure is computed from the offsets and the requested shape, and only the
values are gathered on the device.

Semantics are those of the CPU kernel (``aten/src/ATen/native/sparse/
SparseCsrTensorMath.cpp``), not the scipy convention:

* ``diagonals`` is ``(n_diags,)`` or ``(n_diags, diag_cols)``; a 1-D input is
  one diagonal and ``offsets`` must then be a scalar or length 1.
* ``offsets[i] == k`` places element ``j`` of row ``i`` at output
  ``(j - k, j)``. ``j`` runs over ``[max(0, k), max(0, k) + n_k)`` where
  ``n_k = min(rows + k, diag_cols)`` for ``k <= 0`` and
  ``n_k = min(cols, diag_cols) - k`` for ``k > 0``.
* Entries are emitted diagonal by diagonal in the given offsets order, each
  diagonal with increasing ``j`` -- the same order as the CPU kernel.
* Positive offsets therefore use ``diagonals[i, k:]``, and the count is not
  clipped by ``rows``: a diagonal row longer than the matrix makes the CPU
  kernel emit row indices past ``rows - 1`` on a rectangular shape, and this
  port reproduces that byte for byte rather than silently fixing the structure.
* Offsets outside the matrix give ``n_k == 0`` and add nothing. When
  ``sum(n_k) < 0`` the CPU kernel fails inside ``torch.empty``; the same
  ``RuntimeError`` text is raised here.

One CPU behavior is deliberately not reproduced: when a single ``n_k`` is
negative but the total is not (for example ``shape=(3, 3)``, ``diag_cols=3``,
``offsets=[4, 0]``), the CPU kernel allocates ``sum(n_k)`` entries and then
writes the negative diagonal's block with an out-of-bounds offset. The result
is undefined. This port raises ``RuntimeError`` instead of writing past the
buffer.

``layout`` may be ``torch.sparse_coo`` (default) only. Sparse compressed
tensors cannot be constructed on the MPS backend in 2.14.1
(``torch.sparse_csr_tensor(..., device="mps")`` fails with
``new_compressed_tensor``), so ``torch.sparse_csr`` and ``torch.sparse_csc``
raise ``NotImplementedError`` here instead of returning a COO tensor under a
different layout name.

``diagonals`` must live on MPS. A call whose only MPS tensor is ``offsets``
(diagonals on the CPU) has no sensible output device and raises
``NotImplementedError`` instead of silently moving the result.

Cost: one device-to-host read of ``offsets`` when the offsets live on MPS
(needed because ``nnz`` and the duplicate check are data-dependent), none when
they live on CPU. The per-diagonal counts, the starts of their blocks and the
column at which each diagonal begins are computed on the host from that read
(``O(n_diags)``, not ``O(nnz)``) and uploaded as three ``int64`` rows. One
Metal kernel then writes the ``(2, nnz)`` index matrix and the linear gather
offsets, one ``index_select`` fetches the values, and one sparse constructor
call wraps them; nothing is computed per element on the host.

``_spdiags.out`` copies the result into ``out``. A dense ``out`` fails the same
way the CPU kernel's does (``copy_`` between dense and sparse); a sparse ``out``
whose shape differs from ``shape`` fails in ``resize_``, as on CPU, because
sparse tensors cannot be resized, and an ``out`` on another device fails with
the dispatcher's device-mismatch error. Autograd is not implemented for the
operator in 2.14.1, so backward raises ``derivative for aten::_spdiags is not
implemented`` on MPS exactly as on CPU.

Importing this module neither registers nor compiles anything; the shader is
compiled on the first call, once per process.
"""

from functools import lru_cache

import torch

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// meta = [block start, first column, offset] x n_diags, row-major.
// Thread e writes the (row, column) pair of the e-th stored entry and the
// flat index of its value in diagonals.reshape(-1). The block starts are
// non-decreasing, so an upper-bound binary search finds the diagonal.
kernel void spdiags_indices(device const long* meta,
                            device long* rows, device long* columns,
                            device long* linear,
                            constant uint& nnz, constant uint& n_diags,
                            constant uint& diag_cols,
                            uint i [[thread_position_in_grid]]) {
    if (i >= nnz) return;
    long e = (long)i;
    uint lo = 0, hi = n_diags;
    while (lo < hi) {
        uint mid = (lo + hi) >> 1;
        if (meta[mid] <= e) lo = mid + 1; else hi = mid;
    }
    uint d = lo - 1;
    long j = e - meta[d] + meta[n_diags + d];
    columns[i] = j;
    rows[i] = j - meta[2 * n_diags + d];
    linear[i] = (long)d * (long)diag_cols + j;
}
"""

_THREADS_PER_GROUP = 256

_LAYOUT_NAMES = {
    torch.strided: "Strided",
    torch.sparse_coo: "Sparse",
    torch.sparse_csr: "SparseCsr",
    torch.sparse_csc: "SparseCsc",
    torch.sparse_bsr: "SparseBsr",
    torch.sparse_bsc: "SparseBsc",
}

_DTYPE_NAMES = {
    torch.bool: "Bool",
    torch.int8: "Char",
    torch.uint8: "Byte",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.float16: "Half",
    torch.float32: "Float",
    torch.float64: "Double",
    torch.bfloat16: "BFloat16",
    torch.complex64: "ComplexFloat",
    torch.complex128: "ComplexDouble",
}


def _offsets_host(offsets):
    """The offsets as a Python list. One synchronization when they live on MPS."""
    values = offsets.tolist()
    return [values] if offsets.dim() == 0 else values


def _counts(offsets, rows, cols, diag_cols):
    """Per-diagonal entry counts, exactly the CPU kernel's formula."""
    k = torch.tensor(offsets, dtype=torch.int64)
    positive = k > 0
    non_positive = torch.minimum(rows + k, torch.tensor(diag_cols, dtype=torch.int64))
    positive_counts = torch.minimum(torch.tensor(cols, dtype=torch.int64),
                                    torch.tensor(diag_cols, dtype=torch.int64)) - k
    return torch.where(positive, positive_counts, non_positive)


@lru_cache(maxsize=1)
def _shader():
    return torch.mps.compile_shader(_SOURCE)


def _sparse(diagonals, offsets, shape, layout):
    if diagonals.dim() not in (1, 2):
        raise RuntimeError("Diagonals must be vector or matrix")
    if len(shape) != 2:
        raise RuntimeError("Output shape must be 2d")
    if offsets.dim() not in (0, 1):
        raise RuntimeError("Offsets must be scalar or vector")

    n_diags = 1 if diagonals.dim() == 1 else diagonals.size(0)
    n_offsets = 1 if offsets.dim() == 0 else offsets.size(0)
    if n_diags != n_offsets:
        raise RuntimeError(
            f"Number of diagonals ({n_diags}) does not match the number of "
            f"offsets ({n_offsets})"
        )
    if layout is not None and layout not in (
        torch.sparse_coo, torch.sparse_csr, torch.sparse_csc
    ):
        raise RuntimeError(
            "Only output layouts (Sparse, SparseCsc, SparseCsr) are supported, "
            f"got {_LAYOUT_NAMES.get(layout, layout)}"
        )
    if offsets.dtype != torch.int64:
        raise RuntimeError(
            "Offset Tensor must have dtype Long but got "
            f"{_DTYPE_NAMES.get(offsets.dtype, offsets.dtype)}"
        )
    if diagonals.device.type != "mps":
        raise NotImplementedError(
            "the MPS overlay for aten::_spdiags needs diagonals on the MPS "
            f"device, got {diagonals.device}"
        )

    offsets = _offsets_host(offsets)
    if len(set(offsets)) != len(offsets):
        raise RuntimeError("Offset tensor contains duplicate values")

    if layout is not None and layout is not torch.sparse_coo:
        raise NotImplementedError(
            f"spdiags with layout={_LAYOUT_NAMES[layout]} is not available on "
            "MPS: sparse compressed tensors cannot be constructed on the MPS "
            "backend in torch 2.14.1. Use layout=torch.sparse_coo, or build the "
            "COO result and convert it on a CPU the compressed layout is "
            "actually needed on."
        )

    rows, cols = int(shape[0]), int(shape[1])
    diag_cols = diagonals.size(0) if diagonals.dim() == 1 else diagonals.size(1)
    if rows == 0 or cols == 0 or not offsets:
        counts = torch.empty(0, dtype=torch.int64)
        nnz = 0
    else:
        counts = _counts(offsets, rows, cols, diag_cols)
        nnz = int(counts.sum())
    if nnz < 0:
        raise RuntimeError(
            f"Trying to create tensor with negative dimension {nnz}: [2, {nnz}]"
        )
    if counts.numel() and int(counts.min()) < 0:
        bad = offsets[int(counts.argmin())]
        raise RuntimeError(
            f"spdiags: offset {bad} gives a negative diagonal length "
            f"({int(counts.min())}) for shape [{rows}, {cols}] with "
            f"{diag_cols} diagonal columns; the CPU kernel writes out of "
            "bounds on this input, so the MPS overlay rejects it. Trim the "
            "diagonals row to the longest diagonal that fits, or drop the "
            "offset."
        )

    device = diagonals.device
    if nnz:
        offsets_host = torch.tensor(offsets, dtype=torch.int64)
        meta = torch.stack(
            (counts.cumsum(0) - counts, offsets_host.clamp_min(0), offsets_host)
        ).to(device)
        indices = torch.empty((2, nnz), dtype=torch.int64, device=device)
        linear = torch.empty(nnz, dtype=torch.int64, device=device)
        _shader().spdiags_indices(
            meta, indices[0], indices[1], linear,
            nnz, len(offsets), diag_cols,
            threads=nnz, group_size=_THREADS_PER_GROUP,
        )
        values = diagonals.reshape(-1).index_select(0, linear)
    else:
        indices = torch.empty((2, 0), dtype=torch.int64, device=device)
        values = torch.empty(0, dtype=diagonals.dtype, device=device)

    return torch.ops.aten._sparse_coo_tensor_unsafe.default(
        indices, values, [rows, cols], is_coalesced=nnz <= 1
    )


def spdiags(diagonals, offsets, shape, layout=None):
    """Overlay for ``aten::_spdiags`` on MPS."""
    return _sparse(diagonals, offsets, shape, layout)


def spdiags_out(diagonals, offsets, shape, layout=None, *, out):
    """Overlay for ``aten::_spdiags.out`` on MPS."""
    result = _sparse(diagonals, offsets, shape, layout)
    if tuple(out.shape) != tuple(result.shape):
        out.resize_(result.shape)
    return out.copy_(result)


OVERLAYS = (
    ("_spdiags", spdiags),
    ("_spdiags.out", spdiags_out),
)
