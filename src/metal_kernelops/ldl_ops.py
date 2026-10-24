"""MPS overlays for ``aten::linalg_ldl_factor_ex`` and ``aten::linalg_ldl_solve``.

Both operators have no MPS kernel in torch 2.14.1: every call raises
``NotImplementedError`` before it reaches a backend. The overlays here register
the ``.out`` variants, which is where the functional forms redispatach, and run
the factorization and the solves with aten ops MPS already supports. Nothing is
copied to the CPU and no scalar is read back: every branch of the sequential
pivot search is a device-side mask, so the whole op stays on the GPU.

Semantics are LAPACK's ``sytrf``/``sytrs`` (``hetrf``/``hetrs`` for complex
Hermitian input) with ``uplo='L'``, which is what torch's own CPU and CUDA
kernels call:

* Only the lower triangle of ``self`` is read; the upper triangle is ignored,
  exactly as in ``sytrf``.
* ``LD`` holds ``L`` (unit lower triangular) and the block diagonal ``D`` in
  ``sytrf``'s packed layout: 1x1 blocks on the diagonal, 2x2 blocks on the
  diagonal plus the subdiagonal entry ``LD[k+1, k]``. The strictly upper
  triangle is zero.
* ``pivots`` is LAPACK's 1-based ``ipiv``: positive for a 1x1 block at row
  ``ipiv[k]-1``, negative (both entries) for a 2x2 block.
* ``info`` is the first column that is exactly zero (or has a NaN diagonal).
  ``check_errors=True`` raises a ``RuntimeError``: torch's own message for a
  positive ``info`` is an internal-assert "Unknown error code: N", reproduced
  here without the build path in the prefix.

The algorithms are the unblocked Bunch-Kaufman recurrences of ``dsytf2`` /
``zhetf2`` (factor) and ``dsytrs`` / ``zhetrs`` (solve), expressed as batched
tensor operations. The pivot search is inherently sequential, so the trailing
update is one rank-1 or rank-2 slice per column, and 2x2 blocks are padded into
the same shape as 1x1 blocks so that a whole batch can be advanced in lockstep
whatever its pivot sequence looks like.

This is a correctness-only port. Both ops are sequential dependency chains, so
the factorization costs ``n`` rounds of roughly thirty small kernels and the
launch overhead dominates everything else: measured on this machine against the
same matrices on the CPU, the factor is 200x to 3500x slower and the solve 10x to
500x (see ``measured/ldl.json``). Nothing about that ratio improves with size.
Use these overlays where a hard ``NotImplementedError`` is worse than a slow
answer, not where the op is hot; a Metal kernel that keeps one matrix per
threadgroup is what the timings call for.
"""

import torch

_ALPHA = (1.0 + 17.0 ** 0.5) / 8.0

_FLOAT_NAMES = {
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.float32: "Float",
    torch.float64: "Double",
    torch.complex64: "ComplexFloat",
    torch.complex128: "ComplexDouble",
    torch.int8: "Char",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.bool: "Bool",
}


def _dtype_name(dtype):
    return _FLOAT_NAMES.get(dtype, str(dtype).replace("torch.", ""))


def _use_conj(hermitian, dtype):
    """Whether the factorization is A = L D L^H rather than A = L D L^T.

    For real dtypes ``hermitian`` has no effect on the CPU/CUDA kernels, and the
    real Bunch-Kaufman recurrences are used either way.
    """
    return bool(hermitian) and dtype.is_complex


def _check_dtype(a, name):
    if a.dtype not in (torch.float32, torch.complex64):
        raise NotImplementedError(
            f"{name}: no kernel for MPS tensors of dtype {_dtype_name(a.dtype)}"
        )


def _check_factor_input(a, name):
    if a.dim() < 2:
        raise RuntimeError(
            f"{name}: The input tensor A must have at least 2 dimensions."
        )
    if a.shape[-2] != a.shape[-1]:
        raise RuntimeError(
            f"{name}: A must be batches of square matrices, but they are "
            f"{a.shape[-2]} by {a.shape[-1]} matrices"
        )
    if not (a.is_floating_point() or a.is_complex()):
        raise RuntimeError(
            f"{name}: Expected a floating point or complex tensor as input. "
            f"Got {_dtype_name(a.dtype)}"
        )


def mag(x):
    """LAPACK's CABS1 (``|Re z| + |Im z|``) for complex input, ``|x|`` for real."""
    if x.is_complex():
        return x.real.abs() + x.imag.abs()
    return x.abs()


def _factor(A, use_conj):
    """Batched ``sytrf`` on the lower triangle of ``A`` (``(B, n, n)``, in place).

    Returns ``(ipiv, info)`` with ``ipiv`` ``(B, n)`` int64, 1-based and signed
    as LAPACK does, and ``info`` ``(B,)`` int64.
    """
    B, n, _ = A.shape
    device = A.device
    ipiv = torch.zeros((B, n), dtype=torch.int64, device=device)
    info = torch.zeros((B,), dtype=torch.int64, device=device)
    if n == 0:
        return ipiv, info

    cols = torch.arange(n, device=device)
    # [k, j] -> j >= k, used to isolate row segments of the pivot row.
    ge = cols.view(1, n) >= cols.view(n, 1)
    lower = torch.tril(torch.ones((n, n), dtype=torch.bool, device=device))
    first_row = torch.zeros((1, n), dtype=torch.bool, device=device)
    first_row[:, 0] = True
    diag = A.diagonal(dim1=-2, dim2=-1)  # view
    skip = torch.zeros((B,), dtype=torch.bool, device=device)

    for k in range(n):
        m = n - k - 1
        # LAPACK measures the pivot column with CABS1 for complex input and with
        # the absolute value for real input; the Hermitian routines use the real
        # part of the diagonal alone.
        dmag = mag(diag)
        absakk = diag[:, k].real.abs() if use_conj else dmag[:, k]
        Aabs = mag(A)

        if m > 0:
            colmax, imax = Aabs[:, k + 1:, k].max(-1)
            imax = imax + (k + 1)
        else:
            colmax = torch.zeros_like(absakk)
            imax = torch.full((B,), k, dtype=torch.int64, device=device)

        # ROWMAX: largest off-diagonal magnitude of the pivot row, columns k..n.
        rowabs = Aabs.gather(-2, imax.view(B, 1, 1).expand(B, 1, n)).squeeze(-2)
        keep = ge[k] & (cols != imax.view(B, 1))
        rowmax = rowabs.masked_fill(~keep, 0).max(-1).values
        dimax = dmag.gather(-1, imax.view(B, 1)).squeeze(-1)

        degenerate = (torch.maximum(absakk, colmax) == 0) | (absakk != absakk)
        c1 = absakk >= _ALPHA * colmax
        c2 = absakk >= _ALPHA * colmax * (colmax / rowmax)
        c3 = dimax >= _ALPHA * rowmax

        # A skipped index is never examined by sytrf: it advances in steps of
        # two over a 2x2 block, so only the leading index can set INFO.
        info = torch.where(
            (info == 0) & degenerate & ~skip, torch.full_like(info, k + 1), info
        )
        live = ~skip & ~degenerate
        interchange = live & ~c1 & ~c2
        two = interchange & ~c3

        # A skipped step is the second half of a 2x2 block whose ipiv entry was
        # already written by the step that opened the block.
        entry = torch.where(two, -(imax + 1), torch.where(interchange, imax + 1, k + 1))
        ipiv[:, k] = torch.where(skip, ipiv[:, k], entry)
        if m > 0:
            ipiv[:, k + 1] = torch.where(two, -(imax + 1), ipiv[:, k + 1])

        # Interchange rows/columns KK = k + kstep - 1 and KP in A[k:, k:]. With
        # the matrix kept symmetric this is a plain symmetric permutation, which
        # is exactly the three DSWAPs plus the diagonal exchange of sytrf.
        # Relative to A[k:, k:]: KK - k is 0 for a 1x1 block, 1 for a 2x2 one,
        # and KP - k equals KK - k when no interchange is taken.
        kk = two.to(torch.int64).view(B, 1)
        kprel = torch.where(interchange, imax - k, torch.zeros_like(imax)).view(B, 1)
        base = torch.arange(n - k, device=device).expand(B, n - k)
        idx = torch.where(base == kk, kprel, base)
        idx = torch.where(base == kprel, kk, idx)
        block = A[:, k:, k:]
        block = block.gather(-2, idx.view(B, n - k, 1).expand(B, n - k, n - k))
        block = block.gather(-1, idx.view(B, 1, n - k).expand(B, n - k, n - k))
        A[:, k:, k:] = block

        if m > 0:
            w0 = A[:, k + 1:, k]
            w1 = A[:, k + 1:, k + 1]
            wold = torch.stack((w0, w1), dim=-1)
            d0 = diag[:, k]
            d10 = A[:, k + 1, k]
            if use_conj:
                absd21 = d10.abs()
                e11 = diag[:, k + 1].real / absd21
                e22 = d0.real / absd21
                two_by_two = 1.0 / (e11 * e22 - 1.0)
                e21 = d10 / absd21
                scale = two_by_two / absd21
                wk = scale.view(B, 1) * (
                    e11.view(B, 1) * w0 - e21.view(B, 1) * w1
                )
                wkp1 = scale.view(B, 1) * (
                    e22.view(B, 1) * w1 - e21.conj().view(B, 1) * w0
                )
            else:
                e11 = diag[:, k + 1] / d10
                e22 = d0 / d10
                two_by_two = 1.0 / (e11 * e22 - 1.0)
                scale = two_by_two / d10
                wk = scale.view(B, 1) * (e11.view(B, 1) * w0 - w1)
                wkp1 = scale.view(B, 1) * (e22.view(B, 1) * w1 - w0)

            livev = live.view(B, 1)
            two_v = two.view(B, 1)
            one_by_one = w0 * (
                1.0 / (d0.real if use_conj else d0)
            ).view(B, 1)
            # The 2x2 scaling only applies to a 2x2 block; a 1x1 block that did
            # take an interchange still scales its own column by 1/d.
            new0 = torch.where(two_v, wk, one_by_one)
            # Column k+1 of W is only written by a 2x2 block; it must stay out of
            # the update for a 1x1 block.
            new1 = torch.where(two_v, wkp1, torch.zeros_like(w1))
            # For a 2x2 block the first trailing row holds D's off-diagonal
            # entry and takes part in neither the update nor the scaling. A
            # degenerate or skipped step leaves the whole column alone.
            hold = two_v & first_row[:, :m]
            store0 = torch.where(livev & ~hold, new0, w0)
            store1 = torch.where(hold, w1, new1)
            wnew = torch.stack((store0, new1), dim=-1).masked_fill(
                (hold | ~livev).unsqueeze(-1), 0
            )
            if use_conj:
                update = torch.matmul(wold, wnew.mH)
            else:
                update = torch.matmul(wold, wnew.mT)
            A[:, k + 1:, k + 1:].sub_(update)
            A[:, k + 1:, k] = store0
            A[:, k + 1:, k + 1] = torch.where(
                two.view(B, 1), store1, A[:, k + 1:, k + 1]
            )

            if use_conj:
                # zhetf2 keeps the diagonal of a Hermitian factor real; the
                # rank-2 update here leaves a rounding-level imaginary part.
                diag.copy_(diag.real)
                A.copy_(torch.where(lower, A, A.mT.conj()))
            else:
                A.copy_(torch.where(lower, A, A.mT))

        skip = two

    return ipiv, info


def _solve(LD, pivots, B, nrhs, use_conj):
    """Batched ``sytrs``/``hetrs`` given ``LD``/``pivots`` from :func:`_factor`."""
    Bn, n, _ = LD.shape
    device = LD.device
    X = B
    if n == 0 or nrhs == 0 or Bn == 0:
        return X

    ipiv = pivots.to(torch.int64)
    neg = ipiv < 0
    # 2x2 blocks pair consecutive indices, so a run of negative ipiv entries of
    # length L is floor(L/2) blocks. second[k] marks the trailing index of a
    # block: it is the even position inside the run of negatives ending at k.
    # (Two neighbouring blocks can share a pivot row, so testing the sign of
    # ipiv[k-1] alone is not enough.)
    pos = torch.arange(n, device=device).expand(Bn, n)
    run = pos - torch.where(neg, -1, pos).cummax(-1).values
    second = neg & (run % 2 == 0)
    skipb = torch.zeros((Bn, n), dtype=torch.bool, device=device)
    skipb[:, :-1] = second[:, 1:]
    base_rows = torch.arange(n, device=device).expand(Bn, n)
    first = torch.zeros((1, n), dtype=torch.bool, device=device)
    first[:, 0] = True

    def swap_rows(row, kp):
        idx = torch.where(base_rows == row.view(Bn, 1), kp.view(Bn, 1), base_rows)
        idx = torch.where(base_rows == kp.view(Bn, 1), row.view(Bn, 1), idx)
        X.copy_(X.gather(-2, idx.view(Bn, n, 1).expand(Bn, n, nrhs)))

    def pivot_rows(k, two, idle, row):
        # ipiv is 1-based. A step that is the trailing half of a 2x2 block has
        # already been handled and must leave the rows where they are.
        kp = torch.where(two, -ipiv[:, k] - 1, ipiv[:, k] - 1)
        return torch.where(idle, row, kp)

    def inverse_diagonal(k):
        d = LD[:, k, k]
        return 1.0 / (d.real if use_conj else d)

    # L * D * X = B, forwards.
    for k in range(n):
        idle = second[:, k]
        live = ~idle
        two = live & neg[:, k]
        # A 2x2 block interchanges rows k+1 and KP; a 1x1 block rows k and KP.
        row = torch.where(two, k + 1, k)
        swap_rows(row, pivot_rows(k, two, idle, row))
        if k + 1 < n:
            w0 = LD[:, k + 1:, k]
            w1 = LD[:, k + 1:, k + 1]
            # A 2x2 block leaves its second row alone; an idle step does nothing.
            hold = (two.view(Bn, 1) & first[:, : n - k - 1]) | idle.view(Bn, 1)
            u0 = torch.where(hold, torch.zeros_like(w0), w0)
            u1 = torch.where(hold | ~two.view(Bn, 1), torch.zeros_like(w1), w1)
            X[:, k + 1:, :].sub_(u0.view(Bn, n - k - 1, 1) * X[:, k : k + 1, :])
            X[:, k + 1:, :].sub_(u1.view(Bn, n - k - 1, 1) * X[:, k + 1 : k + 2, :])
            d0 = LD[:, k, k]
            d21 = LD[:, k + 1, k]
            d11 = LD[:, k + 1, k + 1]
            if use_conj:
                akm1 = d0 / d21.conj()
                ak = d11 / d21
                denom = akm1 * ak - 1.0
                bkm1 = X[:, k, :] / d21.conj().view(Bn, 1)
                bk = X[:, k + 1, :] / d21.view(Bn, 1)
            else:
                akm1 = d0 / d21
                ak = d11 / d21
                denom = akm1 * ak - 1.0
                bkm1 = X[:, k, :] / d21.view(Bn, 1)
                bk = X[:, k + 1, :] / d21.view(Bn, 1)
            two_v = two.view(Bn, 1)
            r0 = torch.where(
                two_v,
                (ak.view(Bn, 1) * bkm1 - bk) / denom.view(Bn, 1),
                X[:, k, :] * inverse_diagonal(k).view(Bn, 1),
            )
            r1 = torch.where(
                two_v, (akm1.view(Bn, 1) * bk - bkm1) / denom.view(Bn, 1),
                X[:, k + 1, :],
            )
            X[:, k, :] = torch.where(live.view(Bn, 1), r0, X[:, k, :])
            X[:, k + 1, :] = torch.where(live.view(Bn, 1), r1, X[:, k + 1, :])
        else:
            live_v = live.view(Bn, 1)
            X[:, k, :] = torch.where(
                live_v, X[:, k, :] * inverse_diagonal(k).view(Bn, 1), X[:, k, :]
            )

    # L^H * X = B (L^T for the symmetric case), backwards.
    for k in range(n - 1, -1, -1):
        idle = skipb[:, k]
        live = ~idle
        two = second[:, k] & live
        if k + 1 < n:
            seg = LD[:, k + 1:, k]
            if use_conj:
                seg = seg.conj()
            prod = torch.matmul(seg.unsqueeze(-2), X[:, k + 1 :, :])
            X[:, k : k + 1, :].sub_(prod * live.view(Bn, 1, 1))
            if k > 0:
                seg2 = LD[:, k + 1:, k - 1]
                if use_conj:
                    seg2 = seg2.conj()
                prod2 = torch.matmul(seg2.unsqueeze(-2), X[:, k + 1 :, :])
                X[:, k - 1 : k, :].sub_(prod2 * two.view(Bn, 1, 1))
        row = torch.full_like(ipiv[:, k], k)
        swap_rows(row, pivot_rows(k, two, idle, row))
    return X


def _ldl_factor_ex_out(self, *, hermitian=False, check_errors=False, LD, pivots, info):
    _check_factor_input(self, "torch.linalg.ldl_factor_ex")
    _check_dtype(self, "torch.linalg.ldl_factor_ex")
    use_conj = _use_conj(hermitian, self.dtype)
    n = self.shape[-1]
    batch = self.shape[:-2]
    size = 1
    for dim in batch:
        size *= dim

    work = torch.tril(self.detach()).reshape(size, n, n).clone()
    if size and n:
        # The packed layout only stores one triangle; the recurrences read the
        # pivot row and column, so the work matrix is kept symmetric/Hermitian.
        if use_conj:
            work = torch.where(
                torch.tril(torch.ones((n, n), dtype=torch.bool, device=self.device)),
                work,
                work.mT.conj(),
            )
        else:
            work = torch.where(
                torch.tril(torch.ones((n, n), dtype=torch.bool, device=self.device)),
                work,
                work.mT,
            )

    if size and n:
        ipiv, info_i = _factor(work, use_conj)
    else:
        ipiv = torch.zeros((size, n), dtype=torch.int64, device=self.device)
        info_i = torch.zeros((size,), dtype=torch.int64, device=self.device)

    ld = torch.empty_strided(
        (size, n, n),
        (n * n, 1, n),
        dtype=self.dtype,
        device=self.device,
    )
    if size and n:
        ld.copy_(torch.tril(work))

    LD.copy_(ld.reshape(batch + (n, n)))
    pivots.copy_(ipiv.reshape(batch + (n,)).to(torch.int32))
    info.copy_(info_i.reshape(batch).to(torch.int32))

    if check_errors and size:
        # torch's CUDA kernel only synchronizes when check_errors=True, and so
        # does this one: the readback is the point of the flag.
        failed = (info_i != 0).nonzero()
        if failed.numel():
            code = int(info_i[failed[0, 0]])
            raise RuntimeError(
                f"torch.linalg.ldl_factor_ex: Unknown error code: {code}."
            )
    return LD, pivots, info


def _ldl_solve_out(LD, pivots, B, *, hermitian=False, out):
    _check_factor_input(LD, "torch.linalg.ldl_solve")
    _check_dtype(LD, "torch.linalg.ldl_solve")
    if B.dim() < 2:
        raise RuntimeError(
            "torch.linalg.ldl_solve: Expected B to have at least 2 dimensions, "
            f"but it has {B.dim()} dimensions instead"
        )
    n = LD.shape[-1]
    if B.shape[-2] != n:
        raise RuntimeError(
            "torch.linalg.ldl_solve: Expected B.shape[-2] to be "
            f"{n}, but got {B.shape[-2]} instead"
        )
    if pivots.shape != LD.shape[:-1]:
        raise RuntimeError(
            "torch.linalg.ldl_solve: Expected LD.shape[:-1] and pivots.shape to "
            f"be the same, but got pivots with shape {tuple(pivots.shape)} instead"
        )
    if pivots.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise RuntimeError(
            "torch.linalg.ldl_solve: Expected pivots to be integers. Got "
            f"{_dtype_name(pivots.dtype)}"
        )
    if LD.dtype != B.dtype:
        raise RuntimeError(
            f"torch.linalg.ldl_solve: LD dtype {_dtype_name(LD.dtype)} does not "
            f"match b dtype {_dtype_name(B.dtype)}"
        )
    use_conj = _use_conj(hermitian, LD.dtype)
    nrhs = B.shape[-1]
    batch = torch.broadcast_shapes(LD.shape[:-2], B.shape[:-2])
    size = 1
    for dim in batch:
        size *= dim

    ld = LD.expand(batch + (n, n)).reshape(size, n, n)
    piv = pivots.expand(batch + (n,)).reshape(size, n)
    rhs = B.expand(batch + (n, nrhs)).reshape(size, n, nrhs).clone()
    x = _solve(ld, piv, rhs, nrhs, use_conj)
    out.copy_(x.reshape(batch + (n, nrhs)))
    return out


OVERLAYS = (
    ("linalg_ldl_factor_ex.out", _ldl_factor_ex_out),
    ("linalg_ldl_solve.out", _ldl_solve_out),
)
