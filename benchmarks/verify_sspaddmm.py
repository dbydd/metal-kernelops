"""Verify and measure the SparseMPS ``aten::sspaddmm`` overlay.

The overlay is registered here, not at import of ``metal_kernelops.sparse_addmm_ops``.
The CPU kernel is the reference. Sections: the structural contract (nnz, index order,
values, layout, coalesced flag) over coalesced and uncoalesced sparse operands,
duplicate entries, explicit zeros and a ``beta``/``alpha`` grid; the dtype matrix;
``out=`` behaviour; autograd; the cost of the one data-dependent shape in the call;
warmed, synchronized CPU-vs-MPS timing on realistic sparse operands; and a
``torch.nn.Module`` caller.

Writes ``measured/sspaddmm.json``. Run with:

    uv run --project /Users/dbydd/vibe-agent-working-dir/ResearchChamber \
        python benchmarks/verify_sspaddmm.py
"""

import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.library import Library

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metal_kernelops.sparse_addmm_ops import SPARSE_OVERLAYS  # noqa: E402

JSON_PATH = ROOT / "measured" / "sspaddmm.json"
DEVICE = "mps"
DISPATCH_KEY = "SparseMPS"

_LIBRARY = None


def install():
    library = Library("aten", "FRAGMENT")
    for name, function in SPARSE_OVERLAYS:
        library.impl(name, function, DISPATCH_KEY)
    return library


def median(values):
    return statistics.median(values)


def timed(function, batches, inner=1):
    for _ in range(4):
        function()
    torch.mps.synchronize()
    samples = []
    for _ in range(batches):
        start = time.perf_counter()
        for _ in range(inner):
            function()
        torch.mps.synchronize()
        samples.append((time.perf_counter() - start) / inner)
    return median(samples), min(samples)


def host_only_ms(function, batches, inner):
    """Enqueue without synchronising: dispatch and Python cost, no GPU wait."""
    for _ in range(3):
        function()
    torch.mps.synchronize()
    samples = []
    for _ in range(batches):
        start = time.perf_counter()
        for _ in range(inner):
            function()
        samples.append((time.perf_counter() - start) / inner)
        torch.mps.synchronize()
    return median(samples), min(samples)


def values_for(count, dtype, device="cpu"):
    if dtype.is_complex:
        return torch.randn(count, dtype=dtype, device=device)
    if dtype.is_floating_point:
        return torch.randn(count, dtype=dtype, device=device)
    return torch.randint(-3, 4, (count,), dtype=dtype, device=device)


def make_sparse(rows, columns, count, dtype, device="cpu", coalesce=False):
    row_index = torch.randint(0, rows, (count,), device=device)
    column_index = torch.randint(0, columns, (count,), device=device)
    tensor = torch.sparse_coo_tensor(
        torch.stack([row_index, column_index]),
        values_for(count, dtype, device),
        (rows, columns),
        dtype=dtype,
        device=device,
    )
    return tensor.coalesce() if coalesce else tensor


def structured_cases():
    """Hand-built operands where the CPU index order is known to differ per input order."""
    base_self = torch.tensor([[0, 0, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2]])
    base_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    coalesced_self = torch.sparse_coo_tensor(base_self, base_values, (4, 4)).coalesce()
    tri = torch.tensor([[2, 0, 1], [1, 0, 2]])
    tri_values = torch.tensor([7.0, 1.0, 1.0])
    mat1 = torch.sparse_coo_tensor(tri, tri_values, (4, 3))
    mat2 = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    return {
        "coalesced operands": (coalesced_self, mat1, mat2, 1, 1),
        "unsorted mat1 rows": (
            coalesced_self,
            torch.sparse_coo_tensor(torch.tensor([[2, 0, 1], [1, 0, 2]]), torch.tensor([7.0, 1.0, 1.0]), (4, 3)),
            mat2, 1, 1),
        "duplicate mat1 rows": (
            coalesced_self,
            torch.sparse_coo_tensor(torch.tensor([[2, 0, 2], [1, 0, 0]]), torch.tensor([7.0, 1.0, 3.0]), (4, 3)),
            mat2, 1, 1),
        "uncoalesced self with duplicates": (
            torch.sparse_coo_tensor(torch.tensor([[0, 0, 0, 1], [0, 0, 1, 2]]), torch.tensor([1.0, 2.0, 10.0, 20.0]), (4, 4)),
            mat1, mat2, 1, 1),
        "explicit zeros both sides": (
            torch.sparse_coo_tensor(
                torch.tensor([[2, 0, 0, 0, 1, 2], [1, 0, 0, 2, 1, 0]]),
                torch.tensor([0.0, 5.0, 5.0, 0.0, 7.0, 9.0]), (4, 4)),
            torch.sparse_coo_tensor(
                torch.tensor([[2, 1, 2, 0], [0, 2, 0, 1]]), torch.tensor([0.0, 3.0, 4.0, 5.0]), (4, 3)),
            mat2, 1, 1),
        "mat1 covers every row": (
            coalesced_self,
            torch.sparse_coo_tensor(torch.stack([torch.arange(4), torch.zeros(4, dtype=torch.long)]), torch.ones(4), (4, 3)),
            mat2, 1, 1),
        "mat2 with a zero row": (
            coalesced_self, mat1,
            torch.cat([torch.zeros(1, 4), mat2[1:]], dim=0), 1, 1),
        "empty mat1": (
            coalesced_self,
            torch.sparse_coo_tensor(torch.empty(2, 0, dtype=torch.long), torch.empty(0), (4, 3)),
            mat2, 1, 1),
        "empty self": (
            torch.sparse_coo_tensor(torch.empty(2, 0, dtype=torch.long), torch.empty(0), (4, 4)),
            mat1, mat2, 1, 1),
        "both empty": (
            torch.sparse_coo_tensor(torch.empty(2, 0, dtype=torch.long), torch.empty(0), (4, 4)),
            torch.sparse_coo_tensor(torch.empty(2, 0, dtype=torch.long), torch.empty(0), (4, 3)),
            mat2, 1, 1),
        "beta 0": (coalesced_self, mat1, mat2, 0, 1),
        "alpha 0": (coalesced_self, mat1, mat2, 1, 0),
        "beta and alpha 0": (coalesced_self, mat1, mat2, 0, 0),
        "negative scalars": (coalesced_self, mat1, mat2, -2, 0.5),
        "non-contiguous mat2": (
            coalesced_self, mat1,
            torch.arange(24, dtype=torch.float32).reshape(4, 6).t()[:3], 1, 1),
        "zero columns": (
            torch.sparse_coo_tensor(torch.empty(2, 0, dtype=torch.long), torch.empty(0), (4, 0)),
            mat1, torch.empty(3, 0), 1, 1),
        "zero inner dimension": (
            torch.sparse_coo_tensor(torch.tensor([[0, 2, 3], [1, 4, 2]]), torch.tensor([1.0, 2.0, 3.0]), (4, 5)),
            torch.sparse_coo_tensor(torch.empty(2, 0, dtype=torch.long), torch.empty(0), (4, 0)),
            torch.randn(0, 5), 1, 1),
        "zero rows": (
            torch.sparse_coo_tensor(torch.empty(2, 0, dtype=torch.long), torch.empty(0), (0, 5)),
            torch.sparse_coo_tensor(torch.empty(2, 0, dtype=torch.long), torch.empty(0), (0, 3)),
            torch.randn(3, 5), 1, 1),
    }


def outcome(name, self_cpu, mat1_cpu, mat2_cpu, beta, alpha):
    reference = torch.sspaddmm(self_cpu, mat1_cpu, mat2_cpu, beta=beta, alpha=alpha)
    result = torch.sspaddmm(
        self_cpu.to(DEVICE), mat1_cpu.to(DEVICE), mat2_cpu.to(DEVICE), beta=beta, alpha=alpha)
    difference = (result._values().cpu() - reference._values()).abs()
    dense_reference = beta * self_cpu.to_dense() + alpha * (mat1_cpu.coalesce().to_dense() @ mat2_cpu)
    return {
        "name": name,
        "nnz": [reference._nnz(), result._nnz()],
        "nnz_equal": reference._nnz() == result._nnz(),
        "shape_equal": tuple(reference.shape) == tuple(result.shape),
        "indices_equal": bool(torch.equal(result._indices().cpu(), reference._indices())),
        "index_dtype_equal": result._indices().dtype == reference._indices().dtype,
        "values_max_abs_error": float(difference.max()) if difference.numel() else 0.0,
        "layout_equal": result.layout == reference.layout,
        "coalesced_flag_equal": result.is_coalesced() == reference.is_coalesced(),
        "dtype_equal": result.dtype == reference.dtype,
        "dense_matches_reference": bool(torch.allclose(result.to_dense().cpu(), reference.to_dense())),
        "dense_matches_closed_form": bool(torch.allclose(result.to_dense().cpu(), dense_reference)),
    }


def structure_suite():
    rows = []
    for name, (self_cpu, mat1_cpu, mat2_cpu, beta, alpha) in structured_cases().items():
        row = outcome(name, self_cpu, mat1_cpu, mat2_cpu, beta, alpha)
        row["dense_agreement"] = row.pop("dense_matches_reference") and row.pop("dense_matches_closed_form")
        rows.append(row)

    grid = []
    generated = 0
    exact_index = 0
    bit_exact_values = 0
    value_mismatches = 0
    max_error = 0.0
    for dtype in (torch.float32, torch.int32, torch.int64):
        for rows_count, inner, columns in ((7, 5, 4), (1, 1, 1), (9, 1, 6), (5, 7, 1), (13, 9, 11), (3, 20, 2)):
            for coalesce in (False, True):
                for beta, alpha in ((1, 1), (0, 1), (2, -1), (1, 0), (0, 0), (3, 2)):
                    self_cpu = make_sparse(rows_count, columns, 6, dtype)
                    mat1_cpu = make_sparse(rows_count, inner, 8, dtype, coalesce=coalesce)
                    mat2_cpu = values_for(inner * columns, dtype).reshape(inner, columns)
                    reference = torch.sspaddmm(self_cpu, mat1_cpu, mat2_cpu, beta=beta, alpha=alpha)
                    result = torch.sspaddmm(
                        self_cpu.to(DEVICE), mat1_cpu.to(DEVICE), mat2_cpu.to(DEVICE),
                        beta=beta, alpha=alpha)
                    generated += 1
                    exact_index += int(torch.equal(result._indices().cpu(), reference._indices()))
                    if dtype.is_floating_point:
                        error = float((result._values().cpu() - reference._values()).abs().max())
                        max_error = max(max_error, error)
                        bit_exact_values += int(error == 0.0)
                        value_mismatches += int(error > 1e-5)
                    else:
                        exact = torch.equal(result._values().cpu(), reference._values())
                        bit_exact_values += int(exact)
                        value_mismatches += int(not exact)
                    ok = (
                        result._nnz() == reference._nnz()
                        and tuple(result.shape) == tuple(reference.shape)
                        and result.is_coalesced() == reference.is_coalesced()
                        and result.dtype == reference.dtype
                        and result.layout == reference.layout
                    )
                    if not ok:
                        grid.append({
                            "dtype": str(dtype), "shape": [rows_count, inner, columns],
                            "coalesce": coalesce, "beta": beta, "alpha": alpha,
                            "nnz": [reference._nnz(), result._nnz()],
                        })
    return {
        "structured": rows,
        "structured_failures": [
            row["name"] for row in rows
            if not (row["nnz_equal"] and row["shape_equal"] and row["indices_equal"]
                    and row["index_dtype_equal"] and row["layout_equal"]
                    and row["coalesced_flag_equal"] and row["dtype_equal"]
                    and row["dense_agreement"] and row["values_max_abs_error"] <= 1e-5)
        ],
        "generated_cases": generated,
        "generated_index_mismatches": generated - exact_index,
        "generated_bit_exact_values": bit_exact_values,
        "generated_value_mismatches_beyond_1e-5": value_mismatches,
        "generated_structure_failures": grid,
        "generated_values_max_abs_error": max_error,
    }


def dense_reference(self_cpu, mat1_cpu, mat2_cpu, beta, alpha):
    return beta * self_cpu.to_dense() + alpha * (mat1_cpu.coalesce().to_dense() @ mat2_cpu)


def dtype_suite():
    report = {}
    for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.int8, torch.int16,
                  torch.int32, torch.int64, torch.complex64):
        row = {"served": False}
        try:
            self_cpu = make_sparse(9, 5, 7, dtype)
            mat1_cpu = make_sparse(9, 6, 9, dtype, coalesce=True)
            mat2_cpu = values_for(6 * 5, dtype).reshape(6, 5)
            result = torch.sspaddmm(
                self_cpu.to(DEVICE), mat1_cpu.to(DEVICE), mat2_cpu.to(DEVICE))
            row["served"] = True
            row["dtype_preserved"] = result.dtype == dtype
            row["layout"] = str(result.layout)
            row["coalesced"] = result.is_coalesced()
            if dtype in (torch.float16, torch.bfloat16):
                reference = torch.sspaddmm(self_cpu.float(), mat1_cpu.float(), mat2_cpu.float())
                row["reference"] = "float32 CPU sspaddmm of the same operands"
                row["indices_equal_to_float32_structure"] = (
                    dtype_structure_equal(dtype, self_cpu, mat1_cpu, mat2_cpu))
                row["values_max_abs_error_vs_float32"] = float(
                    (result._values().cpu().float() - reference._values()).abs().max())
            elif dtype.is_complex:
                reference = torch.sspaddmm(self_cpu, mat1_cpu, mat2_cpu)
                row["nnz_equal"] = result._nnz() == reference._nnz()
                row["indices_equal"] = bool(torch.equal(result._indices().cpu(), reference._indices()))
                row["values_max_abs_error"] = float(
                    (result._values().cpu() - reference._values()).abs().max())
                row["values_close"] = bool(torch.allclose(
                    result._values().cpu(), reference._values(), rtol=1e-5, atol=1e-6))
            else:
                reference = torch.sspaddmm(self_cpu, mat1_cpu, mat2_cpu)
                row["nnz_equal"] = result._nnz() == reference._nnz()
                row["indices_equal"] = bool(torch.equal(result._indices().cpu(), reference._indices()))
                error = float((result._values().cpu() - reference._values()).abs().max())
                row["values_max_abs_error"] = error
                row["values_equal"] = bool(torch.equal(result._values().cpu(), reference._values()))
                row["values_close"] = bool(torch.allclose(
                    result._values().cpu(), reference._values(), rtol=1e-5, atol=1e-6))
                row["dense_matches_closed_form"] = bool(torch.allclose(
                    result.to_dense().cpu(),
                    dense_reference(self_cpu, mat1_cpu, mat2_cpu, 1, 1)))
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        report[str(dtype).replace("torch.", "")] = row

    unsupported = {}
    for label, build in {
        "float64": lambda: (
            make_sparse(4, 2, 5, torch.float64), make_sparse(4, 3, 4, torch.float64),
            torch.randn(3, 2, dtype=torch.float64)),
        "bool": lambda: (
            torch.sparse_coo_tensor(torch.tensor([[0, 1], [1, 2]]), torch.ones(2, dtype=torch.bool), (3, 2)).coalesce(),
            torch.sparse_coo_tensor(torch.tensor([[0, 1], [1, 2]]), torch.ones(2, dtype=torch.bool), (3, 3)),
            torch.ones(3, 2, dtype=torch.bool)),
    }.items():
        self_cpu, mat1_cpu, mat2_cpu = build()
        entry = {}
        try:
            torch.sspaddmm(self_cpu.to(DEVICE), mat1_cpu.to(DEVICE), mat2_cpu.to(DEVICE))
            entry["mps"] = "no error"
        except Exception as error:
            entry["mps"] = f"{type(error).__name__}: {str(error)[:200]}"
        try:
            torch.sspaddmm(self_cpu, mat1_cpu, mat2_cpu)
            entry["cpu"] = "no error"
        except Exception as error:
            entry["cpu"] = f"{type(error).__name__}: {str(error)[:200]}"
        unsupported[label] = entry
    report["unsupported"] = unsupported
    return report


def dtype_structure_equal(dtype, self_cpu, mat1_cpu, mat2_cpu):
    reference = torch.sspaddmm(self_cpu.float(), mat1_cpu.float(), mat2_cpu.float())
    result = torch.sspaddmm(self_cpu.to(DEVICE), mat1_cpu.to(DEVICE), mat2_cpu.to(DEVICE))
    return bool(torch.equal(result._indices().cpu(), reference._indices()))


def out_suite():
    base_self = torch.tensor([[0, 0, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2]])
    self_cpu = torch.sparse_coo_tensor(base_self, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), (4, 4)).coalesce()
    mat1_cpu = torch.sparse_coo_tensor(torch.tensor([[2, 0, 1], [1, 0, 2]]), torch.tensor([7.0, 1.0, 1.0]), (4, 3))
    mat2_cpu = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    self_mps, mat1_mps, mat2_mps = self_cpu.to(DEVICE), mat1_cpu.to(DEVICE), mat2_cpu.to(DEVICE)
    expected = torch.sspaddmm(self_cpu, mat1_cpu, mat2_cpu)
    checks = {}

    prefilled = torch.sparse_coo_tensor(torch.tensor([[0, 0], [0, 1]]), torch.tensor([5.0, 5.0]), (4, 4))
    result = torch.sspaddmm(self_mps, mat1_mps, mat2_mps, out=prefilled.to(DEVICE))
    checks["returns_out"] = result is not None
    checks["overwrites_prefilled"] = bool(torch.equal(
        result._values().cpu(), expected._values())) and result._nnz() == expected._nnz()

    small = torch.sparse_coo_tensor(torch.empty(2, 0, dtype=torch.long), torch.empty(0), (1, 1)).to(DEVICE)
    result = torch.sspaddmm(self_mps, mat1_mps, mat2_mps, out=small)
    checks["resizes"] = tuple(small.shape) == (4, 4) and small._nnz() == expected._nnz()

    coalesced_out = torch.sparse_coo_tensor(torch.tensor([[0], [0]]), torch.tensor([1.0]), (4, 4)).coalesce()
    result = torch.sspaddmm(self_mps, mat1_mps, mat2_mps, out=coalesced_out.to(DEVICE))
    checks["coalesced_out_becomes_uncoalesced"] = result.is_coalesced() is False
    checks["coalesced_out_values"] = bool(torch.equal(result._values().cpu(), expected._values()))

    alias = self_cpu.clone().to(DEVICE)
    result = torch.sspaddmm(alias, mat1_mps, mat2_mps, out=alias)
    checks["aliases_self"] = bool(torch.equal(result._indices().cpu(), expected._indices())) and result._nnz() == expected._nnz()

    try:
        torch.sspaddmm(self_mps, mat1_mps, mat2_mps, out=torch.zeros(4, 4, device=DEVICE))
        checks["rejects_dense_out"] = False
    except RuntimeError:
        checks["rejects_dense_out"] = True
    try:
        wrong = torch.sparse_coo_tensor(torch.empty(2, 0, dtype=torch.long, device=DEVICE),
                                        torch.empty(0, dtype=torch.float16, device=DEVICE), (4, 4))
        torch.sspaddmm(self_mps, mat1_mps, mat2_mps, out=wrong)
        checks["rejects_wrong_dtype"] = False
    except RuntimeError:
        checks["rejects_wrong_dtype"] = True
    return checks


def autograd_suite():
    base_self = torch.tensor([[0, 0, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2]])
    base_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    self_cpu = torch.sparse_coo_tensor(base_self, base_values, (4, 4)).coalesce()
    mat1_cpu = torch.sparse_coo_tensor(torch.tensor([[2, 0, 1], [1, 0, 2]]), torch.tensor([7.0, 1.0, 1.0]), (4, 3))
    mat2_cpu = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    def forward_with(which):
        sparse_self = self_cpu.to(DEVICE).requires_grad_(which == "self")
        sparse_mat1 = mat1_cpu.to(DEVICE).requires_grad_(which == "mat1")
        weight = mat2_cpu.to(DEVICE).requires_grad_(which == "mat2")
        return torch.sspaddmm(sparse_self, sparse_mat1, weight)

    report = {}
    for which in ("self", "mat1", "mat2"):
        entry = {}
        try:
            result = forward_with(which)
            entry["forward"] = "ok"
            entry["grad_fn"] = type(result.grad_fn).__name__ if result.grad_fn is not None else None
            try:
                result.to_dense().sum().backward()
                entry["backward"] = "ok"
            except RuntimeError as error:
                entry["backward"] = f"{type(error).__name__}: {error}"
        except Exception as error:
            entry["forward"] = f"{type(error).__name__}: {str(error)[:200]}"
        report[which] = entry
    try:
        weight = mat2_cpu.clone().requires_grad_(True)
        reference = torch.sspaddmm(self_cpu, mat1_cpu, weight)
        reference.to_dense().sum().backward()
        report["cpu_backward"] = "ok"
    except RuntimeError as error:
        report["cpu_backward"] = f"{type(error).__name__}: {error}"
    return report


def sparse_operand(rows, columns, density, device="cpu"):
    count = int(rows * columns * density)
    row_index = torch.randint(0, rows, (count,), device=device)
    column_index = torch.randint(0, columns, (count,), device=device)
    return torch.sparse_coo_tensor(
        torch.stack([row_index, column_index]), torch.randn(count, device=device),
        (rows, columns), device=device).coalesce()


def sync_suite(mat1_mps, mat2_mps):
    """The result nnz is data dependent: the occupied-row count is read back once."""
    row_index = mat1_mps._indices()[0]
    readback, readback_min = timed(lambda: torch.unique_consecutive(row_index), batches=21)
    readback_enqueue, readback_enqueue_min = host_only_ms(
        lambda: torch.unique_consecutive(row_index), batches=21, inner=4)
    scan = []
    for count in (1 << 17, 1 << 20):
        probe = torch.randint(0, 1 << 16, (count,), device=DEVICE, dtype=torch.int64).sort().values
        scan.append({
            "indices": count,
            "unique_consecutive_ms": timed(lambda probe=probe: torch.unique_consecutive(probe), batches=21)[0] * 1e3,
        })
    mm_ms, mm_min = timed(lambda: torch.mm(mat1_mps, mat2_mps), batches=21)
    mm_enqueue, mm_enqueue_min = host_only_ms(
        lambda: torch.mm(mat1_mps, mat2_mps), batches=21, inner=4)
    return {
        "note": "the occupied-row count sets the output nnz, so it is materialised once per "
                "call; torch.unique_consecutive carries that readback, so its enqueue-only time "
                "is already a large fraction of its synchronized time, while mm returns in "
                "microseconds and only the synchronize waits for its kernel",
        "row_index_entries": row_index.numel(),
        "row_count_ms": readback * 1e3,
        "row_count_min_ms": readback_min * 1e3,
        "row_count_enqueue_ms": readback_enqueue * 1e3,
        "row_count_enqueue_min_ms": readback_enqueue_min * 1e3,
        "scan_is_flat_in_nnz": scan,
        "native_mm_ms": mm_ms * 1e3,
        "native_mm_min_ms": mm_min * 1e3,
        "native_mm_enqueue_ms": mm_enqueue * 1e3,
        "native_mm_enqueue_min_ms": mm_enqueue_min * 1e3,
    }


def timing_suite():
    rows = []
    for rows_count, inner, columns, density in (
            (4096, 4096, 256, 0.01),
            (16384, 8192, 128, 0.005),
            (8192, 16384, 512, 0.002),
            (65536, 4096, 64, 0.005)):
        mat1_cpu = sparse_operand(rows_count, inner, density)
        mat2_cpu = torch.randn(inner, columns)
        self_cpu = sparse_operand(rows_count, columns, 0.5)
        mat1_mps, mat2_mps, self_mps = (
            mat1_cpu.to(DEVICE), mat2_cpu.to(DEVICE), self_cpu.to(DEVICE))
        result = torch.sspaddmm(self_mps, mat1_mps, mat2_mps)
        reference = torch.sspaddmm(self_cpu, mat1_cpu, mat2_cpu)
        cpu_median, cpu_min = timed(
            lambda: torch.sspaddmm(self_cpu, mat1_cpu, mat2_cpu), batches=7)
        mps_median, mps_min = timed(
            lambda: torch.sspaddmm(self_mps, mat1_mps, mat2_mps), batches=21)
        enqueue_median, enqueue_min = host_only_ms(
            lambda: torch.sspaddmm(self_mps, mat1_mps, mat2_mps), batches=21, inner=4)
        rows.append({
            "m": rows_count, "k": inner, "n": columns, "density_of_mat1": density,
            "mat1_nnz": mat1_cpu._nnz(),
            "result_nnz": reference._nnz(),
            "parity": bool(torch.equal(result._indices().cpu(), reference._indices())),
            "values_max_abs_error": float(
                (result._values().cpu() - reference._values()).abs().max()),
            "cpu_median_ms": cpu_median * 1e3,
            "cpu_min_ms": cpu_min * 1e3,
            "mps_median_ms": mps_median * 1e3,
            "mps_min_ms": mps_min * 1e3,
            "speedup_median": cpu_median / mps_median,
            "speedup_min": cpu_min / mps_min,
            "mps_enqueue_median_ms": enqueue_median * 1e3,
            "mps_enqueue_min_ms": enqueue_min * 1e3,
            "dense_product_MB": rows_count * columns * 4 / 2 ** 20,
            "estimated_peak_MB": (rows_count * columns * 4 * 2 + 16 * reference._nnz()) / 2 ** 20,
        })
    return {
        "cpu_threads": torch.get_num_threads(),
        "note": "warmed, synchronized; per-call median and best of 21 MPS / 7 CPU batches; "
                "enqueue excludes the wait; measured on a machine shared with other work, "
                "so the minimum is the cleaner kernel estimate",
        "rows": rows,
    }


class Propagation(torch.nn.Module):
    """One graph-propagation step expressed through ``torch.sspaddmm``."""

    def __init__(self, bias, adjacency, weight):
        super().__init__()
        self.register_buffer("bias", bias)
        self.register_buffer("adjacency", adjacency)
        self.weight = torch.nn.Parameter(weight)

    def forward(self):
        return torch.sspaddmm(self.bias, self.adjacency, self.weight)


def caller_suite():
    torch.manual_seed(4242)
    adjacency = sparse_operand(512, 512, 0.02, "cpu")
    bias = sparse_operand(512, 64, 0.1, "cpu")
    weight = torch.randn(512, 64)
    cpu_model = Propagation(bias, adjacency, weight)
    mps_model = Propagation(bias.to(DEVICE), adjacency.to(DEVICE), weight.to(DEVICE))
    reference = cpu_model()
    result = mps_model()
    cpu_median, cpu_min = timed(cpu_model, batches=7)
    mps_median, mps_min = timed(mps_model, batches=21)
    return {
        "module": "Propagation: torch.sspaddmm(bias, adjacency, weight)",
        "shape": list(reference.shape),
        "nnz": [reference._nnz(), result._nnz()],
        "indices_equal": bool(torch.equal(result._indices().cpu(), reference._indices())),
        "values_max_abs_error": float((result._values().cpu() - reference._values()).abs().max()),
        "cpu_median_ms": cpu_median * 1e3,
        "cpu_min_ms": cpu_min * 1e3,
        "mps_median_ms": mps_median * 1e3,
        "mps_min_ms": mps_min * 1e3,
        "speedup_median": cpu_median / mps_median,
    }


def before_overlay():
    base_self = torch.tensor([[0, 0], [0, 1]])
    self_mps = torch.sparse_coo_tensor(base_self, torch.tensor([1.0, 2.0]), (3, 3)).coalesce().to(DEVICE)
    mat1_mps = torch.sparse_coo_tensor(torch.tensor([[0], [0]]), torch.tensor([1.0]), (3, 2)).to(DEVICE)
    mat2_mps = torch.randn(2, 3, device=DEVICE)
    try:
        torch.sspaddmm(self_mps, mat1_mps, mat2_mps)
        return "runs without the overlay"
    except Exception as error:
        return f"{type(error).__name__}: {str(error).splitlines()[0][:200]}"


def main():
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is required for this verification")
    global _LIBRARY
    started = time.time()
    baseline = before_overlay()
    _LIBRARY = install()

    start = time.perf_counter()
    torch.sspaddmm(
        torch.sparse_coo_tensor(torch.tensor([[0], [0]]), torch.tensor([1.0]), (4, 4)).coalesce().to(DEVICE),
        torch.sparse_coo_tensor(torch.tensor([[0], [0]]), torch.tensor([1.0]), (4, 2)).to(DEVICE),
        torch.randn(2, 4, device=DEVICE))
    torch.mps.synchronize()
    first_call_ms = (time.perf_counter() - start) * 1e3

    structure = structure_suite()
    dtypes = dtype_suite()
    out = out_suite()
    autograd = autograd_suite()
    probe_mat1 = sparse_operand(8192, 8192, 0.005)
    sync = sync_suite(probe_mat1.to(DEVICE), torch.randn(8192, 256, device=DEVICE))
    timing = timing_suite()
    caller = caller_suite()

    failures = []
    if structure["structured_failures"]:
        failures.append(f"structured cases failed: {structure['structured_failures']}")
    if structure["generated_index_mismatches"]:
        failures.append(f"{structure['generated_index_mismatches']} generated index mismatches")
    if structure["generated_value_mismatches_beyond_1e-5"]:
        failures.append(
            f"{structure['generated_value_mismatches_beyond_1e-5']} generated value mismatches beyond 1e-5")
    if structure["generated_structure_failures"]:
        failures.append("generated structure flags differ")
    for name, row in dtypes.items():
        if name == "unsupported":
            continue
        if not row.get("served"):
            failures.append(f"{name} not served")
        elif row.get("error"):
            failures.append(f"{name} raised {row['error']}")
        elif "values_close" in row and not row["values_close"]:
            failures.append(f"{name} values differ from CPU beyond 1e-5")
        elif "indices_equal" in row and not row["indices_equal"]:
            failures.append(f"{name} indices differ from CPU")
        elif row.get("indices_equal_to_float32_structure") is False:
            failures.append(f"{name} structure differs from the float32 reference")
        elif row.get("values_max_abs_error_vs_float32", 0.0) > 0.1:
            failures.append(f"{name} values off by more than 0.1 against the float32 reference")
    for name, value in out.items():
        if not value:
            failures.append(f"out= check failed: {name}")
    for which in ("self", "mat1", "mat2"):
        if autograd[which].get("forward") != "ok":
            failures.append(f"autograd forward failed for {which}")
    if not autograd.get("cpu_backward", "").startswith("RuntimeError"):
        failures.append("CPU backward no longer raises, parity statement is stale")
    for row in timing["rows"]:
        if not row["parity"]:
            failures.append(f"timing case {row['m']}x{row['k']}x{row['n']} lost index parity")
    if not caller["indices_equal"]:
        failures.append("caller module indices differ from CPU")

    record = {
        "op": "aten::sspaddmm",
        "device": DEVICE,
        "dispatch_key": DISPATCH_KEY,
        "overlays": [name for name, _ in SPARSE_OVERLAYS],
        "before_overlay": baseline,
        "algorithm": "native SparseMPS mm restricted to the coalesced mat1's occupied rows; "
                     "self's stored entries and the occupied-row block are written into "
                     "preallocated index/value buffers, result left uncoalesced",
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "platform": f"{platform.mac_ver()[0]} {platform.machine()}",
        "first_call_ms_including_pipeline_compile": first_call_ms,
        "elapsed_s": time.time() - started,
        "structure": structure,
        "dtypes": dtypes,
        "out": out,
        "autograd": autograd,
        "sync": sync,
        "timing": timing,
        "caller": caller,
        "limits": [
            "MPS has no float64 tensors, so the float64 signature cannot be served; the CPU "
            "kernel serves it",
            "the CPU kernel rejects float16/bfloat16 (\"sspmm\" not implemented) while this "
            "overlay serves them from the native SparseMPS mm; values are float16/bfloat16 "
            "arithmetic, the structure is identical",
            "bool raises on both backends",
            "invalid-input errors are RuntimeError like the CPU kernel but the shape messages "
            "are the CPU kernel's; dtype-mismatch and out-dtype messages are this overlay's",
            "a hybrid sparse self (sparse_dim other than 2) is rejected as Argument #1; the CPU "
            "kernel instead fails inside an internal size comparison",
            "the output nnz is data dependent, so the occupied-row count is read back to the "
            "host once per call; torch.unique_consecutive blocks for that count, which is the "
            "only synchronization, and its cost is a host round trip rather than a function of "
            "the operand size (see `sync`)",
            "that readback dominates small products, so the MPS call can be slower than the CPU "
            "kernel there; `caller` is such a case",
            "the product is computed by the native sparse mm, which allocates a dense m x n "
            "intermediate before the occupied rows are gathered; memory is therefore "
            "O(m*n) in addition to the COO result",
            "float64/complex128 feed neither MPS tensors nor this overlay; complex64 is served",
        ],
        "failures": failures,
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: record[key] for key in
                      ("first_call_ms_including_pipeline_compile", "structure", "dtypes",
                       "out", "autograd", "sync", "timing", "caller", "failures")}, indent=2))
    print(f"wrote {JSON_PATH}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
