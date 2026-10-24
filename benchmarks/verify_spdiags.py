"""Verify and measure the ``_spdiags`` MPS overlays in ``spdiags_ops.py``.

    uv run --project /Users/dbydd/vibe-agent-working-dir/ResearchChamber \
        python benchmarks/verify_spdiags.py

Registers the two overlays on a temporary ``aten`` fragment for this process
only, then compares the MPS result against the CPU kernel case by case: shape,
dtype, layout, ``nnz``, the coalesced flag, the raw ``(2, nnz)`` index matrix in
construction order and the values. Error paths are compared by exception type
and message. The documented divergences (sparse CSR/CSC, the CPU kernel's
out-of-bounds write on a negative per-diagonal count) are asserted as
divergences rather than counted as failures.

Timing covers four variants per shape: the overlay with offsets on the host,
the overlay with offsets on the MPS device (one extra synchronization), the
offsets-to-host read alone, and the CPU kernel as a scale reference. A
``TorchDispatchMode`` counts the MPS-to-host synchronizations for both offsets
placements, so the "one synchronization" claim is measured, not asserted by
hand.

Writes ``measured/spdiags.json`` and exits non-zero when a check fails.
"""

import contextlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.library import Library
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parents[1]
BROKER = Path("/Users/dbydd/vibe-agent-working-dir/ResearchChamber/shared_scripts")
sys.path.insert(0, str(ROOT / "src"))

from metal_kernelops import spdiags_ops  # noqa: E402

DEVICE = "mps"


def diag(n_diags, diag_cols, dtype=torch.float32):
    values = torch.arange(n_diags * diag_cols, dtype=torch.float32).reshape(n_diags, diag_cols)
    if dtype is torch.bool:
        return values.to(torch.int64) % 3 == 0
    return values.to(dtype)


def parity_cases():
    """Valid calls: every branch of the count formula and every dtype."""
    yield "square_main_up_down", diag(3, 3), torch.tensor([0, 1, -2]), (3, 3)
    yield "square_unsorted", diag(3, 3), torch.tensor([-1, 0, 1]), (3, 3)
    yield "single_up_edge", diag(1, 3), torch.tensor([2]), (3, 3)
    yield "single_down_edge", diag(1, 3), torch.tensor([-2]), (3, 3)
    yield "silent_out_of_range", diag(2, 3), torch.tensor([3, -3]), (3, 3)
    yield "tall_rect_4x3", diag(3, 4), torch.tensor([0, 1, -1]), (4, 3)
    yield "wide_rect_3x5", diag(3, 5), torch.tensor([0, 1, -2]), (3, 5)
    yield "overshoot_rows", diag(1, 5), torch.tensor([1]), (2, 5)
    yield "overshoot_cols", diag(1, 5), torch.tensor([-1]), (5, 2)
    yield "doc_partial", torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([0, -1]), (5, 5)
    yield "long_diag_trimmed", diag(1, 10), torch.tensor([0]), (3, 3)
    yield ("noncontig_2d", torch.arange(18.0).reshape(3, 6)[:, ::2],
           torch.tensor([0, 1, -1]), (4, 3))
    yield "noncontig_1d", torch.arange(6.0)[::2], torch.tensor(0), (3, 3)
    yield "one_d_scalar0", torch.tensor([1.0, 2.0, 3.0]), torch.tensor(0), (3, 3)
    yield "one_d_scalar_neg2", torch.tensor([1.0, 2.0, 3.0]), torch.tensor(-2), (3, 3)
    yield "one_d_scalar_pos1", torch.tensor([1.0, 2.0, 3.0]), torch.tensor(1), (3, 3)
    yield "empty_offsets", torch.ones(0, 3), torch.empty(0, dtype=torch.int64), (3, 3)
    yield "empty_diag_cols", torch.ones(2, 0), torch.tensor([0, -1]), (3, 3)
    yield "one_d_empty", torch.empty(0), torch.tensor(0), (3, 3)
    yield "zero_rows", diag(2, 3), torch.tensor([0, 1]), (0, 3)
    yield "zero_rows_negative_offset", diag(1, 3), torch.tensor([-5]), (0, 3)
    yield "zero_cols", diag(2, 3), torch.tensor([0, 1]), (3, 0)
    yield "zero_cols_positive_offset", diag(1, 3), torch.tensor([4]), (3, 0)
    yield "many_diags", diag(8, 11), torch.arange(-3, 5), (7, 11)
    yield "dtype_float16", diag(2, 3, torch.float16), torch.tensor([0, 1]), (3, 4)
    yield "dtype_bfloat16", diag(2, 3, torch.bfloat16), torch.tensor([0, 1]), (3, 4)
    yield "dtype_int32", diag(2, 3, torch.int32), torch.tensor([0, 1]), (3, 4)
    yield "dtype_int64", diag(2, 3, torch.int64), torch.tensor([0, 1]), (3, 4)
    yield "dtype_bool", diag(2, 3, torch.bool), torch.tensor([0, 1]), (3, 4)
    yield "dtype_complex64", diag(2, 3, torch.complex64), torch.tensor([0, 1]), (3, 4)
    yield "overshoot_mixed", diag(3, 5), torch.tensor([1, -1, 2]), (2, 5)


def error_cases():
    """Calls the CPU kernel rejects; type and message must match on MPS."""
    yield "duplicate_offsets", diag(2, 3), torch.tensor([0, 0]), (3, 3), None
    yield "duplicate_offsets_unsorted", diag(3, 3), torch.tensor([2, -1, 2]), (3, 3), None
    yield "diagonal_count_mismatch", diag(2, 3), torch.tensor([0, 1, 2]), (3, 3), None
    yield "diagonals_3d", torch.ones(1, 2, 3), torch.tensor([0]), (3, 3), None
    yield "offsets_2d", diag(3, 3), torch.tensor([[0]]), (3, 3), None
    yield "shape_1d", diag(1, 3), torch.tensor([0]), (3,), None
    yield "shape_3d", diag(1, 3), torch.tensor([0]), (3, 3, 3), None
    yield "layout_strided", diag(3, 3), torch.tensor([0, 1, -1]), (3, 3), torch.strided
    yield "layout_bsr", diag(3, 3), torch.tensor([0, 1, -1]), (3, 3), torch.sparse_bsr
    yield "offsets_int32", diag(3, 3), torch.tensor([0, 1, -1], dtype=torch.int32), (3, 3), None
    yield "offsets_float32", diag(3, 3), torch.tensor([0.0, 1.0, -1.0]), (3, 3), None
    yield "negative_total", diag(1, 3), torch.tensor([4]), (3, 3), None
    yield "negative_total_negative_offset", diag(1, 3), torch.tensor([-4]), (3, 3), None
    yield ("empty_diag_positive_offset", torch.ones(1, 0), torch.tensor([1]), (3, 3),
           None)
    yield "negative_rows", diag(1, 3), torch.tensor([0]), (-1, 3), None
    yield "negative_cols", diag(1, 3), torch.tensor([0]), (3, -1), None


def divergence_cases():
    """Inputs whose CPU result is either unsupported here or undefined."""
    yield ("layout_sparse_csr", diag(3, 3), torch.tensor([0, 1, -2]), (3, 3),
           torch.sparse_csr)
    yield ("layout_sparse_csc", diag(3, 3), torch.tensor([0, 1, -2]), (3, 3),
           torch.sparse_csc)
    yield "negative_per_diagonal_count", diag(2, 3), torch.tensor([4, 0]), (3, 3)


class Report:
    def __init__(self):
        self.rows = []
        self.failed = 0

    def check(self, name, ok, detail=None):
        ok = bool(ok)
        row = {"check": name, "ok": ok}
        if detail is not None:
            row["detail"] = detail
        self.rows.append(row)
        if not ok:
            self.failed += 1
        return ok


def outcome(fn):
    try:
        return {"ok": True, "value": fn()}
    except Exception as error:  # noqa: BLE001 - the type and text are the assertion
        return {"ok": False, "type": type(error).__name__, "message": str(error)}


def structure(tensor):
    """Raw sparse structure on the host, before any coalescing."""
    host = tensor.to("cpu") if tensor.device.type != "cpu" else tensor
    if host.layout != torch.sparse_coo:
        return {"layout": str(host.layout), "shape": tuple(host.shape),
                "dtype": str(host.dtype)}
    return {
        "layout": str(host.layout),
        "shape": tuple(host.shape),
        "dtype": str(host.dtype),
        "nnz": host._nnz(),
        "coalesced": host.is_coalesced(),
        "indices": host._indices().tolist(),
        "values": host._values().tolist(),
    }


def install():
    had_kernel = torch._C._dispatch_has_kernel_for_dispatch_key("aten::_spdiags", "MPS")
    registered = []
    lib = Library("aten", "FRAGMENT")
    for name, function in spdiags_ops.OVERLAYS:
        operator = f"aten::{name}"
        replaced = torch._C._dispatch_has_kernel_for_dispatch_key(operator, "MPS")
        lib.impl(name, function, "MPS")
        registered.append({"aten": operator, "replaced_existing_kernel": bool(replaced),
                           "kernel": f"{function.__module__}.{function.__name__}"})
    has_kernel = torch._C._dispatch_has_kernel_for_dispatch_key("aten::_spdiags", "MPS")
    return registered, lib, had_kernel, has_kernel


@contextlib.contextmanager
def lease():
    if not (BROKER / "compute_broker.py").exists():
        print("no compute broker here, timing unleased")
        yield
        return
    sys.path.insert(0, str(BROKER))
    from compute_broker import lease_guard

    with lease_guard("mps-kernelops", tier=1, task_name="verify_spdiags"):
        yield


def parity(report):
    for name, diagonals, offsets, shape in parity_cases():
        cpu = torch.ops.aten._spdiags(diagonals, offsets, shape)
        mps = torch.ops.aten._spdiags(diagonals.to(DEVICE), offsets, shape)
        same = structure(cpu) == structure(mps)
        report.check(f"parity: {name}", same,
                     None if same else {"cpu": structure(cpu), "mps": structure(mps)})


def errors(report):
    for name, diagonals, offsets, shape, layout in error_cases():
        cpu = outcome(lambda: torch.ops.aten._spdiags(diagonals, offsets, shape, layout))
        mps = outcome(lambda: torch.ops.aten._spdiags(diagonals.to(DEVICE), offsets, shape, layout))
        same = (not cpu["ok"] and not mps["ok"] and cpu["type"] == mps["type"]
                and cpu["message"] == mps["message"])
        report.check(f"error parity: {name}", same,
                     None if same else {"cpu": cpu.get("message", "returned"),
                                        "mps": mps.get("message", "returned")})


def divergences(report, notes):
    csr = torch.ops.aten._spdiags(diag(3, 3), torch.tensor([0, 1, -2]), (3, 3),
                                  layout=torch.sparse_csr)
    csc = torch.ops.aten._spdiags(diag(3, 3), torch.tensor([0, 1, -2]), (3, 3),
                                  layout=torch.sparse_csc)
    report.check("divergence: CPU builds CSR and CSC", csr.layout == torch.sparse_csr
                 and csc.layout == torch.sparse_csc,
                 {"csr": str(csr.layout), "csc": str(csc.layout)})
    for name, layout in (("csr", torch.sparse_csr), ("csc", torch.sparse_csc)):
        call = outcome(lambda: torch.ops.aten._spdiags(
            diag(3, 3).to(DEVICE), torch.tensor([0, 1, -2]), (3, 3), layout=layout))
        report.check(f"divergence: {name} raises NotImplementedError on MPS",
                     not call["ok"] and call["type"] == "NotImplementedError"
                     and "sparse compressed tensors cannot be constructed" in call["message"],
                     {} if call["ok"] else {"type": call["type"]})
    cpu_partial = outcome(lambda: torch.ops.aten._spdiags(
        diag(2, 3), torch.tensor([4, 0]), (3, 3)))
    mps_partial = outcome(lambda: torch.ops.aten._spdiags(
        diag(2, 3).to(DEVICE), torch.tensor([4, 0]), (3, 3)))
    report.check("divergence: negative per-diagonal count rejected on MPS",
                 cpu_partial["ok"] and not mps_partial["ok"]
                 and mps_partial["type"] == "RuntimeError"
                 and "negative diagonal length" in mps_partial["message"],
                 {"cpu_nnz": cpu_partial["value"]._nnz() if cpu_partial["ok"] else None,
                  "mps": mps_partial.get("message", "returned")})
    notes.extend([
        "CPU builds dense-free CSR and CSC for the same call; MPS 2.14.1 cannot "
        "construct sparse compressed tensors (new_compressed_tensor has no MPS "
        "kernel), so layout=sparse_csr/sparse_csc raises NotImplementedError "
        "instead of returning a COO tensor under another layout name",
        "when one diagonal's count is negative but the total is not, the CPU "
        "kernel allocates sum(counts) entries and writes the negative block out "
        "of bounds; the MPS overlay rejects that input, so the two differ only "
        "on an input whose CPU result is undefined",
    ])


def offsets_device(report):
    diagonals = diag(3, 4)
    offsets = torch.tensor([0, 1, -2])
    cpu = torch.ops.aten._spdiags(diagonals, offsets, (4, 4))
    from_cpu = torch.ops.aten._spdiags(diagonals.to(DEVICE), offsets, (4, 4))
    from_mps = torch.ops.aten._spdiags(diagonals.to(DEVICE), offsets.to(DEVICE), (4, 4))
    report.check("offsets on the device give the same structure",
                 structure(cpu) == structure(from_mps) == structure(from_cpu))
    scalar = torch.tensor(-1, device=DEVICE)
    one_d = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
    scalar_result = torch.ops.aten._spdiags(one_d, scalar, (3, 3))
    expected = torch.ops.aten._spdiags(torch.tensor([1.0, 2.0, 3.0]),
                                       torch.tensor(-1), (3, 3))
    report.check("0-dim offsets on the device", structure(scalar_result) == structure(expected))
    empty = torch.empty(0, dtype=torch.int64, device=DEVICE)
    empty_result = torch.ops.aten._spdiags(torch.ones(0, 3, device=DEVICE), empty, (3, 3))
    report.check("empty offsets on the device", empty_result._nnz() == 0)

    cross = outcome(lambda: torch.ops.aten._spdiags(
        diag(1, 3), torch.tensor([0], device=DEVICE), (3, 3)))
    report.check("offsets on MPS with CPU diagonals is rejected explicitly",
                 not cross["ok"] and cross["type"] == "NotImplementedError"
                 and "diagonals on the MPS device" in cross["message"],
                 {"type": cross.get("type"), "message": cross.get("message")})


def _fuzz_counts(diagonals, offsets, shape):
    rows, cols = shape
    diag_cols = diagonals.size(0) if diagonals.dim() == 1 else diagonals.size(1)
    if rows == 0 or cols == 0 or offsets.numel() == 0:
        return None
    k = offsets.reshape(-1).to(torch.int64)
    return torch.where(k > 0, min(cols, diag_cols) - k,
                       torch.minimum(rows + k, torch.tensor(diag_cols)))


def fuzz(report, total=80):
    """Random shapes, offsets and dtypes against the CPU kernel.

    Inputs whose CPU kernel writes out of bounds (a negative per-diagonal count
    with a non-negative total) are not executed on the CPU at all: the CPU
    kernel under-allocates there and the write is undefined. The overlay must
    reject them.
    """
    import random

    rng = random.Random(20241024)
    dtypes = (torch.float32, torch.float16, torch.int32, torch.int64, torch.bool,
              torch.complex64)
    mismatches = []
    undefined = 0
    for index in range(total):
        rows, cols = rng.randint(0, 7), rng.randint(0, 7)
        n_diags = rng.randint(0, 4)
        diag_cols = rng.randint(0, 8)
        offsets = rng.sample(range(-8, 9), n_diags)
        dtype = rng.choice(dtypes)
        one_d = n_diags == 1 and rng.random() < 0.5
        diagonals = diag(1, diag_cols, dtype)[0] if one_d else diag(n_diags, diag_cols, dtype)
        offsets_t = (torch.tensor(offsets[0]) if rng.random() < 0.5
                     else torch.tensor(offsets, dtype=torch.int64)) if one_d \
            else torch.tensor(offsets, dtype=torch.int64)
        shape = (rows, cols)
        counts = _fuzz_counts(diagonals, offsets_t, shape)
        if counts is not None and int(counts.min()) < 0 and int(counts.sum()) >= 0:
            undefined += 1
            raised = outcome(lambda: torch.ops.aten._spdiags(
                diagonals.to(DEVICE), offsets_t, shape))
            if raised["ok"] or raised["type"] != "RuntimeError" \
                    or "negative diagonal length" not in raised["message"]:
                mismatches.append({"case": index, "expected": "rejected",
                                   "got": raised.get("message", "returned")})
            continue
        cpu = outcome(lambda: torch.ops.aten._spdiags(diagonals, offsets_t, shape))
        mps = outcome(lambda: torch.ops.aten._spdiags(
            diagonals.to(DEVICE), offsets_t, shape))
        if cpu["ok"] != mps["ok"]:
            mismatches.append({"case": index, "cpu": cpu["ok"], "mps": mps["ok"]})
        elif cpu["ok"]:
            if structure(cpu["value"]) != structure(mps["value"]):
                mismatches.append({"case": index, "cpu": structure(cpu["value"]),
                                   "mps": structure(mps["value"])})
        elif cpu["type"] != mps["type"] or cpu["message"] != mps["message"]:
            mismatches.append({"case": index, "cpu": cpu["message"],
                               "mps": mps["message"]})
    report.check(f"fuzz: {total} random cases match the CPU kernel", not mismatches,
                 {"mismatches": mismatches[:3], "rejected_undefined": undefined})


def out_overload(report):
    diagonals_cpu = diag(2, 3)
    diagonals_mps = diagonals_cpu.to(DEVICE)
    offsets = torch.tensor([0, 1])

    def empty_coo(device, shape=(3, 4)):
        return torch.ops.aten._sparse_coo_tensor_unsafe.default(
            torch.empty((2, 0), dtype=torch.int64, device=device),
            torch.empty(0, device=device), list(shape), is_coalesced=True)

    cpu = torch.ops.aten._spdiags.out(diagonals_cpu, offsets, (3, 4), out=empty_coo("cpu"))
    mps_out = empty_coo(DEVICE)
    mps = torch.ops.aten._spdiags.out(diagonals_mps, offsets, (3, 4), out=mps_out)
    report.check("out: same-shape COO out matches CPU and is returned",
                 mps is mps_out and structure(mps) == structure(cpu),
                 None if structure(mps) == structure(cpu) else
                 {"cpu": structure(cpu), "mps": structure(mps)})

    prefilled = torch.ops.aten._sparse_coo_tensor_unsafe.default(
        torch.tensor([[0, 2], [0, 3]], dtype=torch.int64, device=DEVICE),
        torch.tensor([9.0, 9.0], device=DEVICE), [3, 4], is_coalesced=False)
    replaced = torch.ops.aten._spdiags.out(diagonals_mps, offsets, (3, 4), out=prefilled)
    report.check("out: existing nnz is replaced", structure(replaced) == structure(cpu),
                 None if structure(replaced) == structure(cpu) else structure(replaced))

    # Both shapes differ from (3, 4) in one dimension; compare only the class.
    wrong_cpu = outcome(lambda: torch.ops.aten._spdiags.out(
        diagonals_cpu, offsets, (3, 4), out=empty_coo("cpu", (2, 4))))
    wrong_mps = outcome(lambda: torch.ops.aten._spdiags.out(
        diagonals_mps, offsets, (3, 4), out=empty_coo(DEVICE, (2, 4))))
    report.check("out: wrong-shape sparse out fails as on CPU",
                 not wrong_cpu["ok"] and not wrong_mps["ok"]
                 and wrong_cpu["type"] == wrong_mps["type"] == "NotImplementedError",
                 {"cpu": wrong_cpu.get("type"), "mps": wrong_mps.get("type")})

    dense_cpu = outcome(lambda: torch.ops.aten._spdiags.out(
        diagonals_cpu, offsets, (3, 4), out=torch.zeros(3, 4)))
    dense_mps = outcome(lambda: torch.ops.aten._spdiags.out(
        diagonals_mps, offsets, (3, 4), out=torch.zeros(3, 4, device=DEVICE)))
    report.check("out: dense out fails as on CPU",
                 not dense_cpu["ok"] and not dense_mps["ok"]
                 and dense_cpu["type"] == dense_mps["type"] == "RuntimeError"
                 and "dense and sparse" in dense_cpu["message"]
                 and "dense and sparse" in dense_mps["message"])

    single_cpu = torch.ops.aten._spdiags.out(
        diag(1, 3), torch.tensor([2]), (3, 3), out=empty_coo("cpu", (3, 3)))
    single_mps = torch.ops.aten._spdiags.out(
        diag(1, 3).to(DEVICE), torch.tensor([2]), (3, 3), out=empty_coo(DEVICE, (3, 3)))
    report.check("out: nnz <= 1 is marked coalesced",
                 single_cpu._nnz() == single_mps._nnz() == 1
                 and single_cpu.is_coalesced() is True
                 and single_mps.is_coalesced() is True,
                 {"cpu": single_cpu.is_coalesced(), "mps": single_mps.is_coalesced()})

    mismatch_cpu = outcome(lambda: torch.ops.aten._spdiags.out(
        diagonals_cpu, offsets, (3, 4),
        out=torch.ops.aten._sparse_coo_tensor_unsafe.default(
            torch.empty((2, 0), dtype=torch.int64),
            torch.empty(0, dtype=torch.float16), [3, 4], is_coalesced=True)))
    mismatch_mps = outcome(lambda: torch.ops.aten._spdiags.out(
        diagonals_mps, offsets, (3, 4),
        out=torch.ops.aten._sparse_coo_tensor_unsafe.default(
            torch.empty((2, 0), dtype=torch.int64, device=DEVICE),
            torch.empty(0, dtype=torch.float16, device=DEVICE), [3, 4],
            is_coalesced=True)))
    report.check("out: dtype mismatch message matches CPU",
                 not mismatch_cpu["ok"] and not mismatch_mps["ok"]
                 and mismatch_cpu["message"] == mismatch_mps["message"],
                 {"cpu": mismatch_cpu.get("message"), "mps": mismatch_mps.get("message")})


def public_api(report, cpu_reference):
    diagonals = diag(3, 3)
    offsets = torch.tensor([0, 1, -2])
    mps = torch.sparse.spdiags(diagonals.to(DEVICE), offsets, (3, 3))
    report.check("torch.sparse.spdiags reaches the overlay",
                 structure(mps) == cpu_reference)
    cpu_after = torch.sparse.spdiags(diagonals, offsets, (3, 3))
    report.check("CPU dispatch is untouched by the MPS registration",
                 structure(cpu_after) == cpu_reference)


def doc_examples(report):
    # torch.sparse.spdiags documentation, all three examples.
    lower = torch.ops.aten._spdiags(
        torch.arange(9.0).reshape(3, 3).to(DEVICE), torch.tensor([0, -1, -2]),
        (3, 3)).to_dense()
    report.check("doc: lower diagonals", torch.equal(lower, torch.tensor(
        [[0.0, 0.0, 0.0], [3.0, 1.0, 0.0], [6.0, 4.0, 2.0]], device=DEVICE)))
    partial = torch.ops.aten._spdiags(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to(DEVICE), torch.tensor([0, -1]),
        (5, 5)).to_dense()
    report.check("doc: partial diagonals", torch.equal(partial, torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 0.0], [3.0, 2.0, 0.0, 0.0, 0.0],
         [0.0, 4.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0]], device=DEVICE)))
    positive = torch.ops.aten._spdiags(
        torch.tensor([[1.0, 2.0, 3.0]] * 3).to(DEVICE), torch.tensor([0, 1, 2]),
        (5, 5)).to_dense()
    report.check("doc: positive offsets", torch.equal(positive, torch.tensor(
        [[1.0, 2.0, 3.0, 0.0, 0.0], [0.0, 2.0, 3.0, 0.0, 0.0],
         [0.0, 0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0]], device=DEVICE)))


def consumer(report):
    diagonals = diag(3, 512)
    offsets = torch.tensor([-1, 0, 1])
    sparse = torch.ops.aten._spdiags(diagonals.to(DEVICE), offsets, (512, 512))
    dense = sparse.to_dense()
    reference = torch.ops.aten._spdiags(diagonals, offsets, (512, 512)).to_dense()
    report.check("to_dense equals the CPU dense result", torch.equal(dense, reference.to(DEVICE)))
    block = torch.ones(512, 16, device=DEVICE)
    product = torch.sparse.mm(sparse, block)
    report.check("sparse mm consumes the result", torch.equal(product, reference.to(DEVICE) @ block))


def autograd(report):
    cpu = outcome(lambda: _backward(torch.ones(2, 3, requires_grad=True), "cpu"))
    mps = outcome(lambda: _backward(torch.ones(2, 3, device=DEVICE, requires_grad=True), DEVICE))
    report.check("autograd: same unsupported-derivative error as CPU",
                 not cpu["ok"] and not mps["ok"] and cpu["type"] == mps["type"]
                 and cpu["message"] == mps["message"],
                 {"cpu": cpu.get("message"), "mps": mps.get("message")})


def _backward(diagonals, device):
    result = torch.ops.aten._spdiags(diagonals, torch.tensor([0, 1], device=device), (3, 4))
    result.to_dense().sum().backward()
    return diagonals.grad


class Instrument(TorchDispatchMode):
    """Count MPS-visible aten ops and the host reads that synchronize."""

    def __init__(self):
        self.mps_results = 0
        self.h2d = 0
        self.d2h = 0
        self.scalar_reads = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        source = list(_tensors(args)) + list(_tensors(kwargs))
        source_mps = any(t.device.type == "mps" for t in source)
        destination = kwargs.get("device")
        if name == "aten._to_copy.default" and destination is not None:
            if source_mps and destination.type == "cpu":
                self.d2h += 1
            elif not source_mps and destination.type == "mps":
                self.h2d += 1
        if name == "aten._local_scalar_dense.default" and source_mps:
            self.scalar_reads += 1
        result = func(*args, **kwargs)
        if any(t.device.type == "mps" for t in _tensors(result)):
            self.mps_results += 1
        return result


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)


def synchronizations(report):
    host_offsets = torch.tensor([-2, -1, 0, 1, 2])
    diagonals = diag(len(host_offsets), 512).to(DEVICE)
    device_offsets = host_offsets.to(DEVICE)
    spdiags_ops.spdiags(diagonals, host_offsets, [1024, 1024])
    spdiags_ops.spdiags(diagonals, device_offsets, [1024, 1024])
    torch.mps.synchronize()

    with Instrument() as host:
        spdiags_ops.spdiags(diagonals, host_offsets, [1024, 1024])
    with Instrument() as device:
        spdiags_ops.spdiags(diagonals, device_offsets, [1024, 1024])

    report.check("host offsets: no device-to-host read",
                 host.d2h == 0 and host.scalar_reads == 0,
                 {"d2h": host.d2h, "scalar_reads": host.scalar_reads})
    report.check("device offsets: exactly one device-to-host read",
                 device.d2h == 1 and device.scalar_reads == 0,
                 {"d2h": device.d2h, "scalar_reads": device.scalar_reads})
    return {
        "offsets_cpu": {"synchronizations": host.d2h + host.scalar_reads,
                        "mps_aten_results": host.mps_results,
                        "h2d_copies": host.h2d},
        "offsets_mps": {"synchronizations": device.d2h + device.scalar_reads,
                        "mps_aten_results": device.mps_results,
                        "h2d_copies": device.h2d},
    }


def _measure(entries, iters, rounds, warmup=3):
    for _, fn in entries:
        for _ in range(warmup):
            fn()
    torch.mps.synchronize()
    samples = {name: [] for name, _ in entries}
    for _ in range(rounds):
        for name, fn in entries:
            started = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.mps.synchronize()
            samples[name].append((time.perf_counter() - started) / iters * 1e6)
    return {name: {"median": round(statistics.median(values), 1),
                   "min": round(min(values), 1)}
            for name, values in samples.items()}


BENCH = (
    ("1024x1024_band3", (1024, 1024), (-1, 0, 1)),
    ("4096x4096_band5", (4096, 4096), (-2, -1, 0, 1, 2)),
    ("16384x16384_band5", (16384, 16384), (-2, -1, 0, 1, 2)),
    ("1024x1024_band129", (1024, 1024), tuple(range(-64, 65))),
    ("256x256_band511", (256, 256), tuple(range(-255, 256))),
    ("4096x4096_band1025", (4096, 4096), tuple(range(-512, 513))),
)


def timings():
    measured = {}
    for label, shape, offsets in BENCH:
        rows, cols = shape
        diag_cols = min(rows, cols)
        offsets_cpu = torch.tensor(offsets, dtype=torch.int64)
        diagonals_cpu = diag(len(offsets), diag_cols)
        diagonals_mps = diagonals_cpu.to(DEVICE)
        offsets_mps = offsets_cpu.to(DEVICE)
        reference = torch.ops.aten._spdiags(diagonals_cpu, offsets_cpu, shape)
        nnz = reference._nnz()
        small = nnz <= 200_000
        entries = [
            ("mps_overlay_offsets_cpu",
             lambda: torch.ops.aten._spdiags(diagonals_mps, offsets_cpu, shape)),
            ("mps_overlay_offsets_mps",
             lambda: torch.ops.aten._spdiags(diagonals_mps, offsets_mps, shape)),
            ("offsets_to_host", lambda: spdiags_ops._offsets_host(offsets_mps)),
            ("cpu_kernel", lambda: torch.ops.aten._spdiags(diagonals_cpu, offsets_cpu, shape)),
        ]
        row = _measure(entries, iters=10 if small else 3, rounds=9 if small else 5)
        row["nnz"] = nnz
        row["shape"] = list(shape)
        row["diagonals"] = len(offsets)
        row["speedup_overlay_vs_cpu"] = round(
            row["cpu_kernel"]["min"] / row["mps_overlay_offsets_cpu"]["min"], 2)
        measured[label] = row
        print(f"{label:24s} overlay {row['mps_overlay_offsets_cpu']['median']:9.1f} us  "
              f"device-offsets {row['mps_overlay_offsets_mps']['median']:9.1f} us  "
              f"host-read {row['offsets_to_host']['median']:8.1f} us  "
              f"cpu {row['cpu_kernel']['median']:9.1f} us  nnz {nnz}")
    return measured


def main():
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available on this machine")
    torch.manual_seed(0)

    # Computed before registration so the CPU path can be checked afterwards.
    cpu_reference = structure(torch.ops.aten._spdiags(
        diag(3, 3), torch.tensor([0, 1, -2]), (3, 3)))

    registered, _library, had_kernel, has_kernel = install()

    report = Report()
    report.check("MPS had no _spdiags kernel before registration", not had_kernel)
    report.check("MPS kernel present after registration", has_kernel)
    parity(report)
    errors(report)
    fuzz(report)
    notes = []
    divergences(report, notes)
    offsets_device(report)
    out_overload(report)
    public_api(report, cpu_reference)
    doc_examples(report)
    consumer(report)
    autograd(report)
    sync = synchronizations(report)

    print("\nchecks")
    for row in report.rows:
        print(f"  {'ok  ' if row['ok'] else 'FAIL'} {row['check']}"
              + (f"  {row['detail']}" if not row["ok"] and "detail" in row else ""))

    print("\ntimings (warm, median of rounds, torch.mps.synchronize after each round)")
    load_before = round(os.getloadavg()[0], 2)
    with lease():
        measured = timings()
    load_after = round(os.getloadavg()[0], 2)

    payload = {
        "schema": 1,
        "module": "metal_kernelops/spdiags_ops.py",
        "torch": torch.__version__,
        "platform": f"{platform.platform()} {platform.machine()}",
        "device": torch.device(DEVICE).type,
        "operators": ["aten::_spdiags", "aten::_spdiags.out"],
        "overlays": registered,
        "autograd": {
            "overlay": None,
            "reason": "no derivative is registered for aten::_spdiags in 2.14.1; "
                      "forward succeeds and backward raises the same "
                      "'derivative for aten::_spdiags is not implemented' on MPS as on CPU",
        },
        "synchronizations": sync,
        "synchronizations_method": "a TorchDispatchMode counts device-to-host "
                                   "aten._to_copy calls sourced on MPS and "
                                   "aten._local_scalar_dense calls on MPS tensors "
                                   "while the overlay body runs; the offsets read is "
                                   "one bulk copy, not one read per diagonal. The "
                                   "compiled Metal kernel is not an aten op, so the "
                                   "counted MPS results are the aten ops around it",
        "correctness": {"checks": len(report.rows), "failed": report.failed,
                        "rows": report.rows},
        "divergences": notes,
        "timings_us": measured,
        "timings_method": "each variant runs 3 or 10 calls per round with one "
                          "torch.mps.synchronize at the end; variants are interleaved "
                          "round by round, so a load spike hits all of them; median and "
                          "min over 5 or 9 rounds are recorded. speedup_overlay_vs_cpu "
                          "is cpu_kernel.min / mps_overlay_offsets_cpu.min: below 1 "
                          "means the overlay is slower, and the offsets_to_host entry "
                          "measures the synchronization the device-offsets path adds",
        "timings_load_average": [load_before, load_after],
        "notes": [
            "the structure (indices in construction order, values, nnz, coalesced "
            "flag, dtype, shape) is byte-identical to the CPU kernel on every "
            "parity case; the COO result is deliberately not coalesced when "
            "nnz > 1, matching the CPU kernel",
            "offsets on MPS cost one bulk device-to-host read because nnz and the "
            "duplicate check are data-dependent; offsets on the host cost none. "
            "Index construction is one Metal kernel and the values are one "
            "index_select, both on the GPU",
            "out-of-range offsets that still give a non-negative count add nothing, "
            "as on CPU; a diagonal row longer than the matrix produces the same "
            "out-of-bounds row indices as the CPU kernel on rectangular shapes",
            "values keep the diagonals dtype; the MPS backend has no float64, so "
            "float64 diagonals fail on the device before this overlay is reached",
            "of the layouts the operator accepts, only sparse COO can exist on MPS "
            "in 2.14.1; CSR and CSC raise NotImplementedError",
            "timing includes the Python dispatch frame and one sparse constructor "
            "call; the CPU kernel number is a scale reference for the same build, "
            "not a cross-device speed claim",
            "the overlay's per-call cost is dominated by dispatch, one Metal kernel, "
            "one index_select and the sparse constructor rather than by nnz, so it "
            "beats the CPU kernel only once nnz is large; before this overlay the "
            "operator had no MPS implementation at all",
            "the host is shared with other builds, so a round's median is inflated "
            "by contention; the minimum is the closer estimate and the interleaved "
            "ratios are the stable part of these numbers",
        ],
    }

    out = ROOT / "measured" / "spdiags.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}")
    print(f"{len(report.rows)} checks, {report.failed} failed")
    if report.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
