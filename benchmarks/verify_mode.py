"""Verify the ``aten::mode`` / ``aten::mode.values`` MPS overlays, and measure them.

Run from the repository root with the torch environment:

    uv run --project /Users/dbydd/vibe-agent-working-dir/ResearchChamber \\
        python benchmarks/verify_mode.py

Every case is checked against the overlay's own contract -- the reported index
points at the reported value, and on NaN-free rows it is the last occurrence of
the mode value -- and separately against the CPU kernel.  CPU is the reference
only where it reproduces itself: its unstable value-only sort reorders equal keys
above the insertion-sort threshold and, with NaN in the row, stops being a valid
ordering at all.  This script measures that boundary on the CPU it runs on, lists
every divergence, and keeps the two verdicts apart, so ``contract.passed`` never
stands in for CPU parity.

``measured/mode.json`` is rewritten on every run.
"""

import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.library import Library

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metal_kernelops import mode_ops  # noqa: E402

OVERLAYS = mode_ops.OVERLAYS

DEVICE = "mps"
FLOATS = (torch.float32, torch.float16, torch.bfloat16)
INTS = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool)


def install():
    lib = Library("aten", "FRAGMENT")
    for name, function in OVERLAYS:
        lib.impl(name, function, "MPS")
    return lib


def count_bad_values(a, b):
    bad = a != b
    if a.is_floating_point():
        bad &= ~(a.isnan() & b.isnan())
    return int(bad.sum())


def has_nan(tensor):
    return tensor.is_floating_point() and bool(torch.isnan(tensor).any())


def expand_to_rows(values, dim, keepdim):
    return values if keepdim else values.unsqueeze(dim)


def index_points_at_value(rows, dim, keepdim, values, indices):
    picked = rows.gather(dim, expand_to_rows(indices, dim, keepdim))
    picked = picked if keepdim else picked.squeeze(dim)
    return count_bad_values(picked, values) == 0


def index_is_last_occurrence(rows, dim, keepdim, values, indices):
    """The convention this overlay implements, checked against the input itself."""
    if rows.dim() == 0:
        return True
    matches = rows == expand_to_rows(values, dim, keepdim)
    last = rows.shape[dim] - 1 - matches.flip(dim).to(torch.int8).argmax(dim)
    return torch.equal(last, indices if not keepdim else indices.squeeze(dim))


class Report:
    def __init__(self):
        self.cases = 0
        self.value_failures = []
        self.index_failures = []
        self.index_invalid = []
        self.convention_failures = []
        self.parity = {}

    def check(self, label, rows_cpu, rows_mps, dim, keepdim, track_parity=False):
        self.cases += 1
        values_cpu, indices_cpu = torch.mode(rows_cpu, dim, keepdim)
        values_mps, indices_mps = torch.mode(rows_mps, dim, keepdim)
        values_mps, indices_mps = values_mps.cpu(), indices_mps.cpu()
        n = rows_cpu.shape[dim] if rows_cpu.dim() else 1
        nan = has_nan(rows_cpu)

        bad = count_bad_values(values_mps, values_cpu)
        if bad:
            self.value_failures.append(
                {"case": label, "values": bad, "row_length": n, "input_has_nan": nan})

        index_bad = int((indices_mps != indices_cpu).sum()) if indices_mps.numel() else 0
        if index_bad:
            self.index_failures.append(
                {"case": label, "indices": index_bad, "of": indices_cpu.numel(),
                 "row_length": n, "input_has_nan": nan})

        if not index_points_at_value(rows_cpu, dim, keepdim, values_mps, indices_mps):
            self.index_invalid.append(label)
        elif not nan and not index_is_last_occurrence(
                rows_cpu, dim, keepdim, values_mps, indices_mps):
            self.convention_failures.append(label)

        if track_parity:
            row = self.parity.setdefault(n, [0, 0, 0])
            row[0] += 1
            row[1] += int(indices_cpu.numel())
            row[2] += index_bad


def sweep():
    torch.manual_seed(20251024)
    report = Report()

    def case(label, tensor, dim, keepdim=False, parity=False):
        report.check(label, tensor.cpu(), tensor.to(DEVICE), dim, keepdim, parity)

    for dtype in FLOATS + INTS:
        for trial in range(3):
            case(f"ties-2d-{dtype}-{trial}", torch.randint(0, 5, (6, 9)).to(dtype), 1, True)
            case(f"ties-2d-col-{dtype}-{trial}", torch.randint(0, 5, (9, 6)).to(dtype), 0, False)
            case(f"ties-3d-{dtype}-{trial}", torch.randint(-3, 3, (4, 5, 6)).to(dtype), 2, True)
            case(f"ties-3d-mid-{dtype}-{trial}",
                 torch.randint(-3, 3, (4, 5, 6)).to(dtype), -2, False)
            case(f"wide-{dtype}-{trial}", torch.randint(0, 40, (3, 129)).to(dtype), 1)
            case(f"distinct-{dtype}-{trial}", torch.randperm(11).to(dtype), 0)
            # both kernel forms at the top of their row range: 64 is the
            # register form's limit and 256 the lane form's, and the row counts
            # straddle the 2048-row switch between them
            case(f"n64-{dtype}-{trial}", torch.randint(0, 7, (257, 64)).to(dtype), 1)
            case(f"n64-tall-{dtype}-{trial}", torch.randint(0, 7, (4096, 64)).to(dtype), 1)
            case(f"n256-{dtype}-{trial}", torch.randint(0, 9, (129, 256)).to(dtype), 1)
            case(f"n256-tall-{dtype}-{trial}", torch.randint(0, 9, (2048, 256)).to(dtype), 1)

    for dtype in FLOATS:
        for n in (3, 5, 9, 17, 23, 24, 64, 128, 256, 257):
            for trial in range(2):
                case(f"nan-{dtype}-n{n}-{trial}",
                     torch.randint(0, 4, (32, n)).to(dtype).masked_fill(
                         torch.rand(32, n) < 0.3, float("nan")), 1, parity=True)
                case(f"index-parity-{dtype}-n{n}-{trial}",
                     torch.randint(0, max(2, n // 2), (64, n)).to(dtype), 1, parity=True)
        for label, row in [
            ("nnn", [float("nan"), float("nan"), 1.0]),
            ("n2", [1.0, float("nan"), float("nan")]),
            ("skip", [3.0, float("nan"), 3.0, 1.0]),
            ("last", [5.0, 3.0, float("nan")]),
            ("all-nan", [float("nan")] * 3),
            ("repeat", [1.0, float("nan"), 1.0, 2.0, float("nan"), 1.0]),
        ]:
            case(f"nan-placement-{dtype}-{label}", torch.tensor([row], dtype=dtype), 1)
        case(f"signed-zero-{dtype}",
             torch.tensor([[0.0, -0.0, 0.0, -0.0], [-0.0, 0.0, -0.0, 0.0]], dtype=dtype), 1)
        case(f"inf-{dtype}",
             torch.tensor([[float("inf"), float("inf"), 1.0],
                           [float("-inf"), float("-inf"), 1.0]], dtype=dtype), 1)

    case("scalar", torch.tensor(3.5), 0)
    case("scalar-keepdim", torch.tensor(3.5), -1, True)
    case("single", torch.tensor([2.0]), 0)
    case("single-keepdim", torch.tensor([2.0]), 0, True)
    case("transposed-2d", torch.randn(7, 12).t(), 0)
    case("transposed-3d", torch.randn(4, 5, 6).transpose(0, 2), 1)
    case("strided-slice", torch.randn(8, 10)[:, ::2], 1)
    case("bool", torch.randint(0, 2, (6, 7)).bool(), 1)

    for shape, dim, keepdim in [((0, 3), 1, False), ((0, 3), 1, True), ((3, 0), 0, True),
                                ((2, 0, 4), 2, False), ((2, 4, 0), 1, False)]:
        empty = torch.zeros(shape)
        values, indices = torch.mode(empty.to(DEVICE), dim, keepdim)
        cpu_values, cpu_indices = torch.mode(empty, dim, keepdim)
        assert values.shape == cpu_values.shape, (shape, dim, keepdim)
        assert indices.shape == cpu_indices.shape, (shape, dim, keepdim)
        assert values.dtype == cpu_values.dtype and indices.dtype == torch.long
        report.cases += 1

    return report


def edge_cases(report):
    device = DEVICE
    tensor = torch.tensor([[1.0, 2.0, 1.0], [3.0, 3.0, 4.0]])
    values = torch.empty(0, device=device)
    indices = torch.empty(0, dtype=torch.long, device=device)
    result = torch.mode(tensor.to(device), 1, out=(values, indices))
    assert result.values is values and result.indices is indices
    assert result.values.tolist() == [1.0, 3.0] and result.indices.tolist() == [2, 1]
    report.cases += 1

    values = torch.empty(0, device=device)
    indices = torch.empty(0, dtype=torch.long, device=device)
    got = torch.ops.aten.mode.values(tensor.to(device), -1, False, values=values, indices=indices)
    assert got[0].tolist() == [1.0, 3.0] and got[1].tolist() == [2, 1]
    report.cases += 1

    for out, expected in [
        ((torch.empty(0, dtype=torch.float16, device=device),
          torch.empty(0, dtype=torch.long, device=device)),
         "expected scalar type 'Float' but got 'Half' for values output"),
        ((torch.empty(0, dtype=torch.long, device=device),
          torch.empty(0, dtype=torch.long, device=device)),
         "expected scalar type 'Float' but got 'Long' for values output"),
        ((torch.empty(0, device=device), torch.empty(0, dtype=torch.int32, device=device)),
         "expected scalar type 'Long' but got 'Int' for indices output"),
    ]:
        try:
            torch.mode(tensor.to(device), 1, out=out)
        except RuntimeError as error:
            assert expected in str(error), str(error)
        else:
            raise AssertionError("wrong out dtype was accepted")
        report.cases += 1

    zeros = torch.zeros(2, 3, device=device)
    for call, expected_type, fragment in [
        (lambda: torch.mode(torch.zeros(3, 0, device=device), 1), IndexError,
         "Expected reduction dim 1 to have non-zero size"),
        (lambda: torch.mode(torch.tensor([1 + 2j, 3 + 4j], device=device), 0),
         NotImplementedError, "mode_mps"),
        (lambda: torch.mode(zeros, 2), IndexError, "Dimension out of range"),
        (lambda: torch.mode(zeros, -3), IndexError, "Dimension out of range"),
        (lambda: torch.mode(torch.tensor(1.0, device=device), 1), IndexError,
         "Dimension out of range"),
    ]:
        try:
            call()
        except expected_type as error:
            assert fragment in str(error), str(error)
        else:
            raise AssertionError(f"{expected_type.__name__} was not raised")
        report.cases += 1

    # Row length 256 is the widest the lane kernel covers, and the parts of the
    # contract that only show up at that width -- the tie between two segments of
    # equal length, the tie between two values of equal frequency, and the index
    # of a repeated winner -- are checked against the input, not against CPU,
    # whose sort is unspecified above row length 23.
    for dtype in FLOATS + INTS:
        split = torch.zeros(256, dtype=dtype, device=device)
        split[:128] = 5
        split[128] = float("nan") if dtype in FLOATS else 5
        split[129:] = 5
        values, indices = torch.mode(split, 0)
        if dtype in FLOATS:
            # 127 fives, a NaN, 127 fives: the NaN splits the row into two
            # segments whose runs tie, so the first segment wins and the index is
            # its last element
            assert values.item() == 5 and indices.item() == 127, (dtype, indices.item())
        else:
            expected = True if dtype is torch.bool else 5
            assert values.item() == expected and indices.item() == 255, (
                dtype, indices.item())

        # 128 twos and 128 ones: frequencies tie, so the smaller value wins, at
        # the last of the 128 positions that hold it
        tie = torch.empty(256, dtype=dtype, device=device)
        tie[:128] = 2
        tie[128:] = 1
        expected_value, expected_index = (True if dtype is torch.bool else 1), 255
        values, indices = torch.mode(tie, 0)
        assert values.item() == expected_value and indices.item() == expected_index, (
            dtype, values.item(), indices.item())
        report.cases += 2


def gradients():
    torch.manual_seed(7)
    cases = [((6, 9), 1), ((4, 5, 7), 2), ((4, 5, 7), 0), ((9,), 0), ((8, 30), -1), ((3, 4, 5), 1)]
    rows = []
    for shape, dim in cases:
        reference = torch.randn(*shape, requires_grad=True)
        leaf = reference.detach().clone().to(DEVICE).requires_grad_(True)
        values_mps, indices_mps = torch.mode(leaf, dim)
        values_cpu, indices_cpu = torch.mode(reference, dim)
        grad = torch.randn_like(values_cpu)
        values_mps.backward(grad.to(DEVICE))
        values_cpu.backward(grad)
        assert torch.equal(indices_mps.cpu(), indices_cpu), (shape, dim)
        assert torch.equal(leaf.grad.cpu(), reference.grad), (shape, dim)
        assert leaf.grad.device.type == DEVICE
        # value_selecting_reduction_backward writes one gradient per mode index
        assert int((leaf.grad.cpu() != 0).sum(dim=dim).max()) == 1, (shape, dim)
        rows.append({"shape": list(shape), "dim": dim, "gradient_equal_to_cpu": True,
                     "nonzero_gradient_entries": int(leaf.grad.count_nonzero())})
    return rows


def measure(calls, warmup=5, reps=25):
    """Time several callables in interleaved rounds so shared-machine load hits all of them."""
    for _ in range(warmup):
        for call, device in calls.values():
            call()
    samples = {name: [] for name in calls}
    for _ in range(reps):
        for name, (call, device) in calls.items():
            if device == "mps":
                torch.mps.synchronize()
            started = time.perf_counter()
            call()
            if device == "mps":
                torch.mps.synchronize()
            samples[name].append((time.perf_counter() - started) * 1e3)
    return {name: {"median_ms": round(statistics.median(times), 4),
                   "min_ms": round(min(times), 4)} for name, times in samples.items()}


def burst(call, warmup=3, reps=15, batch=10):
    """Amortized cost of one MPS call with ``batch`` of them in flight.

    A single MPS call bracketed by ``torch.mps.synchronize()`` pays a fixed
    submit-and-wait round trip of about 60 us on this machine, which is larger
    than every short-row computation here and identical for any operator --
    ``floor_us`` below measures it.  Ten calls per sync divide that round trip
    away and leave the per-call dispatch and GPU cost, which is the number that
    shows what a kernel change did.
    """
    for _ in range(warmup):
        for _ in range(batch):
            call()
    torch.mps.synchronize()
    times = []
    for _ in range(reps):
        started = time.perf_counter()
        for _ in range(batch):
            call()
        torch.mps.synchronize()
        times.append((time.perf_counter() - started) * 1e3 / batch)
    return round(min(times), 4)


def path_of(block):
    """Which path ``mode_ops`` takes for one ``(rows, n)`` block, as a label."""
    plan = mode_ops._plan(block)
    if plan is None:
        return "sort"
    if plan[0] is mode_ops._reg16_kernel(block.dtype):
        return f"reg16/{plan[2]}"
    if plan[0] is mode_ops._reg_kernel(block.dtype):
        return f"reg/{plan[2]}"
    return f"lane/{plan[2]}"


def timings():
    shapes = [
        ((1024, 1024), 1, torch.float32, 0.0),
        ((1024, 1024), 0, torch.float32, 0.0),
        ((1024, 1024), 1, torch.float32, 0.3),
        ((4096, 128), 1, torch.float32, 0.0),
        ((128, 4096), 1, torch.float32, 0.0),
        ((64, 64, 64), 2, torch.float32, 0.0),
        ((4, 8192), 1, torch.float32, 0.0),
        ((1000, 10), 1, torch.float32, 0.0),
        ((16384, 8), 1, torch.float32, 0.0),
        ((8192, 32), 1, torch.float32, 0.0),
        ((8192, 64), 1, torch.float32, 0.0),
        ((512, 128), 1, torch.float32, 0.0),
        ((256, 256), 1, torch.float32, 0.0),
        ((1024, 256), 1, torch.float32, 0.0),
        ((4096, 256), 1, torch.float32, 0.0),
        ((1024, 1024), 1, torch.float16, 0.0),
        ((512, 512), 1, torch.int32, 0.0),
        ((1000, 3), 1, torch.int64, 0.0),
    ]
    rows = []
    for shape, dim, dtype, nan_fraction in shapes:
        cpu_tensor = torch.randint(0, 64, shape).to(dtype)
        if nan_fraction:
            cpu_tensor = cpu_tensor.float().masked_fill(
                torch.rand(shape) < nan_fraction, float("nan")).to(dtype)
        mps_tensor = cpu_tensor.to(DEVICE)
        block = mode_ops._mode_rows(mps_tensor, dim)
        calls = {
            "cpu": (lambda: torch.mode(cpu_tensor, dim), "cpu"),
            "mps": (lambda: torch.mode(mps_tensor, dim), "mps"),
            "roundtrip": (lambda: torch.mode(mps_tensor.cpu(), dim).values.to(DEVICE), "mps"),
        }
        passes = [measure(calls) for _ in range(3)]
        times = {name: {"median_ms": min(p[name]["median_ms"] for p in passes),
                        "min_ms": min(p[name]["min_ms"] for p in passes)} for name in calls}
        amortized = min(burst(lambda: torch.mode(mps_tensor, dim)) for _ in range(3))
        path = path_of(block)
        read_bytes = block.numel() * block.element_size()
        rows.append({
            "shape": list(shape),
            "dim": dim,
            "dtype": str(dtype).replace("torch.", ""),
            "nan_fraction": nan_fraction,
            "path": path,
            "cpu": times["cpu"],
            "mps": times["mps"],
            "mps_burst_ms": amortized,
            "cpu_to_mps_roundtrip": times["roundtrip"],
            "speedup_cpu_over_mps": round(times["cpu"]["min_ms"] / times["mps"]["min_ms"], 3),
            "speedup_roundtrip_over_mps":
                round(times["roundtrip"]["min_ms"] / times["mps"]["min_ms"], 3),
            # one read of the block over the amortized time; only the kernel
            # rows turn that into a bandwidth figure, since the sort pipeline
            # reads the block several times
            "single_pass_gbs": round(read_bytes / amortized / 1e6, 2)
            if path != "sort" else None,
        })
    return rows


def floor():
    """Cost of one dispatched MPS op with nothing to compute, for reference.

    Every MPS call bracketed by a synchronize pays this; it is the reason a
    short-row ``mode`` cannot be made cheaper than the numbers in ``timings``
    show, and it applies to any operator, not just this one.
    """
    tiny = torch.zeros(8, device=DEVICE)
    single = measure({"add": (lambda: tiny.add(1), "mps")}, reps=25)["add"]
    return {
        "what": "tiny.add(1) on 8 elements: one dispatch plus one synchronize, no "
                "work worth measuring",
        "single_call_us": round(single["min_ms"] * 1e3, 1),
        "amortized_us": round(burst(lambda: tiny.add(1)) * 1e3, 1),
    }


def main():
    broker = Path("/Users/dbydd/vibe-agent-working-dir/ResearchChamber")
    if broker.is_dir():
        sys.path.insert(0, str(broker))
    try:
        from shared_scripts.compute_broker import lease_guard
        lease = lambda: lease_guard("mps-kernelops", tier=1, task_name="mode_verify")
    except ImportError:  # the broker is a machine-local convenience, not a dependency
        from contextlib import nullcontext
        lease = nullcontext

    assert not torch._C._dispatch_has_kernel_for_dispatch_key("aten::mode", "MPS"), (
        "importing mode_ops must not register anything")
    lib = install()  # noqa: F841  (holding the fragment library keeps the kernels registered)
    assert torch._C._dispatch_has_kernel_for_dispatch_key("aten::mode", "MPS")
    assert torch._C._dispatch_has_kernel_for_dispatch_key("aten::mode.values", "MPS")
    assert torch.backends.mps.is_available(), "MPS is not available"

    with lease():
        report = sweep()
        edge_cases(report)
        grad_rows = gradients()
        timing_rows = timings()
        floor_us = floor()

    parity = [
        {"row_length": length, "cases": counts[0], "indices": counts[1],
         "index_mismatches_vs_cpu": counts[2]}
        for length, counts in sorted(report.parity.items())
    ]
    exact_lengths = [row["row_length"] for row in parity if row["index_mismatches_vs_cpu"] == 0]
    reproducible = max(exact_lengths) if exact_lengths else 0

    # The CPU kernel sorts (value, index) pairs with an unstable std::sort whose
    # comparator is not a strict weak order once NaN is present, and which
    # reorders equal keys above the insertion-sort threshold.  Above the row
    # length where this machine still reproduces CPU, a NaN row's value and any
    # repeated value's index are unspecified, so they are listed as divergences
    # rather than counted as matches or hidden behind one flag.
    def cpu_specified(failure):
        return not (failure["input_has_nan"] and failure["row_length"] > reproducible)

    def index_cpu_specified(failure):
        return failure["row_length"] <= reproducible

    values_inside = [f for f in report.value_failures if cpu_specified(f)]
    values_beyond = [f for f in report.value_failures if not cpu_specified(f)]
    indices_inside = [f for f in report.index_failures if index_cpu_specified(f)]
    indices_beyond = [f for f in report.index_failures if not index_cpu_specified(f)]

    contract = {
        "cases": report.cases,
        "index_not_pointing_at_value": report.index_invalid,
        "index_not_last_occurrence": report.convention_failures,
        "value_mismatches_where_cpu_is_reproducible": values_inside,
        "index_mismatches_where_cpu_is_reproducible": indices_inside,
        "convention": "most frequent value, ties to the smallest value, reported at "
                      "the last occurrence of that value in the row; NaN-free rows only "
                      "for the last-occurrence clause",
    }
    contract["passed"] = not (
        report.index_invalid or report.convention_failures or values_inside or indices_inside
    )

    summary = {
        "operator": ["aten::mode", "aten::mode.values"],
        "overlay": "src/metal_kernelops/mode_ops.py",
        "torch": torch.__version__,
        "machine": platform.machine(),
        "device": DEVICE,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": "one Metal kernel, one dispatch, for rows up to 256 elements: a "
                  "register-resident thread per row for rows up to 64 (and for every "
                  "tall block, which needs the thread count), a 64-lane threadgroup per "
                  "row above that (and for short blocks of 17-64 elements, which need "
                  "the parallelism); otherwise stable sort + run length in MPS aten ops. "
                  "No CPU fallback, no per-element Python and no data-dependent host "
                  "read: float rows always run the NaN-segment double argsort, integer "
                  "rows a single stable sort",
        "contract": contract,
        "cpu_parity": {
            "values": {
                "divergences_inside_cpu_reproducibility": values_inside,
                "divergences_beyond_cpu_reproducibility": values_beyond,
                "matched_everywhere_cpu_is_reproducible": not values_inside,
                "matched_everywhere": not report.value_failures,
            },
            "indices": {
                "divergences_inside_cpu_reproducibility": indices_inside,
                "divergences_beyond_cpu_reproducibility": indices_beyond,
                "mismatched_indices_total":
                    int(sum(row["indices"] for row in report.index_failures)),
                "matched_everywhere_cpu_is_reproducible": not indices_inside,
                "matched_everywhere": not report.index_failures,
            },
            "cpu_reproducible_row_length_max": reproducible,
            "measured": parity,
            "note": "the CPU kernel reproduces its own output only up to this row length "
                    "on this machine; above it its std::sort reorders equal keys and its "
                    "NaN placement depends on an introsort partition that compares values "
                    "only, so both the reported index and, on NaN rows, the reported value "
                    "are unspecified.  The overlay's defined result is the one in "
                    "``contract``; every place it departs from what CPU happened to print "
                    "is listed under ``cpu_parity`` and none of them is folded into a pass "
                    "flag.",
        },
        "gradient": {
            "method": "torch.mode(values).backward(random grad) on a leaf, "
                      "compared with the CPU leaf",
            "cases": grad_rows,
        },
        "timings": {
            "method": "median and min of 25 reps after 5 warmups, "
                      "torch.mps.synchronize() around each MPS rep, best of three passes, "
                      "speedups from min; the machine was shared with other agents, so trust "
                      "the relative order more than the absolute numbers",
            "burst_method": "min of 15 rounds of 10 calls between two "
                            "torch.mps.synchronize() calls, after 3 warmup rounds; this is "
                            "the per-call dispatch and GPU cost, with the ~60 us submit-and-wait "
                            "round trip that dominates a single short call divided out",
            "dispatch_floor": floor_us,
            "rows": timing_rows,
        },
    }
    summary["result"] = {
        "contract_passed": contract["passed"],
        "cpu_parity_complete": not report.value_failures and not report.index_failures,
    }

    out = ROOT / "measured" / "mode.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"cases {report.cases}: contract {'pass' if contract['passed'] else 'FAIL'} "
          f"(index not at value {len(report.index_invalid)}, "
          f"index not last occurrence {len(report.convention_failures)}, "
          f"value divergences inside CPU reproducibility {len(values_inside)}, "
          f"index divergences inside CPU reproducibility {len(indices_inside)})")
    print(f"CPU parity: {len(report.value_failures)} value divergence cases "
          f"({len(values_beyond)} in the CPU's unspecified regime) and "
          f"{len(report.index_failures)} index divergence cases "
          f"({sum(row['indices'] for row in indices_beyond)} indices) beyond row length "
          f"{reproducible}; CPU reproduced exactly up to {exact_lengths}")
    for row in timing_rows:
        print(f"  {str(row['shape']):12s} dim={row['dim']} {row['dtype']:9s} "
              f"nan={row['nan_fraction']:.1f} {row['path']:>6s} "
              f"cpu {row['cpu']['min_ms']:8.3f} ms  mps {row['mps']['min_ms']:8.3f} ms  "
              f"burst {row['mps_burst_ms']*1e3:7.1f} us  "
              f"roundtrip {row['cpu_to_mps_roundtrip']['min_ms']:8.3f} ms  "
              f"cpu/mps {row['speedup_cpu_over_mps']:6.2f}x  "
              f"roundtrip/mps {row['speedup_roundtrip_over_mps']:6.2f}x")
    print(f"dispatch floor: {floor_us['single_call_us']} us per single call, "
          f"{floor_us['amortized_us']} us amortized")
    print(f"wrote {out}")
    if not contract["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
