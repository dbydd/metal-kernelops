"""Verify and measure the MPS overlay for ``aten::special_airy_ai``.

Run from the repository root with the environment that has torch 2.14.1 and
MPS::

    /Users/dbydd/vibe-agent-working-dir/ResearchChamber/.venv/bin/python benchmarks/verify_airy.py

The script registers the module's ``OVERLAYS`` the same way
``metal_kernelops.accelerator`` does, then checks three things against the CPU
kernel of the same torch build:

1. parity over the whole finite domain, on grids, random inputs and the branch
   endpoints;
2. aten semantics: result dtype, ``out=`` identity/resize/layout, strided
   input, 0-dim and empty input, integer/bool promotion, the dtypes upstream
   rejects, and the absence of an autograd graph;
3. cost: warm, synchronized CPU and MPS timings, plus the shader compile time.

The result lands in ``measured/airy.json``.
"""

import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch
from torch.library import Library

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metal_kernelops import airy_ops  # noqa: E402

RESULT = ROOT / "measured" / "airy.json"
OP = "aten::special_airy_ai.out"
GRID_TOLERANCE_ABS = 1e-6
GRID_TOLERANCE_REL = 1e-4


def aten_name(dtype):
    return {
        torch.float64: "Double",
        torch.float32: "Float",
        torch.float16: "Half",
        torch.bfloat16: "BFloat16",
        torch.complex64: "ComplexFloat",
        torch.int64: "Long",
        torch.int32: "Int",
        torch.bool: "Bool",
    }.get(dtype, str(dtype))


def ulp_distance(a, b, floor=1e-3):
    """Per-element distance in float32 ulps below the given magnitude floor."""
    fa = a.detach().to("cpu", torch.float32).contiguous()
    fb = b.detach().to("cpu", torch.float32).contiguous()
    a = fa.view(torch.int32)
    b = fb.view(torch.int32)
    key = lambda t: torch.where(t >= 0, t, t ^ 0x7FFFFFFF).to(torch.int64)
    diff = (key(a) - key(b)).abs()
    usable = torch.isfinite(fa) & torch.isfinite(fb) & (fa.abs() >= floor)
    diff = diff[usable]
    if diff.numel() == 0:
        return {"max": 0, "mean": 0.0, "count_over_4": 0, "count": 0}
    return {
        "max": int(diff.max()),
        "mean": float(diff.double().mean()),
        "count_over_4": int((diff > 4).sum()),
        "count": int(diff.numel()),
    }


def compare(name, mps_result, cpu_result, tolerance_rel=GRID_TOLERANCE_REL,
            tolerance_abs=GRID_TOLERANCE_ABS):
    mps_cpu = mps_result.detach().to("cpu", torch.float32)
    cpu_result = cpu_result.detach().to("cpu", torch.float32)
    diff = (mps_cpu - cpu_result).abs()
    both_nan = torch.isnan(mps_cpu) & torch.isnan(cpu_result)
    diff = torch.where(both_nan, torch.zeros_like(diff), diff)
    # Ai has zeros, so relative error is only defined against a magnitude
    # floor; 1e-3 is the scale where a float32 output still carries ~20
    # significant bits.
    envelope = diff / cpu_result.abs().clamp_min(1e-3)
    envelope = envelope[torch.isfinite(envelope)]
    ulp = ulp_distance(mps_cpu, cpu_result)
    return {
        "name": name,
        "numel": int(mps_cpu.numel()),
        "max_abs": float(diff.max()) if diff.numel() else 0.0,
        "max_rel_against_1e-3_floor": float(envelope.max()) if envelope.numel() else 0.0,
        "bit_exact_elements": int((diff == 0).sum()),
        "max_ulp": ulp["max"],
        "mean_ulp": ulp["mean"],
        "ulp_compared_elements": ulp["count"],
        "elements_over_4_ulp": ulp["count_over_4"],
        "nan_matches": bool(torch.equal(torch.isnan(mps_cpu), torch.isnan(cpu_result))),
        "within_tolerance": bool(
            (diff.max() if diff.numel() else 0.0) <= tolerance_abs
            and (envelope.max() if envelope.numel() else 0.0) <= tolerance_rel
            and torch.equal(torch.isnan(mps_cpu), torch.isnan(cpu_result))
        ),
    }


def reference_double(x):
    """High-precision Ai(x): SciPy if present, else the CPU double kernel."""
    try:
        import scipy.special as scipy_special

        return torch.from_numpy(scipy_special.airy(x.double().numpy())[0]), "scipy.special.airy (double)"
    except ImportError:
        return torch.special.airy_ai(x.double()), "torch CPU float64 kernel"


def bench(fn, reps, batch, synchronize):
    for _ in range(3):
        fn()
    if synchronize:
        torch.mps.synchronize()
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        for _ in range(batch):
            fn()
        if synchronize:
            torch.mps.synchronize()
        samples.append((time.perf_counter() - start) / batch)
    return {
        "median_ms": statistics.median(samples) * 1e3,
        "min_ms": min(samples) * 1e3,
        "reps": reps,
        "batch": batch,
        "synchronized": synchronize,
    }


def main():
    report = {
        "torch": torch.__version__,
        "device": "mps",
        "operator": "aten::special_airy_ai",
        "python": sys.version.split()[0],
    }
    checks = []

    def check(name, ok, detail=None, fatal=False):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok and fatal:
            raise AssertionError(f"{name}: {detail}")

    mps_available = torch.backends.mps.is_available()
    check("mps available", mps_available, fatal=True)
    check("import does not compile the shader", airy_ops._library is None)
    check(
        "import does not register a kernel",
        not torch._C._dispatch_has_kernel_for_dispatch_key(OP, "MPS"),
    )

    probe = torch.linspace(-2.0, 2.0, 8, device="mps")
    try:
        torch.special.airy_ai(probe)
        before = "runs"
    except NotImplementedError as exc:
        before = "NotImplementedError: " + str(exc).splitlines()[0]
    report["before_overlay"] = {"mps_public_api": before}

    library = Library("aten", "FRAGMENT")
    registered = {}
    for name, function in airy_ops.OVERLAYS:
        operator = f"aten::{name}"
        registered[operator] = {
            "kernel": function.__name__,
            "replaced_existing": torch._C._dispatch_has_kernel_for_dispatch_key(operator, "MPS"),
        }
        library.impl(name, function, "MPS")
    report["overlays"] = registered
    report["registration_scope"] = (
        "only the .out overload is registered: aten::special_airy_ai has a "
        "CompositeExplicitAutogradNonFunctional kernel that allocates the result "
        "with the meta function and redispatches to .out, so registering the "
        "functional kernel would duplicate that allocation and dtype promotion"
    )
    check("overlay registers one kernel", len(registered) == 1)
    check(
        "overlay fills a gap, does not replace",
        not any(row["replaced_existing"] for row in registered.values()),
    )
    check(
        "kernel installed on MPS",
        torch._C._dispatch_has_kernel_for_dispatch_key(OP, "MPS"),
    )
    check(
        "CPU double path untouched",
        torch.special.airy_ai(torch.tensor([0.5], dtype=torch.float64)).dtype is torch.float64,
    )

    report["upstream"] = {
        "source": "ATen native/Math.h airy_ai_forward<float>, torch 2.14.1",
        "algorithm_origin": "scipy/special/cephes/airy.c",
        "license": (
            "Cephes Math Library, 3-clause BSD, Copyright (c) 2018 Steven Moshier; "
            "SciPy, 3-clause BSD, Copyright (c) 2001-2002 Enthought, Inc., "
            "2003-2019 SciPy Developers (both reproduced in PyTorch's NOTICE)"
        ),
        "regimes": [
            "x >= 26.5 -> +0.0f (dead-tail shortcut; CPU float32 denominator overflows at 25.77)",
            "+-inf -> NaN, NaN propagates",
            "x < -2.09: AFN/AFD/AGN/AGD asymptotic oscillatory expansion",
            "2.09 <= x <= 103.892: AN/AD asymptotic decaying expansion",
            "-2.09 <= x < 2.09: ascending power series in f and g, epsilon FLT_EPSILON",
        ],
        "deviation": (
            "the ascending series is not evaluated when the AN/AD branch already "
            "determines the result; upstream evaluates and discards it"
        ),
        "precision": "Metal sin/cos/exp are not host libm; results are not bit-identical",
        "vendor_coverage": "none: no ATen special function in any of the seven vendor/ submodules",
    }

    # ---- parity over the domain -------------------------------------------
    parity = []
    domains = [
        ("linear [-30, 30], 200001", torch.linspace(-30, 30, 200001)),
        ("linear [-104, 104], 200001", torch.linspace(-104, 104, 200001)),
        ("uniform [-104, 104], 1M", (torch.rand(1 << 20, dtype=torch.float64) * 208 - 104).float()),
        ("normal 1M", torch.randn(1 << 20)),
        ("series only [-2.09, 2.09], 200001", torch.linspace(-2.09, 2.09, 200001)),
        ("decay only [2.09, 8.3203353], 20001", torch.linspace(2.09, 8.3203353, 20001)),
        ("tail [8.3203353, 103.892], 20001", torch.linspace(8.3203353, 103.892, 20001)),
        ("linear [-20, 20], 8M", torch.linspace(-20, 20, 1 << 23)),
    ]
    for name, points in domains:
        cpu = torch.special.airy_ai(points)
        mps = torch.special.airy_ai(points.to("mps"))
        row = compare(name, mps, cpu)
        parity.append(row)
        check(f"parity {name}", row["within_tolerance"], row)
    report["parity"] = parity

    endpoints = torch.tensor(
        [
            float("-inf"), float("inf"), float("nan"),
            -1e10, -103.8921, -103.892, -8.3203353, -2.0900002, -2.09, -2.0899999,
            -1e-8, -0.0, 0.0, 1e-8, 2.0899999, 2.09, 2.0900002,
            8.3203353, 8.3203363, 38.0, 103.891, 103.892, 103.893, 1e10,
        ],
        dtype=torch.float32,
    )
    endpoint_row = compare("branch endpoints", torch.special.airy_ai(endpoints.to("mps")), torch.special.airy_ai(endpoints))
    endpoint_row["points"] = [
        {"x": float(x), "cpu": float(c), "mps": float(m)}
        for x, c, m in zip(endpoints, torch.special.airy_ai(endpoints), torch.special.airy_ai(endpoints.to("mps")).cpu())
    ]
    report["endpoints"] = endpoint_row
    check("branch endpoints", endpoint_row["within_tolerance"], endpoint_row)

    # ---- branch mix profiling ---------------------------------------------
    def classify_mix(x):
        total = x.numel()
        osc = int((x < -2.09).sum())
        ser = int(((x >= -2.09) & (x < 2.09)).sum())
        decay_live = int(((x >= 2.09) & (x < 26.5)).sum())
        decay_dead = int((x >= 26.5).sum())
        # Warp divergence: label each element 0, 1, 2 (live) or 3 (dead)
        tags = torch.empty_like(x, dtype=torch.int8)
        tags[x < -2.09] = 0
        tags[(x >= -2.09) & (x < 2.09)] = 1
        tags[(x >= 2.09) & (x < 26.5)] = 2
        tags[x >= 26.5] = 3
        n_warps = total // 32
        if n_warps > 0:
            w = tags[: n_warps * 32].view(n_warps, 32)
            div = int(((w != w[:, :1]).any(dim=1)).sum())
            div_frac = div / n_warps
        else:
            div_frac = 0.0
        return {
            "numel": total,
            "oscillatory_pct": round(osc / total * 100, 2),
            "series_pct": round(ser / total * 100, 2),
            "decay_live_pct": round(decay_live / total * 100, 2),
            "decay_dead_pct": round(decay_dead / total * 100, 2),
            "divergent_warps_pct": round(div_frac * 100, 2),
        }

    report["branch_mix"] = {
        "uniform_wide_1M": classify_mix((torch.rand(1 << 20, dtype=torch.float64) * 208 - 104).float()),
        "normal_1M": classify_mix(torch.randn(1 << 20)),
        "linear_sorted_8M": classify_mix(torch.linspace(-20, 20, 1 << 23)),
    }

    # ---- reference accuracy: CPU kernel error vs double -------------------
    accuracy_grid = torch.linspace(-30, 30, 20001)
    reference, reference_name = reference_double(accuracy_grid)
    cpu32 = torch.special.airy_ai(accuracy_grid).double()
    mps32 = torch.special.airy_ai(accuracy_grid.to("mps")).cpu().double()
    finite = torch.isfinite(reference)
    accuracy = {"reference": reference_name, "grid": "linear [-30, 30], 20001"}
    for label, values in (("cpu_float32", cpu32), ("mps_float32", mps32)):
        err = (values[finite] - reference[finite]).abs()
        mask = reference[finite].abs() > 1e-6
        accuracy[label] = {
            "max_abs": float(err.max()),
            "max_rel_where_abs_gt_1e-6": float((err[mask] / reference[finite][mask].abs()).max()),
        }
    report["accuracy"] = accuracy
    check(
        "MPS error is on the CPU kernel's scale",
        accuracy["mps_float32"]["max_abs"] <= 2 * accuracy["cpu_float32"]["max_abs"] + 1e-9,
        accuracy,
    )

    # ---- aten semantics ----------------------------------------------------
    semantics = {}

    x = torch.linspace(-4, 4, 12, device="mps").reshape(3, 4)
    cpu_ref = torch.special.airy_ai(x.cpu())
    out = torch.special.airy_ai(x)
    semantics["shape"] = list(out.shape)
    semantics["dtype"] = str(out.dtype)
    check("shape preserved", out.shape == x.shape, list(out.shape))
    check("dtype preserved", out.dtype is torch.float32, str(out.dtype))

    same = torch.empty(3, 4, device="mps")
    returned = torch.special.airy_ai(x, out=same)
    check("out= returns the given tensor", returned.data_ptr() == same.data_ptr())
    resize_source = torch.empty(2, device="mps")
    resized = torch.special.airy_ai(x, out=resize_source)
    semantics["out_resize"] = {
        "asked_from": [2],
        "result_shape": list(resize_source.shape),
        "note": "CPU resizes with a deprecation warning; the overlay resizes silently",
    }
    check("out= resized to the input shape", resize_source.shape == x.shape)

    base = torch.full((3, 8), -1.0, device="mps")
    view = base[:, ::2]
    torch.special.airy_ai(x, out=view)
    semantics["out_strided"] = {
        "matches_logical_order": bool(
            compare("strided out", view, cpu_ref)["within_tolerance"]
        ),
        "gaps_untouched": bool((base[:, 1::2] == -1).all()),
    }
    check("strided out= writes logical order", semantics["out_strided"]["matches_logical_order"])
    check("strided out= leaves gaps alone", semantics["out_strided"]["gaps_untouched"])

    strided_in = torch.linspace(-4, 4, 24, device="mps").reshape(3, 8)[:, ::2]
    strided_row = compare("strided input", torch.special.airy_ai(strided_in), torch.special.airy_ai(strided_in.cpu()))
    semantics["strided_input"] = strided_row
    check("strided input", strided_row["within_tolerance"], strided_row)

    for label, tensor in (
        ("0-dim", torch.tensor(0.5, device="mps")),
        ("empty", torch.empty(0, device="mps")),
        ("large 1-D", torch.linspace(-20, 20, 1 << 20, device="mps")),
    ):
        row = compare(f"{label} input", torch.special.airy_ai(tensor), torch.special.airy_ai(tensor.cpu()))
        semantics[label] = row
        check(f"{label} input", row["within_tolerance"], row)

    dtypes = {}
    for dtype in (torch.int64, torch.int32, torch.bool):
        values = torch.tensor([0, 1, 2], dtype=dtype, device="mps")
        row = compare(
            f"{aten_name(dtype)} input",
            torch.special.airy_ai(values),
            torch.special.airy_ai(values.cpu()),
        )
        dtypes[aten_name(dtype)] = {"result_dtype": str(torch.special.airy_ai(values).dtype), **row}
        check(f"{aten_name(dtype)} input", row["within_tolerance"], row)

    unsupported = {}
    for dtype in (torch.float16, torch.bfloat16, torch.complex64):
        values = torch.tensor([0.5], dtype=dtype, device="mps")
        try:
            torch.special.airy_ai(values)
            unsupported[aten_name(dtype)] = "accepted"
        except NotImplementedError as exc:
            unsupported[aten_name(dtype)] = str(exc).splitlines()[0]
        try:
            torch.special.airy_ai(values.cpu())
            cpu_message = "accepted"
        except NotImplementedError as exc:
            cpu_message = str(exc).splitlines()[0]
        displaced = unsupported[aten_name(dtype)].replace("airy_ai_mps", "airy_ai_cpu")
        check(
            f"{aten_name(dtype)} matches CPU rejection",
            displaced == cpu_message,
            {"mps": unsupported[aten_name(dtype)], "cpu": cpu_message},
        )
    semantics["unsupported_dtypes"] = unsupported

    int_out = {}
    for dtype in (torch.int64, torch.int32, torch.bool):
        try:
            torch.special.airy_ai(x, out=torch.empty(3, 4, dtype=dtype, device="mps"))
            int_out[aten_name(dtype)] = "accepted"
        except RuntimeError as exc:
            int_out[aten_name(dtype)] = str(exc).splitlines()[0]
        try:
            torch.special.airy_ai(x.cpu(), out=torch.empty(3, 4, dtype=dtype))
            cpu_message = "accepted"
        except RuntimeError as exc:
            cpu_message = str(exc).splitlines()[0]
        check(
            f"{aten_name(dtype)} out matches CPU rejection",
            int_out[aten_name(dtype)] == cpu_message,
            {"mps": int_out[aten_name(dtype)], "cpu": cpu_message},
        )
    semantics["integer_out"] = int_out

    half_out = torch.empty(3, 4, dtype=torch.float16, device="mps")
    torch.special.airy_ai(x, out=half_out)
    semantics["half_out"] = {
        "result_dtype": str(half_out.dtype),
        "matches_cpu_half_out": bool(
            torch.equal(half_out.cpu(), torch.special.airy_ai(x.cpu(), out=torch.empty(3, 4, dtype=torch.float16)))
        ),
    }
    check("float16 out= accepted like CPU", semantics["half_out"]["matches_cpu_half_out"], semantics["half_out"])

    xg = torch.linspace(-4, 4, 12, device="mps").reshape(3, 4).requires_grad_(True)
    yg = torch.special.airy_ai(xg)
    xc = torch.linspace(-4, 4, 12).reshape(3, 4).requires_grad_(True)
    yc = torch.special.airy_ai(xc)
    semantics["autograd"] = {
        "mps_grad_fn": repr(yg.grad_fn),
        "mps_requires_grad": bool(yg.requires_grad),
        "cpu_grad_fn": repr(yc.grad_fn),
        "cpu_requires_grad": bool(yc.requires_grad),
    }
    check("no autograd graph, like CPU", yg.grad_fn is None and yc.grad_fn is None, semantics["autograd"])

    grad_out = torch.empty(3, 4, device="mps", requires_grad=True)
    torch.special.airy_ai(x, out=grad_out)
    cpu_grad_out = torch.empty(3, 4, requires_grad=True)
    torch.special.airy_ai(x.cpu(), out=cpu_grad_out)
    semantics["out_requires_grad"] = {
        "mps_requires_grad": bool(grad_out.requires_grad),
        "mps_grad_fn": repr(grad_out.grad_fn),
        "cpu_requires_grad": bool(cpu_grad_out.requires_grad),
        "cpu_grad_fn": repr(cpu_grad_out.grad_fn),
    }
    check(
        "out= keeps requires_grad, like CPU",
        grad_out.requires_grad and grad_out.grad_fn is None
        and cpu_grad_out.requires_grad and cpu_grad_out.grad_fn is None,
        semantics["out_requires_grad"],
    )

    try:
        torch.special.airy_ai(x, out=torch.empty(3, 4))
        semantics["mixed_device_out"] = "accepted"
        mixed_rejected = False
    except RuntimeError as exc:
        semantics["mixed_device_out"] = str(exc).splitlines()[0]
        mixed_rejected = True
    check("mixed-device out rejected", mixed_rejected, semantics["mixed_device_out"])

    report["semantics"] = semantics
    report["dtypes"] = dtypes

    # ---- cost --------------------------------------------------------------
    report["compile"] = {
        "compiled_on": "first MPS call",
        "state_after_import": "no shader, no registration",
    }
    timing = []
    for numel, reps, batch in ((1 << 16, 60, 4), (1 << 20, 30, 2), (1 << 23, 10, 1)):
        host = torch.linspace(-20, 20, numel)
        device = host.to("mps")
        cpu_row = bench(lambda: torch.special.airy_ai(host), reps, batch, False)
        mps_row = bench(lambda: torch.special.airy_ai(device), reps, batch, True)
        buffer = torch.empty_like(device)
        out_row = bench(lambda: torch.special.airy_ai(device, out=buffer), reps, batch, True)
        timing.append(
            {
                "numel": numel,
                "cpu": cpu_row,
                "mps": mps_row,
                "mps_out_reused": out_row,
                "cpu_over_mps": cpu_row["median_ms"] / mps_row["median_ms"],
                "cpu_over_mps_min": cpu_row["min_ms"] / mps_row["min_ms"],
                "out_reuse_gain": mps_row["median_ms"] / out_row["median_ms"],
            }
        )

    # Measure copy_ baseline floor at 8M
    dev8m = torch.linspace(-20, 20, 1 << 23, device="mps")
    buf8m = torch.empty_like(dev8m)
    copy_8m = bench(lambda: buf8m.copy_(dev8m), 10, 1, True)
    del dev8m, buf8m

    # Distribution timing comparisons (1M)
    dist_timing = []
    gen_timing = torch.Generator().manual_seed(99)
    for dname, dpts in (
        ("uniform [-104, 104]", (torch.rand(1 << 20, generator=gen_timing, dtype=torch.float64) * 208 - 104).float()),
        ("normal", torch.randn(1 << 20, generator=gen_timing)),
        ("shuffled [-20, 20]", torch.linspace(-20, 20, 1 << 20)[torch.randperm(1 << 20, generator=gen_timing)]),
    ):
        ddev = dpts.to("mps")
        crow = bench(lambda: torch.special.airy_ai(dpts), 30, 2, False)
        mrow = bench(lambda: torch.special.airy_ai(ddev), 30, 2, True)
        dist_timing.append({"distribution": dname, "cpu": crow, "mps": mrow,
                            "cpu_over_mps": crow["median_ms"] / mrow["median_ms"],
                            "cpu_over_mps_min": crow["min_ms"] / mrow["min_ms"]})
        del ddev, dpts

    report["timing"] = {
        "method": (
            "median and minimum per call; MPS synchronized before and after each "
            "sample; warmup 3. The host can run other GPU work meanwhile, so the "
            "minimum is the least-contended sample and the median is the honest "
            "loaded number"
        ),
        "sizes": timing,
        "copy_floor_8m": copy_8m,
        "distribution_comparisons_1m": dist_timing,
    }

    airy_ops._library = None
    start = time.perf_counter()
    torch.mps.compile_shader(airy_ops._SHADER)
    report["compile"]["compile_shader_ms"] = (time.perf_counter() - start) * 1e3
    airy_ops._library = None
    fresh = torch.linspace(-1, 1, 1 << 20, device="mps")
    start = time.perf_counter()
    torch.special.airy_ai(fresh)
    torch.mps.synchronize()
    report["compile"]["compile_and_first_call_ms"] = (time.perf_counter() - start) * 1e3
    report["compile"]["note"] = (
        "measured in-process after dropping the module cache; the Metal driver "
        "may reuse its on-disk shader cache, so a first-ever run can be slower"
    )
    del fresh

    failed = [row for row in checks if not row["ok"]]
    report["checks"] = {"total": len(checks), "failed": failed}
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(report, indent=1) + "\n")

    for row in parity:
        print(
            f"{row['name']:<38} max_abs {row['max_abs']:.3e}  max_ulp {row['max_ulp']:>2}  "
            f"mean_ulp {row['mean_ulp']:.3f}  over4 {row['elements_over_4_ulp']}"
        )
    print(f"endpoints: max_abs {endpoint_row['max_abs']:.3e} max_ulp {endpoint_row['max_ulp']}")
    print("accuracy:", json.dumps(accuracy))
    for row in timing:
        print(
            f"numel {row['numel']:>9}: cpu {row['cpu']['median_ms']:.3f} ms "
            f"(min {row['cpu']['min_ms']:.3f})  "
            f"mps {row['mps']['median_ms']:.3f} ms (min {row['mps']['min_ms']:.3f})  "
            f"out= {row['mps_out_reused']['min_ms']:.3f} ms  "
            f"cpu/mps {row['cpu_over_mps']:.2f}x (min {row['cpu_over_mps_min']:.2f}x)"
        )
    print(
        f"checks: {len(checks) - len(failed)} passed, {len(failed)} failed; "
        f"wrote {RESULT.relative_to(ROOT)}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
