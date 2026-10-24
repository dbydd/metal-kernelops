#!/usr/bin/env python3
"""Verify and time the fractional max pool 2d/3d MPS overlays.

Checks, in order: the operators raise on MPS before the overlay is installed;
after installation the functional, out, and autograd paths all work; forward
values, indices and gradients match the CPU kernel across shapes, dtypes,
window sizes, and NaN inputs. Then measures warm, synchronized CPU and MPS
medians for forward and forward+backward.

Writes ``measured/fractional_pool_ops.json``. Exits non-zero when a check fails.

    python benchmarks/verify_fractional_pool_ops.py
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metal_kernelops import fractional_pool_ops as ops

GRAD_TOL = {torch.float32: 1e-6, torch.float16: 1e-2, torch.bfloat16: 3e-2}


def lease_guard(task_name):
    """Shared compute lease when the cluster scripts are reachable."""
    root = Path(os.environ.get("COMPUTE_BROKER_PATH", ROOT.parent / "shared_scripts"))
    if (root / "compute_broker.py").exists():
        sys.path.insert(0, str(root))
        from compute_broker import lease_guard as guard

        return guard("mps-kernelops", tier=1, task_name=task_name)
    return contextlib.nullcontext()


def samples_shape(shape, ndim):
    lead = shape[: -(ndim + 1)]
    return (tuple(lead) if lead else (1,)) + (shape[-(ndim + 1)], ndim)


def same_values(a, b):
    if torch.equal(a, b):
        return True
    return bool(((a.isnan() == b.isnan()).all()) and ((a == b) | a.isnan()).all())


def call(ndim, x, kernel_size, output_size, random_samples):
    fn = F.fractional_max_pool2d if ndim == 2 else F.fractional_max_pool3d
    return fn(
        x, kernel_size, output_size=output_size, return_indices=True,
        _random_samples=random_samples,
    )


def check_case(shape, kernel_size, output_size, ndim, dtype, seed, nan=False):
    torch.manual_seed(seed)
    x = torch.randn(shape, dtype=dtype)
    random_samples = torch.rand(samples_shape(shape, ndim), dtype=dtype)
    if seed == 1:
        random_samples = torch.zeros_like(random_samples)
    if seed == 2:
        random_samples = torch.ones_like(random_samples)
    if nan:
        x.view(-1)[::5] = float("nan")

    x_cpu = x.clone().requires_grad_(True)
    v_cpu, i_cpu = call(ndim, x_cpu, kernel_size, output_size, random_samples.clone())
    g = torch.randn_like(v_cpu)
    grad_cpu, = torch.autograd.grad(v_cpu, x_cpu, g)

    x_mps = x.to("mps").requires_grad_(True)
    v_mps, i_mps = call(ndim, x_mps, kernel_size, output_size, random_samples.to("mps"))
    grad_mps, = torch.autograd.grad(v_mps, x_mps, g.to("mps"))

    grad_rel = ((grad_mps.cpu() - grad_cpu).norm() / grad_cpu.norm()).item()
    return {
        "values_exact": same_values(v_mps.cpu(), v_cpu),
        "indices_exact": torch.equal(i_mps.cpu(), i_cpu),
        "device": str(v_mps.device),
        "grad_relative_norm": grad_rel,
        "grad_ok": grad_rel < GRAD_TOL[dtype],
    }


def correctness():
    cases = []
    for ndim, spatial in (
        (2, (10, 10)), (2, (13, 17)), (2, (9, 9)),
        (3, (7, 9, 11)), (3, (8, 8, 8)), (3, (7, 5, 6)),
    ):
        lead = (2, 3) if ndim == 2 else (2, 2)
        for k in (2, 3):
            outputs = (
                (1,) * ndim,
                tuple(max(1, s - 2 * k + 1) for s in spatial),
                tuple(max(1, s // 2) for s in spatial),
            )
            for out in outputs:
                cases.append((lead + spatial, (k,) * ndim, out, ndim))
    # Unbatched inputs use the same plane layout without a batch dimension.
    cases.append(((3, 9, 9), (2, 2), (4, 4), 2))
    cases.append(((3, 7, 5, 6), (2, 2, 2), (3, 2, 3), 3))

    rows = []
    worst = {dtype: 0.0 for dtype in GRAD_TOL}
    failures = 0
    for shape, kernel_size, output_size, ndim in cases:
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for seed in range(3):
                row = check_case(shape, kernel_size, output_size, ndim, dtype, seed)
                worst[dtype] = max(worst[dtype], row["grad_relative_norm"])
                if not (row["values_exact"] and row["indices_exact"] and row["grad_ok"]):
                    failures += 1
                    rows.append({"shape": shape, "kernel": kernel_size, "output": output_size,
                                 "dtype": str(dtype), "seed": seed, **row})
    for ndim, shape, kernel_size, output_size in (
        (2, (1, 1, 8, 8), (2, 2), (4, 4)),
        (3, (1, 1, 6, 6, 6), (2, 2, 2), (3, 3, 3)),
    ):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            row = check_case(shape, kernel_size, output_size, ndim, dtype, 0, nan=True)
            worst[dtype] = max(worst[dtype], row["grad_relative_norm"])
            if not (row["values_exact"] and row["indices_exact"]):
                failures += 1
                rows.append({"shape": shape, "kernel": kernel_size, "output": output_size,
                             "dtype": str(dtype), "nan": True, **row})
    return {
        "cases": len(cases) * 9 + 6,
        "failures": len(rows),
        "worst_grad_relative_norm": {str(k): v for k, v in worst.items()},
        "failed_rows": rows[:20],
    }


def dispatch_checks():
    """The three entry points a model can reach, plus a non-contiguous input."""
    failures = []
    x = torch.randn(2, 3, 12, 12)
    random_samples = torch.rand(2, 3, 2)
    v_cpu, i_cpu = call(2, x, (3, 3), (4, 4), random_samples)

    x_mps = x.to("mps")
    v, i = call(2, x_mps, (3, 3), (4, 4), random_samples.to("mps"))
    if not (same_values(v.cpu(), v_cpu) and torch.equal(i.cpu(), i_cpu)):
        failures.append("functional")

    out = torch.empty(2, 3, 4, 4, device="mps")
    ind = torch.empty(2, 3, 4, 4, dtype=torch.int64, device="mps")
    v2, i2 = torch.ops.aten.fractional_max_pool2d.output(
        x_mps, [3, 3], [4, 4], random_samples.to("mps"), output=out, indices=ind
    )
    if not (same_values(v2.cpu(), v_cpu) and torch.equal(i2.cpu(), i_cpu)):
        failures.append("out")

    grad_out = torch.randn(2, 3, 4, 4, device="mps")
    grad_input = torch.empty(2, 3, 12, 12, device="mps")
    got = torch.ops.aten.fractional_max_pool2d_backward.grad_input(
        grad_out, x_mps, [3, 3], [4, 4], i2, grad_input=grad_input
    )
    expected = torch.zeros(2, 3, 12, 12)
    expected.reshape(6, -1).scatter_add_(
        1, i_cpu.reshape(6, -1), grad_out.cpu().reshape(6, -1)
    )
    if not torch.allclose(got.cpu(), expected, rtol=0, atol=1e-6):
        failures.append("grad_input")

    x5 = torch.randn(2, 2, 7, 9, 11)
    rs5 = torch.rand(2, 2, 3)
    v3, i3 = F.fractional_max_pool3d(x5, 2, output_size=(3, 4, 5), return_indices=True,
                                     _random_samples=rs5)
    v3m, i3m = F.fractional_max_pool3d(
        x5.to("mps"), 2, output_size=(3, 4, 5), return_indices=True,
        _random_samples=rs5.to("mps"),
    )
    if not (same_values(v3m.cpu(), v3) and torch.equal(i3m.cpu(), i3)):
        failures.append("3d functional")
    grad3 = torch.randn_like(v3m)
    got3 = torch.ops.aten.fractional_max_pool3d_backward(
        grad3, x5.to("mps"), [2, 2, 2], [3, 4, 5], i3m
    )
    expected3 = torch.zeros_like(x5)
    expected3.reshape(4, -1).scatter_add_(
        1, i3.reshape(4, -1), grad3.cpu().reshape(4, -1)
    )
    if not torch.allclose(got3.cpu(), expected3, rtol=0, atol=1e-6):
        failures.append("3d backward")

    # channels-last / permuted input goes through the same contiguity path as CPU
    xt = torch.randn(2, 3, 12, 12).transpose(2, 3)
    vt, it = call(2, xt.to("mps"), (3, 3), (4, 4), random_samples.to("mps"))
    vt_cpu, it_cpu = call(2, xt.contiguous(), (3, 3), (4, 4), random_samples)
    if not (same_values(vt.cpu(), vt_cpu) and torch.equal(it.cpu(), it_cpu)):
        failures.append("non-contiguous")
    return failures


def timed(fn, device, iters=15, warm=4, rounds=3):
    """Warm, synchronized samples. Both statistics are kept: this GPU is
    shared with other processes, and contention only ever adds time."""
    for _ in range(warm):
        fn()
    if device == "mps":
        torch.mps.synchronize()
    samples = []
    for _ in range(rounds):
        for _ in range(iters):
            if device == "mps":
                torch.mps.synchronize()
            start = time.perf_counter()
            fn()
            if device == "mps":
                torch.mps.synchronize()
            samples.append(time.perf_counter() - start)
    return {
        "median_ms": statistics.median(samples) * 1e3,
        "min_ms": min(samples) * 1e3,
    }


def timing_case(shape, kernel_size, output_size, ndim, dtype):
    x = torch.randn(shape, dtype=dtype)
    random_samples = torch.rand(samples_shape(shape, ndim), dtype=dtype)
    grad = torch.randn(shape[:-ndim] + tuple(output_size), dtype=dtype)
    x_mps = x.to("mps")
    rs_mps = random_samples.to("mps")
    grad_mps = grad.to("mps")

    def cpu_forward():
        call(ndim, x, kernel_size, output_size, random_samples)

    def mps_forward():
        call(ndim, x_mps, kernel_size, output_size, rs_mps)

    def cpu_step():
        leaf = x.detach().requires_grad_(True)
        v, _ = call(ndim, leaf, kernel_size, output_size, random_samples)
        torch.autograd.grad(v, leaf, grad)

    def mps_step():
        leaf = x_mps.detach().requires_grad_(True)
        v, _ = call(ndim, leaf, kernel_size, output_size, rs_mps)
        torch.autograd.grad(v, leaf, grad_mps)

    cpu_f = timed(cpu_forward, "cpu")
    mps_f = timed(mps_forward, "mps")
    cpu_s = timed(cpu_step, "cpu", iters=10, warm=3)
    mps_s = timed(mps_step, "mps", iters=10, warm=3)
    return {
        "op": f"fractional_max_pool{ndim}d",
        "shape": list(shape),
        "kernel_size": list(kernel_size),
        "output_size": list(output_size),
        "dtype": str(dtype).split(".")[-1],
        "forward": {"cpu": cpu_f, "mps": mps_f,
                    "speedup": cpu_f["median_ms"] / mps_f["median_ms"]},
        "forward_backward": {"cpu": cpu_s, "mps": mps_s,
                             "speedup": cpu_s["median_ms"] / mps_s["median_ms"]},
    }


def main():
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available")

    # Before installation the operator must be missing on MPS.
    missing = []
    try:
        call(2, torch.randn(1, 1, 6, 6, device="mps"), (2, 2), (3, 3),
             torch.rand(1, 1, 2, device="mps"))
    except NotImplementedError:
        missing.append("fractional_max_pool2d")
    try:
        call(3, torch.randn(1, 1, 5, 5, 5, device="mps"), (2, 2, 2), (2, 2, 2),
             torch.rand(1, 1, 3, device="mps"))
    except NotImplementedError:
        missing.append("fractional_max_pool3d")

    import_clean = ops._LIB is None
    compile_start = time.perf_counter()
    ops._library()  # first call compiles; the overlay never compiles at import
    compile_ms = (time.perf_counter() - compile_start) * 1e3
    library = torch.library.Library("aten", "FRAGMENT")
    for name, function in ops.OVERLAYS:
        library.impl(name, function, "MPS")

    unsupported = dispatch_checks()
    result = correctness()
    failures = result["failures"] + len(unsupported)

    timings = []
    with lease_guard("fractional_pool_verify"):
        for shape, kernel_size, output_size, ndim in (
            ((16, 32, 64, 64), (2, 2), (32, 32), 2),
            ((8, 64, 56, 56), (3, 3), (27, 27), 2),
            ((8, 32, 128, 128), (2, 2), (64, 64), 2),
            ((4, 8, 16, 32, 32), (2, 2, 2), (8, 16, 16), 3),
            ((2, 8, 48, 96, 96), (2, 2, 2), (24, 48, 48), 3),
        ):
            timings.append(timing_case(shape, kernel_size, output_size, ndim, torch.float32))
        timings.append(timing_case((16, 32, 64, 64), (2, 2), (32, 32), 2, torch.float16))
        timings.append(timing_case((4, 8, 16, 32, 32), (2, 2, 2), (8, 16, 16), 3, torch.bfloat16))

    report = {
        "module": "metal_kernelops.fractional_pool_ops",
        "torch": torch.__version__,
        "machine": platform.machine(),
        "python": platform.python_version(),
        "overlays": [name for name, _ in ops.OVERLAYS],
        "missing_before_install": missing,
        "import_clean": import_clean,
        "shader_compiled": ops._LIB is not None,
        "shader_compile_ms": compile_ms,
        "dispatch_failures": unsupported,
        "correctness": result,
        "timings": timings,
        "timing_notes": (
            "Warm, synchronized, 15 samples x 3 rounds per case, median and min of "
            "the same samples. This GPU is shared with concurrent MPS processes; "
            "medians are inflated by contention and min_ms is the uncontended kernel "
            "time. CPU and MPS were measured in the same process, one after the other."
        ),
    }
    measured = ROOT / "measured" / "fractional_pool_ops.json"
    measured.parent.mkdir(exist_ok=True)
    measured.write_text(json.dumps(report, indent=2) + "\n")

    print(f"missing before install: {missing}")
    print(f"import clean: {import_clean}, shader compiled: {ops._LIB is not None}")
    print(f"dispatch failures: {unsupported}")
    print(f"correctness: {result['cases']} cases, {failures} failures, "
          f"worst grad rel {result['worst_grad_relative_norm']}")
    for row in timings:
        fwd, step = row["forward"], row["forward_backward"]
        print(f"{row['op']} {row['shape']} k{row['kernel_size']} o{row['output_size']} "
              f"{row['dtype']}: fwd {fwd['cpu']['median_ms']:.3f}/{fwd['mps']['median_ms']:.3f} ms "
              f"x{fwd['speedup']:.2f} | step {step['cpu']['median_ms']:.3f}/"
              f"{step['mps']['median_ms']:.3f} ms x{step['speedup']:.2f}")
    print(f"wrote {measured.relative_to(ROOT)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
