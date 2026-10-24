"""Verify and time the 3-D adaptive pooling overlays against the CPU kernel.

    uv run --project /Users/dbydd/vibe-agent-working-dir/ResearchChamber python \
        benchmarks/verify_adaptive_pool_ops.py [--bench]

Checks, for every case: forward values, max indices (exact), the two backward
operators, the ``.out`` overloads, the autograd path through ``nn`` modules,
and the non-contiguous / 4-D / >input / dtype variants.  ``--bench`` adds warm
synchronized timings against the CPU kernel and writes ``measured/<stem>.json``.
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metal_kernelops import adaptive_pool_ops  # noqa: E402

# Against the CPU kernel; None means the CPU number itself is too loose to
# compare with (it accumulates half and bfloat16 averages in the narrow type).
# EXACT is the bound held against a float64 reference, for every dtype.
TOLERANCE = {torch.float32: 1e-5, torch.float16: 3e-2, torch.bfloat16: 5e-2}
EXACT = {torch.float32: 1e-6, torch.float16: 2e-3, torch.bfloat16: 1e-2}
_DEFAULT = object()
CASES = [
    ((2, 3, 5, 7, 3), (4, 3, 2)),   # nondivisible, mixed shrink
    ((2, 3, 8, 8, 8), (4, 4, 4)),   # divisible
    ((1, 2, 3, 3, 3), (5, 5, 5)),   # output larger than input
    ((2, 3, 5, 7, 3), (5, 7, 3)),   # identity
    ((2, 3, 4, 6, 8), (1, 1, 1)),   # global
    ((3, 6, 9, 7), (3, 4, 5)),      # 4-D input
    ((2, 2, 7, 7, 7), (3, 3, 3)),   # prime input, odd output
    ((1, 1, 13, 17, 19), (6, 8, 9)),
    ((1, 1, 32, 32, 32), (4, 4, 4)),   # cooperative average, divisible
    ((2, 3, 30, 30, 30), (4, 4, 4)),   # cooperative average, nondivisible
    ((1, 1, 32, 32, 32), (2, 2, 2)),   # cooperative average, groups per bin
    ((3, 20, 20, 20), (2, 2, 2)),      # cooperative average, 4-D input
]
DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def check(name, actual, expected, dtype, exact=False, limit=_DEFAULT):
    if exact:
        if not torch.equal(actual.cpu(), expected.cpu()):
            raise AssertionError(f"{name}: indices differ")
        return 0.0
    want = expected.cpu().float()
    scale = want.abs().max().item()
    diff = (actual.cpu().float() - want).abs().max().item() / max(scale, 1e-6)
    if not actual.cpu().float().isfinite().all():
        raise AssertionError(f"{name}: non-finite output")
    bound = TOLERANCE[dtype] if limit is _DEFAULT else limit
    if bound is not None and diff > bound:
        raise AssertionError(f"{name}: max abs diff {diff:.3e} > {bound:.1e}")
    return diff


def avg_forward(x, out, module, dtype):
    actual = module._adaptive_avg_pool3d(x, out)
    expected = torch.ops.aten._adaptive_avg_pool3d(x.cpu(), out)
    # bfloat16 is compared against the exact value below instead: CPU's own
    # bfloat16 average can be tens of percent off the exact mean.
    limit = None if dtype is torch.bfloat16 else _DEFAULT
    diff = check("avg forward", actual, expected, dtype, limit=limit)
    truth = torch.ops.aten._adaptive_avg_pool3d(x.cpu().double(), out)
    scale = max(truth.abs().max().item(), 1e-6)
    ours = (actual.cpu().double() - truth).abs().max().item() / scale
    theirs = (expected.double() - truth).abs().max().item() / scale
    if ours > EXACT[dtype]:
        raise AssertionError(f"avg forward vs float64: {ours:.2e} > {EXACT[dtype]:.1e}")
    if ours > theirs + 1e-3:
        raise AssertionError(
            f"avg forward is less accurate than CPU: {ours:.2e} vs {theirs:.2e}"
        )
    return diff


def run_case(shape, out, dtype):
    """One (shape, output size, dtype) case: every operator and overload."""
    torch.manual_seed(hash((shape, out, dtype)) & 0xFFFF)
    x = torch.randn(*shape, dtype=dtype)
    xm = x.to("mps")
    report = {"input": list(shape), "output": list(out), "dtype": str(dtype)}

    report["avg"] = avg_forward(xm, out, adaptive_pool_ops, dtype)
    avg_expected = torch.ops.aten._adaptive_avg_pool3d(x, out)

    grad = torch.randn(*avg_expected.shape, dtype=dtype)
    actual = adaptive_pool_ops._adaptive_avg_pool3d_backward(grad.to("mps"), xm)
    expected = torch.ops.aten._adaptive_avg_pool3d_backward(grad, x)
    report["avg_backward"] = check("avg backward", actual, expected, dtype)

    got, idx = adaptive_pool_ops._adaptive_max_pool3d(xm, out)
    want, w_idx = torch.ops.aten.adaptive_max_pool3d(x, out)
    report["max"] = check("max forward", got, want, dtype)
    check("max indices", idx, w_idx, dtype, exact=True)
    if idx.dtype != torch.int64 or idx.shape != got.shape:
        raise AssertionError("max indices dtype/shape")

    mgrad = torch.randn(*want.shape, dtype=dtype)
    actual = adaptive_pool_ops._adaptive_max_pool3d_backward(
        mgrad.to("mps"), xm, idx
    )
    expected = torch.ops.aten.adaptive_max_pool3d_backward(mgrad, x, w_idx)
    report["max_backward"] = check("max backward", actual, expected, dtype)

    # .out overloads, including a wrong-shaped destination that has to resize.
    empty = torch.empty(0, device="mps", dtype=dtype)
    avg_out = torch.ops.aten._adaptive_avg_pool3d.out(xm, list(out), out=empty)
    # bfloat16 accumulates in the narrow type on CPU, which drifts away from
    # the exact mean as the bin widens (by tenths for 64^3 bins), so the .out
    # overload is compared against the functional call that was just held to
    # the float64 truth.
    check(
        "avg out",
        avg_out,
        adaptive_pool_ops._adaptive_avg_pool3d(xm, out),
        dtype,
        exact=True,
    )
    if avg_out.data_ptr() != empty.data_ptr():
        raise AssertionError("avg out did not write into the given destination")

    metadata = torch.empty(0, device="mps", dtype=dtype)
    index_slot = torch.empty(0, device="mps", dtype=torch.int64)
    max_out, index_out = torch.ops.aten.adaptive_max_pool3d.out(
        xm, list(out), out=metadata, indices=index_slot
    )
    check("max out", max_out, want, dtype)
    check("max out indices", index_out, w_idx, dtype, exact=True)
    if max_out.data_ptr() != metadata.data_ptr():
        raise AssertionError("max out did not write into the given destination")
    if index_out.data_ptr() != index_slot.data_ptr():
        raise AssertionError("max indices did not write into the given destination")

    grad_in = torch.empty(0, device="mps", dtype=dtype)
    avg_bwd_out = torch.ops.aten._adaptive_avg_pool3d_backward.out(
        grad.to("mps"), xm, out=grad_in
    )
    check(
        "avg backward out",
        avg_bwd_out,
        torch.ops.aten._adaptive_avg_pool3d_backward(grad, x),
        dtype,
    )
    if avg_bwd_out.data_ptr() != grad_in.data_ptr():
        raise AssertionError("avg backward out did not write into the destination")

    grad_in = torch.empty(0, device="mps", dtype=dtype)
    max_bwd_out = torch.ops.aten.adaptive_max_pool3d_backward.grad_input(
        mgrad.to("mps"), xm, idx, grad_input=grad_in
    )
    check("max backward grad_input", max_bwd_out, expected, dtype)
    if max_bwd_out.data_ptr() != grad_in.data_ptr():
        raise AssertionError("max grad_input did not write into the destination")
    return report


def run_reference(dtype, module):
    """Ties, NaN, non-contiguous input and the autograd path."""
    x = torch.zeros(1, 1, 4, 4, 4, dtype=dtype)
    x[0, 0, 0, 0, 0] = 1
    x[0, 0, 0, 1, 1] = 1
    x[0, 0, 1, 0, 1] = 1
    got, idx = module._adaptive_max_pool3d(x.to("mps"), (2, 2, 2))
    want, w_idx = torch.ops.aten.adaptive_max_pool3d(x, (2, 2, 2))
    check("tie values", got, want, dtype)
    check("tie indices", idx, w_idx, dtype, exact=True)

    # Nondivisible bins (4 -> 3) go through the decomposition: NaN values, the
    # last NaN as the index, exactly as the CPU kernel does.
    x = torch.randn(1, 1, 4, 4, 4, dtype=dtype)
    x[0, 0, 0, 0, 0] = float("nan")
    x[0, 0, 1, 1, 1] = float("nan")
    got, idx = module._adaptive_max_pool3d(x.to("mps"), (3, 3, 3))
    want, w_idx = torch.ops.aten.adaptive_max_pool3d(x, (3, 3, 3))
    if not (torch.isnan(got.cpu()) == torch.isnan(want)).all():
        raise AssertionError("nan propagation")
    check("nan indices", idx, w_idx, dtype, exact=True)

    # Divisible bins: the fused kernel keeps CPU's rule -- NaN, last NaN's index.
    z = torch.zeros(1, 1, 2, 2, 2, dtype=dtype)
    z[0, 0, 0, 0, 0] = float("nan")
    z[0, 0, 1, 1, 1] = float("nan")
    got, idx = module._adaptive_max_pool3d(z.to("mps"), (1, 1, 1))
    want, w_idx = torch.ops.aten.adaptive_max_pool3d(z, (1, 1, 1))
    if not (torch.isnan(got).all() and torch.isnan(want).all()):
        raise AssertionError("fast-path nan propagation")
    check("nan index", idx, w_idx, dtype, exact=True)

    # non-contiguous input, both operators
    base = torch.randn(2, 6, 10, 9, dtype=dtype)
    x = base.transpose(0, 1)[:, :2]
    xm = x.to("mps")
    check(
        "strided avg",
        module._adaptive_avg_pool3d(xm, (3, 4, 3)),
        torch.ops.aten._adaptive_avg_pool3d(x, (3, 4, 3)),
        dtype,
    )
    got, idx = module._adaptive_max_pool3d(xm, (3, 4, 3))
    want, w_idx = torch.ops.aten.adaptive_max_pool3d(x, (3, 4, 3))
    check("strided max", got, want, dtype)
    check("strided max indices", idx, w_idx, dtype, exact=True)

    # non-contiguous, divisible: the native kernels take strided input too
    x = torch.randn(2, 3, 8, 16, 16, dtype=dtype).transpose(2, 3)
    xm = x.to("mps")
    check(
        "strided divisible avg",
        module._adaptive_avg_pool3d(xm, (8, 4, 8)),
        torch.ops.aten._adaptive_avg_pool3d(x, (8, 4, 8)),
        dtype,
    )
    got, idx = module._adaptive_max_pool3d(xm, (8, 4, 8))
    want, w_idx = torch.ops.aten.adaptive_max_pool3d(x, (8, 4, 8))
    check("strided divisible max", got, want, dtype)
    check("strided divisible max indices", idx, w_idx, dtype, exact=True)

    # tie-heavy integer data: 40 random divisible shapes, indices must match
    for trial in range(40):
        sizes = [int(torch.randint(2, 7, (1,))) for _ in range(3)]
        kernel = [int(torch.randint(1, 3, (1,))) for _ in range(3)]
        shape = [2, 2] + [s * k for s, k in zip(sizes, kernel)]
        sample = torch.randint(0, 3, shape).to(dtype)
        got, idx = module._adaptive_max_pool3d(sample.to("mps"), tuple(sizes))
        want, w_idx = torch.ops.aten.adaptive_max_pool3d(sample, tuple(sizes))
        check(f"tie-heavy {trial} values", got, want, dtype)
        check(f"tie-heavy {trial} indices", idx, w_idx, dtype, exact=True)

    # autograd: functional and module paths, both operators
    for dtype_name in ("float32", "float16"):
        dt = getattr(torch, dtype_name)
        x = torch.randn(2, 3, 6, 7, 5, dtype=dt, device="mps", requires_grad=True)
        pool = nn.AdaptiveAvgPool3d((3, 4, 2))
        (pool(x).float() ** 2).sum().backward()
        grad_mps = x.grad.clone()
        xc = x.detach().cpu().double().requires_grad_(True)
        (nn.AdaptiveAvgPool3d((3, 4, 2))(xc) ** 2).sum().backward()
        rel = (
            (grad_mps.cpu().double() - xc.grad).norm() / xc.grad.norm().clamp_min(1e-12)
        ).item()
        if rel > (1e-6 if dt == torch.float32 else 5e-3):
            raise AssertionError(f"avg autograd {dtype_name}: relative {rel:.2e}")

        x = torch.randn(2, 3, 6, 7, 5, dtype=dt, device="mps", requires_grad=True)
        y, ind = nn.AdaptiveMaxPool3d((3, 4, 2), return_indices=True)(x)
        y.float().sum().backward()
        grad_mps = x.grad.clone()
        xc = x.detach().cpu().double().requires_grad_(True)
        yc, ic = nn.AdaptiveMaxPool3d((3, 4, 2), return_indices=True)(xc)
        yc.sum().backward()
        if not torch.equal(ind.cpu(), ic):
            raise AssertionError(f"max autograd indices {dtype_name}")
        rel = (
            (grad_mps.cpu().double() - xc.grad).norm() / xc.grad.norm().clamp_min(1e-12)
        ).item()
        if rel > (1e-6 if dt == torch.float32 else 5e-3):
            raise AssertionError(f"max autograd {dtype_name}: relative {rel:.2e}")


def register():
    from torch.library import Library

    library = Library("aten", "FRAGMENT")
    for name, function in adaptive_pool_ops.OVERLAYS:
        library.impl(name, function, "MPS")
    return library


def run_wide(dtype, module):
    """The cooperative average against the decomposition, on a wide box.

    Also asserts the shape really takes that path: the two implementations
    agree to float32 rounding, and a fallback would keep the checks green
    while leaving the new kernel untested.
    """
    for shape, out in (
        ((1, 1, 32, 32, 32), (2, 2, 2)),
        ((2, 3, 30, 30, 30), (4, 4, 4)),
        ((1, 2, 25, 25, 25), (3, 3, 3)),
    ):
        in_shape = list(shape[-3:])
        if not (
            module._window_volume(in_shape, list(out)) > module._AVG_NARROW_LIMIT
            and module._fan_in(in_shape, list(out)) <= module._FANIN_LIMIT
        ):
            raise AssertionError(f"{shape}->{out} does not take the cooperative path")
        x = torch.randn(*shape, dtype=dtype, device="mps")
        plan = module._plan(in_shape, list(out), x.device)[0]
        wide = module._fused_avg_wide(x, out)
        general = module._avg_pool3d_general(
            module._working(x), in_shape, list(out), plan
        ).to(dtype)
        diff = (wide.float() - general.float()).abs().max().item() / max(
            general.float().abs().max().item(), 1e-6
        )
        if diff > EXACT[dtype]:
            raise AssertionError(f"wide average {shape}->{out}: {diff:.2e}")
        check(
            f"wide average {shape}->{out}",
            wide,
            torch.ops.aten._adaptive_avg_pool3d(x.cpu(), out),
            dtype,
            limit=None if dtype is torch.bfloat16 else _DEFAULT,
        )
        grad = torch.randn(*wide.shape, dtype=dtype, device="mps")
        check(
            f"wide average backward {shape}->{out}",
            module._adaptive_avg_pool3d_backward(grad, x),
            torch.ops.aten._adaptive_avg_pool3d_backward(grad.cpu(), x.cpu()),
            dtype,
        )


_LIBRARY = None


def best_of(fn, args, batches=15, repeat=4):
    """Fastest batch, not the mean: the GPU is shared with other work here."""
    for _ in range(2):
        fn(*args)
    torch.mps.synchronize()
    samples = []
    for _ in range(batches):
        start = time.perf_counter()
        for _ in range(repeat):
            fn(*args)
        torch.mps.synchronize()
        samples.append((time.perf_counter() - start) / repeat)
    return min(samples) * 1e3


def bench_one(module_fn, cpu_fn, args_mps, args_cpu, floor=None, batches=15, repeat=4):
    """Interleave every candidate so a slow spell hits all measurements.

    ``floor`` is a same-device ``copy_`` of comparable bytes: its minimum is
    the launch-and-bandwidth floor the row is judged against, taken in the
    same batch so a shared GPU moves both numbers together.
    """
    for _ in range(2):
        module_fn(*args_mps)
        cpu_fn(*args_cpu)
        if floor is not None:
            floor()
    torch.mps.synchronize()
    mps_samples, cpu_samples, floor_samples = [], [], []
    for _ in range(batches):
        start = time.perf_counter()
        for _ in range(repeat):
            module_fn(*args_mps)
        torch.mps.synchronize()
        mps_samples.append((time.perf_counter() - start) / repeat)
        start = time.perf_counter()
        for _ in range(repeat):
            cpu_fn(*args_cpu)
        torch.mps.synchronize()
        cpu_samples.append((time.perf_counter() - start) / repeat)
        if floor is not None:
            start = time.perf_counter()
            for _ in range(repeat):
                floor()
            torch.mps.synchronize()
            floor_samples.append((time.perf_counter() - start) / repeat)
    mps, cpu = min(mps_samples) * 1e3, min(cpu_samples) * 1e3
    lowest = min(floor_samples) * 1e3 if floor_samples else None
    return mps, cpu, cpu / mps, lowest


def benchmark():
    rows = []
    shapes = [
        ((1, 1, 16, 16, 16), (8, 8, 8)),
        ((1, 1, 16, 16, 16), (5, 5, 5)),
        ((2, 32, 32, 32, 32), (16, 16, 16)),
        ((2, 32, 32, 32, 32), (7, 7, 7)),
        ((2, 32, 32, 32, 32), (1, 1, 1)),
        ((1, 8, 64, 64, 64), (4, 64, 64)),
        ((1, 64, 64, 64, 64), (1, 1, 1)),
    ]
    for shape, out in shapes:
        x = torch.randn(*shape)
        xm = x.to("mps")
        grad = torch.randn(*torch.ops.aten._adaptive_avg_pool3d(x, out).shape)
        gm = grad.to("mps")
        idx = torch.ops.aten.adaptive_max_pool3d(x, out)[1]
        im = idx.to("mps")
        in_shape = list(shape[-3:])
        row = {
            "input": list(shape),
            "output": list(out),
            "dtype": "float32",
            "avg_path": (
                "cooperative"
                if adaptive_pool_ops._window_volume(in_shape, list(out))
                > adaptive_pool_ops._AVG_NARROW_LIMIT
                else "thread-per-bin"
            ),
            "max_fused": bool(
                adaptive_pool_ops._window_volume(in_shape, list(out))
                <= adaptive_pool_ops._WINDOW_LIMIT
            ),
            "avg_backward_fused": bool(
                adaptive_pool_ops._fan_in(in_shape, list(out))
                <= adaptive_pool_ops._FANIN_LIMIT
            ),
        }
        # Same-shape copy floor, measured inside every batch below.
        floor_src = torch.empty(xm.numel(), device="mps")
        floor_dst = torch.empty(xm.numel(), device="mps")
        floor = lambda: floor_dst.copy_(floor_src)
        for name, fn, cpu_fn, args, cpu_args in (
            (
                "_adaptive_avg_pool3d",
                adaptive_pool_ops._adaptive_avg_pool3d,
                torch.ops.aten._adaptive_avg_pool3d,
                (xm, out),
                (x, out),
            ),
            (
                "adaptive_max_pool3d",
                adaptive_pool_ops._adaptive_max_pool3d,
                torch.ops.aten.adaptive_max_pool3d,
                (xm, out),
                (x, out),
            ),
            (
                "_adaptive_avg_pool3d_backward",
                adaptive_pool_ops._adaptive_avg_pool3d_backward,
                torch.ops.aten._adaptive_avg_pool3d_backward,
                (gm, xm),
                (grad, x),
            ),
            (
                "adaptive_max_pool3d_backward",
                adaptive_pool_ops._adaptive_max_pool3d_backward,
                torch.ops.aten.adaptive_max_pool3d_backward,
                (gm, xm, im),
                (grad, x, idx),
            ),
        ):
            mps_ms, cpu_ms, speedup, floor_ms = bench_one(
                fn, cpu_fn, args, cpu_args, floor=floor
            )
            row[name + "_ms_mps"] = round(mps_ms, 4)
            row[name + "_ms_cpu"] = round(cpu_ms, 4)
            row[name + "_speedup"] = round(speedup, 2)
            row[name + "_copy_floor_ms"] = round(floor_ms, 4)
            row[name + "_over_floor"] = round(mps_ms / floor_ms, 2)
        rows.append(row)
        print(format_bench(row))
    return rows


def kernel_crossing():
    """Fused paths against the decomposition, by box width.

    Every candidate in a row is measured in one interleaved batch against a
    same-shape ``copy_`` floor, because the GPU here is shared and a lone
    measurement can be several times its own cost.
    """
    rows = []
    x = torch.randn(1, 1, 64, 64, 64)
    xm = x.to("mps")
    floor_src = torch.empty(xm.numel(), device="mps")
    floor_dst = torch.empty(xm.numel(), device="mps")
    floor = lambda: floor_dst.copy_(floor_src)
    for k in (2, 4, 8, 16, 32, 64):
        out = (64 // k,) * 3
        in_shape = [64, 64, 64]
        plan, cover, _ = adaptive_pool_ops._plan(in_shape, list(out), xm.device)
        grad = torch.randn(1, 1, *out)
        gm = grad.to("mps")
        box = adaptive_pool_ops._window_volume(in_shape, list(out))
        row = {
            "kernel": k,
            "box_elements": box,
            "input": [1, 1, 64, 64, 64],
            "output": list(out),
            "avg_path": "cooperative" if box > adaptive_pool_ops._AVG_NARROW_LIMIT else "thread-per-bin",
            "avg_slices": adaptive_pool_ops._wide_shape(in_shape, list(out))[2],
            "avg_group": adaptive_pool_ops._wide_shape(in_shape, list(out))[1],
            "avg_lanes": adaptive_pool_ops._wide_shape(in_shape, list(out))[0],
        }
        average = lambda: adaptive_pool_ops._adaptive_avg_pool3d(xm, out)
        narrow = lambda: adaptive_pool_ops._fused_avg(xm, out)
        cooperative = lambda: adaptive_pool_ops._fused_avg_wide(xm, out)
        general = lambda: adaptive_pool_ops._avg_pool3d_general(
            adaptive_pool_ops._working(xm), in_shape, list(out), plan
        )
        backward = lambda: adaptive_pool_ops._adaptive_avg_pool3d_backward(gm, xm)
        backward_general = lambda: adaptive_pool_ops._avg_pool3d_backward_general(
            adaptive_pool_ops._working(gm), in_shape, list(out), plan, cover
        )
        maximum = lambda: adaptive_pool_ops._fused_max(xm, out)
        maximum_general = lambda: adaptive_pool_ops._max_pool3d_general(
            xm, in_shape, list(out), plan
        )
        measured = interleaved(
            (
                ("avg", average),
                ("avg_narrow", narrow),
                ("avg_cooperative", cooperative),
                ("avg_general", general),
                ("avg_backward", backward),
                ("avg_backward_general", backward_general),
                ("max", maximum),
                ("max_general", maximum_general),
                ("floor", floor),
            )
        )
        for key, ms in measured.items():
            row[key + "_ms"] = round(ms, 4)
        row["avg_gb_per_s"] = round(
            xm.numel() * 4 / (measured["avg"] * 1e-3) / 1e9, 1
        )
        row["avg_backward_gb_per_s"] = round(
            (xm.numel() + gm.numel()) * 4 / (measured["avg_backward"] * 1e-3) / 1e9, 1
        )
        rows.append(row)
        print(
            f"k={k:2d} box={box:6d} {row['avg_path']:14s} "
            f"avg {row['avg_ms']:.3f} (narrow {row['avg_narrow_ms']:.3f}, "
            f"cooperative {row['avg_cooperative_ms']:.3f}, "
            f"general {row['avg_general_ms']:.3f}, floor {row['floor_ms']:.3f}) "
            f"avgb {row['avg_backward_ms']:.3f}/{row['avg_backward_general_ms']:.3f} "
            f"max {row['max_ms']:.3f}/{row['max_general_ms']:.3f} ms"
        )
    return rows


def interleaved(candidates, batches=15, repeat=4):
    """Minimum of one interleaved batch per candidate, in milliseconds."""
    for _, fn in candidates:
        fn()
    torch.mps.synchronize()
    samples = {name: [] for name, _ in candidates}
    for _ in range(batches):
        for name, fn in candidates:
            start = time.perf_counter()
            for _ in range(repeat):
                fn()
            torch.mps.synchronize()
            samples[name].append((time.perf_counter() - start) / repeat)
    return {name: min(values) * 1e3 for name, values in samples.items()}


def format_bench(row):
    parts = [f"{row['input']}->{row['output']} [{row['avg_path']}]"]
    for key in ("_adaptive_avg_pool3d", "adaptive_max_pool3d",
                "_adaptive_avg_pool3d_backward", "adaptive_max_pool3d_backward"):
        parts.append(
            f"{key.replace('_adaptive_', '').replace('adaptive_', '')}: "
            f"{row[key + '_ms_mps']:.3f}ms vs {row[key + '_ms_cpu']:.3f}ms "
            f"({row[key + '_speedup']:.1f}x, floor {row[key + '_copy_floor_ms']:.3f}ms)"
        )
    return "\n  ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available")

    global _LIBRARY
    _LIBRARY = register()
    failures = 0
    if not args.skip_verify:
        for dtype in DTYPES:
            for shape, out in CASES:
                try:
                    report = run_case(shape, out, dtype)
                    print(
                        f"PASS {str(dtype):16s} {shape} -> {out} "
                        f"avg {report['avg']:.1e} bwd {report['avg_backward']:.1e} "
                        f"max {report['max']:.1e} mbwd {report['max_backward']:.1e}"
                    )
                except Exception:
                    failures += 1
                    print(f"FAIL {str(dtype):16s} {shape} -> {out}")
                    traceback.print_exc()
        for dtype in DTYPES:
            try:
                run_reference(dtype, adaptive_pool_ops)
                print(f"PASS reference cases {dtype}")
            except Exception:
                failures += 1
                print(f"FAIL reference cases {dtype}")
                traceback.print_exc()
            try:
                run_wide(dtype, adaptive_pool_ops)
                print(f"PASS cooperative average {dtype}")
            except Exception:
                failures += 1
                print(f"FAIL cooperative average {dtype}")
                traceback.print_exc()

    if args.bench:
        rows = benchmark()
        crossing = kernel_crossing()
        measured = ROOT / "measured" / "adaptive_pool_ops.json"
        measured.parent.mkdir(exist_ok=True)
        measured.write_text(
            json.dumps(
                {
                    "torch": torch.__version__,
                    "device": torch.backends.mps.get_name(),
                    "note": (
                        "min of 15 batches x 4 iterations, synchronized, with the "
                        "operator, its CPU reference and a same-shape copy_ floor "
                        "interleaved inside every batch. The GPU is shared with "
                        "other work, so a row can still be inflated by contention; "
                        "treat a ratio below 1 as that, and re-measure the row "
                        "alone before acting on it. *_copy_floor_ms is a "
                        "copy_ of the input's bytes and *_over_floor is the "
                        "operator's time over it: 1.0 means launch-bound at that "
                        "size, well above 1 means real work or a bad layout."
                    ),
                    "avg_narrow_limit": adaptive_pool_ops._AVG_NARROW_LIMIT,
                    "window_limit": adaptive_pool_ops._WINDOW_LIMIT,
                    "fan_in_limit": adaptive_pool_ops._FANIN_LIMIT,
                    "wide_group": adaptive_pool_ops._WIDE_GROUP,
                    "wide_lanes": adaptive_pool_ops._WIDE_LANES,
                    "wide_treat": adaptive_pool_ops._WIDE_TREAT,
                    "wide_slices": adaptive_pool_ops._WIDE_SLICES,
                    "benchmarks": rows,
                    "kernel_crossing": crossing,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"wrote {measured}")

    print(f"{failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
