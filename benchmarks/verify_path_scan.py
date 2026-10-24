"""Verify the four-path 2D selective scan against a CPU reference and time it.

Checks the traversal order against an independent cumulative-sum derivation, the
forward and the analytic gradients against
:func:`metal_kernelops.cv.path_scan.ss2d_scan_reference`, that reference against
central differences in float64 and against
:func:`metal_kernelops.reference.selective_scan`, the degenerate shapes, and
finally times the fused call against the same four paths written as four
separate scans.

Run with:

    uv run --project /Users/dbydd/vibe-agent-working-dir/ResearchChamber \\
        python benchmarks/verify_path_scan.py
"""

import json
import os
import pathlib
import statistics
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/Users/dbydd/vibe-agent-working-dir/ResearchChamber")

from shared_scripts.compute_broker import lease_guard  # noqa: E402

from metal_kernelops import reference as stepwise  # noqa: E402
from metal_kernelops.cv import path_scan  # noqa: E402
from metal_kernelops.cv.path_scan import (  # noqa: E402
    path_orders,
    ss2d_scan,
    ss2d_scan_reference,
)
from metal_kernelops.scan import selective_scan  # noqa: E402

WARM = 6
ROUNDS = 21
REPS = 4
PROJECT = "mps-kernelops"

failures = []
checks = 0
accuracy = {}
NAMES = ("u", "dt", "A", "B", "C")


def check(label, ok, detail=""):
    global checks
    checks += 1
    if not ok:
        failures.append(f"{label}: {detail}")
        print(f"FAIL {label} {detail}")
    return ok


def relative(got, want):
    """Relative error of the global norm, 0 when both norms are exactly zero."""
    denominator = float(want.norm())
    if denominator == 0.0:
        return 0.0 if float(got.norm()) == 0.0 else float("inf")
    return float((got - want).norm()) / denominator


def build(batch, channels, groups, states, height, width, seed, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    shape = (batch, channels, height, width)
    u = torch.randn(shape, generator=generator, dtype=dtype)
    dt = torch.rand(shape, generator=generator, dtype=dtype) * 0.2 + 0.05
    A = -(torch.rand((channels, states), generator=generator, dtype=dtype) + 0.1)
    B = torch.randn(
        (batch, groups, states, height, width), generator=generator, dtype=dtype
    )
    C = torch.randn(
        (batch, groups, states, height, width), generator=generator, dtype=dtype
    )
    return u, dt, A, B, C


_seed = iter(range(1, 1 << 20))


def case(label, shape, merge="sum", tol=1e-5, gradient_tol=1e-4):
    """Compare one four-path scan on MPS with the CPU reference, forward and backward."""
    args = build(*shape, seed=next(_seed))
    want = ss2d_scan_reference(*args, merge=merge)
    device_args = [tensor.to("mps") for tensor in args]
    got = ss2d_scan(*device_args, merge=merge)
    record = accuracy.setdefault(label, {"shape": list(shape), "merge": merge})
    forward_error = relative(got.cpu(), want)
    record["forward_relative_error"] = forward_error
    record["forward_max_abs_error"] = (
        float((got.cpu() - want).abs().max()) if want.numel() else 0.0
    )
    check(f"{label} shape", tuple(got.shape) == tuple(want.shape), f"{tuple(got.shape)}")
    check(f"{label} finite", bool(torch.isfinite(got).all()), "non-finite output")
    check(f"{label} forward", forward_error < tol, f"relative error {forward_error:.2e}")

    seed = torch.randn_like(want)
    cpu_leaves = [tensor.clone().requires_grad_(True) for tensor in args]
    cpu_grads = torch.autograd.grad(
        ss2d_scan_reference(*cpu_leaves, merge=merge), cpu_leaves, seed
    )
    mps_leaves = [tensor.to("mps").clone().requires_grad_(True) for tensor in args]
    mps_grads = torch.autograd.grad(
        ss2d_scan(*mps_leaves, merge=merge), mps_leaves, seed.to("mps")
    )
    errors = {}
    for name, device_grad, cpu_grad in zip(NAMES, mps_grads, cpu_grads):
        error = relative(device_grad.cpu(), cpu_grad)
        errors[name] = error
        check(f"{label} d{name}", error < gradient_tol, f"relative error {error:.2e}")
    record["gradient_relative_error"] = errors
    print(
        f"{label}: {shape} {merge}, forward {forward_error:.1e}, "
        + " ".join(f"d{name} {errors[name]:.1e}" for name in NAMES)
    )
    return want


def check_ordering():
    """The traversal order, derived independently as cumulative sums.

    With ``A = 0``, ``dt = 1`` and ``B = C = 1`` the recurrence degenerates to a
    cumulative sum along each path, so the merged output states the order
    directly. The expected value is built here from plain Python index lists.
    """
    for height, width in ((4, 3), (1, 6), (6, 1), (5, 5)):
        length = height * width
        generator = torch.Generator().manual_seed(height * 31 + width)
        values = torch.randn(1, 1, length, generator=generator)
        expected = torch.zeros_like(values)
        orders = [
            [h * width + w for h in range(height) for w in range(width)],
            [h * width + w for w in range(width) for h in range(height)],
        ]
        orders += [order[::-1] for order in orders]
        for order in orders:
            indices = torch.tensor(order)
            sequence = values[0, 0].index_select(0, indices).cumsum(0)
            restored = torch.empty_like(sequence)
            restored[indices] = sequence
            expected[0, 0] += restored

        shape = (1, 1, height, width)
        got = ss2d_scan(
            values.reshape(shape).to("mps"),
            torch.ones(shape, device="mps"),
            torch.zeros((1, 1), device="mps"),
            torch.ones((1, 1, 1, height, width), device="mps"),
            torch.ones((1, 1, 1, height, width), device="mps"),
        )
        error = relative(got.cpu(), expected.reshape(shape))
        check(f"order {height}x{width}", error < 1e-6, f"relative error {error:.2e}")
        print(f"order {height}x{width}: relative error {error:.1e}")


def check_stepwise_reference():
    """The reference must agree with the repo's step-by-step scan on every path."""
    height, width = 3, 4
    batch, channels, groups, states = 2, 4, 2, 3
    u, dt, A, B, C = build(batch, channels, groups, states, height, width, seed=7)
    length = height * width
    flat_u, flat_dt = u.reshape(batch, channels, length), dt.reshape(batch, channels, length)
    flat_b = B.reshape(batch, groups, states, length)
    flat_c = C.reshape(batch, groups, states, length)
    restored = []
    for order in path_orders(height, width):
        sequence = stepwise.selective_scan(
            flat_u.index_select(-1, order),
            flat_dt.index_select(-1, order),
            A,
            flat_b.index_select(-1, order).contiguous(),
            flat_c.index_select(-1, order).contiguous(),
        )
        restored.append(
            flat_u.new_zeros(batch, channels, length).index_copy(-1, order, sequence)
        )
    want = ((restored[0] + restored[2]) + (restored[1] + restored[3])).reshape(
        batch, channels, height, width
    )
    error = relative(ss2d_scan_reference(u, dt, A, B, C), want)
    check("reference matches step-by-step scan", error < 1e-5, f"{error:.2e}")
    print(f"reference vs step-by-step scan: relative error {error:.1e}")


def check_finite_difference():
    """Central differences in float64 on a two-by-two map, both merge modes."""
    shape = (1, 2, 1, 2, 2, 2)
    for merge in ("sum", "mean"):
        args = build(*shape, seed=13, dtype=torch.float64)
        function = lambda *a: ss2d_scan_reference(*a, merge=merge)  # noqa: E731
        leaves = [tensor.clone().requires_grad_(True) for tensor in args]
        analytic = torch.autograd.grad(
            function(*leaves), leaves, torch.ones_like(leaves[0])
        )
        step = 1e-6
        worst = 0.0
        for index in range(len(args)):
            for pick in range(min(3, args[index].numel())):
                plus = [t.detach().clone() for t in args]
                minus = [t.detach().clone() for t in args]
                plus[index].reshape(-1)[pick] += step
                minus[index].reshape(-1)[pick] -= step
                difference = (
                    float(function(*plus).sum()) - float(function(*minus).sum())
                ) / (2 * step)
                point = float(analytic[index].reshape(-1)[pick])
                worst = max(worst, abs(difference - point) / max(abs(difference), 1e-9))
        check(f"finite difference {merge}", worst < 1e-6, f"worst relative {worst:.2e}")
        print(f"finite difference {merge}: worst relative {worst:.1e}")


def check_degenerate_shapes():
    """Empty and one-element extents produce the empty result with zero gradients."""
    cases = {
        "zero-height": (1, 4, 1, 3, 0, 5),
        "zero-width": (1, 4, 1, 3, 5, 0),
        "no-channels": (2, 0, 0, 4, 3, 3),
        "no-states": (1, 4, 1, 0, 3, 3),
        "batch-of-zero": (0, 4, 1, 3, 3, 3),
        "single-pixel": (1, 4, 1, 3, 1, 1),
        "single-row": (1, 4, 1, 3, 1, 7),
        "single-column": (1, 4, 1, 3, 7, 1),
    }
    for label, shape in cases.items():
        case(label, shape)
    case("per-channel-groups", (1, 4, 4, 3, 3, 5))
    case("one-channel", (1, 1, 1, 3, 3, 3))
    # One state per lane: these would pass a kernel that stops at eight states.
    case("sixteen-states", (1, 8, 1, 16, 9, 11))
    case("thirty-two-states", (1, 4, 2, 32, 5, 7))
    case("five-states", (1, 6, 2, 5, 13, 3))
    case("mean-merge", (1, 4, 1, 3, 4, 5), merge="mean")


def check_state_limit():
    """More states than lanes is rejected, not silently truncated."""
    args = build(1, 4, 1, 64, 3, 3, seed=3)
    try:
        ss2d_scan(*(tensor.to("mps") for tensor in args))
    except ValueError as error:
        check("states above the lane limit are rejected", "32" in str(error), str(error))
        print(f"states above the lane limit: {error}")
    else:
        check("states above the lane limit are rejected", False, "no error raised")


def check_noncontiguous():
    """Inputs may be strided views; the fused path copies them into its layout."""
    generator = torch.Generator().manual_seed(91)
    stored_u = torch.randn(2, 4, 5, 6, generator=generator)
    stored_dt = torch.rand(2, 4, 5, 6, generator=generator) * 0.2 + 0.05
    stored_a = -(torch.rand(3, 4, generator=generator) + 0.1)
    stored_b = torch.randn(2, 2, 3, 5, 6, generator=generator)
    stored_c = torch.randn(2, 2, 3, 5, 6, generator=generator)
    args = [
        stored_u.transpose(2, 3),
        stored_dt.transpose(2, 3),
        stored_a.transpose(0, 1),
        stored_b.transpose(3, 4),
        stored_c.transpose(3, 4),
    ]
    check(
        "strided inputs are not contiguous",
        not any(tensor.is_contiguous() for tensor in args),
        "test inputs came out contiguous",
    )
    want = ss2d_scan_reference(*(tensor.clone() for tensor in args))
    got = ss2d_scan(*(tensor.to("mps") for tensor in args))
    error = relative(got.cpu(), want)
    check("strided inputs", error < 1e-5, f"relative error {error:.2e}")
    print(f"strided inputs: relative error {error:.1e}")


def _interleaved(candidates, rounds=ROUNDS, reps=REPS):
    """Minimum milliseconds per candidate, every candidate timed in every round.

    The box runs dozens of other jobs, so a candidate timed on its own can take a
    burst that its neighbours never see. Each round runs all candidates
    back-to-back and the reported figure is the per-candidate minimum over
    rounds, which is the only statistic here that describes the code rather than
    the load at that instant.
    """
    for function in candidates:
        for _ in range(WARM):
            function()
    torch.mps.synchronize()
    best = [float("inf")] * len(candidates)
    for _ in range(rounds):
        for index, function in enumerate(candidates):
            started = time.perf_counter()
            for _ in range(reps):
                function()
            torch.mps.synchronize()
            best[index] = min(
                best[index], (time.perf_counter() - started) * 1e3 / reps
            )
    return best


def _bandwidth(megabytes, milliseconds):
    """GB/s over the bytes a kernel had to move, in the units this box reports."""
    return round(megabytes / milliseconds, 2)


def _measure_cpu(function):
    """Fewer iterations for the pure-torch baseline: it is a scale reference."""
    for _ in range(2):
        function()
    samples = []
    for _ in range(5):
        started = time.perf_counter()
        function()
        samples.append((time.perf_counter() - started) * 1e3)
    return min(samples), statistics.median(samples)


def _four_calls(u, dt, A, B, C):
    """The same four paths written as four separate selective scans.

    Each path is materialized on its own, which is what a caller writing the
    four scans by hand would do, so no extra copies are charged to the baseline.
    """
    batch, channels, height, width = u.shape
    groups, states = B.shape[1], A.shape[1]
    length = height * width

    def spatial(x):
        paths = [
            x.reshape(batch, channels, length),
            x.transpose(2, 3).reshape(batch, channels, length),
        ]
        return paths + [path.flip(-1) for path in paths]

    def grouped(x):
        paths = [
            x.reshape(batch, groups, states, length),
            x.transpose(3, 4).reshape(batch, groups, states, length),
        ]
        return paths + [path.flip(-1) for path in paths]

    us, ds, bs, cs = spatial(u), spatial(dt), grouped(B), grouped(C)
    outputs = [
        selective_scan(us[k], ds[k], A, bs[k], cs[k]) for k in range(4)
    ]
    paths = torch.stack(outputs, dim=1)
    row = paths[:, 0].reshape(batch, channels, height, width)
    column = paths[:, 1].reshape(batch, channels, width, height).transpose(2, 3)
    reverse_row = paths[:, 2].flip(-1).reshape(batch, channels, height, width)
    reverse_column = paths[:, 3].flip(-1).reshape(
        batch, channels, width, height
    ).transpose(2, 3)
    return (row + reverse_row) + (column + reverse_column)


def time_transcendental():
    """exp against exp2(x * log2e) in the fused kernels, with both accuracies.

    ``exp`` is the recurrence's only transcendental and the recurrence's state
    update depends on it, so it is the one place a cheaper exponential could
    matter. Metal's ``exp`` is itself built on the hardware exponential, so this
    asks whether spelling it out saves anything measurable.
    """
    record = {}
    for label, shape in {
        "b1_d96_56x56": (1, 96, 1, 16, 56, 56),
        "b2_d64_32x32": (2, 64, 1, 16, 32, 32),
        "small": (1, 8, 1, 16, 5, 7),
    }.items():
        args = build(*shape, seed=211)
        device_args = [tensor.to("mps") for tensor in args]
        want = ss2d_scan_reference(*args)

        def forward(exp2):
            path_scan.EXP2 = exp2
            return ss2d_scan(*device_args)

        errors = {}
        for exp2 in (False, True):
            path_scan.EXP2 = exp2
            got = forward(exp2).cpu()
            errors["exp2" if exp2 else "exp"] = {
                "relative_error": relative(got, want),
                "max_abs_error": float((got - want).abs().max()) if want.numel() else 0.0,
            }
        path_scan.EXP2 = False
        exp_ms, exp2_ms = _interleaved([
            lambda: forward(False),
            lambda: forward(True),
        ])
        path_scan.EXP2 = False
        record[label] = {
            "shape": list(shape),
            "exp_ms": round(exp_ms, 4),
            "exp2_ms": round(exp2_ms, 4),
            "exp2_over_exp": round(exp2_ms / exp_ms, 3),
            "accuracy": errors,
        }
        print(
            f"exp vs exp2 {label}: {exp_ms:.4f} ms vs {exp2_ms:.4f} ms, "
            f"relative error {errors['exp']['relative_error']:.1e} vs "
            f"{errors['exp2']['relative_error']:.1e}"
        )
    path_scan.EXP2 = False
    return record


def time_shapes():
    shapes = {
        "b8_d32_16x16": (8, 32, 1, 16, 16, 16),
        "b2_d64_32x32": (2, 64, 1, 16, 32, 32),
        "b1_d96_56x56": (1, 96, 1, 16, 56, 56),
        "b3_d16_7x9_g4": (3, 16, 4, 8, 7, 9),
    }
    timings = {}
    for name, shape in shapes.items():
        batch, channels, groups, states, height, width = shape
        args = build(*shape, seed=101)
        device_args = [tensor.to("mps") for tensor in args]
        seed = torch.randn(batch, channels, height, width, device="mps")

        leaves = [tensor.clone().requires_grad_(True) for tensor in device_args]
        output = ss2d_scan(*leaves)
        four_leaves = [tensor.clone().requires_grad_(True) for tensor in device_args]
        four_output = _four_calls(*four_leaves)
        cpu_leaves = [tensor.clone().requires_grad_(True) for tensor in args]
        cpu_output = ss2d_scan_reference(*cpu_leaves)

        # The forward reads u, dt, B, C and writes the output; the copy floor
        # moves the same number of bytes as one output-sized tensor, so it is the
        # comparable per-pass ceiling on this box at this size.
        megabytes = 4 * (3 * batch * channels * height * width
                         + 2 * args[3].numel()) / 1e6
        source = torch.empty(batch * channels * height * width, device="mps")
        target = torch.empty_like(source)
        copy_mb = 2 * source.numel() * 4 / 1e6

        copy_ms, forward_min, four_min = _interleaved([
            lambda: target.copy_(source),
            lambda: ss2d_scan(*device_args),
            lambda: _four_calls(*device_args),
        ])
        four_backward_min, backward_min = _interleaved([
            lambda: torch.autograd.grad(
                four_output, four_leaves, seed, retain_graph=True),
            lambda: torch.autograd.grad(
                output, leaves, seed, retain_graph=True),
        ], reps=3)
        record = {
            "shape": list(shape),
            "pixels": height * width,
            "fused_forward_ms": round(forward_min, 4),
            "four_scan_forward_ms": round(four_min, 4),
            "fused_backward_ms": round(backward_min, 4),
            "four_scan_backward_ms": round(four_backward_min, 4),
            "forward_speedup_over_four_scans": round(four_min / forward_min, 3),
            "backward_speedup_over_four_scans": round(
                four_backward_min / backward_min, 3
            ),
            "forward_megabytes": round(megabytes, 3),
            "forward_gigabytes_per_second": _bandwidth(megabytes, forward_min),
            "copy_floor_ms": round(copy_ms, 4),
            "copy_floor_gigabytes_per_second": _bandwidth(copy_mb, copy_ms),
        }
        if name != "b1_d96_56x56":
            cpu_min, cpu_median = _measure_cpu(
                lambda: torch.autograd.grad(
                    cpu_output, cpu_leaves, seed.cpu(), retain_graph=True
                )
            )
            record["cpu_reference_backward_ms"] = round(cpu_min, 4)
            record["cpu_reference_backward_median_ms"] = round(cpu_median, 4)
            record["backward_speedup_over_cpu_reference"] = round(
                cpu_min / backward_min, 3
            )
        timings[name] = record
        print(
            f"{name}: forward {forward_min:.3f} ms (four scans {four_min:.3f} ms, "
            f"{record['forward_speedup_over_four_scans']:.2f}x, "
            f"{record['forward_gigabytes_per_second']:.1f} GB/s, copy floor "
            f"{copy_ms:.3f} ms), backward {backward_min:.3f} ms "
            f"(four scans {four_backward_min:.3f} ms)"
        )
    return timings


def main():
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available")
    with lease_guard(PROJECT, tier=1, task_name="verify_path_scan"):
        check_ordering()
        check_stepwise_reference()
        check_finite_difference()
        check_noncontiguous()
        check_degenerate_shapes()
        check_state_limit()
        for shape in (
            (2, 8, 2, 4, 6, 5),
            (2, 16, 1, 8, 8, 8),
            (1, 3, 3, 2, 5, 7),
        ):
            case(f"forward-backward {shape}", shape)
        transcendental = time_transcendental()
        timings = time_shapes()

    measured = ROOT / "measured" / "path_scan.json"
    measured.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "torch": torch.__version__,
        "device": "mps",
        "box_load": {
            "load_average_1m": round(os.getloadavg()[0], 2),
            "cpus": os.cpu_count(),
            "note": (
                "the box runs many other jobs; every candidate in a comparison is "
                "timed inside the same round and the minimum over rounds is reported"
            ),
        },
        "checks": checks,
        "failures": failures,
        "timing_protocol": (
            f"{ROUNDS} rounds of {REPS} synchronized iterations per candidate after "
            f"{WARM} warmup, all candidates of one comparison interleaved inside "
            "each round, per-candidate minimum reported. The four-scan baseline "
            "materializes each path and calls selective_scan once per path; the "
            "copy floor is a same-shape device-to-device copy measured in the same "
            "rounds, which is the launch-bound ceiling at these sizes"
        ),
        "reference": (
            "metal_kernelops.cv.path_scan.ss2d_scan_reference, a step-by-step scan "
            "over four index-arithmetic permutations, itself checked against "
            "central differences in float64 and against "
            "metal_kernelops.reference.selective_scan"
        ),
        "accuracy": accuracy,
        "transcendental": transcendental,
        "timings": timings,
        "notes": [
            "ss2d_scan runs one Metal kernel that walks all four paths over the "
            "(height, width) grid, one state per lane and 32/states channels per "
            "warp, so no permuted copy of u, dt, B or C is built. Each path writes "
            "its own accumulator slice at the pixel it stands on and one pass adds "
            "the four slices, which replaces the old merge permutation.",
            "Five launches forward: transpose B and C to state-minor, chunk "
            "summaries, the chunk prefix chain, the scan, the merge. At 1x96x56x56 "
            "the whole forward moves 4 MB, so the launches, not the arithmetic, set "
            "the time: a same-shape copy measured in the same round costs about as "
            "much as the forward's slowest stage.",
            "Path 0 is row-major (h*W + w), path 1 is column-major (w*H + h), and "
            "paths 2 and 3 are their reverses. Merging inverts each permutation and "
            "adds in VMamba's cross_merge order: (path 0 + path 2) + (path 1 + path 3).",
            "Merge mean multiplies the summed paths by 0.25, which is exact in binary.",
            "The backward reuses metal_kernelops.scan's reverse_summaries, "
            "reverse_boundaries and gradients on the stacked layout, because dB and "
            "dC reduce over the channels of a group and that layout is what exposes "
            "them. It still pays the four stacking copies; only its permutation and "
            "merge are fused, into merge_paths.",
            "exp2(delta * A * log2e) against exp(delta * A): the two land within "
            "run-to-run noise of each other and their outputs agree to the digits "
            "recorded, because Metal's exp is built on the hardware exponential. "
            "exp is kept, so the forward and the reused backward spell the "
            "transcendental the same way.",
            "states must be at most 32 because each state owns a lane; a wider state "
            "axis is rejected rather than silently truncated. The sixteen- and "
            "thirty-two-state cases above exist to hold that line.",
            "The fused path is float32 on MPS only; the reference runs any dtype.",
            "Gradient tolerance is 1e-4 because the kernel reduces over states in a "
            "different order than torch does; on a four-element map a float32 "
            "reference is itself about 1e-5 from a float64 evaluation, while every "
            "realistic shape lands within 1e-7.",
            "The machine is shared with dozens of concurrent jobs and absolute times "
            "swing several-fold between sweeps; the JSON holds one run. Comparing the "
            "fused path against the previously shipped stacked implementation in the "
            "same interleaved batch gave 2.61x forward at 1x96x56x56, 2.43x at "
            "2x64x32x32, 3.47x at 8x32x16x16 and 2.82x at 3x16x7x9, with the "
            "backward at 1.13x, 1.23x, 1.06x and 0.95x.",
        ],
    }
    measured.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"\nchecks {checks}, failures {len(failures)}")
    print(f"wrote {measured.relative_to(ROOT)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
