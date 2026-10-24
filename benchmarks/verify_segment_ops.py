"""Verify the MPS ``segment_reduce`` overlays against the CPU kernel.

Registers the overlays on the MPS dispatch key for this process only, then
compares forward outputs, gradients and error classes against ``aten``'s CPU
kernel on the same inputs, and finally times both paths.

Run with:

    uv run --project /Users/dbydd/vibe-agent-working-dir/ResearchChamber \\
        python benchmarks/verify_segment_ops.py
"""

import json
import pathlib
import random
import statistics
import sys
import time

import torch
from torch.library import Library

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/Users/dbydd/vibe-agent-working-dir/ResearchChamber")

from shared_scripts.compute_broker import lease_guard  # noqa: E402

from metal_kernelops import segment_ops  # noqa: E402

WARM = 10
TIMED = 60
PROJECT = "mps-kernelops"

failures = []
checks = 0
accuracy = {}


def check(label, ok, detail=""):
    global checks
    checks += 1
    if not ok:
        failures.append(f"{label}: {detail}")
        print(f"FAIL {label} {detail}")
    return ok


def close_reference(got, want, tol):
    if not torch.equal(got.isnan(), want.isnan()):
        return False
    both = ~(got.isnan() | want.isnan())
    if not both.any():
        return True
    return torch.allclose(got[both], want[both], rtol=tol, atol=tol, equal_nan=False)


def _max_diff(got, want):
    """Largest absolute deviation, with a nan in both treated as agreement."""
    device_got, device_want = got.cpu(), want.cpu()
    if not device_got.numel():
        return 0.0
    difference = (device_got - device_want).abs()
    shared_nan = device_got.isnan() & device_want.isnan()
    return float(difference.masked_fill(shared_nan, 0.0).max().item())


def case(label, data, reduce, lengths=None, offsets=None, axis=0, initial=None, unsafe=False, tol=0.0):
    """Compare one invocation on MPS against the CPU kernel, forward and backward."""
    kwargs = dict(lengths=lengths, offsets=offsets, axis=axis, unsafe=unsafe)
    if initial is not None:
        kwargs["initial"] = initial
    want = torch.segment_reduce(data, reduce, **kwargs)
    device_kwargs = {
        key: value.to("mps") if isinstance(value, torch.Tensor) else value
        for key, value in kwargs.items()
    }
    got = torch.segment_reduce(data.to("mps"), reduce, **device_kwargs)
    record = accuracy.setdefault(
        f"{reduce}/{str(data.dtype).split('.')[-1]}",
        {"cases": 0, "forward_exact": 0, "gradient_exact": 0,
         "forward_max_abs_error": 0.0, "gradient_max_abs_error": 0.0},
    )
    record["cases"] += 1
    forward_diff = _max_diff(got, want)
    record["forward_max_abs_error"] = max(record["forward_max_abs_error"], forward_diff)
    record["forward_exact"] += int(close_reference(got.cpu(), want, 0.0))
    check(f"{label} forward shape", got.shape == want.shape, f"{got.shape} vs {want.shape}")
    check(f"{label} forward values", close_reference(got.cpu(), want, tol),
          f"max |diff| {forward_diff:.3e}")

    leaf = data.clone().requires_grad_(True)
    out = torch.segment_reduce(leaf, reduce, **kwargs)
    grad_seed = torch.arange(out.numel(), dtype=out.dtype).reshape(out.shape)
    (want_grad,) = torch.autograd.grad(out, leaf, grad_seed)

    mps_leaf = data.to("mps").requires_grad_(True)
    mps_out = torch.segment_reduce(mps_leaf, reduce, **device_kwargs)
    (got_grad,) = torch.autograd.grad(mps_out, mps_leaf, grad_seed.to("mps"))
    gradient_diff = _max_diff(got_grad, want_grad)
    record["gradient_max_abs_error"] = max(record["gradient_max_abs_error"], gradient_diff)
    record["gradient_exact"] += int(close_reference(got_grad.cpu(), want_grad, 0.0))
    check(f"{label} gradient", close_reference(got_grad.cpu(), want_grad, tol),
          f"max |diff| {gradient_diff:.3e}")
    reference = want_grad.norm().item()
    if reference:
        relative = (got_grad.cpu() - want_grad).norm().item() / reference
        check(f"{label} gradient norm rel err within tolerance", relative < max(tol, 1e-6),
              f"{relative:.3e}")
    return want


def errors_match(label, cpu_call, mps_call):
    try:
        want = cpu_call()
    except Exception as exc:  # noqa: BLE001 - the CPU kernel's own error is the reference
        try:
            mps_call()
        except Exception as got:  # noqa: BLE001
            check(f"{label} error class", type(got).__name__ == type(exc).__name__,
                  f"{type(got).__name__} vs CPU {type(exc).__name__}: {got}")
            return
        check(f"{label} error class", False, "MPS accepted what CPU rejects")
        return
    got = mps_call()
    check(f"{label} runs on both", close_reference(got.cpu(), want, 0.0),
          f"max |diff| {(got.cpu() - want).abs().max().item():.3e}")


def length_table(rows_shape, axis_len, segments, empty=False):
    """A per-row composition of ``axis_len`` into ``segments`` parts, optionally with an empty segment."""
    table = torch.zeros(rows_shape + (segments,), dtype=torch.int64)
    flat = table.reshape(-1, segments)
    for index in range(flat.shape[0]):
        cuts = sorted(random.choices(range(axis_len + 1), k=segments - 1)) if segments > 1 else []
        edges = [0] + cuts + [axis_len]
        row = [edges[i + 1] - edges[i] for i in range(segments)]
        if empty:
            row[0] = 0
            row[-1] = axis_len - sum(row[1:-1])
        flat[index] = torch.tensor(row)
    return table


def offsets_of(lengths, trim=0):
    bounds = torch.cat([torch.zeros(lengths.shape[:-1] + (1,), dtype=torch.int64), lengths], -1)
    bounds = bounds.cumsum(-1)
    if trim:
        bounds[..., 0] += trim
        bounds[..., -1] -= trim
    return bounds.clamp(0, lengths.sum(-1).max().item()).sort(-1).values


def time_pair(mps_call, cpu_call, warm=WARM, iters=TIMED):
    """Wall time in milliseconds for both paths, every iteration synchronized.

    Returns ``(mps_min, mps_median, cpu_min, cpu_median)``. The machine is
    shared, so the minimum is the number that describes the operation and the
    median is reported alongside it to show how contended the run was.
    """
    for _ in range(warm):
        mps_call()
        cpu_call()
    torch.mps.synchronize()
    mps_times, cpu_times = [], []
    for _ in range(iters):
        start = time.perf_counter()
        mps_call()
        torch.mps.synchronize()
        mps_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        cpu_call()
        torch.mps.synchronize()
        cpu_times.append(time.perf_counter() - start)
    return (
        min(mps_times) * 1e3,
        statistics.median(mps_times) * 1e3,
        min(cpu_times) * 1e3,
        statistics.median(cpu_times) * 1e3,
    )


def per_launch(call, iters=TIMED):
    """Milliseconds per launch with one synchronization per batch.

    A synchronized iteration pays one host round trip on top of the work, and on
    this machine that round trip is most of a small kernel's measured time. This
    amortizes it over the whole batch, so the number describes the GPU-side cost
    of one launch.
    """
    for _ in range(WARM):
        call()
    torch.mps.synchronize()
    best = float("inf")
    for _ in range(4):
        start = time.perf_counter()
        for _ in range(iters):
            call()
        torch.mps.synchronize()
        best = min(best, (time.perf_counter() - start) / iters)
    return best * 1e3


def min_synced(call, iters=TIMED):
    """Milliseconds for the fastest synchronized call, with nothing interleaved.

    The interleaved-with-CPU-inputs protocol above moves a whole tensor across
    the bus between every pair of samples, which disturbs the queue; this is the
    number to read when asking what the kernel costs in ordinary use.
    """
    for _ in range(WARM):
        call()
    torch.mps.synchronize()
    best = float("inf")
    for _ in range(iters):
        start = time.perf_counter()
        call()
        torch.mps.synchronize()
        best = min(best, time.perf_counter() - start)
    return best * 1e3


def main():
    if torch.__version__.split("+")[0] != segment_ops.CHECKED_TORCH:
        raise SystemExit(f"overlays checked against torch {segment_ops.CHECKED_TORCH}, running {torch.__version__}")

    library = Library("aten", "FRAGMENT")
    for name, function in segment_ops.OVERLAYS:
        library.impl(name, function, "MPS")
    print(f"registered {[name for name, _ in segment_ops.OVERLAYS]} on MPS, torch {torch.__version__}")

    with lease_guard(PROJECT, tier=1, task_name="verify_segment_ops"):
        random.seed(0)
        torch.manual_seed(0)

        data = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
        for reduce in ("sum", "mean", "prod", "max", "min", "amax", "amin"):
            case(f"{reduce} lengths axis0", data, reduce, lengths=torch.tensor([2, 1, 0]))
            case(f"{reduce} offsets axis0", data, reduce, offsets=torch.tensor([0, 2, 3, 3]))
            case(f"{reduce} initial", data, reduce, lengths=torch.tensor([2, 1, 0]), initial=7)
        case("sum axis1 per-row", data, "sum", lengths=torch.tensor([[2, 2], [1, 3], [4, 0]]), axis=1)
        case("max axis1 per-row", data, "max", lengths=torch.tensor([[2, 2], [1, 3], [4, 0]]), axis=1)
        case("mean axis1 empty", data, "mean", lengths=torch.tensor([[2, 2], [0, 4], [3, 1]]), axis=1)
        case("prod axis1 zero", torch.tensor([[1.0, 3.0, 0.0, 2.0]]), "prod",
             lengths=torch.tensor([[2, 2]]), axis=1)
        case("prod two zeros", torch.tensor([0.0, 0.0, 5.0, 1.0, 2.0, 3.0]), "prod",
             lengths=torch.tensor([3, 3]))
        case("max tie split", torch.tensor([1.0, 3.0, 3.0, 0.0]), "max", lengths=torch.tensor([4]))
        case("min tie split", torch.tensor([1.0, 3.0, 3.0, 0.0]), "min", lengths=torch.tensor([4]))
        # ties in half precision, kernel path: the split itself is exact
        tie = torch.randn(128, 96).half()
        tie[4, 7] = tie[5, 7] = torch.tensor(-3.0, dtype=torch.float16)
        case("fp16 max tie split kernel", tie, "max", lengths=torch.full((32,), 4, dtype=torch.int64))
        case("fp16 min tie split kernel", tie, "min", lengths=torch.full((32,), 4, dtype=torch.int64))
        case("offsets prefix ignored", data, "sum", offsets=torch.tensor([1, 3, 3]))
        case("1-D data", torch.tensor([1.0, 2.0, 3.0, 4.0]), "sum", lengths=torch.tensor([2, 2]))
        case("3-D inner dims", torch.arange(24.0).reshape(2, 3, 4), "prod",
             lengths=torch.tensor([[[2, 2], [2, 2], [2, 2]], [[2, 2], [2, 2], [2, 2]]]), axis=2)
        case("3-D axis1", torch.arange(24.0).reshape(2, 3, 4), "sum",
             lengths=torch.tensor([[1, 2], [2, 1]]), axis=1)
        case("negative axis", data, "sum", lengths=torch.tensor([[2, 2], [2, 2], [2, 2]]), axis=-1)
        case("non-contiguous", torch.arange(8.0).reshape(2, 4).t(), "sum", lengths=torch.tensor([2, 2]))
        case("fp16 sum", data.half(), "sum", lengths=torch.tensor([2, 1, 0]), tol=1e-3)
        case("fp16 mean", data.half(), "mean", lengths=torch.tensor([2, 0, 1]), tol=1e-3)
        case("bf16 prod", data.bfloat16(), "prod", lengths=torch.tensor([2, 1, 0]), tol=1e-2)

        # Explicit kernel-path parity, per-row (inner == 1) and wide (inner > 32).
        # The kernel reduces lanes in parallel, so fp32 sums and products differ
        # from the CPU kernel in association only: 1e-6 absolute does not hold
        # for a 16-element product, the same class of difference the sweep below
        # already allows for fp32.
        for mode in ("sum", "mean", "prod", "max", "min"):
            case(f"{mode} per-row kernel 16x64",
                 torch.randn(16, 64), mode,
                 lengths=torch.full((16, 4), 16, dtype=torch.int64), axis=1, tol=1e-5)
            case(f"{mode} wide kernel 16x32x64",
                 torch.randn(16, 32, 64), mode,
                 lengths=torch.full((16, 4), 8, dtype=torch.int64), axis=1, tol=1e-5)

        # randomized differential sweep
        for trial in range(80):
            axis = random.choice([0, 1])
            shape = [random.choice([1, 2, 3, 4, 8]) for _ in range(axis + 1)]
            shape += [random.choice([1, 2, 3, 8])] * random.choice([0, 1])
            segments = random.choice([1, 2, 3, 5])
            dtype = random.choice([torch.float32] * 6 + [torch.float16, torch.bfloat16])
            values = torch.randn(*shape, dtype=dtype)
            lengths = length_table(tuple(shape[:axis]), shape[axis], segments,
                                   empty=random.random() < 0.35)
            reduce = random.choice(["sum", "mean", "prod", "max", "min"])
            initial = random.choice([None, None, 3.0])
            # fp32 sums accumulate in a different order on the two devices; the
            # exactness counters in the JSON record how often they agree to the bit.
            tolerance = {torch.float32: 1e-6, torch.float16: 1e-2, torch.bfloat16: 6e-2}[dtype]
            case(f"random {trial} {reduce} {dtype} axis{axis}", values, reduce,
                 lengths=lengths, axis=axis, initial=initial, tol=tolerance)
            case(f"random {trial} {reduce} {dtype} axis{axis} offsets", values, reduce,
                 offsets=offsets_of(lengths, trim=random.choice([0, 0, 1])), axis=axis,
                 initial=initial, tol=tolerance)

        # error parity
        device_data = data.to("mps")
        errors_match("axis not last dim",
                     lambda: torch.segment_reduce(data, "sum", lengths=torch.tensor([4]), axis=1),
                     lambda: torch.segment_reduce(device_data, "sum",
                                                  lengths=torch.tensor([4], device="mps"), axis=1))
        errors_match("negative lengths",
                     lambda: torch.segment_reduce(data, "sum", lengths=torch.tensor([3, -1])),
                     lambda: torch.segment_reduce(device_data, "sum",
                                                  lengths=torch.tensor([3, -1], device="mps")))
        errors_match("unsorted offsets",
                     lambda: torch.segment_reduce(data, "sum", offsets=torch.tensor([0, 2, 1])),
                     lambda: torch.segment_reduce(device_data, "sum",
                                                  offsets=torch.tensor([0, 2, 1], device="mps")))
        errors_match("rows do not sum to axis",
                     lambda: torch.segment_reduce(data, "sum", lengths=torch.tensor([1, 1])),
                     lambda: torch.segment_reduce(device_data, "sum",
                                                  lengths=torch.tensor([1, 1], device="mps")))
        errors_match("unsafe skips the sum check",
                     lambda: torch.segment_reduce(data, "sum", lengths=torch.tensor([1, 1]), unsafe=True),
                     lambda: torch.segment_reduce(device_data, "sum",
                                                  lengths=torch.tensor([1, 1], device="mps"), unsafe=True))
        errors_match("indices unsupported",
                     lambda: torch.segment_reduce(data, "sum", indices=torch.tensor([0, 0, 1, 1])),
                     lambda: torch.segment_reduce(device_data, "sum",
                                                  indices=torch.tensor([0, 0, 1, 1], device="mps")))
        errors_match("unknown reduce",
                     lambda: torch.segment_reduce(data, "bogus", lengths=torch.tensor([3])),
                     lambda: torch.segment_reduce(device_data, "bogus",
                                                  lengths=torch.tensor([3], device="mps")))

        check("dispatch path: kernel was exercised", segment_ops._PATHS["kernel"] > 0)
        check("dispatch path: aten fallback was exercised", segment_ops._PATHS["aten"] > 0)

        # out= variants resolve through the composite and must reach the overlay
        out_cpu = torch.empty(2, 4)
        torch.ops.aten.segment_reduce.out(data[:2], "sum", lengths=torch.tensor([1, 1]), out=out_cpu)
        out_mps = torch.empty(2, 4, device="mps")
        torch.ops.aten.segment_reduce.out(device_data[:2], "sum",
                                          lengths=torch.tensor([1, 1], device="mps"), out=out_mps)
        check("segment_reduce.out", torch.equal(out_cpu, out_mps.cpu()))
        bwd_cpu = torch.empty(2, 4)
        torch.ops.aten._segment_reduce_backward.out(torch.ones(2, 4), out_cpu, data[:2], "sum",
                                                    lengths=torch.tensor([1, 1]), out=bwd_cpu)
        bwd_mps = torch.empty(2, 4, device="mps")
        torch.ops.aten._segment_reduce_backward.out(torch.ones(2, 4, device="mps"), out_mps,
                                                    device_data[:2], "sum",
                                                    lengths=torch.tensor([1, 1], device="mps"),
                                                    out=bwd_mps)
        check("_segment_reduce_backward.out", torch.equal(bwd_cpu, bwd_mps.cpu()))

        timings = {}
        torch.mps.empty_cache()
        shapes = {
            "embedding_bag_4096x768": (torch.randn(4096, 768), "sum",
                                       torch.full((512,), 8, dtype=torch.int64), 0),
            "flat_1M": (torch.randn(1_000_000), "sum", torch.full((1000,), 1000, dtype=torch.int64), 0),
            "per_row_1024x4096": (torch.randn(1024, 4096), "sum",
                                  torch.full((1024, 16), 256, dtype=torch.int64), 1),
            "fp16_mean_8192x256": (torch.randn(8192, 256).half(), "mean",
                                   torch.full((1024,), 8, dtype=torch.int64), 0),
        }
        for name, (values, reduce, lengths, axis) in shapes.items():
            device_values = values.to("mps")
            device_lengths = lengths.to("mps")
            kwargs = dict(lengths=lengths, axis=axis)
            device_kwargs = dict(lengths=device_lengths, axis=axis)

            def mps_forward():
                return torch.segment_reduce(device_values, reduce, **device_kwargs)

            def cpu_roundtrip():
                return torch.segment_reduce(device_values.cpu(), reduce, **kwargs).to("mps")

            def mps_unsafe():
                return torch.segment_reduce(device_values, reduce, **device_kwargs, unsafe=True)

            forward_mps, forward_mps_median, forward_cpu, forward_cpu_median = time_pair(
                mps_forward, cpu_roundtrip
            )
            unsafe_mps, _, _, _ = time_pair(mps_unsafe, cpu_roundtrip)

            mps_leaf = device_values.clone().requires_grad_(True)
            mps_out = torch.segment_reduce(mps_leaf, reduce, **device_kwargs)
            seed = torch.randn_like(mps_out)
            cpu_leaf = values.clone().requires_grad_(True)
            cpu_out = torch.segment_reduce(cpu_leaf, reduce, **kwargs)

            def mps_backward():
                return torch.autograd.grad(mps_out, mps_leaf, seed, retain_graph=True)

            def cpu_backward():
                (grad,) = torch.autograd.grad(cpu_out, cpu_leaf, seed.cpu(), retain_graph=True)
                return grad.to("mps")

            backward_mps, backward_mps_median, backward_cpu, backward_cpu_median = time_pair(
                mps_backward, cpu_backward
            )
            # same-batch copy floor for physical bandwidth reference
            copy_buf = torch.empty_like(device_values)
            copy_times = []
            for _ in range(TIMED):
                t0 = time.perf_counter()
                copy_buf.copy_(device_values)
                torch.mps.synchronize()
                copy_times.append(time.perf_counter() - t0)
            copy_floor_ms = min(copy_times) * 1e3
            mb = values.numel() * values.element_size() / 1e6
            copy_gb_s = round(2 * mb / copy_floor_ms, 1) if copy_floor_ms else 0.0
            forward_alone = min_synced(mps_forward)
            backward_alone = min_synced(mps_backward)
            fwd_gb_s = round(mb / forward_alone, 1) if forward_alone else 0.0

            # Where the fused path's GPU time goes: the boundary table plus its
            # validation, then the reduction itself, each without a round trip.
            rows, inner = segment_ops._rows_inner(values.shape, axis)
            segments = int(lengths.shape[-1])
            stage_shape = list(values.shape)
            stage_shape[axis] = segments
            stage = {}
            if segment_ops._kernel_ready(device_values, rows, segments,
                                         values.shape[axis], inner):
                stage_bounds = torch.empty((rows, segments + 1), dtype=torch.int64,
                                           device="mps")
                stage_flag = torch.zeros(2, dtype=torch.int32, device="mps")
                stage_spec = device_lengths.reshape(rows, segments)
                stage_out = torch.empty(stage_shape, dtype=values.dtype, device="mps")
                stage_library = segment_ops._library(values.dtype)

                def scan_bounds():
                    stage_library.scan_bounds(
                        stage_spec, stage_bounds, stage_flag, rows, segments,
                        values.shape[axis], 0, 1, 0, 1, threads=rows)

                def reduce_kernel():
                    segment_ops._forward_kernel(
                        device_values, stage_bounds, stage_out, rows, segments,
                        values.shape[axis], inner, segment_ops._MODES[reduce], None)

                stage = {
                    "scan_bounds_ms": round(per_launch(scan_bounds), 4),
                    "reduce_kernel_ms": round(per_launch(reduce_kernel), 4),
                    "copy_floor_ms": round(per_launch(lambda: copy_buf.copy_(device_values)), 4),
                    "lanes": segment_ops._lanes(inner),
                    "threadgroups": rows * segments,
                }
            timings[name] = {
                "elements": values.numel(),
                "shape": list(values.shape),
                "dtype": str(values.dtype).split(".")[-1],
                "reduce": reduce,
                "axis": axis,
                "segments": int(lengths.shape[-1]),
                "forward_mps_ms": round(forward_mps, 4),
                "forward_mps_median_ms": round(forward_mps_median, 4),
                "forward_mps_unsafe_ms": round(unsafe_mps, 4),
                "forward_cpu_to_mps_ms": round(forward_cpu, 4),
                "forward_cpu_to_mps_median_ms": round(forward_cpu_median, 4),
                "forward_speedup": round(forward_cpu / forward_mps, 3),
                "forward_achieved_gb_s": fwd_gb_s,
                "forward_mps_isolated_ms": round(forward_alone, 4),
                "backward_mps_isolated_ms": round(backward_alone, 4),
                "copy_floor_ms": round(copy_floor_ms, 4),
                "copy_floor_gb_s": copy_gb_s,
                "stages": stage,
                "backward_mps_ms": round(backward_mps, 4),
                "backward_mps_median_ms": round(backward_mps_median, 4),
                "backward_cpu_to_mps_ms": round(backward_cpu, 4),
                "backward_cpu_to_mps_median_ms": round(backward_cpu_median, 4),
                "backward_speedup": round(backward_cpu / backward_mps, 3),
            }
            print(f"{name}: forward {forward_mps:.3f} ms vs {forward_cpu:.3f} ms "
                  f"({forward_cpu / forward_mps:.2f}x), backward {backward_mps:.3f} ms vs "
                  f"{backward_cpu:.3f} ms ({backward_cpu / backward_mps:.2f}x)")

    measured = ROOT / "measured" / "segment_ops.json"
    measured.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "torch": torch.__version__,
        "device": "mps",
        "overlays": [name for name, _ in segment_ops.OVERLAYS],
        "checks": checks,
        "failures": failures,
        "timing_protocol": (
            f"min and median of {TIMED} synchronized iterations after {WARM} warmup, "
            "MPS overlay interleaved with a CPU round trip on the same inputs; "
            "speedup uses the minimum because the machine is shared with other work"
        ),
        "reference": "aten::segment_reduce and its autograd on the CPU kernel, same inputs",
        "paths_exercised": dict(segment_ops._PATHS),
        "notes": [
            "The machine is shared with other benchmark runs; the minimum describes the "
            "operation, the median shows how contended that run was.",
            "Contiguous-lengths descriptors take the fused Metal kernel; non-contiguous "
            "input, non-exact prefixes/suffixes and small shapes take the aten composition.",
            "scan_bounds builds the boundary table and validates it on device; strict "
            "validation uses one deferred host read overlapped with the reduction kernel.",
            "per_row_1024x4096 runs one SIMD group per segment along the axis; wide cases "
            "assign lanes to inner columns in whole SIMD groups up to 512.",
            "backward is bounds-driven with edge filling, avoiding the position map entirely.",
            "fp32 sum/mean/prod accumulate in a different order on the two devices; "
            "max/min are bit-exact.",
            "fp16/bf16 sums accumulate in float32 in the fused kernel then narrow once, "
            "matching the CPU kernel within tolerance.",
            "unsafe=True skips only the lengths-sum check; the negative-length and "
            "unsorted-offset checks and their one host read still run.",
            "index_reduce_ on MPS is reached only on the aten fallback path.",
            "stages.* are per-launch GPU costs amortizing one synchronization over "
            f"{TIMED} submissions, so they exclude the round trip the fused call pays once.",
        ],
        "accuracy": accuracy,
        "timings": timings,
    }
    measured.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"\nchecks {checks}, failures {len(failures)}")
    print(f"wrote {measured.relative_to(ROOT)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
