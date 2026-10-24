"""Verify the MPS Lanczos-3 antialiased resample overlays against the CPU kernel.

Run with the project's torch 2.14.1 environment:

    uv run --project /Users/dbydd/vibe-agent-working-dir/ResearchChamber \
        python benchmarks/verify_lanczos_ops.py

The script registers ``lanczos_ops.OVERLAYS`` on a temporary ``aten`` fragment,
checks that both operators are missing on MPS before registration and correct
afterwards (forward, backward, autograd, layouts, dtype guard, determinism),
then measures CPU and MPS throughput with warm, synchronized timings. It also
repeats the correctness checks on the threadgroup-tile forward and the separable
backward, measures where the tile overtakes the two-pass form (the value of
``_INTERMEDIATE_LIMIT``), and reports the plan caches, the achieved bandwidth and
the machine's measured copy bandwidth as the floor. Results are written to
``measured/lanczos_ops.json``.

The CPU kernel is the reference: same filter, same windows, same scale
semantics. Agreement is reported as max absolute error, max error relative to
the reference's max magnitude, and gradient-norm relative error.
"""

import argparse
import json
import os
import platform
import sys
import time

import torch
from torch.library import Library

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from metal_kernelops import lanczos_ops  # noqa: E402

_BROKER_PATH = os.environ.get(
    "COMPUTE_BROKER_PATH",
    "/Users/dbydd/vibe-agent-working-dir/ResearchChamber/shared_scripts",
)
if _BROKER_PATH not in sys.path:
    sys.path.insert(0, _BROKER_PATH)
try:
    from compute_broker import lease_guard  # noqa: E402
except ImportError:  # the broker lives outside this repository
    lease_guard = None


class _NoLease:
    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return None

    def __exit__(self, *_exc):
        return False


def _guard(task):
    return lease_guard("mps-kernelops", tier=1, task_name=task) if lease_guard else _NoLease()


FORWARD = "aten::_upsample_lanczos2d_aa"
BACKWARD = "aten::_upsample_lanczos2d_aa_backward"

CASES = [
    # (name, input shape, output size, align_corners, scales)
    ("up_2x", (2, 3, 8, 8), (16, 16), False, None),
    ("up_2x_align", (2, 3, 8, 8), (16, 16), True, None),
    ("down_2x", (1, 2, 16, 16), (7, 9), False, None),
    ("down_2x_scales", (1, 2, 16, 16), (7, 9), False, (2.0, 2.0)),
    ("down_2x_align", (1, 1, 16, 16), (7, 9), True, None),
    ("up_3x", (1, 1, 6, 6), (18, 18), False, None),
    ("up_nonuniform", (1, 1, 7, 11), (3, 4), True, None),
    ("equal", (1, 1, 9, 9), (9, 9), False, None),
    ("width_only", (1, 1, 9, 13), (9, 9), False, None),
    ("height_only", (1, 1, 13, 9), (9, 9), False, None),
    ("single_pixel", (1, 3, 1, 1), (5, 7), False, None),
    ("single_pixel_align", (1, 3, 1, 1), (5, 7), True, None),
    ("rect_down", (2, 3, 32, 32), (11, 13), False, None),
    ("rect_up", (1, 4, 40, 24), (80, 9), False, None),
]

TIMING_CASES = [
    ("upsample_2x", (1, 3, 256, 256), (512, 512)),
    ("downsample_2x", (1, 3, 1024, 1024), (512, 512)),
    ("feature_map_2x", (8, 64, 64, 64), (128, 128)),
    ("upsample_4x", (2, 3, 128, 128), (512, 512)),
]


def _compare(got, ref):
    diff = (got.double() - ref.double()).abs()
    denom = ref.double().abs().max().clamp(min=1e-30)
    return {
        "max_abs": diff.max().item(),
        "max_rel": (diff.max() / denom).item(),
        "bit_exact": torch.equal(got, ref),
        "finite": bool(torch.isfinite(got).all()),
    }


def _call(device, shape, out_shape, align_corners, scales, tensor=None):
    x = torch.randn(*shape) if tensor is None else tensor
    out = torch.ops.aten._upsample_lanczos2d_aa
    if scales is None:
        return out(x.to(device), list(out_shape), align_corners)
    return out(x.to(device), list(out_shape), align_corners, scales[0], scales[1])


def _seconds(fn, device, warmup, iters):
    """Best per-call wall time: the machine is shared, so contention only adds."""
    for _ in range(warmup):
        fn()
    if device == "mps":
        torch.mps.synchronize()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        if device == "mps":
            torch.mps.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return min(samples)


def _throughput(fn, warmup, iters):
    """Best steady-state MPS cost: ``iters`` queued calls behind one synchronize."""
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(3):
        torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.mps.synchronize()
        best = min(best, (time.perf_counter() - start) * 1e3 / iters)
    return best


def _paired(fns, warmup, iters, burst=4):
    """Best per-call time of several kernels alternated in one loop, so drift or
    contention on the shared machine hits all of them alike. Each sample queues
    ``burst`` calls behind one synchronize, which keeps the launch floor from
    swamping a sub-0.2 ms kernel. Returns one minimum per function."""
    for _ in range(warmup):
        for fn in fns:
            fn()
    torch.mps.synchronize()
    samples = [[] for _ in fns]
    for _ in range(iters):
        for bucket, fn in zip(samples, fns):
            torch.mps.synchronize()
            start = time.perf_counter()
            for _ in range(burst):
                fn()
            torch.mps.synchronize()
            bucket.append((time.perf_counter() - start) * 1e3 / burst)
    return [min(bucket) for bucket in samples]


def _forward_kernel(stage, x, out, wp, hp, shape, out_shape):
    """The forward kernels of ``stage`` with every buffer preallocated, so the
    measurement isolates the memory traffic from allocator churn."""
    n, c, ih, iw = shape
    oh, ow = out_shape
    if stage == "two_pass":
        temp = torch.empty((n, c, ih, ow), dtype=x.dtype, device=x.device)

        def run():
            lanczos_ops._launch_width(x, temp, wp, ih, ow, n, c)
            lanczos_ops._launch_height(temp, out, hp, oh, ow, n, c)

        return run
    if stage == "tile":
        th = lanczos_ops._tile_rows(hp[4], hp[5])
        return lambda: lanczos_ops._launch_tile(x, out, wp, hp, th, oh, ow, n, c)
    if stage == "width":
        return lambda: lanczos_ops._launch_width(x, out, wp, ih, ow, n, c)
    if stage == "height":
        return lambda: lanczos_ops._launch_height(x, out, hp, oh, ow, n, c)
    wx, xmin, xsize, taps_w, _, _ = wp
    wy, ymin, ysize, taps_h, _, _ = hp
    return lambda: lanczos_ops._library().resample_forward(
        x, out, wx, wy, xmin, ymin, xsize, ysize,
        c, n * c, oh, ow, taps_w, taps_h, *x.stride(), *out.stride(),
        threads=n * c * oh * ow, group_size=256)


def _backward_kernel(stage, grad_out, grad_in, shape, out_shape):
    n, c, ih, iw = shape
    oh, ow = out_shape
    wxt, xidx, btaps_w = lanczos_ops._transposed_plan(iw, ow, False, None)
    wyt, yidx, btaps_h = lanczos_ops._transposed_plan(ih, oh, False, None)
    if stage == "separable":
        rows_first = ih * ow <= oh * iw
        temp = torch.empty(
            (n, c) + ((ih, ow) if rows_first else (oh, iw)),
            dtype=grad_out.dtype, device=grad_out.device,
        )
        if rows_first:
            def run():
                lanczos_ops._launch_gather_rows(
                    grad_out, temp, (wyt, yidx, btaps_h), ih, ow, n, c)
                lanczos_ops._launch_gather_cols(
                    temp, grad_in, (wxt, xidx, btaps_w), ih, iw, n, c)
        else:
            def run():
                lanczos_ops._launch_gather_cols(
                    grad_out, temp, (wxt, xidx, btaps_w), oh, iw, n, c)
                lanczos_ops._launch_gather_rows(
                    temp, grad_in, (wyt, yidx, btaps_h), ih, iw, n, c)
        return run
    return lambda: lanczos_ops._library().resample_backward(
        grad_out, grad_in, wxt, wyt, xidx, yidx, c, n * c, ih, iw, btaps_w, btaps_h,
        *grad_out.stride(), *grad_in.stride(),
        threads=n * c * ih * iw, group_size=256)


def _traffic(stage, src_bytes, dst_bytes, scratch_bytes):
    """DRAM bytes one call must move through, by stage.

    ``two_pass`` and ``separable`` write and re-read their scratch buffer; the
    tile keeps it in threadgroup memory, and the fused forms never build one.
    """
    if stage in ("two_pass", "separable"):
        return src_bytes + 2 * scratch_bytes + dst_bytes
    return src_bytes + dst_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(ROOT, "measured", "lanczos_ops.json"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available on this machine")
    version = torch.__version__.split("+")[0]
    report = {
        "torch": torch.__version__,
        "platform": platform.platform(),
        "date": time.strftime("%Y-%m-%d"),
        "dtype": "float32",
        "timing_note": (
            "shared machine: timings run under a tier-1 compute lease while other "
            "jobs use the GPU, so a single reading swings by several times. "
            "Reported values are minima over iters repeats (contention only adds "
            "time): mps_latency is wall time with a synchronize per call, "
            "mps_steady is iters queued calls behind one synchronize (which pays "
            "allocator churn for a live output per call), kernel_steady is the "
            "same kernels with every buffer reused. The crossover and headline "
            "sections alternate two_pass, tile and a copy_ floor inside one loop "
            "and report per-batch minima, so the floor in the same row is the "
            "memory system's capability under the load that run saw."
        ),
        "load": {
            "loadavg": [round(value, 2) for value in os.getloadavg()],
            "cpu_count": os.cpu_count(),
        },
        "missing_before_register": {},
        "forward": [],
        "backward": [],
        "autograd": [],
        "timing_ms": [],
    }

    # the point of the module: these operators have no MPS kernel upstream
    for name in (FORWARD, BACKWARD):
        report["missing_before_register"][name] = not (
            torch._C._dispatch_has_kernel_for_dispatch_key(name, "MPS")
        )
    if not all(report["missing_before_register"].values()):
        raise SystemExit("expected both operators to be missing on MPS")

    library = Library("aten", "FRAGMENT")
    for name, kernel in lanczos_ops.OVERLAYS:
        library.impl(name, kernel, "MPS")

    torch.manual_seed(0)
    with _guard("lanczos_verify"):
        for name, shape, out_shape, align_corners, scales in CASES:
            x = torch.randn(*shape)
            ref = _call("cpu", shape, out_shape, align_corners, scales, x)
            got = _call("mps", shape, out_shape, align_corners, scales, x).cpu()
            row = {
                "case": name,
                "input": list(shape),
                "output": list(out_shape),
                "align_corners": align_corners,
                "scales": list(scales) if scales else None,
            }
            row.update(_compare(got, ref))
            report["forward"].append(row)

            grad = torch.randn(shape[0], shape[1], *out_shape)
            ref_grad = torch.ops.aten._upsample_lanczos2d_aa_backward(
                grad, list(out_shape), list(shape), align_corners,
                *(scales or (None, None)),
            )
            got_grad = torch.ops.aten._upsample_lanczos2d_aa_backward(
                grad.to("mps"), list(out_shape), list(shape), align_corners,
                *(scales or (None, None)),
            ).cpu()
            row = dict(row)
            row.update(_compare(got_grad, ref_grad))
            report["backward"].append(row)

        # end-to-end autograd: forward values, gradients, gradient-norm error
        for name, shape, out_shape, align_corners in [
            ("autograd_up", (2, 3, 8, 8), (16, 16), False),
            ("autograd_down", (2, 3, 16, 16), (7, 7), False),
            ("autograd_equal", (1, 2, 8, 8), (8, 8), False),
        ]:
            torch.manual_seed(7)
            x = torch.randn(*shape)
            weight = torch.randn(shape[0], shape[1], *out_shape)
            cpu_x = x.clone().requires_grad_(True)
            mps_x = x.clone().to("mps").requires_grad_(True)
            cpu_y = torch.ops.aten._upsample_lanczos2d_aa(cpu_x, list(out_shape), align_corners)
            mps_y = torch.ops.aten._upsample_lanczos2d_aa(mps_x, list(out_shape), align_corners)
            (cpu_y * weight).sum().backward()
            (mps_y * weight.to("mps")).sum().backward()
            value = _compare(mps_y.detach().cpu(), cpu_y.detach())
            grad = _compare(mps_x.grad.cpu(), cpu_x.grad)
            norm_cpu = cpu_x.grad.cpu().double().norm().item()
            norm_mps = mps_x.grad.cpu().double().norm().item()
            report["autograd"].append({
                "case": name,
                "input": list(shape),
                "output": list(out_shape),
                "value_max_abs": value["max_abs"],
                "value_max_rel": value["max_rel"],
                "grad_max_abs": grad["max_abs"],
                "grad_max_rel": grad["max_rel"],
                "grad_norm_cpu": norm_cpu,
                "grad_norm_mps": norm_mps,
                "grad_norm_rel": abs(norm_mps - norm_cpu) / norm_cpu,
            })

        # layouts and shapes the kernel indexes by stride
        layout_checks = {}
        x = torch.randn(2, 3, 8, 8)
        x_cl = x.to(memory_format=torch.channels_last)
        ref = torch.ops.aten._upsample_lanczos2d_aa(x_cl, [16, 16], False)
        got = torch.ops.aten._upsample_lanczos2d_aa(x_cl.to("mps"), [16, 16], False).cpu()
        layout_checks["channels_last_forward"] = _compare(got, ref) | {
            "out_channels_last": got.is_contiguous(memory_format=torch.channels_last),
        }
        x_nc = torch.randn(2, 8, 3, 8).transpose(1, 2)
        ref = torch.ops.aten._upsample_lanczos2d_aa(x_nc, [16, 16], False)
        got = torch.ops.aten._upsample_lanczos2d_aa(x_nc.to("mps"), [16, 16], False).cpu()
        layout_checks["non_contiguous_forward"] = _compare(got, ref)
        g_cl = torch.randn(2, 3, 16, 16).to(memory_format=torch.channels_last)
        ref = torch.ops.aten._upsample_lanczos2d_aa_backward(g_cl, [16, 16], [2, 3, 8, 8], False)
        got = torch.ops.aten._upsample_lanczos2d_aa_backward(
            g_cl.to("mps"), [16, 16], [2, 3, 8, 8], False).cpu()
        layout_checks["channels_last_backward"] = _compare(got, ref)
        empty = torch.ops.aten._upsample_lanczos2d_aa(
            torch.randn(0, 3, 5, 5, device="mps"), [10, 10], False)
        layout_checks["empty_batch"] = {"shape": list(empty.shape)}
        report["layouts"] = layout_checks

        # determinism: the kernels use no atomics, so two runs must agree bitwise
        x = torch.randn(2, 3, 64, 64, device="mps")
        first = torch.ops.aten._upsample_lanczos2d_aa(x, [96, 96], False)
        second = torch.ops.aten._upsample_lanczos2d_aa(x, [96, 96], False)
        report["deterministic"] = bool(torch.equal(first, second))

        try:
            torch.ops.aten._upsample_lanczos2d_aa(x.half(), [96, 96], False)
            report["half_guard"] = "accepted"
        except NotImplementedError:
            report["half_guard"] = "NotImplementedError"

        # the public entry point: F.interpolate(mode="lanczos", antialias=True)
        api = []
        for kwargs in ({"size": (96, 96)}, {"scale_factor": 2.0},
                       {"scale_factor": (1.5, 2.0)}, {"scale_factor": 0.5}):
            x = torch.randn(2, 3, 64, 64)
            ref = torch.nn.functional.interpolate(
                x, mode="lanczos", antialias=True, **kwargs)
            got = torch.nn.functional.interpolate(
                x.to("mps"), mode="lanczos", antialias=True, **kwargs).cpu()
            row = {"kwargs": {key: list(value) if isinstance(value, tuple) else value
                              for key, value in kwargs.items()},
                   "output": list(ref.shape)}
            row.update(_compare(got, ref))
            api.append(row)
        x_cpu = torch.randn(2, 3, 64, 64, requires_grad=True)
        x_mps = x_cpu.detach().clone().to("mps").requires_grad_(True)
        torch.nn.functional.interpolate(
            x_cpu, scale_factor=2.0, mode="lanczos", antialias=True).sum().backward()
        torch.nn.functional.interpolate(
            x_mps, scale_factor=2.0, mode="lanczos", antialias=True).sum().backward()
        report["interpolate"] = {
            "cases": api,
            "grad_max_abs": (x_mps.grad.cpu().double() - x_cpu.grad.double()).abs().max().item(),
        }

        # the fused fallback used when neither a scratch buffer nor a tile fits
        limit = lanczos_ops._INTERMEDIATE_LIMIT
        cap = lanczos_ops._TILE_ROW_CAP
        backward_limit = lanczos_ops._BACKWARD_INTERMEDIATE_LIMIT
        lanczos_ops._INTERMEDIATE_LIMIT = 0
        lanczos_ops._TILE_ROW_CAP = 0
        lanczos_ops._BACKWARD_INTERMEDIATE_LIMIT = 0
        try:
            x = torch.randn(2, 3, 24, 40)
            ref = torch.ops.aten._upsample_lanczos2d_aa(x, [60, 17], False)
            got = torch.ops.aten._upsample_lanczos2d_aa(x.to("mps"), [60, 17], False).cpu()
            grad = torch.randn(2, 3, 60, 17)
            ref_grad = torch.ops.aten._upsample_lanczos2d_aa_backward(
                grad, [60, 17], [2, 3, 24, 40], False)
            got_grad = torch.ops.aten._upsample_lanczos2d_aa_backward(
                grad.to("mps"), [60, 17], [2, 3, 24, 40], False).cpu()
            report["fused_fallback"] = _compare(got, ref) | {
                "grad_" + key: value for key, value in _compare(got_grad, ref_grad).items()
            }
        finally:
            lanczos_ops._INTERMEDIATE_LIMIT = limit
            lanczos_ops._TILE_ROW_CAP = cap
            lanczos_ops._BACKWARD_INTERMEDIATE_LIMIT = backward_limit

        # Output width 1 with a height change: the CPU forward collapses every
        # row onto row 0 (see the module docstring), so check the MPS forward
        # against the adjoint of the CPU backward, which is per-row correct.
        x = torch.randn(1, 1, 6, 1)
        out_shape = [22, 1]
        mps_forward = torch.ops.aten._upsample_lanczos2d_aa(
            x.to("mps"), out_shape, False).cpu()
        adjoint = torch.zeros(out_shape)
        for flat in range(adjoint.numel()):
            seed = torch.zeros(1, 1, *out_shape)
            seed.view(-1)[flat] = 1.0
            grad = torch.ops.aten._upsample_lanczos2d_aa_backward(
                seed, out_shape, list(x.shape), False)
            adjoint.view(-1)[flat] = (grad * x).sum()
        cpu_forward = torch.ops.aten._upsample_lanczos2d_aa(x, out_shape, False)
        report["width_one"] = {
            "mps_vs_adjoint_max_abs": (mps_forward.double() - adjoint.double()).abs().max().item(),
            "cpu_vs_adjoint_max_abs": (cpu_forward.double() - adjoint.double()).abs().max().item(),
            "cpu_rows_constant": bool(torch.allclose(
                cpu_forward.flatten(), cpu_forward.flatten()[0].expand(cpu_forward.numel()))),
        }

        # the threadgroup tile and the separable backward, on the geometries the
        # default thresholds send them: forced here so the sweep above cannot
        # silently leave the tile untested
        limit = lanczos_ops._INTERMEDIATE_LIMIT
        lanczos_ops._INTERMEDIATE_LIMIT = 0
        tile_cases, tile_stages = [], {}
        try:
            for name, shape, out_shape, align_corners in [
                ("tile_up_2x", (2, 3, 8, 8), (16, 16), False),
                ("tile_down_2x", (1, 2, 16, 16), (7, 9), False),
                ("tile_rect_up", (1, 4, 40, 24), (80, 9), False),
                ("tile_up_3x", (1, 1, 6, 6), (18, 18), False),
                ("tile_align", (2, 3, 8, 8), (16, 16), True),
                ("tile_single_pixel", (1, 3, 1, 1), (5, 7), False),
                ("tile_wide_block", (2, 3, 33, 70), (17, 130), False),
                ("tile_thin", (2, 1, 12, 300), (37, 7), False),
            ]:
                n, c, ih, iw = shape
                height_plan = lanczos_ops._axis_plan(
                    ih, out_shape[0], align_corners, None, ih == out_shape[0])
                stage = lanczos_ops._forward_stage(
                    n, c, ih, iw, out_shape[0], out_shape[1], height_plan, 4)
                tile_stages[name] = {
                    "forward": stage,
                    "forward_rows": lanczos_ops._tile_rows(
                        height_plan[4], height_plan[5]),
                    "backward": lanczos_ops._backward_stage(
                        n * c, ih, iw, out_shape[0], out_shape[1], 4),
                }
                x = torch.randn(*shape)
                ref = torch.ops.aten._upsample_lanczos2d_aa(x, list(out_shape), align_corners)
                got = torch.ops.aten._upsample_lanczos2d_aa(
                    x.to("mps"), list(out_shape), align_corners).cpu()
                row = {
                    "case": name, "input": list(shape), "output": list(out_shape),
                    "align_corners": align_corners,
                }
                row.update(_compare(got, ref))
                grad = torch.randn(shape[0], shape[1], *out_shape)
                ref_grad = torch.ops.aten._upsample_lanczos2d_aa_backward(
                    grad, list(out_shape), list(shape), align_corners)
                got_grad = torch.ops.aten._upsample_lanczos2d_aa_backward(
                    grad.to("mps"), list(out_shape), list(shape), align_corners).cpu()
                row.update({
                    "grad_" + key: value
                    for key, value in _compare(got_grad, ref_grad).items()
                })
                tile_cases.append(row)

            # the tile indexes the source and the destination by stride
            x_cl = torch.randn(2, 3, 40, 24).to(memory_format=torch.channels_last)
            ref = torch.ops.aten._upsample_lanczos2d_aa(x_cl, [80, 48], False)
            got = torch.ops.aten._upsample_lanczos2d_aa(x_cl.to("mps"), [80, 48], False).cpu()
            tile_layout = _compare(got, ref) | {
                "out_channels_last": got.is_contiguous(memory_format=torch.channels_last),
            }
            x_nc = torch.randn(2, 24, 3, 40).transpose(1, 2)
            ref = torch.ops.aten._upsample_lanczos2d_aa(x_nc, [80, 48], False)
            got = torch.ops.aten._upsample_lanczos2d_aa(x_nc.to("mps"), [80, 48], False).cpu()
            tile_layout_non_contiguous = _compare(got, ref)
            g_cl = torch.randn(2, 3, 80, 48).to(memory_format=torch.channels_last)
            ref = torch.ops.aten._upsample_lanczos2d_aa_backward(
                g_cl, [80, 48], [2, 3, 40, 24], False)
            got = torch.ops.aten._upsample_lanczos2d_aa_backward(
                g_cl.to("mps"), [80, 48], [2, 3, 40, 24], False).cpu()
            tile_layout_backward = _compare(got, ref)

            # no atomics, so two tile launches agree bitwise
            x = torch.randn(2, 3, 64, 64, device="mps")
            first = torch.ops.aten._upsample_lanczos2d_aa(x, [96, 96], False)
            second = torch.ops.aten._upsample_lanczos2d_aa(x, [96, 96], False)
            tile_deterministic = bool(torch.equal(first, second))

            # width 1 with a height change, against the CPU backward's adjoint
            x = torch.randn(1, 1, 6, 1)
            out_shape = [22, 1]
            mps_forward = torch.ops.aten._upsample_lanczos2d_aa(
                x.to("mps"), out_shape, False).cpu()
            adjoint = torch.zeros(out_shape)
            for flat in range(adjoint.numel()):
                seed = torch.zeros(1, 1, *out_shape)
                seed.view(-1)[flat] = 1.0
                grad = torch.ops.aten._upsample_lanczos2d_aa_backward(
                    seed, out_shape, list(x.shape), False)
                adjoint.view(-1)[flat] = (grad * x).sum()
            tile_adjoint = (mps_forward.double() - adjoint.double()).abs().max().item()
        finally:
            lanczos_ops._INTERMEDIATE_LIMIT = limit
        report["tile_path"] = {
            "cases": tile_cases,
            "stages": tile_stages,
            "channels_last": tile_layout,
            "non_contiguous": tile_layout_non_contiguous,
            "channels_last_backward": tile_layout_backward,
            "deterministic": tile_deterministic,
            "width_one_mps_vs_adjoint": tile_adjoint,
        }

        # the memory system's own floor: one large copy, same best-of-N protocol
        copy_src = torch.empty(8 * 64 * 128 * 128, device="mps")
        copy_dst = torch.empty_like(copy_src)
        copy_ms = _seconds(lambda: copy_dst.copy_(copy_src), "mps", args.warmup, args.iters)
        copy_bytes = copy_src.numel() * copy_src.element_size() * 2
        copy_gbs = copy_bytes / copy_ms / 1e6
        report["memory_floor"] = {
            "copy_bytes": copy_bytes,
            "copy_ms": round(copy_ms, 4),
            "copy_gbs": round(copy_gbs, 1),
        }
        del copy_src, copy_dst
        torch.mps.empty_cache()

        for name, shape, out_shape in TIMING_CASES:
            n, c, ih, iw = shape
            oh, ow = out_shape
            element = 4
            height_plan = lanczos_ops._axis_plan(ih, oh, False, None, ih == oh)
            fwd_stage = lanczos_ops._forward_stage(
                n, c, ih, iw, oh, ow, height_plan, element)
            bwd_stage = lanczos_ops._backward_stage(n * c, ih, iw, oh, ow, element)
            fwd_bytes = _traffic(
                fwd_stage, n * c * ih * iw * element, n * c * oh * ow * element,
                n * c * ih * ow * element)
            bwd_bytes = _traffic(
                bwd_stage, n * c * oh * ow * element, n * c * ih * iw * element,
                n * c * min(ih * ow, oh * iw) * element)
            x_cpu = torch.randn(*shape)
            x_mps = x_cpu.to("mps")
            g_cpu = torch.randn(shape[0], shape[1], *out_shape)
            g_mps = g_cpu.to("mps")
            fwd_cpu_fn = lambda: torch.ops.aten._upsample_lanczos2d_aa(  # noqa: E731
                x_cpu, list(out_shape), False)
            fwd_mps_fn = lambda: torch.ops.aten._upsample_lanczos2d_aa(  # noqa: E731
                x_mps, list(out_shape), False)
            bwd_cpu_fn = lambda: torch.ops.aten._upsample_lanczos2d_aa_backward(  # noqa: E731
                g_cpu, list(out_shape), list(shape), False)
            bwd_mps_fn = lambda: torch.ops.aten._upsample_lanczos2d_aa_backward(  # noqa: E731
                g_mps, list(out_shape), list(shape), False)
            fwd_cpu = _seconds(fwd_cpu_fn, "cpu", args.warmup, args.iters)
            fwd_mps = _seconds(fwd_mps_fn, "mps", args.warmup, args.iters)
            fwd_steady = _throughput(fwd_mps_fn, args.warmup, args.iters)
            bwd_cpu = _seconds(bwd_cpu_fn, "cpu", args.warmup, args.iters)
            bwd_mps = _seconds(bwd_mps_fn, "mps", args.warmup, args.iters)
            bwd_steady = _throughput(bwd_mps_fn, args.warmup, args.iters)
            out_k = torch.empty(n, c, oh, ow, device="mps")
            gin_k = torch.empty(n, c, ih, iw, device="mps")
            wp = lanczos_ops._axis_plan(iw, ow, False, None, iw == ow)
            fwd_kernel = _forward_kernel(
                fwd_stage, x_mps, out_k, wp, height_plan, shape, out_shape)
            bwd_kernel = _backward_kernel(bwd_stage, g_mps, gin_k, shape, out_shape)
            fwd_kernel_steady = _throughput(fwd_kernel, args.warmup, args.iters)
            bwd_kernel_steady = _throughput(bwd_kernel, args.warmup, args.iters)
            del out_k, gin_k
            check = _compare(
                torch.ops.aten._upsample_lanczos2d_aa(x_mps, list(out_shape), False).cpu(),
                torch.ops.aten._upsample_lanczos2d_aa(x_cpu, list(out_shape), False))
            report["timing_ms"].append({
                "case": name,
                "input": list(shape),
                "output": list(out_shape),
                "forward_stage": fwd_stage,
                "backward_stage": bwd_stage,
                "forward": {
                    "cpu": round(fwd_cpu, 3),
                    "mps_latency": round(fwd_mps, 3),
                    "latency_speedup": round(fwd_cpu / fwd_mps, 2),
                    "mps_steady": round(fwd_steady, 3),
                    "steady_speedup": round(fwd_cpu / fwd_steady, 2),
                    "bytes": fwd_bytes,
                    "bandwidth_gbs": round(fwd_bytes / fwd_steady / 1e6, 1),
                    "kernel_steady": round(fwd_kernel_steady, 3),
                    "kernel_bandwidth_gbs": round(fwd_bytes / fwd_kernel_steady / 1e6, 1),
                    "copy_floor_ms": round(fwd_bytes / copy_gbs / 1e6, 4),
                },
                "backward": {
                    "cpu": round(bwd_cpu, 3),
                    "mps_latency": round(bwd_mps, 3),
                    "latency_speedup": round(bwd_cpu / bwd_mps, 2),
                    "mps_steady": round(bwd_steady, 3),
                    "steady_speedup": round(bwd_cpu / bwd_steady, 2),
                    "bytes": bwd_bytes,
                    "bandwidth_gbs": round(bwd_bytes / bwd_steady / 1e6, 1),
                    "kernel_steady": round(bwd_kernel_steady, 3),
                    "kernel_bandwidth_gbs": round(bwd_bytes / bwd_kernel_steady / 1e6, 1),
                    "copy_floor_ms": round(bwd_bytes / copy_gbs / 1e6, 4),
                },
                "value_max_rel": check["max_rel"],
            })
            del x_cpu, x_mps, g_cpu, g_mps
            torch.mps.empty_cache()

        # where the tile overtakes the two-pass form: one geometry, a growing
        # scratch buffer, both kernels and a copy of the same footprint measured
        # interleaved in one loop so the shared machine hits them alike
        copy_src = torch.empty(8 * 64 * 128 * 128, device="mps")
        copy_dst = torch.empty_like(copy_src)
        floor_bytes = copy_src.numel() * copy_src.element_size() * 2

        def copy_floor():
            copy_dst.copy_(copy_src)

        crossover = []
        for channels in (2, 3, 4, 8, 12, 16, 24, 32):
            shape, out_shape = (1, channels, 256, 256), (512, 512)
            x = torch.randn(*shape, device="mps")
            out = torch.empty(1, channels, *out_shape, device="mps")
            scratch = torch.empty(1, channels, 256, 512, device="mps")
            wp = lanczos_ops._axis_plan(256, 512, False, None, False)
            hp = lanczos_ops._axis_plan(256, 512, False, None, False)
            th = lanczos_ops._tile_rows(hp[4], hp[5])

            def two_pass(x=x, out=out, wp=wp, hp=hp, channels=channels):
                lanczos_ops._launch_width(x, scratch, wp, 256, 512, 1, channels)
                lanczos_ops._launch_height(scratch, out, hp, 512, 512, 1, channels)

            def tile(x=x, out=out, wp=wp, hp=hp, th=th, channels=channels):
                lanczos_ops._launch_tile(x, out, wp, hp, th, 512, 512, 1, channels)

            two_ms, tile_ms, floor_ms = _paired(
                [two_pass, tile, copy_floor], args.warmup, args.iters)
            two_bytes = (channels * 256 * 256 * 4 + 2 * channels * 256 * 512 * 4
                         + channels * 512 * 512 * 4)
            tile_bytes = channels * 256 * 256 * 4 + channels * 512 * 512 * 4
            crossover.append({
                "channels": channels,
                "scratch_bytes": channels * 256 * 512 * 4,
                "two_pass_ms": round(two_ms, 4),
                "tile_ms": round(tile_ms, 4),
                "speedup": round(two_ms / tile_ms, 2),
                "two_pass_gbs": round(two_bytes / two_ms / 1e6, 1),
                "tile_gbs": round(tile_bytes / tile_ms / 1e6, 1),
                "floor_ms": round(floor_ms, 4),
                "floor_gbs": round(floor_bytes / floor_ms / 1e6, 1),
            })
            del x, out, scratch
            torch.mps.empty_cache()
        del copy_src, copy_dst

        # The other direction: on a downsampled height a tile spans ``scale``
        # intermediate rows per output row, so the two-pass form keeps the small
        # buffers. This is where that stops being true.
        downscale = []
        for name, shape, out_shape in [
            ("down2_512", (1, 3, 512, 512), (256, 256)),
            ("down2_1024", (1, 3, 1024, 1024), (512, 512)),
            ("down2_2048", (1, 3, 2048, 2048), (1024, 1024)),
            ("down4_1024", (1, 3, 1024, 1024), (256, 256)),
        ]:
            n, c, ih, iw = shape
            oh, ow = out_shape
            x = torch.randn(*shape, device="mps")
            out = torch.empty(n, c, oh, ow, device="mps")
            scratch = torch.empty(n, c, ih, ow, device="mps")
            wp = lanczos_ops._axis_plan(iw, ow, False, None, iw == ow)
            hp = lanczos_ops._axis_plan(ih, oh, False, None, ih == oh)
            th = lanczos_ops._tile_rows(hp[4], hp[5])
            copy_src = torch.empty(n * c * oh * ow, device="mps")
            copy_dst = torch.empty_like(copy_src)
            floor_bytes = copy_src.numel() * copy_src.element_size() * 2

            def two_pass(x=x, out=out, wp=wp, hp=hp):
                lanczos_ops._launch_width(x, scratch, wp, ih, ow, n, c)
                lanczos_ops._launch_height(scratch, out, hp, oh, ow, n, c)

            def tile(x=x, out=out, wp=wp, hp=hp, th=th):
                lanczos_ops._launch_tile(x, out, wp, hp, th, oh, ow, n, c)

            def floor(x=copy_src, out=copy_dst):
                out.copy_(x)

            two_ms, tile_ms, floor_ms = _paired(
                [two_pass, tile, floor], args.warmup, args.iters)
            tile_bytes = n * c * (ih * iw + oh * ow) * 4
            two_bytes = tile_bytes + 2 * n * c * ih * ow * 4
            downscale.append({
                "case": name,
                "input": list(shape),
                "output": list(out_shape),
                "tile_rows": th,
                "scratch_bytes": n * c * ih * ow * 4,
                "two_pass_ms": round(two_ms, 4),
                "tile_ms": round(tile_ms, 4),
                "speedup": round(two_ms / tile_ms, 2),
                "two_pass_gbs": round(two_bytes / two_ms / 1e6, 1),
                "tile_gbs": round(tile_bytes / tile_ms / 1e6, 1),
                "floor_gbs": round(floor_bytes / floor_ms / 1e6, 1),
            })
            del x, out, scratch, copy_src, copy_dst
            torch.mps.empty_cache()

        limit = lanczos_ops._INTERMEDIATE_LIMIT
        downsampled_over = [row for row in downscale if row["scratch_bytes"] > limit]
        downsampled_under = [row for row in downscale if row["scratch_bytes"] <= limit]
        report["crossover"] = {
            "limit_bytes": limit,
            # the rule is right when an upsampled height always prefers the tile
            # and a downsampled one crosses over at the limit
            "upsampling_tile_min_speedup": round(
                min((row["speedup"] for row in crossover), default=0.0), 2),
            "downscale_tile_min_speedup_above": round(
                min((row["speedup"] for row in downsampled_over), default=0.0), 2),
            "downscale_two_pass_max_speedup": round(
                max((row["speedup"] for row in downsampled_under), default=0.0), 2),
            "downscale_crossover_bytes": max(
                (row["scratch_bytes"] for row in downscale if row["speedup"] <= 1.0),
                default=0),
            # counts, not a verdict: one contended sample flips a boolean
            "upsampling_tile_wins": sum(1 for row in crossover if row["speedup"] >= 1.0),
            "upsampling_cases": len(crossover),
            "downscale_tile_wins_above": sum(
                1 for row in downsampled_over if row["speedup"] >= 1.0),
            "downscale_cases_above": len(downsampled_over),
            "cases": crossover,
            "downscale": downscale,
        }

        # headline shape: the two-pass form the module shipped with against the
        # tile it takes now, interleaved with a copy floor in the same loop
        headline_shape, headline_out = (8, 64, 64, 64), (128, 128)
        n, c, ih, iw = headline_shape
        oh, ow = headline_out
        x = torch.randn(*headline_shape, device="mps")
        out = torch.empty(n, c, oh, ow, device="mps")
        scratch = torch.empty(n, c, ih, ow, device="mps")
        wp = lanczos_ops._axis_plan(iw, ow, False, None, False)
        hp = lanczos_ops._axis_plan(ih, oh, False, None, False)
        th = lanczos_ops._tile_rows(hp[4], hp[5])
        copy_src = torch.empty(n * c * oh * ow, device="mps")
        copy_dst = torch.empty_like(copy_src)
        headline_floor_bytes = copy_src.numel() * copy_src.element_size() * 2

        def headline_two_pass():
            lanczos_ops._launch_width(x, scratch, wp, ih, ow, n, c)
            lanczos_ops._launch_height(scratch, out, hp, oh, ow, n, c)

        def headline_tile():
            lanczos_ops._launch_tile(x, out, wp, hp, th, oh, ow, n, c)

        def headline_floor():
            copy_dst.copy_(copy_src)

        two_ms, tile_ms, floor_ms = _paired(
            [headline_two_pass, headline_tile, headline_floor],
            args.warmup, max(args.iters, 20))
        tile_bytes = n * c * ih * iw * 4 + n * c * oh * ow * 4
        two_bytes = tile_bytes + 2 * n * c * ih * ow * 4
        report["headline"] = {
            "input": list(headline_shape),
            "output": list(headline_out),
            "tile_rows": th,
            "two_pass_ms": round(two_ms, 4),
            "tile_ms": round(tile_ms, 4),
            "speedup": round(two_ms / tile_ms, 2),
            "two_pass_gbs": round(two_bytes / two_ms / 1e6, 1),
            "tile_gbs": round(tile_bytes / tile_ms / 1e6, 1),
            "floor_ms": round(floor_ms, 4),
            "floor_gbs": round(headline_floor_bytes / floor_ms / 1e6, 1),
            "tile_floor_ratio": round(tile_ms / (tile_bytes / (headline_floor_bytes / floor_ms)), 2),
        }
        del x, out, scratch, copy_src, copy_dst
        torch.mps.empty_cache()

        # plan construction against the steady state
        axis_info = lanczos_ops._axis_plan.cache_info()
        transposed_info = lanczos_ops._transposed_plan.cache_info()

        def info(entry):
            return {
                "hits": entry.hits,
                "misses": entry.misses,
                "maxsize": entry.maxsize,
                "currsize": entry.currsize,
                "hit_rate": round(entry.hits / max(entry.hits + entry.misses, 1), 3),
            }

        x = torch.randn(8, 64, 64, 64, device="mps")
        resized = lambda: torch.ops.aten._upsample_lanczos2d_aa(x, [128, 128], False)  # noqa: E731
        warm_ms = _seconds(resized, "mps", args.warmup, args.iters)
        lanczos_ops._axis_plan.cache_clear()
        lanczos_ops._transposed_plan.cache_clear()
        cold_ms = _seconds(resized, "mps", 0, 1)
        start = time.perf_counter()
        for _ in range(20000):
            lanczos_ops._axis_plan(64, 128, False, None, False)
        lookup_us = (time.perf_counter() - start) * 1e6 / 20000
        build = []
        for _ in range(5):
            lanczos_ops._axis_plan.cache_clear()
            start = time.perf_counter()
            lanczos_ops._axis_plan(64, 128, False, None, False)
            torch.mps.synchronize()
            build.append((time.perf_counter() - start) * 1e3)
        report["plans"] = {
            "axis_plan": info(axis_info),
            "transposed_plan": info(transposed_info),
            "axis_plan_build_ms": round(min(build), 4),
            "axis_plan_lookup_us": round(lookup_us, 2),
            "feature_map_cold_ms": round(cold_ms, 4),
            "feature_map_warm_ms": round(warm_ms, 4),
            "feature_map_cold_penalty_ms": round(max(cold_ms - warm_ms, 0.0), 4),
        }
        del x
        torch.mps.empty_cache()

    report["summary"] = {
        "forward_max_rel": max(row["max_rel"] for row in report["forward"]),
        "forward_bit_exact_cases": sum(row["bit_exact"] for row in report["forward"]),
        "backward_max_rel": max(row["max_rel"] for row in report["backward"]),
        "backward_bit_exact_cases": sum(row["bit_exact"] for row in report["backward"]),
        "grad_norm_rel_max": max(row["grad_norm_rel"] for row in report["autograd"]),
        "interpolate_max_rel": max(row["max_rel"] for row in report["interpolate"]["cases"]),
        "fused_fallback_max_rel": report["fused_fallback"]["max_rel"],
        "width_one_mps_vs_adjoint": report["width_one"]["mps_vs_adjoint_max_abs"],
        "tile_forward_max_rel": max(row["max_rel"] for row in report["tile_path"]["cases"]),
        "tile_backward_max_rel": max(
            row["grad_max_rel"] for row in report["tile_path"]["cases"]),
        "tile_deterministic": report["tile_path"]["deterministic"],
        "tile_width_one_mps_vs_adjoint": report["tile_path"]["width_one_mps_vs_adjoint"],
        "crossover_limit_bytes": report["crossover"]["limit_bytes"],
        "crossover_upsampling_tile_wins": (
            f"{report['crossover']['upsampling_tile_wins']}/"
            f"{report['crossover']['upsampling_cases']}"),
        "crossover_downscale_tile_wins": (
            f"{report['crossover']['downscale_tile_wins_above']}/"
            f"{report['crossover']['downscale_cases_above']}"),
        "crossover_downscale_bytes": report["crossover"]["downscale_crossover_bytes"],
        "plan_hit_rate": report["plans"]["axis_plan"]["hit_rate"],
        "headline_speedup": report["headline"]["speedup"],
        "headline_tile_gbs": report["headline"]["tile_gbs"],
        "headline_floor_gbs": report["headline"]["floor_gbs"],
        "all_finite": all(
            row["finite"] for row in report["forward"] + report["backward"]
        ),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print(f"torch {version}  mps missing upstream: {report['missing_before_register']}")
    print(f"forward  max rel {report['summary']['forward_max_rel']:.3e}  "
          f"bit-exact {report['summary']['forward_bit_exact_cases']}/{len(report['forward'])}")
    print(f"backward max rel {report['summary']['backward_max_rel']:.3e}  "
          f"bit-exact {report['summary']['backward_bit_exact_cases']}/{len(report['backward'])}")
    print(f"gradient norm rel {report['summary']['grad_norm_rel_max']:.3e}")
    print(f"F.interpolate max rel {report['summary']['interpolate_max_rel']:.3e}  "
          f"fused fallback max rel {report['summary']['fused_fallback_max_rel']:.3e}")
    print(f"width-1 forward vs CPU-backward adjoint "
          f"{report['width_one']['mps_vs_adjoint_max_abs']:.3e} "
          f"(CPU forward {report['width_one']['cpu_vs_adjoint_max_abs']:.3e}, "
          f"rows constant {report['width_one']['cpu_rows_constant']})")
    print(f"deterministic {report['deterministic']}  half guard {report['half_guard']}")
    print(f"tile path  forward max rel {report['summary']['tile_forward_max_rel']:.3e}  "
          f"backward max rel {report['summary']['tile_backward_max_rel']:.3e}  "
          f"deterministic {report['summary']['tile_deterministic']}  "
          f"width-1 adjoint {report['summary']['tile_width_one_mps_vs_adjoint']:.3e}")
    floor = report["memory_floor"]
    print(f"memory floor: copy {floor['copy_ms']:.3f} ms for {floor['copy_bytes'] / 1e6:.0f} MB"
          f" = {floor['copy_gbs']:.0f} GB/s")
    headline = report["headline"]
    print(f"headline {headline['input']} -> {headline['output']} (tile rows "
          f"{headline['tile_rows']}): two-pass {headline['two_pass_ms']:.3f} ms "
          f"({headline['two_pass_gbs']:.0f} GB/s) -> tile {headline['tile_ms']:.3f} ms "
          f"({headline['tile_gbs']:.0f} GB/s) = x{headline['speedup']:.2f}; "
          f"copy floor in the same batch {headline['floor_ms']:.3f} ms "
          f"({headline['floor_gbs']:.0f} GB/s)")
    for row in report["timing_ms"]:
        forward, backward = row["forward"], row["backward"]
        print(f"{row['case']:16s} {str(row['input']):20s} -> {str(row['output']):12s} "
              f"[{row['forward_stage']}/{row['backward_stage']}] "
              f"fwd cpu {forward['cpu']:6.2f} mps {forward['mps_latency']:6.2f}/"
              f"{forward['mps_steady']:6.3f} x{forward['latency_speedup']:5.2f}/"
              f"{forward['steady_speedup']:5.2f} "
              f"kernel {forward['kernel_steady']:6.3f} "
              f"{forward['kernel_bandwidth_gbs']:5.0f}GB/s  "
              f"bwd cpu {backward['cpu']:6.2f} mps {backward['mps_latency']:6.2f}/"
              f"{backward['mps_steady']:6.3f} x{backward['latency_speedup']:5.2f}/"
              f"{backward['steady_speedup']:5.2f} "
              f"kernel {backward['kernel_steady']:6.3f} "
              f"{backward['kernel_bandwidth_gbs']:5.0f}GB/s")
    crossover = report["crossover"]
    print(f"crossover: two-pass limit {crossover['limit_bytes'] / (1 << 20):.1f} MiB, "
          f"downscale crossover "
          f"{crossover['downscale_crossover_bytes'] / (1 << 20):.1f} MiB, "
          f"tile ahead at {crossover['upsampling_tile_wins']}/"
          f"{crossover['upsampling_cases']} upsampling sizes "
          f"(min x{crossover['upsampling_tile_min_speedup']:.2f}) and "
          f"{crossover['downscale_tile_wins_above']}/"
          f"{crossover['downscale_cases_above']} downscale sizes above it "
          f"(min x{crossover['downscale_tile_min_speedup_above']:.2f})")
    for row in crossover["cases"]:
        print(f"  up   scratch {row['scratch_bytes'] / (1 << 20):6.2f} MiB  "
              f"two-pass {row['two_pass_ms']:7.4f}  tile {row['tile_ms']:7.4f}  "
              f"x{row['speedup']:5.2f}  "
              f"({row['two_pass_gbs']:5.0f}/{row['tile_gbs']:5.0f} GB/s, "
              f"floor {row['floor_gbs']:5.0f} GB/s)")
    for row in crossover["downscale"]:
        print(f"  down {row['input'][2]:4d}x{row['input'][3]:<4d}->{row['output'][0]:4d}"
              f"x{row['output'][1]:<4d} th={row['tile_rows']:2d} "
              f"scratch {row['scratch_bytes'] / (1 << 20):6.2f} MiB  "
              f"two-pass {row['two_pass_ms']:7.4f}  tile {row['tile_ms']:7.4f}  "
              f"x{row['speedup']:5.2f}  "
              f"({row['two_pass_gbs']:5.0f}/{row['tile_gbs']:5.0f} GB/s, "
              f"floor {row['floor_gbs']:5.0f} GB/s)")
    plans = report["plans"]
    print(f"plans: axis hit rate {plans['axis_plan']['hit_rate']:.3f} "
          f"({plans['axis_plan']['hits']}/{plans['axis_plan']['hits'] + plans['axis_plan']['misses']}), "
          f"transposed {plans['transposed_plan']['hit_rate']:.3f}, "
          f"lookup {plans['axis_plan_lookup_us']:.2f} us, "
          f"build {plans['axis_plan_build_ms']:.3f} ms, "
          f"feature map cold {plans['feature_map_cold_ms']:.3f} / "
          f"warm {plans['feature_map_warm_ms']:.3f} ms "
          f"(penalty {plans['feature_map_cold_penalty_ms']:.3f})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
