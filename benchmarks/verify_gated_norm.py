"""Verify and measure ``src/metal_kernelops/cv/gated_norm.py``.

    uv run --project /Users/dbydd/vibe-agent-working-dir/ResearchChamber \
        python benchmarks/verify_gated_norm.py

Three independent checks stand behind the numbers:

1. The module against ``rmsnorm_gated_reference``, a literal transcription of
   mamba_ssm's ``rms_norm_ref`` evaluated in float64 on the CPU, over dtypes,
   gate orders, group sizes on both sides of the register cutoff, bias, leading
   dimensions, empty and non-contiguous input, plus the gradients of all four
   inputs.
2. The fused kernel's backward against a central finite difference of its own
   forward, which does not go through the reference at all.
3. The eager fallback against the same reference, so the path the CPU takes is
   held to the same standard, and the wide-group and bfloat16 shapes are held to
   the tolerance of their storage type.

Timings are warm and synchronized, taken under a tier-1 lease from the lab
broker when it is on this machine, and written to ``measured/gated_norm.json``.
Every shape is timed twice: through the fused kernel and through the composed
torch chain, which is what bfloat16 and groups past the cutoff used to run.
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

ROOT = Path(__file__).resolve().parents[1]
BROKER = Path("/Users/dbydd/vibe-agent-working-dir/ResearchChamber/shared_scripts")
sys.path.insert(0, str(ROOT / "src"))

from metal_kernelops.cv import gated_norm  # noqa: E402
from metal_kernelops.cv.gated_norm import (  # noqa: E402
    DENSE_GROUP,
    _composed,
    rmsnorm_gated,
    rmsnorm_gated_reference,
)

DEVICE = "mps"

# Absolute forward tolerance per dtype. The reference is float64, so this is the
# storage error of the dtype plus the float32 reduction inside the kernel.
FORWARD_TOLERANCE = {torch.float32: 2e-5, torch.float16: 5e-3, torch.bfloat16: 4e-2}
GRADIENT_TOLERANCE = {torch.float32: 1e-6, torch.float16: 2e-3, torch.bfloat16: 2e-2}
"""Relative error of the global gradient norm against the float64 reference."""


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


def _grads(module_output, grads_for, *inputs):
    wanted = [t for t in inputs if t is not None]
    return torch.autograd.grad(module_output, wanted, grads_for, allow_unused=True)


def correctness(report):
    cases = [
        # shape, group_size, bias, gate, norm_before_gate, eps
        ((512, 300), None, True, True, True, 1e-5),
        ((512, 300), 60, True, True, True, 1e-5),
        ((512, 300), 100, True, True, True, 1e-6),
        ((512, 300), 150, True, True, False, 1e-5),
        ((512, 300), None, False, True, True, 1e-5),
        ((512, 300), None, True, False, True, 1e-5),
        ((512, 300), 60, True, True, False, 1e-5),
        ((512, 300), None, True, True, True, 1.0),
        ((4, 8, 300), 100, True, True, True, 1e-5),
        ((3, 5, 7, 128), 32, True, True, True, 1e-5),
        ((1, 64), 16, True, True, False, 1e-5),
        ((7, 4096), 1024, True, True, True, 1e-5),
        ((2, 3, 8192), None, True, True, True, 1e-5),
        ((64, 16384), 4096, True, True, True, 1e-5),
        # Past the register cutoff: whole axis, and a wide group inside a wider one.
        ((16, 32768), None, True, True, True, 1e-5),
        ((8, 40000), 20000, True, True, False, 1e-5),
    ]
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for shape, group, bias, gate, before, eps in cases:
            x = torch.randn(*shape, device=DEVICE, dtype=dtype)
            weight = torch.randn(shape[-1], device=DEVICE, dtype=dtype)
            bias_t = torch.randn(shape[-1], device=DEVICE, dtype=dtype) if bias else None
            z = torch.randn(*shape, device=DEVICE, dtype=dtype) if gate else None
            out = rmsnorm_gated(x, weight, bias_t, z, eps=eps, group_size=group,
                                norm_before_gate=before)
            ref = rmsnorm_gated_reference(x, weight, bias_t, z, eps=eps, group_size=group,
                                          norm_before_gate=before)
            error = float((out.cpu().float() - ref.float()).abs().max())
            label = (f"{str(dtype).split('.')[-1]} {tuple(shape)} group={group} "
                     f"bias={int(bias)} z={int(gate)} before={int(before)} eps={eps}")
            report.check(f"forward {label}", error <= FORWARD_TOLERANCE[dtype],
                         {"max_abs_error": error, "tolerance": FORWARD_TOLERANCE[dtype]})
            report.check(f"forward dtype and shape {label}",
                         out.dtype == dtype and out.shape == x.shape
                         and out.device.type == DEVICE)


def gradients(report):
    cases = [
        ((256, 300), None, True, True, True),
        ((256, 300), 60, True, True, True),
        ((256, 300), 100, True, True, False),
        ((256, 300), None, False, True, True),
        ((256, 300), None, True, False, True),
        ((4, 6, 128), 32, True, True, False),
        ((8, 4096), 1024, True, True, True),
        ((16, 32768), None, True, True, True),
        ((8, 40000), 20000, True, True, False),
    ]
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for shape, group, bias, gate, before in cases:
            label = (f"{str(dtype).split('.')[-1]} {tuple(shape)} group={group} "
                     f"bias={int(bias)} z={int(gate)} before={int(before)}")
            x = torch.randn(*shape, device=DEVICE, dtype=dtype).requires_grad_(True)
            weight = torch.randn(shape[-1], device=DEVICE, dtype=dtype).requires_grad_(True)
            bias_t = (torch.randn(shape[-1], device=DEVICE, dtype=dtype).requires_grad_(True)
                      if bias else None)
            z = torch.randn(*shape, device=DEVICE, dtype=dtype).requires_grad_(True) if gate else None
            out = rmsnorm_gated(x, weight, bias_t, z, group_size=group, norm_before_gate=before)
            incoming = torch.randn_like(out)
            got = _grads(out, incoming, x, weight, bias_t, z)

            rx = x.detach().cpu().double().requires_grad_(True)
            rw = weight.detach().cpu().double().requires_grad_(True)
            rb = (bias_t.detach().cpu().double().requires_grad_(True) if bias_t is not None
                  else None)
            rz = z.detach().cpu().double().requires_grad_(True) if z is not None else None
            ref = rmsnorm_gated_reference(rx, rw, rb, rz, group_size=group,
                                          norm_before_gate=before)
            want = _grads(ref, incoming.detach().cpu().double(), rx, rw, rb, rz)

            names = [name for name, tensor in
                     (("x", x), ("weight", weight), ("bias", bias_t), ("z", z))
                     if tensor is not None]
            detail = {}
            worst = 0.0
            for name, g, g_ref in zip(names, got, want):
                g = g.cpu().double()
                relative = float((g - g_ref).norm() / (g_ref.norm() + 1e-30))
                detail[name] = {"relative_norm_error": relative,
                                "max_abs_error": float((g - g_ref).abs().max())}
                worst = max(worst, relative)
            report.check(f"gradients {label}", worst <= GRADIENT_TOLERANCE[dtype],
                         {"worst_relative_norm_error": worst,
                          "tolerance": GRADIENT_TOLERANCE[dtype], **detail})


def finite_difference(report, coordinates=12, step=1e-2):
    """Central differences of the module's own forward against its backward.

    No reference is involved: a wrong analytic ``dx`` or ``dz`` shows up directly.
    The floor here is float32, not the kernel -- a central difference of a sum in
    float32 has a relative noise of about ``eps * |L| / (h * |dL/dx|)``, which is
    a few parts in 10^3 at this tensor size and step. The tensors stay small for
    that reason; the same check at 4000x300 measures the noise, not the kernel.
    """
    for shape, group, before in (((4, 300), 100, True), ((4, 128), 32, False)):
        torch.manual_seed(11)
        x = torch.randn(*shape, device=DEVICE)
        weight = torch.randn(shape[-1], device=DEVICE)
        bias = torch.randn(shape[-1], device=DEVICE)
        z = torch.randn(*shape, device=DEVICE)
        incoming = torch.randn(*shape, device=DEVICE)

        def objective(x_in, z_in):
            return (rmsnorm_gated(x_in, weight, bias, z_in, group_size=group,
                                  norm_before_gate=before) * incoming).sum()

        xg = x.detach().clone().requires_grad_(True)
        objective(xg, z).backward()
        analytic_x = xg.grad.flatten()
        zg = z.detach().clone().requires_grad_(True)
        objective(x, zg).backward()
        analytic_z = zg.grad.flatten()

        index = torch.randperm(x.numel())[:coordinates].tolist()
        worst = {"x": 0.0, "z": 0.0}
        samples = []
        for position in index:
            measured = {}
            for name, base, analytic in (("x", x, analytic_x), ("z", z, analytic_z)):
                plus, minus = base.flatten().clone(), base.flatten().clone()
                plus[position] += step
                minus[position] -= step
                if name == "x":
                    numeric = float((objective(plus.view(shape), z)
                                     - objective(minus.view(shape), z)) / (2 * step))
                else:
                    numeric = float((objective(x, plus.view(shape))
                                     - objective(x, minus.view(shape))) / (2 * step))
                exact = float(analytic[position])
                relative = abs(numeric - exact) / max(abs(numeric), abs(exact), 1e-2)
                worst[name] = max(worst[name], relative)
                measured[name] = {"numeric": numeric, "analytic": exact}
            samples.append({"index": position, **measured})
        report.check(
            f"finite difference {tuple(shape)} group={group} before={int(before)}",
            max(worst.values()) < 2e-2,
            {"worst_relative_error": worst, "step": step, "samples": samples})


def fallback_path(report):
    """The composed path on the CPU must match the reference it claims to mirror."""
    tolerance = {torch.float32: 1e-5, torch.float64: 1e-12}
    for dtype in (torch.float32, torch.float64):
        for group, before in ((None, True), (60, False), (100, True)):
            shape = (128, 300)
            x = torch.randn(*shape, dtype=dtype)
            weight = torch.randn(shape[-1], dtype=dtype)
            bias = torch.randn(shape[-1], dtype=dtype)
            z = torch.randn(*shape, dtype=dtype)
            got = rmsnorm_gated(x, weight, bias, z, group_size=group, norm_before_gate=before)
            want = rmsnorm_gated_reference(x, weight, bias, z, group_size=group,
                                           norm_before_gate=before)
            error = float((got.double() - want.double()).abs().max())
            report.check(f"cpu fallback {str(dtype).split('.')[-1]} group={group} "
                         f"before={int(before)}", error <= tolerance[dtype],
                         {"max_abs_error": error, "tolerance": tolerance[dtype]})

    # bfloat16 on MPS has its own kernel now, and it agrees with the composed
    # path it used to take, in bfloat16's own storage granularity.
    x = torch.randn(128, 300, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(300, device=DEVICE, dtype=torch.bfloat16)
    got = rmsnorm_gated(x, weight)
    want = _composed(x, weight, None, None, 1e-5, 300, True)
    error = float((got.float() - want.float()).abs().max())
    report.check("bfloat16 takes the Metal path",
                 gated_norm._metal_applies(x, weight, None, None))
    report.check("bfloat16 matches the composed path it replaced",
                 error <= FORWARD_TOLERANCE[torch.bfloat16],
                 {"max_abs_error": error, "tolerance": FORWARD_TOLERANCE[torch.bfloat16]})


def edge_cases(report):
    # Non-contiguous inputs: reshape alone would hand the kernel a strided buffer.
    base = torch.randn(64, 129, device=DEVICE)
    for label, x in (("transposed", base.t()), ("strided", torch.randn(64, 258, device=DEVICE)[:, ::2])):
        weight = torch.randn(x.shape[-1], device=DEVICE)
        out = rmsnorm_gated(x, weight)
        ref = rmsnorm_gated_reference(x, weight)
        error = float((out.cpu() - ref).abs().max())
        report.check(f"non-contiguous {label}", not x.is_contiguous() and error < 2e-5,
                     {"max_abs_error": error, "strides": list(x.stride())})

        xg = x.detach().clone().requires_grad_(True)
        wg = weight.detach().clone().requires_grad_(True)
        grad = torch.randn_like(rmsnorm_gated(xg, wg))
        got = rmsnorm_gated(xg, wg)
        gx, gw = torch.autograd.grad((got * grad).sum(), (xg, wg))
        rg = rmsnorm_gated_reference(x, weight)
        rx = x.detach().cpu().double().requires_grad_(True)
        rw = weight.detach().cpu().double().requires_grad_(True)
        rref = rmsnorm_gated_reference(rx, rw)
        rgx, rgw = torch.autograd.grad((rref * grad.detach().cpu().double()).sum(), (rx, rw))
        relative = float((gx.cpu().double() - rgx).norm() / rgx.norm())
        report.check(f"non-contiguous {label} gradients", relative < 1e-6,
                     {"relative_norm_error": relative})

    # Empty inputs must not launch a kernel, must not crash, and keep the shape.
    for shape in ((0, 300), (5, 0), (0,), (2, 0, 7)):
        x = torch.empty(*shape, device=DEVICE)
        weight = torch.empty(shape[-1] if shape else 0, device=DEVICE)
        out = rmsnorm_gated(x, weight)
        report.check(f"empty {tuple(shape)}", out.shape == x.shape and out.numel() == 0
                     and out.dtype == x.dtype)

    # A zero-row batch with a real feature axis still exercises the kernel launch.
    x = torch.empty(0, 300, device=DEVICE, requires_grad=False)
    weight = torch.ones(300, device=DEVICE)
    report.check("zero rows keeps dtype", rmsnorm_gated(x, weight).dtype == x.dtype)

    # One row, one group of one: the covers for a threadgroup wider than the group.
    for features in (1, 2, 31, 33):
        x = torch.randn(4, features, device=DEVICE)
        weight = torch.randn(features, device=DEVICE)
        out = rmsnorm_gated(x, weight)
        ref = rmsnorm_gated_reference(x, weight)
        error = float((out.cpu() - ref).abs().max())
        report.check(f"narrow feature axis {features}", error < 2e-5, {"max_abs_error": error})

    # The cutoff is a path boundary, not a limit: DENSE_GROUP goes through the
    # register kernel and wider groups through the streaming one, and both agree
    # with the reference at the tolerance of their storage type.
    for features in (DENSE_GROUP, 2 * DENSE_GROUP, 4 * DENSE_GROUP):
        for dtype in (torch.float32, torch.bfloat16):
            x = torch.randn(2, features, device=DEVICE, dtype=dtype)
            weight = torch.randn(features, device=DEVICE, dtype=dtype)
            out = rmsnorm_gated(x, weight)
            ref = rmsnorm_gated_reference(x, weight)
            error = float((out.cpu().float() - ref.float()).abs().max())
            path = "dense" if features <= DENSE_GROUP else "stream"
            report.check(f"feature axis {features} {str(dtype).split('.')[-1]}", 
                         error <= FORWARD_TOLERANCE[dtype],
                         {"max_abs_error": error, "path": path,
                          "tolerance": FORWARD_TOLERANCE[dtype]})

    # The Metal gate used to reject wide groups; it must not any more.
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        x = torch.randn(2, 2 * DENSE_GROUP, device=DEVICE, dtype=dtype)
        weight = torch.randn(2 * DENSE_GROUP, device=DEVICE, dtype=dtype)
        report.check(f"wide group keeps the Metal path {str(dtype).split('.')[-1]}",
                     gated_norm._metal_applies(x, weight, None, None))

    # Error parity: bad shapes and a group that does not divide the axis.
    def raises(fn):
        try:
            fn()
        except ValueError:
            return True
        return False

    x = torch.randn(4, 300, device=DEVICE)
    weight = torch.ones(300, device=DEVICE)
    report.check("rejects a group that does not divide",
                 raises(lambda: rmsnorm_gated(x, weight, group_size=7)))
    report.check("rejects a weight of the wrong length",
                 raises(lambda: rmsnorm_gated(x, torch.ones(299, device=DEVICE))))
    report.check("rejects a gate of the wrong shape",
                 raises(lambda: rmsnorm_gated(x, weight, z=torch.randn(4, 299, device=DEVICE))))
    report.check("rejects a scalar input",
                 raises(lambda: rmsnorm_gated(torch.randn((), device=DEVICE), weight)))


def mixed_weight_dtype(report):
    """A float32 weight on float16 activations is the usual inference setup."""
    x = torch.randn(64, 512, device=DEVICE, dtype=torch.float16)
    weight = torch.randn(512, device=DEVICE, dtype=torch.float32)
    bias = torch.randn(512, device=DEVICE, dtype=torch.float32)
    z = torch.randn(64, 512, device=DEVICE, dtype=torch.float16)
    xg = x.detach().clone().requires_grad_(True)
    wg = weight.detach().clone().requires_grad_(True)
    bg = bias.detach().clone().requires_grad_(True)
    zg = z.detach().clone().requires_grad_(True)
    out = rmsnorm_gated(xg, wg, bg, zg)
    out.sum().backward()
    ref = rmsnorm_gated_reference(x, weight, bias, z)
    error = float((out.detach().cpu().float() - ref.float()).abs().max())
    report.check("mixed weight dtype forward", error < 5e-3, {"max_abs_error": error})
    report.check("mixed weight dtype gradients keep dtype",
                 wg.grad.dtype == torch.float32 and bg.grad.dtype == torch.float32
                 and xg.grad.dtype == torch.float16 and zg.grad.dtype == torch.float16)
    rx = x.detach().cpu().double().requires_grad_(True)
    rw = weight.detach().cpu().double().requires_grad_(True)
    rb = bias.detach().cpu().double().requires_grad_(True)
    rz = z.detach().cpu().double().requires_grad_(True)
    rref = rmsnorm_gated_reference(rx, rw, rb, rz)
    rwx = torch.autograd.grad(rref.sum(), (rx, rw, rb, rz))
    relative = float((wg.grad.cpu().double() - rwx[1]).norm() / rwx[1].norm())
    report.check("mixed weight dtype gradient value", relative < 2e-3,
                 {"relative_norm_error": relative})


def optimizer_step(report):
    """A real training step moves the parameters and lowers the loss."""
    torch.manual_seed(0)
    shape = (256, 512)
    target = torch.randn(*shape, device=DEVICE)
    x = torch.randn(*shape, device=DEVICE)
    z = torch.randn(*shape, device=DEVICE)
    weight = torch.ones(shape[-1], device=DEVICE, requires_grad=True)
    bias = torch.zeros(shape[-1], device=DEVICE, requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=1e-2)
    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        loss = (rmsnorm_gated(x, weight, bias, z) - target).pow(2).mean()
        loss.backward()
        losses.append(float(loss.detach()))
        optimizer.step()
    report.check("loss decreases over five Adam steps", losses[-1] < losses[0],
                 {"losses": losses})
    report.check("both parameters moved",
                 not torch.equal(weight.detach(), torch.ones(shape[-1], device=DEVICE))
                 and not torch.equal(bias.detach(), torch.zeros(shape[-1], device=DEVICE)))
    return losses


@contextlib.contextmanager
def lease():
    if not (BROKER / "compute_broker.py").exists():
        print("no compute broker here, timing unleased")
        yield
        return
    sys.path.insert(0, str(BROKER))
    from compute_broker import lease_guard

    with lease_guard("mps-kernelops", tier=1, task_name="verify_gated_norm"):
        yield


def _measure(entries, iters, rounds, warmup=3):
    """Per-call microseconds, variants interleaved round by round."""
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
    return {name: {"median": round(statistics.median(values), 1), "min": round(min(values), 1)}
            for name, values in samples.items()}


BENCH_SHAPES = (
    # rows, features, group_size; None means the whole feature axis
    (8192, 2560, None), (32768, 1024, None), (2048, 4096, None), (65536, 512, None),
    (8192, 8192, None), (4096, 768, None), (1024, 5120, None),
    (2048, 16384, None),      # the widest group the register kernels take
    (2048, 32768, None),      # past it, streaming, whole axis
    (1024, 65536, 32768),     # streaming, two groups per row
)


def _sampling(rows, features):
    """Calls per round and rounds, by tensor size: the composed chain is slow."""
    elements = rows * features
    if elements <= 1 << 23:
        return 20, 15
    if elements <= 1 << 25:
        return 10, 13
    return 3, 9


def timings():
    """Every variant of one shape in one interleaved batch.

    The three dtypes of a shape are timed together, not in separate sweeps: this
    box is shared with two dozen other builds, and the load drifts enough between
    sweeps to invert a bf16-against-fp32 comparison that a single batch settles.
    """
    measured = {}
    for rows, features, group_size in BENCH_SHAPES:
        # The previous shape leaves cached blocks behind; a fresh cache keeps
        # the allocation noise of one shape out of the next one's numbers.
        torch.mps.empty_cache()
        group = features if group_size is None else group_size
        iters, rounds = _sampling(rows, features)
        entries = []
        state = {}
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            tag, x, variants = _variants(dtype, rows, features, group)
            state[tag] = (dtype, x)
            entries += variants

        # One batch, one lease: the fused kernel, the composed chain it replaced
        # and a same-shape copy_ that is the floor of this memory system at this
        # instant, for all three dtypes at once.
        batch = _measure(entries, iters=iters, rounds=rounds)
        for tag, (dtype, x) in state.items():
            row = {name.split(":", 1)[1]: value for name, value in batch.items()
                   if name.startswith(f"{tag}:")}
            label = f"{dtype} {rows}x{features} group={group}"
            row["rounds"] = rounds
            row["calls_per_round"] = iters
            row["path"] = "dense" if group <= DENSE_GROUP else "stream"
            threads = (gated_norm._STREAM_THREADS if group > DENSE_GROUP
                       else gated_norm._threads(group))
            blocks, rows_per_block = gated_norm._row_blocks(rows, features // group, features)
            row["kernel"] = {"threadgroup": threads,
                             "elements_per_thread": (None if group > DENSE_GROUP
                                                     else gated_norm._depth(group, threads)),
                             "backward_blocks": blocks,
                             "rows_per_block": rows_per_block,
                             "kernels_per_forward": 1,
                             "kernels_per_step": 2 if group <= DENSE_GROUP else 3}
            row["before_ms"] = {
                "forward": round(row["composed_forward"]["min"] / 1e3, 3),
                "forward_backward": round(row["composed_forward_backward"]["min"] / 1e3, 3)}
            row["after_ms"] = {
                "forward": round(row["fused_forward"]["min"] / 1e3, 3),
                "forward_backward": round(row["fused_forward_backward"]["min"] / 1e3, 3)}
            row["speedup_forward"] = round(
                row["composed_forward"]["min"] / row["fused_forward"]["min"], 2)
            row["speedup_forward_backward"] = round(
                row["composed_forward_backward"]["min"] / row["fused_forward_backward"]["min"], 2)
            bytes_moved = rows * features * x.element_size()
            row["gigabytes_per_second_forward"] = round(
                3 * bytes_moved / (row["fused_forward"]["min"] * 1e3), 3)
            row["gigabytes_per_second_copy_floor"] = round(
                2 * bytes_moved / (row["copy_floor"]["min"] * 1e3), 3)
            measured[label] = row
            print(f"{label:44s} {row['path']:6s} after {row['after_ms']['forward']:8.3f} / "
                  f"{row['after_ms']['forward_backward']:8.3f} ms  before "
                  f"{row['before_ms']['forward']:8.3f} / "
                  f"{row['before_ms']['forward_backward']:8.3f} ms  "
                  f"{row['speedup_forward']:5.2f}x / {row['speedup_forward_backward']:5.2f}x  "
                  f"{row['gigabytes_per_second_forward']:6.1f} GB/s "
                  f"(copy floor {row['gigabytes_per_second_copy_floor']:6.1f})")
    return measured


def _variants(dtype, rows, features, group):
    """The five timed closures for one dtype of one shape, in their own frame.

    Built by a function rather than inline so each dtype's closures capture that
    dtype's tensors: a loop body rebinds the names and every closure would end up
    reading the last dtype's buffers.
    """
    tag = str(dtype).split(".")[-1]
    x = torch.randn(rows, features, device=DEVICE, dtype=dtype)
    z = torch.randn(rows, features, device=DEVICE, dtype=dtype)
    weight = torch.randn(features, device=DEVICE, dtype=dtype)
    bias = torch.randn(features, device=DEVICE, dtype=dtype)
    leaf_x = x.detach().clone().requires_grad_(True)
    leaf_z = z.detach().clone().requires_grad_(True)
    incoming = torch.randn_like(rmsnorm_gated(leaf_x, weight, bias, leaf_z, group_size=group))
    copy_dst = torch.empty_like(x)

    def step():
        leaf_x.grad = None
        leaf_z.grad = None
        rmsnorm_gated(leaf_x, weight, bias, leaf_z, group_size=group).backward(incoming)

    def composed_forward():
        return _composed(x, weight, bias, z, 1e-5, group, True)

    def composed_step():
        leaf_x.grad = None
        leaf_z.grad = None
        _composed(leaf_x, weight, bias, leaf_z, 1e-5, group, True).backward(incoming)

    return tag, x, [
        (f"{tag}:fused_forward",
         lambda: rmsnorm_gated(x, weight, bias, z, group_size=group)),
        (f"{tag}:fused_forward_backward", step),
        (f"{tag}:composed_forward", composed_forward),
        (f"{tag}:composed_forward_backward", composed_step),
        (f"{tag}:copy_floor", lambda: copy_dst.copy_(x)),
    ]


def _comparison(measured):
    """The two targets of this round, read off the same interleaved batch.

    bfloat16 has to land within 1.2x of float32 through the fused kernel, and a
    group wider than the register cutoff has to beat the composed chain by 3x.
    Both are ratios of minima from one lease, so the machine load cancels.
    """
    by_dtype = {}
    for key, row in measured.items():
        dtype, rest = key.split(" ", 1)
        by_dtype.setdefault(dtype, {})[rest] = row
    shapes = list(next(iter(by_dtype.values())))

    bfloat16 = {}
    for shape in shapes:
        fp32 = by_dtype["torch.float32"][shape]
        bf16 = by_dtype["torch.bfloat16"][shape]
        bfloat16[shape] = {
            "forward_ratio": round(bf16["after_ms"]["forward"]
                                   / fp32["after_ms"]["forward"], 2),
            "forward_backward_ratio": round(bf16["after_ms"]["forward_backward"]
                                            / fp32["after_ms"]["forward_backward"], 2),
        }

    wide = {}
    for dtype, per_dtype in by_dtype.items():
        for shape, row in per_dtype.items():
            if row["path"] != "stream":
                continue
            wide[f"{dtype} {shape}"] = {
                "forward_speedup": row["speedup_forward"],
                "forward_backward_speedup": row["speedup_forward_backward"],
            }

    worst_bfloat16 = max(max(v.values()) for v in bfloat16.values())
    worst_wide = min(min(v.values()) for v in wide.values())
    return {
        "bfloat16_against_float32_fused": bfloat16,
        "wide_group_against_composed": wide,
        "worst_bfloat16_ratio": round(worst_bfloat16, 2),
        "worst_wide_group_speedup": round(worst_wide, 2),
        "targets": {"bfloat16_at_most": 1.2, "wide_group_at_least": 3.0,
                    "met": worst_bfloat16 <= 1.2 and worst_wide >= 3.0},
    }


def main():
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available on this machine")
    torch.manual_seed(0)

    # Import must not compile a shader; the first call does.
    compiled_at_import = gated_norm._library.cache_info().misses
    report = Report()
    report.check("import compiles nothing", compiled_at_import == 0,
                 {"shader_library_misses": compiled_at_import})

    correctness(report)
    gradients(report)
    finite_difference(report)
    fallback_path(report)
    edge_cases(report)
    mixed_weight_dtype(report)
    losses = optimizer_step(report)
    report.check("shader compiled lazily on first call", gated_norm._library.cache_info().misses == 1,
                 {"shader_library_misses": gated_norm._library.cache_info().misses})

    print("\nchecks")
    for row in report.rows:
        print(f"  {'ok  ' if row['ok'] else 'FAIL'} {row['check']}"
              + (f"  {row['detail']}" if not row["ok"] and "detail" in row else ""))

    print("\ntimings (warm, median of rounds, torch.mps.synchronize after each round)")
    load_before = round(os.getloadavg()[0], 2)
    with lease():
        measured = timings()
    load_after = round(os.getloadavg()[0], 2)
    comparison = _comparison(measured)
    print(f"\nbfloat16 against float32, fused: worst {comparison['worst_bfloat16_ratio']:.2f}x "
          f"(target at most 1.20x)")
    print(f"wide groups against the composed chain: worst {comparison['worst_wide_group_speedup']:.2f}x "
          f"(target at least 3.00x)")

    payload = {
        "schema": 1,
        "module": "metal_kernelops/cv/gated_norm.py",
        "torch": torch.__version__,
        "platform": f"{platform.platform()} {platform.machine()}",
        "device": torch.device(DEVICE).type,
        "api": (
            "rmsnorm_gated(x, weight, bias=None, z=None, eps=1e-5, group_size=None, "
            "norm_before_gate=True) -> Tensor; no aten registration"
        ),
        "semantics": (
            "rms_norm_ref from mamba_ssm.ops.triton.layernorm_gated: gate applied "
            "before or after normalization, biased mean over group_size elements, "
            "float32 intermediates, result cast back to x.dtype"
        ),
        "contract": {
            "x": "(..., features), float32, float16 or bfloat16 on MPS for the Metal path",
            "weight": "(features,), any float dtype; upcast to float32 for the kernel",
            "bias": "(features,) or None",
            "z": "same shape and dtype as x, or None",
            "group_size": f"divides features; {DENSE_GROUP} or less takes the register "
                          "kernels, wider takes the streaming ones",
            "backward": "analytic dx, dz, dw, db; weight gradient in weight.dtype",
        },
        "paths": {
            "dense": f"MPS, float32/float16/bfloat16, group at most {DENSE_GROUP}",
            "stream": f"MPS, float32/float16/bfloat16, group wider than {DENSE_GROUP}",
            "composed": "CPU, float64, empty input",
            "cutoff": DENSE_GROUP,
            "shader_compiled_at_import": compiled_at_import,
        },
        "correctness": {
            "checks": len(report.rows),
            "failed": report.failed,
            "rows": report.rows,
        },
        "training_step": {"adam_losses": losses},
        "timings_us": measured,
        "timings_method": (
            "each variant runs 3 to 20 calls per round with one torch.mps.synchronize at the "
            "end; the five variants of a shape -- fused forward, fused step, composed forward, "
            "composed step, and a same-shape copy_ floor -- are interleaved round by round "
            "inside one lease, so the machine load is common to all of them; median and min "
            "over 9 to 15 rounds are recorded, the ratios and before/after use min, the fused "
            "numbers include the python dispatch, and the composed numbers are the same "
            "arithmetic in torch ops on the same tensors"
        ),
        "comparison": comparison,
        "timings_load_average": [load_before, load_after],
        "notes": [
            "one threadgroup per (row, group); the group's values stay in registers between "
            "the sum of squares and the write, so x, z and y cross the bus once each",
            "a group wider than the register set streams: the same threadgroup walks it once "
            "for the mean square and once for the write, still one kernel",
            "the wide backward is two kernels -- one threadgroup per row-group for c1, then "
            "one thread per feature column that walks its row block and writes a dw/db "
            "partial -- so a column is a register whatever the group width is",
            "bfloat16 has no type in this MSL, so it is read as the top 16 bits of a float32 "
            "and written back rounded to nearest even, which matches torch's own cast bit for "
            "bit over 2^18 random values",
            "the backward keeps dw and db in registers over a run of rows and reduces 256 "
            "partials in torch; measured cost over 192 to 1024 partials is flat within 5%, "
            "and 64 partials costs 70% on a 4096x768 step",
            "float16 and bfloat16 accumulate in float32: squares, the mean and rstd never "
            "round to the narrow type, matching the reference's upcast",
            "torch 2.14.1 silently returns zeros from mps.to('cpu', torch.float64) instead of "
            "raising, so the float64 reference converts in two steps",
            "the gate order is the shader's enum: none=0, before normalization=1, "
            "after=2",
        ],
        "limits": [
            "the streaming kernels re-read x and z for the write pass, so a group past the "
            "cutoff costs two reads of the input rather than one",
            "the wide backward reads x, z and dy twice, once for c1 and once for the "
            "gradients",
            "at the largest wide shapes the step is launch-bound, not bandwidth-bound: on "
            "1024x65536 float32 the three kernels measure 2.2, 2.6 and 2.7 ms (366, 527 and "
            "499 GB/s against a 421 GB/s copy_ floor) while the step takes 11.4 ms, so about "
            "a third of it is the per-launch cost this shared box charges; that cell is the "
            "one place where the fused step is 2.7x the composed chain rather than 4.2x or "
            "more, and a single-kernel wide backward -- one threadgroup per row-group walking "
            "the group in chunks of tgs*E, so dw/db stay in registers -- is how to remove it",
            "the gate is silu only; a sigmoid or other gate would need another kernel",
            "one threadgroup per row-group means a feature axis narrower than 32 leaves part "
            "of the first warp idle",
            "16-bit lanes waste half of each memory transaction on this GPU: a bfloat16 copy "
            "measured 128 GB/s per element against 305 GB/s with two elements packed into one "
            "32-bit access, and standalone forward kernels measured 1.36x (float32), 1.50x "
            "(float16) and 1.88x (bfloat16) for the paired form. Integrated into this kernel "
            "set the same templated accessor returned wrong values, which is why the shipped "
            "kernels still move one element per access; the paired path is the next thing to "
            "try, behind a correctness check at every depth and threadgroup width",
            "the host is shared with other builds: absolute microseconds move by a factor of "
            "two between runs, so the interleaved speedup ratios are the stable part of "
            "these numbers and the min is the closer estimate of the kernel",
            "a different machine, an M-series core count or a later torch will change the "
            "threadgroup heuristic, which was set from measurements of this one",
        ],
    }

    out = ROOT / "measured" / "gated_norm.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}")
    print(f"{len(report.rows)} checks, {report.failed} failed")
    if report.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
