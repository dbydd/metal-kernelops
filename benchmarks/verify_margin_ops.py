"""Verify the margin-loss MPS overlays against the CPU kernels and time them.

Run with the workspace interpreter::

    uv run --project /Users/dbydd/vibe-agent-working-dir/ResearchChamber \
        python benchmarks/verify_margin_ops.py

Registers the eight overlays in this process only, then checks forward values,
``is_target``, gradients through autograd, the backward aten ops directly, the
``.out``/``.grad_input``/``.output`` overloads, optimizer steps, and the
float16/bfloat16 kernels. ``run_path_cases`` records the kernels each dtype and
shape actually launches, so the one-launch reduction stays a measured property
rather than a comment. Timings are warm and synchronized, reported both per
queued call and per call with a synchronization at the end; the fused and
composed paths are timed against each other in the same process, alternating
which one runs first, and the cost of one kernel launch is measured directly.

The JSON written to ``measured/margin_ops.json`` carries the numbers; the
process exit code is non-zero when any check fails.
"""

import argparse
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
from torch.library import Library

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import metal_kernelops.margin_ops as margin_ops  # noqa: E402

REDUCTIONS = {"none": 0, "mean": 1, "sum": 2}
REDUCTION_NAMES = {0: "none", 1: "mean", 2: "sum"}
FAILURES = []
CHECKS = {"count": 0, "failed": 0, "max_abs": 0.0, "max_rel": 0.0}
REGISTRATION = None  # the Library must outlive install(); dropping it unregisters
BROKER_SCRIPTS = Path(
    os.environ.get(
        "COMPUTE_BROKER_SCRIPTS",
        "/Users/dbydd/vibe-agent-working-dir/ResearchChamber/shared_scripts",
    )
)


def compare(name, got, want, atol, rtol=0.0, exact=False, bucket=None):
    stats = CHECKS if bucket is None else bucket
    CHECKS["count"] += 1
    got = got.detach().to("cpu", torch.float32)
    want = want.detach().to("cpu", torch.float32)
    if got.shape != want.shape:
        FAILURES.append(f"{name}: shape {tuple(got.shape)} != {tuple(want.shape)}")
        CHECKS["failed"] += 1
        return False
    if want.numel() == 0:
        return True
    diff = (got - want).abs()
    max_abs = diff.max().item()
    max_rel = (diff / want.abs().clamp_min(1e-12)).max().item()
    stats["max_abs"] = max(stats.get("max_abs", 0.0), max_abs)
    stats["max_rel"] = max(stats.get("max_rel", 0.0), max_rel)
    if exact:
        ok = max_abs == 0.0
    else:
        ok = bool(torch.allclose(got, want, rtol=rtol, atol=atol))
    if not ok:
        CHECKS["failed"] += 1
        FAILURES.append(
            f"{name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
            f"got={got.flatten()[:4].tolist()} want={want.flatten()[:4].tolist()}"
        )
    return ok


def describe():
    return {
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "device": "mps",
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "load_average": [round(value, 2) for value in os.getloadavg()],
    }


def before_registration():
    names = [name for name, _ in margin_ops.OVERLAYS]
    return {
        "overloads": names,
        "mps_kernel_before": {
            name: torch._C._dispatch_has_kernel_for_dispatch_key(f"aten::{name}", "MPS")
            for name in names
        },
    }


def install():
    global REGISTRATION
    library = Library("aten", "FRAGMENT")
    for name, function in margin_ops.OVERLAYS:
        library.impl(name, function, "MPS")
    REGISTRATION = library
    return library


def broker_lease():
    """Tier-1 probe lease from the workspace broker, when it is on this machine."""
    if not (BROKER_SCRIPTS / "compute_broker.py").exists():
        return contextlib.nullcontext(), {"acquired": False, "reason": "broker not found"}
    sys.path.insert(0, str(BROKER_SCRIPTS))
    from compute_broker import lease_guard

    return (
        lease_guard("mps-kernelops", tier=1, task_name="verify_margin_ops"),
        {"acquired": True, "project": "mps-kernelops", "tier": 1},
    )


def multilabel_targets(n, d, labels, duplicates=False, empty_rows=False):
    generator = torch.Generator().manual_seed(n * 1000 + d * 10 + labels)
    target = torch.full((n, d), -1, dtype=torch.int64)
    for i in range(n):
        count = 0 if (empty_rows and i % 4 == 0) else min(labels, d)
        picked = torch.randperm(d, generator=generator)[:count]
        if duplicates and count > 1:
            picked[0] = picked[-1]
        target[i, :count] = picked
    return target


def parity(name, cpu_fn, mps_fn, cpu_inputs, mps_inputs, atol, grad_atol):
    cpu_out = cpu_fn(*cpu_inputs)
    mps_out = mps_fn(*mps_inputs)
    if isinstance(cpu_out, tuple):
        compare(f"{name} output", mps_out[0], cpu_out[0], atol, 1e-5)
        compare(f"{name} is_target", mps_out[1], cpu_out[1], 0.0, exact=True)
        cpu_value, mps_value = cpu_out[0], mps_out[0]
    else:
        compare(f"{name} output", mps_out, cpu_out, atol, 1e-5)
        cpu_value, mps_value = cpu_out, mps_out
    if not cpu_inputs[0].requires_grad:
        return
    seed = torch.ones_like(cpu_value)
    cpu_grad = torch.autograd.grad(cpu_value, cpu_inputs[0], seed)[0]
    mps_grad = torch.autograd.grad(mps_value, mps_inputs[0], seed.to("mps"))[0]
    compare(f"{name} grad", mps_grad, cpu_grad, grad_atol, 1e-4)


def run_multi_margin_cases(device):
    for n, d in [(0, 3), (1, 3), (8, 5), (64, 37), (16, 100), (4, 257),
                 (256, 10), (2, 1000), (1024, 5)]:
        torch.manual_seed(n * 31 + d)
        x = torch.randn(n, d)
        target = torch.randint(0, d, (n,))
        for p in (1, 2):
            for margin in (1.0, 0.5, 2.0):
                for reduction in ("none", "mean", "sum"):
                    for weighted in (False, True):
                        weight = torch.rand(d) + 0.25 if weighted else None
                        label = f"mm {n}x{d} p{p} m{margin} {reduction} w{int(weighted)}"
                        x_cpu = x.clone().requires_grad_(True)
                        x_mps = x.to(device).requires_grad_(True)
                        w_mps = weight.to(device) if weight is not None else None

                        def cpu_fn(a, b, p=p, margin=margin, w=weight, r=reduction):
                            return F.multi_margin_loss(
                                a, b, p=p, margin=margin, weight=w, reduction=r)

                        def mps_fn(a, b, p=p, margin=margin, w=w_mps, r=reduction):
                            return F.multi_margin_loss(
                                a, b, p=p, margin=margin, weight=w, reduction=r)

                        # Reduction.sum adds nframe/dim*dim terms; error grows
                        # with the reduction tree depth, so the tolerance is
                        # relative.
                        tol = 1e-5 if reduction != "sum" else 1e-4
                        parity(
                            label, cpu_fn, mps_fn,
                            (x_cpu, target), (x_mps, target.to(device)),
                            tol, 1e-4,
                        )


def run_multi_margin_edges(device):
    for shape, target_shape, reduction in [
        ((3,), (), "mean"), ((3,), (1,), "none"), ((3,), (1,), "mean"),
        ((), (), "mean"), ((), (1,), "none"), ((1, 3), (), "none"),
    ]:
        x = torch.randn(torch.Size(shape))
        target = torch.tensor(0).reshape(target_shape)
        x_cpu = x.clone().requires_grad_(True)
        x_mps = x.to(device).requires_grad_(True)
        parity(
            f"mm edge {shape} target{target_shape} {reduction}",
            lambda a, b, r=reduction: F.multi_margin_loss(a, b, reduction=r),
            lambda a, b, r=reduction: F.multi_margin_loss(a, b, reduction=r),
            (x_cpu, target), (x_mps, target.to(device)),
            1e-5, 1e-4,
        )

    x = torch.randn(8, 37).t()
    target = torch.randint(0, 8, (37,))
    parity(
        "mm non-contiguous",
        lambda a, b: F.multi_margin_loss(a, b),
        lambda a, b: F.multi_margin_loss(a, b),
        (x.clone().requires_grad_(True), target),
        (x.to(device).requires_grad_(True), target.to(device)),
        1e-5, 1e-4,
    )

    x = torch.randn(12, 24)
    target = torch.randint(0, 24, (12,))
    weight = torch.rand(48)[::2]
    parity(
        "mm strided weight",
        lambda a, b: F.multi_margin_loss(a, b, weight=weight),
        lambda a, b: F.multi_margin_loss(a, b, weight=weight.to(device)),
        (x.clone().requires_grad_(True), target),
        (x.to(device).requires_grad_(True), target.to(device)),
        1e-5, 1e-4,
    )

    for n, d in [(5, 9), (7, 64)]:
        torch.manual_seed(n * 5 + d)
        x = torch.randn(n, d)
        target = torch.randint(0, d, (n,))
        for p in (1, 2):
            for margin in (-0.5, 0.0):
                for weight in (torch.zeros(d), torch.full((d,), -1.5)):
                    for reduction in ("none", "mean", "sum"):
                        parity(
                            f"mm weight {n}x{d} p{p} m{margin} "
                            f"w{weight[0].item():+.1f} {reduction}",
                            lambda a, b, p=p, margin=margin, w=weight, r=reduction:
                                F.multi_margin_loss(a, b, p=p, margin=margin,
                                                    weight=w, reduction=r),
                            lambda a, b, p=p, margin=margin, w=weight.to(device),
                                   r=reduction:
                                F.multi_margin_loss(a, b, p=p, margin=margin,
                                                    weight=w, reduction=r),
                            (x.clone().requires_grad_(True), target),
                            (x.to(device).requires_grad_(True), target.to(device)),
                            1e-5, 1e-4,
                        )


def run_multilabel_cases(device):
    for n, d in [(0, 3), (1, 1), (4, 5), (8, 37), (3, 64), (2, 257), (2, 1000)]:
        for labels in (1, 3, d):
            for reduction in ("none", "mean", "sum"):
                target = multilabel_targets(n, d, labels, duplicates=n > 1, empty_rows=True)
                torch.manual_seed(n * 7 + d + labels)
                x = torch.randn(n, d)
                code = REDUCTIONS[reduction]
                parity(
                    f"ml {n}x{d} labels{labels} {reduction}",
                    lambda a, b, r=code: torch.ops.aten.multilabel_margin_loss_forward(a, b, r),
                    lambda a, b, r=code: torch.ops.aten.multilabel_margin_loss_forward(a, b, r),
                    (x.clone().requires_grad_(True), target),
                    (x.to(device).requires_grad_(True), target.to(device)),
                    1e-5, 1e-4,
                )

    for shape, values in [((4,), [0, -1, 2, 1]), ((), [0]), ((1, 4), [0, -1, 2, 1])]:
        x = torch.randn(torch.Size(shape))
        target = torch.tensor(values, dtype=torch.int64).reshape(shape)
        for reduction in ("none", "mean", "sum"):
            code = REDUCTIONS[reduction]
            parity(
                f"ml edge {shape} {reduction}",
                lambda a, b, r=code: torch.ops.aten.multilabel_margin_loss_forward(a, b, r),
                lambda a, b, r=code: torch.ops.aten.multilabel_margin_loss_forward(a, b, r),
                (x.clone().requires_grad_(True), target),
                (x.to(device).requires_grad_(True), target.to(device)),
                1e-5, 1e-4,
            )

    x = torch.randn(8, 37).t()
    target = multilabel_targets(37, 8, 3)
    parity(
        "ml non-contiguous",
        lambda a, b: torch.ops.aten.multilabel_margin_loss_forward(a, b, 1),
        lambda a, b: torch.ops.aten.multilabel_margin_loss_forward(a, b, 1),
        (x.clone().requires_grad_(True), target),
        (x.to(device).requires_grad_(True), target.to(device)),
        1e-5, 1e-4,
    )

    target = torch.randint(0, 8, (8, 37)).t()
    parity(
        "ml strided target",
        lambda a, b: torch.ops.aten.multilabel_margin_loss_forward(a, b, 1),
        lambda a, b: torch.ops.aten.multilabel_margin_loss_forward(a, b, 1),
        (x.clone().requires_grad_(True), target),
        (x.to(device).requires_grad_(True), target.to(device)),
        1e-5, 1e-4,
    )


def run_direct_backward_cases(device):
    for n, d in [(8, 5), (64, 37), (16, 100)]:
        torch.manual_seed(n + d)
        x = torch.randn(n, d)
        target = torch.randint(0, d, (n,))
        weight = torch.rand(d) + 0.25
        for p in (1, 2):
            for reduction in ("none", "mean", "sum"):
                code = REDUCTIONS[reduction]
                seed = torch.rand(n) + 0.5 if code == 0 else torch.tensor(0.7)
                cpu = torch.ops.aten.multi_margin_loss_backward(
                    seed, x, target, p, 1.0, weight, code)
                mps = torch.ops.aten.multi_margin_loss_backward(
                    seed.to(device), x.to(device), target.to(device), p, 1.0,
                    weight.to(device), code)
                compare(f"mm backward {n}x{d} p{p} {reduction}",
                        mps, cpu, 1e-5, 1e-5)

        strided = torch.rand(n, 2)[:, 0]
        cpu = torch.ops.aten.multi_margin_loss_backward(
            strided, x, target, 1, 1.0, weight, 0)
        mps = torch.ops.aten.multi_margin_loss_backward(
            strided.to(device), x.to(device), target.to(device), 1, 1.0,
            weight.to(device), 0)
        compare(f"mm backward {n}x{d} strided grad_output", mps, cpu, 1e-5, 1e-5)

    for n, d in [(8, 5), (8, 37), (4, 100)]:
        target = multilabel_targets(n, d, 3, duplicates=True, empty_rows=True)
        torch.manual_seed(n * 13 + d)
        x = torch.randn(n, d)
        _, is_target = torch.ops.aten.multilabel_margin_loss_forward(x, target, 1)
        for reduction in ("none", "mean", "sum"):
            code = REDUCTIONS[reduction]
            seed = torch.rand(n) + 0.5 if code == 0 else torch.tensor(0.7)
            cpu = torch.ops.aten.multilabel_margin_loss_backward(
                seed, x, target, code, is_target)
            mps = torch.ops.aten.multilabel_margin_loss_backward(
                seed.to(device), x.to(device), target.to(device), code,
                is_target.to(device))
            compare(f"ml backward {n}x{d} {reduction}", mps, cpu, 1e-5, 1e-5)


def run_out_cases(device):
    torch.manual_seed(5)
    x = torch.randn(6, 7)
    target = torch.randint(0, 7, (6,))
    weight = torch.rand(7) + 0.5
    x_mps, target_mps, weight_mps = x.to(device), target.to(device), weight.to(device)

    expected = F.multi_margin_loss(x_mps, target_mps, weight=weight_mps)
    out = torch.empty(3, device=device)
    returned = torch.ops.aten.multi_margin_loss.out(
        x_mps, target_mps, 1, 1, weight_mps, 1, out=out)
    compare("mm out value", returned, expected, 0.0, exact=True)
    CHECKS["count"] += 1
    if returned.data_ptr() != out.data_ptr() or out.shape != expected.shape:
        FAILURES.append("mm out did not reuse/resize the out tensor")
        CHECKS["failed"] += 1

    seed = torch.tensor(0.5, device=device)
    grad_expected = torch.ops.aten.multi_margin_loss_backward(
        seed, x_mps, target_mps, 1, 1.0, weight_mps, 1)
    grad_out = torch.empty(0, device=device)
    returned = torch.ops.aten.multi_margin_loss_backward.grad_input(
        seed, x_mps, target_mps, 1, 1.0, weight_mps, 1, grad_input=grad_out)
    compare("mm grad_input value", returned, grad_expected, 0.0, exact=True)
    CHECKS["count"] += 1
    if returned.data_ptr() != grad_out.data_ptr():
        FAILURES.append("mm grad_input did not reuse the grad_input tensor")
        CHECKS["failed"] += 1

    mt = multilabel_targets(6, 7, 3, duplicates=True)
    expected, mask = torch.ops.aten.multilabel_margin_loss_forward(
        x_mps, mt.to(device), 1)
    output = torch.empty(2, device=device)
    mask_out = torch.empty(2, 2, device=device)
    got, got_mask = torch.ops.aten.multilabel_margin_loss_forward.output(
        x_mps, mt.to(device), 1, output=output, is_target=mask_out)
    compare("ml output value", got, expected, 0.0, exact=True)
    compare("ml is_target value", got_mask, mask, 0.0, exact=True)
    CHECKS["count"] += 1
    if got.data_ptr() != output.data_ptr() or got_mask.data_ptr() != mask_out.data_ptr():
        FAILURES.append("multilabel output did not reuse the out tensors")
        CHECKS["failed"] += 1

    grad_expected = torch.ops.aten.multilabel_margin_loss_backward(
        seed, x_mps, mt.to(device), 1, mask)
    grad_out = torch.empty(0, device=device)
    returned = torch.ops.aten.multilabel_margin_loss_backward.grad_input(
        seed, x_mps, mt.to(device), 1, mask, grad_input=grad_out)
    compare("ml grad_input value", returned, grad_expected, 0.0, exact=True)
    CHECKS["count"] += 1
    if returned.data_ptr() != grad_out.data_ptr():
        FAILURES.append("multilabel grad_input did not reuse the tensor")
        CHECKS["failed"] += 1


def run_dtype_cases(device):
    ext = {
        "float16": {"max_abs": 0.0, "max_rel": 0.0, "tol": 2e-2},
        "bfloat16": {"max_abs": 0.0, "max_rel": 0.0, "tol": 1e-1},
    }
    torch.manual_seed(11)
    # A quarter grid keeps every margin gap far from zero, so the "z > 0"
    # predicate agrees between the float32 and the half-precision arithmetic;
    # otherwise one near-zero gap flips a gradient element by a whole scale
    # unit and no tolerance distinguishes that from a broken kernel.
    x = torch.randn(16, 40).mul(4).round().div_(4)
    target = torch.randint(0, 40, (16,))
    weight = torch.rand(40) + 0.5
    reference = F.multi_margin_loss(x, target, p=2, weight=weight)
    x_reference = x.clone().requires_grad_(True)
    F.multi_margin_loss(x_reference, target, p=2, weight=weight).backward()
    target_mps, weight_mps = target.to(device), weight.to(device)
    for dtype, atol in ((torch.float16, 2e-2), (torch.bfloat16, 1e-1)):
        grad_atol = 1e-4 if dtype == torch.float16 else 4e-4
        x_mps = x.to(device, dtype).requires_grad_(True)
        got = F.multi_margin_loss(
            x_mps, target_mps, p=2, weight=weight_mps.to(dtype))
        CHECKS["count"] += 1
        if got.dtype != dtype or not torch.isfinite(got):
            FAILURES.append(f"mm {dtype} output dtype/isfinite")
            CHECKS["failed"] += 1
        compare(f"mm {dtype} vs float32", got, reference, atol, atol,
                bucket=ext[str(dtype).split(".")[-1]])
        got.backward()
        CHECKS["count"] += 1
        if x_mps.grad.dtype != dtype or not torch.isfinite(x_mps.grad).all():
            FAILURES.append(f"mm {dtype} gradient dtype/isfinite")
            CHECKS["failed"] += 1
        compare(f"mm {dtype} grad vs float32", x_mps.grad, x_reference.grad,
                grad_atol, atol,
                bucket=ext[str(dtype).split(".")[-1]])

    mt = multilabel_targets(16, 40, 3, duplicates=True)
    reference, mask_reference = torch.ops.aten.multilabel_margin_loss_forward(x, mt, 1)
    x_reference = x.clone().requires_grad_(True)
    torch.ops.aten.multilabel_margin_loss_forward(x_reference, mt, 1)[0].backward()
    for dtype, atol in ((torch.float16, 2e-2), (torch.bfloat16, 1e-1)):
        grad_atol = 1e-4 if dtype == torch.float16 else 4e-4
        x_mps = x.to(device, dtype).requires_grad_(True)
        got, mask = torch.ops.aten.multilabel_margin_loss_forward(
            x_mps, mt.to(device), 1)
        CHECKS["count"] += 1
        if got.dtype != dtype or mask.dtype != dtype or not torch.isfinite(got):
            FAILURES.append(f"ml {dtype} output dtype/isfinite")
            CHECKS["failed"] += 1
        compare(f"ml {dtype} vs float32", got, reference, atol, atol,
                bucket=ext[str(dtype).split(".")[-1]])
        compare(f"ml {dtype} is_target", mask, mask_reference, 0.0, exact=True)
        got.backward()
        CHECKS["count"] += 1
        if x_mps.grad.dtype != dtype or not torch.isfinite(x_mps.grad).all():
            FAILURES.append(f"ml {dtype} gradient dtype/isfinite")
            CHECKS["failed"] += 1
        compare(f"ml {dtype} grad vs float32", x_mps.grad, x_reference.grad,
                grad_atol, atol,
                bucket=ext[str(dtype).split(".")[-1]])
    return ext


def run_path_cases(device):
    """Which fused kernel each dtype and shape takes, and how many launches."""
    torch.manual_seed(13)
    results = {}
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        name = str(dtype).split(".")[-1]
        for n, d in [(64, 10), (256, 10), (256, 100), (16, 1000)]:
            x = torch.randn(n, d, device=device, dtype=dtype).requires_grad_(True)
            t = torch.randint(0, d, (n,), device=device)
            for reduction in ("none", "mean", "sum"):
                case = f"mm {name} {n}x{d} {reduction}"
                CHECKS["count"] += 1
                margin_ops._TRACE = trace = []
                try:
                    value = F.multi_margin_loss(x, t, reduction=reduction)
                    value.backward(torch.ones_like(value))
                finally:
                    margin_ops._TRACE = None
                results[case] = trace
                suffix = {"float32": "", "float16": "_h", "bfloat16": "_b"}[name]
                fused_total = (
                    reduction != "none"
                    and d <= margin_ops._ROW_DIM
                    and n * d <= margin_ops._TOTAL_ELEMENTS
                )
                want_rows = 1 if (reduction == "none" or fused_total) else 2
                # forward takes want_rows launches, the backward always one.
                if len(trace) != want_rows + 1:
                    FAILURES.append(
                        f"path {case}: {len(trace)} launches {trace}, expected "
                        f"{want_rows + 1}")
                    CHECKS["failed"] += 1
                if not all(entry.endswith(suffix) for entry in trace):
                    FAILURES.append(
                        f"path {case}: kernels {trace} not all {name}")
                    CHECKS["failed"] += 1

        for n, d, labels in [(64, 10, 3), (256, 100, 3), (16, 1000, 3)]:
            x = torch.randn(n, d, device=device, dtype=dtype).requires_grad_(True)
            t = multilabel_targets(n, d, labels, duplicates=True).to(device)
            for reduction in ("none", "mean", "sum"):
                case = f"ml {name} {n}x{d} l{labels} {reduction}"
                CHECKS["count"] += 1
                margin_ops._TRACE = trace = []
                try:
                    value, _ = torch.ops.aten.multilabel_margin_loss_forward(
                        x, t, REDUCTIONS[reduction])
                    value.backward(torch.ones_like(value))
                finally:
                    margin_ops._TRACE = None
                results[case] = trace
                fused_total = (
                    reduction != "none"
                    and n >= margin_ops._TOTAL_MIN_ROWS
                    and n * d <= margin_ops._TOTAL_ELEMENTS
                )
                want = 1 if (reduction == "none" or fused_total) else 2
                if len(trace) != want + 1:
                    FAILURES.append(
                        f"path {case}: {len(trace)} launches {trace}, expected "
                        f"{want + 1}")
                    CHECKS["failed"] += 1
                if not all(entry.endswith(suffix) for entry in trace):
                    FAILURES.append(
                        f"path {case}: kernels {trace} not all {name}")
                    CHECKS["failed"] += 1
    return results


def run_error_cases(device):
    x = torch.randn(4, 5, device=device)
    target = torch.tensor([0, 1, 2, 3], device=device)
    cases = [
        ("p=1.5", ValueError, lambda: F.multi_margin_loss(
            x, target, p=1.5)),
        ("p=3", ValueError, lambda: F.multi_margin_loss(x, target, p=3)),
        ("int target", RuntimeError, lambda: F.multi_margin_loss(
            x, target.to(torch.int32))),
        ("bad weight", RuntimeError, lambda: F.multi_margin_loss(
            x, target, weight=torch.ones(3, device=device))),
        ("3-D input", RuntimeError, lambda: F.multi_margin_loss(
            x.reshape(1, 4, 5), target)),
        ("bad multilabel shape", RuntimeError, lambda: F.multilabel_margin_loss(
            x, torch.zeros(4, 4, dtype=torch.int64, device=device))),
    ]
    results = {}
    for name, expected, call in cases:
        CHECKS["count"] += 1
        try:
            call()
        except expected:
            results[name] = expected.__name__
        except Exception as error:  # noqa: BLE001 - report the wrong type
            results[name] = f"unexpected {type(error).__name__}"
            CHECKS["failed"] += 1
            FAILURES.append(f"error case {name}: raised {type(error).__name__}")
        else:
            results[name] = "no error"
            CHECKS["failed"] += 1
            FAILURES.append(f"error case {name}: did not raise")
    return results


def run_optimizer_case(device):
    torch.manual_seed(17)
    init = torch.randn(5, 8)
    inputs = torch.randn(32, 8)
    target = torch.randint(0, 5, (32,))

    def train(dev):
        torch.manual_seed(3)
        linear = torch.nn.Linear(8, 5)
        linear.weight.data.copy_(init)
        linear.bias.data.zero_()
        linear.to(dev)
        loss_fn = torch.nn.MultiMarginLoss(p=2, margin=1.0)
        optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)
        x = inputs.to(dev)
        t = target.to(dev)
        first = None
        for _ in range(4):
            optimizer.zero_grad()
            loss = loss_fn(linear(x), t)
            loss.backward()
            optimizer.step()
            if first is None:
                first = loss.item()
        return first, loss.item(), linear.weight.detach().cpu().clone()

    cpu_first, cpu_last, cpu_weight = train("cpu")
    mps_first, mps_last, mps_weight = train(device)
    compare("optimizer first loss", torch.tensor(mps_first), torch.tensor(cpu_first), 1e-5)
    compare("optimizer final loss", torch.tensor(mps_last), torch.tensor(cpu_last), 1e-5)
    compare("optimizer weight", mps_weight, cpu_weight, 1e-5, 1e-5)
    moved = not torch.equal(mps_weight, init)
    CHECKS["count"] += 1
    if not moved:
        FAILURES.append("optimizer did not move the weights")
        CHECKS["failed"] += 1

    mt = multilabel_targets(16, 6, 2, duplicates=True)

    def train_multilabel(dev):
        torch.manual_seed(5)
        linear = torch.nn.Linear(6, 6)
        linear.to(dev)
        loss_fn = torch.nn.MultiLabelMarginLoss()
        optimizer = torch.optim.SGD(linear.parameters(), lr=0.05)
        x = torch.randn(16, 6).to(dev)
        t = mt.to(dev)
        optimizer.zero_grad()
        loss = loss_fn(linear(x), t)
        loss.backward()
        optimizer.step()
        return loss.item(), linear.weight.detach().cpu().clone()

    torch.manual_seed(23)
    torch.randn(16, 6)
    cpu_loss, cpu_ml_weight = train_multilabel("cpu")
    torch.manual_seed(23)
    torch.randn(16, 6)
    mps_loss, mps_ml_weight = train_multilabel(device)
    compare("multilabel module loss", torch.tensor(mps_loss), torch.tensor(cpu_loss),
            1e-5, 1e-5)
    compare("multilabel module weight", mps_ml_weight, cpu_ml_weight, 1e-5, 1e-5)

    return {
        "first_loss_cpu": cpu_first, "first_loss_mps": mps_first,
        "final_loss_cpu": cpu_last, "final_loss_mps": mps_last,
        "weights_moved": moved,
        "multilabel_module_loss_cpu": cpu_loss,
        "multilabel_module_loss_mps": mps_loss,
    }


def timed_cpu(fn, reps=30, inner=20, warmup=5):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        for _ in range(inner):
            fn()
        samples.append((time.perf_counter() - start) / inner * 1e6)
    return statistics.median(samples), min(samples)


def timed_mps(fn, reps=30, inner=0, warmup=10):
    """Queued and synchronized cost per call.

    With ``inner`` left at 0 the batch is sized to about 5 ms of queued work,
    so a background workload inflates the sample instead of a single call.
    """
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    if inner <= 0:
        start = time.perf_counter()
        fn()
        torch.mps.synchronize()
        inner = max(1, min(200, int(0.005 / max(time.perf_counter() - start, 1e-6))))
    queued, synced = [], []
    for _ in range(reps):
        start = time.perf_counter()
        for _ in range(inner):
            fn()
        torch.mps.synchronize()
        queued.append((time.perf_counter() - start) / inner * 1e6)
        torch.mps.synchronize()
        start = time.perf_counter()
        fn()
        torch.mps.synchronize()
        synced.append((time.perf_counter() - start) * 1e6)
    return statistics.median(queued), min(queued), statistics.median(synced)


def measure(device):
    results = []

    def record(name, cpu_fn, mps_fn):
        cpu, cpu_min = timed_cpu(cpu_fn)
        queued, queued_min, synced = timed_mps(mps_fn)
        results.append({
            "case": name,
            "cpu_us": round(cpu, 2),
            "cpu_us_min": round(cpu_min, 2),
            "mps_queued_us": round(queued, 2),
            "mps_queued_us_min": round(queued_min, 2),
            "mps_synced_us": round(synced, 2),
            "speedup": round(cpu / queued, 3),
        })
        print(f"  {name:34s} cpu {cpu:8.1f}  mps {queued:8.1f}  "
              f"synced {synced:8.1f}  x{cpu / queued:5.2f}")

    print("multi_margin forward")
    for n, d, weighted in [(64, 10, False), (4096, 10, False), (256, 100, True),
                           (256, 1000, False)]:
        x = torch.randn(n, d)
        t = torch.randint(0, d, (n,))
        w = torch.rand(d) + 0.5 if weighted else None
        xm, tm = x.to(device), t.to(device)
        wm = w.to(device) if w is not None else None
        record(f"mm fwd {n}x{d} w{int(weighted)}",
               lambda x=x, t=t, w=w: F.multi_margin_loss(x, t, weight=w),
               lambda x=xm, t=tm, w=wm: F.multi_margin_loss(x, t, weight=w))

    print("multi_margin forward+backward")
    for n, d in [(64, 10), (256, 100), (256, 1000)]:
        x = torch.randn(n, d)
        t = torch.randint(0, d, (n,))
        x_cpu = x.clone().requires_grad_(True)
        x_mps = x.to(device).requires_grad_(True)
        t_mps = t.to(device)

        def cpu_run():
            x_cpu.grad = None
            F.multi_margin_loss(x_cpu, t).backward()

        def mps_run():
            x_mps.grad = None
            F.multi_margin_loss(x_mps, t_mps).backward()

        record(f"mm fwd+bwd {n}x{d}", cpu_run, mps_run)

    print("multilabel forward")
    for n, d, labels in [(64, 10, 3), (256, 100, 3), (64, 1000, 3), (16, 1000, 20)]:
        x = torch.randn(n, d)
        t = multilabel_targets(n, d, labels, duplicates=True)
        xm, tm = x.to(device), t.to(device)
        record(f"ml fwd {n}x{d} l{labels}",
               lambda x=x, t=t: F.multilabel_margin_loss(x, t),
               lambda x=xm, t=tm: F.multilabel_margin_loss(x, t))

    print("multilabel forward+backward")
    for n, d, labels in [(64, 10, 3), (256, 100, 3), (64, 1000, 3)]:
        x = torch.randn(n, d)
        t = multilabel_targets(n, d, labels, duplicates=True)
        x_cpu = x.clone().requires_grad_(True)
        x_mps = x.to(device).requires_grad_(True)
        t_mps = t.to(device)

        def cpu_run():
            x_cpu.grad = None
            F.multilabel_margin_loss(x_cpu, t).backward()

        def mps_run():
            x_mps.grad = None
            F.multilabel_margin_loss(x_mps, t_mps).backward()

        record(f"ml fwd+bwd {n}x{d} l{labels}", cpu_run, mps_run)

    return results


def no_metal(self, weight):
    """Dispatch switch that forces the aten composition on the MPS device."""
    return False


def round_time(fn, inner):
    start = time.perf_counter()
    for _ in range(inner):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - start) / inner * 1e6


def enqueue_us(fn, samples=40):
    """CPU-side cost of one call, timed from an empty queue to the enqueue.

    This is the part of a small batch that no amount of GPU load can inflate:
    the Python and dispatch work of getting the command onto the queue. A
    second launch adds it again, which is what a fused reduction saves.
    """
    best = float("inf")
    for _ in range(samples):
        torch.mps.synchronize()
        start = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - start) * 1e6)
    torch.mps.synchronize()
    return best


def measure_paths(device):
    """Fused against composed on the same tensors, alternating which runs first.

    One round per direction, so a slowdown that belongs to the machine rather
    than to either path lands on both halves of the pair.
    """
    cases = []
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        tag = str(dtype).split(".")[-1]
        for n, d, weighted in [(64, 10, False), (256, 100, True), (256, 1000, False)]:
            x = torch.randn(n, d).to(device, dtype)
            t = torch.randint(0, d, (n,)).to(device)
            w = (torch.rand(d) + 0.5).to(device, dtype) if weighted else None
            cases.append((
                f"mm {tag} fwd {n}x{d} w{int(weighted)}",
                lambda x=x, t=t, w=w: F.multi_margin_loss(x, t, weight=w),
            ))
        for n, d, labels in [(64, 10, 3), (256, 100, 3), (64, 1000, 3)]:
            x = torch.randn(n, d).to(device, dtype)
            t = multilabel_targets(n, d, labels, duplicates=True).to(device)
            cases.append((
                f"ml {tag} fwd {n}x{d} l{labels}",
                lambda x=x, t=t: F.multilabel_margin_loss(x, t),
            ))

    enabled = margin_ops._metal_ready
    rounds = 8
    results = []
    print("fused versus composed (microseconds, median)")
    try:
        for name, fn in cases:
            for _ in range(3):
                fn()
            margin_ops._metal_ready = no_metal
            for _ in range(3):
                fn()
            torch.mps.synchronize()
            # Size the batch to about 10 ms of queued work, so a background
            # workload lands on both halves of the pair instead of on one call.
            start = time.perf_counter()
            fn()
            torch.mps.synchronize()
            inner = max(1, min(50, int(0.01 / max(time.perf_counter() - start, 1e-6))))
            fused, composed = [], []
            ratios = []
            for index in range(rounds):
                fused_first = index % 2 == 0
                margin_ops._metal_ready = enabled if fused_first else no_metal
                first = round_time(fn, inner)
                margin_ops._metal_ready = no_metal if fused_first else enabled
                second = round_time(fn, inner)
                one, other = (first, second) if fused_first else (second, first)
                fused.append(one)
                composed.append(other)
                ratios.append(other / one)
            fused_us, composed_us = statistics.median(fused), statistics.median(composed)
            paired = statistics.median(ratios)
            results.append({
                "case": name,
                "calls_per_sample": inner,
                "fused_us": round(fused_us, 2),
                "fused_us_min": round(min(fused), 2),
                "composed_us": round(composed_us, 2),
                "composed_us_min": round(min(composed), 2),
                "speedup_vs_composed": round(paired, 2),
                "speedup_of_medians": round(composed_us / fused_us, 2),
                "speedup_min": round(min(composed) / min(fused), 2),
            })
            print(f"  {name:30s} fused {fused_us:9.2f}  composed {composed_us:9.2f}  "
                  f"x{paired:5.2f} paired")
    finally:
        margin_ops._metal_ready = enabled
    return results


def measure_launch_floor(device):
    """The cost of one kernel launch, and what a second launch adds.

    One launch and two launches are timed against each other round by round and
    the reported figure is the median of the per-round ratio: the box is
    shared, and a difference of a few microseconds does not survive a
    comparison of two numbers taken minutes apart.
    """
    library = margin_ops._library()
    scratch = torch.ones(1, device=device)
    scalar = torch.empty((), device=device)

    def one_launch():
        library.row_reduce(scratch, scalar, 1, 0, threads=128, group_size=128)

    floor, floor_min, _ = timed_mps(one_launch, reps=20)

    tiny = torch.randn(1, 1, device=device)
    tiny_target = torch.zeros(1, dtype=torch.int64, device=device)
    overlay, overlay_min, _ = timed_mps(
        lambda: F.multi_margin_loss(tiny, tiny_target, reduction="none"), reps=20)

    pairs = []
    for n, d in [(64, 10), (256, 10), (64, 32)]:
        x = torch.randn(n, d, device=device)
        target = torch.randint(0, d, (n,), device=device)
        fn = lambda x=x, target=target: F.multi_margin_loss(x, target)
        elements = margin_ops._TOTAL_ELEMENTS
        single, two, ratios = [], [], []
        inner = 5
        try:
            for index in range(8):
                margin_ops._TOTAL_ELEMENTS = elements if index % 2 == 0 else 0
                first = round_time(fn, inner)
                margin_ops._TOTAL_ELEMENTS = 0 if index % 2 == 0 else elements
                second = round_time(fn, inner)
                one, pair = (first, second) if index % 2 == 0 else (second, first)
                single.append(one)
                two.append(pair)
                ratios.append(pair / one)
        finally:
            margin_ops._TOTAL_ELEMENTS = elements
        # Same loop, adjacent samples: the difference between the two paths is
        # then immune to CPU load drifting across the case.
        ones, twos = [], []
        for _ in range(40):
            for value, bucket in ((elements, ones), (0, twos)):
                margin_ops._TOTAL_ELEMENTS = value
                torch.mps.synchronize()
                start = time.perf_counter()
                fn()
                bucket.append((time.perf_counter() - start) * 1e6)
        increment = statistics.median([t - o for o, t in zip(ones, twos)])
        one_enqueue = round(min(ones), 2)
        two_enqueue = round(min(twos), 2)
        margin_ops._TOTAL_ELEMENTS = elements
        pairs.append({
            "case": f"mm fwd {n}x{d} mean",
            "one_launch_us": round(statistics.median(single), 2),
            "two_launch_us": round(statistics.median(two), 2),
            "one_launch_us_min": round(min(single), 2),
            "two_launch_us_min": round(min(two), 2),
            "two_over_one": round(statistics.median(ratios), 2),
            "one_launch_enqueue_us": one_enqueue,
            "two_launch_enqueue_us": two_enqueue,
            "second_launch_enqueue_us": round(increment, 2),
        })
        print(f"  {n}x{d} one launch {statistics.median(single):7.2f}  "
              f"two launches {statistics.median(two):7.2f}  "
              f"ratio {statistics.median(ratios):5.2f}  "
              f"enqueue {one_enqueue:5.2f}/{two_enqueue:5.2f} us  "
              f"second {increment:5.2f} us")
    return {
        "single_kernel_launch_us": round(floor, 2),
        "single_kernel_launch_us_min": round(floor_min, 2),
        "one_element_overlay_us": round(overlay, 2),
        "one_element_overlay_us_min": round(overlay_min, 2),
        "bare_kernel_enqueue_us": round(enqueue_us(one_launch), 2),
        "one_element_overlay_enqueue_us": round(enqueue_us(
            lambda: F.multi_margin_loss(tiny, tiny_target, reduction="none")), 2),
        "one_launch_vs_two": pairs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=str(ROOT / "measured" / "margin_ops.json"))
    parser.add_argument("--skip-timing", action="store_true")
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available")
    device = "mps"
    report = {"environment": describe(),
              "registration": before_registration()}
    lease, lease_record = broker_lease()
    with lease as lease_id:
        if lease_record["acquired"]:
            lease_record["id"] = lease_id
        report["lease"] = lease_record

        started = time.perf_counter()
        install()
        compile_x = torch.randn(4, 8, device=device)
        compile_t = torch.randint(0, 8, (4,), device=device)
        F.multi_margin_loss(compile_x, compile_t)
        report["shader_compile_s"] = round(time.perf_counter() - started, 3)

        run_multi_margin_cases(device)
        run_multi_margin_edges(device)
        run_multilabel_cases(device)
        run_direct_backward_cases(device)
        run_out_cases(device)
        report["dtype_extensions"] = run_dtype_cases(device)
        report["kernel_paths"] = run_path_cases(device)
        report["error_paths"] = run_error_cases(device)
        report["optimizer"] = run_optimizer_case(device)

        if not args.skip_timing:
            print("launch floor (microseconds)")
            report["launch_floor"] = measure_launch_floor(device)
            print("timing (microseconds, median)")
            report["timings"] = measure(device)
            report["fused_vs_composed"] = measure_paths(device)

    report["correctness"] = {
        "checks": CHECKS["count"],
        "failed": CHECKS["failed"],
        "max_abs_error_float32": CHECKS["max_abs"],
        "max_rel_error_float32": CHECKS["max_rel"],
        "failures": FAILURES,
    }
    report["notes"] = [
        "float32, float16 and bfloat16 inputs of at most two dimensions take "
        "the fused Metal kernels; the aten-op composition is left for other "
        "dtypes, a weight tensor whose dtype differs from the input, and "
        "backward calls whose is_target mask dtype differs from the input.",
        "The half kernels accumulate in float32 and store once: each row value, "
        "each gradient element and the reduced scalar is rounded a single time "
        "from the float32 result.",
        "multi_margin rows of at most 32 classes run one thread per row; wider "
        "rows run one threadgroup per row.",
        "multilabel_margin_loss runs one threadgroup per row; the group holds "
        "32 threads for rows of at most 32 classes and 64 otherwise.",
        "A reduction on a batch whose nframe*dim fits 4096 elements (and, for "
        "multi_margin, whose rows are at most 32 classes wide) is produced by "
        "the forward kernel itself; multilabel also needs at least 8 rows, or "
        "the single threadgroup would be half idle. Larger batches write "
        "per-row partials and take a second reduce pass.",
        "kernel_paths records the launch sequence per dtype and shape, so a "
        "regression that silently adds a launch or drops to the composition "
        "fails the run instead of only costing time.",
        "launch_floor times a single trivial kernel launch and compares the "
        "one-launch reduction against the two-pass path on the same shape, "
        "round by round, and reports the median of the per-round ratio; a small "
        "case cannot go faster than that floor, whatever the kernel does. "
        "environment.load_average is the CPU load when the run started.",
        "launch_floor.*_enqueue_us is the CPU-side cost of one call, timed from "
        "an empty queue to the enqueue with the minimum of 40 samples. It is "
        "the one number a busy GPU cannot inflate, and it is what a second "
        "launch adds: on an idle device the queued cost of a small batch is "
        "this figure plus a microsecond or two of kernel time.",
        "The CPU kernels dispatch float32 and float64 only, so float16 and "
        "bfloat16 are extensions of this overlay and are compared with the "
        "float32 CPU result.",
        "Target entries are assumed in range, as on CPU. The fused kernels "
        "bounds-check out-of-range entries and contribute nothing for them; "
        "the composed path raises through tensor indexing. Valid inputs match "
        "the CPU kernel.",
        "The machine is shared. timings are medians and minima of 30 samples "
        "of 20 queued calls plus one synchronized call; fused_vs_composed "
        "times both paths on the same tensors and alternates which one runs "
        "first, so its ratio survives background load better than its "
        "absolute numbers. launch_floor.single_kernel_launch_us is the "
        "calibration for the run: near 2-3 us the machine was idle, in the "
        "tens or hundreds of us it was busy and every row is inflated by "
        "roughly that much. Compare rows inside one JSON, or ratios, not "
        "absolute times across runs.",
    ]
    output = Path(args.json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {output}")
    print(f"checks {CHECKS['count']}, failed {CHECKS['failed']}, "
          f"max abs {CHECKS['max_abs']:.3e}, max rel {CHECKS['max_rel']:.3e}")
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
