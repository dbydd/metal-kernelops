"""Verify and measure the RReLU MPS overlays in ``src/metal_kernelops/rrelu_ops.py``.

    uv run --project /Users/dbydd/vibe-agent-working-dir/ResearchChamber \
        python benchmarks/verify_rrelu.py

Registers the three overlays on a temporary ``aten`` fragment for this process
only, checks them against the CPU kernel and against the explicit formula, then
times them warm and synchronized. Writes ``measured/rrelu.json`` and exits
non-zero when a check fails.

The reference for training mode is ``where(self > 0, self, self * noise)`` with
``noise`` read back after the call: that is exactly what the CPU kernel
computes, and it is what the sample stored in ``noise`` has to satisfy. Eval
mode is ``leaky_relu(self, (lower + upper) / 2)`` with ``noise`` untouched.
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

ROOT = Path(__file__).resolve().parents[1]
BROKER = Path("/Users/dbydd/vibe-agent-working-dir/ResearchChamber/shared_scripts")
sys.path.insert(0, str(ROOT / "src"))

from metal_kernelops import rrelu_ops  # noqa: E402

DEVICE = "mps"
DTYPES = (torch.float32, torch.float16, torch.bfloat16)
SHAPES = ((4,), (32, 64), (3, 7, 13, 11), (2, 3, 4, 5, 6))
LOWER, UPPER = 0.125, 0.3
SLOPE = (LOWER + UPPER) / 2
BENCH_DTYPES = ((torch.float32, (1 << 20, 1 << 24, 16 * 64 * 112 * 112)),
                (torch.float16, (1 << 20, 1 << 22)),
                (torch.bfloat16, (1 << 20, 1 << 22)))


def reference(self, noise, training):
    if training:
        return torch.where(self > 0, self, self * noise)
    return torch.nn.functional.leaky_relu(self, SLOPE)


def aten_composition(self, noise, training):
    """What the overlay does when no fused kernel applies (non-contiguous input)."""
    if not training:
        return torch.nn.functional.leaky_relu(self, SLOPE)
    mask = self > 0
    noise.uniform_(LOWER, UPPER)
    noise.masked_fill_(mask, 1.0)
    return self * noise


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


def bounds(dtype):
    """The interval as that dtype stores it; uniform_ rounds both ends."""
    return float(torch.tensor(LOWER, dtype=dtype)), float(torch.tensor(UPPER, dtype=dtype))


def noise_contract(noise, mask, dtype):
    lo, hi = bounds(dtype)
    neg = ~mask
    ones = bool((noise[mask] == 1).all()) if bool(mask.any()) else True
    inside = bool((noise[neg] >= lo).all()) and bool((noise[neg] <= hi).all()) if bool(neg.any()) else True
    return ones and inside, {"positives_are_one": ones, "negatives_in_range": inside,
                             "lo": lo, "hi": hi}


def install():
    registered = []
    lib = Library("aten", "FRAGMENT")
    for name, function in rrelu_ops.OVERLAYS:
        operator = f"aten::{name}"
        replaced = torch._C._dispatch_has_kernel_for_dispatch_key(operator, "MPS")
        lib.impl(name, function, "MPS")
        registered.append({"aten": operator, "replaced_existing_kernel": bool(replaced),
                           "kernel": function.__name__})
    return registered, lib


@contextlib.contextmanager
def lease():
    """Hold a tier-1 lease while timing, when the lab's compute broker exists.

    The timing section is the only part that needs the GPU to itself; correctness
    checks run unleased because they assert values, not speed.
    """
    if not (BROKER / "compute_broker.py").exists():
        print("no compute broker here, timing unleased")
        yield
        return
    sys.path.insert(0, str(BROKER))
    from compute_broker import lease_guard

    with lease_guard("mps-kernelops", tier=1, task_name="verify_rrelu"):
        yield


def correctness(report):
    # Training: the output and the sample stored in noise both match the formula.
    for dtype in DTYPES:
        for shape in SHAPES:
            x = torch.randn(shape, device=DEVICE, dtype=dtype)
            noise = torch.empty_like(x)
            out = torch.ops.aten.rrelu_with_noise(x, noise, LOWER, UPPER, True)
            mask = x > 0
            report.check(f"train formula {tuple(shape)} {dtype}", torch.equal(out, reference(x, noise, True)))
            ok, detail = noise_contract(noise, mask, dtype)
            report.check(f"train noise contract {tuple(shape)} {dtype}", ok, detail)
            report.check(f"train out is new storage {tuple(shape)} {dtype}",
                         out.data_ptr() != x.data_ptr() and out.dtype == dtype and out.shape == x.shape)

    # Eval: bit-exact against leaky_relu, noise untouched, and against the CPU kernel.
    for dtype in DTYPES:
        for shape in SHAPES:
            x = torch.randn(shape, device=DEVICE, dtype=dtype)
            noise = torch.full_like(x, 7.0)
            out = torch.ops.aten.rrelu_with_noise(x, noise, LOWER, UPPER, False)
            report.check(f"eval formula {tuple(shape)} {dtype}",
                         torch.equal(out, torch.nn.functional.leaky_relu(x, SLOPE)))
            report.check(f"eval leaves noise alone {tuple(shape)} {dtype}",
                         torch.equal(noise, torch.full_like(x, 7.0)))

    # Eval against the CPU kernel. float32 agrees bit for bit. bfloat16 does not,
    # because MPS's leaky_relu kernel rounds the slope to bfloat16 before the
    # multiply and the CPU kernel keeps it in float; the overlay delegates to that
    # kernel, so it can only match it, not improve on it.
    for dtype in (torch.float32, torch.bfloat16):
        x = torch.randn(4096, device=DEVICE, dtype=dtype)
        noise = torch.empty_like(x)
        out = torch.ops.aten.rrelu_with_noise(x, noise, LOWER, UPPER, False)
        cpu = torch.ops.aten.rrelu_with_noise(x.cpu(), noise.cpu(), LOWER, UPPER, False)
        leaky = torch.nn.functional.leaky_relu(x, SLOPE)
        delta = float((out.cpu().float() - cpu.float()).abs().max())
        floor = float((leaky.cpu().float() - cpu.float()).abs().max())
        report.check(f"eval matches CPU kernel {dtype} up to the MPS leaky_relu floor",
                     torch.equal(out, leaky) and delta <= floor,
                     {"max_abs_diff": delta, "mps_leaky_relu_vs_cpu": floor})

    # In-place: same storage, self rewritten, noise carries the sample.
    for dtype in DTYPES:
        x = torch.randn(1 << 16, device=DEVICE, dtype=dtype)
        pristine = x.clone()
        noise = torch.empty_like(x)
        out = torch.ops.aten.rrelu_with_noise_(x, noise, LOWER, UPPER, True)
        report.check(f"inplace keeps storage {dtype}", out.data_ptr() == x.data_ptr())
        report.check(f"inplace formula {dtype}", torch.equal(x, reference(pristine, noise, True)))
        ok, detail = noise_contract(noise, pristine > 0, dtype)
        report.check(f"inplace noise contract {dtype}", ok, detail)

        x = torch.randn(1 << 16, device=DEVICE, dtype=dtype)
        pristine = x.clone()
        noise = torch.full_like(x, 7.0)
        torch.ops.aten.rrelu_with_noise_(x, noise, LOWER, UPPER, False)
        report.check(f"inplace eval formula {dtype}",
                     torch.equal(x, torch.nn.functional.leaky_relu(pristine, SLOPE)))
        report.check(f"inplace eval leaves noise alone {dtype}",
                     torch.equal(noise, torch.full_like(x, 7.0)))

    # Non-contiguous input takes the aten composition and stays on the device.
    for dtype in DTYPES:
        base = torch.randn(64, 96, device=DEVICE, dtype=dtype)
        x = base.t()
        noise = torch.empty_like(x)
        out = torch.ops.aten.rrelu_with_noise(x, noise, LOWER, UPPER, True)
        report.check(f"non-contiguous train {dtype}",
                     not x.is_contiguous() and torch.equal(out, reference(x, noise, True)))
        report.check(f"non-contiguous noise {dtype}", *noise_contract(noise, x > 0, dtype))

        x = base.t().clone().t()
        pristine = x.clone()
        noise = torch.empty_like(x)
        torch.ops.aten.rrelu_with_noise_(x, noise, LOWER, UPPER, True)
        report.check(f"non-contiguous inplace {dtype}", torch.equal(x, reference(pristine, noise, True)))

    xc = torch.randn(8, 16, 14, 14, device=DEVICE).to(memory_format=torch.channels_last)
    nc = torch.empty_like(xc)
    out = torch.ops.aten.rrelu_with_noise(xc, nc, LOWER, UPPER, True)
    report.check("channels_last train", torch.equal(out, reference(xc, nc, True)))

    # Empty tensor must not launch anything and must not crash.
    x = torch.empty(0, 5, device=DEVICE)
    noise = torch.empty_like(x)
    out = torch.ops.aten.rrelu_with_noise(x, noise, LOWER, UPPER, True)
    report.check("empty tensor", out.shape == x.shape and out.numel() == 0)

    # Generator: the MPS generator drives the draw, and it is reproducible.
    x = torch.randn(1 << 16, device=DEVICE)
    g1 = torch.Generator(device=DEVICE).manual_seed(1234)
    first = torch.ops.aten.rrelu_with_noise(x, torch.empty_like(x), LOWER, UPPER, True, g1)
    second = torch.ops.aten.rrelu_with_noise(x, torch.empty_like(x), LOWER, UPPER, True, g1)
    g2 = torch.Generator(device=DEVICE).manual_seed(1234)
    replay = torch.ops.aten.rrelu_with_noise(x, torch.empty_like(x), LOWER, UPPER, True, g2)
    report.check("generator replay", torch.equal(first, replay))
    report.check("generator advances", not torch.equal(first, second))
    torch.manual_seed(7)
    noise = torch.empty_like(x)
    a = torch.ops.aten.rrelu_with_noise(x, noise, LOWER, UPPER, True)
    torch.manual_seed(7)
    b = torch.ops.aten.rrelu_with_noise(x, torch.empty_like(x), LOWER, UPPER, True)
    report.check("manual_seed replay", torch.equal(a, b))

    # The sample is spread over the interval, not stuck at one value.
    neg = x <= 0
    lo, hi = bounds(torch.float32)
    mean = float(noise[neg].mean())
    report.check("sample mean near interval midpoint", abs(mean - (lo + hi) / 2) < 2e-3,
                 {"mean": mean, "expected": (lo + hi) / 2})

    # .out overload: writes out, keeps the sample, resizes, handles aliasing and strides.
    x = torch.randn(777, device=DEVICE)
    noise = torch.empty_like(x)
    out = torch.empty(777, device=DEVICE)
    got = torch.ops.aten.rrelu_with_noise.out(x, noise, LOWER, UPPER, True, out=out)
    report.check("out returns out", got.data_ptr() == out.data_ptr())
    report.check("out formula", torch.equal(out, reference(x, noise, True)))
    report.check("out noise contract", *noise_contract(noise, x > 0, torch.float32))

    small = torch.empty(3, device=DEVICE)
    got = torch.ops.aten.rrelu_with_noise.out(x, torch.empty_like(x), LOWER, UPPER, False, out=small)
    report.check("out resizes", tuple(got.shape) == (777,) and tuple(small.shape) == (777,)
                 and torch.equal(got, torch.nn.functional.leaky_relu(x, SLOPE)))

    gen = torch.Generator(device=DEVICE).manual_seed(99)
    aliased = torch.empty_like(x)
    got = torch.ops.aten.rrelu_with_noise.out(x, aliased, LOWER, UPPER, True, generator=gen, out=aliased)
    gen = torch.Generator(device=DEVICE).manual_seed(99)
    expected = torch.ops.aten.rrelu_with_noise(x, torch.empty_like(x), LOWER, UPPER, True, gen)
    report.check("out aliasing noise", torch.equal(got, expected))

    strided = torch.empty(777 * 2, device=DEVICE)[::2]
    gen = torch.Generator(device=DEVICE).manual_seed(5)
    n_strided = torch.empty_like(x)
    got = torch.ops.aten.rrelu_with_noise.out(x, n_strided, LOWER, UPPER, True, generator=gen, out=strided)
    gen = torch.Generator(device=DEVICE).manual_seed(5)
    n_expected = torch.empty_like(x)
    expected = torch.ops.aten.rrelu_with_noise(x, n_expected, LOWER, UPPER, True, gen)
    report.check("out non-contiguous", not strided.is_contiguous() and torch.equal(got, expected)
                 and torch.equal(n_strided, n_expected))

    # Error parity with the CPU kernel.
    x = torch.randn(5, device=DEVICE)
    cpu_message = mps_message = None
    try:
        torch.ops.aten.rrelu_with_noise(x.cpu(), torch.zeros(3), LOWER, UPPER, True)
    except RuntimeError as error:
        cpu_message = str(error)
    try:
        torch.ops.aten.rrelu_with_noise(x, torch.zeros(3, device=DEVICE), LOWER, UPPER, True)
    except RuntimeError as error:
        mps_message = str(error)
    report.check("noise shape error matches CPU", cpu_message == mps_message,
                 {"cpu": cpu_message, "mps": mps_message})
    try:
        torch.ops.aten.rrelu_with_noise(x, torch.zeros(5, dtype=torch.int64, device=DEVICE),
                                        LOWER, UPPER, True)
        raised = False
    except RuntimeError:
        raised = True
    report.check("noise dtype error", raised)

    # Registering on MPS must leave the CPU kernel alone.
    cpu_x, cpu_noise = torch.randn(1000), torch.empty(1000)
    cpu_out = torch.ops.aten.rrelu_with_noise(cpu_x, cpu_noise, LOWER, UPPER, True)
    report.check("cpu kernel untouched", torch.equal(cpu_out, reference(cpu_x, cpu_noise, True)))

    return report.rows


def gradients(report):
    for dtype in DTYPES:
        x = torch.randn(1 << 16, device=DEVICE, dtype=dtype, requires_grad=True)
        noise = torch.empty_like(x.detach())
        out = torch.ops.aten.rrelu_with_noise(x, noise, LOWER, UPPER, True)
        out.sum().backward()
        report.check(f"train gradient is the sample {dtype}", torch.equal(x.grad, noise))

        x = torch.randn(1 << 16, device=DEVICE, dtype=dtype, requires_grad=True)
        noise = torch.empty_like(x.detach())
        out = torch.ops.aten.rrelu_with_noise(x, noise, LOWER, UPPER, False)
        out.sum().backward()
        expected = torch.where(x.detach() > 0, torch.ones_like(x), torch.full_like(x, SLOPE))
        report.check(f"eval gradient is the slope {dtype}", torch.equal(x.grad, expected))

    # In-place on a non-leaf, both modes.
    for training in (True, False):
        leaf = torch.randn(1 << 16, device=DEVICE, requires_grad=True)
        x = leaf * 1.0
        noise = torch.empty_like(x.detach())
        out = torch.ops.aten.rrelu_with_noise_(x, noise, LOWER, UPPER, training)
        reference_grad = (noise if training
                          else torch.where(out.detach() > 0, torch.ones_like(out),
                                           torch.full_like(out, SLOPE)))
        out.sum().backward()
        report.check(f"inplace gradient training={training}", torch.equal(leaf.grad, reference_grad))

    # A weighted loss, so the gradient is not just a mask: relative error of the
    # global gradient norm against the same computation spelled out in torch.
    x = torch.randn(1 << 18, device=DEVICE, requires_grad=True)
    noise = torch.empty_like(x.detach())
    weight = torch.randn_like(x)
    (torch.ops.aten.rrelu_with_noise(x, noise, LOWER, UPPER, True) * weight).sum().backward()

    xr = x.detach().clone().requires_grad_(True)
    (torch.where(xr > 0, xr, xr * noise) * weight).sum().backward()
    mps_norm, ref_norm = float(x.grad.norm()), float(xr.grad.norm())
    report.check("gradient matches explicit reference",
                 torch.equal(x.grad, xr.grad),
                 {"mps_norm": mps_norm, "reference_norm": ref_norm,
                  "relative_error": abs(mps_norm - ref_norm) / ref_norm})

    # nn.RReLU through a forward/backward pass, plus an optimizer step.
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32), torch.nn.RReLU(LOWER, UPPER), torch.nn.Linear(32, 4)
    ).to(DEVICE)
    model.train()
    inputs = torch.randn(64, 16, device=DEVICE)
    target = torch.randn(64, 4, device=DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    losses = []
    before = [p.detach().clone() for p in model.parameters()]
    for _ in range(5):
        loss = torch.nn.functional.mse_loss(model(inputs), target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(round(float(loss.detach()), 6))

    parameters = list(model.parameters())
    nonzero = sum(1 for p in parameters if p.grad is not None and bool((p.grad != 0).any()))
    moved = sum(1 for p, old in zip(parameters, before) if not torch.equal(p.detach(), old))
    report.check("all parameters receive gradient", nonzero == len(parameters),
                 {"parameters": len(parameters), "nonzero_grad": nonzero})
    report.check("optimizer step moves every parameter", moved == len(parameters),
                 {"moved": moved, "parameters": len(parameters)})
    report.check("loss decreases", losses[-1] < losses[0], {"losses": losses})
    model.eval()
    with torch.no_grad():
        report.check("eval forward finite", bool(torch.isfinite(model(inputs)).all()))

    return {"losses": losses, "parameters": len(parameters), "nonzero_grad": nonzero,
            "moved": moved}


def _measure(entries, iters, rounds, warmup=3):
    """Per-call microseconds for several variants, timed round by round.

    The variants are interleaved so a shared machine's load hits each of them
    alike; a ratio between two of the numbers stays meaningful even when the
    absolute ones move.
    """
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


def timings():
    measured = {}
    for dtype, numels in BENCH_DTYPES:
        for numel in numels:
            label = f"{dtype} {numel}"
            x = torch.randn(numel, device=DEVICE, dtype=dtype)
            noise = torch.empty_like(x)
            inplace_x = torch.randn(numel, device=DEVICE, dtype=dtype)
            grad_out = torch.randn_like(x)
            leaf = x.detach().clone().requires_grad_(True)

            def step():
                out = torch.ops.aten.rrelu_with_noise(leaf, noise, LOWER, UPPER, True)
                out.sum().backward()
                leaf.grad = None

            gpu_entries = [
                ("mps_fused_train",
                 lambda: torch.ops.aten.rrelu_with_noise(x, noise, LOWER, UPPER, True)),
                ("mps_aten_composition_train", lambda: aten_composition(x, noise, True)),
                ("mps_fused_inplace_train",
                 lambda: torch.ops.aten.rrelu_with_noise_(inplace_x, noise, LOWER, UPPER, True)),
                ("mps_eval",
                 lambda: torch.ops.aten.rrelu_with_noise(x, noise, LOWER, UPPER, False)),
                ("mps_backward_train",
                 lambda: torch.ops.aten.rrelu_with_noise_backward(
                     grad_out, x, noise, LOWER, UPPER, True, False)),
                ("mps_backward_eval",
                 lambda: torch.ops.aten.rrelu_with_noise_backward(
                     grad_out, x, noise, LOWER, UPPER, False, False)),
                ("mps_forward_and_backward", step),
            ]
            cpu_entries = []
            if dtype != torch.float16:  # the CPU kernel has no Half path
                cpu_x = x.detach().cpu()
                cpu_noise = torch.empty_like(cpu_x)
                cpu_entries = [
                    ("cpu_train", lambda: torch.ops.aten.rrelu_with_noise(
                        cpu_x, cpu_noise, LOWER, UPPER, True)),
                    ("cpu_eval", lambda: torch.ops.aten.rrelu_with_noise(
                        cpu_x, cpu_noise, LOWER, UPPER, False)),
                ]

            small = numel <= 1 << 20
            # This host is shared with other builds; more rounds give the minimum a
            # better chance of landing in a quiet window.
            row = _measure(gpu_entries, iters=20 if small else 10, rounds=15)
            if cpu_entries:
                row.update(_measure(cpu_entries, iters=5 if small else 2,
                                    rounds=3 if small else 2, warmup=1))

            row["rounds"] = 15
            row["speedup_fused_over_aten_composition"] = round(
                row["mps_aten_composition_train"]["min"] / row["mps_fused_train"]["min"], 2)
            if "cpu_train" in row:
                row["speedup_mps_over_cpu_train"] = round(
                    row["cpu_train"]["min"] / row["mps_fused_train"]["min"], 2)
            measured[label] = row
            print(f"{label:26s} fused {row['mps_fused_train']['median']:9.1f} us  "
                  f"composition {row['mps_aten_composition_train']['median']:9.1f} us  "
                  f"eval {row['mps_eval']['median']:8.1f} us  "
                  f"fwd+bwd {row['mps_forward_and_backward']['median']:9.1f} us"
                  + (f"  cpu {row['cpu_train']['median']:9.1f} us" if "cpu_train" in row else ""))
    return measured


def main():
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available on this machine")
    torch.manual_seed(0)

    # The fragment dies with the Library object, so it has to outlive install().
    registered, _library = install()
    report = Report()
    checks = correctness(report)
    grad_summary = gradients(report)

    print("\nchecks")
    for row in checks:
        print(f"  {'ok  ' if row['ok'] else 'FAIL'} {row['check']}"
              + (f"  {row['detail']}" if not row["ok"] and "detail" in row else ""))

    print("\ntimings (warm, median of rounds, torch.mps.synchronize after each round)")
    load_before = round(os.getloadavg()[0], 2)
    with lease():
        measured = timings()
    load_after = round(os.getloadavg()[0], 2)

    payload = {
        "schema": 1,
        "module": "metal_kernelops/rrelu_ops.py",
        "torch": torch.__version__,
        "platform": f"{platform.platform()} {platform.machine()}",
        "device": torch.device(DEVICE).type,
        "operators": ["aten::rrelu_with_noise", "aten::rrelu_with_noise_",
                      "aten::rrelu_with_noise.out"],
        "overlays": registered,
        "backward": {
            "overlay": None,
            "reason": "aten::rrelu_with_noise_backward is CompositeExplicitAutograd and already "
                      "runs on MPS as one kernel (mul in training, leaky_relu_backward in eval); "
                      "a Python body would only add a dispatch frame",
        },
        "kernel_launches": {
            "training": 2,
            "training_aten_composition": 4,
            "eval": 1,
            "backward_training": 1,
            "backward_eval": 1,
        },
        "correctness": {
            "checks": len(checks),
            "failed": report.failed,
            "rows": checks,
        },
        "gradients": grad_summary,
        "timings_us": measured,
        "timings_method": "each variant runs 10 or 20 calls per round with one "
                          "torch.mps.synchronize at the end; variants are interleaved round by "
                          "round, so a load spike hits all of them; median and min over rounds are "
                          "recorded and the ratios use min",
        "timings_load_average": [load_before, load_after],
        "notes": [
            "training: noise is overwritten with the slope used, 1.0 where self > 0, so the "
            "backward is grad * noise",
            "eval: noise is untouched and the output is leaky_relu with (lower + upper) / 2 in "
            "double precision, as the CPU kernel computes it",
            "noise is non-differentiable",
            "the draw comes from MPS's uniform_, so the MPS generator (default or explicit) "
            "advances; the stream differs from the CPU kernel's",
            "non-contiguous inputs and dtypes without a Metal kernel take the 4-launch aten "
            "composition, still entirely on the GPU",
            "the CPU training path is dominated by the per-element draw from the CPU generator, "
            "not the arithmetic; compare cpu_train with cpu_eval at the same size",
            "the host is shared with other builds, so a round's median is inflated by contention; "
            "the minimum is the closer estimate of the kernel and the round-by-round ratios are the "
            "stable part of these numbers",
        ],
    }

    out = ROOT / "measured" / "rrelu.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}")
    print(f"{len(checks)} checks, {report.failed} failed")
    if report.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
