"""Verify and time the Householder MPS overlays.

The reference is ``torch.ormqr`` on the CPU, which is LAPACK ``dormqr``: the
reflectors are applied implicitly, exactly as ``ormqr`` defines them. No test
here reconstructs ``Q`` through ``torch.linalg.qr``. The second overlay,
``linalg_householder_product``, is checked against the same CPU routine of that
name, including ``k < n``.

Run:

    uv run --project /Users/dbydd/vibe-agent-working-dir/ResearchChamber python benchmarks/verify_householder.py

Every check must pass; the timings are appended to ``measured/householder.json``.
"""

import json
import platform
import statistics
import sys
import time
from datetime import date
from pathlib import Path

import torch
from torch.library import Library

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import metal_kernelops.householder_ops as householder_ops  # noqa: E402
from metal_kernelops.householder_ops import OVERLAYS, ormqr_mps  # noqa: E402

MEASURED = ROOT / "measured" / "householder.json"

ATOL = 5e-5
RTOL = 5e-5

failures = []
checks = []
_library = None


def check(name, condition, detail=""):
    print(("PASS " if condition else "FAIL ") + name + (f"  {detail}" if detail else ""))
    checks.append({"check": name, "pass": bool(condition), "detail": detail})
    if not condition:
        failures.append(name + (f": {detail}" if detail else ""))


def install():
    global _library
    _library = Library("aten", "FRAGMENT")
    for name, function in OVERLAYS:
        _library.impl(name, function, "MPS")
    return _library


def close(got, ref):
    return torch.allclose(got, ref, atol=ATOL, rtol=RTOL)


def max_error(got, ref):
    return (got.double() - ref.double()).abs().max().item()


def correctness():
    """Every case: same reflectors on CPU and MPS, compare the two outputs."""
    cases = []
    for m, n in [(8, 8), (12, 9), (9, 12), (33, 65), (128, 32), (300, 200)]:
        A = torch.randn(m, n)
        inp, tau = torch.geqrf(A)
        for nrhs in (1, 3, 8, 17):
            for left in (True, False):
                other = torch.randn(m, nrhs) if left else torch.randn(nrhs, m)
                for transpose in (False, True):
                    ref = torch.ormqr(inp, tau, other, left=left, transpose=transpose)
                    got = torch.ormqr(
                        inp.to("mps"), tau.to("mps"), other.to("mps"),
                        left=left, transpose=transpose,
                    )
                    cases.append(((m, n, nrhs, left, transpose), got.cpu(), ref))
    worst = max(max_error(got, ref) for _, got, ref in cases)
    check("unbatched left/right x transpose x nrhs", all(close(g, r) for _, g, r in cases),
          f"{len(cases)} cases, max abs error {worst:.2e}")

    # Batched reflectors and targets, two batch dimensions.
    A = torch.randn(2, 3, 24, 16)
    inp, tau = torch.geqrf(A)
    other = torch.randn(2, 3, 24, 5)
    for transpose in (False, True):
        ref = torch.ormqr(inp, tau, other, transpose=transpose)
        got = torch.ormqr(inp.to("mps"), tau.to("mps"), other.to("mps"), transpose=transpose)
        check(f"batched (2,3) left transpose={transpose}", close(got.cpu(), ref),
              f"max abs error {max_error(got.cpu(), ref):.2e}")
        ref_r = torch.ormqr(inp, tau, other.transpose(-1, -2).contiguous(), left=False,
                            transpose=transpose)
        got_r = torch.ormqr(inp.to("mps"), tau.to("mps"),
                            other.transpose(-1, -2).contiguous().to("mps"),
                            left=False, transpose=transpose)
        check(f"batched (2,3) right transpose={transpose}", close(got_r.cpu(), ref_r),
              f"max abs error {max_error(got_r.cpu(), ref_r):.2e}")

    # Non-contiguous target: a transposed view is handed to the kernel as-is.
    A = torch.randn(40, 20)
    inp, tau = torch.geqrf(A)
    other = torch.randn(7, 40).t()
    ref = torch.ormqr(inp, tau, other)
    got = torch.ormqr(inp.to("mps"), tau.to("mps"), other.to("mps"))
    check("non-contiguous other", close(got.cpu(), ref),
          f"max abs error {max_error(got.cpu(), ref):.2e}")

    # End to end on the device: geqrf and ormqr both through their MPS kernels.
    A = torch.randn(64, 48)
    inp_mps, tau_mps = torch.geqrf(A.to("mps"))
    other = torch.randn(64, 6)
    got = torch.ormqr(inp_mps, tau_mps, other.to("mps")).cpu()
    ref = torch.ormqr(*torch.geqrf(A), other)
    check("mps geqrf -> mps ormqr", close(got, ref),
          f"max abs error {max_error(got, ref):.2e}")

    # m above 4096 leaves threadgroup memory and takes the global-memory kernel.
    for m in (4096, 4200):
        A = torch.randn(m, 24)
        inp, tau = torch.geqrf(A)
        for left, other in (
            (True, torch.randn(m, 4)),
            (False, torch.randn(3, m)),
        ):
            ref = torch.ormqr(inp, tau, other, left=left)
            got = torch.ormqr(inp.to("mps"), tau.to("mps"), other.to("mps"), left=left)
            check(f"m={m} left={left} kernel path", close(got.cpu(), ref),
                  f"max abs error {max_error(got.cpu(), ref):.2e}")

    # out= routes through aten::ormqr.out and writes the caller's tensor.
    A = torch.randn(20, 12)
    inp, tau = torch.geqrf(A)
    other = torch.randn(20, 4)
    out = torch.empty(20, 4, device="mps")
    torch.ormqr(inp.to("mps"), tau.to("mps"), other.to("mps"), out=out)
    check("out= writes the provided tensor", close(out.cpu(), torch.ormqr(inp, tau, other)),
          f"max abs error {max_error(out.cpu(), torch.ormqr(inp, tau, other)):.2e}")

    # Gradients: the ATen autograd formula wraps the kernel because the overlay
    # sits on the MPS key, below AutogradMPS.
    A = torch.randn(30, 18)
    inp, tau = torch.geqrf(A)
    x_cpu = torch.randn(30, 5, requires_grad=True)
    x_mps = x_cpu.detach().clone().to("mps").requires_grad_()
    torch.ormqr(inp, tau, x_cpu).sum().backward()
    torch.ormqr(inp.to("mps"), tau.to("mps"), x_mps).sum().backward()
    check("gradient flows on MPS",
          x_mps.grad is not None and close(x_mps.grad.cpu(), x_cpu.grad),
          f"max abs error {max_error(x_mps.grad.cpu(), x_cpu.grad):.2e}")

    # Error contract matches the CPU kernel rather than silently differing.
    try:
        torch.ormqr(inp.to("mps"), tau.to("mps"), torch.randn(30, 5, device="mps"), left=False)
        check("left=False shape check", False, "no error raised")
    except RuntimeError as error:
        check("left=False shape check", "other.shape[-1] must be equal to input.shape[-2]" in str(error))
    try:
        torch.ormqr(inp.to("mps"), tau[:-1].to("mps"), torch.randn(30, 5, device="mps"))
        check("tau length check", False, "no error raised")
    except RuntimeError as error:
        check("tau length check", "tau.shape[-1] must be equal to" in str(error))

    # Direct call, used by the timing pass, agrees with the operator.
    A = torch.randn(50, 20)
    inp, tau = torch.geqrf(A)
    other = torch.randn(50, 3)
    got = ormqr_mps(inp.to("mps"), tau.to("mps"), other.to("mps"))
    check("ormqr_mps direct call", close(got.cpu(), torch.ormqr(inp, tau, other)),
          f"max abs error {max_error(got.cpu(), torch.ormqr(inp, tau, other)):.2e}")


def product_correctness():
    """``linalg_householder_product`` against the CPU kernel, including k < n."""
    cases = 0
    worst = 0.0
    for m, n, k in [
        (8, 6, 6), (8, 6, 3), (8, 6, 0), (12, 12, 12),
        (33, 20, 20), (33, 20, 7), (256, 128, 64), (512, 512, 512),
    ]:
        A = torch.randn(m, n)
        inp, tau = torch.geqrf(A)
        ref = torch.linalg.householder_product(inp, tau[:k])
        got = torch.linalg.householder_product(inp.to("mps"), tau[:k].to("mps"))
        cases += 1
        worst = max(worst, max_error(got.cpu(), ref))
        if not close(got.cpu(), ref):
            check(f"householder_product ({m},{n},k={k})", False,
                  f"max abs error {max_error(got.cpu(), ref):.2e}")
        if got.stride() != ref.stride():
            check(f"householder_product ({m},{n},k={k}) strides", False,
                  f"{got.stride()} vs {ref.stride()}")
    check("householder_product shapes and k <= n", True, f"{cases} cases, max abs error {worst:.2e}")

    A = torch.randn(2, 3, 24, 16)
    inp, tau = torch.geqrf(A)
    ref = torch.linalg.householder_product(inp, tau[:, :, :9])
    got = torch.linalg.householder_product(inp.to("mps"), tau[:, :, :9].to("mps"))
    check("householder_product batched (2,3) k<n", close(got.cpu(), ref) and got.stride() == ref.stride(),
          f"max abs error {max_error(got.cpu(), ref):.2e}")

    A = torch.randn(30, 18)
    inp, tau = torch.geqrf(A)
    x_cpu = inp.clone().requires_grad_()
    x_mps = inp.clone().to("mps").requires_grad_()
    torch.linalg.householder_product(x_cpu, tau).sum().backward()
    torch.linalg.householder_product(x_mps, tau.to("mps")).sum().backward()
    check("householder_product gradient on MPS",
          x_mps.grad is not None and close(x_mps.grad.cpu(), x_cpu.grad),
          f"max abs error {max_error(x_mps.grad.cpu(), x_cpu.grad):.2e}")

    try:
        torch.linalg.householder_product(torch.randn(4, 6, device="mps"), torch.randn(4, device="mps"))
        check("householder_product m>=n check", False, "no error raised")
    except RuntimeError as error:
        check("householder_product m>=n check",
              "input.shape[-2] must be greater than or equal to input.shape[-1]" in str(error))


def native_product_timing():
    """The native MPS kernel, timed before the overlay replaces it."""
    cases, rows = [], []
    for size in (128, 256, 512):
        A = torch.randn(size, size)
        inp, tau = torch.geqrf(A)
        inp_mps, tau_mps = inp.to("mps"), tau.to("mps")
        torch.linalg.householder_product(inp_mps, tau_mps)
        torch.mps.synchronize()
        start = time.perf_counter()
        torch.linalg.householder_product(inp_mps, tau_mps)
        torch.mps.synchronize()
        native_ms = (time.perf_counter() - start) * 1e3
        cpu_ms = median_time(
            lambda: torch.linalg.householder_product(inp, tau), repeats=5, warmup=1
        )
        cases.append((size, inp, tau))
        rows.append({"m": size, "n": size, "cpu_ms": round(cpu_ms, 4),
                     "native_mps_ms": round(native_ms, 3)})
        print(f"householder_product {size}x{size}: native mps={native_ms:9.3f}  cpu={cpu_ms:7.3f}")
    return cases, rows


def orgqr_composite():
    """``torch.orgqr`` is a composite over ``linalg_householder_product``."""
    A = torch.randn(256, 256)
    inp, tau = torch.geqrf(A)
    ref = torch.orgqr(inp, tau)
    got = torch.orgqr(inp.to("mps"), tau.to("mps"))
    check("orgqr composite reaches the overlay", close(got.cpu(), ref),
          f"max abs error {max_error(got.cpu(), ref):.2e}")
    inp_mps, tau_mps = inp.to("mps"), tau.to("mps")
    mps_ms = median_time(lambda: torch.orgqr(inp_mps, tau_mps), sync=True)
    cpu_ms = median_time(lambda: torch.orgqr(inp, tau))
    print(f"orgqr 256x256: mps={mps_ms:7.3f}  cpu={cpu_ms:7.3f}")
    return {"m": 256, "n": 256, "mps_ms": round(mps_ms, 4), "cpu_ms": round(cpu_ms, 4)}


def product_overlay_timing(cases, rows):
    for row, (size, inp, tau) in zip(rows, cases):
        inp_mps, tau_mps = inp.to("mps"), tau.to("mps")
        overlay_ms = median_time(lambda: torch.linalg.householder_product(inp_mps, tau_mps), sync=True)
        row["overlay_mps_ms"] = round(overlay_ms, 4)
        row["speedup_vs_native_mps"] = round(row["native_mps_ms"] / overlay_ms, 1)
        row["overlay_vs_cpu"] = round(row["cpu_ms"] / overlay_ms, 2)
        print(f"householder_product {size}x{size}: overlay={overlay_ms:8.3f}  "
              f"{row['speedup_vs_native_mps']:8.1f}x vs native")


def median_time(fn, repeats=20, warmup=5, sync=False):
    """Best of five warm median batches; this host is shared with other work."""
    for _ in range(warmup):
        fn()
    if sync:
        torch.mps.synchronize()
    best = float("inf")
    for _ in range(5):
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            if sync:
                torch.mps.synchronize()
            samples.append(time.perf_counter() - start)
        best = min(best, statistics.median(samples) * 1e3)
    return best


def queued_time(fn, repeats=10, warmup=3):
    """Marginal cost of one call when the queue stays busy, as in a model step."""
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        torch.mps.synchronize()
        best = min(best, (time.perf_counter() - start) / repeats * 1e3)
    return best


def stage_timing():
    """Split one call into its two launches: the block factors and the update.

    ``wy_factor`` computes the compact WY factor of every reflector block once
    per call, ``ormqr_wy`` (or ``ormqr_wy_reg`` for small ``m``) replays them
    for every column of the target. The split says which of the two to attack
    next: the factorisation is target independent, so it loses share as the
    target widens.
    """
    rows = []
    for m, n, nrhs in [(512, 512, 8), (1024, 1024, 8), (2048, 512, 8)]:
        A = torch.randn(m, n)
        inp, tau = torch.geqrf(A)
        inp_mps, tau_mps = inp.to("mps"), tau.to("mps")
        other_mps = torch.randn(m, nrhs, device="mps")
        extra, ib, group = householder_ops._shape(m)
        blocks = -(-n // ib)
        factors = torch.empty(blocks * ib * ib, dtype=torch.float32, device="mps")
        out = torch.empty_strided((1, m, nrhs), (m * nrhs, 1, m),
                                  dtype=torch.float32, device="mps")
        kernel = householder_ops._library(extra == 0 and m <= 4096, True, extra, ib)
        factor_ms = median_time(
            lambda: kernel.wy_factor(inp_mps, tau_mps, factors, m, n, n, blocks,
                                     int(True), threads=blocks * ib * ib,
                                     group_size=ib * ib),
            sync=True,
        )
        apply = kernel.ormqr_wy_reg if extra else kernel.ormqr_wy
        apply_ms = median_time(
            lambda: apply(inp_mps, factors, other_mps, out, m, n, n, nrhs, 1, 1,
                          blocks, threads=nrhs * group, group_size=group),
            sync=True,
        )
        row = {
            "m": m, "n": n, "nrhs": nrhs, "block": ib, "threadgroup": group,
            "rows_per_thread": extra,
            "factor_ms": round(factor_ms, 4),
            "apply_ms": round(apply_ms, 4),
        }
        rows.append(row)
        print(f"stages m={m:5d} nrhs={nrhs:3d}: factor={factor_ms:7.3f}  "
              f"apply={apply_ms:7.3f}  (block {ib}, group {group}, rows {extra})")
    return rows


def timing():
    rows = []
    shapes = [(256, 256), (512, 512), (1024, 1024), (1024, 256), (2048, 512)]
    for m, n in shapes:
        A = torch.randn(m, n)
        inp, tau = torch.geqrf(A)
        inp_mps, tau_mps = inp.to("mps"), tau.to("mps")
        for nrhs in (1, 8, 64):
            for left in (True, False):
                other = torch.randn(m, nrhs) if left else torch.randn(nrhs, m)
                other_mps = other.to("mps")
                cpu_ms = median_time(lambda: torch.ormqr(inp, tau, other, left=left))
                mps_ms = median_time(
                    lambda: torch.ormqr(inp_mps, tau_mps, other_mps, left=left), sync=True
                )
                mps_queued_ms = queued_time(
                    lambda: torch.ormqr(inp_mps, tau_mps, other_mps, left=left)
                )
                fallback_ms = median_time(
                    lambda: torch.ormqr(
                        inp_mps.cpu(), tau_mps.cpu(), other_mps.cpu(), left=left
                    ).to("mps"),
                    repeats=10, warmup=2, sync=True,
                )
                row = {
                    "m": m, "n": n, "nrhs": nrhs, "left": left,
                    "cpu_ms": round(cpu_ms, 4),
                    "mps_ms": round(mps_ms, 4),
                    "mps_queued_ms": round(mps_queued_ms, 4),
                    "cpu_fallback_with_transfers_ms": round(fallback_ms, 4),
                    "speedup_vs_cpu_fallback": round(fallback_ms / mps_ms, 2),
                    "mps_vs_cpu_compute": round(cpu_ms / mps_ms, 2),
                    "mps_queued_vs_cpu_compute": round(cpu_ms / mps_queued_ms, 2),
                }
                rows.append(row)
                print(
                    f"m={m:5d} n={n:4d} nrhs={nrhs:3d} left={str(left):5s} "
                    f"cpu={cpu_ms:7.3f}  mps={mps_ms:7.3f}  queued={mps_queued_ms:7.3f}  "
                    f"fallback={fallback_ms:7.3f}  mps/fallback={fallback_ms / mps_ms:5.2f}x  "
                    f"mps/cpu={cpu_ms / mps_ms:5.2f}x"
                )
    return rows


def main():
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available")

    # Importing the module must not register or compile anything.
    print(f"torch {torch.__version__}, mps available")
    check("import compiles no shader",
          householder_ops._library.cache_info().currsize == 0)
    try:
        torch.ormqr(*[t.to("mps") for t in torch.geqrf(torch.randn(8, 6))], torch.randn(8, 2, device="mps"))
        check("ormqr raises on MPS before install", False, "call succeeded without the overlay")
    except NotImplementedError:
        check("ormqr raises on MPS before install", True)

    product_cases, product_rows = native_product_timing()
    install()
    correctness()
    product_correctness()
    product_overlay_timing(product_cases, product_rows)
    orgqr_row = orgqr_composite()
    stage_rows = stage_timing()
    rows = timing()

    payload = {
        "module": "metal_kernelops.householder_ops",
        "operators": ["aten::ormqr", "aten::ormqr.out", "aten::linalg_householder_product"],
        "torch": torch.__version__,
        "device": "mps",
        "machine": f"{platform.machine()} {platform.platform()}",
        "measured": date.today().isoformat(),
        "note": (
            "This machine was shared with other agents while measuring. Each figure is the "
            "best of five batches of 20 warm runs, but single rows still move by up to 2x "
            "between runs. mps_ms includes the commit and wakeup of an idle stream; "
            "mps_queued_ms is the marginal cost with the queue kept busy, which is what a "
            "model step pays."
        ),
        "method": (
            "best of five batches of 20 warm runs, torch.mps.synchronize() around each MPS run; "
            "cpu is torch.ormqr on cpu tensors with no transfer; "
            "cpu_fallback_with_transfers_ms is the PYTORCH_ENABLE_MPS_FALLBACK path: "
            "move both operands to cpu, run, move the result back; "
            "native_mps_ms for householder_product is one warm call to the built-in MPS "
            "kernel, timed before the overlay is installed"
        ),
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "checks": checks,
        "ormqr": rows,
        "ormqr_stages": stage_rows,
        "householder_product": product_rows,
        "orgqr": orgqr_row,
        "failures": failures,
    }
    MEASURED.parent.mkdir(parents=True, exist_ok=True)
    MEASURED.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {MEASURED}")

    if failures:
        raise SystemExit(f"{len(failures)} check(s) failed")
    print("all checks passed")


if __name__ == "__main__":
    main()
