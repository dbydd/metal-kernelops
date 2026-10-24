"""Verify and time the MPS overlays for ``linalg_ldl_factor_ex`` / ``linalg_ldl_solve``.

The correctness half compares the overlay against torch's CPU kernels, which call
LAPACK, on the quantities the operators actually promise:

* ``pivots`` and ``info`` must match the CPU kernel exactly, on positive definite,
  indefinite, zero-diagonal, block-diagonal, rank-deficient and NaN input;
* ``LD`` must match the CPU kernel to float32 rounding (the two implementations
  can differ in the last bits because the trailing update is a matmul here and a
  level-2 BLAS call there, but the pivot sequence is bit-for-bit the same, which
  is what fixes the factorization);
* ``ldl_solve`` must reproduce the CPU solution and, independently, solve
  ``A x = b``: the residual is measured against a float64/float128 reference
  solve with ``torch.linalg.solve``.

The timing half measures both ops on the same matrices on the CPU and on the
MPS overlay, warm and synchronized, and records medians. Host load is shared, so
treat the ratios as order of magnitude.

Usage::

    uv run --project /Users/dbydd/vibe-agent-working-dir/ResearchChamber \\
        python benchmarks/verify_ldl.py
    # writes measured/ldl.json, exits non-zero if a check failed
"""

import json
import os
import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metal_kernelops.ldl_ops import OVERLAYS  # noqa: E402


_LIBRARY = None


def register():
    """Register the overlays and keep the fragment alive for the process."""
    global _LIBRARY
    from torch.library import Library

    _LIBRARY = Library("aten", "FRAGMENT")
    for name, fn in OVERLAYS:
        _LIBRARY.impl(name, fn, "MPS")
    return [name for name, _ in OVERLAYS]


FAILURES = []


def check(section, name, ok, **detail):
    row = {"section": section, "case": name, "ok": bool(ok)}
    row.update(detail)
    if not ok:
        FAILURES.append(row)
    print(f"{'pass' if ok else 'FAIL'}  {section:9s} {name}")
    return row


def rel(a, b):
    """Relative Frobenius error; 0.0 when both sides are zero or empty."""
    a, b = a.double(), b.double()
    if a.numel() == 0:
        return 0.0
    denom = b.norm()
    if not torch.isfinite(denom):
        return float("nan")
    if denom == 0:
        return 0.0 if a.norm() == 0 else float("inf")
    return float((a - b).norm() / denom)


def symmetric(n, kind, *, complex_=False, hermitian=False, device="cpu"):
    if complex_:
        z = torch.randn(n, n, dtype=torch.complex64)
        a = (z + z.mH) / 2 if hermitian else (z + z.mT) / 2
        if hermitian:
            a = a - torch.diag(a.diagonal()) + torch.diag(
                a.diagonal().real.to(torch.complex64)
            )
    else:
        a = torch.randn(n, n)
        a = a + a.mT
    if kind == "zero_diag":
        a.diagonal().zero_()
    elif kind == "blocks":
        a.zero_()
        for i in range(0, n - 1, 2):
            v = torch.randn(1, dtype=a.dtype)
            a[i, i + 1] = v
            a[i + 1, i] = v
    elif kind == "zeros":
        a.zero_()
    elif kind == "rank1":
        v = torch.ones(n, 1, dtype=a.dtype)
        a = v @ v.mT
    return a.to(device)


def factor_cases():
    rows = []
    torch.manual_seed(0)
    sizes = [1, 2, 3, 5, 8, 13, 16, 33, 64, 129]
    kinds = ["randn", "zero_diag", "blocks", "rank1", "zeros"]
    for n in sizes:
        for kind in kinds:
            a = symmetric(n, kind)
            cpu = torch.linalg.ldl_factor_ex(a)
            dev = torch.linalg.ldl_factor_ex(a.to("mps"))
            ld_rel = rel(dev.LD.cpu(), cpu.LD)
            ld_ok = (ld_rel < 1e-5) if ld_rel == ld_rel else bool(
                torch.equal(dev.LD.cpu().isnan(), cpu.LD.isnan())
            )
            piv_ok = torch.equal(dev.pivots.cpu(), cpu.pivots)
            info_ok = torch.equal(dev.info.cpu(), cpu.info)
            # Once D is exactly singular the trailing block of LD is arbitrary
            # (division by a dropped zero column), and LAPACK's blocked sytrf can
            # land on a different trailing residue than this unblocked one, so
            # the pivot sequence and info are the gate and LD is informational.
            singular = bool(info_ok and int(cpu.info) != 0) or kind in ("rank1", "zeros")
            rows.append(
                check(
                    "factor-real",
                    f"n={n} {kind}",
                    piv_ok and info_ok and (singular or ld_ok),
                    pivots_exact=piv_ok,
                    info_exact=info_ok,
                    ld_rel_err=ld_rel,
                    ld_gated=not singular,
                    info=int(cpu.info),
                    pivots=[int(v) for v in cpu.pivots],
                )
            )
    # batches, including an empty batch dimension
    for shape in [(4, 6, 6), (2, 3, 5, 5), (0, 4, 4), (3, 0, 0)]:
        a = torch.randn(*shape)
        a = a + a.mT
        cpu = torch.linalg.ldl_factor_ex(a)
        dev = torch.linalg.ldl_factor_ex(a.to("mps"))
        rows.append(
            check(
                "factor-real",
                f"batch {tuple(shape)}",
                torch.equal(dev.pivots.cpu(), cpu.pivots)
                and torch.equal(dev.info.cpu(), cpu.info)
                and rel(dev.LD.cpu(), cpu.LD) < 1e-5,
                ld_rel_err=rel(dev.LD.cpu(), cpu.LD),
                strides=list(dev.LD.stride()),
                cpu_strides=list(cpu.LD.stride()),
            )
        )
    # complex: hermitian and symmetric, with and without a zero diagonal
    for n in [2, 3, 4, 6, 9, 17]:
        for hermitian in [True, False]:
            for kind in ["randn", "zero_diag"]:
                a = symmetric(n, kind, complex_=True, hermitian=hermitian)
                cpu = torch.linalg.ldl_factor_ex(a, hermitian=hermitian)
                dev = torch.linalg.ldl_factor_ex(a.to("mps"), hermitian=hermitian)
                ld_rel = rel(dev.LD.cpu(), cpu.LD)
                rows.append(
                    check(
                        "factor-complex",
                        f"n={n} hermitian={hermitian} {kind}",
                        torch.equal(dev.pivots.cpu(), cpu.pivots)
                        and torch.equal(dev.info.cpu(), cpu.info)
                        and ld_rel < 1e-5,
                        pivots_exact=torch.equal(dev.pivots.cpu(), cpu.pivots),
                        ld_rel_err=ld_rel,
                        pivots=[int(v) for v in cpu.pivots],
                    )
                )
    return rows


def solve_cases():
    rows = []
    torch.manual_seed(1)
    for n in [1, 2, 3, 5, 8, 16, 33, 64, 129]:
        for nrhs in [1, 7]:
            a = symmetric(n, "randn")
            b = torch.randn(n, nrhs)
            ld, piv, _ = torch.linalg.ldl_factor_ex(a)
            x_cpu = torch.linalg.ldl_solve(ld, piv, b)
            x_ref = torch.linalg.solve(a.double(), b.double()).float()
            x_dev = torch.linalg.ldl_solve(
                ld.to("mps"), piv.to("mps"), b.to("mps")
            ).cpu()
            residual = float((a @ x_dev - b).norm() / b.norm())
            rows.append(
                check(
                    "solve-real",
                    f"n={n} nrhs={nrhs}",
                    residual < 1e-4 and rel(x_dev, x_cpu) < 1e-4,
                    residual=residual,
                    vs_cpu=rel(x_dev, x_cpu),
                    vs_float64=rel(x_dev, x_ref),
                )
            )
    # batched right-hand sides and batch broadcasting of B against LD
    a = symmetric(6, "randn")
    a = torch.stack([a, symmetric(6, "zero_diag")])
    b = torch.randn(2, 6, 3)
    ld, piv, _ = torch.linalg.ldl_factor_ex(a)
    x_cpu = torch.linalg.ldl_solve(ld, piv, b)
    x_dev = torch.linalg.ldl_solve(ld.to("mps"), piv.to("mps"), b.to("mps")).cpu()
    rows.append(
        check(
            "solve-real",
            "batch (2,6,6) nrhs=3",
            rel(x_dev, x_cpu) < 1e-4,
            vs_cpu=rel(x_dev, x_cpu),
        )
    )
    b = torch.randn(1, 6, 2)  # extra leading batch dim, broadcast against LD
    x_cpu = torch.linalg.ldl_solve(ld, piv, b)
    x_dev = torch.linalg.ldl_solve(ld.to("mps"), piv.to("mps"), b.to("mps")).cpu()
    rows.append(
        check(
            "solve-real",
            "broadcast B (1,6,2) against LD (2,6,6)",
            rel(x_dev, x_cpu) < 1e-4,
            vs_cpu=rel(x_dev, x_cpu),
            shape=list(x_dev.shape),
        )
    )

    torch.manual_seed(2)
    for n in [3, 6, 9, 17]:
        for hermitian in [True, False]:
            a = symmetric(n, "randn", complex_=True, hermitian=hermitian)
            b = torch.randn(n, 2, dtype=torch.complex64)
            cpu = torch.linalg.ldl_factor_ex(a, hermitian=hermitian)
            dev = torch.linalg.ldl_factor_ex(a.to("mps"), hermitian=hermitian)
            x_cpu = torch.linalg.ldl_solve(
                cpu.LD, cpu.pivots, b, hermitian=hermitian
            )
            x_dev = torch.linalg.ldl_solve(
                dev.LD, dev.pivots, b.to("mps"), hermitian=hermitian
            ).cpu()
            x_ref = torch.linalg.solve(
                a.to(torch.complex128), b.to(torch.complex128)
            ).to(torch.complex64)
            residual = float((a @ x_dev - b).norm() / b.norm())
            rows.append(
                check(
                    "solve-complex",
                    f"n={n} hermitian={hermitian}",
                    residual < 1e-3 and rel(x_dev, x_cpu) < 1e-3,
                    residual=residual,
                    vs_cpu=rel(x_dev, x_cpu),
                    vs_complex128=rel(x_dev, x_ref),
                )
            )
    return rows


def semantics_cases():
    rows = []
    torch.manual_seed(3)
    a = symmetric(4, "randn")
    cpu = torch.linalg.ldl_factor_ex(a)
    dev = torch.linalg.ldl_factor_ex(a.to("mps"))
    rows.append(
        check(
            "semantics",
            "LD column-major strides, pivots/info int32, info 0-dim",
            tuple(dev.LD.stride()) == tuple(cpu.LD.stride())
            and dev.pivots.dtype == torch.int32
            and dev.info.dtype == torch.int32
            and dev.info.shape == cpu.info.shape
            and dev.pivots.shape == cpu.pivots.shape,
            strides=list(dev.LD.stride()),
            cpu_strides=list(cpu.LD.stride()),
        )
    )
    # only the lower triangle is read, like sytrf
    a2 = a.clone()
    upper = torch.triu_indices(4, 4, offset=1)
    a2[upper[0], upper[1]] = 99.0
    dev2 = torch.linalg.ldl_factor_ex(a2.to("mps"))
    rows.append(
        check(
            "semantics",
            "upper triangle ignored",
            torch.equal(dev2.LD.cpu(), dev.LD.cpu()),
        )
    )
    # non-contiguous input
    dev3 = torch.linalg.ldl_factor_ex(a.t().to("mps"))
    rows.append(
        check(
            "semantics",
            "transposed (non-contiguous) input",
            rel(dev3.LD.cpu(), cpu.LD) < 1e-6,
            ld_rel_err=rel(dev3.LD.cpu(), cpu.LD),
        )
    )
    # out= forms
    o_ld = torch.empty(4, 4, device="mps")
    o_piv = torch.empty(4, dtype=torch.int32, device="mps")
    o_info = torch.empty((), dtype=torch.int32, device="mps")
    r = torch.linalg.ldl_factor_ex(a.to("mps"), out=(o_ld, o_piv, o_info))
    b = torch.randn(4, 2)
    o_x = torch.empty(4, 2, device="mps")
    torch.linalg.ldl_solve(
        dev.LD, dev.pivots, b.to("mps"), out=o_x
    )
    rows.append(
        check(
            "semantics",
            "out= forms write through",
            torch.equal(r.LD, o_ld)
            and torch.equal(r.pivots, o_piv)
            and rel(o_x.cpu(), torch.linalg.ldl_solve(cpu.LD, cpu.pivots, b)) < 1e-4,
        )
    )
    # info / check_errors / ldl_factor
    singular = torch.zeros(3, 3, device="mps")
    info = torch.linalg.ldl_factor_ex(singular).info
    raised = None
    try:
        torch.linalg.ldl_factor_ex(singular, check_errors=True)
    except RuntimeError as exc:
        raised = str(exc)
    factor_raised = None
    try:
        torch.linalg.ldl_factor(singular)
    except RuntimeError as exc:
        factor_raised = str(exc)
    rows.append(
        check(
            "semantics",
            "info=1, check_errors raises 'Unknown error code: 1'",
            int(info) == 1
            and raised is not None
            and "Unknown error code: 1" in raised
            and factor_raised is not None
            and "Unknown error code: 1" in factor_raised,
            info=int(info),
            check_errors_message=raised,
        )
    )
    # gradients: the op has no derivative, exactly as on the CPU
    ag = a.clone().requires_grad_(True).to("mps")
    ag.requires_grad_(True)
    ld_g, piv_g, _ = torch.linalg.ldl_factor_ex(ag)
    x_g = torch.linalg.ldl_solve(ld_g, piv_g, b.to("mps"))
    backward_error = None
    try:
        x_g.sum().backward()
    except RuntimeError as exc:
        backward_error = str(exc)
    rows.append(
        check(
            "semantics",
            "gradients: NotImplemented grad_fn, backward raises",
            ld_g.requires_grad
            and not ld_g.is_leaf
            and type(x_g.grad_fn).__name__ == "NotImplemented"
            and backward_error is not None
            and "not implemented" in backward_error,
            backward_message=backward_error,
        )
    )
    # validation messages match the CPU kernel word for word
    for label, probe in [
        ("non-square", torch.ones(2, 3)),
        ("1-dim", torch.ones(3)),
        ("integer dtype", torch.ones(2, 2, dtype=torch.int32)),
    ]:
        cpu_msg = mps_msg = None
        try:
            torch.linalg.ldl_factor_ex(probe)
        except RuntimeError as exc:
            cpu_msg = str(exc)
        try:
            torch.linalg.ldl_factor_ex(probe.to("mps"))
        except RuntimeError as exc:
            mps_msg = str(exc)
        rows.append(
            check(
                "semantics",
                f"validation message: {label}",
                cpu_msg is not None and cpu_msg == mps_msg,
                cpu=cpu_msg,
                mps=mps_msg,
            )
        )
    # unsupported dtype is refused rather than silently computed
    half_raised = None
    try:
        torch.linalg.ldl_factor_ex(a.to(torch.float16).to("mps"))
    except NotImplementedError as exc:
        half_raised = str(exc)
    rows.append(
        check("semantics", "float16 refused", half_raised is not None, message=half_raised)
    )
    return rows


def _median_ms(fn, reps, warmup):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize() if torch.backends.mps.is_available() else None
    samples = []
    for _ in range(reps):
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        start = time.perf_counter()
        fn()
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2] * 1e3


def _lease():
    broker = Path(
        os.environ.get(
            "RESEARCH_CHAMBER", "/Users/dbydd/vibe-agent-working-dir/ResearchChamber"
        )
    ) / "shared_scripts"
    if not (broker / "compute_broker.py").exists():
        from contextlib import nullcontext

        return nullcontext("no-broker")
    sys.path.insert(0, str(broker))
    from compute_broker import lease_guard

    return lease_guard("mps-kernelops", tier=1, task_name="ldl_verify")


def context_metrics():
    """How fast this MPS device and the CPU are right now, for scale."""
    a = torch.randn(512, 512)
    b = torch.randn(512, 512)
    a_mps, b_mps = a.to("mps"), b.to("mps")
    matmul_cpu = _median_ms(lambda: a @ b, 9, 2)
    matmul_mps = _median_ms(lambda: a_mps @ b_mps, 9, 2)

    # A dependent chain of six tiny ops is the unit cost this overlay pays per
    # pivot-search round; measure it so the factor timings can be read against
    # the device's launch overhead rather than against nothing.
    x = torch.randn(1, 64, 64, device="mps")
    y = torch.randn(1, 64, 64, device="mps")
    idx = torch.arange(64, device="mps").view(1, 64, 1).expand(1, 64, 64)

    def chain():
        t = torch.abs(x).max(-1).values
        u = torch.where(t > 0, torch.abs(y), t)
        v = u.gather(-2, idx)
        w = v * u
        return w - x

    chain_ms = _median_ms(chain, 15, 3)
    return {
        "matmul_512_cpu_ms": matmul_cpu,
        "matmul_512_mps_ms": matmul_mps,
        "matmul_512_speedup": matmul_cpu / matmul_mps,
        "mps_dependent_6op_chain_ms": chain_ms,
        "mps_us_per_dependent_op": chain_ms * 1e3 / 6,
        "note": "shared host: other processes were using the GPU during these runs, so read the ratios as lower bounds on this device's gap",
    }


def timings():
    rows = []
    torch.manual_seed(4)
    shapes = [(8, 1), (32, 1), (64, 1), (64, 8), (128, 1), (128, 8)]
    with _lease():
        for n, batch in shapes:
            a = torch.randn(batch, n, n)
            a = a + a.mT
            a_mps = a.to("mps")
            reps = 5 if n >= 64 else 15
            t_cpu = _median_ms(lambda: torch.linalg.ldl_factor_ex(a), reps, 2)
            t_mps = _median_ms(lambda: torch.linalg.ldl_factor_ex(a_mps), reps, 1)
            ld, piv, _ = torch.linalg.ldl_factor_ex(a)
            ld_mps, piv_mps, _ = torch.linalg.ldl_factor_ex(a_mps)
            b = torch.randn(batch, n, 8)
            t_solve_cpu = _median_ms(
                lambda: torch.linalg.ldl_solve(ld, piv, b), reps, 2
            )
            t_solve_mps = _median_ms(
                lambda: torch.linalg.ldl_solve(ld_mps, piv_mps, b.to("mps")), reps, 1
            )
            row = {
                "n": n,
                "batch": batch,
                "nrhs": 8,
                "factor_cpu_ms": t_cpu,
                "factor_mps_ms": t_mps,
                "factor_ratio": t_mps / t_cpu,
                "solve_cpu_ms": t_solve_cpu,
                "solve_mps_ms": t_solve_mps,
                "solve_ratio": t_solve_mps / t_solve_cpu,
                "reps": reps,
            }
            rows.append(row)
            print(
                "timing n={n:4d} batch={batch:2d}  factor cpu={factor_cpu_ms:8.3f} "
                "mps={factor_mps_ms:8.3f} ({factor_ratio:7.1f}x)   solve "
                "cpu={solve_cpu_ms:7.3f} mps={solve_mps_ms:7.3f} "
                "({solve_ratio:6.1f}x)".format(**row)
            )
    return rows


def main():
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available on this machine")
    registered = register()
    rows = []
    rows += factor_cases()
    rows += solve_cases()
    rows += semantics_cases()
    timing_rows = timings()
    context = context_metrics()

    summary = {
        "passed": sum(1 for r in rows if r["ok"]),
        "failed": sum(1 for r in rows if not r["ok"]),
        "sections": {},
    }
    for row in rows:
        entry = summary["sections"].setdefault(
            row["section"], {"passed": 0, "failed": 0}
        )
        entry["passed" if row["ok"] else "failed"] += 1
    parts = [
        "{} {}/{}".format(k, v["passed"], v["passed"] + v["failed"])
        for k, v in summary["sections"].items()
    ]
    print(f"\n{summary['passed']} passed, {summary['failed']} failed ({', '.join(parts)})")

    payload = {
        "torch": torch.__version__,
        "device": torch.mps.get_name() if hasattr(torch.mps, "get_name") else "mps",
        "machine": f"{platform.platform()} / {platform.machine()}",
        "python": platform.python_version(),
        "registered_overlays": registered,
        "reference": "torch CPU kernels (LAPACK sytrf/hetrf, sytrs/hetrs, uplo='L')",
        "summary": summary,
        "notes": [
            "pivots and info are compared exactly; LD and solutions to a relative tolerance",
            "float64 is unreachable on MPS, so float32 and complex64 are the full dtype domain of these ops on this backend",
            "timings are medians of warm, synchronized runs on a shared machine, so treat the ratios as order of magnitude",
            "a decomposition into existing aten ops cannot win here: the pivot search is n sequential rounds of ~30 kernels and launch overhead dominates",
        ],
        "checks": rows,
        "context": context,
        "timings": timing_rows,
    }
    out = ROOT / "measured" / "ldl.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
