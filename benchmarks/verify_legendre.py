"""Verify and measure the MPS overlay for ``aten::special_legendre_polynomial_p``.

Run from the repository root::

    uv run --project /Users/dbydd/vibe-agent-working-dir/ResearchChamber \
        python benchmarks/verify_legendre.py

The script registers :data:`metal_kernelops.legendre_ops.OVERLAYS` on the MPS
dispatch key in this process only, then:

1. checks every one of the six aten overloads against the CPU kernel, element
   by element on the float32 bit pattern;
2. times the CPU kernel and the MPS kernel on the same inputs, warmed and
   synchronized, and writes ``measured/legendre.json``.

Exit status is 1 when any check fails.
"""

import json
import sys
import time
from pathlib import Path

import torch
from torch.library import Library

ROOT = Path(__file__).resolve().parent.parent
CHAMBER = Path("/Users/dbydd/vibe-agent-working-dir/ResearchChamber")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(CHAMBER))

from metal_kernelops import legendre_ops  # noqa: E402
from shared_scripts.compute_broker import lease_guard  # noqa: E402

MEASURED = ROOT / "measured" / "legendre.json"

leg = torch.special.legendre_polynomial_p
aten = torch.ops.aten.special_legendre_polynomial_p

_checks = []
_library = None


def install():
    """Register the overlays for this process; the handle must outlive the call."""
    global _library
    _library = Library("aten", "FRAGMENT")
    for name, kernel in legendre_ops.OVERLAYS:
        _library.impl(name, kernel, "MPS")


def check(label, got, want):
    """Compare an MPS result with a CPU result: shape, dtype, then bit pattern."""
    ok = got.shape == want.shape and got.dtype == want.dtype
    if ok:
        a, b = got.cpu(), want
        ok = torch.equal(a != a, b != b) and torch.equal(
            a.view(torch.int32)[a == a] if a.dtype == torch.float32 else a[a == a],
            b.view(torch.int32)[b == b] if b.dtype == torch.float32 else b[b == b],
        )
    _checks.append({"case": label, "elements": want.numel(), "bit_exact": ok})
    if not ok:
        print(f"FAIL {label}")
    return ok


def raises_same(label, mps_call, cpu_call):
    """Same exception type, and the same message apart from the kernel's name."""
    try:
        mps_call()
    except Exception as mps_error:
        try:
            cpu_call()
        except Exception as cpu_error:
            # The CPU kernel names itself in the message; the overlay names itself.
            ok = type(mps_error) is type(cpu_error) and (
                str(mps_error).split('"')[-1] == str(cpu_error).split('"')[-1]
            )
            _checks.append({"case": label, "elements": None, "bit_exact": ok})
            if not ok:
                print(f"FAIL {label}: {mps_error} vs {cpu_error}")
            return ok
        _checks.append({"case": label, "elements": None, "bit_exact": False})
        print(f"FAIL {label}: mps raised, cpu did not")
        return False
    _checks.append({"case": label, "elements": None, "bit_exact": False})
    print(f"FAIL {label}: mps did not raise")
    return False


def verify():
    ok = True
    generator = torch.Generator().manual_seed(20261024)

    x = torch.rand(2, 3, device="mps")
    degrees = torch.tensor([0.0, 1.0, 2.0], device="mps")
    cpu_x, cpu_degrees = x.cpu(), degrees.cpu()

    print("== overloads ==")
    ok &= check("Tensor", leg(x, degrees), leg(cpu_x, cpu_degrees))
    ok &= check("out", aten.out(x, degrees, out=torch.empty(2, 3, device="mps")),
                aten.out(cpu_x, cpu_degrees, out=torch.empty(2, 3)))
    ok &= check("x_scalar", leg(0.5, degrees), leg(0.5, cpu_degrees))
    ok &= check("n_scalar", leg(x, 2.0), leg(cpu_x, 2.0))
    ok &= check("x_scalar_out",
                aten.x_scalar_out(0.5, degrees, out=torch.empty(3, device="mps")),
                aten.x_scalar_out(0.5, cpu_degrees, out=torch.empty(3)))
    ok &= check("n_scalar_out",
                aten.n_scalar_out(x, 2.0, out=torch.empty(2, 3, device="mps")),
                aten.n_scalar_out(cpu_x, 2.0, out=torch.empty(2, 3)))
    ok &= check("integer degree", leg(x, 3), leg(cpu_x, 3))
    ok &= check("integer input scalar", leg(2, degrees), leg(2, cpu_degrees))
    ok &= check("bool input scalar", leg(True, degrees), leg(True, cpu_degrees))

    print("== recurrence, bit-exact against the CPU kernel ==")
    inside = torch.rand(50_000, generator=generator) * 2 - 1
    for degree in (0, 1, 2, 3, 5, 8, 17, 33, 64, 100):
        ok &= check(f"degree {degree} on [-1, 1]", leg(inside.to("mps"), float(degree)),
                    leg(inside, float(degree)))
    outside = torch.rand(20_000, generator=generator) * 6 - 3
    for degree in (2, 5, 12):
        ok &= check(f"degree {degree} on [-3, 3]", leg(outside.to("mps"), float(degree)),
                    leg(outside, float(degree)))
    mixed = torch.randint(0, 40, (40_000,), generator=generator).float()
    mixed_x = torch.rand(40_000, generator=generator) * 2 - 1
    ok &= check("mixed degrees in one call", leg(mixed_x.to("mps"), mixed.to("mps")),
                leg(mixed_x, mixed))
    small = torch.rand(49, generator=generator) * 2 - 1
    for degree in (0.5, 1.5, 2.7, -0.5, -3.0, 6.999):
        ok &= check(f"non-integer degree {degree}", leg(small.to("mps"), degree),
                    leg(small, degree))
    # A group of eight takes the vector recurrence only when all eight share a
    # degree and none is at |x| == 1, so both conditions are placed to straddle
    # a group boundary.
    for stride in (1, 3, 8):
        unit = (torch.rand(64, generator=generator) * 2 - 1)
        hits = list(range(0, 64, stride))
        unit[hits] = torch.tensor([1.0, -1.0] * len(hits))[:len(hits)]
        for degree in (3.0, 8.0, 33.0):
            ok &= check(f"|x| == 1 every {stride} at degree {degree}",
                        leg(unit.to("mps"), degree), leg(unit, degree))
    per_group = torch.tensor([3.0] * 8 + [3.0] * 7 + [4.0] + [4.0] + [3.0] * 7
                             + [3.0] * 32, device="mps")
    per_group_x = torch.rand(56, generator=generator) * 2 - 1
    ok &= check("one degree change per group", leg(per_group_x.to("mps"), per_group),
                leg(per_group_x, per_group.cpu()))
    ragged = torch.rand(70, generator=generator) * 2 - 1
    for count in (1, 7, 8, 9, 15, 17, 31, 33):
        ok &= check(f"{count} elements, partial group",
                    leg(ragged[:count].to("mps"), 6.0), leg(ragged[:count], 6.0))

    print("== special values ==")
    special = torch.tensor([0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0,
                            float("nan"), float("inf"), -float("inf")])
    for degree in (-4.0, -1.0, -0.5, 0.0, 1.0, 2.0, 3.0):
        ok &= check(f"special values at degree {degree}",
                    leg(special.to("mps"), degree), leg(special, degree))
    ok &= check("NaN degree", leg(torch.tensor([0.3, 2.0], device="mps"), float("nan")),
                leg(torch.tensor([0.3, 2.0]), float("nan")))
    ok &= check("-inf degree", leg(torch.tensor([0.3, 2.0], device="mps"), -float("inf")),
                leg(torch.tensor([0.3, 2.0]), -float("inf")))

    print("== dtypes ==")
    for dtype in (torch.int64, torch.int32, torch.int8, torch.bool):
        a = torch.ones(5, dtype=dtype, device="mps")
        b = torch.tensor([0, 1, 2, 3, 4], dtype=dtype, device="mps")
        ok &= check(f"{dtype} inputs", leg(a, b), leg(a.cpu(), b.cpu()))
    integral_degree = torch.tensor([1, 2, 3, 4, 5], device="mps")
    ok &= check("int64 degree", leg(x.new_tensor([0.1, 0.2, 0.3, 0.4, 0.5]), integral_degree),
                leg(x.new_tensor([0.1, 0.2, 0.3, 0.4, 0.5]).cpu(), integral_degree.cpu()))
    for dtype in (torch.float16, torch.bfloat16):
        ok &= raises_same(
            f"{dtype} rejected like cpu",
            lambda d=dtype: leg(torch.rand(3, dtype=d, device="mps"),
                                torch.tensor(2.0, dtype=d, device="mps")),
            lambda d=dtype: leg(torch.rand(3, dtype=d), torch.tensor(2.0, dtype=d)))
    ok &= raises_same(
        "complex inputs rejected like cpu",
        lambda: leg(torch.rand(3, dtype=torch.complex64, device="mps"), 2.0),
        lambda: leg(torch.rand(3, dtype=torch.complex64), 2.0))
    ok &= raises_same(
        "complex scalar rejected like cpu",
        lambda: leg(1 + 2j, x), lambda: leg(1 + 2j, cpu_x))

    print("== shapes ==")
    ok &= check("empty", leg(torch.empty(0, device="mps"), 3.0), leg(torch.empty(0), 3.0))
    ok &= check("2x0", leg(torch.empty(2, 0, device="mps"), 3.0),
                leg(torch.empty(2, 0), 3.0))
    ok &= check("0-dim", leg(torch.tensor(0.5, device="mps"), 3.0),
                leg(torch.tensor(0.5), 3.0))
    grid = torch.rand(64, 128, 4, device="mps")
    ok &= check("3-D", leg(grid, 7.0), leg(grid.cpu(), 7.0))
    column = torch.rand(4, 1, 1, device="mps")
    row = torch.rand(1, 3, 5, device="mps")
    ok &= check("broadcast x against n", leg(column, row), leg(column.cpu(), row.cpu()))
    transposed = torch.rand(4, 6, device="mps").t()
    ok &= check("non-contiguous x", leg(transposed, 3.0), leg(transposed.cpu(), 3.0))
    # A one-element slice starts at a 4-byte offset, which the float4 kernel
    # cannot take, so these exercise the one-element-per-thread path.
    unaligned = torch.rand(1025, device="mps")[1:]
    ok &= check("x starting at a 4-byte offset", leg(unaligned, 5.0),
                leg(unaligned.cpu(), 5.0))
    unaligned_degree = torch.full((1025,), 5.0, device="mps")[1:]
    ready = torch.rand(1025, device="mps")
    ok &= check("both operands offset", leg(ready[1:], unaligned_degree),
                leg(ready[1:].cpu(), unaligned_degree.cpu()))

    print("== out semantics ==")
    strided = torch.empty(3, 2, device="mps").t()
    result = aten.out(x, degrees, out=strided)
    ok &= result.data_ptr() == strided.data_ptr()
    ok &= check("strided out", result, aten.out(cpu_x, cpu_degrees, out=torch.empty(3, 2).t()))
    resized = aten.out(x, degrees, out=torch.empty(1, device="mps"))
    ok &= check("out resized to the broadcast shape", resized,
                aten.out(cpu_x, cpu_degrees, out=torch.empty(1)))
    half_out = torch.empty(2, 3, dtype=torch.float16, device="mps")
    ok &= check("float16 out", aten.out(x, degrees, out=half_out),
                aten.out(cpu_x, cpu_degrees, out=torch.empty(2, 3, dtype=torch.float16)))
    ok &= check("complex64 out", aten.out(x, degrees, out=torch.empty(
                    2, 3, dtype=torch.complex64, device="mps")),
                aten.out(cpu_x, cpu_degrees, out=torch.empty(2, 3, dtype=torch.complex64)))
    for dtype in (torch.int64, torch.int32, torch.bool):
        ok &= raises_same(
            f"unsafe cast to {dtype} rejected like cpu",
            lambda d=dtype: aten.out(x, degrees, out=torch.empty(2, 3, dtype=d, device="mps")),
            lambda d=dtype: aten.out(cpu_x, cpu_degrees, out=torch.empty(2, 3, dtype=d)))
    alias = torch.rand(4, device="mps")
    reference = alias.cpu().clone()
    returned = aten.out(alias, torch.ones(4, device="mps"), out=alias)
    ok &= returned.data_ptr() == alias.data_ptr()
    ok &= check("out aliasing the input", alias, leg(reference, torch.ones(4)))

    print("== autograd ==")
    tracked = torch.rand(4, device="mps", requires_grad=True)
    mps_out = leg(tracked, 2.0)
    cpu_out = leg(tracked.cpu(), 2.0)
    same = (mps_out.requires_grad, mps_out.grad_fn is None) == (
        cpu_out.requires_grad, cpu_out.grad_fn is None)
    _checks.append({"case": "requires_grad matches cpu", "elements": None, "bit_exact": same})
    if not same:
        print(f"FAIL autograd flag: {mps_out.requires_grad} vs {cpu_out.requires_grad}")
    ok &= same
    return ok


def _per_call_ms(call, repeat=10, trials=15, warmup=3):
    """Per-call milliseconds: the least-contended trial, repeats queued behind one sync."""
    for _ in range(warmup):
        call()
    torch.mps.synchronize()
    samples = []
    for _ in range(trials):
        start = time.perf_counter()
        for _ in range(repeat):
            call()
        torch.mps.synchronize()
        samples.append((time.perf_counter() - start) * 1e3 / repeat)
    return min(samples)


def measure():
    rows = []
    shapes = [(1 << 16,), (1 << 20,), (1 << 22,), (1024, 4096)]
    for degree in (4, 32):
        for shape in shapes:
            cpu_x = torch.rand(shape, generator=torch.Generator().manual_seed(3))
            mps_x = cpu_x.to("mps")
            cpu_ms = _per_call_ms(lambda: leg(cpu_x, float(degree)))
            mps_ms = _per_call_ms(lambda: leg(mps_x, float(degree)))
            mul_ms = _per_call_ms(lambda: mps_x * 2.0)
            rows.append({
                "shape": list(shape),
                "elements": cpu_x.numel(),
                "degree": degree,
                "cpu_ms": round(cpu_ms, 4),
                "mps_ms": round(mps_ms, 4),
                "speedup": round(cpu_ms / mps_ms, 2),
                "mps_mul_ms": round(mul_ms, 4),
            })
            print(f"  {tuple(shape)} n={degree}: cpu {cpu_ms:.3f} ms, mps {mps_ms:.3f} ms, "
                  f"{cpu_ms / mps_ms:.2f}x, elementwise mul on mps {mul_ms:.3f} ms")
    return rows


def bandwidth():
    """Achieved read-plus-write bandwidth, beside a same-shape copy as the floor.

    A copy moves the same bytes as ``out`` plus one float operand: 8 bytes per
    element. The kernel moves 8 bytes per element when the degree is a scalar,
    12 when it is a tensor.
    """
    rows = []
    for degree in (4, 32):
        for shape in ((1 << 20,), (1 << 22,)):
            mps_x = torch.rand(shape, device="mps")
            source = torch.rand(shape, device="mps")
            bytes_moved = mps_x.numel() * 8
            copy_ms = _per_call_ms(lambda: source.copy_(mps_x))
            ms = _per_call_ms(lambda: leg(mps_x, float(degree)))
            rows.append({
                "elements": mps_x.numel(),
                "degree": degree,
                "mps_ms": round(ms, 4),
                "gb_per_s": round(bytes_moved / (ms * 1e6), 1),
                "copy_ms": round(copy_ms, 4),
                "copy_gb_per_s": round(bytes_moved / (copy_ms * 1e6), 1),
            })
            print(f"  {mps_x.numel():>8} elements n={degree}: {ms * 1e3:7.1f} us "
                  f"{rows[-1]['gb_per_s']:6.0f} GB/s, copy floor "
                  f"{rows[-1]['copy_gb_per_s']:6.0f} GB/s")
    return rows


def overloads():
    """Per-call cost of the six overloads against the elementwise-multiply floor.

    Every overload runs on the same 4096-element input, where the call is
    launch-bound and the differences are host-side work: the scalar forms
    skip building an operand tensor, and ``.out`` skips the allocation.
    """
    mps_x = torch.rand(4096, device="mps")
    degrees = torch.full((4096,), 4.0, device="mps")
    single_degree = torch.tensor(4.0, device="mps")
    out = torch.empty(4096, device="mps")
    cases = {
        "tensor_tensor": lambda: aten.default(mps_x, degrees),
        "out": lambda: aten.out(mps_x, degrees, out=out),
        "x_scalar": lambda: aten.x_scalar(0.5, single_degree),
        "x_scalar_out": lambda: aten.x_scalar_out(0.5, degrees, out=out),
        "n_scalar": lambda: aten.n_scalar(mps_x, 4.0),
        "n_scalar_out": lambda: aten.n_scalar_out(mps_x, 4.0, out=out),
        "elementwise_multiply": lambda: mps_x * 2.0,
    }
    row = {}
    for label, call in cases.items():
        us = _per_call_ms(call, repeat=200, trials=15, warmup=20) * 1e3
        row[label] = round(us, 2)
        print(f"  4096 elements {label:<22} {us:6.2f} us")
    return row


def launch_cost():
    """Single-call latency at 4096 elements, where dispatch dominates."""
    cpu_x = torch.rand(4096, generator=torch.Generator().manual_seed(4))
    mps_x = cpu_x.to("mps")
    row = {"shape": [4096], "elements": 4096, "degree": 4}
    for label, call in (("cpu_us", lambda: leg(cpu_x, 4.0)),
                        ("mps_us", lambda: leg(mps_x, 4.0)),
                        ("mps_mul_us", lambda: mps_x * 2.0)):
        repeats = 50
        for _ in range(5):
            call()
        torch.mps.synchronize()
        samples = []
        for _ in range(7):
            start = time.perf_counter()
            for _ in range(repeats):
                call()
            torch.mps.synchronize()
            samples.append((time.perf_counter() - start) * 1e6 / repeats)
        row[label] = round(min(samples), 2)
    print(f"  4096 elements n=4: cpu {row['cpu_us']} us, mps {row['mps_us']} us, "
          f"elementwise mul on mps {row['mps_mul_us']} us")
    return row


def main():
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available on this machine")
    install()
    with lease_guard("mps-kernelops", tier=1, task_name="verify_legendre"):
        ok = verify()
        failures = [row for row in _checks if not row["bit_exact"]]
        print(f"\n{len(_checks) - len(failures)}/{len(_checks)} checks pass")
        print("== timing ==")
        rows = measure()
        print("== bandwidth ==")
        rates = bandwidth()
        print("== overloads ==")
        costs = overloads()
        print("== 4096-element latency ==")
        floor = launch_cost()
    MEASURED.parent.mkdir(parents=True, exist_ok=True)
    MEASURED.write_text(json.dumps({
        "torch": torch.__version__,
        "generated": time.strftime("%Y-%m-%d"),
        "op": "aten::special_legendre_polynomial_p",
        "device": "mps",
        "upstream": {
            "source": "aten/src/ATen/native/Math.h",
            "function": "legendre_polynomial_p_forward",
            "license": "BSD-3-Clause, PyTorch contributors",
        },
        "overlays": [name for name, _ in legendre_ops.OVERLAYS],
        "checks": {
            "total": len(_checks),
            "failed": len(failures),
            "cases": _checks,
        },
        "timing": {
            "method": "least-contended of 15 trials of 10 queued calls after 3 warmup calls, "
                      "one torch.mps.synchronize per trial, device-resident tensors so "
                      "transfers are excluded; measured while other work shared the GPU",
            "rows": rows,
            "bandwidth": rates,
            "overload_us": costs,
            "launch_floor": floor,
        },
        "boundaries": [
            "float32 only; float16 and bfloat16 raise, as the CPU kernel does; "
            "float64 cannot exist on MPS; complex inputs and complex scalar "
            "operands raise the same NotImplementedError",
            "a degree at or above about 1e12, and +inf, do not terminate: the "
            "recurrence runs once per degree and |x| <= 1 never turns NaN, exactly "
            "as upstream",
            "a 4096-element call is dispatch-bound, not kernel-bound: compare it "
            "with the elementwise-multiply floor in launch_floor",
            "a copy of the same length is the floor for this kernel's traffic; the "
            "bandwidth rows report the ratio, and a row near the copy is at the "
            "memory roofline",
            "timing rows carry an elementwise multiply on the same mps shape as a "
            "floor reference, because other agents were sharing the GPU during the "
            "run; a single row can be several times the uncontended time",
        ],
    }, indent=2) + "\n")
    print(f"wrote {MEASURED.relative_to(ROOT)}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
