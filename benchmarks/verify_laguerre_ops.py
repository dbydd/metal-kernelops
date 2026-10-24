"""Verify and measure the MPS ``special_laguerre_polynomial_l`` overlay.

Run with the MPS torch build:

    uv run --project /Users/dbydd/vibe-agent-working-dir/ResearchChamber python \
        benchmarks/verify_laguerre_ops.py

The script registers the six overlays on a throwaway ``aten`` fragment for the
lifetime of the process, checks every overload against the CPU kernel, times
the kernel against the CPU kernel and against the torch-ops loop that is the
only alternative on MPS today, and writes ``measured/laguerre_ops.json``. It
exits non-zero on the first real numerical disagreement.
"""

import json
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.library import Library

from metal_kernelops.laguerre_ops import MAX_DEGREE, OVERLAYS, _LANES, _TABLE, _THREADS

BROKER = Path(
    os.environ.get(
        "KERNELOPS_BROKER", "/Users/dbydd/vibe-agent-working-dir/ResearchChamber/shared_scripts"
    )
)

failures = []
results = {}
_libraries = []


def check(name, condition, detail=""):
    if condition:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        failures.append(f"{name} {detail}")


@contextmanager
def lease(task):
    """Hold a tier-1 compute lease when the local broker is installed."""
    if not (BROKER / "compute_broker.py").is_file():
        yield
        return
    sys.path.insert(0, str(BROKER))
    from compute_broker import lease_guard

    with lease_guard("mps-kernelops", tier=1, task_name=task):
        yield


def register():
    before = {
        name: torch._C._dispatch_has_kernel_for_dispatch_key(
            f"aten::{name}", "MPS"
        )
        for name, _ in OVERLAYS
    }
    library = Library("aten", "FRAGMENT")
    _libraries.append(library)
    for name, function in OVERLAYS:
        library.impl(name, function, "MPS")
    after = {
        name: torch._C._dispatch_has_kernel_for_dispatch_key(f"aten::{name}", "MPS")
        for name, _ in OVERLAYS
    }
    return before, after


def same(a, b):
    """Bitwise equality with NaN treated as equal."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return torch.equal(
        torch.nan_to_num(a, nan=math.pi, posinf=math.pi, neginf=-math.pi),
        torch.nan_to_num(b, nan=math.pi, posinf=math.pi, neginf=-math.pi),
    )


def verify_values():
    print("\nvalues: float32, scalar degree")
    base = torch.cat(
        [
            torch.linspace(-4.0, 4.0, 257),
            torch.randn(4096) * 3,
            torch.randn(512) * 30,
            torch.rand(256) * 0.25,
        ]
    )
    specials = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, 1e-30, -1e-30, 1e30, -1e30,
         float("inf"), float("-inf"), float("nan")]
    )
    x = torch.cat([base, specials]).to(torch.float32)
    xm = x.to("mps")

    worst = {"degree": None, "max_abs": 0.0, "max_rel": 0.0}
    for degree in [0, 1, 2, 3, 4, 5, 6, 8, 12, 16, 32, 64, 128, 500, 2000]:
        want = torch.special.laguerre_polynomial_l(x, degree)
        got = torch.special.laguerre_polynomial_l(xm, degree).cpu()
        ok = same(want, got)
        finite = torch.isfinite(want) & torch.isfinite(got)
        abs_err = (want[finite] - got[finite]).abs().max().item() if finite.any() else 0.0
        rel_err = (
            ((want[finite] - got[finite]).abs() / want[finite].abs().clamp_min(1e-38)).max().item()
            if finite.any()
            else 0.0
        )
        check(f"degree {degree} bit-exact over {x.numel()} values", ok,
              f"max_abs={abs_err:.3e} max_rel={rel_err:.3e}")
        if abs_err > worst["max_abs"]:
            worst = {"degree": degree, "max_abs": abs_err, "max_rel": rel_err}
    results["f32_scalar_degree"] = worst

    print("\nvalues: float64 has no MPS representation, half and bfloat16")
    for dtype in [torch.float16, torch.bfloat16]:
        xh = x.to(dtype)
        try:
            torch.special.laguerre_polynomial_l(xh, 3)
            cpu_raises = False
        except NotImplementedError:
            cpu_raises = True
        got = torch.special.laguerre_polynomial_l(xh.to("mps"), 3).cpu()
        want = torch.special.laguerre_polynomial_l(xh.float(), 3).to(dtype)
        check(f"cpu has no {dtype} kernel", cpu_raises)
        check(f"{dtype} matches float32 recurrence rounded to {dtype}", same(want, got))
    results["low_precision"] = "float32 recurrence, stored in the promoted dtype"


def verify_degrees():
    print("\ndegree semantics")
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 1.5, 9.0], dtype=torch.float32)
    xm = x.to("mps")
    for degree, label in [
        (2.9, "2.9 truncates to 2"),
        (2.5, "2.5 truncates to 2"),
        (-0.5, "-0.5 truncates to 0"),
        (-2.9, "-2.9 truncates to -2"),
        (-3, "-3 is negative"),
        (True, "bool True is degree 1"),
        (0.999, "0.999 truncates to 0"),
    ]:
        want = torch.special.laguerre_polynomial_l(x, degree)
        got = torch.special.laguerre_polynomial_l(xm, degree).cpu()
        check(f"scalar n={label}", same(want, got))

    zero = torch.tensor([0.0, -0.0], dtype=torch.float32).to("mps")
    for degree in [0, 1, 7, MAX_DEGREE, 10 ** 9, float("inf")]:
        got = torch.special.laguerre_polynomial_l(zero, degree).cpu()
        check(f"x==0 returns 1 at degree {degree}", torch.equal(got, torch.ones(2)))
    got = torch.special.laguerre_polynomial_l(zero, -3).cpu()
    check("x==0 with negative degree returns 0", torch.equal(got, torch.zeros(2)))

    one = torch.tensor([1.5], device="mps")
    check("nan degree behaves as 0", torch.equal(
        torch.special.laguerre_polynomial_l(one, float("nan")).cpu(), torch.ones(1)))
    check("-inf degree behaves as negative", torch.equal(
        torch.special.laguerre_polynomial_l(one, float("-inf")).cpu(), torch.zeros(1)))
    check("+inf degree is capped to nan", torch.isnan(
        torch.special.laguerre_polynomial_l(one, float("inf")).cpu()).all())
    check(f"degree >= {MAX_DEGREE} returns nan", torch.isnan(
        torch.special.laguerre_polynomial_l(one, MAX_DEGREE).cpu()).all())
    print("\ndegree tensor")
    xn = torch.tensor([1.5, -0.7, 0.0, 2.0], dtype=torch.float32).to("mps")
    for values, dtype, label in [
        ([0, 1, 2, 5], torch.int64, "int64"),
        ([0, 1, 2, 5], torch.int32, "int32"),
        ([0.0, 1.9, 2.9, -0.5], torch.float32, "float32 with truncation"),
        ([True, False, True, False], torch.bool, "bool"),
    ]:
        nt = torch.tensor(values, dtype=dtype)
        want = torch.special.laguerre_polynomial_l(xn.cpu(), nt)
        got = torch.special.laguerre_polynomial_l(xn, nt.to("mps")).cpu()
        check(f"tensor n {label} -> {want.dtype}", same(want, got))

    nt = torch.tensor([float("inf"), float("nan"), float("-inf"), 3.0], device="mps")
    got = torch.special.laguerre_polynomial_l(xn, nt).cpu()
    want = torch.special.laguerre_polynomial_l(xn.cpu()[1:], torch.tensor([0, -1, 3]))
    check("tensor n: inf hits the cap, nan behaves as 0, -inf as negative",
          torch.isnan(got[0]).all() and same(want, got[1:]))


def verify_shapes():
    print("\nshapes, strides, dtypes")
    shapes = [
        ((4,), (4,), "1-D"),
        ((2, 1), (3,), "broadcast x (2,1) with n (3,)"),
        ((3,), (3, 1), "broadcast x (3,) with n (3,1)"),
        ((), (), "0-dim both"),
        ((3, 5), (), "0-dim n"),
        ((), (3, 5), "0-dim x"),
        ((1, 1), (5,), "single-element x"),
    ]
    for xs, ns, label in shapes:
        a = torch.randn(xs) * 2
        b = torch.randint(-2, 6, ns).to(torch.float32) if ns else torch.tensor(3.0)
        want = torch.special.laguerre_polynomial_l(a, b)
        got = torch.special.laguerre_polynomial_l(a.to("mps"), b.to("mps")).cpu()
        check(f"{label} -> {tuple(want.shape)} {want.dtype}", same(want, got))

    for dtype in [torch.float32, torch.int64, torch.int32, torch.bool]:
        a = torch.tensor([1, 2, 3], dtype=dtype)
        b = torch.tensor([2, 2, 2], dtype=dtype)
        want = torch.special.laguerre_polynomial_l(a, b)
        got = torch.special.laguerre_polynomial_l(a.to("mps"), b.to("mps")).cpu()
        check(f"x and n dtype {dtype} -> {want.dtype}", same(want, got))

    x = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    want = torch.special.laguerre_polynomial_l(x, 3)
    check("non-contiguous x", same(want, torch.special.laguerre_polynomial_l(x.to("mps"), 3).cpu()))
    xe = torch.tensor([[1.5]], device="mps").expand(3, 5)
    want = torch.special.laguerre_polynomial_l(xe.cpu(), 3)
    check("expanded x with stride 0", same(want, torch.special.laguerre_polynomial_l(xe, 3).cpu()))
    nt = torch.arange(6, dtype=torch.float32).reshape(2, 3).t()
    want = torch.special.laguerre_polynomial_l(torch.tensor([1.5]), nt)
    got = torch.special.laguerre_polynomial_l(torch.tensor([1.5], device="mps"), nt.to("mps"))
    check("non-contiguous n", same(want, got.cpu()))

    empty = torch.empty(0, device="mps")
    got = torch.special.laguerre_polynomial_l(empty, 3)
    check("empty returns an empty float32 tensor",
          got.shape == (0,) and got.dtype == torch.float32)
    got = torch.special.laguerre_polynomial_l(
        torch.empty(0, 3, device="mps"), torch.tensor(3, device="mps"))
    check("empty 2-D keeps the shape", got.shape == (0, 3))

    # Vectorisation edge cases: large count, non-multiple-of-4 tail, and misaligned slice
    xl = torch.randn(65539, device="mps")
    want = torch.special.laguerre_polynomial_l(xl.cpu(), 32)
    got = torch.special.laguerre_polynomial_l(xl, 32).cpu()
    check("vectorised tail with count%4!=0 bit-exact", same(want, got))

    x_slice = torch.randn(10000, device="mps")[1:]
    want = torch.special.laguerre_polynomial_l(x_slice.cpu(), 8)
    got = torch.special.laguerre_polynomial_l(x_slice, 8).cpu()
    check("misaligned slice x[1:] scalar fallback bit-exact", same(want, got))

    xh = torch.tensor([1.5, 2.5], dtype=torch.float16, device="mps")
    nf = torch.tensor([2, 3], dtype=torch.float32, device="mps")
    got = torch.special.laguerre_polynomial_l(xh, nf)
    want = torch.special.laguerre_polynomial_l(xh.cpu().float(), nf.cpu())
    check("half x with float n promotes to float32", got.dtype == torch.float32 and same(want, got.cpu()))

    # The vector path is only taken at four elements or more, so the low-precision
    # vector types need a count that reaches it.
    for dtype in (torch.float16, torch.bfloat16):
        for count in (1000, 1001):
            xv = (torch.rand(count) * 4 - 2).to(dtype).to("mps")
            want = torch.special.laguerre_polynomial_l(xv.cpu().float(), 33).to(dtype)
            check(f"{dtype} vectorised path, count {count}", same(
                want, torch.special.laguerre_polynomial_l(xv, 33).cpu()))


def verify_overloads():
    print("\noverloads")
    x = torch.tensor([1.5, -0.5, 2.0], dtype=torch.float32)
    n = torch.tensor([2, 3, 4], dtype=torch.int64)
    ops = torch.ops.aten.special_laguerre_polynomial_l
    xm, nm = x.to("mps"), n.to("mps")

    check("x_scalar matches CPU",
          same(ops.x_scalar(1.5, n).float(), ops.x_scalar(1.5, nm).cpu()))
    check("n_scalar matches CPU",
          same(ops.n_scalar(x, 3), ops.n_scalar(xm, 3).cpu()))
    check("x_scalar integer scalar -> float32",
          ops.x_scalar(2, nm).dtype == torch.float32
          and same(ops.x_scalar(2, n).float(), ops.x_scalar(2, nm).cpu()))

    out = torch.empty(3, device="mps")
    got = ops.out(xm, nm, out=out)
    check("out= writes the same storage",
          got.data_ptr() == out.data_ptr()
          and same(torch.special.laguerre_polynomial_l(x, n), got.cpu()))
    out16 = torch.empty(3, dtype=torch.float16, device="mps")
    got = ops.out(xm, nm, out=out16)
    check("out= float16 casts the float32 result",
          got.dtype == torch.float16
          and same(torch.special.laguerre_polynomial_l(x, n).to(torch.float16), got.cpu()))
    og = torch.empty(2, device="mps")
    got = ops.n_scalar_out(xm, 3, out=og)
    check("n_scalar_out resizes a wrong-sized out", got.shape == (3,) and got.data_ptr() == og.data_ptr())
    base = torch.empty(4, 2, device="mps")
    view = base[:, 0]
    got = ops.out(xm, nm, out=view)
    check("out= accepts a non-contiguous view", same(torch.special.laguerre_polynomial_l(x, n), got.cpu()))
    inplace = x.clone().to("mps")
    got = ops.out(inplace, nm, out=inplace)
    check("out= aliasing x", same(torch.special.laguerre_polynomial_l(x, n), got.cpu()))

    for dtype, label in [(torch.int32, "Int"), (torch.int64, "Long"), (torch.bool, "Bool")]:
        try:
            ops.out(xm, nm, out=torch.empty(3, dtype=dtype, device="mps"))
            check(f"out= {dtype} rejected", False)
        except RuntimeError as error:
            check(f"out= {dtype} rejected like CPU", f"desired output type {label}" in str(error),
                  str(error))
    try:
        ops.out(xm, nm, out=torch.empty(3))
        check("out= on cpu rejected", False)
    except RuntimeError:
        check("out= on cpu rejected", True)

    xg = torch.tensor([1.5], device="mps", requires_grad=True)
    check("no autograd graph, as on CPU",
          not torch.special.laguerre_polynomial_l(xg, 2).requires_grad)

    try:
        torch.special.laguerre_polynomial_l(xm, n)
        check("mixed-device n rejected", False)
    except RuntimeError as error:
        check("mixed-device n rejected", "same device" in str(error))


def cpu_batch(fn, reps):
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - start) * 1e3 / reps


def mps_batch(fn, reps):
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - start) * 1e3 / reps


def interleaved(configs, rounds, batch):
    """Measure every config once per round; keep the fastest round.

    Other processes share this GPU, and their interference only ever adds time,
    so the minimum over interleaved rounds is the honest estimate.
    """
    best = {key: float("inf") for key in configs}
    shuffle = random.Random(0)
    items = list(configs.items())
    for _ in range(rounds):
        shuffle.shuffle(items)
        for key, fn in items:
            best[key] = min(best[key], mps_batch(fn, batch))
    return best


def torch_loop(x, degree):
    """The decomposition of this operator a user has to write on MPS today."""
    if degree == 0:
        return torch.ones_like(x)
    p = torch.ones_like(x)
    t = 1.0 - x
    q = t
    r = q
    for k in range(1, degree):
        r = ((float(k + k) + t) * q - float(k) * p) / float(k + 1)
        p, q = q, r
    return r


def benchmark():
    print("\ntimings: fastest of interleaved rounds of queued calls, see measured JSON")
    sizes = [1 << 16, 1 << 20, 1 << 22, 1 << 24]
    degrees = [0, 1, 2, 4, 8, 32, 128]
    data = {}
    configs = {}
    for count in sizes:
        x = torch.rand(count) * 4 - 2
        xm = x.to("mps")
        data[count] = (x, xm)
        buffer = torch.empty_like(xm)
        for degree in degrees:
            configs[(count, degree)] = (
                lambda xm=xm, degree=degree: torch.special.laguerre_polynomial_l(xm, degree)
            )
        configs[(count, "copy")] = lambda xm=xm: torch.empty_like(xm).copy_(xm)
        # Device floors on one reused buffer: no allocator cost, so these are the
        # fastest a kernel touching the same bytes can run.
        configs[(count, "floor_copy")] = lambda xm=xm, buffer=buffer: buffer.copy_(xm)
        configs[(count, "floor_fill")] = lambda buffer=buffer: buffer.fill_(1.0)
        for degree in [4, 32]:
            configs[(count, f"loop{degree}")] = lambda xm=xm, degree=degree: torch_loop(xm, degree)

    for _ in range(2):
        for fn in configs.values():
            fn()
    torch.mps.synchronize()
    mps = interleaved(configs, rounds=10, batch=15)

    rows = []
    for count in sizes:
        x, _ = data[count]
        floor_copy = mps[(count, "floor_copy")]
        floor_fill = mps[(count, "floor_fill")]
        for degree in degrees:
            mps_ms = mps[(count, degree)]
            single = cpu_batch(lambda: torch.special.laguerre_polynomial_l(x, degree), 1)
            reps = max(3, min(20, int(0.2 / max(single / 1e3, 1e-9))))
            cpu_ms = min(cpu_batch(lambda: torch.special.laguerre_polynomial_l(x, degree), reps)
                         for _ in range(3))
            # Degree 0 answers without reading x, so it moves half the bytes.
            moved = count * (4 if degree == 0 else 8)
            row = {
                "count": count,
                "degree": degree,
                "mps_ms": round(mps_ms, 5),
                "cpu_ms": round(cpu_ms, 5),
                "cpu_reps": reps,
                "speedup": round(cpu_ms / mps_ms, 2),
                "mps_GB_per_s": round(count * 8 / (mps_ms / 1e3) / 1e9, 1),
                "mps_bytes_moved": moved,
                "mps_moved_GB_per_s": round(moved / (mps_ms / 1e3) / 1e9, 1),
                "vs_copy_floor": round(floor_copy / mps_ms, 2),
            }
            rows.append(row)
            print(
                f"  count={count:>9} degree={degree:>4}  mps={mps_ms:8.4f} ms  "
                f"cpu={cpu_ms:8.4f} ms  {row['speedup']:6.2f}x  "
                f"{row['mps_moved_GB_per_s']:7.1f} GB/s moved  "
                f"{row['vs_copy_floor']:5.2f}x of copy floor"
            )
        rows[-1]["mps_copy_ms"] = round(mps[(count, "copy")], 5)
        rows[-1]["floor_copy_ms"] = round(floor_copy, 5)
        rows[-1]["floor_fill_ms"] = round(floor_fill, 5)
        rows[-1]["floor_copy_GB_per_s"] = round(count * 8 / (floor_copy / 1e3) / 1e9, 1)
        rows[-1]["floor_fill_GB_per_s"] = round(count * 4 / (floor_fill / 1e3) / 1e9, 1)
        rows[-1]["mps_torch_loop_ms"] = {
            str(degree): round(mps[(count, f"loop{degree}")], 5) for degree in [4, 32]
        }
        print(f"  count={count:>9} floor: copy {floor_copy:.4f} ms "
              f"({rows[-1]['floor_copy_GB_per_s']:.1f} GB/s), fill {floor_fill:.4f} ms "
              f"({rows[-1]['floor_fill_GB_per_s']:.1f} GB/s), "
              f"allocating copy_ {rows[-1]['mps_copy_ms']:.4f} ms")

    x = torch.rand(1 << 16, device="mps")
    for _ in range(20):
        torch.special.laguerre_polynomial_l(x, 2)
    torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(200):
        torch.special.laguerre_polynomial_l(x, 2)
    dispatch_us = (time.perf_counter() - start) * 1e6 / 200
    start = time.perf_counter()
    for _ in range(200):
        x * 2.0
    torch_us = (time.perf_counter() - start) * 1e6 / 200
    torch.mps.synchronize()
    print(f"  python dispatch: {dispatch_us:.2f} us per call, torch elementwise {torch_us:.2f} us")
    return rows, {"mps_dispatch_us": round(dispatch_us, 2), "torch_elementwise_us": round(torch_us, 2)}


def gpu_load():
    """Concurrent lease holders, so a reader can tell a loaded box from an idle one."""
    if not (BROKER / "compute_broker.py").is_file():
        return None
    sys.path.insert(0, str(BROKER))
    try:
        from compute_broker import ComputeBroker

        status = ComputeBroker().status()
    except Exception as error:  # the broker is a shared external script
        return f"unavailable: {error}"
    return {
        "active_leases": status.get("active_leases_count"),
        "tasks": sorted(lease["task_name"] for lease in status.get("leases", [])),
    }


def main():
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available on this machine")
    print(f"torch {torch.__version__}")
    before, after = register()
    print("\nregistration")
    for name, _ in OVERLAYS:
        check(f"{name}: no MPS kernel before, one after", not before[name] and after[name])

    verify_values()
    verify_degrees()
    verify_shapes()
    verify_overloads()

    with lease("laguerre_verify"):
        rows, dispatch = benchmark()
    load = gpu_load()

    payload = {
        "operator": "aten::special_laguerre_polynomial_l",
        "overloads": [name for name, _ in OVERLAYS],
        "torch": torch.__version__,
        "device": torch.backends.mps.is_available() and "mps" or "unavailable",
        "chip": os.popen("sysctl -n machdep.cpu.brand_string").read().strip(),
        "max_degree": MAX_DEGREE,
        "kernel": {"lanes": _LANES, "reciprocal_table": _TABLE, "group_size": _THREADS},
        "optimizations": {
            "vectorization": "four elements per thread through float4/half4/bfloat4 loads and "
                             "stores, with a scalar tail and a scalar fallback when the buffers "
                             "are not vector-aligned",
            "low_degree": "degree 0 writes 1.0 without reading x, since the result cannot depend "
                          "on it; the uniform scalar degree is resolved once per thread",
            "high_degree": "the recurrence divides by a tabulated correctly rounded reciprocal "
                           "plus one fma correction (Markstein), replacing the hardware divide "
                           "with three full-rate operations that round identically",
            "ilp": "four independent recurrence chains per thread, so the divide latency of one "
                   "lane overlaps the arithmetic of the other three",
        },
        "exactness": {
            "float32": "bit-identical to the CPU kernel for every finite input and degree tested",
            "float16_bfloat16": "CPU has no kernel; float32 recurrence stored at the promoted dtype",
            "deviations": [
                "non-finite degree: CPU is undefined behaviour and hangs, MPS maps nan to 0, "
                "-inf to 0, +inf to the degree cap",
                f"degree >= {MAX_DEGREE}: MPS returns nan instead of looping; only visible for "
                f"x > 0, where L_n(x) stays bounded instead of overflowing",
            ],
        },
        "method": "timings are the fastest of 10 interleaved rounds of 15 queued calls, each "
                  "round followed by torch.mps.synchronize(); other processes share this GPU, "
                  "so the best round isolates this kernel from their interference. The copy and "
                  "fill floors are torch ops on one reused buffer, i.e. the fastest anything "
                  "touching those bytes can run; mps_moved_GB_per_s counts the bytes the kernel "
                  "actually moves, which is count*4 rather than count*8 at degree 0",
        "gpu_load_during_measurement": load,
        "dispatch": dispatch,
        "timings": rows,
        "failures": failures,
    }
    out = ROOT / "measured" / "laguerre_ops.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}")
    if failures:
        print(f"\n{len(failures)} FAILURES")
        for failure in failures:
            print(" ", failure)
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
