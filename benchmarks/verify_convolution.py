"""Correctness, gradient and timing check for ``metal_kernelops.convolution_ops``.

Run from the repository root:

    uv run --project /Users/dbydd/vibe-agent-working-dir/ResearchChamber \
        python benchmarks/verify_convolution.py

It records, for each of the six legacy ``slow_conv*`` entry points:

* the aten schema and what the call did on MPS before the overlay (the baseline);
* CPU fp64 forward against the MPS overlay, plus shape and dtype;
* gradients: CPU fp32 against MPS fp32 for the same inputs and upstream grad;
* kernel timing on MPS, the native ``aten::convolution`` path as the baseline,
  and a forward+backward step, all warm and synchronized.

The reference is ``aten::convolution`` / ``aten::convolution_backward`` on CPU,
not the legacy CPU kernels. ``semantics`` checks that the two agree on CPU, which
is what makes that reference valid.

Results go to ``measured/convolution.json``.
"""

import json
import os
from statistics import median
import sys
import time
from pathlib import Path

import torch
from torch.library import Library
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metal_kernelops.convolution_ops import OVERLAYS  # noqa: E402

ATEN = torch.ops.aten
DEVICE = "mps"
MPS = torch.device(DEVICE)
WARMUP = 5
ITERS = 30
BATCHES = 12
CALLS = {}


class DeviceTrace(TorchDispatchMode):
    """Records the device of every tensor the overlay dispatches on."""

    def __init__(self):
        super().__init__()
        self.devices = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        for value in list(args) + list(kwargs.values()):
            if isinstance(value, torch.Tensor):
                self.devices.add(value.device.type)
        return func(*args, **kwargs)


# --------------------------------------------------------------------------- #
# Case definitions. ``args`` live on CPU; every run gets its own copy.
# --------------------------------------------------------------------------- #

CASES = [
    {
        "name": "conv3d",
        "op": "slow_conv3d_forward",
        "shapes": {
            "input": (2, 8, 12, 12, 12),
            "weight": (16, 8, 3, 3, 3),
            "bias": (16,),
        },
        "legacy": lambda a: ATEN.slow_conv3d_forward(
            a["input"], a["weight"], [3, 3, 3], a["bias"], [1, 1, 1], [1, 1, 1]
        ),
        "native": lambda a: ATEN.convolution(
            a["input"], a["weight"], a["bias"], [1, 1, 1], [1, 1, 1], [1, 1, 1],
            False, [0, 0, 0], 1,
        ),
        "backward_native": lambda a, g: ATEN.convolution_backward(
            g, a["input"], a["weight"], [16], [1, 1, 1], [1, 1, 1], [1, 1, 1],
            False, [0, 0, 0], 1, [True, True, True],
        ),
    },
    {
        "name": "dilated2d",
        "op": "slow_conv_dilated2d",
        "shapes": {
            "input": (2, 8, 32, 32),
            "weight": (16, 8, 3, 3),
            "bias": (16,),
        },
        "legacy": lambda a: ATEN.slow_conv_dilated2d(
            a["input"], a["weight"], [3, 3], a["bias"], [1, 1], [2, 2], [2, 2]
        ),
        "native": lambda a: ATEN.convolution(
            a["input"], a["weight"], a["bias"], [1, 1], [2, 2], [2, 2],
            False, [0, 0], 1,
        ),
        "backward_native": lambda a, g: ATEN.convolution_backward(
            g, a["input"], a["weight"], [16], [1, 1], [2, 2], [2, 2],
            False, [0, 0], 1, [True, True, True],
        ),
    },
    {
        "name": "dilated3d",
        "op": "slow_conv_dilated3d",
        "shapes": {
            "input": (2, 8, 10, 10, 10),
            "weight": (16, 8, 3, 3, 3),
            "bias": (16,),
        },
        "legacy": lambda a: ATEN.slow_conv_dilated3d(
            a["input"], a["weight"], [3, 3, 3], a["bias"], [1, 1, 1], [2, 2, 2], [2, 2, 2]
        ),
        "native": lambda a: ATEN.convolution(
            a["input"], a["weight"], a["bias"], [1, 1, 1], [2, 2, 2], [2, 2, 2],
            False, [0, 0, 0], 1,
        ),
        "backward_native": lambda a, g: ATEN.convolution_backward(
            g, a["input"], a["weight"], [16], [1, 1, 1], [2, 2, 2], [2, 2, 2],
            False, [0, 0, 0], 1, [True, True, True],
        ),
    },
    {
        "name": "transpose2d",
        "op": "slow_conv_transpose2d",
        "shapes": {
            "input": (2, 8, 16, 16),
            "weight": (8, 16, 3, 3),
            "bias": (16,),
        },
        "legacy": lambda a: ATEN.slow_conv_transpose2d(
            a["input"], a["weight"], [3, 3], a["bias"], [2, 2], [1, 1], [1, 1], [1, 1]
        ),
        "native": lambda a: ATEN.convolution(
            a["input"], a["weight"], a["bias"], [2, 2], [1, 1], [1, 1],
            True, [1, 1], 1,
        ),
        "backward_native": lambda a, g: ATEN.convolution_backward(
            g, a["input"], a["weight"], [16], [2, 2], [1, 1], [1, 1],
            True, [1, 1], 1, [True, True, True],
        ),
    },
    {
        "name": "transpose3d",
        "op": "slow_conv_transpose3d",
        "shapes": {
            "input": (2, 8, 8, 8, 8),
            "weight": (8, 16, 3, 3, 3),
            "bias": (16,),
        },
        "legacy": lambda a: ATEN.slow_conv_transpose3d(
            a["input"], a["weight"], [3, 3, 3], a["bias"], [2, 2, 2], [1, 1, 1],
            [1, 1, 1], [1, 1, 1],
        ),
        "native": lambda a: ATEN.convolution(
            a["input"], a["weight"], a["bias"], [2, 2, 2], [1, 1, 1], [1, 1, 1],
            True, [1, 1, 1], 1,
        ),
        "backward_native": lambda a, g: ATEN.convolution_backward(
            g, a["input"], a["weight"], [16], [2, 2, 2], [1, 1, 1], [1, 1, 1],
            True, [1, 1, 1], 1, [True, True, True],
        ),
    },
]

BACKWARD_CASE = {
    "name": "conv2d_backward",
    "op": "_slow_conv2d_backward",
    "shapes": {
        "grad_output": (2, 16, 16, 16),
        "input": (2, 8, 18, 18),
        "weight": (16, 8, 3, 3),
    },
    "legacy": lambda a: ATEN._slow_conv2d_backward.output_mask(
        a["grad_output"], a["input"], a["weight"], [3, 3], [1, 1], [0, 0],
        [True, True, True],
    ),
    "native": lambda a: ATEN.convolution_backward(
        a["grad_output"], a["input"], a["weight"], [16], [1, 1], [0, 0], [1, 1],
        False, [0, 0], 1, [True, True, True],
    ),
}

OP_NAMES = {
    "slow_conv3d_forward": "aten::slow_conv3d_forward",
    "slow_conv_dilated2d": "aten::slow_conv_dilated2d",
    "slow_conv_dilated3d": "aten::slow_conv_dilated3d",
    "slow_conv_transpose2d": "aten::slow_conv_transpose2d",
    "slow_conv_transpose3d": "aten::slow_conv_transpose3d",
    "_slow_conv2d_backward": "aten::_slow_conv2d_backward.output_mask",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def make_args(case, dtype, device):
    generator = torch.Generator().manual_seed(1234)
    args = {}
    for name, shape in case["shapes"].items():
        tensor = torch.randn(shape, generator=generator, dtype=torch.float64)
        args[name] = tensor.to(device=device, dtype=dtype)
    return args


def rel_error(actual, expected):
    actual = actual.detach().cpu().double()
    expected = expected.detach().cpu().double()
    denominator = expected.norm()
    if denominator == 0:
        return float((actual - expected).norm())
    return float((actual - expected).norm() / denominator)


def time_mps(*fns, batches=BATCHES):
    """Per-batch samples for each function, the functions interleaved.

    A batch is ITERS calls and one wait. The functions alternate batch by batch,
    so a busy stretch on this shared machine lands on whichever function was
    running, and every function sees the same spread. ``min`` over batches is
    the number to compare; the rest of the samples are kept so the spread is
    visible instead of being averaged away.

    ``wall`` is submission plus execution, the number the alias is judged on.
    ``enqueue`` is submission only, the Python work the alias adds; it is the
    number to read when the GPU is too busy for ``wall`` to say anything.
    """
    for fn in fns:
        for _ in range(WARMUP):
            fn()
    torch.mps.synchronize()
    samples = [{"wall": [], "enqueue": []} for _ in fns]
    for _ in range(batches):
        for index, fn in enumerate(fns):
            start = time.perf_counter()
            for _ in range(ITERS):
                fn()
            submitted = time.perf_counter()
            torch.mps.synchronize()
            done = time.perf_counter()
            samples[index]["wall"].append((done - start) / ITERS * 1000.0)
            samples[index]["enqueue"].append((submitted - start) / ITERS * 1000.0)
    return samples


def time_cpu(fn, rounds=2):
    for _ in range(WARMUP):
        fn()
    best = float("inf")
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(ITERS):
            fn()
        best = min(best, (time.perf_counter() - start) / ITERS * 1000.0)
    return best


def step_closure(call, args):
    def step():
        tensors = {name: value.detach().requires_grad_(True) for name, value in args.items()}
        out = call(tensors)
        torch.autograd.grad(out, list(tensors.values()), torch.ones_like(out))
        return out

    return step


# --------------------------------------------------------------------------- #
# Baseline: what each call did before the overlay, and its schema.
# --------------------------------------------------------------------------- #

def baseline(case):
    name = OP_NAMES[case["op"]]
    base, _, overload = name.partition(".")
    row = {
        "op": case["op"],
        "schema": str(torch._C._get_schema(base, overload)),
        "mps_kernel_before": bool(
            torch._C._dispatch_has_kernel_for_dispatch_key(name, "MPS")
        ),
    }
    args = make_args(case, torch.float32, MPS)
    try:
        case["legacy"](args)
        torch.mps.synchronize()
        row["baseline"] = "ran"
    except Exception as exc:  # the point is to record the failure
        row["baseline"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
    return row


def _counted(name, kernel):
    def wrapper(*args, **kwargs):
        CALLS[name] = CALLS.get(name, 0) + 1
        return kernel(*args, **kwargs)

    return wrapper


def install_overlays():
    library = Library("aten", "FRAGMENT")
    for name, kernel in OVERLAYS:
        library.impl(name, _counted(name, kernel), "MPS")
    return library


def copy_floor(outputs):
    """A device copy of the overlay's own output, as (callable, bytes moved).

    The cheapest thing that touches that much memory, measured in the same batch
    as the alias: the floor a memory-bound kernel on this machine sits on, and
    the reference for the achieved bandwidth reported with it.
    """
    tensors = [t for t in (outputs if isinstance(outputs, tuple) else (outputs,)) if t is not None]
    buffers = [(torch.empty_like(t), t) for t in tensors]

    def copy():
        for dst, src in buffers:
            dst.copy_(src)

    moved = 2 * sum(t.numel() * t.element_size() for t in tensors)
    return copy, moved


def timing_row(legacy_fn, native_fn, floor=None):
    """Overlay against the native op, with the native call measured twice.

    The repeat is the noise control: it is the same call, so anything the alias
    has to show must be larger than the gap between those two. The copy is the
    other floor - a device copy of the same output, the cheapest thing that
    moves that many bytes.

    Every pair of numbers here comes from one batch, where both functions ran
    back to back. The reported value is the median of the per-batch differences
    (or ratios), not a difference of two minima taken at different moments: on a
    machine with other jobs on the GPU the second form reports tens of
    microseconds of "overhead" for two identical calls.
    """
    fns = [legacy_fn, native_fn, native_fn]
    if floor is not None:
        fns.append(floor[0])
    samples = time_mps(*fns)

    def delta_us(first, second, stat):
        """Median of the per-batch difference first - second."""
        return median(
            (a - b) * 1000.0
            for a, b in zip(samples[first][stat], samples[second][stat])
        )

    def spread_us(index, stat):
        values = samples[index][stat]
        return (max(values) - min(values)) * 1000.0

    row = {
        "mps_ms": median(samples[0]["wall"]),
        "native_ms": median(samples[1]["wall"]),
        "native_repeat_ms": median(samples[2]["wall"]),
        "overhead_us": delta_us(0, 1, "wall"),
        "overhead_ratio": median(
            a / b for a, b in zip(samples[0]["wall"], samples[1]["wall"])
        ),
        "mps_min_ms": min(samples[0]["wall"]),
        "native_min_ms": min(samples[1]["wall"]),
        "enqueue_ms": median(samples[0]["enqueue"]),
        "native_enqueue_ms": median(samples[1]["enqueue"]),
        "enqueue_overhead_us": delta_us(0, 1, "enqueue"),
        "noise_us": median(
            abs(a - b) * 1000.0 for a, b in zip(samples[1]["wall"], samples[2]["wall"])
        ),
        "batch_spread_us": {
            "mps": spread_us(0, "wall"),
            "native": spread_us(1, "wall"),
        },
    }
    if floor is not None:
        floor_ms = median(samples[3]["wall"])
        row["copy_floor_ms"] = floor_ms
        row["copy_floor_gbps"] = floor[1] / (floor_ms * 1e6)
    return row


# --------------------------------------------------------------------------- #
# Correctness, gradients, timing.
# --------------------------------------------------------------------------- #

def run_case(case):
    args64 = make_args(case, torch.float64, torch.device("cpu"))
    reference = case["native"](args64).detach()

    args32_cpu = make_args(case, torch.float32, torch.device("cpu"))
    args_f16_cpu = make_args(case, torch.float16, torch.device("cpu"))

    rows = []
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        cpu_args = {name: value.to(dtype) for name, value in args64.items()}
        if dtype is torch.float32:
            cpu_args = args32_cpu
        elif dtype is torch.float16:
            cpu_args = args_f16_cpu
        mps_args = {name: value.to(MPS) for name, value in cpu_args.items()}

        try:
            actual = case["legacy"](mps_args)
            torch.mps.synchronize()
        except RuntimeError as exc:  # dtype the native Metal kernel does not have
            rows.append(
                {
                    "name": case["name"],
                    "op": case["op"],
                    "dtype": str(dtype).replace("torch.", ""),
                    "supported": False,
                    "error": str(exc).splitlines()[0][:160],
                }
            )
            continue
        row = {
            "name": case["name"],
            "op": case["op"],
            "dtype": str(dtype).replace("torch.", ""),
            "supported": True,
            "shape": list(actual.shape),
            "reference_shape": list(reference.shape),
            "shape_match": tuple(actual.shape) == tuple(reference.shape),
            "dtype_match": actual.dtype == dtype,
            "value_rel_error": rel_error(actual.cpu(), reference),
            "cpu_ms": time_cpu(lambda: case["legacy"](cpu_args)),
        }
        row.update(
            timing_row(
                lambda: case["legacy"](mps_args),
                lambda: case["native"](mps_args),
                copy_floor(actual),
            )
        )
        if dtype is torch.float32:
            step_samples = time_mps(
                step_closure(case["legacy"], mps_args),
                step_closure(case["native"], mps_args),
            )
            row["step_mps_ms"], row["step_native_ms"] = (
                median(s["wall"]) for s in step_samples
            )
            row["step_cpu_ms"] = time_cpu(step_closure(case["legacy"], cpu_args))
        rows.append(row)

    # Gradients: same float32 inputs, same upstream grad, CPU versus MPS.
    cpu_args = {name: value.clone().requires_grad_(True) for name, value in args32_cpu.items()}
    mps_args = {name: value.to(MPS).requires_grad_(True) for name, value in args32_cpu.items()}

    cpu_out = case["legacy"](cpu_args)
    seed = torch.randn(cpu_out.shape, generator=torch.Generator().manual_seed(99))
    cpu_grads = torch.autograd.grad(cpu_out, list(cpu_args.values()), seed)
    mps_out = case["legacy"](mps_args)
    mps_grads = torch.autograd.grad(mps_out, list(mps_args.values()), seed.to(MPS))
    names = list(cpu_args)
    rows[0]["grad_rel_error"] = {
        name: rel_error(mps_grad, cpu_grad)
        for name, mps_grad, cpu_grad in zip(names, mps_grads, cpu_grads)
    }
    rows[0]["grad_norms_mps"] = {
        name: float(grad.detach().cpu().double().norm()) for name, grad in zip(names, mps_grads)
    }
    return rows


def run_backward_case():
    case = BACKWARD_CASE
    args64 = make_args(case, torch.float64, torch.device("cpu"))
    reference = case["native"](args64)

    cpu_args = {name: value.to(torch.float32) for name, value in args64.items()}
    mps_args = {name: value.to(MPS) for name, value in cpu_args.items()}
    actual = case["legacy"](mps_args)
    torch.mps.synchronize()

    row = {
        "name": case["name"],
        "op": case["op"],
        "dtype": "float32",
        "supported": True,
        "shape": [list(t.shape) for t in actual],
        "reference_shape": [list(t.shape) for t in reference],
        "shape_match": all(a.shape == b.shape for a, b in zip(actual, reference)),
        "dtype_match": all(a.dtype == torch.float32 for a in actual),
        "value_rel_error": max(rel_error(a.cpu(), b) for a, b in zip(actual, reference)),
        "cpu_ms": time_cpu(lambda: case["legacy"](cpu_args)),
        "grad_rel_error": {
            name: rel_error(a.cpu(), b)
            for name, a, b in zip(("input", "weight", "bias"), actual, reference)
        },
    }
    row.update(
        timing_row(
            lambda: case["legacy"](mps_args),
            lambda: case["native"](mps_args),
            copy_floor(actual),
        )
    )
    return [row]


# --------------------------------------------------------------------------- #
# Semantics that are not covered by one batched case.
# --------------------------------------------------------------------------- #

def check_semantics():
    checks = []

    def record(name, passed, detail=""):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    # The reference is the native convolution on CPU; show the legacy CPU kernel
    # agrees with it on the same fp64 tensors.
    for case in CASES:
        args = make_args(case, torch.float64, torch.device("cpu"))
        legacy = case["legacy"](args)
        native = case["native"](args)
        record(
            f"cpu_legacy_equals_native/{case['name']}",
            torch.equal(legacy, native),
            f"max |diff| {float((legacy - native).abs().max()):.3e}",
        )

    bwd = BACKWARD_CASE
    args = make_args(bwd, torch.float64, torch.device("cpu"))
    record(
        "cpu_legacy_equals_native/conv2d_backward",
        all(torch.equal(a, b) for a, b in zip(bwd["legacy"](args), bwd["native"](args))),
    )

    # Unbatched rules: dilated ops and transpose3d squeeze the batch back,
    # transpose2d does not. MPS has no float64, so these compare float32 to
    # float32.
    def cpu_tensors(shapes, seed):
        generator = torch.Generator().manual_seed(seed)
        return {
            name: torch.randn(shape, generator=generator)
            for name, shape in shapes.items()
        }

    def on_mps(tensors):
        return {name: value.to(MPS) for name, value in tensors.items()}

    def compare(name, cpu_call, mps_call, rank):
        cpu_out = cpu_call()
        mps_out = mps_call()
        torch.mps.synchronize()
        record(f"{name}_rank", mps_out.dim() == cpu_out.dim() == rank, f"{list(cpu_out.shape)}")
        error = rel_error(mps_out, cpu_out)
        record(f"{name}_value", error < 1e-5, f"{error:.3e}")

    args2 = cpu_tensors(
        {
            "input": (8, 20, 20),
            "weight": (16, 8, 3, 3),
            "weight_t": (8, 16, 3, 3),
            "bias": (16,),
        },
        21,
    )
    mps2 = on_mps(args2)
    compare(
        "unbatched_dilated2d",
        lambda: ATEN.slow_conv_dilated2d(args2["input"], args2["weight"], [3, 3], args2["bias"], [1, 1], [1, 1], [1, 1]),
        lambda: ATEN.slow_conv_dilated2d(mps2["input"], mps2["weight"], [3, 3], mps2["bias"], [1, 1], [1, 1], [1, 1]),
        3,
    )
    compare(
        "unbatched_transpose2d_stays_batched",
        lambda: ATEN.slow_conv_transpose2d(args2["input"], args2["weight_t"], [3, 3], args2["bias"], [1, 1], [0, 0], [0, 0], [1, 1]),
        lambda: ATEN.slow_conv_transpose2d(mps2["input"], mps2["weight_t"], [3, 3], mps2["bias"], [1, 1], [0, 0], [0, 0], [1, 1]),
        4,
    )
    record(
        "unbatched_transpose2d_batch_one",
        ATEN.slow_conv_transpose2d(mps2["input"], mps2["weight_t"], [3, 3], mps2["bias"], [1, 1], [0, 0], [0, 0], [1, 1]).shape[0] == 1,
        "3-D in, 4-D out with batch 1",
    )

    args3 = cpu_tensors(
        {
            "input": (8, 10, 10, 10),
            "weight": (16, 8, 3, 3, 3),
            "weight_t": (8, 16, 3, 3, 3),
            "bias": (16,),
        },
        22,
    )
    mps3 = on_mps(args3)
    compare(
        "unbatched_dilated3d",
        lambda: ATEN.slow_conv_dilated3d(args3["input"], args3["weight"], [3, 3, 3], args3["bias"], [1, 1, 1], [1, 1, 1], [1, 1, 1]),
        lambda: ATEN.slow_conv_dilated3d(mps3["input"], mps3["weight"], [3, 3, 3], mps3["bias"], [1, 1, 1], [1, 1, 1], [1, 1, 1]),
        4,
    )
    compare(
        "unbatched_transpose3d_stays_unbatched",
        lambda: ATEN.slow_conv_transpose3d(args3["input"], args3["weight_t"], [3, 3, 3], args3["bias"], [1, 1, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1]),
        lambda: ATEN.slow_conv_transpose3d(mps3["input"], mps3["weight_t"], [3, 3, 3], mps3["bias"], [1, 1, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1]),
        4,
    )

    # output_mask: false entries are undefined tensors, exposed as None.
    masked = ATEN._slow_conv2d_backward.output_mask(
        *(t.to(MPS) for t in (torch.randn(1, 16, 8, 8),
                              torch.randn(1, 8, 10, 10),
                              torch.randn(16, 8, 3, 3))),
        [3, 3], [1, 1], [0, 0], [False, True, False],
    )
    record(
        "backward_output_mask_returns_undefined",
        masked[0] is None and masked[2] is None and masked[1] is not None,
        f"{[type(value).__name__ for value in masked]}",
    )

    # Flattened 2-D weight: forward and backward stay flat.
    flat_w = torch.randn(16, 8 * 3 * 3)
    grad_flat = ATEN._slow_conv2d_backward.output_mask(
        torch.randn(1, 16, 8, 8, device=MPS),
        torch.randn(1, 8, 10, 10, device=MPS),
        flat_w.to(MPS), [3, 3], [1, 1], [0, 0], [True, True, True],
    )
    record(
        "backward_flat_weight_shape",
        tuple(grad_flat[1].shape) == tuple(flat_w.shape),
        f"{list(grad_flat[1].shape)}",
    )

    # grad_input overload writes the caller's tensors and returns them.
    grad_output = torch.randn(1, 16, 8, 8, device=MPS)
    grad_input = torch.empty(0, device=MPS)
    grad_weight = torch.empty(0, device=MPS)
    grad_bias = torch.empty(0, device=MPS)
    returned = ATEN._slow_conv2d_backward.grad_input(
        grad_output,
        torch.randn(1, 8, 10, 10, device=MPS),
        torch.randn(16, 8, 3, 3, device=MPS),
        [3, 3], [1, 1], [0, 0],
        grad_input=grad_input, grad_weight=grad_weight, grad_bias=grad_bias,
    )
    record(
        "backward_grad_input_writes_in_place",
        all(a is b for a, b in zip(returned, (grad_input, grad_weight, grad_bias)))
        and tuple(grad_input.shape) == (1, 8, 10, 10),
        f"{[list(t.shape) for t in returned]}",
    )

    # .out overloads write the caller's tensor, return it, and produce the same
    # values as the functional overload. The dilated ones are autogenerated and
    # route through the default overload.
    def check_out(name, functional, out_call):
        expected = functional()
        out = torch.empty(0, device=MPS)
        returned = out_call(out)
        torch.mps.synchronize()
        diff = float((out - expected).abs().max().item())
        record(
            f"out_overload/{name}",
            returned is out and tuple(out.shape) == tuple(expected.shape) and diff == 0.0,
            f"{list(out.shape)}, max |diff| {diff:.3e}",
        )

    x3o = torch.randn(1, 8, 8, 8, 8, device=MPS)
    w3o = torch.randn(16, 8, 3, 3, 3, device=MPS)
    check_out(
        "slow_conv3d_forward.output",
        lambda: ATEN.slow_conv3d_forward(x3o, w3o, [3, 3, 3], None, [1, 1, 1], [0, 0, 0]),
        lambda out: ATEN.slow_conv3d_forward.output(
            x3o, w3o, [3, 3, 3], None, [1, 1, 1], [0, 0, 0], output=out
        ),
    )

    x2o = torch.randn(1, 8, 8, 8, device=MPS)
    w2o = torch.randn(8, 16, 3, 3, device=MPS)
    check_out(
        "slow_conv_transpose2d.out",
        lambda: ATEN.slow_conv_transpose2d(x2o, w2o, [3, 3], None, [2, 2], [1, 1], [1, 1], [1, 1]),
        lambda out: ATEN.slow_conv_transpose2d.out(
            x2o, w2o, [3, 3], None, [2, 2], [1, 1], [1, 1], [1, 1], out=out
        ),
    )
    x3t = torch.randn(1, 8, 6, 6, 6, device=MPS)
    w3t = torch.randn(8, 16, 3, 3, 3, device=MPS)
    check_out(
        "slow_conv_transpose3d.out",
        lambda: ATEN.slow_conv_transpose3d(
            x3t, w3t, [3, 3, 3], None, [2, 2, 2], [1, 1, 1], [1, 1, 1], [1, 1, 1],
        ),
        lambda out: ATEN.slow_conv_transpose3d.out(
            x3t, w3t, [3, 3, 3], None, [2, 2, 2], [1, 1, 1], [1, 1, 1], [1, 1, 1], out=out,
        ),
    )
    w2d = torch.randn(16, 8, 3, 3, device=MPS)
    check_out(
        "slow_conv_dilated2d.out",
        lambda: ATEN.slow_conv_dilated2d(
            x2o, w2d, [3, 3], None, [1, 1], [0, 0], [1, 1]
        ),
        lambda out: ATEN.slow_conv_dilated2d.out(
            x2o, w2d, [3, 3], None, [1, 1], [0, 0], [1, 1], out=out,
        ),
    )

    # Argument checks that the CPU kernel also rejects.
    for name, call in (
        (
            "reject_kernel_size_mismatch",
            lambda: ATEN.slow_conv3d_forward(
                torch.randn(1, 8, 8, 8, 8, device=MPS),
                torch.randn(16, 8, 3, 3, 3, device=MPS),
                [2, 2, 2], None, [1, 1, 1], [0, 0, 0],
            ),
        ),
        (
            "reject_negative_padding",
            lambda: ATEN.slow_conv_dilated2d(
                torch.randn(1, 8, 8, 8, device=MPS),
                torch.randn(16, 8, 3, 3, device=MPS),
                [3, 3], None, [1, 1], [-1, -1], [1, 1],
            ),
        ),
        (
            "reject_output_padding",
            lambda: ATEN.slow_conv_transpose2d(
                torch.randn(1, 8, 8, 8, device=MPS),
                torch.randn(8, 16, 3, 3, device=MPS),
                [3, 3], None, [1, 1], [0, 0], [1, 1], [1, 1],
            ),
        ),
    ):
        try:
            call()
            record(name, False, "no error")
        except ValueError as exc:
            record(name, True, str(exc)[:100])
        except Exception as exc:  # unexpected type
            record(name, False, f"{type(exc).__name__}: {exc}")

    # No recursion: the dispatcher enters each overlay once per call. The
    # counters wrap every overlay installed by ``install_overlays``.
    CALLS.clear()
    ATEN.slow_conv3d_forward(
        torch.randn(1, 8, 6, 6, 6, device=MPS),
        torch.randn(16, 8, 3, 3, 3, device=MPS),
        [3, 3, 3], None, [1, 1, 1], [0, 0, 0],
    )
    record("no_recursion", CALLS == {"slow_conv3d_forward": 1}, f"entries {CALLS}")

    # No CPU round trip: every op dispatched by the overlay sees only MPS
    # tensors.
    trace = DeviceTrace()
    with trace:
        ATEN.slow_conv_transpose3d(
            torch.randn(1, 8, 6, 6, 6, device=MPS),
            torch.randn(8, 16, 3, 3, 3, device=MPS),
            [3, 3, 3], None, [2, 2, 2], [1, 1, 1], [1, 1, 1], [1, 1, 1],
        )
    torch.mps.synchronize()
    record("no_cpu_round_trip", trace.devices == {"mps"}, f"{sorted(trace.devices)}")
    return checks


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available")

    results = {
        "torch": torch.__version__,
        "device": DEVICE,
        "module": "metal_kernelops.convolution_ops",
        "machine_load_at_start": [round(value, 2) for value in os.getloadavg()],
        "timing": {
            "clock": "time.perf_counter",
            "warmup": WARMUP,
            "iterations": ITERS,
            "batches": {"mps": BATCHES, "cpu": 2},
            "sync": "torch.mps.synchronize() after warmup and after each batch",
            "statistic": "median of per-batch samples; overhead_us and "
            "overhead_ratio are the median of the paired differences within "
            "each batch; mps_min_ms and native_min_ms preserve the minima",
            "mps_fields": {
                "mps_ms": "overlay: submission plus execution",
                "native_ms": "aten::convolution / convolution_backward, same inputs",
                "native_repeat_ms": "the same native call measured a second time",
                "overhead_us": "mps_ms - native_ms, in microseconds",
                "enqueue_ms": "overlay submission only, no wait for the GPU",
                "enqueue_overhead_us": "enqueue_ms - native_enqueue_ms: the Python "
                "work the alias adds",
                "noise_us": "|native_ms - native_repeat_ms|: two identical calls "
                "differ by this much on this machine",
                "batch_spread_us": "max - min over batches, per function: how far "
                "the load moved during the run",
                "copy_floor_ms": "a device copy of the same output, same batch",
                "copy_floor_gbps": "that copy's achieved bandwidth, read plus write",
            },
            "note": "shared machine: other MPS jobs run concurrently. An overhead "
            "smaller than noise_us is not a measurement of the alias.",
        },
        "overlays": [baseline(case) for case in CASES + [BACKWARD_CASE]],
        "limitations": [
            "slow_conv_transpose3d: the native MPS kernel has no fp16/bf16 "
            "variant ('ConvTranspose 3D with BF16 or FP16 types is not "
            "supported on MPS'); fp32 only.",
            "slow_conv_transpose2d with an unbatched (3-D) input: forward "
            "works, backward raises because the generated autograd formula "
            "passes the 3-D input to aten::convolution_backward, which "
            "rejects unbatched input on CPU and MPS alike.",
            "slow_conv_dilated2d / slow_conv_dilated3d with an unbatched "
            "input: backward raises for the same reason on CPU and MPS; "
            "forward handles the unbatched case.",
            "A kernel larger than the padded input: the CPU kernels raise, the "
            "Metal kernel returns a zero-sized output, so MPS and CPU disagree "
            "on that input.",
        ],
    }

    # The Library object owns the registrations: dropping it unregisters them.
    library = install_overlays()

    cases = []
    for case in CASES:
        cases.extend(run_case(case))
    cases.extend(run_backward_case())
    results["cases"] = cases

    semantics = check_semantics()
    results["semantics"] = semantics

    value_limits = {
        "float32": 1e-5,
        "float16": 2e-3,
        "bfloat16": 2e-2,
    }
    passed = True
    for row in cases:
        if not row["supported"]:
            continue
        passed &= row["shape_match"] and row["dtype_match"]
        passed &= row["value_rel_error"] <= value_limits[row["dtype"]]
        if "grad_rel_error" in row:
            passed &= all(value <= 1e-5 for value in row["grad_rel_error"].values())
    for check in semantics:
        passed &= check["passed"]
    results["verdict"] = "pass" if passed else "fail"
    results["machine_load_at_end"] = [round(value, 2) for value in os.getloadavg()]

    out = ROOT / "measured" / "convolution.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1) + "\n")

    print(
        f"{'case':22s} {'dtype':9s} {'value':>10s} {'grad':>10s} {'mps ms':>8s} "
        f"{'native':>8s} {'over us':>8s} {'enq us':>7s} {'enq ov':>7s} {'noise':>7s} "
        f"{'copy ms':>8s} {'GB/s':>6s} {'spread':>7s} {'cpu':>8s}"
    )
    for row in cases:
        if not row["supported"]:
            print(f"{row['name']:22s} {row['dtype']:9s} unsupported by the native MPS kernel: {row['error']}")
            continue
        grad = ""
        if "grad_rel_error" in row:
            grad = f"{max(row['grad_rel_error'].values()):.2e}"
        spread = max(row["batch_spread_us"].values())
        print(
            f"{row['name']:22s} {row['dtype']:9s} {row['value_rel_error']:10.2e} "
            f"{grad:>10s} {row['mps_ms']:8.3f} {row['native_ms']:8.3f} "
            f"{row['overhead_us']:8.2f} {row['enqueue_ms'] * 1000:7.2f} "
            f"{row['enqueue_overhead_us']:7.2f} {row['noise_us']:7.2f} "
            f"{row['copy_floor_ms']:8.3f} {row['copy_floor_gbps']:6.0f} "
            f"{spread:7.2f} {row['cpu_ms']:8.3f}"
        )
    for check in semantics:
        print(f"{'PASS' if check['passed'] else 'FAIL'}  {check['check']}  {check['detail']}")
    print(f"verdict: {results['verdict']}  ->  {out.relative_to(ROOT)}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
