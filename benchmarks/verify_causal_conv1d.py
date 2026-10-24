"""Correctness, gradient, step-parity and timing check for causal_conv1d.

Run from the repository root:

    uv run --project /Users/dbydd/vibe-agent-working-dir/ResearchChamber \
        python benchmarks/verify_causal_conv1d.py

What it records:

* forward parity of the Metal kernels against ``causal_conv1d_ref`` evaluated in
  CPU float64, including the final state;
* the reference's own gradient against central finite differences in float64,
  then the Metal analytic gradient against the reference gradient;
* decode parity: ``causal_conv1d_update`` applied token by token (and in chunks)
  must reproduce ``causal_conv1d_fn`` over the whole sequence, output and state;
* layout and edge semantics: empty sequence, one-token sequence, ``K = 1``,
  non-contiguous input, ``(channels, K)`` weight, a state that is a slice of a
  larger cache, and the absence of ``aten::convolution`` on the kernel path;
* warmed, synchronized timings for forward, forward+backward and the decode
  update, against the equivalent ``conv1d`` composition.
* a per-stage profile of the same three shapes: the forward kernel launch, the
  fused backward launch (``dx`` and ``dweight`` in one pass), the ``dweight``
  finalize launch, and the whole autograd step, each against a streaming copy
  of the same byte count measured in the same batch.  The copy is the physical
  floor those stages are measured against.

Timing runs under a tier-1 compute lease when the workspace broker is on this
machine.  Results go to ``measured/causal_conv1d.json``.
"""

import contextlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metal_kernelops.cv import causal_conv1d as cc  # noqa: E402

BROKER = Path("/Users/dbydd/vibe-agent-working-dir/ResearchChamber/shared_scripts")
DEVICE = "mps"
MPS = torch.device(DEVICE)
WARMUP = 5
ITERS = 30
ROUNDS = 9
PROFILE_ROUNDS = 20

VALUE_LIMITS = {"float32": 1e-6, "float16": 3e-3}
GRAD_LIMITS = {"float32": 1e-5, "float16": 3e-2}

_STREAM = r"""
#include <metal_stdlib>
using namespace metal;

kernel void stream_copy(device const float4* src, device float4* dst,
                        uint gid [[thread_position_in_grid]]) {
    dst[gid] = src[gid];
}
"""
_stream_library = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def rel_error(actual, expected):
    actual = actual.detach().cpu().double()
    expected = expected.detach().cpu().double()
    denominator = expected.norm()
    if denominator == 0:
        return float((actual - expected).norm())
    return float((actual - expected).norm() / denominator)


def time_mps(*fns, rounds=ROUNDS, iters=ITERS):
    """Milliseconds per call, rounds interleaved across the functions."""
    for fn in fns:
        for _ in range(WARMUP):
            fn()
    torch.mps.synchronize()
    best = [float("inf")] * len(fns)
    for _ in range(rounds):
        for index, fn in enumerate(fns):
            start = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.mps.synchronize()
            best[index] = min(best[index], (time.perf_counter() - start) / iters * 1000.0)
    return best


def aten_forward(x, weight, bias, activation, initial_state):
    """The composition the kernels replace: conv1d with an explicit causal pad."""
    if weight.dim() == 3:
        weight = weight[:, 0, :]
    channels, width = weight.shape
    taps = width - 1
    length = x.shape[-1]
    if initial_state is not None:
        joined = torch.cat([initial_state, x], dim=-1)
        out = F.conv1d(joined, weight.unsqueeze(1), bias, groups=channels)
    else:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=taps, groups=channels)[..., :length]
    return F.silu(out) if activation is not None else out


def aten_update(x, state, weight, bias, activation):
    if weight.dim() == 3:
        weight = weight[:, 0, :]
    squeezed = x.dim() == 2
    if squeezed:
        x = x.unsqueeze(-1)
    channels = weight.shape[0]
    length = x.shape[-1]
    joined = torch.cat([state, x], dim=-1)
    out = F.conv1d(joined, weight.unsqueeze(1), bias, groups=channels)[..., -length:]
    state.copy_(joined[..., -state.shape[-1]:])
    out = F.silu(out) if activation is not None else out
    return out.squeeze(-1) if squeezed else out


class OpTrace(TorchDispatchMode):
    """Records the aten operators and tensor devices a call actually touches."""

    def __init__(self):
        super().__init__()
        self.ops = []
        self.devices = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        for value in list(args) + list((kwargs or {}).values()):
            if isinstance(value, torch.Tensor):
                self.devices.add(value.device.type)
        return func(*args, **(kwargs or {}))


def make(shape, dtype, device, seed=0, scale=1.0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float64).to(
        device=device, dtype=dtype
    ) * scale


def name_of(dtype):
    return str(dtype).replace("torch.", "")


def lease(task_name):
    """Tier-1 lease when the workspace broker is installed, nullcontext otherwise."""
    if not (BROKER / "compute_broker.py").is_file():
        return contextlib.nullcontext({"acquired": False, "reason": "broker not found"})
    sys.path.insert(0, str(BROKER))
    from compute_broker import lease_guard

    return lease_guard("mps-kernelops", tier=1, task_name=task_name)


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #

# name, batch, channels, length, taps, activation, initial_state, dtype
CASES = [
    ("decode-1x2048", 1, 2048, 1, 4, "silu", True, torch.float32),
    ("decode-8x2048", 8, 2048, 1, 4, "silu", True, torch.float32),
    ("decode-1x768", 1, 768, 1, 4, "silu", True, torch.float32),
    ("decode-16x1536-fp16", 16, 1536, 1, 4, "silu", True, torch.float16),
    ("short-2x256x16", 2, 256, 16, 4, "silu", False, torch.float32),
    ("prefill-1x768x4096", 1, 768, 4096, 4, "silu", True, torch.float32),
    ("prefill-8x512x2048", 8, 512, 2048, 4, "silu", False, torch.float32),
    ("prefill-16x768x1024-fp16", 16, 768, 1024, 4, "silu", True, torch.float16),
    ("narrow-4x128x200-k8", 4, 128, 200, 8, None, False, torch.float32),
    ("k1-2x64x33", 2, 64, 33, 1, None, True, torch.float32),
]

# name, batch, channels, length, taps, activation, state columns, dtype
UPDATES = [
    ("update-1x2048x1", 1, 2048, 1, 4, "silu", 3, torch.float32),
    ("update-8x2048x1", 8, 2048, 1, 4, "silu", 3, torch.float32),
    ("update-64x2048x1", 64, 2048, 1, 4, "silu", 3, torch.float32),
    ("update-1x768x1", 1, 768, 1, 4, "silu", 3, torch.float32),
    ("update-1x2048x4", 1, 2048, 4, 4, "silu", 3, torch.float32),
    ("update-8x2048x1-wide", 8, 2048, 1, 4, "silu", 8, torch.float32),
    ("update-16x1536x1-fp16", 16, 1536, 1, 4, "silu", 3, torch.float16),
]

# name, batch, channels, length, taps, activation, initial_state, dtype
PROFILES = [
    ("prefill-1x768x4096", 1, 768, 4096, 4, "silu", True, torch.float32),
    ("prefill-8x512x2048", 8, 512, 2048, 4, "silu", False, torch.float32),
    ("prefill-16x768x1024-fp16", 16, 768, 1024, 4, "silu", True, torch.float16),
]


# --------------------------------------------------------------------------- #
# Forward, gradient, update
# --------------------------------------------------------------------------- #

def run_case(name, batch, channels, length, taps, activation, has_state, dtype):
    weight = make((channels, 1, taps), dtype, MPS, seed=1, scale=0.5)
    bias = make((channels,), dtype, MPS, seed=2)
    x = make((batch, channels, length), dtype, MPS, seed=3)
    state = make((batch, channels, taps - 1), dtype, MPS, seed=4) if has_state else None

    out, final = cc.causal_conv1d_fn(x, weight, bias, activation, state, True)
    torch.mps.synchronize()
    ref_out, ref_final = cc.causal_conv1d_ref(
        x.cpu().double(), weight.cpu().double(), bias.cpu().double(), activation,
        state.cpu().double() if state is not None else None, True,
    )
    kernel_ms, aten_ms = time_mps(
        lambda: cc.causal_conv1d_fn(x, weight, bias, activation, state),
        lambda: aten_forward(x, weight, bias, activation, state),
    )
    return {
        "name": name,
        "dtype": name_of(dtype),
        "shape": [batch, channels, length],
        "taps": taps,
        "activation": activation,
        "initial_state": has_state,
        "out_rel_error": rel_error(out, ref_out),
        "final_state_rel_error": rel_error(final, ref_final) if ref_final.numel() else 0.0,
        "kernel_ms": kernel_ms,
        "aten_ms": aten_ms,
        "speedup": aten_ms / kernel_ms,
    }


def run_gradient(name, batch, channels, length, taps, activation, has_state, dtype):
    weight = make((channels, 1, taps), dtype, MPS, seed=11, scale=0.5).requires_grad_(True)
    bias = make((channels,), dtype, MPS, seed=12).requires_grad_(True)
    x = make((batch, channels, length), dtype, MPS, seed=13).requires_grad_(True)
    state = (
        make((batch, channels, taps - 1), dtype, MPS, seed=14).requires_grad_(True)
        if has_state else None
    )
    seed_out = make((batch, channels, length), torch.float32, MPS, seed=15).to(dtype)
    seed_final = (
        make((batch, channels, taps - 1), torch.float32, MPS, seed=16).to(dtype)
        if has_state and taps > 1 else None
    )

    out, final = cc.causal_conv1d_fn(x, weight, bias, activation, state, True)
    loss = (out * seed_out).sum()
    if seed_final is not None:
        loss = loss + (final * seed_final).sum()
    loss.backward()
    torch.mps.synchronize()
    got = {"x": x.grad.clone(), "weight": weight.grad.clone(), "bias": bias.grad.clone()}
    if state is not None:
        got["initial_state"] = (
            state.grad.clone() if state.grad is not None else torch.zeros_like(state)
        )

    rx = x.detach().cpu().double().requires_grad_(True)
    rw = weight.detach().cpu().double().requires_grad_(True)
    rb = bias.detach().cpu().double().requires_grad_(True)
    rs = state.detach().cpu().double().requires_grad_(True) if state is not None else None
    ref_out, ref_final = cc.causal_conv1d_ref(rx, rw, rb, activation, rs, True)
    ref_loss = (ref_out * seed_out.cpu().double()).sum()
    if seed_final is not None:
        ref_loss = ref_loss + (ref_final * seed_final.cpu().double()).sum()
    ref_loss.backward()
    want = {"x": rx.grad, "weight": rw.grad, "bias": rb.grad}
    if rs is not None:
        want["initial_state"] = rs.grad if rs.grad is not None else torch.zeros_like(state)
    errors = {key: rel_error(got[key], want[key]) for key in want}

    def step(fn):
        def run():
            detached_x = x.detach().requires_grad_(True)
            detached_w = weight.detach().requires_grad_(True)
            detached_b = bias.detach().requires_grad_(True)
            detached_s = state.detach().requires_grad_(True) if state is not None else None
            produced = fn(detached_x, detached_w, detached_b, activation, detached_s, True)
            if isinstance(produced, tuple):
                produced = produced[0]
            torch.autograd.grad(produced, (detached_x, detached_w, detached_b), seed_out)
        return run

    kernel_ms, aten_ms = time_mps(
        step(cc.causal_conv1d_fn),
        step(lambda *args: aten_forward(*args[:5])),
    )
    return {
        "name": name,
        "dtype": name_of(dtype),
        "shape": [batch, channels, length],
        "taps": taps,
        "activation": activation,
        "initial_state": has_state,
        "grad_rel_error": errors,
        "kernel_ms": kernel_ms,
        "aten_ms": aten_ms,
        "speedup": aten_ms / kernel_ms,
    }


def run_update(name, batch, channels, length, taps, activation, columns, dtype):
    weight = make((channels, 1, taps), dtype, MPS, seed=21, scale=0.5)
    bias = make((channels,), dtype, MPS, seed=22)
    x = make((batch, channels, length), dtype, MPS, seed=23)
    state_reference = make((batch, channels, columns), dtype, MPS, seed=24)
    state_kernel = state_reference.clone()
    reference = aten_update(x, state_reference, weight, bias, activation)
    out = cc.causal_conv1d_update(x, state_kernel, weight, bias, activation)
    torch.mps.synchronize()
    kernel_ms, aten_ms = time_mps(
        lambda: cc.causal_conv1d_update(x, state_kernel, weight, bias, activation),
        lambda: aten_update(x, state_reference, weight, bias, activation),
    )
    return {
        "name": name,
        "dtype": name_of(dtype),
        "shape": [batch, channels, length],
        "taps": taps,
        "activation": activation,
        "state_columns": columns,
        "out_rel_error": rel_error(out, reference),
        "state_rel_error": rel_error(state_kernel, state_reference),
        "kernel_ms": kernel_ms,
        "aten_ms": aten_ms,
        "speedup": aten_ms / kernel_ms,
    }


def stream_copy(nbytes):
    """A float4 copy moving ``nbytes`` in total, half read and half written."""
    global _stream_library
    if _stream_library is None:
        _stream_library = torch.mps.compile_shader(_STREAM)
    words = max(1, nbytes // 8 // 4)
    source = torch.randn(words * 4, device=MPS)
    target = torch.empty_like(source)
    library = _stream_library
    return lambda: library.stream_copy(source, target, words, threads=words, group_size=256)


def run_profile(name, batch, channels, length, taps, activation, has_state, dtype):
    """Every launch of one shape on its own, against a copy of the same bytes.

    ``dx`` and ``dweight`` share one launch, so the fused backward is one row
    here; the finalize that folds the per-threadgroup ``dweight`` partials is
    the other.
    """
    weight = make((channels, 1, taps), dtype, MPS, seed=1, scale=0.5)
    kernel_weight = weight.view(channels, taps)
    bias = make((channels,), dtype, MPS, seed=2)
    x = make((batch, channels, length), dtype, MPS, seed=3)
    state = make((batch, channels, taps - 1), dtype, MPS, seed=4) if has_state else None
    dout = make((batch, channels, length), dtype, MPS, seed=5)
    seed = make((batch, channels, length), dtype, MPS, seed=6)

    out = torch.empty_like(x)
    final = x.new_empty((batch, channels, taps - 1))
    vector = cc._vector((x,), length, 16 // x.element_size())
    back_vector = cc._vector((x, dout), length, 4)
    blocks = length // back_vector
    groups = min(-(-blocks // cc._TILE), max(1, cc._TARGET_GROUPS // (batch * channels)))
    npc = batch * groups
    dx = torch.empty_like(x)
    dstate = torch.empty((batch, channels, taps - 1), dtype=dtype, device=MPS)
    partial = torch.empty(channels * npc * (taps + 1), dtype=torch.float32, device=MPS)
    dweight = torch.empty((channels, taps), dtype=dtype, device=MPS)
    dbias = torch.empty((channels,), dtype=dtype, device=MPS)
    back = cc._shader(cc._BACKWARD, dtype, taps, back_vector)
    fold = cc._shader(cc._FINALIZE, dtype, taps)
    init = state if state is not None else x
    silu = 1 if activation is not None else 0

    def forward_kernel():
        cc._forward_metal(x, kernel_weight, bias, state, activation, out, final, False)

    def backward_kernel():
        back.causal_conv1d_bwd(
            dout, x, init, kernel_weight, bias, dx, dstate, partial,
            channels, length, blocks, groups, npc,
            1 if state is not None else 0, 1, silu, 1,
            threads=batch * channels * groups * cc._TILE, group_size=cc._TILE,
        )

    def dweight_finalize():
        fold.causal_conv1d_dwfinal(
            partial, dweight, dbias, npc, 1, 1,
            threads=channels * (taps + 1), group_size=64,
        )

    def step():
        detached_x = x.detach().requires_grad_(True)
        detached_w = weight.detach().requires_grad_(True)
        detached_b = bias.detach().requires_grad_(True)
        produced = cc.causal_conv1d_fn(detached_x, detached_w, detached_b, activation, state)
        torch.autograd.grad(produced, (detached_x, detached_w, detached_b), seed)

    one = batch * channels * length * x.element_size()
    staged = {
        "forward": (lambda: cc.causal_conv1d_fn(x, weight, bias, activation, state), 2 * one),
        "forward_kernel": (forward_kernel, 2 * one),
        "backward_kernel": (backward_kernel, 3 * one),
        "dweight_finalize": (
            dweight_finalize,
            channels * npc * (taps + 1) * 4 + channels * taps * x.element_size(),
        ),
        "forward_backward": (step, 5 * one),
        "stream_copy": (stream_copy(2 * one), 2 * one),
    }
    times = time_mps(*(fn for fn, _ in staged.values()), rounds=PROFILE_ROUNDS)
    measured = dict(zip(staged, times))
    floor = 2 * one / measured["stream_copy"] / 1e6

    stages = []
    for label, (_, nbytes) in staged.items():
        ms = measured[label]
        stages.append({
            "stage": label,
            "bytes": nbytes,
            "ms": ms,
            "gbs": nbytes / ms / 1e6,
            "of_floor": (nbytes / ms / 1e6) / floor,
        })
    return {
        "name": name,
        "dtype": name_of(dtype),
        "shape": [batch, channels, length],
        "taps": taps,
        "activation": activation,
        "initial_state": has_state,
        "vector": vector,
        "backward_vector": back_vector,
        "blocks_per_row": blocks,
        "groups_per_row": groups,
        "floor_gbs": floor,
        "stages": stages,
    }


# --------------------------------------------------------------------------- #
# Checks that one batched case cannot express
# --------------------------------------------------------------------------- #

def record(checks, name, passed, detail=""):
    checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})


def check_semantics():
    checks = []

    # Empty sequence: no convolution, the state passes through untouched.
    weight = make((8, 1, 4), torch.float32, MPS, seed=31)
    x = torch.empty(2, 8, 0, device=MPS)
    state = make((2, 8, 3), torch.float32, MPS, seed=32)
    out, final = cc.causal_conv1d_fn(x, weight, None, "silu", state, True)
    record(checks, "empty_sequence",
           tuple(out.shape) == (2, 8, 0) and torch.equal(final, state),
           f"out {tuple(out.shape)}, final == state")
    buffer = make((2, 8, 3), torch.float32, MPS, seed=33)
    before = buffer.clone()
    cc.causal_conv1d_update(torch.empty(2, 8, 0, device=MPS), buffer, weight)
    record(checks, "empty_update", torch.equal(buffer, before), "state unchanged")

    # One token, no state: the output sees only the last tap.
    x = make((1, 4, 1), torch.float32, MPS, seed=34)
    weight = make((4, 1, 4), torch.float32, MPS, seed=35, scale=0.5)
    out = cc.causal_conv1d_fn(x, weight, None, None)
    expect = x * weight[:, 0, 3].view(1, 4, 1)
    record(checks, "length_one_takes_last_tap", rel_error(out, expect) < 1e-6,
           f"{rel_error(out, expect):.2e}")

    # K = 1: no state, output is a scaled copy.
    weight = make((4, 1, 1), torch.float32, MPS, seed=36, scale=0.5)
    x = make((2, 4, 7), torch.float32, MPS, seed=37)
    out, final = cc.causal_conv1d_fn(x, weight, None, None, None, True)
    expect = x * weight[:, 0, 0].view(1, 4, 1)
    record(checks, "k1_scaled_copy",
           rel_error(out, expect) < 1e-6 and tuple(final.shape) == (2, 4, 0),
           f"{rel_error(out, expect):.2e}, final {tuple(final.shape)}")

    # Non-contiguous input: a transposed view, and a strided slice.
    weight = make((16, 1, 4), torch.float32, MPS, seed=38, scale=0.5)
    transposed = make((2, 40, 16), torch.float32, MPS, seed=39).transpose(1, 2)
    out = cc.causal_conv1d_fn(transposed, weight, None, "silu")
    ref = cc.causal_conv1d_ref(transposed.cpu().double(), weight.cpu().double(), None, "silu")
    record(checks, "non_contiguous_input", rel_error(out, ref) < 1e-6,
           f"stride {transposed.stride()} err {rel_error(out, ref):.2e}")
    strided = make((2, 16, 64), torch.float32, MPS, seed=40)[..., 1::2]
    out = cc.causal_conv1d_fn(strided, weight, None, None)
    ref = cc.causal_conv1d_ref(strided.cpu().double(), weight.cpu().double(), None, None)
    record(checks, "strided_input", rel_error(out, ref) < 1e-6,
           f"stride {strided.stride()} err {rel_error(out, ref):.2e}")

    # The upstream (channels, K) weight layout is the same convolution.
    packed = make((16, 1, 4), torch.float32, MPS, seed=41, scale=0.5)
    x = make((2, 16, 40), torch.float32, MPS, seed=42)
    a = cc.causal_conv1d_fn(x, packed, None, "silu")
    b = cc.causal_conv1d_fn(x, packed[:, 0, :], None, "silu")
    record(checks, "weight_layout", torch.equal(a, b), "(C,1,K) == (C,K)")

    # A state that is a slice of a larger cache is read and written in place.
    cache = make((2, 16, 9), torch.float32, MPS, seed=43)
    view = cache[..., :3]
    x = make((2, 16), torch.float32, MPS, seed=44)
    weight = make((16, 1, 4), torch.float32, MPS, seed=45, scale=0.5)
    expected = cache.clone()
    before = cache.clone()
    reference = aten_update(x, expected[..., :3], weight, None, "silu")
    out = cc.causal_conv1d_update(x, view, weight, None, "silu")
    torch.mps.synchronize()
    record(checks, "cache_slice_state",
           not torch.equal(cache[..., :3], before[..., :3])
           and torch.equal(cache[..., 3:], before[..., 3:])
           and rel_error(out, reference) < 1e-6
           and rel_error(cache[..., :3], expected[..., :3]) < 1e-6,
           f"err {rel_error(out, reference):.2e}, view {view.shape} stride {view.stride()}")

    # The kernel path must not reach aten::convolution, and must stay on the device.
    probe = x.unsqueeze(-1).expand(-1, -1, 8).contiguous()
    trace = OpTrace()
    with trace:
        cc.causal_conv1d_fn(probe, weight, None, "silu")
    torch.mps.synchronize()
    record(checks, "no_convolution_on_kernel_path",
           not any("convolution" in op for op in trace.ops) and trace.devices == {"mps"},
           f"{len(trace.ops)} aten ops, devices {sorted(trace.devices)}")
    trace = OpTrace()
    with trace:
        aten_forward(probe, weight[:, 0, :], None, "silu", None)
    torch.mps.synchronize()
    record(checks, "baseline_uses_convolution",
           any("convolution" in op for op in trace.ops), f"{len(trace.ops)} aten ops")

    # Decode is inference only, and says so.
    try:
        cc.causal_conv1d_update(
            make((2, 16), torch.float32, MPS, seed=46).requires_grad_(True),
            make((2, 16, 3), torch.float32, MPS, seed=47), weight,
        )
        raised = False
    except RuntimeError:
        raised = True
    record(checks, "update_rejects_grad", raised, "requires_grad x raises")

    # bfloat16 has no Metal kernel and must still be correct through the composition.
    weight = make((8, 1, 4), torch.bfloat16, MPS, seed=48, scale=0.5)
    x = make((2, 8, 24), torch.bfloat16, MPS, seed=49)
    out = cc.causal_conv1d_fn(x, weight, None, "silu")
    ref = cc.causal_conv1d_ref(x.cpu().double(), weight.cpu().double(), None, "silu")
    record(checks, "bfloat16_fallback",
           out.dtype == torch.bfloat16 and rel_error(out, ref) < 2e-2,
           f"dtype {out.dtype}, err {rel_error(out, ref):.2e}")

    # Mixed dtypes: arithmetic in weight.dtype, result back in x.dtype.
    weight = make((8, 1, 4), torch.float32, MPS, seed=52, scale=0.5).requires_grad_(True)
    x = make((2, 8, 12), torch.float16, MPS, seed=53).requires_grad_(True)
    out = cc.causal_conv1d_fn(x, weight, None, "silu")
    out.sum().backward()
    ref = cc.causal_conv1d_ref(
        x.detach().cpu().double(), weight.detach().cpu().double(), None, "silu"
    )
    record(checks, "mixed_dtype",
           out.dtype == torch.float16 and x.grad.dtype == torch.float16
           and weight.grad.dtype == torch.float32 and rel_error(out, ref) < 3e-3,
           f"out {out.dtype}, grads {x.grad.dtype}/{weight.grad.dtype}, "
           f"err {rel_error(out, ref):.2e}")

    # A CPU tensor runs the composition with autograd intact.
    cpu = torch.device("cpu")
    weight = make((8, 1, 4), torch.float32, cpu, seed=50, scale=0.5).requires_grad_(True)
    x = make((2, 8, 24), torch.float32, cpu, seed=51)
    out = cc.causal_conv1d_fn(x, weight, None, "silu")
    out.sum().backward()
    record(checks, "cpu_autograd",
           weight.grad is not None and bool(torch.isfinite(weight.grad).all()),
           f"grad norm {float(weight.grad.norm()):.3e}")
    return checks


def check_step_parity():
    checks = []
    for dtype in (torch.float32, torch.float16):
        tolerance = VALUE_LIMITS[name_of(dtype)]
        for batch, channels, length in ((2, 64, 9), (1, 256, 17), (3, 32, 4)):
            weight = make((channels, 1, 4), dtype, MPS, seed=51, scale=0.5)
            bias = make((channels,), dtype, MPS, seed=52)
            x = make((batch, channels, length), dtype, MPS, seed=53)
            state = make((batch, channels, 3), dtype, MPS, seed=54)
            full, final = cc.causal_conv1d_fn(x, weight, bias, "silu", state, True)

            stepping = state.clone()
            pieces = [cc.causal_conv1d_update(x[..., t], stepping, weight, bias, "silu")
                      for t in range(length)]
            step_out = torch.stack(pieces, dim=-1)
            torch.mps.synchronize()
            out_error = rel_error(step_out, full)
            state_error = rel_error(stepping, final)
            record(checks, f"step_vs_full_{name_of(dtype)}_{batch}x{channels}x{length}",
                   out_error < tolerance and state_error < tolerance,
                   f"out {out_error:.2e}, state {state_error:.2e}")

            chunked = state.clone()
            chunks = [cc.causal_conv1d_update(x[..., i:i + 3], chunked, weight, bias, "silu")
                      for i in range(0, length, 3)]
            torch.mps.synchronize()
            out_error = rel_error(torch.cat(chunks, dim=-1), full)
            state_error = rel_error(chunked, final)
            record(checks, f"chunked_vs_full_{name_of(dtype)}_{batch}x{channels}x{length}",
                   out_error < tolerance and state_error < tolerance,
                   f"out {out_error:.2e}, state {state_error:.2e}")
    return checks


def check_reference_finite_difference():
    """Central differences in float64 on the reference, all four inputs."""
    torch.manual_seed(7)
    batch, channels, length, taps = 2, 3, 7, 4
    x = torch.randn(batch, channels, length, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(channels, 1, taps, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(channels, dtype=torch.float64, requires_grad=True)
    state = torch.randn(batch, channels, taps - 1, dtype=torch.float64, requires_grad=True)
    out, final = cc.causal_conv1d_ref(x, weight, bias, "silu", state, True)
    seed_out = torch.randn_like(out)
    seed_final = torch.randn_like(final)
    (out * seed_out).sum().add((final * seed_final).sum()).backward()

    def loss():
        out, final = cc.causal_conv1d_ref(
            x.detach(), weight.detach(), bias.detach(), "silu", state.detach(), True
        )
        return float((out * seed_out).sum() + (final * seed_final).sum())

    epsilon = 1e-6
    rows = []
    for name, tensor in (("x", x), ("weight", weight), ("bias", bias), ("initial_state", state)):
        numeric = torch.zeros_like(tensor)
        flat = tensor.detach().view(-1)
        for index in range(flat.numel()):
            original = float(flat[index])
            flat[index] = original + epsilon
            plus = loss()
            flat[index] = original - epsilon
            minus = loss()
            flat[index] = original
            numeric.view(-1)[index] = (plus - minus) / (2 * epsilon)
        error = rel_error(numeric, tensor.grad)
        rows.append({"input": name, "rel_error": error, "passed": error < 1e-7})
    return rows


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available")

    results = {
        "torch": torch.__version__,
        "device": DEVICE,
        "module": "metal_kernelops.cv.causal_conv1d",
        "op": "causal_conv1d_fn / causal_conv1d_update (Dao-AILab/causal-conv1d semantics)",
        "api": {
            "causal_conv1d_fn": "x (B, C, L); weight (C, 1, K) or (C, K); bias (C,) or None; "
                                "activation None|'silu'|'swish'; initial_state (B, C, K-1) or None; "
                                "return_final_state -> out (B, C, L) [, final_state (B, C, K-1)]",
            "causal_conv1d_update": "x (B, C) or (B, C, q); state (B, C, s >= K-1) updated in place; "
                                    "weight (C, 1, K) or (C, K); bias (C,) or None; "
                                    "activation None|'silu'|'swish' -> out with the rank of x",
            "causal_conv1d_ref": "same contract as causal_conv1d_fn; explicit taps, no convolution, "
                                 "differentiable, any device and dtype",
        },
        "timing": {
            "clock": "time.perf_counter",
            "warmup": WARMUP,
            "iterations": ITERS,
            "rounds": ROUNDS,
            "sync": "torch.mps.synchronize() after warmup and after each round",
            "statistic": "best round per call, rounds interleaved across the two functions",
            "baseline": "conv1d with an explicit causal pad (aten_forward / aten_update in the "
                        "benchmark), which is what causal_conv1d_fn lowers to without the kernels",
            "traffic": {
                "forward": "read x, write out (2 tensors); the final state rides along",
                "backward": "read dout, read x, write dx (3 tensors), one fused pass for "
                            "dx and dweight, with the pre-activation recomputed from x",
                "dweight_finalize": "the per-threadgroup partials of every batch row, folded",
                "floor": "a float4 copy moving the same byte count in the same batch",
            },
            "note": "shared machine, about 25 concurrent MPS jobs: each row is a minimum over "
                    "interleaved rounds inside its own batch, so rows from different batches "
                    "are not directly comparable and a busy batch inflates every number in "
                    "it. The profile compares each stage with of_floor, a copy of the same "
                    "traffic measured in the same batch, which survives the load.",
        },
    }

    with lease("verify_causal_conv1d") as lease_id:
        results["lease"] = {"acquired": True, "id": str(lease_id)}
        results["forward_cases"] = [run_case(*case) for case in CASES]
        results["gradient_cases"] = [run_gradient(*case) for case in CASES]
        results["update_cases"] = [run_update(*case) for case in UPDATES]
        results["profile"] = [run_profile(*case) for case in PROFILES]
        results["semantics"] = check_semantics()
        results["step_parity"] = check_step_parity()
        results["finite_difference"] = check_reference_finite_difference()

    passed = True
    for row in results["forward_cases"]:
        limit = VALUE_LIMITS[row["dtype"]]
        passed &= row["out_rel_error"] <= limit
        passed &= row["final_state_rel_error"] <= limit
    for row in results["gradient_cases"]:
        limit = GRAD_LIMITS[row["dtype"]]
        passed &= all(error <= limit for error in row["grad_rel_error"].values())
    for row in results["update_cases"]:
        limit = VALUE_LIMITS[row["dtype"]]
        passed &= row["out_rel_error"] <= limit and row["state_rel_error"] <= limit
    for check in results["semantics"] + results["step_parity"]:
        passed &= check["passed"]
    for row in results["finite_difference"]:
        passed &= row["passed"]
    results["verdict"] = "pass" if passed else "fail"
    results["limits"] = [
        "float32 and float16 use the Metal kernels; float64, bfloat16 and every non-MPS "
        "device run the conv1d composition, which is exact but not the fast path.",
        "causal_conv1d_update is the decode path and carries no autograd; it raises when "
        "x, weight or state requires grad.",
        "the state of causal_conv1d_update must have stride 1 in its last dimension: a slice "
        "of a larger cache is fine because the kernel reads the strides, a transpose is not, "
        "because the buffer is written in place.",
        "seq_idx (sequence-boundary masks for packed batches) is not implemented.",
        "the tap count is a compile-time constant in the Metal sources, so a new K compiles "
        "one shader the first time it is used.",
        "a row length that is not a multiple of four (eight for the fp16 forward) drops to one "
        "element per thread: correct, and slower by the loads it then issues one at a time.",
        "fp16 accumulates the taps and the bias in float32 and rounds once; values agree with "
        "the float64 reference to about 2e-4 relative, not bit-exactly.",
    ]

    out = ROOT / "measured" / "causal_conv1d.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1) + "\n")

    header = f"{'case':26s} {'dtype':8s} {'value':>9s} {'state':>9s} {'kernel':>8s} {'aten':>8s} {'x':>6s}"
    print(header)
    for row in results["forward_cases"]:
        print(f"{row['name']:26s} {row['dtype']:8s} {row['out_rel_error']:9.2e} "
              f"{row['final_state_rel_error']:9.2e} {row['kernel_ms']:8.3f} {row['aten_ms']:8.3f} "
              f"{row['speedup']:6.2f}")
    print()
    print(f"{'gradient':26s} {'dtype':8s} {'worst':>9s} {'kernel':>8s} {'aten':>8s} {'x':>6s}")
    for row in results["gradient_cases"]:
        print(f"{row['name']:26s} {row['dtype']:8s} "
              f"{max(row['grad_rel_error'].values()):9.2e} {row['kernel_ms']:8.3f} "
              f"{row['aten_ms']:8.3f} {row['speedup']:6.2f}")
    print()
    print(f"{'update':26s} {'dtype':8s} {'out':>9s} {'state':>9s} {'kernel':>8s} {'aten':>8s} {'x':>6s}")
    for row in results["update_cases"]:
        print(f"{row['name']:26s} {row['dtype']:8s} {row['out_rel_error']:9.2e} "
              f"{row['state_rel_error']:9.2e} {row['kernel_ms']:8.3f} {row['aten_ms']:8.3f} "
              f"{row['speedup']:6.2f}")
    print()
    print(f"{'stage':26s} {'dtype':8s} {'ms':>8s} {'GB/s':>8s} {'of floor':>9s}")
    for row in results["profile"]:
        for stage in row["stages"]:
            print(f"{row['name'] + ' ' + stage['stage']:26s} {row['dtype']:8s} "
                  f"{stage['ms']:8.4f} {stage['gbs']:8.1f} {stage['of_floor']:9.2f}")
    print()
    for check in results["semantics"] + results["step_parity"]:
        print(f"{'PASS' if check['passed'] else 'FAIL'}  {check['check']}  {check['detail']}")
    for row in results["finite_difference"]:
        print(f"{'PASS' if row['passed'] else 'FAIL'}  finite_difference[{row['input']}]  "
              f"{row['rel_error']:.2e}")
    print(f"verdict: {results['verdict']}  ->  {out.relative_to(ROOT)}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
