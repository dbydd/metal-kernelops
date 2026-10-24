"""Verify ``mamba2_chunk_scan`` against the step recurrence, then measure it.

Run with the torch 2.14.1 environment:

    PYTHONPATH=src uv run --project <venv> python benchmarks/verify_mamba2_chunk.py

Checks, all against ``mamba2_chunk_scan_reference`` (the per-token recurrence,
evaluated in float64 on the CPU):

* forward and final state over the full option matrix: grouped-query heads
  (``ngroups`` 1, ``nheads/2`` and ``nheads``), ``D`` as ``(nheads,)`` and
  ``(nheads, d_head)``, ``z``, ``dt_bias``, ``dt_softplus``, ``initial_states``,
  lengths that do not divide the chunk size and a chunk size larger than the
  sequence;
* the same for float16 and bfloat16 on MPS, plus output dtype/shape, padded
  output length, the ``initial_states`` spellings and the shape errors;
* agreement with a literal transcription of ``ssd_minimal_discrete`` on a
  permutation-free case: the layout and the discretization are what upstream
  computes, not just what our own reference computes;
* input gradients for every differentiable argument, MPS autograd against
  float64 autograd of the recurrence, plus central finite differences of
  ``x``/``dt``/``A``/``B``/``C`` against the reference to prove the reference
  really is the function being differentiated;
* timings: chunk-size sweep, model-like shapes, float32/float16/bfloat16,
  forward and forward+backward, a sequential-recurrence baseline, and the peak
  footprint of the buffers that are live together (``torch.mps`` has no
  per-call high-water mark, so that one is modelled from the shape and the live
  allocation of the returned tensors is measured next to it); all inside a
  compute-broker tier-1 lease and written to ``measured/mamba2_chunk.json``.

Tolerance
---------
The scan keeps ``dt``, the cumulative decay and the chunk states in float32
whatever the input dtype, and rounds only the score matrix and the head-state
products to the input dtype. Float32 is therefore compared at 1e-5 relative,
half at 2e-3 and bfloat16 at 2e-2; the recorded numbers are what actually came
out, the thresholds only decide pass or fail.
"""

import json
import os
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CHAMBER = Path(os.environ.get("RESEARCH_CHAMBER", "/Users/dbydd/vibe-agent-working-dir/ResearchChamber"))
sys.path.insert(0, str(CHAMBER))

from metal_kernelops.cv.mamba2_chunk import (  # noqa: E402
    mamba2_chunk_scan,
    mamba2_chunk_scan_reference,
)
from shared_scripts.compute_broker import lease_guard  # noqa: E402

CHECKED_TORCH = "2.14.1"
TOL = 1e-5
HALF_TOL = 2e-3
BFLOAT_TOL = 2e-2


def _max_err(actual, expected):
    """(max abs error, max error relative to the largest expected magnitude)."""
    actual = actual.float()
    expected = expected.float()
    scale = expected.abs().max().item()
    return ((actual - expected).abs().max().item(),
            ((actual - expected).abs().max() / max(scale, 1e-30)).item())


def _record(failures, report, name, passed, detail=""):
    report[name] = passed
    if not passed:
        failures.append(f"{name}{': ' + detail if detail else ''}")


def _compare(failures, report, name, actual, expected, tol, detail=""):
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    same_special = bool(torch.equal(torch.isfinite(actual), torch.isfinite(expected)))
    worst_abs, worst_rel = _max_err(actual, expected)
    ok = same_special and worst_rel <= tol
    _record(failures, report, name, ok,
            f"rel {worst_rel:.3e} abs {worst_abs:.3e}{' ' + detail if detail else ''}")
    return worst_abs, worst_rel


def _problem(batch, length, nheads, headdim, ngroups, dstate, seed=0, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, length, nheads, headdim, generator=generator)
    dt = torch.randn(batch, length, nheads, generator=generator)
    A = -torch.exp(torch.rand(nheads, generator=generator))
    B = torch.randn(batch, length, ngroups, dstate, generator=generator)
    C = torch.randn(batch, length, ngroups, dstate, generator=generator)
    return [t.to(dtype) for t in (x, dt, A, B, C)]


def _options(batch, length, nheads, headdim, dstate, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return dict(
        D=torch.randn(nheads, generator=generator),
        z=torch.randn(batch, length, nheads, headdim, generator=generator),
        dt_bias=torch.randn(nheads, generator=generator) * 0.1,
        initial_states=torch.randn(batch, nheads, headdim, dstate, generator=generator) * 0.1,
    )


def check_forward(failures, report):
    """Forward and final state against the float64 recurrence, on CPU then MPS."""
    shapes = [
        (2, 13, 4, 6, 2, 5),      # length not a multiple of the chunk size
        (1, 16, 4, 4, 1, 4),      # grouped-query with one group
        (2, 9, 4, 3, 4, 3),       # one group per head
        (1, 5, 2, 2, 1, 3),       # chunk larger than the sequence
    ]
    cases = []
    for seed, (batch, length, nheads, headdim, ngroups, dstate) in enumerate(shapes):
        for chunk in (1, 4, 16, 64):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                cases.append((seed, batch, length, nheads, headdim, ngroups, dstate, chunk, dtype))
    for seed, batch, length, nheads, headdim, ngroups, dstate, chunk, dtype in cases:
        args = [t.to("mps", dtype) for t in
                _problem(batch, length, nheads, headdim, ngroups, dstate, seed, dtype)]
        options = {k: v.to("mps", dtype) for k, v in
                   _options(batch, length, nheads, headdim, dstate, seed=seed + 10).items()}
        tolerance = {torch.float32: TOL, torch.float16: HALF_TOL, torch.bfloat16: BFLOAT_TOL}[dtype]
        cpu = [t.cpu().double() for t in args]
        cpu_options = {k: v.cpu().double() for k, v in options.items()}
        y_ref, state_ref = mamba2_chunk_scan_reference(
            *cpu, D=cpu_options["D"], z=cpu_options["z"],
            dt_bias=cpu_options["dt_bias"], dt_softplus=True,
            initial_states=cpu_options["initial_states"],
        )
        y, state = mamba2_chunk_scan(
            *args, chunk, D=options["D"], z=options["z"],
            dt_bias=options["dt_bias"], dt_softplus=True,
            initial_states=options["initial_states"],
        )
        tag = f"forward_b{batch}l{length}h{nheads}p{headdim}g{ngroups}n{dstate}" \
              f"_c{chunk}_{str(dtype).removeprefix('torch.')}"
        _compare(failures, report, tag, y, y_ref, tolerance)
        _compare(failures, report, tag + "_state", state, state_ref, tolerance)


def check_without_options(failures, report):
    """Defaults: no D, z, bias, softplus or initial state; and each one alone."""
    args = [t.to("mps") for t in _problem(2, 9, 4, 6, 2, 5, seed=7)]
    cpu = [t.cpu().double() for t in args]
    options = {k: v.to("mps") for k, v in _options(2, 9, 4, 6, 5, seed=17).items()}
    y_ref, s_ref = mamba2_chunk_scan_reference(*cpu)
    y, s = mamba2_chunk_scan(*args, 4)
    _compare(failures, report, "defaults_y", y, y_ref, TOL)
    _compare(failures, report, "defaults_state", s, s_ref, TOL)

    y_ref, s_ref = mamba2_chunk_scan_reference(*cpu, D=options["D"].cpu().double())
    y, s = mamba2_chunk_scan(*args, 4, D=options["D"])
    _compare(failures, report, "D_1d_y", y, y_ref, TOL)
    _compare(failures, report, "D_1d_state", s, s_ref, TOL)

    dp = torch.randn(4, 6, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    y_ref, _ = mamba2_chunk_scan_reference(*cpu, D=dp)
    y, _ = mamba2_chunk_scan(*args, 4, D=dp.float().to("mps"))
    _compare(failures, report, "D_2d_y", y, y_ref, TOL)

    y_ref, _ = mamba2_chunk_scan_reference(*cpu, dt_bias=options["dt_bias"].cpu().double())
    y, _ = mamba2_chunk_scan(*args, 4, dt_bias=options["dt_bias"])
    _compare(failures, report, "dt_bias_without_softplus_y", y, y_ref, TOL)

    y_ref, s_ref = mamba2_chunk_scan_reference(
        *cpu, initial_states=options["initial_states"].cpu().double()
    )
    y, s = mamba2_chunk_scan(*args, 4, initial_states=options["initial_states"])
    _compare(failures, report, "initial_states_y", y, y_ref, TOL)
    _compare(failures, report, "initial_states_state", s, s_ref, TOL)


def check_ssd_minimal(failures, report):
    """A literal ``ssd_minimal_discrete`` agrees with the reference and the scan."""

    def segsum(x):
        size = x.size(-1)
        ones = torch.ones(size, size, dtype=torch.bool, device=x.device)
        x = x.unsqueeze(-1).expand(*x.shape, size).masked_fill(~ones.tril(-1), 0)
        x = torch.cumsum(x, dim=-2)
        return x.masked_fill(~ones.tril(), float("-inf"))

    def ssd_minimal_discrete(X, A, B, C, block_len, initial_states=None):
        batch, length, nheads, headdim = X.shape
        chunks, ngroups, dstate = length // block_len, B.shape[2], B.shape[3]
        X = X.reshape(batch, chunks, block_len, nheads, headdim)
        A = A.reshape(batch, chunks, block_len, nheads).permute(0, 3, 1, 2)
        B = B.reshape(batch, chunks, block_len, ngroups, dstate)
        C = C.reshape(batch, chunks, block_len, ngroups, dstate)
        A_cumsum = torch.cumsum(A, dim=-1)
        decay = torch.exp(segsum(A))
        Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, decay, X)
        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)
        decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        states, final_state = new_states[:, :-1], new_states[:, -1]
        state_decay_out = torch.exp(A_cumsum)
        Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)
        return (Y_diag + Y_off).reshape(batch, length, nheads, headdim), final_state

    batch, length, nheads, headdim, ngroups, dstate = 1, 16, 2, 3, 1, 4
    x = torch.randn(batch, length, nheads, headdim, generator=torch.Generator().manual_seed(5),
                    dtype=torch.float64)
    dt = F.softplus(torch.randn(batch, length, nheads, generator=torch.Generator().manual_seed(6),
                                dtype=torch.float64) - 2)
    A = -torch.exp(torch.rand(nheads, generator=torch.Generator().manual_seed(7), dtype=torch.float64))
    B = torch.randn(batch, length, ngroups, dstate, generator=torch.Generator().manual_seed(8),
                    dtype=torch.float64)
    C = torch.randn(batch, length, ngroups, dstate, generator=torch.Generator().manual_seed(9),
                    dtype=torch.float64)
    initial = torch.randn(batch, nheads, headdim, dstate,
                          generator=torch.Generator().manual_seed(10), dtype=torch.float64) * 0.1
    y_min, s_min = ssd_minimal_discrete(
        x * dt.unsqueeze(-1), A * dt, B, C, 4, initial[:, None]
    )
    y_ref, s_ref = mamba2_chunk_scan_reference(x, dt, A, B, C, 4, initial_states=initial)
    y_scan, s_scan = mamba2_chunk_scan(x, dt, A, B, C, 4, initial_states=initial)
    _compare(failures, report, "ssd_minimal_y", y_min, y_ref, TOL)
    _compare(failures, report, "ssd_minimal_state", s_min, s_ref, TOL)
    _compare(failures, report, "ssd_minimal_scan_y", y_scan, y_min, TOL)
    _compare(failures, report, "ssd_minimal_scan_state", s_scan, s_min, TOL)


def check_layout(failures, report):
    args = _problem(2, 13, 4, 6, 2, 5, seed=11)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        mps = [t.to("mps", dtype) for t in args]
        y, state = mamba2_chunk_scan(*mps, 5)
        _record(failures, report, f"layout_dtype_{str(dtype).removeprefix('torch.')}",
                y.dtype == dtype and state.dtype == dtype and y.shape == (2, 13, 4, 6)
                and state.shape == (2, 4, 6, 5),
                f"{y.dtype} {tuple(y.shape)} {tuple(state.shape)}")
    # non-contiguous inputs reach the same answer
    mps = [t.to("mps") for t in args]
    wide = torch.zeros(2, 13, 4, 12, device="mps")
    wide[:, :, :, ::2] = mps[0]
    y_strided, state_strided = mamba2_chunk_scan(wide[:, :, :, ::2], *mps[1:], 5)
    y_plain, state_plain = mamba2_chunk_scan(*mps, 5)
    _compare(failures, report, "layout_strided_x", y_strided, y_plain, 0.0)
    # both initial_states spellings and the ssd_minimal 5-D form
    state0 = torch.randn(2, 4, 6, 5, generator=torch.Generator().manual_seed(12)).to("mps")
    y_a, s_a = mamba2_chunk_scan(*mps, 5, initial_states=state0)
    y_b, s_b = mamba2_chunk_scan(*mps, 5, initial_states=state0.transpose(-1, -2))
    y_c, s_c = mamba2_chunk_scan(*mps, 5, initial_states=state0[:, None])
    _compare(failures, report, "layout_init_transposed", y_b, y_a, 0.0)
    _compare(failures, report, "layout_init_5d", y_c, y_a, 0.0)
    # a wrong shape must raise, not silently broadcast
    try:
        mamba2_chunk_scan(*mps, 5, initial_states=torch.zeros(2, 4, 5, 5, device="mps"))
        raised = False
    except ValueError:
        raised = True
    _record(failures, report, "layout_init_shape_checked", raised)


def check_gradients(failures, report):
    """MPS autograd against float64 autograd of the recurrence, every input."""
    batch, length, nheads, headdim, ngroups, dstate = 2, 21, 4, 4, 2, 5
    args = _problem(batch, length, nheads, headdim, ngroups, dstate, seed=13)
    options = _options(batch, length, nheads, headdim, dstate, seed=14)

    def run(device, dtype):
        x, dt, A, B, C = (t.to(device=device, dtype=dtype).requires_grad_(True) for t in args)
        D = options["D"].to(device, dtype).requires_grad_(True)
        z = options["z"].to(device, dtype).requires_grad_(True)
        bias = options["dt_bias"].to(device, dtype).requires_grad_(True)
        init = options["initial_states"].to(device, dtype).requires_grad_(True)
        y, state = mamba2_chunk_scan(x, dt, A, B, C, 8, D=D, z=z, dt_bias=bias,
                                     dt_softplus=True, initial_states=init)
        loss = (y * y).sum() + (state * state).sum()
        return torch.autograd.grad(loss, (x, dt, A, B, C, D, z, bias, init))

    fast = run("mps", torch.float32)
    slow = run("cpu", torch.float64)
    names = ("x", "dt", "A", "B", "C", "D", "z", "dt_bias", "initial_states")
    for name, grad, expected in zip(names, fast, slow):
        _compare(failures, report, f"grad_{name}", grad, expected, TOL)


def check_finite_difference(failures, report):
    """Central differences of the float64 recurrence, to confirm it is the target."""
    batch, length, nheads, headdim, ngroups, dstate = 1, 8, 2, 3, 1, 4
    args = _problem(batch, length, nheads, headdim, ngroups, dstate, seed=16, dtype=torch.float64)
    names = ("x", "dt", "A", "B", "C")
    leaves = [t.clone().requires_grad_(True) for t in args]
    mamba2_chunk_scan_reference(*leaves)[0].sum().backward()
    analytic = [t.grad.clone() for t in leaves]
    for position, name in enumerate(names):
        generator = torch.Generator().manual_seed(20 + position)
        for _ in range(3):
            index = tuple(torch.randint(0, size, (1,), generator=generator).item()
                          for size in args[position].shape)
            step = 1e-6
            plus = [t.clone() for t in args]
            minus = [t.clone() for t in args]
            plus[position][index] += step
            minus[position][index] -= step
            numeric = (
                mamba2_chunk_scan_reference(*plus)[0].sum()
                - mamba2_chunk_scan_reference(*minus)[0].sum()
            ) / (2 * step)
            exact = analytic[position][index].item()
            _record(failures, report, f"fd_{name}_{index}",
                    abs(numeric.item() - exact) < 1e-5 * max(1.0, abs(exact)),
                    f"numeric {numeric.item():.6f} analytic {exact:.6f}")


def _time(fn, device="mps", warmup=5, reps=15, batches=3):
    """Best of ``reps`` runs in each of ``batches`` passes, synchronized per run."""
    best = None
    for _ in range(batches):
        for _ in range(warmup):
            fn()
        if device == "mps":
            torch.mps.synchronize()
        for _ in range(reps):
            start = time.perf_counter()
            fn()
            if device == "mps":
                torch.mps.synchronize()
            sample = time.perf_counter() - start
            best = sample if best is None else min(best, sample)
    return best * 1e3


def _problem_mps(batch, length, nheads, headdim, ngroups, dstate, dtype):
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(batch, length, nheads, headdim, generator=generator, dtype=torch.float32)
    dt = torch.randn(batch, length, nheads, generator=generator, dtype=torch.float32)
    A = -torch.exp(torch.rand(nheads, generator=generator, dtype=torch.float32))
    B = torch.randn(batch, length, ngroups, dstate, generator=generator, dtype=torch.float32)
    C = torch.randn(batch, length, ngroups, dstate, generator=generator, dtype=torch.float32)
    z = torch.randn(batch, length, nheads, headdim, generator=generator, dtype=torch.float32)
    D = torch.randn(nheads, generator=generator, dtype=torch.float32)
    return [t.to("mps", dtype) for t in (x, dt, A, B, C, z, D)]


def _memory(batch, length, nheads, headdim, ngroups, dstate, chunk, itemsize):
    """Peak footprint of one forward, from the buffers that are live together.

    ``torch.mps`` reports live and driver allocations but no high-water mark for
    a call, and in this build the driver count jumps to a fixed reservation on
    the first allocation, so neither counter can be sampled across a call. The
    peak is therefore the sum of the sizes below, each fixed by the shape:

    * ``xw`` (input dtype) and the float32 decay ``weight``, both
      ``batch * length * nheads * d_head`` elements;
    * the chunk state and the entering state, both
      ``batch * nheads * nchunks * d_state * d_head`` float32;
    * the causal decay and the score matrix, both
      ``batch * nheads * length * chunk`` float32/input dtype;
    * the per-group ``C @ B`` block, ``batch * ngroups * length * chunk``.

    ``torch.mps.current_allocated_memory()`` after the call is reported
    alongside it as the measured live part (the returned tensors).
    """
    nchunks = -(-length // chunk)
    state = batch * nheads * nchunks * dstate * headdim * 4
    xw = batch * length * nheads * headdim
    scores = batch * nheads * length * chunk
    return dict(
        chunk_state_mib=round(state / 2**20, 1),
        x_weight_mib=round((xw * (itemsize + 4)) / 2**20, 1),
        scores_mib=round((scores * (4 + itemsize)) / 2**20, 1),
        cb_mib=round(batch * ngroups * length * chunk * 4 / 2**20, 1),
        peak_model_mib=round((2 * state + xw * (itemsize + 4)) / 2**20
                             + (scores * (4 + itemsize)) / 2**20
                             + batch * ngroups * length * chunk * 4 / 2**20, 1),
    )


def measure():
    shapes = [
        ("mamba2-130m", 1, 2048, 24, 64, 1, 128),
        ("gqa-h32-g8", 4, 1024, 32, 64, 8, 128),
        ("wide-heads", 1, 4096, 64, 64, 1, 64),
    ]
    sweep, models, dtypes, baseline, memory = [], [], [], [], []
    for name, batch, length, nheads, headdim, ngroups, dstate in shapes:
        x, dt, A, B, C, z, D = _problem_mps(batch, length, nheads, headdim, ngroups, dstate,
                                            torch.float32)
        for chunk in (16, 32, 64, 128, 256):
            with torch.no_grad():
                ms = _time(lambda chunk=chunk: mamba2_chunk_scan(
                    x, dt, A, B, C, chunk, D=D, z=z, dt_softplus=True))
            sweep.append(dict(shape=name, chunk=chunk, dtype="float32", ms=round(ms, 3),
                              tokens_per_s=round(batch * length / ms * 1e3)))
        # ties inside 5% go to the smaller chunk: less score-matrix memory, and
        # the choice then does not flip on a noisy sample
        rows = sweep[-5:]
        fastest = min(row["ms"] for row in rows)
        best_chunk = min(row["chunk"] for row in rows if row["ms"] <= fastest * 1.05)
        with torch.no_grad():
            ms = _time(lambda: mamba2_chunk_scan(x, dt, A, B, C, best_chunk, D=D, z=z,
                                                 dt_softplus=True))
        torch.mps.empty_cache()
        live = torch.mps.current_allocated_memory()
        torch.mps.synchronize()
        with torch.no_grad():
            out = mamba2_chunk_scan(x, dt, A, B, C, best_chunk, D=D, z=z, dt_softplus=True)
        torch.mps.synchronize()
        memory.append(dict(
            shape=name, chunk=best_chunk, dtype="float32",
            returned_mib=round((torch.mps.current_allocated_memory() - live) / 2**20, 1),
            **_memory(batch, length, nheads, headdim, ngroups, dstate, best_chunk, 4),
        ))
        del out
        models.append(dict(shape=name, chunk=best_chunk, dtype="float32", ms=round(ms, 3),
                           tokens_per_s=round(batch * length / ms * 1e3)))
        # forward + backward
        for tensor in (x, dt, A, B, C, z, D):
            tensor.requires_grad_(True)

        def step():
            y, state = mamba2_chunk_scan(x, dt, A, B, C, best_chunk, D=D, z=z,
                                         dt_softplus=True)
            ((y * y).sum() + (state * state).sum()).backward()
            for tensor in (x, dt, A, B, C, z, D):
                tensor.grad = None

        models.append(dict(shape=name, chunk=best_chunk, dtype="float32",
                           ms=round(_time(step, warmup=2, reps=5), 3), pass_="forward+backward"))
        for tensor in (x, dt, A, B, C, z, D):
            tensor.requires_grad_(False)

        # sequential per-token recurrence on MPS, the thing chunking replaces
        with torch.no_grad():
            baseline.append(dict(shape=name, chunk=None, dtype="float32",
                                 ms=round(_time(lambda: mamba2_chunk_scan_reference(
                                     x, dt, A, B, C, D=D, z=z, dt_softplus=True,
                                     dtype=torch.float32), warmup=1, reps=3, batches=1), 3)))

    baseline_ms = {row["shape"]: row["ms"] for row in baseline}
    for row in models:
        if "pass_" not in row:
            row["sequential_speedup"] = round(baseline_ms[row["shape"]] / row["ms"], 1)

    # dtype comparison on the largest shape
    name, batch, length, nheads, headdim, ngroups, dstate = shapes[2]
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        x, dt, A, B, C, z, D = _problem_mps(batch, length, nheads, headdim, ngroups, dstate, dtype)
        with torch.no_grad():
            ms = _time(lambda: mamba2_chunk_scan(x, dt, A, B, C, 32, D=D, z=z, dt_softplus=True))
        dtypes.append(dict(shape=name, chunk=32, dtype=str(dtype).removeprefix("torch."),
                           ms=round(ms, 3), tokens_per_s=round(batch * length / ms * 1e3)))
    return sweep, models, dtypes, baseline, memory


def main():
    version = torch.__version__.split("+")[0]
    if version != CHECKED_TORCH:
        raise SystemExit(f"scan was checked against torch {CHECKED_TORCH}, running {version}")
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available on this machine")
    failures, report = [], {}
    check_forward(failures, report)
    check_without_options(failures, report)
    check_ssd_minimal(failures, report)
    check_layout(failures, report)
    check_gradients(failures, report)
    check_finite_difference(failures, report)

    # Two rounds under separate leases; the sequential baseline is the most
    # contention-sensitive number in the set, so the faster of the two rounds is
    # the one that saw the quieter device.
    rounds, lease = [], None
    for _ in range(2):
        with lease_guard("metal-kernelops", tier=1, task_name="mamba2_chunk_scan") as lease:
            rounds.append(measure())
    sweep, models, dtypes, baseline, memory = min(
        rounds, key=lambda result: sum(row["ms"] for row in result[3])
    )
    measured = {
        "torch": version,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "date": time.strftime("%Y-%m-%d"),
        "lease": lease,
        "api": ("mamba2_chunk_scan(x, dt, A, B, C, chunk_size=64, D=None, z=None, "
                "dt_bias=None, dt_softplus=False, initial_states=None) -> (y, final_state)"),
        "layout": {
            "x": "(batch, length, nheads, d_head)",
            "dt": "(batch, length, nheads)",
            "A": "(nheads,)",
            "B, C": "(batch, length, ngroups, d_state)",
            "D": "(nheads,) or (nheads, d_head)",
            "z": "(batch, length, nheads, d_head)",
            "initial_states": "(batch, nheads, d_head, d_state)",
            "y": "(batch, length, nheads, d_head)",
            "final_state": "(batch, nheads, d_head, d_state)",
        },
        "tolerance": {
            "float32": TOL,
            "float16": HALF_TOL,
            "bfloat16": BFLOAT_TOL,
            "note": ("every dtype is compared with the float64 per-token recurrence; dt, the "
                     "cumulative decay and the chunk states stay in float32 and only the "
                     "score matrix and the head-state products follow the input dtype"),
        },
        "timing": (
            "two rounds, each best of 15 runs after 5 warm-up runs with torch.mps.synchronize() "
            "per run in three separate passes, whole call including autograd bookkeeping; the "
            "round with the faster sequential baseline is kept because the device is shared "
            "with other MPS work, so the milliseconds are still upper bounds on an idle device; "
            "the sequential baseline is one launch per token, so contention inflates it more "
            "than the scan and the ratios are upper bounds too"
        ),
        "checks": report,
        "chunk_sweep": sweep,
        "models": models,
        "dtypes": dtypes,
        "sequential_baseline": baseline,
        "memory": memory,
    }
    target = ROOT / "measured" / "mamba2_chunk.json"
    target.write_text(json.dumps(measured, indent=2) + "\n")

    print(f"{sum(report.values())}/{len(report)} checks passed")
    for failure in failures:
        print(f"  FAIL {failure}")
    for row in sweep:
        print(f"  {row['shape']:<14} chunk {row['chunk']:>4}  {row['ms']:7.3f} ms"
              f"  {row['tokens_per_s']:>9} tok/s")
    for row in models:
        print(f"  {row['shape']:<14} chunk {row['chunk']:>4}  {row['ms']:7.3f} ms"
              f"  {row.get('pass_', 'forward'):<16}"
              f"  {row['tokens_per_s'] if 'tokens_per_s' in row else ''}"
              f"  {row.get('sequential_speedup', '')}")
    for row in baseline:
        print(f"  {row['shape']:<14} sequential   {row['ms']:11.3f} ms")
    for row in dtypes:
        print(f"  {row['shape']:<14} {row['dtype']:<9} {row['ms']:7.3f} ms")
    for row in memory:
        print(f"  {row['shape']:<14} peak {row['peak_model_mib']:7.1f} MiB"
              f"  (state {row['chunk_state_mib']:.1f} + scores {row['scores_mib']:.1f}"
              f" + x {row['x_weight_mib']:.1f} + cb {row['cb_mib']:.1f})")
    print(f"wrote {target}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
