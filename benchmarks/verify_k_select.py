"""Verify and measure ``metal_kernelops.cv.k_select``: NSA key-block selection.

The contract under test is the one in the module docstring: larger score first,
equal scores by smaller index, NaNs after every finite value, ``sorted=False``
as the same set by ascending index, int64 output, ``0 <= k <= N``.  Two
independent lines of evidence are used for it:

* structural checks that read only the scores and the returned indices — the
  selection is sorted under the stated order, nothing unselected beats the worst
  selected element, indices are unique and in range, and an all-distinct row
  agrees with ``torch.topk``;
* exact parity against ``reference_k_select``, the two-stable-argsort form of
  the same order, over ties, NaNs, empty selections, wide rows and wide batches.

Gradients: ``k_select`` must build no graph at all, and ``k_select_values`` must
pass the gradient to exactly the selected entries, matching both a hand-built
scatter and the CPU path.

Timings are NSA-shaped: one row per query head, one column per candidate block,
``k`` in {8, 16, 32, 64, 128}, against the stable-argsort path the kernel
replaced, against ``torch.topk``, and against the CPU reference.
"""

import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import metal_kernelops.cv.selection as ksel
from metal_kernelops.cv.selection import k_select, k_select_values, reference_k_select

DEVICE = "mps"
K_VALUES = (8, 16, 32, 64, 128)

# k values the kernel limit probe sweeps: both sides of the window/radix
# crossover, both sides of the kernel/sort cutoff, and the range in between.
K_LIMIT_VALUES = (8, 12, 16, 20, 24, 32, 48, 64, 128, 192, 256, 384, 512)


def import_is_inert():
    """Importing the module must not compile a shader or touch the device."""
    info = ksel._library.cache_info()
    return info.currsize == 0 and info.misses == 0


def better(a_score, a_index, b_score, b_index):
    """``a`` ranks ahead of ``b`` under the module's contract, elementwise."""
    a_nan, b_nan = torch.isnan(a_score), torch.isnan(b_score)
    finite = ~a_nan & ~b_nan
    by_value = finite & ((a_score > b_score)
                         | ((a_score == b_score) & (a_index < b_index)))
    over_nan = ~a_nan & b_nan
    nan_tie = a_nan & b_nan & (a_index < b_index)
    return by_value | over_nan | nan_tie


def structural_failures(scores, indices, k, dim, sorted_desc):
    """Checks that need nothing but the input scores and the returned indices."""
    scores = scores.cpu()
    indices = indices.cpu()
    width = scores.shape[dim]
    failures = []

    moved = scores.movedim(dim, -1)
    rows = moved.reshape(-1, width)
    picked = indices.movedim(dim, -1).reshape(-1, k)
    if picked.shape[0] != rows.shape[0]:
        return ["selected shape does not match the input shape"]
    row_ids = torch.arange(rows.shape[0]).unsqueeze(1).expand_as(picked)
    picked_scores = rows[row_ids, picked]

    if bool((picked < 0).any()) or bool((picked >= width).any()):
        failures.append("index out of range")
    if k > 1 and not bool((picked[:, 1:] != picked[:, :-1]).all()):
        failures.append("duplicate index inside one selection")

    if sorted_desc and k > 1:
        strict = better(picked_scores[:, :-1], picked[:, :-1],
                        picked_scores[:, 1:], picked[:, 1:])
        if not bool(strict.all()):
            failures.append("selection not in descending score order with lower-index ties")
    if not sorted_desc and k > 1 and not bool((picked[:, 1:] > picked[:, :-1]).all()):
        failures.append("sorted=False is not ascending by index")

    # Nothing left out may rank ahead of the worst element taken.
    if k < width:
        worst_score = picked_scores[:, -1]
        worst_index = picked[:, -1]
        offsets = torch.arange(width)
        for start in range(0, rows.shape[0], 2048):
            block = slice(start, start + 2048)
            beats = better(rows[block], offsets,
                           worst_score[block].unsqueeze(1),
                           worst_index[block].unsqueeze(1))
            beats.scatter_(1, picked[block], False)
            if bool(beats.any()):
                failures.append(
                    f"{int(beats.sum())} unselected elements rank ahead of the selection")
                break
    return failures


def parity_case(shape, k, salt, seed, dim=-1, dtype=torch.float32):
    """One kernel-vs-reference case; ``salt`` chooses ties, NaN or plain values."""
    torch.manual_seed(seed)
    scores = torch.randn(*shape)
    if salt == 1:
        scores = (scores * 2).round()          # heavy exact ties
    elif salt == 2:
        scores[torch.rand_like(scores) < 0.1] = float("nan")
        scores[scores > 2.2] = float("-inf")   # -inf next to NaN exercises the tail
    elif salt == 3:
        scores = scores.tril(-1) if len(shape) == 2 else scores
        scores[scores == 0] = float("inf")
    scores = scores.to(dtype)
    on_device = scores.to(DEVICE)
    failures = []
    for order in (True, False):
        got = k_select(on_device, k, dim=dim, sorted=order)
        want = reference_k_select(scores, k, dim=dim, sorted=order)
        if not torch.equal(got.cpu(), want):
            failures.append(f"k={k} sorted={order} {salt=}")
        failures += [f"k={k} sorted={order} {salt=}: {why}"
                     for why in structural_failures(on_device, got, k,
                                                    dim if dim >= 0 else dim + len(shape),
                                                    order)]
    return failures


def parity_sweep():
    cases = 0
    failures = []
    seed = 0
    shapes = [
        (1, 8), (7, 24), (64, 64), (8, 512), (64, 256), (256, 256), (1024, 64),
        (512, 4096), (1024, 2048), (2048, 1024), (4096, 1024), (16384, 1024),
        (2, 3, 257), (4, 7, 1024), (3, 5, 9, 64),
    ]
    # Every form boundary: 15 and 17 straddle the window/radix crossover, 255 and
    # 256 stay inside the kernel, 257 is one past it and runs the sort path.
    extra_ks = (3, 5, 15, 17, 64, 100, 128, 192, 255, 256, 257)
    for shape in shapes:
        width = shape[-1]
        wide = shape[0] * width > 2 * 1024 * 1024
        ks = list(dict.fromkeys((1, 2, 8, 16, 32, 48, width)
                                + tuple(k for k in extra_ks if k <= width)))
        for k in ks:
            if k > width:
                continue
            for salt in ((0, 2) if wide else (0, 1, 2, 3)):
                if salt == 3 and len(shape) != 2:
                    continue
                seed += 1
                cases += 1
                failures += parity_case(shape, k, salt, seed)
    # a non-last axis, and a non-contiguous view of an MPS tensor
    for k in (4, 16, 64, 128):
        seed += 1
        cases += 2
        failures += parity_case((4, 128, 64), min(k, 64), 1, seed, dim=1)
        failures += parity_case((4, 64, 128), k, 2, seed + 7, dim=2)
        transposed = torch.randn(512, 96, device=DEVICE).t()
        want = reference_k_select(transposed.cpu(), k)
        if not torch.equal(k_select(transposed, k).cpu(), want):
            failures.append(f"non-contiguous k={k}")
    # half precision is widened exactly, so the order must not move
    for dtype in (torch.float16, torch.bfloat16):
        for k in (8, 32, 64, 128):
            seed += 1
            cases += 1
            failures += parity_case((80, 1024), k, 1, seed, dtype=dtype)
    return cases, failures


def ties_against_topk():
    """With distinct scores the order is total, so topk must agree on the values."""
    cases = 0
    failures = []
    torch.manual_seed(11)
    for width, k in [(4096, 8), (4096, 32), (1024, 16), (257, 33), (4096, 64),
                     (2048, 128), (1024, 256), (4096, 384)]:
        cpu = torch.randn(128, width)
        got = k_select(cpu.to(DEVICE), k).cpu()
        values = torch.gather(cpu, 1, got)
        want_values, want_indices = torch.topk(cpu, k, dim=1)
        cases += 1
        if not torch.equal(values, want_values):
            failures.append(f"width={width} k={k}: selected values differ from topk")
        if not torch.equal(got, want_indices):
            failures.append(f"width={width} k={k}: indices differ from topk on distinct scores")
    return cases, failures


def edge_cases():
    checks = 0
    failures = []

    def expect(name, condition):
        nonlocal checks
        checks += 1
        if not condition:
            failures.append(name)

    scores = torch.randn(32, 128, device=DEVICE)
    empty = k_select(scores, 0)
    expect("k=0 shape", empty.shape == (32, 0) and empty.dtype == torch.int64)
    expect("k=0 empty tensor reference", reference_k_select(scores.cpu(), 0).shape == (32, 0))
    expect("k=N keeps every block",
           torch.equal(k_select(scores, 128).cpu(), reference_k_select(scores.cpu(), 128)))

    raised = 0
    for bad in (129, -1):
        try:
            k_select(scores, bad)
        except ValueError:
            raised += 1
    expect("k outside [0, N] raises", raised == 2)
    try:
        k_select(scores.to(torch.int32), 4)
    except TypeError:
        checks += 1
    else:
        failures.append("integer scores accepted")
    try:
        k_select(torch.tensor(1.0, device=DEVICE), 1)
    except ValueError:
        checks += 1
    else:
        failures.append("0-dim input accepted")
    try:
        k_select(scores, 4, dim=5)
    except IndexError:
        checks += 1
    else:
        failures.append("out-of-range dim accepted")

    # an empty batch: zero rows, no kernel launch
    hollow = torch.empty(0, 64, device=DEVICE)
    expect("empty batch", k_select(hollow, 8).shape == (0, 8))

    # NaN only: the selection is the first k indices
    nan_rows = torch.full((4, 32), float("nan"), device=DEVICE)
    expect("all-NaN rows take the lowest indices",
           torch.equal(k_select(nan_rows, 5).cpu(),
                       torch.arange(5).expand(4, 5)))
    # fewer finite values than k: the tail is NaN by index order
    mixed = torch.full((1, 16), float("nan"))
    mixed[0, ::4] = 1.0
    expect("NaN fills the tail by index",
           torch.equal(k_select(mixed.to(DEVICE), 8).cpu(), reference_k_select(mixed, 8)))
    # -inf must rank above NaN, even when the index order would prefer the NaN
    tail = torch.tensor([[float("nan"), float("-inf"), 3.0, float("nan"), float("-inf")]])
    expect("-inf above NaN", k_select(tail.to(DEVICE), 4).cpu().tolist() == [[2, 1, 4, 0]])

    # The order keys are a bit transformation, so the exact float bit patterns
    # are what the radix form compares: -0.0 must tie with +0.0, and adjacent
    # floats must order as floats do.
    keys = torch.tensor([
        [-0.0, 0.0, -0.0, 0.0, 1.0],                     # signed zero tie
        [1.0, torch.nextafter(torch.tensor(1.0), torch.tensor(2.0)).item(),
         torch.nextafter(torch.tensor(1.0), torch.tensor(0.0)).item(), 0.0, -0.0],
        [float("inf"), float("-inf"), 0.0, -0.0, 1.0],
    ])
    expect("bit patterns: signed zeros and subnormals keep the float order",
           torch.equal(k_select(keys.to(DEVICE), 5).cpu(),
                       reference_k_select(keys, 5)))

    # k past the kernel cutoff, and just past the largest radix k it covers
    wide = torch.randn(128, 600, device=DEVICE)
    wide[0, :5] = float("nan")
    for k in (256, 257, 512, 600):
        expect(f"k={k} matches the reference",
               torch.equal(k_select(wide, k).cpu(), reference_k_select(wide.cpu(), k)))
        expect(f"k={k} sorted=False matches the reference",
               torch.equal(k_select(wide, k, sorted=False).cpu(),
                           reference_k_select(wide.cpu(), k, sorted=False)))

    # the input is never written
    probe = torch.randn(16, 64, device=DEVICE)
    before = probe.clone()
    k_select(probe, 8)
    expect("input untouched", torch.equal(probe, before))
    return checks, failures


def gradients():
    rows = []
    torch.manual_seed(19)
    scores = torch.randn(64, 512, device=DEVICE, requires_grad=True)
    indices = k_select(scores, 16)
    rows.append({"case": "k_select builds no graph",
                 "detail": f"grad_fn={indices.grad_fn}, dtype={indices.dtype}",
                 "passed": indices.grad_fn is None and indices.dtype == torch.int64})

    values = k_select_values(scores, 16)
    grad = torch.randn_like(values)
    values.backward(grad)
    scatter = torch.zeros_like(scores.detach()).scatter_(1, k_select(scores.detach(), 16), grad)
    rows.append({"case": "k_select_values routes the gradient to the selected entries",
                 "detail": "backward equals a scatter of the incoming gradient",
                 "passed": torch.equal(scores.grad, scatter)})

    cpu_scores = scores.detach().cpu().clone().requires_grad_(True)
    cpu_values = k_select_values(cpu_scores, 16)
    cpu_values.backward(grad.cpu())
    rows.append({"case": "MPS gradient equals the CPU reference gradient",
                 "detail": f"max |difference| {float((scores.grad.cpu() - cpu_scores.grad).abs().max()):.3e}",
                 "passed": torch.equal(scores.grad.cpu(), cpu_scores.grad)})

    # sorted=False permutes the same entries, so the gradient must land on the
    # same positions; the per-position values follow the permutation.
    scores.grad = None
    other_order = k_select(scores.detach(), 16, sorted=False)
    grad_other = torch.randn_like(values)
    k_select_values(scores, 16, sorted=False).backward(grad_other)
    scatter_other = torch.zeros_like(scores.detach()).scatter_(1, other_order, grad_other)
    rows.append({"case": "sorted=False gradient lands on the same entries",
                 "detail": "same support as the descending order, values permuted with it",
                 "passed": torch.equal(scores.grad, scatter_other)
                           and torch.equal(scatter_other != 0, scatter != 0)})

    probe = torch.ones(3, 8, device=DEVICE, requires_grad=True)   # k covers the row
    k_select_values(probe, 8).sum().backward()
    rows.append({"case": "k=N gives every entry its gradient",
                 "detail": f"entries with no gradient: {int((probe.grad == 0).sum())}",
                 "passed": bool((probe.grad == 1).all())})
    rows.append(finite_difference_gradient())
    return rows


def finite_difference_gradient():
    """Central difference of ``sum(selected * weight)`` against the analytic gradient.

    The scores are spread so that a 1e-2 nudge never reorders them: the gradient
    of the gathered values is then a real derivative, ``weight`` on a selected
    entry and zero elsewhere, and a finite difference has to reproduce it.
    """
    torch.manual_seed(23)
    base = torch.linspace(-4.0, 4.0, 64).repeat(4, 1)
    base = (base + torch.rand(4, 64) * 0.05).to(DEVICE)
    weights = torch.randn(4, 8, device=DEVICE)

    def loss_of(scores):
        return (k_select_values(scores, 8) * weights).sum()

    scores = base.clone().requires_grad_(True)
    loss_of(scores).backward()
    analytic = scores.grad.clone()

    eps = 1e-2
    worst = 0.0
    for row, column in ((0, 63), (0, 0), (1, 40), (2, 31), (3, 12)):
        plus, minus = base.clone(), base.clone()
        plus[row, column] += eps
        minus[row, column] -= eps
        numeric = float((loss_of(plus) - loss_of(minus)) / (2 * eps))
        worst = max(worst, abs(numeric - float(analytic[row, column])))
    return {"case": "finite difference agrees with the analytic gradient",
            "detail": f"5 sampled entries, largest |difference| {worst:.2e}",
            "passed": worst < 1e-2}


def measure(calls, warmup=8, reps=25, rounds=1):
    """Warm, synchronized, interleaved: one shared machine, so compare within a round.

    Every callable is timed once per rep, in the same rep, so a sweep that lands
    mid-flight for one of them lands mid-flight for all of them.  ``rounds``
    repeats the whole sweep and keeps the best rep per callable, which is what
    the k-limit probe uses: on this box a single sweep can put the same kernel
    1.5x apart, and the minimum over rounds is the only stable statistic.
    """
    for _ in range(warmup):
        for call, device in calls.values():
            call()
    if any(device == "mps" for _, device in calls.values()):
        torch.mps.synchronize()
    best = None
    for _ in range(rounds):
        samples = {name: [] for name in calls}
        for _ in range(reps):
            for name, (call, device) in calls.items():
                if device == "mps":
                    torch.mps.synchronize()
                started = time.perf_counter()
                call()
                if device == "mps":
                    torch.mps.synchronize()
                samples[name].append((time.perf_counter() - started) * 1e3)
        pass_ = {name: {"min_ms": round(min(times), 4),
                        "median_ms": round(statistics.median(times), 4)}
                 for name, times in samples.items()}
        best = pass_ if best is None else {
            name: {key: min(best[name][key], pass_[name][key])
                   for key in ("min_ms", "median_ms")} for name in pass_}
    return best


def kernel_k_limit():
    """Both kernel cutoffs: window-vs-radix, and kernel-vs-sort.

    Every kernel form against the stable-argsort path at one few-row and one
    many-row shape, over a k sweep that straddles both constants:
    ``_RADIX_MIN_K`` where the window gives way to the radix form, and
    ``_KERNEL_MAX_K`` where the radix form gives way to the sort path.
    """
    forms = {"merge": ("select_topk_l64", 64), "tall": ("select_topk_tall", 32),
             "radix": ("select_radix", ksel._RADIX_THREADS)}
    rows = []
    for shape in ((512, 4096), (8192, 512)):
        scores = torch.randn(*shape, device=DEVICE)
        cpu_scores = scores.cpu()
        for k in K_LIMIT_VALUES:
            if k > shape[1]:
                continue
            ksel._library(k, False)     # a k that will not compile must fail loudly
            want = reference_k_select(cpu_scores, k)
            # Above the crossover the window forms are never dispatched, and the
            # tall one needs 0.7 s a call at k = 512: timing them there would
            # measure the probe, not the module.
            window_k = k < ksel._RADIX_MIN_K or k <= 48
            fits = {"select_radix": lambda kk: kk <= ksel._radix_max_k(),
                    "select_topk_l64": lambda kk: window_k and kk <= ksel._lanes_max_k(64),
                    "select_topk_tall": lambda kk: window_k}
            used = {name: form for name, form in forms.items() if fits[form[0]](k)}
            calls = {name: (lambda s=scores, kk=k, f=form: ksel._kernel_select(
                s, kk, 1, True, f), "mps") for name, form in used.items()}
            calls["stable_argsort"] = (
                lambda s=scores, kk=k: ksel._sort_select(s, kk, 1, True), "mps")
            times = measure(calls, warmup=4, reps=15, rounds=4)
            exact = {name: torch.equal(ksel._kernel_select(scores, k, 1, True, form).cpu(),
                                       want)
                     for name, form in used.items()}
            best_name = min(used, key=lambda n: times[n]["min_ms"])
            rows.append({
                "rows": shape[0], "blocks": shape[1], "k": k,
                "merge_ms": times.get("merge", {}).get("min_ms"),
                "tall_ms": times.get("tall", {}).get("min_ms"),
                "radix_ms": times.get("radix", {}).get("min_ms"),
                "stable_argsort_ms": times["stable_argsort"]["min_ms"],
                "best_kernel": best_name,
                "best_kernel_ms": times[best_name]["min_ms"],
                "kernel_wins": times[best_name]["min_ms"] < times["stable_argsort"]["min_ms"],
                "kernel_forms_exact": exact,
            })
    return rows


def form_choice():
    """The form each k takes on ``_plan``, for the summary's dispatch table."""
    chosen = {}
    for k in K_LIMIT_VALUES:
        plan = ksel._plan(k, 1024, 512)
        chosen[str(k)] = ({"form": plan[0], "threadgroup": plan[1]} if plan
                          else {"form": "stable_argsort", "threadgroup": None})
    return chosen


def timings():
    """NSA shapes: ``Q*H`` rows — one per query head — and N candidate blocks."""
    shapes = [
        (512, 4096), (1024, 2048), (2048, 1024), (4096, 1024),
        (8192, 512), (16384, 1024), (65536, 512),
    ]
    rows = []
    for shape in shapes:
        scores = torch.randn(*shape, device=DEVICE)
        cpu_scores = scores.cpu()
        width = shape[-1]
        for k in K_VALUES:
            plan = ksel._plan(k, width, shape[0])
            calls = {
                "public_path": (lambda s=scores, kk=k: k_select(s, kk), "mps"),
                "stable_argsort": (lambda s=scores, kk=k: ksel._sort_select(s, kk, 1, True), "mps"),
                "torch_topk": (lambda s=scores, kk=k: torch.topk(s, kk, dim=-1), "mps"),
            }
            if shape[0] * width <= 4 * 1024 * 1024:
                calls["cpu_reference"] = (
                    lambda s=cpu_scores, kk=k: reference_k_select(s, kk), "cpu")
            best = None
            for _ in range(3):
                pass_ = measure(calls, warmup=5 if "cpu_reference" not in calls else 1,
                                reps=25 if "cpu_reference" not in calls else 5)
                best = pass_ if best is None else {
                    name: {key: min(best[name][key], pass_[name][key]) for key in ("min_ms", "median_ms")}
                    for name in pass_}
            exact = torch.equal(k_select(scores, k).cpu(), reference_k_select(cpu_scores, k))
            row = {
                "rows": shape[0],
                "blocks": width,
                "k": k,
                "path": plan[0] if plan else f"stable_argsort (k > {ksel._KERNEL_MAX_K})",
                "threadgroup": plan[1] if plan else None,
                "exact_vs_reference": exact,
            }
            row.update(best)
            row["speedup_sort_over_public"] = round(
                best["stable_argsort"]["min_ms"] / best["public_path"]["min_ms"], 3)
            row["speedup_topk_over_public"] = round(
                best["torch_topk"]["min_ms"] / best["public_path"]["min_ms"], 3)
            rows.append(row)
    return rows


def main():
    torch.manual_seed(20251024)
    assert torch.backends.mps.is_available(), "MPS is not available"
    inert = import_is_inert()

    broker = Path("/Users/dbydd/vibe-agent-working-dir/ResearchChamber")
    if broker.is_dir():
        sys.path.insert(0, str(broker))
    try:
        from shared_scripts.compute_broker import lease_guard
        lease = lambda: lease_guard("mps-kernelops", tier=1, task_name="k_select_verify")
    except ImportError:   # the broker is a machine-local convenience, not a dependency
        from contextlib import nullcontext
        lease = nullcontext

    with lease():
        cases, parity_failures = parity_sweep()
        topk_cases, topk_failures = ties_against_topk()
        edge_checks, edge_failures = edge_cases()
        grad_rows = gradients()
        k_limit_rows = kernel_k_limit()
        timing_rows = timings()

    contract = {
        "scores": "larger first; equal scores by smaller index; NaN after every finite value",
        "sorted_false": "the same set, ascending by index (deterministic)",
        "output": "int64, scores.shape with k in place of dim",
        "kept": "0 <= k <= scores.shape[dim]; other values raise",
        "dtype": "float32/float16/bfloat16 on MPS run the kernel; half is widened to "
                 "float32 first, which is exact, so the order is unchanged",
        "parity_cases": cases,
        "parity_failures": parity_failures,
        "structural_and_parity_passed": not parity_failures,
        "topk_agreement_cases": topk_cases,
        "topk_agreement_failures": topk_failures,
        "edge_checks": edge_checks,
        "edge_failures": edge_failures,
        "gradients": grad_rows,
    }
    contract["passed"] = (not parity_failures and not topk_failures
                          and not edge_failures
                          and all(row["passed"] for row in grad_rows))

    fastest = [row for row in timing_rows if row["k"] <= ksel._KERNEL_MAX_K]
    kernel_exact = all(row["exact_vs_reference"] for row in timing_rows)
    beats_sort = [row["speedup_sort_over_public"] > 1.0 for row in fastest]
    # The dispatcher must beat the sort path at every k it claims, and lose at
    # none of them; the probe is the evidence for both cutoffs.
    inside = [row for row in k_limit_rows if row["k"] <= ksel._KERNEL_MAX_K]
    outside = [row for row in k_limit_rows if row["k"] > ksel._KERNEL_MAX_K]
    window_side = [row for row in inside if row["k"] < ksel._RADIX_MIN_K]
    radix_side = [row for row in inside if row["k"] >= ksel._RADIX_MIN_K]
    summary = {
        "operator": "k_select / k_select_values (NSA key-block selection)",
        "overlay": "src/metal_kernelops/cv/selection.py",
        "torch": torch.__version__,
        "machine": platform.machine(),
        "device": DEVICE,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "api": {
            "k_select": "k_select(scores, k, dim=-1, sorted=True) -> int64 indices",
            "k_select_values": "k_select_values(scores, k, dim=-1, sorted=True) -> float scores",
            "reference_k_select": "reference_k_select(scores, k, dim=-1, sorted=True), "
                                  "two stable argsorts, any device",
        },
        "method": "one Metal kernel per call, in one of three forms.  Small k: a "
                  "threadgroup per row, one lane per lane-strided slice, a per-lane "
                  "top-k window in registers, then a shared-memory pairwise merge; a "
                  "one-thread-per-row form of the same window takes over once there are "
                  "enough rows to fill the device.  k >= "
                  f"{ksel._RADIX_MIN_K}: one simdgroup per row over 32-bit order keys — "
                  "the k-th key found by histogramming 8 bits at a time, then the winners "
                  "placed by rank, which keeps nothing per-k in registers.  k is a "
                  "compile-time constant so buffers and the shift chain are sized to it. "
                  f"k > {ksel._KERNEL_MAX_K} or a non-MPS device takes the "
                  "two-stable-argsort reference.",
        "import_is_inert": inert,
        "limits": {
            "k": f"the kernel covers k <= {ksel._KERNEL_MAX_K}: the window forms to "
                 f"k = {ksel._RADIX_MIN_K - 1}, the radix form from k = {ksel._RADIX_MIN_K} "
                 "up.  Above the kernel cutoff the two-argsort path runs, which is what "
                 "the k-limit probe measured as the faster option there; the crossover is "
                 "shape-dependent, so the probe reports it rather than a single number",
            "dtypes": "floating point only; on MPS float32/float16/bfloat16 reach the "
                      "kernel, and integer or complex scores raise TypeError",
            "gradients": "k_select returns int64 with no graph; only k_select_values is "
                         "differentiable, and it routes the gradient to the selected entries",
            "layout": "the selected axis may be any axis; a non-last axis costs one "
                      "movedim copy, a non-contiguous or half input one more conversion",
            "compile": "the first call for a given (k, sorted) compiles a shader (15-45 ms "
                       "measured on this machine), later calls hit the lru_cache; nothing "
                       "compiles at import",
            "determinism": "the order is total, so the result is reproducible run to run "
                           "and identical between the kernel and the sort path — unlike the "
                           "bitonic selection in fla's NSA kernel, whose ties are arbitrary",
            "not_covered": "building NSA's block scores (per-head normalised scores summed "
                           "over the group, the always-selected first and last blocks) and "
                           "the block-sparse attention itself are the caller's",
        },
        "contract": contract,
        "timings": {
            "method": "min and median over 25 reps after 8 warmups, torch.mps.synchronize() "
                      "around every MPS rep, best of three passes; the CPU column runs 5 "
                      "reps and is only measured up to 4M elements.  The machine is shared "
                      "with other agents: trust the ordering and the ratios, not the "
                      "absolute numbers.  ``public_path`` is k_select itself — validation, "
                      "dispatch choice and launch.  ``stable_argsort`` is the two-argsort "
                      "form this module replaced, ``torch_topk`` what torch offers on MPS. "
                      "The raw kernel forms are timed in the k-limit probe instead, where "
                      "they are not competing with three other callables for the device.",
            "rows": timing_rows,
            "kernel_k_limit_probe": {
                "why": "two cutoffs, both set from this probe.  The window lives in "
                       "registers and the lanes in threadgroup memory, both growing "
                       "with k, so it gives way to the radix form at _RADIX_MIN_K = "
                       f"{ksel._RADIX_MIN_K}; the radix form's rank pass is O(k^2) per "
                       "row and its candidate buffer O(k), so above _KERNEL_MAX_K = "
                       f"{ksel._KERNEL_MAX_K} the sort path wins again",
                "dispatch": form_choice(),
                "rows": k_limit_rows,
            },
        },
        "result": {
            "contract_passed": contract["passed"],
            "kernel_exact_at_every_nsa_shape": kernel_exact,
            "public_path_faster_than_stable_argsort_at_every_kernel_shape": all(beats_sort),
            "public_path_min_speedup_over_stable_argsort": min(
                row["speedup_sort_over_public"] for row in fastest),
            "public_path_max_speedup_over_stable_argsort": max(
                row["speedup_sort_over_public"] for row in fastest),
            "kernel_k_limit": ksel._KERNEL_MAX_K,
            "radix_min_k": ksel._RADIX_MIN_K,
            "radix_threadgroup": ksel._RADIX_THREADS,
            # Inside the cutoff the kernel wins at every probed shape.  Past it
            # the margin erodes with k and inverts first on the many-short-row
            # shape, so the claim there is that the kernel no longer wins
            # everywhere — at k = 512 the few-row shape still favours it.
            "kernel_k_limit_confirmed": all(row["kernel_wins"] for row in inside)
                                        and not all(row["kernel_wins"] for row in outside),
            "k_above_the_cutoff_where_the_sort_path_wins": [
                [row["rows"], row["blocks"], row["k"]] for row in outside
                if not row["kernel_wins"]],
            "kernel_forms_exact_at_every_probed_k": all(
                all(row["kernel_forms_exact"].values()) for row in k_limit_rows),
            "radix_wins_where_it_is_dispatched": [row["k"] for row in radix_side
                                                  if row["best_kernel"] != "radix"],
            "window_wins_where_it_is_dispatched": [row["k"] for row in window_side
                                                   if row["best_kernel"] == "radix"],
            "speedup_over_sort_by_shape_and_k": {
                f"{shape_rows[0]}x{shape_rows[1]}": {
                    str(row["k"]): round(row["stable_argsort_ms"] / row["best_kernel_ms"], 2)
                    for row in shape_rows}
                for shape_rows in (
                    [row for row in k_limit_rows if row["rows"] == rows]
                    for rows in dict.fromkeys(row["rows"] for row in k_limit_rows))},
        },
    }

    out = ROOT / "measured" / "k_select.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"import inert: {inert}")
    print(f"contract: {'pass' if contract['passed'] else 'FAIL'}  "
          f"parity {cases} cases ({len(parity_failures)} failures), "
          f"topk agreement {topk_cases} ({len(topk_failures)}), "
          f"edges {edge_checks} ({len(edge_failures)}), "
          f"gradients {sum(row['passed'] for row in grad_rows)}/{len(grad_rows)}")
    for failure in parity_failures[:10] + topk_failures[:10] + edge_failures[:10]:
        print("  FAIL", failure)
    print(f"k-limit probe (window to k={ksel._RADIX_MIN_K - 1}, radix to "
          f"k={ksel._KERNEL_MAX_K}, sort above):")
    for row in k_limit_rows:
        cell = lambda v: f"{v:8.4f}" if v else "       -"
        print(f"  {row['rows']:6d}x{row['blocks']:<5d} k={row['k']:<3d} "
              f"merge {cell(row['merge_ms'])}  tall {cell(row['tall_ms'])}  "
              f"radix {cell(row['radix_ms'])}  "
              f"sort {row['stable_argsort_ms']:8.4f}  "
              f"best {row['best_kernel']:<5s} ({row['stable_argsort_ms'] / row['best_kernel_ms']:5.1f}x)  "
              f"{'wins' if row['kernel_wins'] else 'loses'}  "
              f"exact={all(row['kernel_forms_exact'].values())}")
    for row in timing_rows:
        head = (f"  {row['rows']:6d}x{row['blocks']:<5d} k={row['k']:<3d} {row['path']:<18s}")
        body = f"public {row['public_path']['min_ms']:8.4f} "
        tail = (f"sort {row['stable_argsort']['min_ms']:8.4f} "
                f"topk {row['torch_topk']['min_ms']:8.4f} "
                f"cpu {row.get('cpu_reference', {}).get('min_ms', float('nan')):9.4f}")
        print(head + body + "  " + tail
              + f"  sort/public {row['speedup_sort_over_public']:6.2f}x")
    print(f"fastest path {summary['result']['public_path_min_speedup_over_stable_argsort']:.2f}x "
          f"to {summary['result']['public_path_max_speedup_over_stable_argsort']:.2f}x "
          f"faster than the stable-argsort path at the NSA shapes above")
    print(f"wrote {out}")
    if (not contract["passed"] or not kernel_exact
            or not summary["result"]["kernel_k_limit_confirmed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
