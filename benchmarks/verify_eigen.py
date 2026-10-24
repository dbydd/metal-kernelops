"""Verify and measure the MPS eigen overlays in ``metal_kernelops.eigen_ops``.

Run with ``python benchmarks/verify_eigen.py`` from the repository root (with
``src`` on ``PYTHONPATH`` or the package installed). It registers the overlays
in this process, checks the forward and backward semantics against the CPU
reference, and writes ``measured/eigen.json``.

The forward check is residual-based: ``linalg_eig`` does not define an
eigenvector phase or an eigenvalue order, so eigenvalues are compared after
sorting and eigenvectors through ``A V - V diag(w)``.

The backward check has two halves. Against the CPU reference it uses matrices
whose eigenvector matrix is well conditioned, because the gradient of ``eig``
divides by eigenvalue differences and amplifies any forward error. On a
non-normal matrix it instead compares the MPS gradient with PyTorch's
registered formula evaluated in double precision on the same decomposition,
which isolates the implementation from the conditioning of the problem.

Timings are wall clock around ``torch.mps.synchronize()``, minimum of repeated
runs. This machine is shared with other work, so the ratios are indicative
rather than isolated measurements. Each timing row also records how many sweeps
and rotations the kernel spent against its ``30n + 100`` sweep budget, and the
kernel dispatch time with the input refresh subtracted, because the kernel
overwrites its input: a repeat on an already-triangular matrix does no work and
would report a fraction of the real cost.
"""

import json
import os
import platform
import sys
import time
from pathlib import Path

import torch
from torch.library import Library

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from metal_kernelops import eigen_ops

CHAMBER = Path(
    os.environ.get(
        "RESEARCH_CHAMBER_ROOT", "/Users/dbydd/vibe-agent-working-dir/ResearchChamber"
    )
)
if str(CHAMBER) not in sys.path:
    sys.path.insert(0, str(CHAMBER))

try:  # the broker is a machine-local convenience, not a dependency
    from shared_scripts.compute_broker import lease_guard
except ImportError:
    import contextlib

    def lease_guard(*args, **kwargs):
        return contextlib.nullcontext()

CHECKED_TORCH = "2.14.1"
OUTPUT = ROOT / "measured" / "eigen.json"

_OVERLAYS = ["aten::" + name for name, _ in eigen_ops.OVERLAYS]
_AUTOGRAD_OVERLAYS = ["aten::" + name for name, _ in eigen_ops.AUTOGRAD_OVERLAYS]

#: Registrations are dropped when the ``Library`` object is collected.
_LIBRARY = None


def install():
    global _LIBRARY
    library = Library("aten", "FRAGMENT")
    for name, function in eigen_ops.OVERLAYS:
        library.impl(name, function, "MPS")
    for name, function in eigen_ops.AUTOGRAD_OVERLAYS:
        library.impl(name, function, "AutogradMPS")
    _LIBRARY = library
    return library


def eigenvalue_error(ours, reference):
    """Largest distance from an eigenvalue to the nearest reference one, both ways.

    ``eig`` defines neither the order nor the phase, and conjugate pairs share a
    real part to the last bit, so sorting is not a reliable pairing. Nearest
    neighbour is.
    """
    ours = ours.reshape(-1, ours.shape[-1]).to(torch.complex128)
    reference = reference.reshape(-1, reference.shape[-1]).to(torch.complex128)
    distances = (ours.unsqueeze(-1) - reference.unsqueeze(-2)).abs()
    forward = distances.amin(dim=-1).amax().item()
    backward = distances.amin(dim=-2).amax().item()
    return max(forward, backward)



def residual(matrix, values, vectors):
    """Relative ``max |A V - V diag(w)|``."""
    dtype = torch.complex128 if matrix.dtype in (torch.float64, torch.complex128) else torch.complex64
    matrix = matrix.detach().cpu().to(dtype)
    values = values.detach().cpu().to(dtype)
    vectors = vectors.detach().cpu().to(dtype)
    matrices = matrix.reshape(-1, matrix.shape[-1], matrix.shape[-1])
    values = values.reshape(matrices.shape[:-1])
    vectors = vectors.reshape(matrices.shape)
    left = matrices @ vectors
    right = vectors * values.unsqueeze(-2)
    scale = matrices.abs().amax(dim=(-2, -1)).clamp_min(1e-30)
    return ((left - right).abs().amax(dim=(-2, -1)) / scale).max().item()


def forward_case(name, matrix):
    """Returns the row for the JSON report."""
    device_matrix = matrix.to("mps")
    values, vectors = torch.linalg.eig(device_matrix)
    torch.mps.synchronize()
    reference_dtype = torch.complex128 if matrix.is_complex() else torch.float64
    ref_values, ref_vectors = torch.linalg.eig(matrix.to(reference_dtype))
    values = values.reshape(-1, values.shape[-1]).cpu()
    ref_values = ref_values.reshape(-1, ref_values.shape[-1])
    error = eigenvalue_error(values, ref_values)
    scale = ref_values.abs().amax().item()
    row = {
        "case": name,
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "result_dtype": str(values.dtype),
        "eigenvalue_rel_err": error / max(scale, 1e-30),
        "residual": residual(device_matrix, values, vectors),
        "reference_residual": residual(matrix, ref_values, ref_vectors),
    }
    return row


def backward_case(name, matrix, weights):
    """Compare the MPS gradient with CPU autograd on a well conditioned matrix."""
    a, b, w = weights
    double_dtype = torch.complex128 if matrix.is_complex() else torch.float64

    def loss_from(input_matrix):
        device = input_matrix.device
        values, vectors = torch.linalg.eig(input_matrix)
        dtype = values.real.dtype
        return (values.real * a.to(device=device, dtype=dtype)
                + values.imag * b.to(device=device, dtype=dtype)).sum() + (
            vectors.abs() ** 2 * w.to(device=device, dtype=dtype)
        ).sum()

    cpu = matrix.to(double_dtype).clone().requires_grad_(True)
    loss_from(cpu).backward()
    device_matrix = matrix.to(torch.complex64 if matrix.is_complex() else torch.float32)
    device_matrix = device_matrix.to("mps").requires_grad_(True)
    loss_from(device_matrix).backward()
    difference = (device_matrix.grad.cpu().to(cpu.grad.dtype) - cpu.grad).abs().amax().item()
    scale = cpu.grad.abs().amax().item()
    return {"case": name, "shape": list(matrix.shape), "reference_dtype": str(matrix.dtype),
            "grad_rel_err": difference / max(scale, 1e-30)}


def formula_case(matrix):
    """MPS gradient against PyTorch's formula evaluated in double on the same V."""
    values, vectors, inverse = eigen_ops._decompose(matrix.to("mps"), need_inverse=True)
    torch.mps.synchronize()
    torch.manual_seed(11)
    n = matrix.shape[-1]
    a = torch.randn(n, dtype=torch.float64)
    b = torch.randn(n, dtype=torch.float64)
    w = torch.randn(n, n, dtype=torch.float64)
    gw = (a + 1j * b).to(torch.complex64)
    gv = 2 * w.to(device="mps", dtype=torch.float32).to(torch.complex64) * vectors
    got = eigen_ops._EigBackward._grad(gw, gv, values, vectors, inverse, False)

    v = vectors.cpu().to(torch.complex128)
    wh = inverse.cpu().to(torch.complex128)
    lc = values.cpu().to(torch.complex128)
    gw = gw.cpu().to(torch.complex128)
    gv = gv.cpu().to(torch.complex128)
    vh = v.conj().transpose(-1, -2)
    vhgv = vh @ gv
    diagonal = torch.diagonal(vhgv)
    projected = vhgv - vh @ (v * diagonal.real.unsqueeze(-2))
    econj = lc.conj().unsqueeze(-2) - lc.conj().unsqueeze(-1)
    econj.diagonal().fill_(1.0)
    mixed = projected / econj
    mixed.diagonal().copy_(gw)
    expected = (wh.conj().transpose(-1, -2) @ (mixed @ vh)).real
    difference = (got.cpu() - expected).abs().amax().item()
    return {"case": "formula consistency", "shape": list(matrix.shape),
            "grad_rel_err": difference / max(expected.abs().amax().item(), 1e-30)}


def eigvals_cases():
    torch.manual_seed(5)
    n = 6
    # Symmetric: the eigenvalue problem is well conditioned, so the MPS result
    # can be compared with the double-precision reference directly.
    factors = torch.randn(n, n, dtype=torch.float64)
    matrix = (factors + factors.T).to(torch.float32)
    device = matrix.to("mps")
    values = torch.linalg.eigvals(device)
    torch.mps.synchronize()
    reference = torch.linalg.eigvals(matrix.to(torch.float64))
    error = eigenvalue_error(values.cpu(), reference)
    out = torch.empty_like(values)
    returned = torch.linalg.eigvals(device, out=out)
    out_ok = returned.data_ptr() == out.data_ptr() and torch.equal(returned, values)

    a = torch.randn(n)
    cpu = matrix.to(torch.float64).clone().requires_grad_(True)
    (torch.linalg.eigvals(cpu).real * a).sum().backward()
    grad_device = matrix.clone().to("mps").requires_grad_(True)
    (torch.linalg.eigvals(grad_device).real * a.to("mps")).sum().backward()
    grad_error = (grad_device.grad.cpu().to(torch.float64) - cpu.grad).abs().amax().item()
    grad_scale = cpu.grad.abs().amax().item()
    return {
        "eigenvalue_rel_err": error / reference.abs().amax().item(),
        "out_variant_writes_in_place": bool(out_ok),
        "grad_rel_err": grad_error / grad_scale,
    }


def out_case():
    torch.manual_seed(6)
    matrix = torch.randn(2, 4, 4, device="mps")
    values = torch.empty(2, 4, dtype=torch.complex64, device="mps")
    vectors = torch.empty(2, 4, 4, dtype=torch.complex64, device="mps")
    returned = torch.linalg.eig(matrix, out=(values, vectors))
    torch.mps.synchronize()
    return {
        "eig_out_writes_in_place": bool(returned[0].data_ptr() == values.data_ptr()
                                        and returned[1].data_ptr() == vectors.data_ptr()),
    }


def error_cases():
    results = {}
    try:
        torch.linalg.eig(torch.randn(3, 4, device="mps"))
        results["non_square"] = "no error"
    except RuntimeError as error:
        results["non_square"] = type(error).__name__
    try:
        torch.linalg.eig(torch.randn(4, 4, device="mps", dtype=torch.float16))
        results["float16"] = "no error"
    except TypeError as error:
        results["float16"] = type(error).__name__
    try:
        torch.linalg.eig(torch.randn(0, 0, device="mps"))
        results["order_0"] = "no error"
    except RuntimeError as error:
        results["order_0"] = type(error).__name__
    return results


def min_ms(function, repeats=9, calls=1, warmup=2):
    """Minimum milliseconds per call over ``repeats`` synchronized batches."""
    for _ in range(warmup):
        function()
    torch.mps.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(calls):
            function()
        torch.mps.synchronize()
        samples.append((time.perf_counter() - start) * 1e3 / calls)
    return min(samples)


def kernel_case(n, batch, calls=8):
    """Sweeps, rotations and dispatch milliseconds for one shape.

    Every call refreshes the working buffer from a pristine copy first: the
    kernel overwrites its input, so a repeat on an already-triangular matrix
    does no QR work and would report a fraction of the real cost. The refresh is
    timed in the same batch and subtracted.
    """
    threads = eigen_ops._threads_for(n)
    kernel = eigen_ops._kernel(n, threads)
    source = torch.randn(batch, n, n, dtype=torch.complex64, device="mps")
    work = source.clone()
    unitary = torch.empty(batch, n, n, dtype=torch.complex64, device="mps")
    vectors_y = torch.empty_like(unitary)
    stats = torch.empty(batch, 3, dtype=torch.int32, device="mps")
    work_real, z_real, y_real = (torch.view_as_real(t) for t in (work, unitary, vectors_y))
    budget = eigen_ops._SWEEPS_PER_ROW * n + eigen_ops._SWEEPS_FLOOR

    def dispatch():
        work.copy_(source)
        kernel(work_real, z_real, y_real, y_real, stats, eigen_ops._QR_TOL, budget, 0,
               threads=batch * threads, group_size=threads)

    total = min_ms(dispatch, calls=calls)
    copy = min_ms(lambda: work.copy_(source), calls=calls)
    sweeps = int(stats[:, 1].max())
    rotations = int(stats[:, 2].max())
    return {
        "kernel_ms": round(total - copy, 3),
        "copy_ms": round(copy, 3),
        "sweeps": sweeps,
        "rotations": rotations,
        "sweep_budget": budget,
        "rotations_per_sweep": round(rotations / sweeps, 1) if sweeps else None,
    }


def timing_case(n, batch, repeats=7, warmup=2):
    torch.manual_seed(7)
    matrix = torch.randn(batch, n, n, device="mps")
    overlay = min_ms(lambda: torch.linalg.eig(matrix), repeats=repeats, warmup=warmup)

    host = matrix.cpu()
    torch.linalg.eig(host)

    def fallback():
        values, vectors = torch.linalg.eig(host)
        return values.to("mps"), vectors.to("mps")

    cpu_ms = min_ms(fallback, repeats=repeats, warmup=warmup)
    row = {
        "n": n,
        "batch": batch,
        "overlay_ms": round(overlay, 3),
        "cpu_fallback_ms": round(cpu_ms, 3),
        "speedup": round(cpu_ms / overlay, 2),
    }
    row.update(kernel_case(n, batch))
    return row


def primitive_cases():
    """The MPS features this kernel depends on, checked on the device."""
    torch.manual_seed(9)
    complex_tensor = torch.complex(torch.randn(4, 4), torch.randn(4, 4)).to("mps")
    real_view = torch.view_as_real(complex_tensor)
    round_trip = torch.view_as_complex(real_view)
    other = torch.complex(torch.randn(4, 4), torch.randn(4, 4)).to("mps")
    product = complex_tensor @ other
    reference = complex_tensor.cpu().to(torch.complex128) @ other.cpu().to(torch.complex128)
    torch.mps.synchronize()
    return {
        "complex64_tensor": str(complex_tensor.dtype),
        "view_as_real_shape": list(real_view.shape),
        "view_round_trip": bool(torch.equal(round_trip, complex_tensor)),
        "complex_matmul_rel_err": (product.cpu().to(torch.complex128) - reference).abs().amax().item()
        / reference.abs().amax().item(),
        "compile_shader": hasattr(torch.mps, "compile_shader"),
    }


def well_conditioned(n, complex_input, seed):
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(n, n, dtype=torch.float64, generator=generator)
    if complex_input:
        q = q + 1j * torch.randn(n, n, dtype=torch.float64, generator=generator)
    diagonal = torch.linspace(-3, 3, n, dtype=torch.float64)
    if complex_input:
        diagonal = diagonal + 0.5j * torch.linspace(-1, 1, n, dtype=torch.float64)
    matrix = q.to(torch.complex128) @ torch.diag(diagonal.to(torch.complex128))
    matrix = matrix @ torch.linalg.inv(q.to(torch.complex128))
    return matrix if complex_input else matrix.real


def main():
    if torch.__version__.split("+")[0] != CHECKED_TORCH:
        raise SystemExit(f"checked against torch {CHECKED_TORCH}, running {torch.__version__}")
    with lease_guard("mps-kernelops", tier=1, task_name="verify_eigen"):
        report = measure_all()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {OUTPUT}")


def measure_all():
    install()

    torch.manual_seed(0)
    report = {
        "torch": torch.__version__,
        "device": "mps",
        "machine": platform.machine(),
        "overlays": _OVERLAYS,
        "autograd_overlays": _AUTOGRAD_OVERLAYS,
        "dtype": {
            "input": ["float32", "complex64"],
            "output": "complex64",
            "note": "MPS has no float64; other dtypes raise TypeError",
        },
        "mps_primitives": primitive_cases(),
    }

    print("forward")
    forward = []
    for shape, complex_input in (
        ((1, 1), False),
        ((4, 4), False),
        ((4, 4), True),
        ((3, 8, 8), False),
        ((2, 3, 16, 16), False),
        ((32, 32), False),
    ):
        torch.manual_seed(1)
        matrix = torch.randn(*shape, dtype=torch.float32)
        if complex_input:
            matrix = torch.complex(matrix, torch.randn(*shape, dtype=torch.float32))
        forward.append(forward_case(f"{shape}{' complex' if complex_input else ''}", matrix))
        print(f"  {forward[-1]['case']}: rel_err={forward[-1]['eigenvalue_rel_err']:.2e} "
              f"residual={forward[-1]['residual']:.2e} (reference {forward[-1]['reference_residual']:.2e})")
    report["forward"] = forward

    print("backward")
    backward = []
    for n, complex_input in ((4, False), (6, False), (4, True), (8, True)):
        matrix = well_conditioned(n, complex_input, seed=n)
        torch.manual_seed(2)
        weights = (torch.randn(n, dtype=torch.float64),
                   torch.randn(n, dtype=torch.float64),
                   torch.randn(n, n, dtype=torch.float64))
        backward.append(backward_case(f"n={n}{' complex' if complex_input else ''}", matrix, weights))
        print(f"  {backward[-1]['case']}: grad_rel_err={backward[-1]['grad_rel_err']:.2e}")

    non_normal = (torch.randn(8, 8, dtype=torch.float64)
                  @ torch.diag(torch.linspace(-2, 2, 8, dtype=torch.float64))
                  @ torch.linalg.inv(torch.randn(8, 8, dtype=torch.float64)))
    formula = formula_case(non_normal.real.to(torch.float32))
    backward.append(formula)
    print(f"  {formula['case']}: grad_rel_err={formula['grad_rel_err']:.2e}")
    report["backward"] = backward
    report["eigvals"] = eigvals_cases()
    print(f"  eigvals: {report['eigvals']}")

    print("out variants and errors")
    report["out"] = out_case()
    report["errors"] = error_cases()
    print(f"  {report['out']} {report['errors']}")

    torch.manual_seed(3)
    symmetric = torch.randn(16, 16)
    symmetric = symmetric + symmetric.T
    symmetric = symmetric.to("mps")
    values = torch.linalg.eigvals(symmetric).cpu().real
    reference = torch.linalg.eigvalsh(symmetric.cpu())
    report["symmetric"] = {
        "max_abs_err_vs_eigvalsh": (torch.sort(values).values - reference).abs().amax().item()
    }
    repeated = torch.eye(6, device="mps")
    repeated[0, 1] = 1.0  # defective Jordan block
    values, vectors = torch.linalg.eig(repeated)
    torch.mps.synchronize()
    report["defective"] = {
        "values": sorted(values.cpu().real.tolist()),
        "finite": bool(torch.isfinite(vectors).all()),
        "order_0": error_cases()["order_0"],
    }
    print(f"  symmetric: {report['symmetric']} defective: {report['defective']}")

    print("timing (shared machine, contended)")
    timings = []
    for n, batch in ((16, 1), (32, 1), (64, 1), (128, 1), (32, 64), (64, 64), (128, 8)):
        row = timing_case(n, batch)
        timings.append(row)
        print(f"  n={n} batch={batch}: overlay={row['overlay_ms']:.2f} ms "
              f"cpu_fallback={row['cpu_fallback_ms']:.2f} ms speedup={row['speedup']}x "
              f"kernel={row['kernel_ms']:.2f} ms copy={row['copy_ms']:.2f} ms "
              f"sweeps={row['sweeps']}/{row['sweep_budget']} rotations={row['rotations']}")
    report["timing_ms"] = timings
    report["timing_note"] = (
        "wall clock including torch.mps.synchronize(); minimum of 7 synchronized runs; "
        "kernel_ms excludes the input refresh copy measured in the same batch; "
        "machine shared, ratios indicative"
    )
    return report


if __name__ == "__main__":
    main()
