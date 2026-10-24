"""MPS dispatch-key overlays.

Importing this module changes nothing. Call :func:`install` to take over MPS
kernels for operators this package implements; from then on ordinary PyTorch
code reaches those kernels with no call-site change.

``install()`` registers only kernels that are exact and worth running on the
device. ``install(compatibility=True)`` also registers exact kernels whose cost
is worse than the missing-error path they replace, so callers choose that trade.

Registration is permanent for the life of the process. There is no uninstall.
"""

from importlib import import_module

import torch
from torch.library import Library

CHECKED_TORCH = "2.14.1"

_installed = {}
_library = None

_MODULES = (
    "index_ops", "ndtri_ops", "frexp_ops", "airy_ops", "laguerre_ops", "rrelu_ops",
    "decomposition_ops", "log_ndtr_ops", "fractional_pool_ops", "lanczos_ops",
    "legendre_ops", "matrix_sqrt_ops", "mode_ops", "segment_ops", "convolution_ops",
    "margin_ops", "householder_ops", "attention_ops", "adaptive_pool_ops", "eigen_ops",
    "distance_ops",
    "spdiags_ops",
    "sparse_addmm_ops",
)

_COMPATIBILITY_MODULES = ("ldl_ops",)


def _previous(operator, dispatch):
    table = torch._C._dispatch_dump_table(operator)
    return next(
        (line for line in table.splitlines() if line.startswith(f"{dispatch}:")),
        None,
    )


def install(compatibility=False):
    """Register overlays on MPS. Returns the coverage table.

    ``compatibility`` adds exact implementations whose measured cost is worse
    than leaving the operator unsupported, such as the sequential LDL
    factorization. The default table contains only kernels that provide a real
    device path with competitive performance.
    """
    version = torch.__version__.split("+")[0]
    if version != CHECKED_TORCH:
        raise RuntimeError(
            f"overlays were checked against torch {CHECKED_TORCH}, running {version}"
        )
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available on this machine")

    if _installed:
        return coverage()

    overlays = []
    autograd_overlays = []
    sparse_overlays = []
    names = set()
    autograd_names = set()
    sparse_names = set()
    for module_name in _MODULES + (_COMPATIBILITY_MODULES if compatibility else ()):
        module = import_module(f".{module_name}", __package__)
        for name, function in getattr(module, "OVERLAYS", ()):
            if name in names:
                raise RuntimeError(f"duplicate MPS overlay: aten::{name}")
            names.add(name)
            overlays.append((name, function, module_name in _COMPATIBILITY_MODULES))
        for name, function in getattr(module, "AUTOGRAD_OVERLAYS", ()):
            if name in autograd_names:
                raise RuntimeError(f"duplicate AutogradMPS overlay: aten::{name}")
            autograd_names.add(name)
            autograd_overlays.append((name, function, module_name in _COMPATIBILITY_MODULES))
        for name, function in getattr(module, "SPARSE_OVERLAYS", ()):
            if name in sparse_names:
                raise RuntimeError(f"duplicate SparseMPS overlay: aten::{name}")
            sparse_names.add(name)
            sparse_overlays.append((name, function, module_name in _COMPATIBILITY_MODULES))

    global _library
    _library = Library("aten", "FRAGMENT")
    tables = (
        ("MPS", overlays, ""),
        ("AutogradMPS", autograd_overlays, "@AutogradMPS"),
        ("SparseMPS", sparse_overlays, "@SparseMPS"),
    )
    for dispatch, entries, suffix in tables:
        for name, function, compatibility_only in entries:
            operator = f"aten::{name}"
            replaced = torch._C._dispatch_has_kernel_for_dispatch_key(operator, dispatch)
            _library.impl(name, function, dispatch)
            _installed[operator + suffix] = {
                "replaced_existing": replaced,
                "previous_kernel": _previous(operator, dispatch),
                "kernel": f"{function.__module__}.{function.__name__}",
                "compatibility": compatibility_only,
            }
    return coverage()


def coverage():
    """What this package has taken over: one row per operator and dispatch key."""
    return {
        operator: dict(row) | {"torch": torch.__version__}
        for operator, row in _installed.items()
    }
