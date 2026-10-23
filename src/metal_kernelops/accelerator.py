"""MPS dispatch-key overlays.

Importing this module changes nothing. Call :func:`install` to take over MPS
kernels for operators this package implements; from then on ordinary PyTorch
code reaches those kernels with no call-site change.

Every overlay here exists because the operator raises ``NotImplementedError``
on the MPS device today. Operators that run but run slowly belong in a Metal
kernel, not in this table.

Registration is permanent for the life of the process. There is no uninstall.
"""

import torch
from torch.library import Library

CHECKED_TORCH = "2.14.1"

_installed = {}
_library = None


def _take(self, index):
    return self.reshape(-1).index_select(0, index.reshape(-1)).reshape(index.shape)


def _channel_shuffle(self, groups):
    channels = self.shape[1]
    if channels % groups:
        raise ValueError(f"channel count {channels} is not divisible by {groups}")
    shape = self.shape
    shuffled = (
        self.reshape(shape[0], groups, channels // groups, *shape[2:])
        .transpose(1, 2)
        .reshape(shape)
    )
    return shuffled.contiguous()


def _put_(self, index, source, accumulate=False):
    flat = self.reshape(-1)
    flat.index_put_((index.reshape(-1),), source.reshape(-1), accumulate)
    if flat.data_ptr() != self.data_ptr():
        self.copy_(flat.reshape(self.shape))
    return self


_OVERLAYS = (
    ("take", _take),
    ("channel_shuffle", _channel_shuffle),
    ("put_", _put_),
)


def install():
    """Register the overlays on the MPS dispatch key. Returns the coverage table."""
    version = torch.__version__.split("+")[0]
    if version != CHECKED_TORCH:
        raise RuntimeError(
            f"overlays were checked against torch {CHECKED_TORCH}, running {version}"
        )
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available on this machine")

    if _installed:
        return coverage()

    global _library
    _library = Library("aten", "FRAGMENT")
    for name, function in _OVERLAYS:
        operator = f"aten::{name}"
        replaced = torch._C._dispatch_has_kernel_for_dispatch_key(operator, "MPS")
        _library.impl(name, function, "MPS")
        _installed[operator] = {"replaced_existing": replaced, "kernel": function.__name__}
    return coverage()


def coverage():
    """What this package has taken over: one row per operator, current process only."""
    return {
        operator: dict(row) | {"torch": torch.__version__}
        for operator, row in _installed.items()
    }
