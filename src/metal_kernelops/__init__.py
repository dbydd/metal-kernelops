"""Front end for the Metal selective scan kernels and the MPS accelerator.

Tree-external CV operators live in ``metal_kernelops.cv``: Mamba-2 chunk scan
and decode update, causal convolution, gated RMSNorm, VMamba path scan, and
NSA key-block selection. They are explicit imports, not dispatch overlays.

The recurrence is the one from mamba_ssm's ``selective_scan_fn``:

    x_t = exp(dt_t * A) * x_{t-1} + dt_t * B_t * u_t
    y_t = <x_t, C_t>

``D`` (skip), ``z`` (gate) and ``delta_bias`` are applied afterwards as pointwise
torch ops rather than inside the kernel.

Layouts:
    u, dt:  (batch, dim, length)   contiguous
    A:      (dim, dstate)          negative reals
    B, C:   (batch, groups, dstate, length)
    D:      (dim,)
    z:      (batch, dim, length)
"""

from . import reference
from .scan import (
    BLOCK,
    CHUNK,
    scan_mps,
    scan_reference,
    selective_scan,
    selective_scan_fn,
)

__all__ = [
    "BLOCK",
    "CHUNK",
    "reference",
    "scan_mps",
    "scan_reference",
    "selective_scan",
    "selective_scan_fn",
]

__version__ = "0.3.0"
