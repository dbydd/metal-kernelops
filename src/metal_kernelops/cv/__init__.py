"""Tree-external computer-vision operators for Apple Silicon.

These functions are not aten operators. They are explicit imports from
``metal_kernelops.cv``; nothing here registers a dispatch key or changes
PyTorch when it is imported.

Families:

* Mamba-2 chunk scan and one-step state update, from state-spaces/mamba.
* Causal depthwise convolution and gated RMSNorm used by Mamba-2 blocks.
* Four-direction 2D path scan from VMamba's SS2D.
* Deterministic top-k key-block selection used by DeepSeek NSA.
"""

from .causal_conv1d import causal_conv1d_fn, causal_conv1d_ref, causal_conv1d_update
from .gated_norm import rmsnorm_gated, rmsnorm_gated_reference
from .selection import k_select, k_select_values, reference_k_select
from .mamba2_chunk import mamba2_chunk_scan, mamba2_chunk_scan_reference
from .mamba2_state import mamba2_state_update, mamba2_state_update_reference
from .path_scan import path_orders, ss2d_scan, ss2d_scan_reference
from .roi_align import roi_align

__all__ = [
    "causal_conv1d_fn",
    "causal_conv1d_ref",
    "causal_conv1d_update",
    "k_select",
    "k_select_values",
    "mamba2_chunk_scan",
    "mamba2_chunk_scan_reference",
    "mamba2_state_update",
    "mamba2_state_update_reference",
    "path_orders",
    "reference_k_select",
    "roi_align",
    "rmsnorm_gated",
    "rmsnorm_gated_reference",
    "ss2d_scan",
    "ss2d_scan_reference",
]
