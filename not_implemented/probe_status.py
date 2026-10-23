"""Classify candidates from portable_candidates.txt as running or raising on MPS.

Each entry calls the public PyTorch API, not the aten operator, because the
public API is what a model touches. Run with the environment that has torch
installed and MPS available:

    python not_implemented/probe_status.py > not_implemented/mps_status.json
"""

import json

import torch
import torch.nn.functional as F

d = "mps"
f32 = {"dtype": torch.float32, "device": d}
i64 = {"dtype": torch.int64, "device": d}

calls = {
    "take": lambda: torch.take(torch.arange(12.0, **f32).reshape(3, 4), torch.tensor([0, 5], device=d)),
    "narrow_copy": lambda: torch.narrow_copy(torch.arange(12.0, **f32).reshape(3, 4), 1, 1, 2),
    "logit": lambda: torch.logit(torch.rand(4, **f32) * 0.5 + 0.25),
    "heaviside": lambda: torch.heaviside(torch.randn(4, **f32), torch.zeros(4, **f32)),
    "frexp": lambda: torch.frexp(torch.randn(4, **f32)),
    "logspace": lambda: torch.logspace(-2, 2, 5, **f32),
    "mode": lambda: torch.mode(torch.randn(3, 4, **f32), dim=1),
    "channel_shuffle": lambda: F.channel_shuffle(torch.randn(1, 4, 2, 2, **f32), 2),
    "put": lambda: torch.zeros(6, **f32).put_(torch.tensor([0, 2], device=d), torch.ones(2, **f32)),
    "special_ndtri": lambda: torch.special.ndtri(torch.rand(4, **f32) * 0.5 + 0.25),
    "special_log_ndtr": lambda: torch.special.log_ndtr(torch.randn(4, **f32)),
    "special_airy_ai": lambda: torch.special.airy_ai(torch.randn(4, **f32)),
    "special_laguerre": lambda: torch.special.laguerre_polynomial_l(torch.rand(4, **f32), 2),
    "special_legendre": lambda: torch.special.legendre_polynomial_p(torch.rand(4, **f32), 2),
    "adaptive_max_pool3d": lambda: F.adaptive_max_pool3d(torch.randn(1, 2, 4, 4, 4, **f32), 2),
    "adaptive_avg_pool3d": lambda: F.adaptive_avg_pool3d(torch.randn(1, 2, 4, 4, 4, **f32), 2),
    "fractional_max_pool2d": lambda: F.fractional_max_pool2d(torch.randn(1, 2, 6, 6, **f32), 2, output_size=(3, 3)),
    "segment_reduce": lambda: torch.segment_reduce(torch.randn(6, **f32), "sum", lengths=torch.tensor([2, 4], device=d)),
    "linalg_eig": lambda: torch.linalg.eig(torch.randn(4, 4, **f32)),
    "linalg_eigvals": lambda: torch.linalg.eigvals(torch.randn(4, 4, **f32)),
    "linalg_ldl_solve": lambda: torch.linalg.solve(
        torch.randn(4, 4, **f32) @ torch.randn(4, 4, **f32).T + torch.eye(4, **f32), torch.randn(4, 2, **f32)
    ),
    "aminmax": lambda: torch.aminmax(torch.randn(4, **f32), dim=0),
    "sspaddmm": lambda: torch.addmm(torch.randn(3, 3, **f32), torch.randn(3, 2, **f32), torch.randn(2, 3, **f32)),
    "rrelu_with_noise": lambda: F.rrelu(torch.randn(4, **f32)),
    "multi_margin_loss": lambda: F.multi_margin_loss(torch.randn(4, 3, **f32), torch.tensor([0, 1, 2, 0], device=d)),
    "multilabel_margin_loss": lambda: F.multilabel_margin_loss(
        torch.randn(3, 4, **f32), torch.tensor([[0, 1, -1, -1]] * 3, device=d)
    ),
    "batch_norm_update_stats": lambda: torch.ops.aten.batch_norm_update_stats(
        torch.randn(2, 3, **f32), None, None, 0.1
    ),
    "masked_softmax": lambda: torch.ops.aten._masked_softmax(
        torch.randn(2, 3, **f32), torch.zeros(2, 3, dtype=torch.bool, device=d), None, None
    ),
    "transform_bias_rescale_qkv": lambda: torch.ops.aten._transform_bias_rescale_qkv(
        torch.randn(2, 3, 4, **f32), torch.randn(12, **f32), 3
    ),
    "upsample_lanczos2d_aa": lambda: F.interpolate(torch.randn(1, 1, 8, 8, **f32), scale_factor=2, mode="bicubic"),
    "slow_conv_transpose2d": lambda: F.conv_transpose2d(
        torch.randn(1, 2, 4, 4, **f32), torch.randn(2, 3, 3, 3, **f32)
    ),
    "slow_conv_dilated2d": lambda: F.conv2d(
        torch.randn(1, 2, 6, 6, **f32), torch.randn(3, 2, 3, 3, **f32), dilation=2
    ),
    "spdiags": lambda: torch.ops.aten._spdiags(
        torch.randn(3, 3, **f32), torch.tensor([0, 1, -2], **i64), torch.tensor([3, 3], **i64)
    ),
    "add_relu": lambda: F.relu(torch.randn(4, **f32)),
    "addmm_activation": lambda: torch.addmm(
        torch.randn(2, 3, **f32), torch.randn(2, 4, **f32), torch.randn(4, 3, **f32)
    ),
    "pdist": lambda: F.pdist(torch.randn(5, 3, **f32)),
    "glu_jvp": lambda: F.glu(torch.randn(4, 6, **f32), dim=1),
    "matrix_sqrth": lambda: torch.linalg.matrix_exp(torch.randn(4, 4, **f32)),
    "ormqr": lambda: torch.linalg.qr(torch.randn(4, 3, **f32)),
}

status = {}
for name, call in calls.items():
    try:
        call()
        status[name] = "runs"
    except NotImplementedError as exc:
        status[name] = "NotImplementedError: " + str(exc).splitlines()[0][:80]
    except Exception as exc:  # a probe bug or a different failure, kept visible
        status[name] = f"{type(exc).__name__}: " + str(exc).splitlines()[0][:80]

print(json.dumps({"torch": torch.__version__, "device": d, "status": status}, indent=1))
