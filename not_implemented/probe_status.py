"""Probe every portable candidate against the exact aten operator on MPS.

The candidate list in ``portable_candidates.txt`` is the deduplicated set of
operators that have no native MPS kernel.  Each candidate is invoked through
its real aten schema, not through a public helper that might route elsewhere
(``F.relu`` for ``_add_relu``, ``linalg.solve`` for ``linalg_ldl_solve``,
``linalg.qr`` for ``ormqr``, ...).  Every tensor argument lives on MPS; sparse
operators receive sparse MPS tensors, so a result of "runs" or an error always
describes the operator itself.

Two invocations are meant to be compared::

    python not_implemented/probe_status.py --accelerator disabled > not_implemented/mps_status.json
    python not_implemented/probe_status.py --accelerator enabled
    python not_implemented/probe_status.py --accelerator enabled --compatibility

``enabled`` imports this checkout's ``src`` first and installs the MPS
overlays in ``metal_kernelops.accelerator`` before probing.  ``disabled`` is
the baseline: plain torch, no overlay import.  ``--compatibility`` adds the
exact but slow fallback table.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "not_implemented" / "portable_candidates.txt"

# Candidate name -> exact aten overload.  Every other candidate is the default
# overload of the aten operator with the same name; these four are not, so the
# probe must name the overload explicitly.  At startup each resolution is
# checked for existence and for collisions, so a candidate can never silently
# stand for a different operator.
OVERLOAD = {
    "_add_relu": "Tensor",
    "_slow_conv2d_backward": "output_mask",
    "adaptive_avg_pool3d_backward": "grad_input",
    "frexp": "Tensor",
}


def _calls(device):
    """One schema-valid call per candidate.  All tensors are created on ``device``."""
    f32 = {"dtype": torch.float32, "device": device}
    i64 = {"dtype": torch.int64, "device": device}
    bool_ = {"dtype": torch.bool, "device": device}

    def r(*shape):
        return torch.randn(*shape, **f32)

    def u(*shape):
        return torch.rand(*shape, **f32)

    def ldl_input():
        a = r(4, 4)
        return a @ a.T + torch.eye(4, **f32)

    def glu_parts():
        x = r(4, 6)
        dx = r(4, 6)
        glu = x[:, :3] * torch.sigmoid(x[:, 3:])
        return glu, x, dx

    return {
        "_add_relu": lambda: torch.ops.aten._add_relu.Tensor(r(4), r(4)),
        "_addmm_activation": lambda: torch.ops.aten._addmm_activation(
            r(2, 3), r(2, 4), r(4, 3), use_gelu=False
        ),
        "_aminmax": lambda: torch.ops.aten._aminmax(r(4, 4)),
        "_fused_sdp_choice": lambda: torch.ops.aten._fused_sdp_choice(
            r(1, 2, 4, 8), r(1, 2, 4, 8), r(1, 2, 4, 8)
        ),
        "_masked_softmax": lambda: torch.ops.aten._masked_softmax(
            r(2, 4), torch.zeros(2, 4, **bool_), dim=1, mask_type=0
        ),
        "_masked_softmax_backward": lambda: torch.ops.aten._masked_softmax_backward(
            r(2, 4), torch.softmax(r(2, 4), dim=1), torch.zeros(2, 4, **bool_), dim=1
        ),
        "_native_multi_head_attention": lambda: torch.ops.aten._native_multi_head_attention(
            r(2, 3, 4), r(2, 3, 4), r(2, 3, 4),
            4, 2, r(12, 4), r(12), r(4, 4), r(4),
            mask=None, need_weights=True, average_attn_weights=True, mask_type=None,
        ),
        "_pdist_backward": lambda: torch.ops.aten._pdist_backward(
            torch.ones(6, **f32),
            torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [3.0, 4.0]], **f32),
            2.0,
            torch.tensor([3.0, 4.0, 5.0, 5.0, 4.0, 3.0], **f32),
        ),
        "_pdist_forward": lambda: torch.ops.aten._pdist_forward(r(4, 3), 2.0),
        "_segment_reduce_backward": lambda: torch.ops.aten._segment_reduce_backward(
            r(2, 3), torch.zeros(2, 3, **f32), r(6, 3), "sum",
            lengths=torch.tensor([2, 4], **i64),
        ),
        "_slow_conv2d_backward": lambda: torch.ops.aten._slow_conv2d_backward.output_mask(
            r(1, 3, 4, 4), r(1, 2, 6, 6), r(3, 2, 3, 3),
            [3, 3], [1, 1], [0, 0], [True, True, True],
        ),
        "_spdiags": lambda: torch.ops.aten._spdiags(
            r(3, 4), torch.tensor([0, 1, -2], **i64), [4, 4]
        ),
        "_transform_bias_rescale_qkv": lambda: torch.ops.aten._transform_bias_rescale_qkv(
            r(2, 3, 12), r(12), 2
        ),
        "_transformer_encoder_layer_fwd": lambda: torch.ops.aten._transformer_encoder_layer_fwd(
            r(2, 3, 4), 4, 2, r(12, 4), r(12), r(4, 4), r(4),
            False, False, 1e-5,
            r(4), r(4), r(4), r(4),
            r(8, 4), r(8), r(4, 8), r(4),
            mask=None, mask_type=None,
        ),
        "_upsample_lanczos2d_aa": lambda: torch.ops.aten._upsample_lanczos2d_aa(
            r(1, 1, 4, 4), [8, 8], False
        ),
        "_upsample_lanczos2d_aa_backward": lambda: torch.ops.aten._upsample_lanczos2d_aa_backward(
            r(1, 1, 8, 8), [8, 8], [1, 1, 4, 4], False
        ),
        "adaptive_avg_pool3d": lambda: torch.ops.aten.adaptive_avg_pool3d(
            r(1, 2, 4, 4, 4), [2, 2, 2]
        ),
        "adaptive_avg_pool3d_backward": lambda: torch.ops.aten.adaptive_avg_pool3d_backward.grad_input(
            r(1, 2, 2, 2, 2), r(1, 2, 4, 4, 4), grad_input=torch.empty(1, 2, 4, 4, 4, **f32)
        ),
        "adaptive_max_pool3d": lambda: torch.ops.aten.adaptive_max_pool3d(
            r(1, 2, 4, 4, 4), [2, 2, 2]
        ),
        "adaptive_max_pool3d_backward": lambda: torch.ops.aten.adaptive_max_pool3d_backward(
            r(1, 2, 2, 2, 2), r(1, 2, 4, 4, 4),
            torch.zeros(1, 2, 2, 2, 2, **i64),
        ),
        "batch_norm_update_stats": lambda: torch.ops.aten.batch_norm_update_stats(
            r(2, 3, 4, 4), torch.zeros(3, **f32), torch.ones(3, **f32), 0.1
        ),
        "channel_shuffle": lambda: torch.ops.aten.channel_shuffle(r(1, 4, 2, 2), 2),
        "fractional_max_pool2d": lambda: torch.ops.aten.fractional_max_pool2d(
            r(1, 2, 6, 6), [2, 2], [3, 3], u(1, 2, 2)
        ),
        "fractional_max_pool2d_backward": lambda: torch.ops.aten.fractional_max_pool2d_backward(
            r(1, 2, 3, 3), r(1, 2, 6, 6), [2, 2], [3, 3],
            torch.zeros(1, 2, 3, 3, **i64),
        ),
        "fractional_max_pool3d": lambda: torch.ops.aten.fractional_max_pool3d(
            r(1, 2, 6, 6, 6), [2, 2, 2], [3, 3, 3], u(1, 2, 3)
        ),
        "fractional_max_pool3d_backward": lambda: torch.ops.aten.fractional_max_pool3d_backward(
            r(1, 2, 3, 3, 3), r(1, 2, 6, 6, 6), [2, 2, 2], [3, 3, 3],
            torch.zeros(1, 2, 3, 3, 3, **i64),
        ),
        "frexp": lambda: torch.ops.aten.frexp.Tensor(r(4)),
        "glu_backward_jvp": lambda: torch.ops.aten.glu_backward_jvp(
            r(4, 6), r(4, 3), r(4, 6), r(4, 3), r(4, 6), 1
        ),
        "glu_jvp": lambda: torch.ops.aten.glu_jvp(*glu_parts(), 1),
        "heaviside": lambda: torch.ops.aten.heaviside(r(4), torch.full((4,), 0.5, **f32)),
        "linalg__powsum": lambda: torch.ops.aten.linalg__powsum(r(4, 4), 2, None, False),
        "linalg_eig": lambda: torch.ops.aten.linalg_eig(r(4, 4)),
        "linalg_eigvals": lambda: torch.ops.aten.linalg_eigvals(r(4, 4)),
        "linalg_ldl_factor_ex": lambda: torch.ops.aten.linalg_ldl_factor_ex(
            ldl_input(), hermitian=True, check_errors=True
        ),
        "linalg_ldl_solve": lambda: torch.ops.aten.linalg_ldl_solve(
            2 * torch.eye(4, **f32),
            torch.arange(1, 5, dtype=torch.int32, device=device),
            r(4, 2),
        ),
        "linalg_matrix_sqrth": lambda: torch.ops.aten.linalg_matrix_sqrth(ldl_input()),
        "logit": lambda: torch.ops.aten.logit(u(4) * 0.5 + 0.25),
        "logspace": lambda: torch.ops.aten.logspace(-2.0, 2.0, 5, 10.0, **f32),
        "mode": lambda: torch.ops.aten.mode(r(3, 4), 1),
        "multi_margin_loss": lambda: torch.ops.aten.multi_margin_loss(
            r(4, 3), torch.tensor([0, 1, 2, 0], **i64), 1, 1.0, None, 1
        ),
        "multi_margin_loss_backward": lambda: torch.ops.aten.multi_margin_loss_backward(
            torch.ones(1, **f32), r(4, 3), torch.tensor([0, 1, 2, 0], **i64),
            1, 1.0, None, 1,
        ),
        "multilabel_margin_loss_backward": lambda: torch.ops.aten.multilabel_margin_loss_backward(
            torch.ones(1, **f32),
            r(3, 4),
            torch.tensor([[0, 1, -1, -1], [1, 2, -1, -1], [0, 3, -1, -1]], **i64),
            1,
            torch.tensor(
                [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
                **f32,
            ),
        ),
        "multilabel_margin_loss_forward": lambda: torch.ops.aten.multilabel_margin_loss_forward(
            r(3, 4),
            torch.tensor([[0, 1, -1, -1], [1, 2, -1, -1], [0, 3, -1, -1]], **i64),
            1,
        ),
        "narrow_copy": lambda: torch.ops.aten.narrow_copy(r(3, 4), 1, 1, 2),
        "ormqr": lambda: torch.ops.aten.ormqr(
            torch.triu(r(4, 4)), torch.zeros(4, **f32), r(4, 2), True, False
        ),
        "put": lambda: torch.ops.aten.put(
            torch.zeros(6, **f32), torch.tensor([0, 2], **i64), torch.ones(2, **f32)
        ),
        "rrelu_with_noise": lambda: torch.ops.aten.rrelu_with_noise(
            r(4), torch.empty(4, **f32), 0.125, 0.3333, True
        ),
        "segment_reduce": lambda: torch.ops.aten.segment_reduce(
            r(6), "sum", lengths=torch.tensor([2, 4], **i64)
        ),
        "slow_conv3d_forward": lambda: torch.ops.aten.slow_conv3d_forward(
            r(1, 2, 4, 4, 4), r(3, 2, 3, 3, 3), [3, 3, 3], r(3), [1, 1, 1], [0, 0, 0]
        ),
        "slow_conv_dilated2d": lambda: torch.ops.aten.slow_conv_dilated2d(
            r(1, 2, 6, 6), r(3, 2, 3, 3), [3, 3], r(3), [1, 1], [1, 1], [2, 2]
        ),
        "slow_conv_dilated3d": lambda: torch.ops.aten.slow_conv_dilated3d(
            r(1, 1, 4, 4, 4), r(2, 1, 2, 2, 2), [2, 2, 2], r(2),
            [1, 1, 1], [0, 0, 0], [2, 2, 2],
        ),
        "slow_conv_transpose2d": lambda: torch.ops.aten.slow_conv_transpose2d(
            r(1, 2, 4, 4), r(2, 3, 3, 3), [3, 3], r(3), [2, 2], [0, 0], [0, 0], [1, 1]
        ),
        "slow_conv_transpose3d": lambda: torch.ops.aten.slow_conv_transpose3d(
            r(1, 1, 3, 3, 3), r(1, 2, 2, 2, 2), [2, 2, 2], r(2),
            [2, 2, 2], [0, 0, 0], [0, 0, 0], [1, 1, 1],
        ),
        "special_airy_ai": lambda: torch.ops.aten.special_airy_ai(
            torch.linspace(-2.0, 2.0, 4, **f32)
        ),
        "special_laguerre_polynomial_l": lambda: torch.ops.aten.special_laguerre_polynomial_l(
            u(4), torch.full((4,), 2, **i64)
        ),
        "special_legendre_polynomial_p": lambda: torch.ops.aten.special_legendre_polynomial_p(
            u(4) * 2 - 1, torch.full((4,), 2, **i64)
        ),
        "special_log_ndtr": lambda: torch.ops.aten.special_log_ndtr(r(4)),
        "special_ndtri": lambda: torch.ops.aten.special_ndtri(u(4) * 0.5 + 0.25),
        "sspaddmm": lambda: torch.ops.aten.sspaddmm.default(
            torch.sparse_coo_tensor(
                torch.tensor([[0, 2], [1, 0]], **i64),
                torch.ones(2, **f32),
                (3, 2),
                **f32,
            ),
            torch.sparse_coo_tensor(
                torch.tensor([[0, 2], [0, 3]], **i64),
                torch.ones(2, **f32),
                (3, 4),
                **f32,
            ),
            r(4, 2),
        ),
        "take": lambda: torch.ops.aten.take(
            torch.arange(12.0, **f32).reshape(3, 4), torch.tensor([0, 5], **i64)
        ),
    }


def _resolve(name):
    op = getattr(torch.ops.aten, name)
    overload = OVERLOAD.get(name, "default")
    if overload not in op.overloads():
        raise SystemExit(f"{name} has no {overload} overload: {op.overloads()}")
    return getattr(op, overload)


def _candidate_names():
    names = [line.strip() for line in CANDIDATES.read_text().splitlines() if line.strip()]
    if len(names) != len(set(names)):
        raise SystemExit("duplicate candidate in portable_candidates.txt")
    return names


def _classify(call):
    try:
        call()
    except NotImplementedError as exc:
        return "NotImplementedError: " + (str(exc).splitlines() or [""])[0][:120]
    except Exception as exc:
        return f"{type(exc).__name__}: " + (str(exc).splitlines() or [""])[0][:120]
    return "runs"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--accelerator",
        choices=("disabled", "enabled"),
        default="disabled",
        help="install this checkout's MPS overlays before probing (default: disabled)",
    )
    parser.add_argument(
        "--compatibility",
        action="store_true",
        help="with --accelerator enabled, also install exact but slow fallbacks",
    )
    args = parser.parse_args()

    coverage = None
    if args.accelerator == "enabled":
        sys.path.insert(0, str(ROOT / "src"))
        from metal_kernelops import accelerator

        coverage = accelerator.install(compatibility=args.compatibility)

    names = _candidate_names()
    calls = _calls("mps")
    missing = [name for name in names if name not in calls]
    extra = [name for name in calls if name not in names]
    if missing or extra:
        raise SystemExit(f"candidate/call mismatch: missing={missing} extra={extra}")

    resolved = {name: _resolve(name) for name in names}
    schemas = {name: resolved[name]._schema.name for name in names}
    if len(set(schemas.values())) != len(schemas):
        raise SystemExit("two candidates resolve to the same aten schema")

    status = {name: _classify(calls[name]) for name in names}
    payload = {
        "torch": torch.__version__,
        "device": "mps",
        "accelerator": args.accelerator,
        "compatibility": args.compatibility,
        "aten": schemas,
        "status": status,
    }
    if coverage is not None:
        payload["coverage"] = coverage
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
