"""Verify the MPS attention overlays against the CPU kernels, then measure them.

Run with the torch 2.14.1 environment:

    PYTHONPATH=src uv run --project <venv> python benchmarks/verify_attention_ops.py

Checks, all against the CPU kernel on the same values:

* ``_masked_softmax``: mask types 0, 1 and 2, softmax on and off the last axis,
  the compact ``(L, L)`` and ``(B, L)`` layouts, bool and float masks,
  all-masked rows (NaN), 0-d and empty inputs, and the CPU kernel's error paths;
* ``_masked_softmax_backward``: the CPU formula, dropped positions exactly
  zero, an all-masked row and an all-masked tensor, the mask each type expands
  to, strided and 3-D inputs, and autograd through the overlay;
* ``_transform_bias_rescale_qkv``: float32 and float16, split bias, output
  strides;
* ``_native_multi_head_attention``: self, cross and encoder-decoder attention,
  ``need_weights``, ``average_attn_weights``, every mask layout, empty batch;
* ``_transformer_encoder_layer_fwd``: all four ``norm_first``/``use_gelu``
  combinations, every mask type, float32, float16 and bfloat16, the head width
  the attention kernel has no form for, and an empty batch;
* ``_fused_sdp_choice``: the backend ``scaled_dot_product_attention`` reaches;
* end to end: ``nn.MultiheadAttention`` and ``nn.TransformerEncoderLayer`` on
  MPS against CPU, and the autograd rule the CPU op does have;
* importing the module compiles no shader: every cache is checked empty before
  the first operator call;
* timings: CPU and MPS, warmed and synchronized, plus one interleaved
  before/after batch per operator with a same-batch ``copy_`` floor, the launch
  count on both sides and the load average, written to
  ``measured/attention_ops.json``.

Process-wide installation is ``accelerator.install()``; this script registers
the same ``OVERLAYS`` table itself so the module can be checked alone.
"""

import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metal_kernelops import attention_ops  # noqa: E402

CHECKED_TORCH = "2.14.1"
TOL = 3e-5
HALF_TOL = 2e-3
BFLOAT_TOL = 2e-2


def install():
    library = torch.library.Library("aten", "FRAGMENT")
    for name, kernel in attention_ops.OVERLAYS:
        library.impl(name, kernel, "MPS")
    return library  # the caller must keep this alive: dropping it deregisters


def mha_weights(embed_dim, dtype=torch.float32):
    generator = torch.Generator().manual_seed(7)
    shapes = (3 * embed_dim, embed_dim), (3 * embed_dim,), (embed_dim, embed_dim), (embed_dim,)
    return [
        torch.randn(shape, generator=generator, dtype=dtype) for shape in shapes
    ]


def encoder_weights(embed_dim, dtype=torch.float32):
    generator = torch.Generator().manual_seed(11)
    shapes = (
        (3 * embed_dim, embed_dim),
        (3 * embed_dim,),
        (embed_dim, embed_dim),
        (embed_dim,),
        (4 * embed_dim, embed_dim),
        (4 * embed_dim,),
        (embed_dim, 4 * embed_dim),
        (embed_dim,),
    )
    return [torch.randn(shape, generator=generator, dtype=dtype) for shape in shapes]


def _diff(actual, expected):
    if not expected.numel():
        return 0.0
    scale = expected.abs().max().clamp(min=1e-12)
    return ((actual - expected).abs() / scale).max().item()


def _record(failures, report, name, passed, detail=""):
    report[name] = passed
    if not passed:
        failures.append(f"{name}{': ' + detail if detail else ''}")


def _matches(failures, report, name, actual, expected, tol=TOL):
    """Shape, NaN/inf pattern and values against the CPU result."""
    if actual.shape != expected.shape:
        _record(failures, report, name, False, f"{tuple(actual.shape)} vs {tuple(expected.shape)}")
        return
    same_special = torch.equal(torch.isnan(actual), torch.isnan(expected)) and torch.equal(
        torch.isinf(actual), torch.isinf(expected)
    )
    finite = torch.isfinite(expected)
    value = _diff(actual[finite], expected[finite]) if finite.any() else 0.0
    _record(failures, report, name, bool(same_special and value < tol), f"rel diff {value:.3e}")


def check_masked_softmax(failures, report):
    torch.manual_seed(0)
    for dtype, tol in ((torch.float32, TOL), (torch.float16, HALF_TOL)):
        scores = torch.randn(2, 3, 5, 5, dtype=dtype)
        rows = torch.zeros(5, 5, dtype=torch.bool)
        rows[0, 0] = True
        columns = torch.zeros(2, 5, dtype=torch.bool)
        columns[:, 0] = True
        exact = torch.zeros(2, 3, 5, 5, dtype=torch.bool)
        exact[:, :, :, 0] = True
        cases = {
            "type2_exact": (scores, exact, None, 2),
            "type0_layout": (scores, rows, -1, 0),
            "type1_padding": (scores, columns, -1, 1),
            "type0_default_dim": (scores, rows, None, 0),
            "type1_default_dim": (scores, columns, None, 1),
            "type2_default_dim": (scores, exact, None, None),
        }
        for label, (x, mask, dim, mask_type) in cases.items():
            cpu = torch.ops.aten._masked_softmax(x, mask, dim, mask_type)
            mps = torch.ops.aten._masked_softmax(x.to("mps"), mask.to("mps"), dim, mask_type)
            _matches(failures, report, f"masked_softmax_{label}_{dtype}", mps.cpu(), cpu, tol)

    # softmax along a non-last axis with the compact layouts
    scores = torch.randn(2, 3, 5, 5)
    rows = torch.zeros(5, 5, dtype=torch.bool)
    rows[0, 0] = True
    padding = torch.zeros(2, 5, dtype=torch.bool)
    padding[:, 0] = True
    exact = torch.zeros(2, 3, 5, 5, dtype=torch.bool)
    exact[:, :, :, 0] = True
    for mask_type, mask in ((0, rows), (1, padding), (2, exact)):
        cpu = torch.ops.aten._masked_softmax(scores, mask, 1, mask_type)
        mps = torch.ops.aten._masked_softmax(scores.to("mps"), mask.to("mps"), 1, mask_type)
        _matches(failures, report, f"masked_softmax_dim1_type{mask_type}", mps.cpu(), cpu)

    # float masks are not part of this kernel's contract
    try:
        torch.ops.aten._masked_softmax(scores, torch.zeros(2, 3, 5, 5))
        cpu_error = None
    except RuntimeError as error:
        cpu_error = str(error)
    try:
        torch.ops.aten._masked_softmax(
            scores.to("mps"), torch.zeros(2, 3, 5, 5, device="mps")
        )
        mps_error = None
    except RuntimeError as error:
        mps_error = str(error)
    _record(
        failures,
        report,
        "masked_softmax_error_float_mask",
        mps_error == cpu_error and cpu_error is not None,
        f"cpu {cpu_error!r} mps {mps_error!r}",
    )

    # strided input: the CPU kernel copies to contiguous memory first
    strided = torch.randn(2, 3, 5, 10)[..., ::2]
    strided_mask = torch.zeros(2, 3, 5, 5, dtype=torch.bool)
    strided_mask[..., 0] = True
    _matches(
        failures,
        report,
        "masked_softmax_strided_input",
        torch.ops.aten._masked_softmax(strided.to("mps"), strided_mask.to("mps")).cpu(),
        torch.ops.aten._masked_softmax(strided, strided_mask),
    )

    # every element masked: the CPU kernel divides by a NaN sum
    full = torch.ones(2, 3, 5, 4, dtype=torch.bool)
    cpu = torch.ops.aten._masked_softmax(scores[..., :4], full)
    mps = torch.ops.aten._masked_softmax(scores[..., :4].to("mps"), full.to("mps"))
    _record(
        failures,
        report,
        "masked_softmax_all_masked_nan",
        torch.isnan(cpu).all().item() and torch.isnan(mps.cpu()).all().item(),
    )

    # scalar and empty inputs
    scalar_cpu = torch.ops.aten._masked_softmax(torch.tensor(3.0), torch.tensor(True))
    scalar_mps = torch.ops.aten._masked_softmax(
        torch.tensor(3.0, device="mps"), torch.tensor(True, device="mps")
    )
    _record(
        failures,
        report,
        "masked_softmax_scalar",
        scalar_cpu.shape == scalar_mps.shape and torch.isnan(scalar_mps.cpu()).item(),
    )
    empty = torch.zeros(0, 3)
    empty_mask = torch.zeros(0, 3, dtype=torch.bool)
    _record(
        failures,
        report,
        "masked_softmax_empty",
        torch.ops.aten._masked_softmax(empty.to("mps"), empty_mask.to("mps")).shape
        == torch.ops.aten._masked_softmax(empty, empty_mask).shape,
    )

    # autograd through the overlay uses _masked_softmax_backward on MPS
    reference = torch.randn(2, 3, 6, 6, requires_grad=True)
    mask = torch.zeros(2, 3, 6, 6, dtype=torch.bool)
    mask[..., 0] = True
    grad = torch.randn(2, 3, 6, 6)
    expected = torch.ops.aten._masked_softmax(reference, mask)
    expected.backward(grad)
    live = reference.detach().to("mps").requires_grad_(True)
    actual = torch.ops.aten._masked_softmax(live, mask.to("mps"))
    actual.backward(grad.to("mps"))
    _matches(failures, report, "masked_softmax_autograd", actual.cpu().detach(), expected.detach())
    _matches(failures, report, "masked_softmax_grad", live.grad.cpu(), reference.grad)
    _record(
        failures,
        report,
        "masked_softmax_metal_path",
        attention_ops._ms_library.cache_info().currsize > 0,
        str(attention_ops._ms_library.cache_info()),
    )
 


def check_masked_softmax_backward(failures, report):
    torch.manual_seed(1)
    for dtype, tol in ((torch.float32, TOL), (torch.float16, HALF_TOL)):
        grad = torch.randn(2, 3, 6, 6, dtype=dtype)
        scores = torch.randn(2, 3, 6, 6, dtype=dtype)
        mask = torch.zeros(2, 3, 6, 6, dtype=torch.bool)
        mask[..., 0] = True
        mask[0, 0, 1, 1] = True
        output = torch.ops.aten._masked_softmax(scores, mask)
        cpu = torch.ops.aten._masked_softmax_backward(grad, output, mask)
        mps = torch.ops.aten._masked_softmax_backward(
            grad.to("mps"), output.to("mps"), mask.to("mps")
        )
        _matches(failures, report, f"masked_softmax_backward_{dtype}", mps.cpu(), cpu, tol)
        _record(
            failures,
            report,
            f"masked_softmax_backward_zeroed_{dtype}",
            bool((mps.cpu()[mask] == 0).all()),
        )
        shifted = torch.ops.aten._masked_softmax_backward(grad, output, mask, 1)
        shifted_mps = torch.ops.aten._masked_softmax_backward(
            grad.to("mps"), output.to("mps"), mask.to("mps"), 1
        )
        _matches(
            failures, report, f"masked_softmax_backward_dim1_{dtype}", shifted_mps.cpu(), shifted, tol
        )

    # a row the mask blocks completely: the reduction is empty, so the gradient
    # there is exactly zero on both backends, not NaN
    grad = torch.randn(2, 3, 4, 5)
    scores = torch.randn(2, 3, 4, 5)
    mask = torch.zeros(2, 3, 4, 5, dtype=torch.bool)
    mask[0, 0, 0] = True
    output = torch.ops.aten._masked_softmax(scores, mask, -1, 2)
    cpu = torch.ops.aten._masked_softmax_backward(grad, output, mask, -1)
    mps = torch.ops.aten._masked_softmax_backward(
        grad.to("mps"), output.to("mps"), mask.to("mps"), -1
    )
    _matches(failures, report, "masked_softmax_backward_blocked_row", mps.cpu(), cpu)
    _record(
        failures,
        report,
        "masked_softmax_backward_blocked_row_zero",
        bool((mps.cpu()[0, 0, 0] == 0).all()),
        f"{mps.cpu()[0, 0, 0].tolist()}",
    )

    # three dimensions, and a mask that is not contiguous: both leave the
    # kernel for the composite, which has to produce the same values
    grad = torch.randn(3, 4, 6)
    scores = torch.randn(3, 4, 6)
    mask = torch.zeros(3, 4, 6, dtype=torch.bool)
    mask[..., -1] = True
    output = torch.ops.aten._masked_softmax(scores, mask, -1, 2)
    cpu = torch.ops.aten._masked_softmax_backward(grad, output, mask, -1)
    mps = torch.ops.aten._masked_softmax_backward(
        grad.to("mps"), output.to("mps"), mask.to("mps"), -1
    )
    _matches(failures, report, "masked_softmax_backward_3d", mps.cpu(), cpu)
    wide = torch.zeros(3, 4, 12, dtype=torch.bool)
    wide[..., -1] = True
    strided_mask = wide[..., ::2]
    output = torch.ops.aten._masked_softmax(scores, strided_mask, -1, 2)
    cpu = torch.ops.aten._masked_softmax_backward(grad, output, strided_mask, -1)
    mps = torch.ops.aten._masked_softmax_backward(
        grad.to("mps"), output.to("mps"), strided_mask.to("mps"), -1
    )
    _matches(failures, report, "masked_softmax_backward_strided_mask", mps.cpu(), cpu)

    # backward after a forward that took each mask type: the gradient arrives
    # with the mask expanded to the score shape, which is the only layout the
    # aten backward accepts, and each type has to reduce over its own rows
    scores = torch.randn(2, 3, 5, 5)
    grad = torch.randn(2, 3, 5, 5)
    rows = torch.zeros(5, 5, dtype=torch.bool)
    rows[0, 0] = True
    rows[1, :] = True
    padding = torch.zeros(2, 5, dtype=torch.bool)
    padding[:, 0] = True
    exact = torch.zeros(2, 3, 5, 5, dtype=torch.bool)
    exact[:, :, :, 0] = True
    for mask_type, compact in ((0, rows), (1, padding), (2, exact)):
        output = torch.ops.aten._masked_softmax(scores, compact, -1, mask_type)
        expanded = attention_ops._mask_for_scores(
            scores.shape, compact, mask_type, -1
        ).expand(scores.shape).contiguous()
        cpu = torch.ops.aten._masked_softmax_backward(grad, output, expanded, -1)
        mps = torch.ops.aten._masked_softmax_backward(
            grad.to("mps"), output.to("mps"), expanded.to("mps"), -1
        )
        _matches(failures, report, f"masked_softmax_backward_after_type{mask_type}", mps.cpu(), cpu)

    # every element masked: the forward is all NaN and the backward is all zero,
    # because the reduction that would have produced the row sum is empty
    full = torch.ones(2, 3, 4, 6, dtype=torch.bool)
    scores = torch.randn(2, 3, 4, 6)
    output = torch.ops.aten._masked_softmax(scores, full, -1, 2)
    grad = torch.randn(2, 3, 4, 6)
    mps = torch.ops.aten._masked_softmax_backward(
        grad.to("mps"), output.to("mps"), full.to("mps"), -1
    )
    _record(
        failures,
        report,
        "masked_softmax_backward_all_masked",
        torch.isnan(output).all().item() and bool((mps.cpu() == 0).all()),
        f"nan={torch.isnan(output).all().item()} zero={bool((mps.cpu() == 0).all())}",
    )


def check_transform_bias_rescale_qkv(failures, report):
    torch.manual_seed(2)
    for dtype, tol in ((torch.float32, TOL), (torch.float16, HALF_TOL)):
        for batch, length, embed, heads in ((2, 4, 12, 3), (3, 7, 64, 8)):
            qkv = torch.randn(batch, length, 3 * embed, dtype=dtype)
            bias = torch.randn(3 * embed, dtype=dtype)
            cpu = torch.ops.aten._transform_bias_rescale_qkv(qkv, bias, heads)
            mps = torch.ops.aten._transform_bias_rescale_qkv(
                qkv.to("mps"), bias.to("mps"), heads
            )
            name = f"transform_bias_rescale_qkv_{embed}h{heads}_{dtype}"
            for index, axis in enumerate("qkv"):
                _matches(failures, report, f"{name}_{axis}", mps[index].cpu(), cpu[index], tol)
            _record(
                failures,
                report,
                f"{name}_layout",
                tuple(mps[0].shape) == tuple(cpu[0].shape) and mps[0].is_contiguous(),
                f"{tuple(mps[0].shape)} contiguous={mps[0].is_contiguous()}",
            )
    # the reciprocal scaling, not a division, is what the CPU kernel multiplies by
    qkv = torch.randn(1, 2, 144)
    bias = torch.randn(144)
    q, _, _ = torch.ops.aten._transform_bias_rescale_qkv(qkv.to("mps"), bias.to("mps"), 3)
    expected = (
        (qkv.reshape(1, 2, 3, 3, 16).permute(2, 0, 3, 1, 4)[0] + bias.reshape(3, 3, 1, 16)[0])
        * (1.0 / 4.0)
    )
    _record(failures, report, "transform_bias_rescale_qkv_exact_scaling", torch.equal(q.cpu(), expected))

    # half precision: the bias add and the rescale accumulate in float32 and
    # round once, which is the CPU kernel's result bit for bit
    half_qkv = torch.randn(2, 5, 72, dtype=torch.float16)
    half_bias = torch.randn(72, dtype=torch.float16)
    half_cpu = torch.ops.aten._transform_bias_rescale_qkv(half_qkv, half_bias, 3)
    half_mps = torch.ops.aten._transform_bias_rescale_qkv(
        half_qkv.to("mps"), half_bias.to("mps"), 3
    )
    _record(
        failures,
        report,
        "transform_bias_rescale_qkv_half_exact",
        all(torch.equal(a.cpu(), b) for a, b in zip(half_mps, half_cpu)),
    )

    # all three outputs carry the (batch, heads, length, head_dim) layout the
    # pipeline indexes, not just q
    for dtype in (torch.float32, torch.float16):
        qkv = torch.randn(2, 5, 3 * 40, dtype=dtype)
        bias = torch.randn(3 * 40, dtype=dtype)
        cpu = torch.ops.aten._transform_bias_rescale_qkv(qkv, bias, 8)
        mps = torch.ops.aten._transform_bias_rescale_qkv(qkv.to("mps"), bias.to("mps"), 8)
        _record(
            failures,
            report,
            f"transform_bias_rescale_qkv_layouts_{dtype}",
            all(
                a.dtype == dtype and a.is_contiguous() and a.stride() == b.stride()
                for a, b in zip(mps, cpu)
            ),
            f"{[tuple(a.shape) for a in mps]} contiguous={[a.is_contiguous() for a in mps]}",
        )

    # every head width the packed layout can produce: the four-wide path when
    # the width divides by four, the scalar path otherwise, and the
    # decomposition for the widths neither takes. Half is bit-exact on all of
    # them; float32 lands within one ulp of the CPU at head widths 6, 7, 17, 18,
    # 21 and 24, where the CPU result follows neither the reciprocal product nor
    # the quotient.
    for dtype, tol in ((torch.float32, TOL), (torch.float16, HALF_TOL)):
        worst, exact, total, inexact = 0.0, 0, 0, []
        for embed in list(range(1, 65)) + [96, 128]:
            for heads in (1, 2, 4, 8):
                if embed % heads:
                    continue
                qkv = torch.randn(2, 3, 3 * embed, dtype=dtype)
                bias = torch.randn(3 * embed, dtype=dtype)
                cpu = torch.ops.aten._transform_bias_rescale_qkv(qkv, bias, heads)
                mps = torch.ops.aten._transform_bias_rescale_qkv(
                    qkv.to("mps"), bias.to("mps"), heads
                )
                for a, b in zip(mps, cpu):
                    a = a.cpu()
                    total += 1
                    same = torch.equal(a, b)
                    exact += int(same)
                    worst = max(worst, _diff(a, b))
                    if not same and embed // heads not in inexact:
                        inexact.append(embed // heads)
        _record(
            failures,
            report,
            f"transform_bias_rescale_qkv_head_widths_{dtype}",
            worst < tol,
            f"worst rel diff {worst:.3e}, bit-exact {exact}/{total}, "
            f"one-ulp head widths {sorted(inexact)}",
        )

    # a strided input falls back to the decomposition and still matches
    wide = torch.randn(4, 8, 3 * 32)
    strided = wide[:, ::2]
    bias = torch.randn(3 * 32)
    cpu = torch.ops.aten._transform_bias_rescale_qkv(strided, bias, 4)
    mps = torch.ops.aten._transform_bias_rescale_qkv(strided.to("mps"), bias.to("mps"), 4)
    _record(
        failures,
        report,
        "transform_bias_rescale_qkv_strided",
        all(torch.equal(a.cpu(), b) for a, b in zip(mps, cpu))
        and all(a.is_contiguous() for a in mps),
        f"contiguous={[a.is_contiguous() for a in mps]}",
    )

    # a bias of another dtype is cast to the input dtype before the add. The CPU
    # kernel reads the bias through the input's pointer without converting, so
    # its mixed-dtype result is not a contract; this compares the cast call with
    # the same call given an already-cast bias.
    qkv16 = torch.randn(2, 3, 48, dtype=torch.float16)
    bias32 = torch.randn(48)
    cast = torch.ops.aten._transform_bias_rescale_qkv(qkv16.to("mps"), bias32.to("mps"), 4)
    recast = torch.ops.aten._transform_bias_rescale_qkv(
        qkv16.to("mps"), bias32.to(torch.float16).to("mps"), 4
    )
    _record(
        failures,
        report,
        "transform_bias_rescale_qkv_bias_cast",
        all(torch.equal(a, b) for a, b in zip(cast, recast))
        and all(a.dtype == torch.float16 for a in cast),
        f"{[a.dtype for a in cast]}",
    )

    # a bias that is a contiguous slice of a larger tensor keeps its values
    qkv = torch.randn(2, 3, 48)
    sliced = torch.randn(3 * 16 + 4)[: 3 * 16]
    cpu = torch.ops.aten._transform_bias_rescale_qkv(qkv, sliced, 4)
    mps = torch.ops.aten._transform_bias_rescale_qkv(qkv.to("mps"), sliced.to("mps"), 4)
    _record(
        failures,
        report,
        "transform_bias_rescale_qkv_bias_slice",
        all(torch.equal(a.cpu(), b) for a, b in zip(mps, cpu)),
    )

    # bfloat16 is not a shader pointer: the decomposition takes it
    bf16_qkv = torch.randn(2, 3, 48, dtype=torch.bfloat16)
    bf16_bias = torch.randn(48, dtype=torch.bfloat16)
    bf16_cpu = torch.ops.aten._transform_bias_rescale_qkv(bf16_qkv, bf16_bias, 4)
    bf16_mps = torch.ops.aten._transform_bias_rescale_qkv(
        bf16_qkv.to("mps"), bf16_bias.to("mps"), 4
    )
    for index, axis in enumerate("qkv"):
        _matches(
            failures,
            report,
            f"transform_bias_rescale_qkv_bf16_{axis}",
            bf16_mps[index].cpu(),
            bf16_cpu[index],
            HALF_TOL,
        )

    # an empty batch or an empty sequence is a legal shape and returns empty
    # outputs of the same layout
    for shape in ((0, 5, 24), (2, 0, 24)):
        empty = torch.ops.aten._transform_bias_rescale_qkv(
            torch.zeros(shape, device="mps"), torch.zeros(24, device="mps"), 4
        )
        _record(
            failures,
            report,
            f"transform_bias_rescale_qkv_empty_{shape[0]}x{shape[1]}",
            all(a.is_contiguous() and a.numel() == 0 for a in empty)
            and tuple(empty[0].shape) == (shape[0], 4, shape[1], shape[2] // 3 // 4),
            f"{[tuple(a.shape) for a in empty]}",
        )

    # the two error contracts, with the message the overlay documents
    for label, embed, heads, message in (
        ("heads", 12, 5, "embedding dimension 4 is not divisible by 5 heads"),
        ("triple", 10, 1, "expected 3 * num_embedding to be divisible by 3, got 10"),
    ):
        try:
            torch.ops.aten._transform_bias_rescale_qkv(
                torch.zeros(2, 3, embed, device="mps"),
                torch.zeros(embed, device="mps"),
                heads,
            )
        except RuntimeError as error:
            _record(
                failures,
                report,
                f"transform_bias_rescale_qkv_error_{label}",
                str(error) == message,
                f"{str(error)!r} vs {message!r}",
            )
        else:
            _record(failures, report, f"transform_bias_rescale_qkv_error_{label}", False, "no error")

    # the shader path, not the decomposition, is what ran
    _record(
        failures,
        report,
        "transform_bias_rescale_qkv_metal_path",
        attention_ops._library.cache_info().hits > 0,
        str(attention_ops._library.cache_info()),
    )


def _legacy_split(qkv, qkv_bias, num_heads):
    """The four-pass decomposition the overlay used before the fused kernel.

    Kept for the paired measurement: the two have to run in one process, against
    the same device state, for the comparison to mean anything.
    """
    batch, length, triple = qkv.shape
    embed = triple // 3
    head_dim = embed // num_heads
    bias = qkv_bias.to(qkv.dtype).reshape(3, num_heads, 1, head_dim)
    packed = qkv.reshape(batch, length, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    scale = head_dim**-0.5
    wide = torch.float32 if qkv.dtype in (torch.float16, torch.bfloat16) else qkv.dtype
    q = torch.add(packed[0].to(wide), bias[0].to(wide)).contiguous()
    q.mul_(scale)
    return (
        q.to(qkv.dtype),
        torch.add(packed[1], bias[1]).contiguous(),
        torch.add(packed[2], bias[2]).contiguous(),
    )


def _paired_sync(calls, rounds=25, reps=20):
    """Minimum per candidate with ``synchronize()`` after every call.

    This is the protocol the ops table uses. Every candidate is timed inside the
    same round, so a neighbour's burst lands on all of them and the minimum of
    each is the sample least disturbed by it. A round's own minima also give a
    ratio; the median of those is reported next to the ratio of the overall
    minima, because one burst across a whole window can leave a minimum that
    says nothing about either candidate.
    """
    best = {name: float("inf") for name in calls}
    ratios = []
    for _ in range(rounds):
        for fn in calls.values():
            for _ in range(4):
                fn()
        torch.mps.synchronize()
        sample = {name: float("inf") for name in calls}
        for _ in range(reps):
            for name, fn in calls.items():
                start = time.perf_counter()
                fn()
                torch.mps.synchronize()
                elapsed = time.perf_counter() - start
                sample[name] = min(sample[name], elapsed)
                best[name] = min(best[name], elapsed)
        ratios.append(sample["legacy"] / sample["fused"])
    ratios.sort()
    return (
        {name: value * 1e3 for name, value in best.items()},
        ratios[len(ratios) // 2],
    )


def _paired_queued(calls, count=300, rounds=8):
    """Minimum per candidate with the queue kept full: the kernel's own cost."""
    best = {name: float("inf") for name in calls}
    for _ in range(rounds):
        for name, fn in calls.items():
            for _ in range(20):
                fn()
            torch.mps.synchronize()
            start = time.perf_counter()
            for _ in range(count):
                fn()
            torch.mps.synchronize()
            best[name] = min(best[name], (time.perf_counter() - start) / count)
    return {name: value * 1e3 for name, value in best.items()}


def measure_transform_bias_rescale_qkv():
    """The fused split against the decomposition it replaces, and against a copy.

    A plain ``copy_`` of the same byte count is the floor one launch of this
    shape reaches on the device: the operator reads one 3 * embed slab per
    position and writes another, which is the copy's traffic exactly. Both
    protocols are recorded because they answer different questions -- the
    synchronize-per-call minimum is what the ops table reports, and the queued
    minimum is the kernel's own cost with the device round trip taken out.
    """
    torch.manual_seed(7)
    batch, length, embed, heads = 4, 128, 128, 8
    rows = []
    for dtype in (torch.float32, torch.float16):
        qkv = torch.randn(batch, length, 3 * embed, dtype=dtype, device="mps")
        bias = torch.randn(3 * embed, dtype=dtype, device="mps")
        source = torch.randn(batch * length * 3 * embed, dtype=dtype, device="mps")
        destination = torch.empty_like(source)
        moved = 2 * qkv.numel() * qkv.element_size()
        calls = {
            "legacy": lambda: _legacy_split(qkv, bias, heads),
            "fused": lambda: torch.ops.aten._transform_bias_rescale_qkv(qkv, bias, heads),
            "copy": lambda: destination.copy_(source),
        }
        sync, sync_median = _paired_sync(calls)
        queued = _paired_queued(calls)
        rows.append(
            {
                "dtype": str(dtype).removeprefix("torch."),
                "batch": batch,
                "length": length,
                "embed": embed,
                "heads": heads,
                "head_dim": embed // heads,
                "bytes_moved": moved,
                "legacy_sync_ms": round(sync["legacy"], 4),
                "fused_sync_ms": round(sync["fused"], 4),
                "copy_sync_ms": round(sync["copy"], 4),
                "legacy_queued_ms": round(queued["legacy"], 4),
                "fused_queued_ms": round(queued["fused"], 4),
                "copy_queued_ms": round(queued["copy"], 4),
                "legacy_gbs": round(moved / (queued["legacy"] * 1e-3) / 1e9, 1),
                "fused_gbs": round(moved / (queued["fused"] * 1e-3) / 1e9, 1),
                "copy_gbs": round(moved / (queued["copy"] * 1e-3) / 1e9, 1),
                "speedup_sync": round(sync["legacy"] / sync["fused"], 2),
                "speedup_sync_median": round(sync_median, 2),
                "speedup_queued": round(queued["legacy"] / queued["fused"], 2),
            }
        )
    return {
        "protocol": (
            "legacy, fused and a same-size copy_ timed inside the same round, "
            "minimum per candidate; sync = torch.mps.synchronize() after every "
            "call (25 rounds of 20 calls, ratio of the minima and median of the "
            "per-round ratios), queued = 300 calls per candidate with one "
            "synchronize at the end"
        ),
        "box_load": (
            "the device is shared with other jobs, so one synchronize-per-call "
            "window anywhere in the file moves by several times between sweeps: "
            "the copy floor below has been seen anywhere from 0.02 to 0.55 ms. "
            "The ratios come from the interleaved loop, not from two separate "
            "runs, and the queued numbers are the kernel's own cost with the "
            "device round trip taken out"
        ),
        "launches": {"legacy": 4, "fused": 1, "copy": 1},
        "rows": rows,
        "load_average": round(os.getloadavg()[0], 2),
    }


def check_native_multi_head_attention(failures, report):
    torch.manual_seed(3)
    batch, length, embed, heads = 2, 5, 24, 3
    rows = torch.zeros(length, length, dtype=torch.bool)
    rows[0, 0] = True
    padding = torch.zeros(batch, length, dtype=torch.bool)
    padding[:, 0] = True
    exact = torch.zeros(batch, heads, length, length, dtype=torch.bool)
    exact[:, :, :, 0] = True
    for dtype, tol in ((torch.float32, TOL), (torch.float16, HALF_TOL)):
        qkv_weight, qkv_bias, proj_weight, proj_bias = mha_weights(embed, dtype=dtype)
        query = torch.randn(batch, length, embed, dtype=dtype)
        other = torch.randn(batch, length, embed, dtype=dtype)
        cases = {
            "weights": (query, query, query, None, True, True, None),
            "weights_unaveraged": (query, query, query, None, True, False, None),
            "no_weights": (query, query, query, None, False, True, None),
            "mask_type0": (query, query, query, rows, True, True, 0),
            "mask_type0_no_weights": (query, query, query, rows, False, True, 0),
            "mask_type1": (query, query, query, padding, True, True, 1),
            "mask_type1_no_weights": (query, query, query, padding, False, True, 1),
            "mask_type2": (query, query, query, exact, True, True, 2),
            "cross": (query, other, other, None, True, True, None),
            "encoder_decoder": (query, other, other, rows, False, True, 0),
        }
        for label, (q, k, v, mask, need_weights, average, mask_type) in cases.items():
            plain = torch.ops.aten._native_multi_head_attention(
                q, k, v, embed, heads, qkv_weight, qkv_bias, proj_weight, proj_bias,
                mask, need_weights, average, mask_type,
            )
            on_mps = torch.ops.aten._native_multi_head_attention(
                q.to("mps"), k.to("mps"), v.to("mps"), embed, heads,
                qkv_weight.to("mps"), qkv_bias.to("mps"), proj_weight.to("mps"),
                proj_bias.to("mps"),
                None if mask is None else mask.to("mps"), need_weights, average, mask_type,
            )
            name = f"nmha_{label}_{str(dtype).removeprefix('torch.')}"
            _matches(failures, report, f"{name}_out", on_mps[0].cpu(), plain[0], tol)
            if need_weights:
                _matches(failures, report, f"{name}_weights", on_mps[1].cpu(), plain[1], tol)
            else:
                _record(failures, report, f"{name}_no_weights_none", on_mps[1] is None)

    qkv_weight, qkv_bias, proj_weight, proj_bias = mha_weights(embed)
    query = torch.randn(batch, length, embed)

    # a query row the mask blocks completely stays NaN: the CPU kernel divides
    # by an empty sum, and the fused-kernel path has to put that NaN back
    blocked = torch.zeros(length, length, dtype=torch.bool)
    blocked[0, :] = True
    plain = torch.ops.aten._native_multi_head_attention(
        query, query, query, embed, heads, qkv_weight, qkv_bias, proj_weight, proj_bias,
        blocked, False, True, 0,
    )[0]
    on_mps = torch.ops.aten._native_multi_head_attention(
        query.to("mps"), query.to("mps"), query.to("mps"), embed, heads,
        qkv_weight.to("mps"), qkv_bias.to("mps"), proj_weight.to("mps"), proj_bias.to("mps"),
        blocked.to("mps"), False, True, 0,
    )[0]
    _record(
        failures,
        report,
        "nmha_blocked_row_nan",
        torch.equal(torch.isnan(on_mps.cpu()), torch.isnan(plain))
        and bool(torch.isnan(plain)[:, 0].all()),
    )

    empty = torch.zeros(0, 4, embed)
    empty_mps = torch.ops.aten._native_multi_head_attention(
        empty.to("mps"), empty.to("mps"), empty.to("mps"), embed, heads,
        qkv_weight.to("mps"), qkv_bias.to("mps"), proj_weight.to("mps"), proj_bias.to("mps"),
        None, True, True, None,
    )
    _record(
        failures,
        report,
        "nmha_empty",
        tuple(empty_mps[0].shape) == tuple(empty.shape) and empty_mps[1] is None,
    )

    # a mask without a mask type is refused, as on CPU
    try:
        torch.ops.aten._native_multi_head_attention(
            query, query, query, embed, heads, qkv_weight, qkv_bias, proj_weight, proj_bias,
            rows, True, True, None,
        )
        cpu_error = None
    except RuntimeError as error:
        cpu_error = str(error)
    try:
        torch.ops.aten._native_multi_head_attention(
            query.to("mps"), query.to("mps"), query.to("mps"), embed, heads,
            qkv_weight.to("mps"), qkv_bias.to("mps"), proj_weight.to("mps"), proj_bias.to("mps"),
            rows.to("mps"), True, True, None,
        )
        mps_error = None
    except RuntimeError as error:
        mps_error = str(error)
    _record(
        failures,
        report,
        "nmha_error_mask_without_type",
        mps_error == cpu_error and "Mask Type" in (cpu_error or ""),
        f"cpu {cpu_error!r} mps {mps_error!r}",
    )

    # the op has no autograd rule on any backend; the overlay must not invent one
    live = query.clone().to("mps").requires_grad_(True)
    out = torch.ops.aten._native_multi_head_attention(
        live, live, live, embed, heads,
        qkv_weight.to("mps"), qkv_bias.to("mps"), proj_weight.to("mps"), proj_bias.to("mps"),
        None, False, True, None,
    )[0]
    try:
        out.sum().backward()
        raised = None
    except RuntimeError as error:
        raised = str(error)
    _record(
        failures,
        report,
        "nmha_no_autograd_rule",
        raised is not None and "derivative for aten::_native_multi_head_attention" in raised,
        f"{raised!r}",
    )


def check_transformer_encoder_layer(failures, report):
    torch.manual_seed(4)
    batch, length, embed, heads = 2, 5, 24, 3
    rows = torch.zeros(length, length, dtype=torch.bool)
    rows[0, 0] = True
    padding = torch.zeros(batch, length, dtype=torch.bool)
    padding[:, 0] = True
    exact = torch.zeros(batch, heads, length, length, dtype=torch.bool)
    exact[:, :, :, 0] = True
    masks = {
        "plain": (None, None),
        "type0": (rows, 0),
        "type1": (padding, 1),
        "type2": (exact, 2),
    }
    # bfloat16 is the one dtype the fused pass does not take -- it has no
    # bfloat16 form -- so that row checks the composed fallback, which is the
    # arithmetic the fused pass reproduces.
    for dtype, tol in ((torch.float32, TOL), (torch.float16, HALF_TOL),
                       (torch.bfloat16, BFLOAT_TOL)):
        qkv_weight, qkv_bias, proj_weight, proj_bias, f1, b1, f2, b2 = encoder_weights(
            embed, dtype=dtype
        )
        src = torch.randn(batch, length, embed, dtype=dtype)
        norm_weight = torch.ones(embed, dtype=dtype)
        norm_bias = torch.zeros(embed, dtype=dtype)
        for norm_first in (False, True):
            for use_gelu in (False, True):
                for mask_label, (mask, mask_type) in masks.items():
                    label = (
                        f"norm_first{int(norm_first)}_gelu{int(use_gelu)}"
                        f"_{mask_label}"
                        f"_{str(dtype).removeprefix('torch.')}"
                    )
                    plain = torch.ops.aten._transformer_encoder_layer_fwd(
                        src, embed, heads, qkv_weight, qkv_bias, proj_weight, proj_bias,
                        use_gelu, norm_first, 1e-5, norm_weight, norm_bias,
                        norm_weight, norm_bias, f1, b1, f2, b2, mask,
                        mask_type,
                    )
                    on_mps = torch.ops.aten._transformer_encoder_layer_fwd(
                        src.to("mps"), embed, heads,
                        qkv_weight.to("mps"), qkv_bias.to("mps"),
                        proj_weight.to("mps"), proj_bias.to("mps"),
                        use_gelu, norm_first, 1e-5,
                        norm_weight.to("mps"), norm_bias.to("mps"),
                        norm_weight.to("mps"), norm_bias.to("mps"),
                        f1.to("mps"), b1.to("mps"), f2.to("mps"), b2.to("mps"),
                        None if mask is None else mask.to("mps"), mask_type,
                    )
                    _matches(failures, report, f"encoder_layer_{label}", on_mps.cpu(), plain, tol)

    # the two shader stages, not the composed fallback, are what the last-axis
    # rows above ran
    _record(
        failures,
        report,
        "encoder_layer_metal_path",
        attention_ops._encoder_library.cache_info().currsize > 0,
        str(attention_ops._encoder_library.cache_info()),
    )

    # a head width the attention kernel has no form for takes the composed path
    # and still matches the CPU kernel
    ew12 = encoder_weights(12)
    src12 = torch.randn(2, 4, 12)
    ones12, zeros12 = torch.ones(12), torch.zeros(12)
    plain = torch.ops.aten._transformer_encoder_layer_fwd(
        src12, 12, 1, *ew12[:4], True, False, 1e-5,
        ones12, zeros12, ones12, zeros12, *ew12[4:],
    )
    on_mps = torch.ops.aten._transformer_encoder_layer_fwd(
        src12.to("mps"), 12, 1, *[t.to("mps") for t in ew12[:4]], True, False, 1e-5,
        ones12.to("mps"), zeros12.to("mps"), ones12.to("mps"), zeros12.to("mps"),
        *[t.to("mps") for t in ew12[4:]],
    )
    _matches(failures, report, "encoder_layer_head_width_fallback", on_mps.cpu(), plain)

    # an empty batch is a legal shape: the overlay returns it unchanged
    ew24 = encoder_weights(24)
    nw24, nb24 = torch.ones(24), torch.zeros(24)
    empty24 = torch.zeros(0, 4, 24)
    empty_mps = torch.ops.aten._transformer_encoder_layer_fwd(
        empty24.to("mps"), 24, 3,
        *[t.to("mps") for t in ew24[:4]], True, False, 1e-5,
        nw24.to("mps"), nb24.to("mps"), nw24.to("mps"), nb24.to("mps"),
        *[t.to("mps") for t in ew24[4:]],
    )
    _record(
        failures,
        report,
        "encoder_layer_empty",
        tuple(empty_mps.shape) == tuple(empty24.shape) and empty_mps.device.type == "mps",
        f"{tuple(empty_mps.shape)}",
    )

    # the fast path runs under inference_mode in real models
    src24 = torch.randn(batch, length, 24)
    ones24, zeros24 = torch.ones(24), torch.zeros(24)
    with torch.inference_mode():
        eager = torch.ops.aten._transformer_encoder_layer_fwd(
            src24, 24, 3, *ew24[:4], True, False, 1e-5,
            ones24, zeros24, ones24, zeros24, *ew24[4:],
        )
        fast = torch.ops.aten._transformer_encoder_layer_fwd(
            src24.to("mps"), 24, 3, *[t.to("mps") for t in ew24[:4]], True, False, 1e-5,
            ones24.to("mps"), zeros24.to("mps"), ones24.to("mps"), zeros24.to("mps"),
            *[t.to("mps") for t in ew24[4:]],
        )
    _matches(failures, report, "encoder_layer_inference_mode", fast.cpu(), eager, HALF_TOL)


def check_fused_sdp_choice(failures, report):
    query = torch.randn(2, 3, 4, 8, device="mps")
    choice = torch.ops.aten._fused_sdp_choice(query, query, query)
    plain = F.scaled_dot_product_attention(query, query, query)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        forced = F.scaled_dot_product_attention(query, query, query)
    _record(
        failures,
        report,
        "fused_sdp_choice_math",
        choice == int(torch.nn.attention.SDPBackend.MATH),
        f"choice {choice}",
    )
    _matches(failures, report, "fused_sdp_choice_matches_sdpa", plain.cpu(), forced.cpu())


def check_models(failures, report):
    torch.manual_seed(5)
    embed, heads, batch, length = 96, 4, 2, 6
    attention = nn.MultiheadAttention(embed, heads, batch_first=True).eval()
    layer = nn.TransformerEncoderLayer(
        embed, heads, dim_feedforward=4 * embed, batch_first=True, dropout=0.0
    ).eval()
    src = torch.randn(batch, length, embed)
    rows = torch.zeros(length, length, dtype=torch.bool)
    rows[0, 0] = True
    with torch.no_grad():
        plain_attn = attention(src, src, src, need_weights=False)[0]
        plain_weights = attention(src, src, src, need_weights=True)
        plain_layer = layer(src, src_mask=rows)
    on_mps = nn.MultiheadAttention(embed, heads, batch_first=True).eval().to("mps")
    on_mps.load_state_dict(attention.state_dict())
    layer_mps = (
        nn.TransformerEncoderLayer(
            embed, heads, dim_feedforward=4 * embed, batch_first=True, dropout=0.0
        )
        .eval()
        .to("mps")
    )
    layer_mps.load_state_dict(layer.state_dict())
    with torch.no_grad():
        mps_attn = on_mps(src.to("mps"), src.to("mps"), src.to("mps"), need_weights=False)[0]
        mps_weights = on_mps(src.to("mps"), src.to("mps"), src.to("mps"), need_weights=True)
        mps_layer = layer_mps(src.to("mps"), src_mask=rows.to("mps"))
    _matches(failures, report, "model_multihead_attention", mps_attn.cpu(), plain_attn)
    _matches(failures, report, "model_multihead_attention_weights", mps_weights[1].cpu(), plain_weights[1])
    _matches(failures, report, "model_transformer_layer_masked", mps_layer.cpu(), plain_layer)
    with torch.no_grad():
        plain_plain = layer(src)
        mps_plain = layer_mps(src.to("mps"))
    _matches(failures, report, "model_transformer_layer", mps_plain.cpu(), plain_plain)

    # training path: nn.MultiheadAttention computes its own gradients above the
    # overlays, so this is the only gradient the fast path offers
    train = nn.MultiheadAttention(embed, heads, batch_first=True)
    train_mps = nn.MultiheadAttention(embed, heads, batch_first=True).to("mps")
    train_mps.load_state_dict(train.state_dict())
    expected = train(src, src, src, need_weights=False)[0].sum()
    expected.backward()
    actual = train_mps(src.to("mps"), src.to("mps"), src.to("mps"), need_weights=False)[0].sum()
    actual.backward()
    weight_grad = train_mps.in_proj_weight.grad.cpu()
    _matches(failures, report, "model_multihead_attention_grad", weight_grad, train.in_proj_weight.grad)


def _time(fn, device, warmup=10, reps=30, batches=2):
    """Best of ``reps`` runs in each of ``batches`` passes, synchronized per run.

    The machine is shared, so the best sample is the least disturbed one; the
    median of a contended run moves by more than the difference being measured,
    and a neighbouring job's burst can swallow a whole pass, which is why the
    passes are separate.
    """
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


def _lease():
    """A tier-1 compute lease for the timing phase, when the broker is here."""
    from contextlib import nullcontext

    broker = Path(
        os.environ.get(
            "RESEARCH_CHAMBER", "/Users/dbydd/vibe-agent-working-dir/ResearchChamber"
        )
    ) / "shared_scripts"
    if not (broker / "compute_broker.py").exists():
        return nullcontext("no-broker")
    sys.path.insert(0, str(broker))
    from compute_broker import lease_guard

    return lease_guard("mps-kernelops", tier=1, task_name="attention_verify")


def _paired(before, after, floor, pairs=20, reps=8):
    """Interleaved before/after/floor: minima, and the median of per-pair ratios.

    Two variants timed in separate sweeps compare the box's load, not the code:
    a neighbour's burst can cost one variant a multiple of its own runtime and
    the box drifts over minutes. Every pair runs all three callables back to
    back inside the same window, the order alternates every pair so no variant
    systematically gets the quiet half, and each keeps its own minimum. ``floor``
    is a ``copy_`` of the same bytes in that same window: it is the ceiling a
    single launch of this size reaches here, so the pair means something even on
    a loaded box.
    """
    calls = (before, after, floor)
    for _ in range(5):
        for call in calls:
            call()
    torch.mps.synchronize()
    best = [None, None, None]
    ratios = []
    for pair in range(pairs):
        order = (0, 1, 2) if pair % 2 == 0 else (1, 0, 2)
        sample = {}
        for which in order:
            start = time.perf_counter()
            for _ in range(reps):
                calls[which]()
            torch.mps.synchronize()
            sample[which] = (time.perf_counter() - start) / reps
            best[which] = sample[which] if best[which] is None else min(best[which],
                                                                       sample[which])
        ratios.append(sample[0] / sample[1])
    ratios.sort()
    return {
        "before_ms": round(best[0] * 1e3, 4),
        "after_ms": round(best[1] * 1e3, 4),
        "floor_ms": round(best[2] * 1e3, 4),
        "ratio": round(best[0] / best[1], 2),
        "median_ratio": round(ratios[len(ratios) // 2], 2),
    }


def measure():
    torch.manual_seed(6)
    ops = []
    embed, heads, batch, length = 128, 8, 4, 128
    qkv_weight, qkv_bias, proj_weight, proj_bias = mha_weights(embed)
    f1, b1, f2, b2 = (
        torch.randn(4 * embed, embed),
        torch.randn(4 * embed),
        torch.randn(embed, 4 * embed),
        torch.randn(embed),
    )
    src = torch.randn(batch, length, embed)
    norm_weight, norm_bias = torch.ones(embed), torch.zeros(embed)
    rows = torch.zeros(length, length, dtype=torch.bool)
    rows[0, 0] = True
    scores = torch.randn(batch, heads, length, length)
    mask = torch.zeros(batch, heads, length, length, dtype=torch.bool)
    mask[..., 0] = True
    grad = torch.randn(batch, heads, length, length)
    softmaxed = torch.ops.aten._masked_softmax(scores, mask, -1, 2)
    qkv = torch.randn(batch, length, 3 * embed)

    cpu = dict(
        src=src, qkv=qkv, scores=scores, mask=mask, rows=rows, grad=grad, softmaxed=softmaxed,
        qkv_weight=qkv_weight, qkv_bias=qkv_bias,
        proj_weight=proj_weight, proj_bias=proj_bias,
        norm_weight=norm_weight, norm_bias=norm_bias,
        f1=f1, b1=b1, f2=f2, b2=b2,
    )
    mps = {name: tensor.to("mps") for name, tensor in cpu.items()}
    head_dim = embed // heads

    # helpers for the interleaved before/after comparison
    def ms_composite_fwd():
        return torch.softmax(mps["scores"].masked_fill(mps["mask"], float("-inf")), -1)
    def ms_composite_bwd():
        total = (mps["grad"] * mps["softmaxed"]).masked_fill(mps["mask"], 0.0).sum(-1, keepdim=True)
        return (mps["softmaxed"] * (mps["grad"] - total)).masked_fill(mps["mask"], 0.0)

    def _legacy_split():
        packed = mps["qkv"].reshape(batch, length, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
        bias = mps["qkv_bias"].reshape(3, heads, 1, head_dim)
        q = torch.add(packed[0].float(), bias[0].float()).contiguous().mul_(1.0 / math.sqrt(head_dim))
        return q.to(mps["qkv"].dtype), torch.add(packed[1], bias[1]).contiguous(), torch.add(packed[2], bias[2]).contiguous()

    legacy_sink_f32 = torch.empty_like(mps["scores"])
    legacy_sink_src = torch.empty_like(mps["src"])
    legacy_sink_qkv = torch.empty_like(mps["qkv"])

    def attention(t, **kwargs):
        return torch.ops.aten._native_multi_head_attention(
            t["src"], t["src"], t["src"], embed, heads,
            t["qkv_weight"], t["qkv_bias"], t["proj_weight"], t["proj_bias"], **kwargs,
        )

    def encoder(t):
        return torch.ops.aten._transformer_encoder_layer_fwd(
            t["src"], embed, heads, t["qkv_weight"], t["qkv_bias"],
            t["proj_weight"], t["proj_bias"], True, False, 1e-5,
            t["norm_weight"], t["norm_bias"], t["norm_weight"], t["norm_bias"],
            t["f1"], t["b1"], t["f2"], t["b2"],
        )

    cases = {
        "_masked_softmax": (
            lambda: torch.ops.aten._masked_softmax(scores, mask, -1, 2),
            lambda: torch.ops.aten._masked_softmax(mps["scores"], mps["mask"], -1, 2),
            ms_composite_fwd,
            lambda: legacy_sink_f32.copy_(mps["scores"]),
        ),
        "_masked_softmax_backward": (
            lambda: torch.ops.aten._masked_softmax_backward(grad, softmaxed, mask, -1),
            lambda: torch.ops.aten._masked_softmax_backward(
                mps["grad"], mps["softmaxed"], mps["mask"], -1
            ),
            ms_composite_bwd,
            lambda: legacy_sink_f32.copy_(mps["grad"]),
        ),
        "_transform_bias_rescale_qkv": (
            lambda: torch.ops.aten._transform_bias_rescale_qkv(qkv, qkv_bias, heads),
            lambda: torch.ops.aten._transform_bias_rescale_qkv(
                mps["qkv"], mps["qkv_bias"], heads
            ),
            _legacy_split,
            lambda: legacy_sink_qkv.copy_(mps["qkv"]),
        ),
        "_native_multi_head_attention_weights": (
            lambda: attention(cpu),
            lambda: attention(mps),
            lambda: _nmha_legacy(mps_dict, embed, heads),
            lambda: legacy_sink_src.copy_(mps["src"]),
        ),
        "_native_multi_head_attention": (
            lambda: attention(cpu, need_weights=False),
            lambda: attention(mps, need_weights=False),
            lambda: _nmha_legacy(mps_dict, embed, heads, need_weights=False),
            lambda: legacy_sink_src.copy_(mps["src"]),
        ),
        "_native_multi_head_attention_masked": (
            lambda: attention(cpu, mask=rows, need_weights=False, mask_type=0),
            lambda: attention(mps, mask=mps["rows"], need_weights=False, mask_type=0),
            lambda: _nmha_legacy(mps_dict, embed, heads, mask=mps["rows"], mask_type=0, need_weights=False),
            lambda: legacy_sink_src.copy_(mps["src"]),
        ),
        "_transformer_encoder_layer_fwd": (
            lambda: encoder(cpu),
            lambda: encoder(mps),
            lambda: attention_ops._encoder_composed(
                mps["src"], embed, heads, mps["qkv_weight"], mps["qkv_bias"],
                mps["proj_weight"], mps["proj_bias"], True, False, 1e-5,
                mps["norm_weight"], mps["norm_bias"], mps["norm_weight"], mps["norm_bias"],
                mps["f1"], mps["b1"], mps["f2"], mps["b2"],
            ),
            lambda: legacy_sink_src.copy_(mps["src"]),
        ),
    }
    # a dict the nmha legacy helper can index
    mps_dict = mps
    with torch.no_grad():
        for name, (cpu_fn, mps_fn, legacy_fn, floor_fn) in cases.items():
            cpu_ms = _time(cpu_fn, "cpu")
            mps_ms = _time(mps_fn, "mps")
            pair = _paired(legacy_fn, mps_fn, floor_fn)
            launches_before = _dispatches(legacy_fn)
            launches_after = _dispatches(mps_fn)
            load = os.getloadavg()
            ops.append(
                {
                    "op": name,
                    "batch": batch,
                    "length": length,
                    "embed": embed,
                    "heads": heads,
                    "dtype": "float32",
                    "cpu_ms": round(cpu_ms, 4),
                    "mps_ms": round(mps_ms, 4),
                    "speedup": round(cpu_ms / mps_ms, 2),
                    # interleaved paired measurement (before = aten composite, after = fused)
                    "before_ms": pair["before_ms"],
                    "after_ms": pair["after_ms"],
                    "ratio": pair["ratio"],
                    "ratio_median": pair["median_ratio"],
                    "floor_ms": pair["floor_ms"],
                    "launches_before": len(launches_before),
                    "launches_after": len(launches_after),
                    "load_average": round(load[0], 2),
                    # cpu_ms has no queue to drain; after_ms amortizes the MPS
                    # round trip over eight launches, so this is the two costs
                    # compared on the same footing
                    "speedup_amortized": round(cpu_ms / pair["after_ms"], 2),
                }
            )

    # why the pipeline uses the fused kernel: the attention core alone, with q
    # already carrying the 1/sqrt(head_dim) rescale
    core = []
    for shape, dtype in (
        ((4, 8, 128, 64), torch.float32),
        ((8, 8, 512, 64), torch.float32),
        ((8, 8, 512, 64), torch.float16),
    ):
        q = torch.randn(*shape, device="mps", dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        def explicit():
            weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)), -1)
            return torch.matmul(weights, v)

        with torch.no_grad():
            explicit_ms = _time(explicit, "mps")
            fused_ms = _time(
                lambda: F.scaled_dot_product_attention(q, k, v, scale=1.0), "mps"
            )
        core.append(
            {
                "batch": shape[0],
                "heads": shape[1],
                "length": shape[2],
                "head_dim": shape[3],
                "dtype": str(dtype).removeprefix("torch."),
                "explicit_ms": round(explicit_ms, 4),
                "fused_ms": round(fused_ms, 4),
                "speedup": round(explicit_ms / fused_ms, 2),
            }
        )

    models = []
    for length in (128, 512):
        for dtype in (torch.float32, torch.float16):
            src = torch.randn(4, length, embed, dtype=dtype)
            rows = torch.zeros(length, length, dtype=torch.bool)
            rows[0, 0] = True
            attention = nn.MultiheadAttention(embed, heads, batch_first=True).eval().to(dtype)
            attention_mps = nn.MultiheadAttention(embed, heads, batch_first=True).eval().to("mps")
            attention_mps.load_state_dict(attention.state_dict())
            attention_mps = attention_mps.to(dtype)
            layer = nn.TransformerEncoderLayer(
                embed, heads, dim_feedforward=4 * embed, batch_first=True, dropout=0.0
            ).eval().to(dtype)
            layer_mps = (
                nn.TransformerEncoderLayer(
                    embed, heads, dim_feedforward=4 * embed, batch_first=True, dropout=0.0
                )
                .eval()
                .to("mps")
            )
            layer_mps.load_state_dict(layer.state_dict())
            layer_mps = layer_mps.to(dtype)
            src_mps = src.to("mps")
            rows_mps = rows.to("mps")
            for name, cpu_fn, mps_fn in (
                (
                    "nn.MultiheadAttention",
                    lambda: attention(src, src, src, need_weights=False)[0],
                    lambda: attention_mps(
                        src_mps, src_mps, src_mps, need_weights=False
                    )[0],
                ),
                (
                    "nn.MultiheadAttention(need_weights)",
                    lambda: attention(src, src, src, need_weights=True),
                    lambda: attention_mps(src_mps, src_mps, src_mps),
                ),
                (
                    "nn.TransformerEncoderLayer",
                    lambda: layer(src, src_mask=rows),
                    lambda: layer_mps(src_mps, src_mask=rows_mps),
                ),
            ):
                with torch.no_grad():
                    cpu_ms = _time(cpu_fn, "cpu")
                    mps_ms = _time(mps_fn, "mps")
                models.append(
                    {
                        "model": name,
                        "batch": 4,
                        "length": length,
                        "embed": embed,
                        "heads": heads,
                        "dtype": str(dtype).removeprefix("torch."),
                        "cpu_ms": round(cpu_ms, 4),
                        "mps_ms": round(mps_ms, 4),
                        "speedup": round(cpu_ms / mps_ms, 2),
                    }
                )
    return ops, core, models


_DISPATCH_META = {
    "aten::as_strided", "aten::view", "aten::alias", "aten::select", "aten::t",
    "aten::permute", "aten::transpose", "aten::expand", "aten::_unsafe_view",
    "aten::_reshape_alias", "aten::flatten", "aten::empty", "aten::empty_like",
    "aten::empty_strided", "aten::reshape", "aten::detach", "aten::lift_fresh",
    "aten::slice", "aten::squeeze", "aten::unsqueeze",
}
"""Leaf aten calls that move no data; every other leaf is one Metal dispatch."""


def _dispatches(fn):
    """Metal dispatches one call issues, counted from the aten events it raises.

    A dispatch counts when nothing under it launches anything: an op whose
    children are all views, aliases or allocations is itself the enqueue, which
    is what an overlay's body looks like once it holds a Metal kernel, while an
    op that fans out into other kernel-bearing ops is counted through them. The
    leaf walk alone missed the first case and reported an overlay with one
    kernel as zero dispatches.
    """
    from torch.profiler import ProfilerActivity, profile

    with torch.no_grad():
        fn()
        torch.mps.synchronize()
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            fn()
            torch.mps.synchronize()
    names = []

    def walk(event):
        children = [
            child for child in getattr(event, "cpu_children", [])
            if child.name not in _DISPATCH_META
        ]
        if not children:
            names.append(event.name)
            return
        for child in children:
            walk(child)

    for event in prof.events():
        if getattr(event, "cpu_parent", None) is None and event.name not in _DISPATCH_META:
            walk(event)
    return [name for name in names if name.startswith("aten::")]


def _nmha_legacy(t, embed, heads, mask=None, mask_type=None, need_weights=True):
    """The pipeline the fused core replaced, rebuilt from the module's helpers.

    Those helpers still serve the over-long-row path, so this is the code the
    operator ran before the rewrite; timing it beside the rewrite on one batch
    of tensors is what makes the comparison evidence on a shared device.
    """
    qkv = attention_ops._project_qkv(t["src"], t["src"], t["src"], embed, t["qkv_weight"])
    q, k, v = attention_ops._transform_bias_rescale_qkv(qkv, t["qkv_bias"], heads)
    context, weights = attention_ops._attention(
        q, k, v, mask, mask_type, need_weights, True, heads
    )
    merged = context.permute(0, 2, 1, 3).reshape(t["src"].shape[0], t["src"].shape[1], embed)
    return F.linear(merged, t["proj_weight"], t["proj_bias"]), weights


def native_multi_head_report():
    """Dispatch counts and a same-batch before/after for the fused operator.

    Absolute milliseconds move by several times between sweeps while the device
    is shared, so every row is timed inside one batch next to a ``copy_`` of the
    same shape as a floor, and reported as that batch's minimum. The core row
    times the fused kernel alone and gives the bandwidth over the traffic it
    must move: the packed projection in and the merged context out.
    """
    torch.manual_seed(12)
    embed, heads, batch, length = 128, 8, 4, 128
    head_dim = embed // heads
    qkv_weight, qkv_bias, proj_weight, proj_bias = mha_weights(embed)
    src = torch.randn(batch, length, embed)
    rows = torch.zeros(length, length, dtype=torch.bool)
    rows[0, 0] = True
    t = {
        "src": src.to("mps"),
        "qkv_weight": qkv_weight.to("mps"),
        "qkv_bias": qkv_bias.to("mps"),
        "proj_weight": proj_weight.to("mps"),
        "proj_bias": proj_bias.to("mps"),
    }
    mask = rows.to("mps")
    sink = torch.empty(batch, length, embed, device="mps")

    def new(**kwargs):
        return torch.ops.aten._native_multi_head_attention(
            t["src"], t["src"], t["src"], embed, heads, t["qkv_weight"], t["qkv_bias"],
            t["proj_weight"], t["proj_bias"], **kwargs,
        )

    def old(**kwargs):
        return _nmha_legacy(t, embed, heads, **kwargs)

    packed = torch.matmul(t["src"], t["qkv_weight"].t())
    merged = torch.empty(batch, length, embed, device="mps")
    kernel, kernel_rows, _ = attention_ops._mha_plan(torch.float32, head_dim, length)
    tiles = (length + kernel_rows - 1) // kernel_rows
    spare_f = attention_ops._mha_spare(torch.float32, "mps")
    spare_u8 = attention_ops._mha_spare(torch.uint8, "mps")

    def core():
        kernel(
            packed, t["qkv_bias"], spare_u8, spare_f, merged, length, embed, heads,
            tiles, 3, 0, float(head_dim ** -0.5),
            threads=batch * heads * tiles * kernel_rows * 32, group_size=kernel_rows * 32,
        )

    cases = {
        "copy_floor": lambda: sink.copy_(t["src"]),
        "no_weights": lambda: new(need_weights=False),
        "weights": lambda: new(need_weights=True),
        "masked": lambda: new(mask=mask, need_weights=False, mask_type=0),
        "legacy_no_weights": lambda: old(need_weights=False),
        "legacy_weights": lambda: old(need_weights=True),
        "legacy_masked": lambda: old(mask=mask, need_weights=False, mask_type=0),
        "core": core,
    }
    best = {}
    with torch.no_grad():
        for _ in range(4):
            for name, fn in cases.items():
                value = _time(fn, "mps", warmup=5, reps=15, batches=1)
                best[name] = min(best.get(name, value), value)
        dispatches = {
            name: len(_dispatches(fn)) for name, fn in cases.items() if name != "copy_floor"
        }

    core_bytes = packed.numel() * 4 + merged.numel() * 4
    table = []
    for name in ("no_weights", "weights", "masked"):
        table.append(
            {
                "case": name,
                "legacy_ms": round(best[f"legacy_{name}"], 4),
                "fused_ms": round(best[name], 4),
                "speedup": round(best[f"legacy_{name}"] / best[name], 2),
                "legacy_dispatches": dispatches[f"legacy_{name}"],
                "fused_dispatches": dispatches[name],
            }
        )
    return {
        "shape": {"batch": batch, "length": length, "embed": embed, "heads": heads,
                  "head_dim": head_dim, "dtype": "float32"},
        "method": (
            "minimum of four interleaved batches, torch.mps.synchronize() per call, "
            "each row timed next to a copy_ floor of the same shape; the device was "
            "shared with about twenty other MPS jobs, so the milliseconds are upper "
            "bounds and the paired ratios are the measurement"
        ),
        "copy_floor_ms": round(best["copy_floor"], 4),
        "core": {
            "ms": round(best["core"], 4),
            "dispatches": dispatches["core"],
            "bytes": core_bytes,
            "gb_per_s": round(core_bytes / (best["core"] * 1e6), 1),
        },
        "cases": table,
    }


def main():
    version = torch.__version__.split("+")[0]
    if version != CHECKED_TORCH:
        raise SystemExit(f"overlays were checked against torch {CHECKED_TORCH}, running {version}")
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available on this machine")

    # before the overlay: every operator in the table has to be missing
    missing = [
        name
        for name, _ in attention_ops.OVERLAYS
        if not torch._C._dispatch_has_kernel_for_dispatch_key(f"aten::{name}", "MPS")
    ]
    library = install()

    failures, report = [], {}
    # importing the module must compile nothing and register nothing: every
    # shader cache is still empty here, before the first operator call
    caches = (
        attention_ops._ms_library.cache_info().currsize,
        attention_ops._encoder_library.cache_info().currsize,
        attention_ops._library.cache_info().currsize,
        len(attention_ops._MHA_KERNELS),
    )
    _record(
        failures,
        report,
        "import_compiles_nothing",
        caches == (0, 0, 0, 0),
        f"shader caches at import {caches}",
    )
    _record(failures, report, "baseline_missing_on_mps", len(missing) == len(attention_ops.OVERLAYS), str(missing))
    check_masked_softmax(failures, report)
    check_masked_softmax_backward(failures, report)
    check_transform_bias_rescale_qkv(failures, report)
    check_native_multi_head_attention(failures, report)
    check_transformer_encoder_layer(failures, report)
    check_fused_sdp_choice(failures, report)
    check_models(failures, report)
    with _lease():
        ops, core, models = measure()
        split = measure_transform_bias_rescale_qkv()
        nmha = native_multi_head_report()

    measured = {
        "torch": version,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "date": time.strftime("%Y-%m-%d"),
        "overlays": [name for name, _ in attention_ops.OVERLAYS],
        "baseline": "each operator raises NotImplementedError on an MPS tensor without the overlay",
        "timing": (
            "three numbers per operator, because the box is shared and one number "
            "cannot separate code from load. cpu_ms is the best of 30 CPU runs; mps_ms "
            "is the best of 30 MPS runs with torch.mps.synchronize() after each, so it "
            "is dominated by the device round trip (about 0.12 ms here) rather than by "
            "the kernel, and speedup = cpu_ms / mps_ms carries that bias against MPS. "
            "before_ms/after_ms come from one interleaved batch: the aten composite and "
            "the fused operator run back to back, order alternating, eight launches "
            "between synchronizations, minimum per variant, next to a copy_ of the same "
            "bytes as the floor; ratio = before_ms / after_ms is the code comparison "
            "and ratio_median is the median of the per-pair ratios. speedup_amortized = "
            "cpu_ms / after_ms compares the CPU cost, which has no queue to drain, with "
            "the amortized MPS cost"
        ),
        "not_measured": {
            "_fused_sdp_choice": "returns an SDPBackend enum value; no tensor work to time"
        },
        "model_scope": (
            "torch 2.14.1 routes nn.MultiheadAttention and nn.TransformerEncoderLayer "
            "away from these private operators on MPS; the model rows measure the native "
            "MPS module path against CPU and do not exercise the overlays"
        ),
        "checks": report,
        "native_multi_head": nmha,
        "ops": ops,
        "transform_bias_rescale_qkv": split,
        "attention_core": core,
        "models": models,
    }
    target = ROOT / "measured" / "attention_ops.json"
    target.write_text(json.dumps(measured, indent=2) + "\n")

    print(f"{len(report) - len(failures)}/{len(report)} checks passed")
    for failure in failures:
        print(f"  FAIL {failure}")
    for row in ops:
        print(f"  {row['op']:<45} cpu {row['cpu_ms']:8.3f} ms  mps {row['mps_ms']:8.3f} ms  {row['speedup']:5.2f}x")
    for row in ops:
        if "before_ms" not in row:
            continue
        print(
            f"  {row['op']:<40} composite {row['before_ms']:7.4f} -> fused {row['after_ms']:7.4f} ms"
            f"  {row['ratio']:5.2f}x (median {row['ratio_median']:5.2f})"
            f"  cpu-vs-fused {row['speedup_amortized']:5.2f}x"
            f"  floor {row['floor_ms']:7.4f}  launches {row['launches_before']} -> {row['launches_after']}"
        )
    for row in core:
        print(
            f"  attention core {row['dtype']:<8} L={row['length']:<4}"
            f" explicit {row['explicit_ms']:8.3f} ms  fused {row['fused_ms']:8.3f} ms"
            f"  {row['speedup']:5.2f}x"
        )
    for row in models:
        print(
            f"  {row['model']:<38} {row['dtype']:<8} L={row['length']:<4}"
            f" cpu {row['cpu_ms']:8.3f} ms  mps {row['mps_ms']:8.3f} ms  {row['speedup']:5.2f}x"
        )
    nmha = measured["native_multi_head"]
    print(f"  native_multi_head (copy_ floor {nmha['copy_floor_ms']:.4f} ms, core {nmha['core']['ms']:.4f} ms @ {nmha['core']['gb_per_s']} GB/s):")
    for row in nmha["cases"]:
        print(
            f"    {row['case']:<16} legacy {row['legacy_ms']:7.4f} ms ({row['legacy_dispatches']} disp) "
            f" fused {row['fused_ms']:7.4f} ms ({row['fused_dispatches']} disp)  {row['speedup']:5.2f}x"
        )
    print(f"wrote {target}")
    load = os.getloadavg()
    print(f"  load average {load[0]:.2f}")
    for row in split["rows"]:
        print(
            f"  split {row['dtype']:<8} legacy {row['legacy_sync_ms']:7.4f} ms"
            f"  fused {row['fused_sync_ms']:7.4f} ms  {row['speedup_sync']:5.2f}x"
            f" (rounds median {row['speedup_sync_median']:5.2f}x)"
            f"  copy {row['copy_sync_ms']:7.4f} ms"
            f" | queued legacy {row['legacy_queued_ms']:7.4f}"
            f"  fused {row['fused_queued_ms']:7.4f}  {row['speedup_queued']:5.2f}x"
            f"  {row['fused_gbs']:5.1f} GB/s  copy {row['copy_queued_ms']:7.4f} ms"
        )
    keep = library  # noqa: F841 - the registration lives as long as the library
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
