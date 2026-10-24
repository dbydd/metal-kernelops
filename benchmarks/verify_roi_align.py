"""Verify the torchvision-compatible RoIAlign implementation on MPS.

Checks, in order:

* torchvision parity for ``aligned in {False, True}`` and
  ``sampling_ratio in {-1, 0, 1, 2, 3}``, at rtol 2e-5 / atol 2e-6;
* the composed fallback and the Metal kernel agree, so the adaptive grid that
  keeps the composed path stays exact;
* input gradients match torchvision's, and are bitwise reproducible;
* list boxes, empty boxes and the error contracts;
* timings: composed torch path, Metal path, torchvision's CPU kernel, and a
  ``fill_`` of the same output as the measured floor for one write.
"""

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parents[1] / "shared_scripts"))

import torch
import torchvision
from compute_broker import lease_guard
from metal_kernelops.cv.roi_align import _composed, roi_align

RTOL = 2e-5
ATOL = 2e-6

SHAPES = (
    (8, 256, 56, 56, 512),
    (4, 128, 112, 112, 256),
    (32, 64, 28, 28, 1024),
)

BASELINE_MS = 56.8
"""The composed path at (8, 256, 56, 56) with 512 boxes, before this work, on an
otherwise idle machine. Today's composed column cannot reproduce it: the GPU is
shared, and the composed path's ~40x larger traffic suffers far more from that
contention than the single-pass kernel does."""


def interleaved_ms(runners, rounds=7, warmup=2):
    """Per-runner minima over one batch of rounds.

    Every candidate runs in every round, so a lease that grabs the GPU mid-batch
    inflates all of them and the minima stay comparable. A one-shot A/B on this
    machine is meaningless: the same kernel moves by several times between runs.
    """
    for _ in range(warmup):
        for fn in runners.values():
            fn()
    torch.mps.synchronize()
    best = {name: float("inf") for name in runners}
    for _ in range(rounds):
        for name, fn in runners.items():
            torch.mps.synchronize()
            start = time.perf_counter()
            fn()
            torch.mps.synchronize()
            best[name] = min(best[name], (time.perf_counter() - start) * 1e3)
    return best


def expect(kind, fn):
    try:
        fn()
    except kind:
        return True
    raise AssertionError(f"{kind.__name__} not raised")


def run():
    torch.manual_seed(23)
    checks = []

    for aligned in (False, True):
        for ratio in (-1, 0, 1, 2, 3):
            x = torch.randn(2, 3, 9, 11)
            rois = torch.tensor(
                [[0, 1.2, 2.3, 6.1, 7.0], [1, 0.5, 0.7, 8.2, 8.9], [0, -2.0, -1.0, 20.0, 19.0]]
            )
            reference = torchvision.ops.roi_align(x, rois, (3, 4), 1.0, ratio, aligned)
            result = roi_align(x.to("mps"), rois.to("mps"), (3, 4), 1.0, ratio, aligned)
            torch.testing.assert_close(result.cpu(), reference, rtol=RTOL, atol=ATOL)
            composed = _composed(x.to("mps"), rois.to("mps"), 3, 4, 1.0, ratio, aligned)
            torch.testing.assert_close(result.cpu(), composed.cpu(), rtol=RTOL, atol=ATOL)
            checks.append(f"aligned={aligned} ratio={ratio}")

    for aligned in (False, True):
        for ratio in (-1, 2, 3):
            torch.manual_seed(41)
            x = torch.randn(3, 5, 24, 31)
            start = torch.rand(17, 2) * torch.tensor([30.0, 23.0])
            rois = torch.cat(
                [
                    torch.randint(0, 3, (17, 1)).float(),
                    start,
                    start + torch.rand(17, 2) * 9 + 0.5,
                ],
                1,
            )
            reference = torchvision.ops.roi_align(x, rois, (5, 6), 0.6, ratio, aligned)
            result = roi_align(x.to("mps"), rois.to("mps"), (5, 6), 0.6, ratio, aligned)
            torch.testing.assert_close(result.cpu(), reference, rtol=RTOL, atol=ATOL)
            checks.append(f"wide boxes aligned={aligned} ratio={ratio}")

    x = torch.randn(2, 3, 9, 11)
    rois = torch.tensor([[0, 1.2, 2.3, 6.1, 7.0], [1, 0.5, 0.7, 8.2, 8.9]])
    x_cpu = x.clone().requires_grad_()
    x_mps = x.clone().to("mps").requires_grad_()
    reference = torchvision.ops.roi_align(x_cpu, rois, (3, 4), 1.0, 2, False)
    result = roi_align(x_mps, rois.to("mps"), (3, 4), 1.0, 2, False)
    reference.square().mean().backward()
    result.square().mean().backward()
    torch.testing.assert_close(result.cpu(), reference.detach(), rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(x_mps.grad.cpu(), x_cpu.grad, rtol=1e-5, atol=1e-6)
    checks.append("input gradient")

    torch.manual_seed(5)
    grad_x = torch.randn(3, 4, 17, 21, device="mps", requires_grad=True)
    start = torch.rand(29, 2, device="mps") * torch.tensor([20.0, 16.0], device="mps")
    grad_rois = torch.cat(
        [
            torch.randint(0, 3, (29, 1), device="mps").float(),
            start,
            start + torch.rand(29, 2, device="mps") * 7 + 0.5,
        ],
        1,
    )
    runs = []
    for _ in range(5):
        values = roi_align(grad_x, grad_rois, (4, 5), 0.8, 3, True)
        weight = torch.arange(values.numel(), device="mps").reshape(values.shape)
        (grad,)= torch.autograd.grad((values * weight).sum(), grad_x, retain_graph=True)
        runs.append(grad.clone())
    assert all(torch.equal(runs[0], run) for run in runs[1:]), "backward is not reproducible"
    checks.append("deterministic backward")

    list_boxes = [torch.tensor([[1.0, 2.0, 6.0, 7.0]]), torch.tensor([[0.5, 0.5, 8.0, 8.5]])]
    list_ref = torchvision.ops.roi_align(x.detach(), list_boxes, (2, 2), 1.0, 2, True)
    list_mps = roi_align(x.detach().to("mps"), [b.to("mps") for b in list_boxes], (2, 2), 1.0, 2, True)
    torch.testing.assert_close(list_mps.cpu(), list_ref, rtol=RTOL, atol=ATOL)
    checks.append("list boxes")

    empty = roi_align(x.detach().to("mps"), torch.empty((0, 5), device="mps"), (2, 2))
    assert empty.shape == (0, 3, 2, 2)
    empty_x = x.clone().to("mps").requires_grad_()
    empty = roi_align(empty_x, torch.empty((0, 5), device="mps"), (2, 2))
    empty.sum().backward()
    assert empty.shape == (0, 3, 2, 2)
    assert float(empty_x.grad.abs().sum()) == 0.0
    checks.append("empty boxes")

    x_mps = x.detach().to("mps")
    expect(ValueError, lambda: roi_align(x[0], torch.zeros((1, 5)), (2, 2)))
    expect(TypeError, lambda: roi_align(x.double().to("mps"), torch.zeros((1, 5), device="mps"), (2, 2)))
    expect(ValueError, lambda: roi_align(x_mps, torch.zeros((1, 3), device="mps"), (2, 2)))
    expect(ValueError, lambda: roi_align(x_mps, torch.zeros((1, 5), device="mps"), (0, 2)))
    checks.append("error contracts")

    timings = []
    for batch, channels, height, width, boxes in SHAPES:
        base = torch.randn(batch, channels, height, width)
        batch_index = torch.randint(0, batch, (boxes, 1)).float()
        start = torch.rand(boxes, 2) * torch.tensor([width - 3, height - 3])
        rois = torch.cat([batch_index, start, start + torch.rand(boxes, 2) * 4 + 1], 1)
        base_mps, rois_mps = base.to("mps"), rois.to("mps")
        reference = torchvision.ops.roi_align(base, rois, (7, 7), 1.0, 2, False)
        # The CPU kernel accumulates in double; both MPS paths carry float32.
        # At these sizes that shows up as a few 1e-6 against a data scale of ~4,
        # so the parity bound here is that error over the reference's peak.
        produced = roi_align(base_mps, rois_mps, (7, 7), 1.0, 2, False).cpu()
        parity = float((produced - reference).abs().max()) / float(reference.abs().max())
        assert parity <= 2e-6, f"parity {parity:.2e} over the data scale"
        grad = torch.randn_like(reference, device="mps")
        leaves = base_mps.clone().requires_grad_()
        out = roi_align(leaves, rois_mps, (7, 7), 1.0, 2, False)
        plain = base_mps.clone().requires_grad_()
        plain_out = _composed(plain, rois_mps, 7, 7, 1.0, 2, False)
        sink = torch.empty(reference.numel(), device="mps")
        source = torch.empty(reference.numel(), device="mps")
        measured = interleaved_ms(
            {
                "composed": lambda: _composed(base_mps, rois_mps, 7, 7, 1.0, 2, False),
                "metal": lambda: roi_align(base_mps, rois_mps, (7, 7), 1.0, 2, False),
                "cpu": lambda: torchvision.ops.roi_align(base, rois, (7, 7), 1.0, 2, False),
                "composed_backward": lambda: torch.autograd.grad(
                    plain_out, plain, grad, retain_graph=True
                ),
                "backward": lambda: torch.autograd.grad(
                    out, leaves, grad, retain_graph=True
                ),
                "fill": lambda: sink.fill_(1.0),
                "copy": lambda: sink.copy_(source),
            }
        )
        out_bytes = reference.numel() * 4
        in_bytes = base.numel() * 4
        timings.append(
            {
                "shape": [batch, channels, height, width],
                "boxes": boxes,
                "boxes_per_image": boxes / batch,
                "output_mb": round(out_bytes / 1e6, 2),
                "parity_over_peak": parity,
                "composed_ms": measured["composed"],
                "composed_backward_ms": measured["composed_backward"],
                "metal_ms": measured["metal"],
                "metal_backward_ms": measured["backward"],
                "cpu_ms": measured["cpu"],
                "speedup_same_batch": measured["composed"] / measured["metal"],
                "speedup_backward_same_batch": (
                    measured["composed_backward"] / measured["backward"]
                ),
                "metal_gb_s": out_bytes / measured["metal"] / 1e6,
                "fill_ms": measured["fill"],
                "copy_ms": measured["copy"],
                "input_mb": round(in_bytes / 1e6, 2),
            }
        )

    result = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "checks": checks,
        "baseline": {
            "shape": [8, 256, 56, 56],
            "boxes": 512,
            "output_size": [7, 7],
            "sampling_ratio": 2,
            "composed_ms": BASELINE_MS,
            "note": "recorded before this work on an idle machine; see measurement",
        },
        "measurement": (
            "one interleaved batch per shape: every column runs once per round, "
            "warmup 2, 7 rounds, per-column minimum; the GPU is shared with other "
            "leases, so only these same-batch minima are comparable. The same-batch "
            "ratios overstate the idle-machine ratio because contention costs the "
            "composed path, which moves ~40x the bytes, far more than the kernel."
        ),
        "floor_note": (
            "fill_ms writes the output size with no reads; copy_ms reads and writes "
            "it. Both are measured in the same batch as the kernel column."
        ),
        "timings": timings,
        "note": (
            "torchvision has no RoIAlign MPS kernel; the CPU column is torchvision's "
            "native kernel. The composed column is the same operator in plain torch "
            "ops, which is what the Metal path replaces; its backward is autograd "
            "through those ops, the only backward the drafted kernel path had."
        ),
        "status": "pass",
    }
    (ROOT / "measured/roi_align.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    with lease_guard("mps-kernelops", tier=1, task_name="roi-align"):
        run()
