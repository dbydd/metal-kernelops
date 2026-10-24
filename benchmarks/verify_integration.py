"""Run a model step across independent overlay families after one install."""

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parents[1] / "shared_scripts"))

import torch
import torch.nn.functional as F
from compute_broker import lease_guard
from metal_kernelops import accelerator


class MixedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 8, 3, padding=1)
        self.head = torch.nn.Linear(8 * 12 * 12, 11)

    def forward(self, x, samples):
        x = self.conv(x)
        x = F.channel_shuffle(x, 2)
        x = F.rrelu(x, training=False)
        x = F.fractional_max_pool2d(x, 2, output_size=(12, 12), _random_samples=samples)
        x = F.interpolate(x, size=(15, 15), mode="lanczos", antialias=True)
        x = F.interpolate(x, size=(12, 12), mode="lanczos", antialias=True)
        x = self.head(x.flatten(1))
        p = x.sigmoid().clamp(.01, .99)
        return torch.special.log_ndtr(torch.special.ndtri(p)).mean()


def run():
    torch.manual_seed(719)
    assert accelerator.coverage() == {}
    coverage = accelerator.install()
    a = MixedModel()
    b = MixedModel().to("mps")
    b.load_state_dict(a.state_dict())
    x = torch.randn(3, 4, 24, 24)
    samples = torch.rand(3, 8, 2)
    oa, ob = torch.optim.SGD(a.parameters(), lr=.01), torch.optim.SGD(b.parameters(), lr=.01)
    initial = {n: v.detach().clone() for n, v in b.named_parameters()}
    la, lb = a(x, samples), b(x.to("mps"), samples.to("mps"))
    la.backward()
    lb.backward()
    torch.testing.assert_close(lb.cpu(), la, rtol=3e-5, atol=3e-6)
    grad_rows = []
    sq_a, sq_b = 0., 0.
    for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        torch.testing.assert_close(pb.grad.cpu(), pa.grad, rtol=2e-4, atol=3e-6)
        assert torch.isfinite(pb.grad).all() and pb.grad.count_nonzero() > 0
        sq_a += pa.grad.double().square().sum().item()
        sq_b += pb.grad.double().cpu().square().sum().item() if pb.grad.dtype == torch.float64 else pb.grad.cpu().double().square().sum().item()
        grad_rows.append(name)
    oa.step()
    ob.step()
    moved = all(not torch.equal(initial[n], p) for n, p in b.named_parameters())
    assert moved
    torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(5):
        ob.zero_grad(set_to_none=True)
        loss = b(x.to("mps"), samples.to("mps"))
        loss.backward()
        ob.step()
    torch.mps.synchronize()
    result = {"torch": torch.__version__, "overlays": len(coverage), "cpu_loss": la.item(), "mps_loss": lb.item(), "gradient_parameters": grad_rows, "gradient_norm_relative_error": abs(sq_b ** .5 - sq_a ** .5) / sq_a ** .5, "all_parameters_moved": moved, "mps_step_ms": (time.perf_counter() - start) * 200, "status": "pass"}
    (ROOT / "measured/integration.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    with lease_guard("mps-kernelops", tier=1, task_name="model-integration"):
        run()
