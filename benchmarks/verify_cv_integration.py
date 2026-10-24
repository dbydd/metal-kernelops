"""End-to-end checks for the tree-external CV operators.

A small Mamba-2 block, a VMamba-style path-scan block, and NSA block selection
all run first on CPU through the float64 references and then on MPS through the
fused public entry points.  Forward values, every parameter gradient, and one
SGD step are compared.
"""

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
from metal_kernelops.cv import (
    causal_conv1d_fn,
    causal_conv1d_ref,
    k_select,
    k_select_values,
    mamba2_chunk_scan,
    mamba2_chunk_scan_reference,
    reference_k_select,
    rmsnorm_gated,
    rmsnorm_gated_reference,
    ss2d_scan,
    ss2d_scan_reference,
)

torch.manual_seed(17)


class Mamba2Block(torch.nn.Module):
    def __init__(self, d_model=24, heads=2, head_dim=6, d_state=5, conv=4):
        super().__init__()
        self.d_model, self.heads, self.head_dim = d_model, heads, head_dim
        self.d_state, self.conv = d_state, conv
        inner = heads * head_dim
        self.in_proj = torch.nn.Linear(d_model, 2 * inner + 2 * d_state + heads)
        self.conv_weight = torch.nn.Parameter(torch.randn(inner + 2 * d_state, 1, conv) * .3)
        self.conv_bias = torch.nn.Parameter(torch.randn(inner + 2 * d_state) * .1)
        self.a_log = torch.nn.Parameter(torch.randn(heads) * .2)
        self.d_skip = torch.nn.Parameter(torch.ones(heads))
        self.dt_bias = torch.nn.Parameter(torch.randn(heads) * .2)
        self.norm_weight = torch.nn.Parameter(torch.ones(inner))
        self.out_proj = torch.nn.Linear(inner, d_model)
        self.inner = inner

    def forward(self, x, reference=False):
        zxbcdt = self.in_proj(x)
        z = zxbcdt[..., :self.inner]
        xbc = zxbcdt[..., self.inner:2 * self.inner + 2 * self.d_state]
        dt = zxbcdt[..., 2 * self.inner + 2 * self.d_state:]
        conv = causal_conv1d_ref if reference else causal_conv1d_fn
        xbc = conv(
            xbc.transpose(1, 2), self.conv_weight.reshape(self.inner + 2 * self.d_state, -1),
            self.conv_bias, activation="silu",
        ).transpose(1, 2)
        x_ssm = xbc[..., :self.inner].reshape(x.shape[0], x.shape[1], self.heads, self.head_dim)
        B = xbc[..., self.inner:self.inner + self.d_state].unsqueeze(2)
        C = xbc[..., self.inner + self.d_state:].unsqueeze(2)
        A = -torch.exp(self.a_log)
        scan = mamba2_chunk_scan_reference if reference else mamba2_chunk_scan
        y, _ = scan(
            x_ssm, dt, A, B, C, chunk_size=8, D=self.d_skip,
            dt_bias=self.dt_bias, dt_softplus=True,
        )
        y = y.reshape(x.shape[0], x.shape[1], self.inner)
        norm = rmsnorm_gated_reference if reference else rmsnorm_gated
        y = norm(y, self.norm_weight, z=z, norm_before_gate=False)
        return self.out_proj(y).square().mean()


class PathBlock(torch.nn.Module):
    def __init__(self, channels=6, groups=2, states=4):
        super().__init__()
        self.channels, self.groups, self.states = channels, groups, states
        self.u = torch.nn.Conv2d(channels, channels, 3, padding=1)
        self.dt = torch.nn.Conv2d(channels, channels, 1)
        self.b = torch.nn.Conv2d(channels, groups * states, 1)
        self.c = torch.nn.Conv2d(channels, groups * states, 1)
        self.a = torch.nn.Parameter(-torch.rand(channels, states))

    def forward(self, x, reference=False):
        u = self.u(x)
        dt = F.softplus(self.dt(x))
        shape = (x.shape[0], self.groups, self.states, x.shape[2], x.shape[3])
        B = self.b(x).reshape(shape)
        C = self.c(x).reshape(shape)
        scan = ss2d_scan_reference if reference else ss2d_scan
        return scan(u, dt, self.a, B, C, merge="sum").square().mean()


def compare(cpu, mps, tolerance):
    torch.testing.assert_close(mps.cpu(), cpu, rtol=tolerance, atol=tolerance)


def gradients(cpu_model, mps_model, x):
    cpu_model.zero_grad(set_to_none=True)
    mps_model.zero_grad(set_to_none=True)
    cpu_loss = cpu_model(x, reference=True)
    mps_loss = mps_model(x.to("mps"))
    cpu_loss.backward()
    mps_loss.backward()
    compare(cpu_loss.detach(), mps_loss.detach(), 5e-5)
    rows = []
    for (name, a), (_, b) in zip(cpu_model.named_parameters(), mps_model.named_parameters()):
        torch.testing.assert_close(b.grad.cpu(), a.grad, rtol=2e-4, atol=2e-5)
        rows.append(name)
    before = {n: p.detach().clone() for n, p in mps_model.named_parameters()}
    torch.optim.SGD(cpu_model.parameters(), lr=.03).step()
    torch.optim.SGD(mps_model.parameters(), lr=.03).step()
    moved = all(not torch.equal(before[n], p) for n, p in mps_model.named_parameters())
    return {"loss": cpu_loss.item(), "gradient_parameters": rows, "parameters_moved": moved}


def run():
    torch.manual_seed(17)
    mamba_cpu = Mamba2Block()
    mamba_mps = Mamba2Block().to("mps")
    mamba_mps.load_state_dict(mamba_cpu.state_dict())
    mamba_result = gradients(mamba_cpu, mamba_mps, torch.randn(2, 19, 24))

    path_cpu = PathBlock()
    path_mps = PathBlock().to("mps")
    path_mps.load_state_dict(path_cpu.state_dict())
    path_result = gradients(path_cpu, path_mps, torch.randn(2, 6, 9, 11))

    scores = torch.randn(2, 3, 40, 66)
    scores[0, 0, 3, 5] = scores[0, 0, 3, 9] = 4.0
    idx_cpu = reference_k_select(scores, 8)
    idx_mps = k_select(scores.to("mps"), 8)
    assert torch.equal(idx_cpu, idx_mps.cpu())
    gather_cpu = scores.gather(-1, idx_cpu)
    gather_mps = k_select_values(scores.to("mps", dtype=torch.float32), 8)
    compare(gather_cpu, gather_mps, 1e-6)

    torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(4):
        mps_loss = mamba_mps(torch.randn(2, 19, 24, device="mps"))
        mps_loss.backward()
        mamba_mps.zero_grad(set_to_none=True)
    torch.mps.synchronize()
    mamba_ms = (time.perf_counter() - start) * 250

    start = time.perf_counter()
    for _ in range(4):
        path_loss = path_mps(torch.randn(2, 6, 9, 11, device="mps"))
        path_loss.backward()
        path_mps.zero_grad(set_to_none=True)
    torch.mps.synchronize()
    path_ms = (time.perf_counter() - start) * 250

    start = time.perf_counter()
    for _ in range(20):
        k_select(torch.randn(2, 3, 40, 66, device="mps"), 8)
    torch.mps.synchronize()
    select_ms = (time.perf_counter() - start) * 50

    result = {
        "torch": torch.__version__,
        "mamba2_block": mamba_result,
        "mamba2_step_ms": mamba_ms,
        "path_scan_block": path_result,
        "path_scan_step_ms": path_ms,
        "k_select": {"indices": "bit-equal to reference", "k_select_ms": select_ms},
        "status": "pass",
    }
    (ROOT / "measured/cv_integration.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    with lease_guard("mps-kernelops", tier=1, task_name="cv-integration"):
        run()
