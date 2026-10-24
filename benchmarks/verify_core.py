"""Exercise ordinary PyTorch entry points through the installed accelerator."""

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parents[1] / "shared_scripts"))

import torch
from compute_broker import lease_guard
from metal_kernelops import accelerator


def check(cpu, mps, *, atol=2e-6, rtol=2e-6):
    if isinstance(cpu, (tuple, list)):
        for a, b in zip(cpu, mps):
            check(a, b, atol=atol, rtol=rtol)
    else:
        torch.testing.assert_close(mps.cpu(), cpu, atol=atol, rtol=rtol, equal_nan=True)


def run():
    torch.manual_seed(82)
    before = accelerator.coverage()
    assert before == {}
    coverage = accelerator.install()
    assert coverage == accelerator.install()
    checks = []
    x = torch.randn(8, 6, requires_grad=True)
    xm = x.detach().to("mps").requires_grad_()
    index = torch.tensor([[-1, 0, 7], [2, 2, 9]])
    check(torch.take(x.t(), index), torch.take(xm.t(), index.to("mps")))
    torch.take(x, index).sum().backward()
    torch.take(xm, index.to("mps")).sum().backward()
    check(x.grad, xm.grad)
    checks.append("take negative/multidimensional/noncontiguous/repeated-index backward")
    for accumulate in (False, True):
        src = torch.tensor([2., 3., 4.])
        idx = torch.tensor([-1, 2, 7])
        a = torch.zeros(3, 4).t()
        b = a.to("mps")
        a.put_(idx, src, accumulate=accumulate)
        b.put_(idx.to("mps"), src.to("mps"), accumulate=accumulate)
        check(a, b)
    checks.append("put noncontiguous/negative/accumulate")
    a = torch.randn(2, 8, 9, 7, requires_grad=True)
    b = a.detach().to("mps").requires_grad_()
    ca, cb = torch.channel_shuffle(a, 4), torch.channel_shuffle(b, 4)
    check(ca, cb)
    ca.square().mean().backward()
    cb.square().mean().backward()
    check(a.grad, b.grad)
    checks.append("channel shuffle forward/backward")
    v = torch.tensor([-float("inf"), -1., -0., 0., 1., float("inf"), float("nan")])
    check(torch.heaviside(v, torch.tensor(.25)), torch.heaviside(v.to("mps"), torch.tensor(.25, device="mps")))
    for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.int64):
        for steps in (0, 1, 17):
            check(torch.logspace(-2, 2, steps, dtype=dtype), torch.logspace(-2, 2, steps, dtype=dtype, device="mps"), atol=.02, rtol=.01)
    checks.append("heaviside boundaries and logspace steps/dtypes")
    for shape in ((7, 5), (4, 5, 3, 2)):
        a = torch.randn(shape)
        ma, va = torch.randn(5), torch.rand(5)
        mb, vb = ma.to("mps"), va.to("mps")
        check(torch.ops.aten.batch_norm_update_stats(a, ma, va, .1), torch.ops.aten.batch_norm_update_stats(a.to("mps"), mb, vb, .1))
        check((ma, va), (mb, vb))
    checks.append("batch normalization biased outputs/unbiased running stats")
    x = torch.randn(4, 10)
    dx = torch.randn_like(x)
    glu = torch.randn(4, 5)
    gx = torch.randn_like(x)
    gg = torch.randn_like(glu)
    dgg = torch.randn_like(glu)
    check(torch.ops.aten.glu_jvp(glu, x, dx, 1), torch.ops.aten.glu_jvp(glu.to("mps"), x.to("mps"), dx.to("mps"), 1))
    args = (gx, gg, x, dgg, dx)
    check(torch.ops.aten.glu_backward_jvp(*args, 1), torch.ops.aten.glu_backward_jvp(*(a.to("mps") for a in args), 1))
    checks.append("GLU JVP and backward JVP with arbitrary supplied primals")
    x, y = torch.randn(7, 9), torch.randn(9)
    check(torch.ops.aten._add_relu(x, y, alpha=.4), torch.ops.aten._add_relu(x.to("mps"), y.to("mps"), alpha=.4))
    for gelu in (False, True):
        a, b, c = torch.randn(9), torch.randn(7, 3), torch.randn(3, 9)
        check(torch.ops.aten._addmm_activation(a, b, c, alpha=.7, beta=.3, use_gelu=gelu), torch.ops.aten._addmm_activation(a.to("mps"), b.to("mps"), c.to("mps"), alpha=.7, beta=.3, use_gelu=gelu))
    check(torch.ops.aten._aminmax(x), torch.ops.aten._aminmax(x.to("mps")))
    checks.append("private add-ReLU/addmm-activation/aminmax exact schemas")
    x = torch.randn(1024, 1024, device="mps")
    for _ in range(4):
        torch.heaviside(x, x)
    torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(30):
        torch.heaviside(x, x)
    torch.mps.synchronize()
    timing = (time.perf_counter() - start) * 1e3 / 30
    result = {"torch": torch.__version__, "checks": checks, "coverage": coverage, "heaviside_ms": timing, "status": "pass"}
    (ROOT / "measured/core.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"checks": len(checks), "overlays": len(coverage), "status": "pass", "heaviside_ms": timing}))


if __name__ == "__main__":
    with lease_guard("mps-kernelops", tier=1, task_name="core-integration"):
        run()
