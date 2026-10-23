# AGENTS.md — metal-kernelops

This is an independent repository. It has its own `.git`; it is not tracked by any parent workspace repository, and nothing here belongs in one.

## Decision: the target is an MPS dispatch-key accelerator

Settled 2026-10-24. The goal is not a bag of functions a model must call. The goal is: run ordinary PyTorch, move a model with `model.to("mps")`, and get fast kernels with no model-side change.

That is done by registering kernels on the MPS dispatch key from a fragment library over `aten`:

```python
from torch.library import Library
lib = Library("aten", "FRAGMENT")
lib.impl("take", take_mps, "MPS")       # fills a missing kernel
lib.impl("cumsum", cumsum_mps, "MPS")   # replaces an existing one
```

Verified on torch 2.14.1: an overlay kernel is reached by a plain `torch.take` on an mps tensor, the same call on a cpu tensor still reaches the built-in kernel, replacing an existing MPS kernel only warns once, and gradients flow because the overlay body is written in aten ops MPS already supports.

### What gets built, in order

| Layer | Scope | Cost |
|---|---|---|
| MPS overlay kernels, one per candidate in `not_implemented/portable_candidates.txt`, plus replacements for existing MPS kernels that are slower than a Metal kernel | the 60 candidates, then whatever profiling justifies | process-global; must be an explicit import |
| Decompositions registered for missing ops that can be expressed in supported ops | most of the remaining 118 | removes the CPU round trip and the hard error, no speed promise |
| `TorchDispatchMode` catch-all for the long tail | everything else | a Python frame per op; used to discover what a model touches |

A private dispatch key is rejected: it changes the device name and breaks every `device.type == "mps"` branch in the ecosystem. That is the opposite of this repo's purpose.

## Rules

1. **Track the latest stable torch.** Kernel gaps close with every release. Re-measure the whole table on each new torch; never patch the old numbers incrementally. When a new torch implements an op this repo overlays, delete the overlay and record why.
2. **Check `vendor/` before writing a kernel.** Seven community implementations are attached as submodules. A duplicate is wasted work. If a submodule covers the op, import it or point at it; do not retype the kernel.
3. **The overlay is opt-in and auditable.** Importing the package must not change process behaviour. The accelerator is a separate import, and it exposes the table of ops it actually took over: operator name, previous kernel, new kernel, torch version it was checked against.
4. **Pin and check the torch version.** Overlay registration depends on schemas. Verify at import that the running torch matches the version the overlays were validated on, and fail loudly when it does not.
5. **Kernels are `compile_shader` first, precompiled `.metallib` second.** `torch.mps.compile_shader(source: str)` takes a source string and nothing else in 2.14.1; it needs no build step. Move to a precompiled metallib inside an extension when shader compile or dispatch overhead becomes measurable against kernel time, not before.
6. **No promise of `torch.compile`.** This build has no MPS codegen: `torch._inductor.codegen.mps` does not exist and `torch._inductor.config` has no MPS keys. Fusion is manual, so an operator worth fusing gets one kernel that absorbs its neighbours.
7. **Verification standard.** Real shapes, forward and backward, against `reference.py`: outputs finite and bit-exact where arithmetic allows, gradient coverage explained, relative error of the global gradient norm around 1e-7, optimizer step moves the weights. Slow-but-correct is not a pass. Do not measure only the kernel; measure the model step it belongs to.

## Current state

- `src/metal_kernelops/scan.py`: Mamba selective scan, 6 Metal kernels, analytic backward. Measured on HSRMamba: 16.09 s → 1.01 s per step, loss difference 0.0, 288/288 parameters with non-zero gradients, gradient-norm relative error 9.79e-8.
- `not_implemented/`: the gap lists. 2199 aten overloads have no MPS kernel; 1230 base names; 126 with a CPU kernel and no MPS coverage; 118 after deduplication; 60 portable candidates, 54 CPU-backend-only.
- `vendor/`: seven mpsops submodules, read-only.
- No accelerator package yet. `__init__.py` exports the scan functions only.

## Writing

Documentation and commit messages state facts, decisions, and risks. Numbers carry their torch version. Do not describe a planned layer as if it exists.
