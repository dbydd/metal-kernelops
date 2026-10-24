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
6. **Do not rebuild what Inductor already fuses.** This build does ship a Metal codegen: `torch/_inductor/codegen/mps.py`, 1267 lines, self-described as an early prototype for elementwise ops, wired in at `codegen/common.py:620` and compiled through `torch.mps.compile_shader`. Measured on 1024x4096 fp32: a pointwise chain is 3.64x faster compiled, softmax 1.10x, amax 1.28x, layer_norm 0.81x, matmul 1.00x, conv2d 0.71x. Pointwise fusion is therefore taken; work here targets operators that raise `NotImplementedError`, fusions blocked by one such operator, and scan-shaped work the SIMD scheduler cannot express.
7. **Verification standard.** Real shapes, forward and backward, against `reference.py`: outputs finite and bit-exact where arithmetic allows, gradient coverage explained, relative error of the global gradient norm around 1e-7, optimizer step moves the weights. Slow-but-correct is not a pass. Do not measure only the kernel; measure the model step it belongs to.
8. **Compare a change in the same process and batch.** This machine runs other GPU work, so absolute times move by a factor of two between runs. Keep the previous implementation importable, run old and new interleaved over several rounds, take the minimum per round, and record a same-shape `copy_` or single-launch floor measured in the same batch. A ratio from two separate runs is not evidence. State the load average next to the number.

## Current state

- `src/metal_kernelops/scan.py`: Mamba selective scan, 6 Metal kernels, analytic backward. Measured on HSRMamba: 16.09 s → 1.01 s per step, loss difference 0.0, 288/288 parameters with non-zero gradients, gradient-norm relative error 9.79e-8.
- `not_implemented/`: the gap lists. 2199 aten overloads have no MPS kernel; 1230 base names; 126 with a CPU kernel and no MPS coverage; 118 after deduplication; 60 portable candidates, 54 CPU-backend-only. The corrected exact-schema probe records 57 failures and three native composites at baseline. `install()` raises the supported count to 58; the last two candidates are LDL factor and solve, available only through `install(compatibility=True)`.
- `vendor/`: seven mpsops submodules, read-only.
- `src/metal_kernelops/accelerator.py`: one explicit install point for 110 MPS overloads across index operations, pooling, resampling, reductions, losses, special functions, attention, convolution aliases, sparse algebra, and linear algebra. It registers table entries at `MPS`, `SparseMPS` for `sspaddmm`, and `AutogradMPS` for the eigen backward. The compatibility table holds exact implementations that are slower than an explicit CPU call. `__init__.py` does not import the accelerator, so installation never changes process behaviour by accident.
- `benchmarks/`: one verifier per operator family, plus `verify_core.py` for capability-only decompositions and `verify_integration.py` for one mixed model step. `measured/` holds the recorded runs.
- `src/metal_kernelops/cv/`: tree-external CV operators with no dispatch registration: Mamba-2 chunk scan and decode update, causal conv, gated RMSNorm, VMamba path scan, NSA key-block selection, and RoIAlign. Each has a CPU reference, a verifier, and a `measured/` record; `verify_cv_integration.py` composes them into block-level steps.

## Writing

Documentation and commit messages state facts, decisions, and risks. Numbers carry their torch version. Do not describe a planned layer as if it exists.
