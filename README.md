# metal-kernelops

Native Metal kernels for PyTorch on Apple Silicon: operators that MPS either lacks or runs through a slow generic path.

The first kernel is Mamba's selective scan, ported from the CUDA-only `mamba_ssm` extension.

## Status

| Module | Operator | State |
|---|---|---|
| `metal_kernelops/scan.py` | Mamba selective scan (SSM), forward + analytic backward | working, bit-exact against the reference at `(1, 192, 65536)` |
| `metal_kernelops/reference.py` | Pure PyTorch step loop | reference for alignment checks |
| `metal_kernelops/accelerator.py` | MPS dispatch-key overlays: `take`, `channel_shuffle`, `put_` | working; each turns `NotImplementedError` into a native path |

Measured on HSRMamba, same weights, real slices: full-model forward + backward dropped from 16.09 s to 1.01 s (15.9x). Loss difference 0.0, gradients non-zero for 288/288 parameters, relative error of the global gradient norm 9.79e-8.

## Install

```bash
pip install torch>=2.1
pip install .
```

Python 3.9+. MPS needs macOS 12.3 or newer.

## Usage

```python
import torch
from metal_kernelops import selective_scan_fn, scan_mps, scan_reference

# u, dt: (batch, dim, length); A: (dim, dstate); B, C: (batch, groups, dstate, length)
out = selective_scan_fn(u, dt, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False)
```

`selective_scan_fn` has three routes behind it:

- `scan_mps(u, dt, A, B, C)` runs the Metal kernels. It requires tensors on `mps`, `float32`, contiguous, and returns `(batch, dim, length)`.
- `selective_scan(u, dt, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False)` is the drop-in replacement for `mamba_ssm`'s function of the same name. `delta_bias` and `delta_softplus` are applied before the scan, `D` and `z` afterwards, as pointwise torch ops.
- `scan_reference(u, dt, A, B, C)` runs the plain PyTorch loop for alignment checks.

Returns the scan output only. The last-state return value of the CUDA extension is not implemented here.

To take over MPS kernels process-wide, opt in explicitly:

```python
from metal_kernelops import accelerator

accelerator.install()   # raises if the running torch is not the checked version
accelerator.coverage()  # {"aten::take": {"replaced_existing": False, "kernel": "_take", ...}}
```

Importing `accelerator` changes nothing. `install()` registers the overlays for the life of the process; there is no uninstall. Each row in `coverage()` names the operator, whether an MPS kernel already existed, and the function that replaced it.

## Implementation routes, cheapest to most aggressive

`torch.mps.compile_shader(source: str)` compiles MSL at runtime. Concretely in torch 2.14.1 it takes a source string and nothing else: no compile options, no fast-math switch, no precompiled library handle. Each process pays the compile at first call, and each operator dispatch is a Python-level call.

The routes past it, in increasing order of effort and headroom:

| Route | What it buys | Cost |
|---|---|---|
| Precompiled `.metallib` inside an Objective-C++/C++ extension, dispatched as a PyTorch op | No runtime shader compile, `MTLComputePipelineState` reuse, direct `MTLCommandBuffer` control, custom argument buffers | Build step per macOS/Xcode target, C++/ObjC++ glue, wheel platform tagging |
| Metal Performance Shaders and MPSGraph primitives | Apple's own tuned and fused kernels for conv, matmul, normalization | Limited operator set; a scan or a custom backward cannot be expressed |
| Fusing across operators instead of one op at a time (conv1d + activation + scan in a single kernel) | Removes intermediate memory traffic, which dominates these workloads | Kernel complexity, shape specialization |
| Full custom backend behind a private dispatch key | Whole-graph control, autograd integration | Very large, and it duplicates what upstream may later add |

What this repository does: `compile_shader` first, because it is the only route that needs no build step and no Xcode project, and its measured speedup is already 15.9x. A precompiled metallib is the next step when shader compile time or dispatch overhead becomes measurable against kernel time.

One thing this build cannot do: `torch.mps` exposes no API to load an external metallib, so a precompiled route needs an extension module.

### What `torch.compile` already does on MPS

Inductor ships a Metal codegen here: `torch/_inductor/codegen/mps.py`, 1267 lines. Its own header calls it "an early prototype that shows that one can compile elementwise ops into a Metal shader". It is wired into device dispatch (`MetalScheduling` in `torch/_inductor/codegen/common.py:620`), compiles through `torch.mps.compile_shader` (`torch/_inductor/runtime/runtime_utils.py:214`), and has a C++ wrapper (`codegen/cpp_wrapper_mps.py`) plus a dedicated flex-attention template (`codegen/metal_flex_attention_template.py`, 1048 lines).

Measured on torch 2.14.1, fp32, 1024x4096 matrices, 20 iterations after warmup:

| Graph | eager ms | compiled ms | speedup |
|---|---|---|---|
| `(sin(a) + cos(b)) * 2 - tanh(a)` | 0.480 | 0.132 | 3.64x |
| `relu(a * b + a)` | 0.283 | 0.160 | 1.77x |
| `softmax(a, -1)` | 0.181 | 0.165 | 1.10x |
| `amax(a, -1)` | 0.095 | 0.074 | 1.28x |
| `layer_norm(a)` | 0.214 | 0.263 | 0.81x |
| `sum(a, -1)` | 0.049 | 0.078 | 0.63x |
| `a @ b.T` | 0.730 | 0.731 | 1.00x |
| `conv2d` | 0.072 | 0.101 | 0.71x |

Numerics hold: `sum` matches to a relative Frobenius error of 1.5e-7, `conv2d` and `softmax` bit-exactly.

So the compiler already fuses pointwise chains, and that is where it wins. It generates no gemm or conv kernels: those stay extern aten calls, which is why their speedup is 1.00x and 0.71x. Reductions come out correct but not faster than the tuned MPS kernels.

That defines this repository's work: overlays and kernels for operators the compiler cannot reach at all (the `NotImplementedError` list), fusions blocked by one unsupported operator, and scan-shaped work, which the SIMD scheduler has no parallel-scan support for. Rebuilding pointwise fusion would be duplicate work.

### The route that keeps model code untouched

The three routes above all assume the model calls a function this package provides. The most aggressive option that still runs ordinary PyTorch is the opposite: register kernels on the MPS dispatch key itself, so `model.to("mps")` picks them up with no code change at all.

That works from Python. A fragment library over `aten` may register a kernel for `MPS`, and it may also replace a kernel that already exists:

```python
from torch.library import Library

lib = Library("aten", "FRAGMENT")
lib.impl("take", take_mps, "MPS")        # take has no MPS kernel today
lib.impl("cumsum", cumsum_mps, "MPS")    # replaces the existing one
```

Verified on torch 2.14.1: the second line replaces the registered MPS kernel and only emits a one-time `UserWarning`; the same call on a CPU tensor still reaches the built-in kernel; gradients flow because the overlay body is written in aten ops that MPS already supports.

Shape of the accelerator, most aggressive first:

| Layer | Coverage | Notes |
|---|---|---|
| Overlay kernels for every candidate in `not_implemented/portable_candidates.txt`, written on top of `compile_shader` or a precompiled metallib | the 60 candidates plus any existing MPS kernel worth replacing | Process-global. Changes behaviour for every model in the process, so it must be an explicit import, not a side effect of installing the package |
| Decomposition registrations for ops that can be expressed in supported ops | most of the remaining 118 | No speed promise, but removes the CPU round trip and the hard error |
| `TorchDispatchMode` catch-all for the long tail | everything else | A Python frame per op, tens of microseconds. Last resort, useful to find what a model touches |

Limits worth stating in advance: the overlay is bound to a torch version and breaks loudly on schema changes; it cannot restore what has no kernel-level expression on Metal at all, such as the quantization stack and the oneDNN paths listed in `not_implemented/backend_specific.txt`; and `torch.compile` will not fuse any of it, because this build has no MPS codegen. Coverage has to be auditable, so the accelerator must expose the table of ops it actually took over.

## Vendored community implementations

These are git submodules under `vendor/`, kept read-only. Two rules: check whether an implementation already exists before writing one, and do not duplicate what a submodule covers.

| Repository | Operator | Language |
|---|---|---|
| [mpsops/mps-flash-attention](https://github.com/mpsops/mps-flash-attention) | Flash attention, O(N) memory | Python + Metal |
| [mpsops/mps-linear-attention](https://github.com/mpsops/mps-linear-attention) | Flash linear attention (DeltaNet, Qwen3.5) | Objective-C++ |
| [mpsops/mps-deform-conv](https://github.com/mpsops/mps-deform-conv) | Deformable convolution 2D | Objective-C++ |
| [mpsops/mps-correlation](https://github.com/mpsops/mps-correlation) | Correlation layer for optical flow | Objective-C++ |
| [mpsops/mps-conv3d](https://github.com/mpsops/mps-conv3d) | 3D convolution | Python + Metal |
| [mpsops/mps-carafe](https://github.com/mpsops/mps-carafe) | CARAFE upsampling | Objective-C++ |
| [mpsops/mps-bitsandbytes](https://github.com/mpsops/mps-bitsandbytes) | 8-bit quantization | Python |

```bash
git clone --recurse-submodules <this repo>
# or, in an existing clone
git submodule update --init --recursive
```

## Operator gap

Counted on torch 2.14.1 (git `5c488690`). A kernel counts as present only when it is registered for the MPS dispatch key; a Composite implementation that reaches MPS through other operators counts as missing.

| Measure | overloads | base names |
|---|---|---|
| aten total | 3113 | 1589 |
| registered for MPS | 914 | 533 |
| missing for MPS | 2199 | 1230 |
| missing while CPU has a kernel | 186 | 133 (126 of them missing every overload) |

After merging in-place variants and private aliases, 118 base names remain. 54 of those are CPU-only by nature (quantization, oneDNN, nested and sparse layouts, RNG helpers). 60 are genuine candidates for a Metal kernel.

`not_implemented/` holds the lists and the counting rules.

## Verification standard

Every kernel is checked on real shapes, forward and backward, against `reference.py`:

- outputs finite and bit-exact where the arithmetic allows it;
- gradient coverage: which parameters get non-zero gradients, and why any do not;
- relative error of the global gradient norm around 1e-7;
- optimizer step actually moves the weights.

Slow-but-correct is not a pass. The reference implementation exists to prove the kernel, not to ship.

## Layout

```
src/metal_kernelops/   scan.py (Metal kernels), reference.py (pure PyTorch)
not_implemented/       operator gap lists and counting rules
vendor/                community implementations, git submodules
benchmarks/            timing scripts
measured/              alignment results, one JSON per operator
```

## License

MIT.
