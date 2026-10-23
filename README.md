# metal-kernelops

Native Metal kernels for PyTorch on Apple Silicon: operators that MPS either lacks or runs through a slow generic path.

The first kernel is Mamba's selective scan, ported from the CUDA-only `mamba_ssm` extension.

## Status

| Module | Operator | State |
|---|---|---|
| `metal_kernelops/scan.py` | Mamba selective scan (SSM), forward + analytic backward | working, bit-exact against the reference at `(1, 192, 65536)` |
| `metal_kernelops/reference.py` | Pure PyTorch step loop | reference for alignment checks |

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

Two things this build cannot do: torch 2.14.1 on macOS ships no MPS codegen for `torch.compile` (`torch._inductor.codegen.mps` does not exist, `torch._inductor.config` has no MPS keys), so automatic fusion is not available; and `torch.mps` exposes no API to load an external metallib, so a precompiled route needs an extension module.

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
