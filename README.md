# metal-kernelops

Metal kernels and dispatch-key overlays for PyTorch on Apple Silicon. The package fills operators that MPS lacks or runs through a path that costs more than a native kernel.

The verified torch version is 2.14.1, which PyPI reported as the current release on 2026-10-24. Registration is checked against that exact version and raises on any other.

## Status

`metal_kernelops/scan.py` ships the original Mamba selective scan. Its forward matches a pure PyTorch reference to 4.8e-7 on a 2x8x64 grouped test, and all five inputs receive finite gradients. The earlier HSRMamba run recorded a step drop from 16.09 s to 1.01 s (15.9x) with a gradient-norm relative error of 9.79e-8; that run is not re-measured in `measured/`.

The MPS accelerator expands that idea to ordinary PyTorch calls. It registers kernels on the `aten` MPS dispatch key, so a model moved with `model.to("mps")` reaches them without a call-site change. The default table currently takes over 110 operator overloads across indexing, pooling, resampling, reductions, losses, special functions, attention, convolution aliases, sparse algebra, and linear algebra.

The separate `metal_kernelops.cv` package adds tree-external CV operators that torch does not define at all: Mamba-2 chunk scan and decode update, causal convolution, gated RMSNorm, VMamba-style path scan, NSA key-block selection, and RoIAlign. These are explicit imports, not dispatch registrations.

## Install

```bash
pip install torch==2.14.1
pip install .
```

Python 3.10 or newer. MPS needs macOS 12.3 or newer.

## Enable

```python
from metal_kernelops import accelerator

table = accelerator.install()          # fast, exact implementations only
table = accelerator.install(compatibility=True)  # also exact but slow fallbacks
```

Importing `accelerator` changes nothing. `install()` is process-global and has no uninstall. `coverage()` returns one row per operator and dispatch key with the previous kernel, the replacement, and whether the row came from the compatibility table.

The compatibility table exists for operators whose exact algorithm is strictly sequential and therefore slower than an explicit CPU call. `linalg_ldl_factor_ex` and `linalg_ldl_solve` are the current members. The default table contains kernels that remove a hard error with a useful device path or replace a measured bottleneck.

`install()` must be the first registration in the process. A second call returns the existing table and does not add compatibility entries.

## What the accelerator covers

Most routes below run their arithmetic entirely on MPS. `Metal` means a kernel compiled with `torch.mps.compile_shader` on first use. `aten` means a composition of operators that already have MPS kernels. `Native` means the existing MPS kernel is called with the missing schema translated to it. The matrix square root delegates to upstream `eigh`, which keeps the work on the device only inside its Metal region; the boundary is stated under Boundaries.

| Area | Module | Operator surface | Route |
|---|---|---|---|
| Indexing | `index_ops.py` | `take`, `take.out`, `channel_shuffle`, `put`, `put_`, `put.out`, `narrow_copy.out` | aten gather and scatter with negative-index and alias checks |
| Gaussian functions | `ndtri_ops.py`, `log_ndtr_ops.py` | `special_ndtri`, `special_ndtri.out`, `special_log_ndtr`, `special_log_ndtr.out` | Metal; Cephes inverse CDF, fused log CDF with Chebyshev and asymptotic tails |
| Float decomposition | `frexp_ops.py` | `frexp.Tensor`, `frexp.Tensor_out` | Metal over raw exponent bits, including subnormals |
| Airy and polynomials | `airy_ops.py`, `laguerre_ops.py`, `legendre_ops.py` | Airy Ai, Laguerre L, Legendre P, with default, scalar and `out` overloads | Metal; upstream recurrences with explicit FMA |
| Randomized activation | `rrelu_ops.py` | `rrelu_with_noise`, in-place and `out` | one Metal pass for noise, mask and output in training; native leaky ReLU in evaluation |
| Missing elementwise and fused operators | `decomposition_ops.py` | `heaviside`, `logspace.out`, `_add_relu` family, `_addmm_activation`, `_aminmax` family, `batch_norm_update_stats`, `glu_jvp`, `glu_backward_jvp` | aten compositions matching the CPU kernels, including denormal classification and out dtype contracts |
| Fractional pooling | `fractional_pool_ops.py` | 2D and 3D forward plus backward overloads | Metal forward with exact interval and NaN rules; scatter-add backward |
| Adaptive pooling | `adaptive_pool_ops.py` | 3D average and max, forward, backward, `out` and `grad_input` | Metal for ordinary windows, aten decomposition for wide windows |
| Lanczos resampling | `lanczos_ops.py` | `_upsample_lanczos2d_aa` and backward | Metal; Lanczos-3 with antialias scaling and a deterministic transpose backward |
| Reductions | `mode_ops.py`, `segment_ops.py` | `mode`, `mode.values`, `segment_reduce`, `_segment_reduce_backward` | Metal for short rows; sort/run-length and masked index reductions in aten |
| Pairwise distance | `distance_ops.py` | `_pdist_forward`, `_pdist_backward` | Metal; packed pair map, no N x N x D tensor |
| Margin losses | `margin_ops.py` | `multi_margin_loss` and `multilabel_margin_loss`, forward, backward and `out` | fused Metal for float32 batch rows; aten equations for the remaining dtypes |
| Attention | `attention_ops.py` | `_masked_softmax`, its backward, `_transform_bias_rescale_qkv`, `_native_multi_head_attention`, `_transformer_encoder_layer_fwd`, `_fused_sdp_choice` | one fused Metal kernel each for masked softmax and its backward, the packed-QKV split, the attention core with online softmax, and the whole encoder layer, with exact mask, packed-QKV and all-masked-row semantics; torch 2.14.1 does not route `nn.MultiheadAttention` or `nn.TransformerEncoderLayer` to these private ops on MPS |
| Convolution aliases | `convolution_ops.py` | `slow_conv3d_forward`, dilated 2D and 3D, transposed 2D and 3D, `_slow_conv2d_backward` | native `aten::convolution` and `convolution_backward`; no recursive dispatch |
| Householder | `householder_ops.py` | `ormqr`, `ormqr.out`, `linalg_householder_product` | Metal reflector application, also replaces a native MPS path that was seconds slow |
| Eigendecomposition | `eigen_ops.py` | `linalg_eig`, `linalg_eig.out`, `_linalg_eigvals`, `linalg_eigvals.out` | Metal Hessenberg plus shifted QR; custom backward on `AutogradMPS` because the built-in complex solve is unavailable on MPS |
| Matrix square root | `matrix_sqrt_ops.py` | `linalg_matrix_sqrth` | native `eigh` decomposition with CPU-identical clamping and validation |
| Sparse diagonals | `spdiags_ops.py` | `_spdiags`, `_spdiags.out` | Metal index construction plus SparseMPS construction and gather |
| Sparse addmm | `sparse_addmm_ops.py` | `sspaddmm`, `sspaddmm.out` | native sparse matrix multiplication with the CPU kernel's uncoalesced output structure |
| LDL (opt-in) | `ldl_ops.py` | `linalg_ldl_factor_ex.out`, `linalg_ldl_solve.out` | exact Bunch-Kaufman in Metal; compatibility table only |

The `AutogradMPS` rows are separate dispatch registrations. `linalg_eig` needs one because PyTorch's generated backward calls a complex LU solve that MPS cannot run, while the CPU and CUDA formulas stay untouched.

## Tree-external CV operators

`metal_kernelops.cv` implements operators from model families that ship their own CUDA extensions. Importing the package does not register a dispatch key, compile a shader, or allocate a tensor. Every function has a pure-PyTorch reference alongside it.

| API | Upstream family | Shape contract | Route |
|---|---|---|---|
| `mamba2_chunk_scan` | state-spaces/mamba, `mamba_chunk_scan_combined` | `x (B,L,H,P)`, `dt (B,L,H)`, `A (H,)`, `B/C (B,L,G,N)`, optional `D`, `z`, `dt_bias`, `initial_states (B,H,P,N)` | chunked SSD factorization; two Metal kernels compute the masked decay, the `<C,B>` block and the gated output, MPS matmuls carry the discretization and the chunk transfer; returns `(y, final_state)` |
| `mamba2_state_update` | state-spaces/mamba, `selective_state_update` | `state (B,H,P,N)`, `x (B,H,P)`, `dt (B,H)` or `(B,H,P)`, `A (H,)`, `B/C (B,G,N)` | one fused Metal step with analytic backward |
| `causal_conv1d_fn`, `causal_conv1d_update` | Dao-AILab/causal-conv1d | depthwise `x (B,C,L)`, weight `(C,1,K)` or `(C,K)`, state `(B,C,K-1)` | fused Metal forward, backward, and decode update with optional SiLU |
| `rmsnorm_gated` | state-spaces/mamba, `layernorm_gated.rms_norm_ref` | `x (...,D)`, optional gate `z`, `group_size` divides `D` | fused Metal RMSNorm with SiLU gate before or after normalization |
| `ss2d_scan` | VMamba SS2D | `u/dt (B,C,H,W)`, `A (C,N)`, `B/C (B,G,N,H,W)` | one Metal kernel walking the four VMamba traversal orders over the grid with one state per lane, writing each path at the pixel it stands on, then a four-way add in `(0+2)+(1+3)` order; `sum` or `mean` merge |
| `k_select`, `k_select_values` | DeepSeek NSA block selection | scores `(...,N)` on MPS, `k` candidates per row | Metal top-k with total order: higher score, then lower index, NaNs last |
| `roi_align` | torchvision `roi_align` | input `(N,C,H,W)`, boxes `(K,5)` or a list of `(L,4)` | differentiable bilinear sampling with `aligned`, `spatial_scale`, and exact or adaptive grids |

Measured on this machine, minimum of warmed synchronized rounds. Each ratio compares the current implementation against the one it replaced, timed in the same process and batch:

| Operator | Case | Result |
|---|---|---|
| `mamba2_chunk_scan` | 130M model, 1x2048, heads 24, head dim 64, state 128, chunk 32 | 1.80 ms, 1.14M tokens/s, 92.7x over a per-token recurrence |
| `mamba2_chunk_scan` | GQA, heads 32, 8 groups, chunk 32 | 4.82 ms, 43.2x over a per-token recurrence |
| `mamba2_state_update` | 1.3B decode, batch 1 | 17.6 us amortized forward, 109 us for forward and backward, 2.91x over the torch-op composition; gradients bit-reproducible |
| `causal_conv1d_fn` | 1x768x4096 fp32 | 0.041 ms forward, 0.149 ms forward and backward: 1.94x and 2.12x over the previous kernel |
| `causal_conv1d_update` | 64x2048 decode | 16.9 us, 15x over the ATen composition |
| `rmsnorm_gated` | 2048x32768, group 32768, bfloat16 | 2.61 ms forward, 8.47 ms step: 10.1x and 9.0x over the composed fallback this replaces |
| `ss2d_scan` | 1x96x56x56 | 2.61x forward and 1.13x backward over the previous stacked implementation |
| `ss2d_scan` | 2x64x32x32 | 2.43x forward and 1.23x backward over the previous stacked implementation |
| `k_select` | NSA shapes, k = 64 and 128 | 2.04x to 12.0x faster than the stable argsort it replaces, bit-exact ties, now for k up to 256 |
| `roi_align` | 8x256x56x56, 512 boxes, 7x7 output, ratio 2 | 0.30 ms forward against 408 ms for the composed MPS path and 29.7 ms for torchvision on CPU; backward 1.93 ms |

`benchmarks/verify_cv_integration.py` composes a Mamba-2 block (projection, causal conv, chunk scan, gated norm) and a path-scan block. It compares CPU float64 references with MPS forward and every parameter gradient, then takes an SGD step. Both blocks report all parameters moved and gradients within `2e-4`; K-select indices are bit-equal to the CPU reference.

Limits of this package:

- `mamba2_chunk_scan` fuses the decay, the `<C,B>` block, the carried-state term, the skip and the gate into two Metal kernels, and keeps the discretization, the chunk states and the chunk transfer on torch matmuls. The fused path runs only without autograd: with gradients enabled the function falls back to the composition, so backward is unaccelerated. The chunk transfer is `O(nchunks^2)`, so `chunk_size` must grow with sequence length. Packed sequences (`seq_idx`) and variable-length states are not implemented.
- `mamba2_state_update` uses scalar `A` per head, as Mamba-2 does. Its default output state is a fresh tensor; pass `out_state` for in-place cache updates. The backward is two dispatches with one writer per gradient element, so repeated runs give bit-identical gradients.
- `causal_conv1d` does not implement `seq_idx` boundaries. `causal_conv1d_update` is an inference path and rejects autograd.
- `rmsnorm_gated` implements the SiLU gate for groups up to `max(x.shape[-1], 16384)` in float32, float16 and bfloat16; groups wider than 16384 use a streaming kernel that walks the group instead of holding it in registers.
- `ss2d_scan` applies one `dt`, `A`, `B`, `C` to all four paths, and needs `states <= 32` because each state owns a lane. Per-path parameters require a direct `selective_scan` call with four times the channels and groups.
- `k_select` selects indices with no gradient; `k_select_values` is differentiable. Metal covers `k <= 256`; larger `k` uses a stable argsort. Building NSA's block scores is the caller's job.
- `roi_align` exists because torchvision 0.29.1 has no MPS kernel. It matches torchvision within `5.4e-7` of the data peak on the measured shapes, differentiates the input with a deterministic gather, and is 99x faster than torchvision's CPU kernel at 8x256x56x56 with 512 boxes.
- None of these functions appear in `accelerator.coverage()`; they are explicit imports from `metal_kernelops.cv`.

## Gap catalog

`not_implemented/portable_candidates.txt` lists 60 base operators that had a CPU kernel and no MPS kernel on torch 2.14.1. The old probe used unrelated public functions for several entries: `F.relu` for `_add_relu`, `linalg.qr` for `ormqr`, `matrix_exp` for `linalg_matrix_sqrth`, `linalg.solve` for LDL, dense `addmm` for `sspaddmm`, and `bicubic` for Lanczos. Those calls passed or failed for reasons unrelated to the target operator.

`not_implemented/probe_status.py` now calls the exact aten schema for every entry, constructs sparse inputs as sparse, and keeps all tensor arguments on MPS.

```bash
uv run python not_implemented/probe_status.py --accelerator disabled > not_implemented/mps_status.json
uv run python not_implemented/probe_status.py --accelerator enabled
```

On the baseline, 57 of the 60 candidates raise `NotImplementedError` or a related runtime error. Three run: `linalg__powsum`, `logit`, and `narrow_copy`. With the default accelerator, the supported set rises to 58 of 60. The two remaining candidates are the exact but sequential LDL factor and solve, available through `install(compatibility=True)`, which brings the set to 60 of 60.

The candidate file stays as the historical baseline so the two probe runs remain comparable. A new operator family gets its own module, a measured record under `measured/`, a verifier under `benchmarks/`, and one entry in `accelerator._MODULES`.

## Verification

Every operator family has a verifier under `benchmarks/` and a machine-readable record under `measured/`. The verifiers call ordinary PyTorch entry points, `torch.nn.functional`, or the exact private aten schema, with the accelerator installed.

The standard is:

- values against the CPU kernel on real shapes, with exact comparison where the arithmetic permits and stated tolerances where summation order differs;
- endpoints, empty tensors, non-contiguous layouts, `out` dtypes, and alias rules;
- gradients through the ordinary autograd graph, gradient-norm relative error against float64 or CPU, and at least one optimizer step that moves weights;
- timing after warmup and `torch.mps.synchronize()`, with a CPU baseline and an explicit note when the machine is contended.

`benchmarks/verify_core.py` exercises the capability-only operators and their gradients. `benchmarks/verify_integration.py` runs one model step through convolution, channel shuffle, RReLU, fractional pooling, Lanczos resizing, ndtri, and log-ndtr. Its recorded run reports a loss difference of 1.2e-7, a gradient-norm relative error of 6.7e-8, and non-zero movement for all four parameter tensors.

Representative measured results, using the minimum of warmed synchronized rounds. Each ratio names the implementation it is measured against. A ratio against "the previous kernel" was taken in one process with both implementations live in the same batch, because cross-run absolute times on this machine move by a factor of two with neighbouring GPU work:

| Operator | Shape or model step | Result |
|---|---|---|
| `special_airy_ai` | 65,536 to 8,388,608 float32 values | 13.3x to 66.2x over the CPU kernel; values above 26.5 return zero without entering the decaying branch, which is what the CPU kernel computes there |
| `special_log_ndtr` | 16M float32 values | 62.1x over CPU and 10.9x over the previous 14-pass composition; 0.363 ms to move 134 MB |
| `special_ndtri` | 16M float32 values | 1.9x to 3.2x over the previous one-element kernel, within 1.3x of a float4 copy of the same bytes |
| `frexp` | 16M values | 6.26x over CPU at 99% of a same-shape copy; the contract moves 12 bytes per element, so 1.5x over the previous kernel would need 719 GB/s against a measured 497 GB/s memcpy |
| `special_laguerre_polynomial_l` | 16.7M values, degree 128 | 1.83x over the previous kernel, 156x to 180x over CPU; exact reciprocal replaces the hardware divide |
| `special_legendre_polynomial_p` | 4.19M values, degree 32 | 4.19x over the previous kernel, 42.1x over CPU |
| `mode` | 4096 rows of 64 | 5.32x over the previous kernel in one thread per row; wider rows use a 64-lane threadgroup up to 256, and the sort path stays above that |
| `segment_reduce` | per-row, 1024x4096 | 7.6x forward and 5.2x backward over the previous composition, 25x to 36x for max |
| `_pdist_forward` / backward | 4096 rows, 256 columns | 1.6x to 1.8x forward, 1.8x to 2.5x backward over the previous kernels |
| adaptive 3D average pooling | (2,32,32,32,32) to 16 cubed | the fused average wins up to kernel 64: 5.4x to 7.7x over the decomposition; backward 3.6x to 6.7x |
| `_upsample_lanczos2d_aa` | 8x64x64x64 to 8x64x128x128 | 2.26x forward and 2.86x backward over the two-pass form, both interleaved against it |
| `multi_margin_loss` | 256 rows, 1000 classes | 2.13x over the previous kernel; float16 and bfloat16 run 4.2x to 50x over the composition they used to take |
| `_masked_softmax` and its backward | 4x8 heads, 128 tokens, float32 | 1.92x and 2.78x over the aten composite, 3 and 8 launches reduced to 1 |
| `_native_multi_head_attention` | 4x8 heads, 128 tokens, float32 | 1.40x plain, 1.70x masked, 1.25x with weights averaged, at 3 launches instead of 4 to 12 |
| `ormqr` | 512 squared, 8 right-hand sides | 2.9x over the sequential reflector kernel, 3.0x to 4.3x at 2048x512 |
| `linalg_householder_product` | 512 squared | 23026x over the native MPS kernel |
| `linalg_eig` | 64 matrices of size 64 | 6.9x over an explicit CPU round trip; a single matrix stays slower than CPU because one threadgroup holds one matrix |
| convolution aliases | 3D, dilated 2D and 3D, transposed | enqueue 1.4 to 2.4 us below `aten::convolution` because the alias calls `aten::_mps_convolution` directly, so the wrapper now costs less than the native entry point |

Small tensors lose to dispatch. The measured kernels generally need thousands of elements or a batched operation before MPS wins, and the verifiers report the crossover instead of hiding it. Absolute times on this machine move with concurrent GPU load, so the `measured/` files record both the median and the minimum and identify which one supports a ratio.

## Boundaries

- **Torch version.** Overlay schemas are checked against 2.14.1. A different version raises at install rather than registering mismatched kernels. Re-measure the whole table on each new torch; delete overlays that upstream has implemented.
- **Dtype.** MPS has no float64, so no overlay provides it. Most kernels support float32 and, where the operator and Metal allow it, float16 and bfloat16. Each module docstring states its exact set.
- **Upstream CPU fallbacks inside native kernels.** `matrix_sqrt_ops.py` uses MPS `eigh`. That kernel is Metal only while its Jacobi workspace fits 32 KiB and the batch is large enough; outside that region PyTorch moves the work to the CPU. The module docstring and `measured/matrix_sqrt_ops.json` identify the affected shapes.
- **Small-matrix linear algebra.** `linalg_eig` and `linalg_matrix_sqrth` remove a hard error, and their batched shapes beat an explicit CPU round trip. A single small matrix is slower than that round trip: `linalg_eig` is about 0.2x to 0.4x and `linalg_matrix_sqrth` about 0.5x to 0.95x below batch 8. The crossover is recorded in `measured/eigen.json` and `measured/matrix_sqrt_ops.json`.
- **Attention reachability.** torch 2.14.1 routes `nn.MultiheadAttention` and `nn.TransformerEncoderLayer` only to the CPU, CUDA, XPU, or PrivateUse1 fast paths. On MPS those modules use `scaled_dot_product_attention` directly and never call the six private attention overlays. The attention speedups in `measured/attention_ops.json` are measurements of the direct aten calls, not of the nn modules.
- **Deferred device errors.** An out-of-bounds index reaches MPS device-side, so `take` and `put_` may return before the error surfaces at the next synchronization. The accelerator does not add a host bounds read to hide this behavior.
- **Layout.** `channel_shuffle` returns a contiguous tensor even when the input uses channels-last. This costs one copy outside the identity case and is faster than preserving channels-last on this backend.
- **Private `_add_relu` edge case.** The overlay is add followed by ReLU. On float32, the CPU vector kernel maps NaN to zero and saturates infinities to the dtype maximum; the overlay lets NaN and infinity propagate. Finite results match.
- **Sparse layouts.** Sparse COO construction, `to_dense`, sparse matrix multiplication, `sspaddmm`, and `_spdiags` work on MPS. CSR and CSC construction do not exist on MPS in 2.14.1, so `_spdiags` rejects those layouts instead of returning a differently typed tensor.
- **Non-convergence.** `linalg_eig` raises a convergence error when its sweep budget is exhausted. It does not return a non-triangular factor.
- **Compiler interaction.** `torch.compile` with the MPS Inductor backend does not see these Python-level MPS kernels as fusions. Pointwise chains are still fused by Inductor; the overlays remove hard errors and cover scan-like or private operators outside its reach.

## Layout

```
src/metal_kernelops/    accelerator.py, scan.py, cv/ for tree-external operators
not_implemented/        baseline gap catalog, corrected probe, baseline results
benchmarks/             one verifier per family plus core and model integration
measured/               recorded runs, torch version, timings, and limits
vendor/                 seven community implementations, read-only
```

## Vendored implementations

The seven submodules under `vendor/` are kept for reference and reuse. None registers an `aten` kernel: they patch Python functions or register private namespaces such as `mfa`. No symbol or dispatch collision exists with this package.

| Repository | Scope |
|---|---|
| [mpsops/mps-flash-attention](https://github.com/mpsops/mps-flash-attention) | Flash attention and quantized KV cache |
| [mpsops/mps-linear-attention](https://github.com/mpsops/mps-linear-attention) | DeltaNet and Qwen linear attention |
| [mpsops/mps-deform-conv](https://github.com/mpsops/mps-deform-conv) | Deformable convolution 2D |
| [mpsops/mps-correlation](https://github.com/mpsops/mps-correlation) | Optical-flow correlation |
| [mpsops/mps-conv3d](https://github.com/mpsops/mps-conv3d) | 3D convolution |
| [mpsops/mps-carafe](https://github.com/mpsops/mps-carafe) | CARAFE upsampling |
| [mpsops/mps-bitsandbytes](https://github.com/mpsops/mps-bitsandbytes) | Quantization and 8-bit optimizers |

## License

MIT for this repository. Metal ports that follow ATen, Cephes, or SciPy algorithms carry the upstream notice in their module docstring.
