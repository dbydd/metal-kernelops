# not_implemented: MPS operator gap catalog

Measured on torch 2.14.1 (git `5c488690`), macOS, `mps_available=True`.

## Counting rule

A kernel counts as present only when it is registered for the MPS dispatch key. A Composite implementation that reaches MPS through other operators counts as missing. The check is:

```python
torch._C._dispatch_has_kernel_for_dispatch_key(operator, "MPS")
```

| Measure | overloads | base names |
|---|---:|---:|
| aten total | 3113 | 1589 |
| registered for MPS | 914 | 533 |
| missing for MPS | 2199 | 1230 |
| missing while CPU has a kernel | 186 | 133 |

The 126 base names that are entirely missing from MPS merge to 118 after in-place variants, private aliases, and test entry points are folded back. Fifty-four are CPU-only by nature: quantization, oneDNN, nested and sparse-layout conversions, and RNG helpers. The remaining 60 are Metal candidates. Four private aliases, `_adaptive_avg_pool3d`, `_adaptive_avg_pool3d_backward`, `_linalg_eigvals`, and `native_channel_shuffle`, are covered through their public counterparts and therefore appear in neither list.

## Correct probe

The first version of this catalog called public functions that were not the target operator. It used `F.relu` for `_add_relu`, `linalg.qr` for `ormqr`, `matrix_exp` for `linalg_matrix_sqrth`, `linalg.solve` for LDL, dense `addmm` for `sspaddmm`, and `bicubic` for Lanczos. A result from those calls says nothing about the missing operator.

`probe_status.py` now:

- reads all 60 names from `portable_candidates.txt` and fails if the call table does not cover them exactly;
- resolves each name to its exact aten overload, including `_add_relu.Tensor`, `_slow_conv2d_backward.output_mask`, `adaptive_avg_pool3d_backward.grad_input`, and `frexp.Tensor`;
- creates sparse inputs as sparse tensors on MPS and never substitutes a dense tensor;
- creates every tensor on MPS with small, schema-valid shapes;
- reports `runs` or the first error line under the candidate name.

```bash
uv run python not_implemented/probe_status.py --accelerator disabled > not_implemented/mps_status.json
uv run python not_implemented/probe_status.py --accelerator enabled
uv run python not_implemented/probe_status.py --accelerator enabled --compatibility
```

`disabled` is the baseline. `enabled` inserts this checkout's `src` first and calls `metal_kernelops.accelerator.install()`.

## Baseline and accelerator

Baseline results, recorded in `mps_status.json`: 57 of 60 raise `NotImplementedError` or a related runtime error. Three run without an overlay:

- `linalg__powsum` through CompositeExplicitAutograd;
- `logit`, which has a native MPS kernel;
- `narrow_copy`, whose default overload reaches a composite.

With `accelerator.install()`, 58 of 60 run. The two exceptions are `linalg_ldl_factor_ex` and `linalg_ldl_solve`, whose exact Bunch-Kaufman factor is sequential and measured hundreds of times slower than a CPU call. They are available only through:

```python
from metal_kernelops import accelerator
accelerator.install(compatibility=True)
```

With that table, all 60 candidates run.

## Files

| File | Content |
|---|---|
| `portable_candidates.txt` | the 60 candidates, kept as the historical baseline so both probe runs remain comparable |
| `backend_specific.txt` | 54 entries that have no meaning on the Apple GPU, with the exclusion rule above |
| `cpu_only_base_names.txt` | all 118 merged base names |
| `probe_status.py` | exact-schema probe with optional accelerator install |
| `mps_status.json` | baseline output for the current torch version |

## Disposition

Implemented in `src/metal_kernelops/`:

- indexing: `take`, `put`, `channel_shuffle`, `narrow_copy.out`
- pooling: fractional 2D and 3D, adaptive 3D average and max
- resampling: `_upsample_lanczos2d_aa` and backward
- reductions: `mode`, `segment_reduce`, `_segment_reduce_backward`, `_pdist_forward`, `_pdist_backward`
- losses: both margin loss families and their backward passes
- attention: `_masked_softmax` and backward, packed QKV, native MHA, encoder layer, SDPA choice
- special functions: Airy Ai, Laguerre L, Legendre P, `log_ndtr`, `ndtri`, `frexp`
- arithmetic: `heaviside`, `logspace`, `_add_relu`, `_addmm_activation`, `_aminmax`, `rrelu_with_noise`
- derivatives and statistics: `glu_jvp`, `glu_backward_jvp`, `batch_norm_update_stats`
- convolution: slow 3D, dilated 2D and 3D, transposed 2D and 3D, slow 2D backward
- linear algebra: `ormqr`, `linalg_householder_product`, `linalg_matrix_sqrth`, `linalg_eig`, eigenvalue-only variants
- sparse: `_spdiags`, `sspaddmm`

Already native or composed, left unchanged: `logit`, `narrow_copy`, `linalg__powsum`.

Opt-in exact but slow: `linalg_ldl_factor_ex`, `linalg_ldl_solve`.

No candidate is silently dropped.
