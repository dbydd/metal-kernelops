# not_implemented/ — MPS 没有原生内核的算子目录

统计环境：torch 2.14.1（git `5c488690`），macOS，`mps_available=True`。

## 计数口径

只统计真正注册 MPS 内核的 overload，PyTorch 用 Composite 组合拼出来的实现不算已实现。判定用
`torch._C._dispatch_has_kernel_for_dispatch_key(op, "MPS")`。

| 口径 | overload | 基础算子名 |
|---|---|---|
| aten 全量 | 3113 | 1589 |
| MPS 有内核 | 914 | 533 |
| MPS 缺内核 | 2199 | 1230（1056 完全缺 + 174 部分缺） |
| CPU 有内核、MPS 完全没有 | 186 | 126 |

## 去重后的数字

126 个基础算子名按下面三条规则合并，结果是 **118**：

1. 并地（in-place）变体并回函数版本：`_add_relu_` 并入 `_add_relu`，`heaviside_` 并入 `heaviside` 等；
2. 纯测试入口剔除：`_foobar`、`_test_*`；
3. 私有别名并回公开名：`_adaptive_avg_pool3d` 并入 `adaptive_avg_pool3d`，`_linalg_eigvals` 并入 `linalg_eigvals`，`_adaptive_avg_pool3d_backward` 并入 `adaptive_avg_pool3d_backward`。

118 再按「这个算子是不是只对 CPU 后端有意义」筛一遍，剔除 54 个永远不会在 Apple GPU 上有意义的入口：量化栈（`quantize_per_tensor`、`quantized_lstm`、`fake_quantize_*`、`_dyn_quant_*`）、oneDNN（`mkldnn_rnn_layer`、`to_mkldnn`）、嵌套张量与稀疏格式转换、`_philox_*` 随机数辅助、`*_for_cpu` 变体。剩下的 **60** 个是真正可以写 Metal 内核的候选。

## 文件

| 文件 | 内容 | 行数 |
|---|---|---|
| `portable_candidates.txt` | 可以移植的候选算子，按字母序 | 60 |
| `backend_specific.txt` | 只对 CPU 后端有意义的入口，排除理由见上 | 54 |
| `cpu_only_base_names.txt` | 去重后的完整名单（上两者之和） | 118 |

## 怎么用

先看 `portable_candidates.txt`。挑一个之前，先确认三件事：

1. 该算子在 MPS 上是不是通过 Composite 走了别的路径、实际能跑（跑得慢不等于不能用）；
2. 目标模型里有没有真实调用（没有调用就不值得写内核）；
3. `vendor/` 下的社区实现有没有已经覆盖（重复实现没有意义）。

移植完成一个，就从 `portable_candidates.txt` 删掉，并在 `measured/` 写对账读数。
