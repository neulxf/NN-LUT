# 简单对比测试脚本使用说明

## 脚本1: simple_comparison.sh

**用途**：对 Qwen2.5-0.5B-Instruct 进行三组对比测试

**测试内容**：
1. **Baseline**：使用标准激活函数（原始模型）
2. **NN-LUT**：使用神经网络学习的查找表
3. **FP-LUT**：使用预定义浮点查找表

**运行方式**：
```bash
bash simple_comparison.sh
```

**输出**：
- 结果保存在 `results/comparison_YYYYMMDD_HHMMSS/` 目录
- 三个日志文件：
  - `baseline.log` - Baseline 结果
  - `nnlut.log` - NN-LUT 结果
  - `fplut.log` - FP-LUT 结果

---

## 脚本2: test_gelu.sh

**用途**：测试 GELU 激活函数的模型（使用 Phi-2）

**测试内容**：
1. **Baseline**：Phi-2 原始模型
2. **NN-LUT + GELU**：使用 GELU LUT 的 NN-LUT
3. **FP-LUT + GELU**：自动检测 GELU 的 FP-LUT

**运行方式**：
```bash
bash test_gelu.sh
```

**输出**：
- 结果保存在 `results/gelu_test_YYYYMMDD_HHMMSS/` 目录
- 三个日志文件：
  - `baseline.log` - Baseline 结果
  - `nnlut_gelu.log` - NN-LUT + GELU 结果
  - `fplut_gelu.log` - FP-LUT + GELU 结果

**验证要点**：
- 检查日志中是否出现：`[INFO] Detected activation function: GELU`
- 检查是否使用正确的 GELU LUT：`[INFO] Using GELU LUT: ...`
- 对比三组结果的性能指标

---

## 快速运行

```bash
# 运行 Qwen2.5-0.5B-Instruct 对比
bash simple_comparison.sh

# 运行 GELU 测试（Phi-2）
bash test_gelu.sh
```

## 注意事项

1. **GELU 检测**：代码会自动检测 Phi-2 使用的 `gelu_new` 激活函数
2. **LUT 文件**：确保 `nnlut_bench/` 目录下有对应的 LUT 文件：
   - `lut_details_silu_H32_sub.json`
   - `lut_details_gelu_H32_sub.json`
   - `lut_details_exp_H32_sub.json`
3. **GPU 内存**：如果内存不足，可以修改 `batch_size`（当前为 1）


