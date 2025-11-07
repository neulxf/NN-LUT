# 激活函数使用情况总结

## 检测结果

根据对配置中模型的检测，以下是激活函数使用情况：

### ✅ 使用 SiLU 的模型（大多数）

以下模型使用 **SiLU** 激活函数：

- **Qwen 系列**：Qwen2.5-7B-Instruct, Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct, Qwen2-1.5B-Instruct
- **Qwen3 系列**：Qwen3-8B
- **LLaMA 系列**：Llama-3.2-1B, Llama-3.2-3B, TinyLlama-1.1B, llama3-8b-hf
- **Mistral 系列**：Mistral-7B-v0.1

### ⚠️ 使用 GELU 的模型

以下模型使用 **GELU** 激活函数：

1. **Phi-2** (`microsoft/phi-2`)
   - 检测到：`gelu_new`（GELU 的一个变体）
   - ⚠️ **需要注意**：当前评估脚本会检测到 GELU，需要提供 `--lut_gelu_path` 参数

### ❓ 需要进一步确认的模型

以下模型由于权限或路径问题无法检测，但根据架构知识推测：

- **Gemma 系列** (google/gemma-2b, google/gemma-2b-it, google/gemma-7b)
  - 推测：**可能使用 GELU**（Google 的模型通常使用 GELU）
  - ⚠️ **建议**：如果使用 Gemma 模型，请提供 `--lut_gelu_path` 参数

- **StableLM 系列** (stabilityai/stablelm-3b-4e1t)
  - 需要实际检测确认

- **OLMo 系列** (allenai/OLMo-1B)
  - 需要实际检测确认

- **OpenELM 系列** (apple/OpenELM-1_1B, apple/OpenELM-3B)
  - 需要实际检测确认

- **Pythia 系列** (EleutherAI/pythia-*)
  - 需要实际检测确认

- **RedPajama 系列** (togethercomputer/RedPajama-INCITE-Base-3B-v1)
  - 基于 LLaMA 架构，推测使用 SiLU

- **SmolLM 系列** (HuggingFaceTB/SmolLM-*)
  - 需要实际检测确认

## 当前配置中的模型激活函数分布

基于检测结果和架构知识：

| 模型系列 | 激活函数 | 备注 |
|---------|---------|------|
| Qwen 系列 | SiLU | ✅ 已确认 |
| LLaMA 系列 | SiLU | ✅ 已确认 |
| Qwen3 系列 | SiLU | ✅ 已确认 |
| Mistral 系列 | SiLU | ✅ 已确认 |
| Phi-2 | GELU | ✅ 已确认（gelu_new） |
| Gemma 系列 | GELU | ⚠️ 推测（需要确认） |
| StableLM | ? | ❓ 未知 |
| OLMo | ? | ❓ 未知 |
| OpenELM | ? | ❓ 未知 |
| Pythia | ? | ❓ 未知 |
| RedPajama | SiLU | ⚠️ 推测（基于LLaMA） |
| SmolLM | ? | ❓ 未知 |

## 建议

1. **对于 Phi-2**：运行评估时请提供 `--lut_gelu_path` 参数
   ```bash
   python eval.py --model_name Phi-2 --use_nnlut \
       --lut_silu_path nnlut_bench/lut_details_silu_H32_sub.json \
       --lut_gelu_path nnlut_bench/lut_details_gelu_H32_sub.json \
       --lut_exp_path nnlut_bench/lut_details_exp_H32_sub.json \
       --batch_size 1
   ```

2. **对于 Gemma 系列**：如果使用，建议提供 `--lut_gelu_path` 参数

3. **代码已支持自动检测**：当前代码会自动检测激活函数类型，如果检测到 GELU 但没有提供 GELU LUT，会给出警告并使用 SiLU LUT 作为回退

4. **对于未知模型**：代码会默认使用 SiLU，如果实际是 GELU，结果可能不准确

## 验证方法

运行评估时，代码会输出检测到的激活函数类型：
```
[INFO] Detected activation function: GELU
```
或
```
[INFO] Detected activation function: SILU
```

如果看到警告：
```
[WARN] Model uses GELU but --lut_gelu_path not provided. Using SiLU LUT as fallback.
```
说明模型使用 GELU，但未提供 GELU LUT，建议提供正确的 LUT 文件。


