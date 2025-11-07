# FPLUT 详解

## 什么是 FPLUT？

**FPLUT** 是 **Floating Point Look-Up Table（浮点查找表）** 的缩写，是一种用于加速激活函数计算的优化技术。

## 核心原理

### 1. 基本思想

FPLUT 将复杂的激活函数（如 SiLU、GELU）的计算替换为**预计算的查找表（LUT）**，通过**查表 + 线性插值**的方式近似原始函数，从而加速推理。

### 2. 工作流程

```
原始计算: f(x) = SiLU(x)  ← 需要复杂的数学运算
         ↓
FPLUT方法:
1. 预计算查找表：在关键切分点（cut_points）处计算函数值
2. 输入 x → 找到 x 所在的区间 [cut_i, cut_i+1]
3. 查表获取区间端点的函数值
4. 线性插值得到最终结果
```

### 3. 关键组件

#### `NewTable` 类
- **作用**：构建和管理查找表
- **切分点（cut_points）**：预定义的 11 个关键点，将函数定义域分成 10 个区间
- **表大小**：259 个条目（table_size=259）
- **插值方式**：线性插值

#### `FPLUT` 类
- **作用**：包装原始激活函数，提供 LUT 版本的 forward
- **初始化**：接收原始函数和函数名称（如 "silu", "gelu"）
- **运行时**：首次调用时构建查找表，后续调用直接查表

### 4. 支持的函数

FPLUT 支持多种激活函数和数学函数：

- **激活函数**：`silu`, `gelu`, `sigmoid`, `tanh`, `relu`, `leaky_relu`, `elu`, `hardswish`, `mish`, `swish`
- **数学函数**：`exp`, `log`, `sin`, `cos`
- **其他**：`divide`, `inversesigmoid`, `pow_2.0`, `softplus`

### 5. 精度与性能权衡

- **精度**：通过精心选择的切分点和线性插值，可以达到接近原始函数的精度
- **性能**：查表操作比复杂数学运算快得多，特别是在 GPU 上
- **内存**：需要存储查找表（通常很小，259 个 float16 值）

## 代码示例

### 使用 FPLUT 替换 SiLU

```python
from models.fp_lut import FPLUT
import torch.nn as nn

# 原始激活函数
original_silu = nn.SiLU()

# 创建 FPLUT 版本
fplut_silu = FPLUT(function=original_silu, func_name="silu")

# 使用
x = torch.tensor([1.0, 2.0, 3.0])
y_original = original_silu(x)      # 精确计算
y_fplut = fplut_silu(x)            # LUT 近似
```

### 在模型中使用

```python
# 替换模型中的激活函数
for layer in model.model.layers:
    layer.mlp.act_fn = FPLUT(func_name="silu", function=layer.mlp.act_fn)
```

## FPLUT vs NN-LUT

| 特性 | FPLUT | NN-LUT |
|------|-------|--------|
| **全称** | Floating Point LUT | Neural Network LUT |
| **查找表生成方式** | 预定义切分点 + 函数值采样 | 神经网络学习得到 |
| **切分点** | 固定（11个切分点） | 可学习（cut_points） |
| **插值方式** | 线性插值 | 分段线性（slopes + biases） |
| **精度** | 较高 | 通常更高（可学习） |
| **灵活性** | 较低（固定切分点） | 较高（可优化） |
| **使用场景** | 快速实现，固定函数 | 需要高精度，可训练优化 |

## 技术细节

### 1. 切分点选择

每种函数都有预定义的切分点（`cut_points_dict`），这些切分点基于：
- 函数的曲率变化
- 数值精度要求
- 计算效率考虑

例如 SiLU 的切分点：
```python
"silu": [
    -20.359375,   # 左边界
    -17.109375,
    -8.3671875,
    -1.9755859375,
    -0.255615234375,
    -0.007244110107421875,
    0.0072174072265625,
    0.228515625,
    1.58203125,
    10.46875,
    65504.0,      # 右边界
]
```

### 2. 查找过程

1. **区间定位**：使用 `torch.bucketize` 找到输入值所在的区间
2. **归一化**：将输入值映射到区间内的局部坐标
3. **查表**：获取区间端点的函数值
4. **插值**：线性插值计算最终结果

### 3. FP16 优化

- 使用 FP16（半精度）进行计算，提高性能
- 使用 Triton 库进行高效的 FP16 运算（`float16_add_triton`, `float16_mul_triton`）

## 在项目中的使用

### 启用 FPLUT

```bash
# Zero-shot 评估
python eval.py --model_name Qwen2.5-0.5B-Instruct --use_fplut --batch_size 1

# PPL 评估
python eval_ppl.py --model_name Qwen2.5-0.5B-Instruct --use_fplut --seqlen 512
```

### 支持的激活函数

- **SiLU**：`func_name="silu"`（默认）
- **GELU**：`func_name="gelu"`（自动检测）
- 其他支持的函数见 `cut_points_dict`

## 优缺点

### ✅ 优点

1. **加速推理**：查表比复杂数学运算快
2. **内存友好**：查找表很小（259 个值）
3. **易于实现**：无需额外训练
4. **通用性强**：支持多种函数

### ⚠️ 缺点

1. **精度损失**：相比精确计算有微小误差
2. **固定切分点**：无法针对特定模型优化
3. **内存访问**：需要访问查找表（但通常很快）

## 总结

FPLUT 是一种**实用的加速技术**，通过**预计算查找表 + 线性插值**来近似激活函数，在保持较高精度的同时显著提升推理速度。它是 NN-LUT 的基础版本，适合快速部署和通用场景。

如果对精度要求更高，可以使用 **NN-LUT**，它通过学习的方式优化查找表，通常能达到更高的精度。


