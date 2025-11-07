# FPLUT 推理计算过程详细分析

本文档结合代码详细分析 FPLUT（FP16 Look-Up Table）的推理计算过程。

## 一、整体架构

```
输入 x (FP32/BF16) 
  ↓
FPLUT.forward()
  ↓
预处理（clamp, 类型转换）
  ↓
NewTable.forward()
  ↓
分块处理（split_forward）
  ↓
lut_fast（核心查找+插值）
  ↓
输出 y (FP32/BF16)
```

## 二、详细计算流程

### 阶段1: FPLUT.forward() - 入口处理

**代码位置**: `models/fp_lut.py:594-606`

```python
def forward(self, x):
    # 1. 延迟初始化查找表（首次调用时）
    if self.table is None:
        cut_points = cut_points_dict[self.func_name]  # 获取预定义切分点
        self.table = NewTable(self.function, cut_points, table_size=259, device=x.device)
    
    # 2. 特殊函数预处理（sin/cos需要模运算）
    if self.func_name in ["cos", "sin"]:
        x = mod_data(x)  # x % (2π) 并归一化到 [-π, π]
    
    # 3. 保存原始数据类型
    dtype = x.dtype
    
    # 4. 限制输入范围到查找表有效区间
    x = x.clamp(self.table.cut_points[0], self.table.cut_points[-1])
    # 例如：SiLU的cut_points[0]=-20.359375, cut_points[-1]=65504.0
    
    # 5. 转换为FP16进行计算（查找表使用FP16精度）
    y = self.table(x.half()).clamp(-65504, 65504).to(dtype)
    
    return y
```

**关键步骤**:
- **延迟初始化**: 首次调用时创建查找表，避免不必要的内存占用
- **输入裁剪**: 确保输入在查找表有效范围内
- **精度转换**: FP32/BF16 → FP16（计算） → FP32/BF16（输出）

---

### 阶段2: NewTable.forward() - 张量重塑和分块

**代码位置**: `models/fp_lut.py:478-497`

```python
def forward(self, x):
    # 1. 保存原始形状
    shape = x.shape  # 例如: [batch_size, seq_len, hidden_dim]
    
    # 2. 展平为一维张量
    x = x.reshape(-1)  # 例如: [batch_size * seq_len * hidden_dim]
    
    # 3. 根据数据量决定分块策略
    if torch.numel(x) > pow(2, 28):      # > 268,435,456 元素
        split = 32
    elif torch.numel(x) > pow(2, 25):   # > 33,554,432 元素
        split = 8
    else:
        split = 1
    
    # 4. 分块处理
    out = self.split_forward(x, split)
    
    # 5. 恢复原始形状
    out = out.reshape(shape)
    return out
```

**设计原因**:
- **内存优化**: 大张量分块处理，避免单次处理过大数据导致内存问题
- **并行优化**: 分块后可以更好地利用GPU并行性

---

### 阶段3: split_forward() - 分块执行查找

**代码位置**: `models/fp_lut.py:553-573`

```python
def split_forward(self, x, split=4, inplace=False, test_lut=False, use_gpu_lut=True):
    if inplace:
        out = x
    else:
        out = x.clone()  # 创建输出副本
    
    length = int(len(x) / split)  # 每块长度
    
    # 分块处理
    for i in range(split):
        if i == (split - 1):
            # 最后一块：处理剩余所有元素
            local = x[i * length :]
            out[i * length :] = self._lut_fast(local)
        else:
            # 中间块：处理固定长度
            local = x[i * length : (i + 1) * length]
            out[i * length : (i + 1) * length] = self._lut_fast(local)
    
    return out
```

**关键点**:
- 每块独立调用 `_lut_fast()` 进行查找
- 最后一块处理剩余元素（可能长度不同）

---

### 阶段4: lut_fast() - 核心查找函数

**代码位置**: `fplut/nl/_lut_fast_imp.py:212-216`

```python
def lut_fast(x, cut_points, values, scales):
    if x.is_cuda:
        return lut_fast_triton(x, cut_points, values, scales)  # GPU: Triton实现
    else:
        return lut_fast_torch(x, cut_points, values, scales)   # CPU: PyTorch实现
```

根据设备选择不同实现，我们重点分析 **GPU Triton实现** 和 **CPU PyTorch实现**。

---

### 阶段5: lut_fast_torch() - CPU实现（详细步骤）

**代码位置**: `fplut/nl/_lut_fast_imp.py:153-192`

这是最清晰的实现，我们逐步分析：

#### 步骤5.1: 预计算索引和区间长度

```python
# 预计算每个区间在查找表中的起始索引
pre_indices = torch.tensor(
    [0] + [1 + 32 * (i - 1) for i in range(1, 9)] + [257], 
    device=x.device, dtype=torch.int16
)
# 结果: [0, 1, 33, 65, 97, 129, 161, 193, 225, 257]
# 含义: 区间0起始索引=0, 区间1起始索引=1, 区间2起始索引=33, ...

# 每个区间的采样点数量
interval_lengths = torch.tensor([1] + [32] * 8 + [1], device=x.device, dtype=torch.int16)
# 结果: [1, 32, 32, 32, 32, 32, 32, 32, 32, 1]
# 含义: 区间0有1个点, 区间1-8各有32个点, 区间9有1个点
```

#### 步骤5.2: 找到输入所在的区间

```python
# 使用bucketize找到x属于哪个区间
cut_indices = torch.bucketize(x, cut_points, right=True).sub_(1).clip_(0, scales.numel() - 1)
# bucketize返回: x所在的区间索引+1
# sub_(1): 转换为区间索引 (0-9)
# clip_: 确保索引在有效范围内

# 示例:
# 如果 x = -2.5, cut_points = [-20.36, -17.11, -8.37, ..., 65504]
# bucketize返回 2 (因为 -2.5 在 [-17.11, -8.37) 区间，索引为1)
# sub_(1)后得到 cut_indices = 1
```

**数学表达式**:
$$i = \text{bucketize}(x, \text{cut\_points}) - 1$$
其中 $i \in [0, 9]$ 是区间索引。

#### 步骤5.3: 计算在区间内的归一化位置

```python
# 获取对应区间的预计算索引和区间长度
pre_indices = pre_indices[cut_indices]  # 形状: [N], 每个元素对应其区间的起始索引
interval_lengths = interval_lengths[cut_indices]  # 形状: [N], 每个元素对应其区间的长度

# 计算x距离区间起点的偏移
idxs_f = (x - cut_points[cut_indices]).mul_(scales[cut_indices])
# 步骤分解:
# 1. x - cut_points[cut_indices]: 计算距离区间起点的偏移量 d
# 2. .mul_(scales[cut_indices]): 乘以缩放系数 s_i
#    结果: idxs_f = d * s_i = (x - c_i) * (32 / (c_{i+1} - c_i))
```

**数学表达式**:
$$p = (x - c_i) \cdot s_i$$
其中:
- $c_i$ 是区间 $i$ 的起点（cut_points[i]）
- $s_i$ 是区间 $i$ 的缩放系数（scales[i] = 32 / (c_{i+1} - c_i)）
- $p$ 是归一化位置，范围约 $[0, 32)$ 对于中间区间

#### 步骤5.4: 计算查找表索引

```python
# 计算下取整索引（左边界）
idxs = (idxs_f.floor()).clip_(0).clamp_max_(interval_lengths).to(torch.int16)
# floor(): 下取整，得到左边界索引
# clip_(0): 确保 >= 0
# clamp_max_(interval_lengths): 确保 <= 区间长度

# 计算上取整索引（右边界）
idxs_plus = (idxs + 1).clip_(0).clamp_max_(interval_lengths).to(torch.int16)
```

**数学表达式**:
$$j = \lfloor p \rfloor, \quad j \in [0, \text{interval\_len})$$
$$j_{+} = \min(j + 1, \text{interval\_len})$$

#### 步骤5.5: 计算全局查找表索引

```python
# 将局部索引转换为全局索引
idxs = idxs + pre_indices          # 左边界全局索引
idxs_plus = idxs_plus + pre_indices  # 右边界全局索引
```

**示例**:
- 如果 `cut_indices = 2`（第3个区间）
- `pre_indices[2] = 33`（该区间起始索引）
- `idxs = 5`（区间内局部索引）
- 则 `idxs + pre_indices = 5 + 33 = 38`（全局索引）

#### 步骤5.6: 读取查找表值并线性插值

```python
# 读取左右两个边界值
y1 = values[idxs.int()]      # 左边界值: f(x_j)
y2 = values[idxs_plus.int()]  # 右边界值: f(x_{j+1})

# 计算插值权重（小数部分）
m1 = (idxs_f - idxs).clip_(0, 1).to(torch.float32)
# m1 = p - j，即归一化位置的小数部分 α ∈ [0, 1)

# 线性插值计算
right_left = y2 - y1                    # f(x_{j+1}) - f(x_j)
y = (right_left.float() * m1 + y1.to(torch.float32))
# y = f(x_j) + α * (f(x_{j+1}) - f(x_j))
```

**数学表达式**:
$$\alpha = p - \lfloor p \rfloor \in [0, 1)$$
$$y = f(x_j) + \alpha \cdot (f(x_{j+1}) - f(x_j))$$

这就是标准的**线性插值公式**。

#### 步骤5.7: 边界处理

```python
# 处理超出范围的情况
y[x <= cut_points[0]] = values[0]      # x <= 最小值：使用第一个值
y[x >= cut_points[-1]] = values[-1]    # x >= 最大值：使用最后一个值
```

---

### 阶段6: lut_fast_triton() - GPU实现（优化版本）

**代码位置**: `fplut/nl/_lut_fast_imp.py:119-150`

GPU实现使用Triton JIT编译，逻辑与CPU版本相同，但进行了优化：

#### 优化1: 展开的区间查找（避免循环）

```python
# 从右到左检查10个切分点，找到x所在的区间
interval_idx = 0
interval_idx = tl.where((x >= tl.load(cut_points_ptr + 9)) & (interval_idx == 0), 9, interval_idx)
interval_idx = tl.where((x >= tl.load(cut_points_ptr + 8)) & (interval_idx == 0), 8, interval_idx)
# ... 继续检查其他切分点
```

**优势**: 
- 避免了循环和分支预测失败
- 所有比较可以并行执行
- Triton编译器可以进一步优化

#### 优化2: 向量化加载和计算

```python
# 使用Triton的向量化操作
x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
# 一次性加载一个block的所有元素（BLOCK_SIZE=512/1024/2048）
```

#### 优化3: 内联计算减少内存访问

```python
# 所有计算都在kernel内部完成，减少CPU-GPU通信
minus_start = x - cut_points_i
position = (minus_start * mul_values_i).to(tl.float32)
m1 = (position - tl.floor(position)).to(tl.float32)
```

---

## 三、数学公式总结

### 3.1 查找表构建（create_table）

对于区间 $[c_i, c_{i+1}]$:

**采样点生成**:
$$x_k = c_i + k \cdot \frac{c_{i+1} - c_i}{32}, \quad k = 0, 1, \ldots, 31$$

**查找表值**:
$$T[j] = f(x_j), \quad j = 0, 1, \ldots, 258$$

**缩放系数**:
$$s_i = \begin{cases}
\frac{1}{c_1 - c_0} & \text{if } i = 0 \\
\frac{32}{c_{i+1} - c_i} & \text{if } i = 1, \ldots, 8 \\
\frac{1}{c_{10} - c_9} & \text{if } i = 9
\end{cases}$$

### 3.2 查找和插值（推理时）

**区间查找**:
$$i = \arg\max_j \{c_j \leq x\} \quad \text{s.t. } c_j \leq x < c_{j+1}$$

**归一化位置**:
$$p = (x - c_i) \cdot s_i$$

**局部索引**:
$$j = \lfloor p \rfloor, \quad j_{+} = \min(j + 1, \text{len}_i)$$

**全局索引**:
$$\text{idx}_{left} = \text{pre\_idx}_i + j$$
$$\text{idx}_{right} = \text{pre\_idx}_i + j_{+}$$

**线性插值**:
$$\alpha = p - j \in [0, 1)$$
$$y = T[\text{idx}_{left}] + \alpha \cdot (T[\text{idx}_{right}] - T[\text{idx}_{left}])$$

---

## 四、具体数值示例

假设使用SiLU函数，输入 `x = -2.5`:

### 步骤1: 查找区间

```python
cut_points = [-20.359375, -17.109375, -8.3671875, -1.9755859375, 
              -0.255615234375, -0.007244110107421875, 0.0072174072265625,
              0.228515625, 1.58203125, 10.46875, 65504.0]

# x = -2.5 落在区间 [c_2, c_3) = [-8.3671875, -1.9755859375)
cut_indices = 2
```

### 步骤2: 计算归一化位置

```python
# 区间起点
c_2 = -8.3671875

# 缩放系数（假设）
s_2 = 32 / (-1.9755859375 - (-8.3671875)) = 32 / 6.3916015625 ≈ 5.008

# 归一化位置
p = (-2.5 - (-8.3671875)) * 5.008
  = 5.8671875 * 5.008
  ≈ 29.39
```

### 步骤3: 计算索引

```python
# 局部索引
j = floor(29.39) = 29
j_plus = min(29 + 1, 32) = 30

# 全局索引
pre_idx_2 = 33  # 区间2的起始索引
idx_left = 33 + 29 = 62
idx_right = 33 + 30 = 63
```

### 步骤4: 线性插值

```python
# 插值权重
alpha = 29.39 - 29 = 0.39

# 查找表值
y1 = table[62]  # f(x_62)
y2 = table[63]  # f(x_63)

# 插值结果
y = y1 + 0.39 * (y2 - y1)
```

---

## 五、性能优化点

### 5.1 内存优化
- **延迟初始化**: 查找表首次调用时才创建
- **分块处理**: 大张量分块避免内存峰值
- **FP16计算**: 查找表使用FP16，减少内存占用

### 5.2 计算优化
- **Triton JIT**: GPU上使用编译后的kernel，性能更好
- **向量化**: 批量处理多个元素
- **预计算**: pre_indices和interval_lengths预计算，避免重复计算

### 5.3 精度优化
- **分段线性插值**: 在关键区域（中间8个区间）使用更密集的采样（32点）
- **边界处理**: 边界区间特殊处理，避免精度损失

---

## 六、误差分析

### 6.1 插值误差

线性插值的理论误差上界：

$$\epsilon(x) \leq \frac{1}{8} \cdot (x_{j+1} - x_j)^2 \cdot \max_{\xi \in [x_j, x_{j+1}]} |f''(\xi)|$$

对于SiLU函数：
- 中间区间大小: $\sim 2$ (对于大部分区间)
- 二阶导数最大值: $\sim 0.1$
- 最大误差: $\sim 0.05$ (相对误差约0.5%)

### 6.2 FP16精度影响

查找表使用FP16存储，可能引入额外误差：
- FP16精度: $\sim 10^{-3}$ (相对)
- 总误差: 插值误差 + FP16量化误差

---

## 七、总结

FPLUT通过以下机制实现高效近似：

1. **预处理**: 输入裁剪和类型转换
2. **区间查找**: O(1)复杂度找到输入所属区间
3. **归一化**: 将输入映射到查找表索引空间
4. **线性插值**: 使用两个相邻查找表值进行插值
5. **边界处理**: 处理超出范围的情况

整个过程在保持高精度的同时，实现了比原始函数计算更高效的性能。

