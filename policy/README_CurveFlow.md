# CurveFlow CFM Canonical UNet Policy

这是基于 CurveFlow 的 CFM (Conditional Flow Matching) Canonical UNet Policy 实现，相比标准的 Flow Matching 方法，CurveFlow 提供了更平滑的生成轨迹和更好的边界条件控制。

## 主要特性

### 1. CurveFlow 轨迹参数化
- **标准 FM**: `x_t = (1-t)*x_0 + t*x_1` (线性插值)
- **CurveFlow**: `x_t = a_phi(t)*x_0 + b_psi(t)*epsilon` (非线性参数化)
  - `a_phi(t) = (1-t) + t(1-t)*delta*raw_a`
  - `b_psi(t) = t + t(1-t)*delta*raw_b`

### 2. 曲率正则化
- 添加曲率正则化损失: `L_curv = λ∫(a'*b'' - b'*a'')²dt`
- 确保生成轨迹的平滑性
- 减少模式崩塌问题

### 3. 改进的时间采样
- 使用 Beta(0.5, 0.5) 分布采样时间步
- 在轨迹两端更密集采样
- 提高边界条件的学习效果

## 文件结构

```
canonical_policy/policy/
├── cfm_curveflow_canonical_unet_policy.py  # 主要实现
├── example_curveflow_usage.py             # 使用示例
├── curveflow_config.yaml                   # 配置文件
└── README_CurveFlow.md                     # 本文档
```

## 核心组件

### 1. CurveFlowCoefficientNet
```python
class CurveFlowCoefficientNet(nn.Module):
    """神经网络直接参数化CurveFlow系数"""
    def __init__(self, hidden_dim=64, output_activation="tanh"):
        # 使用深度网络学习系数函数
```

### 2. CurveFlow
```python
class CurveFlow(nn.Module):
    """实现CurveFlow轨迹参数化"""
    def a_phi(self, t):
        """数据样本系数"""
        return (1 - t) + t * (1 - t) * (self.delta_scale * raw_a)
    
    def b_psi(self, t):
        """噪声样本系数"""
        return t + t * (1 - t) * (self.delta_scale * raw_b)
```

### 3. CurveFlowTrainer
```python
class CurveFlowTrainer:
    """计算曲率正则化损失"""
    def compute_curvature_loss(self):
        # 使用自动微分计算曲率
```

## 使用方法

### 1. 基本使用

```python
from canonical_policy.policy.cfm_curveflow_canonical_unet_policy import CFMCurveFlowCanonicalUNetPolicy

# 创建策略
policy = CFMCurveFlowCanonicalUNetPolicy(
    shape_meta=shape_meta,
    horizon=16,
    n_action_steps=8,
    n_obs_steps=2,
    # CurveFlow 参数
    curveflow_lambda=0.1,
    curveflow_hidden_dim=64,
    curveflow_delta_scale=0.5,
    curveflow_m_grid_points=100,
)
```

### 2. 训练

```python
# 计算损失（包含曲率正则化）
loss, loss_dict = policy.compute_loss(batch)

# 损失包含：
# - fm_loss: Flow Matching 损失
# - curvature_loss: 曲率正则化损失
# - total_loss: 总损失
```

### 3. 推理

```python
# 预测动作
result = policy.predict_action(obs_dict)
action = result['action']
```

## 参数配置

### 关键参数

| 参数 | 描述 | 推荐范围 | 默认值 |
|------|------|----------|--------|
| `curveflow_lambda` | 曲率正则化权重 | 0.01-0.5 | 0.1 |
| `curveflow_delta_scale` | 扰动幅度 | 0.1-1.0 | 0.5 |
| `curveflow_hidden_dim` | 网络隐藏层维度 | 32-128 | 64 |
| `curveflow_m_grid_points` | 曲率计算网格点数 | 50-200 | 100 |

### 参数调优指南

#### 1. curveflow_lambda (曲率正则化权重)
- **小值 (0.01-0.05)**: 轻微正则化，保持生成多样性
- **中值 (0.1-0.2)**: 平衡平滑性和多样性
- **大值 (0.3-0.5)**: 强正则化，生成更平滑的轨迹

#### 2. curveflow_delta_scale (扰动幅度)
- **小值 (0.1-0.3)**: 接近线性轨迹，训练稳定
- **中值 (0.4-0.6)**: 平衡线性和非线性
- **大值 (0.7-1.0)**: 强非线性，需要更多训练

#### 3. curveflow_hidden_dim (网络容量)
- **小值 (32-48)**: 简单函数，训练快
- **中值 (64-96)**: 平衡表达能力和效率
- **大值 (96-128)**: 复杂函数，需要更多数据

## 使用场景

### 1. 简单任务
```yaml
curveflow_lambda: 0.05
curveflow_delta_scale: 0.3
curveflow_hidden_dim: 48
```

### 2. 复杂任务
```yaml
curveflow_lambda: 0.2
curveflow_delta_scale: 0.7
curveflow_hidden_dim: 96
```

### 3. 平滑轨迹
```yaml
curveflow_lambda: 0.3
curveflow_delta_scale: 0.4
curveflow_hidden_dim: 64
```

### 4. 多样化轨迹
```yaml
curveflow_lambda: 0.05
curveflow_delta_scale: 0.8
curveflow_hidden_dim: 80
```

## 优势对比

| 特性 | 标准 Flow Matching | CurveFlow |
|------|-------------------|-----------|
| 轨迹参数化 | 线性插值 | 非线性参数化 |
| 正则化 | 无 | 曲率正则化 |
| 时间采样 | 均匀分布 | Beta分布 |
| 边界条件 | 基础控制 | 精确控制 |
| 平滑性 | 一般 | 更好 |
| 训练稳定性 | 一般 | 更好 |

## 注意事项

1. **计算开销**: CurveFlow 需要计算高阶导数，比标准 FM 稍慢
2. **参数敏感**: 需要仔细调优 `curveflow_lambda` 和 `curveflow_delta_scale`
3. **内存使用**: 曲率计算需要额外的内存存储梯度图
4. **训练时间**: 由于额外的正则化项，训练时间可能稍长

## 示例代码

完整的使用示例请参考 `example_curveflow_usage.py` 文件。

## 参考文献

- CurveFlow: 基于曲率正则化的 Flow Matching 方法
- 相关论文和实现细节请参考原始 CurveFlow 论文
