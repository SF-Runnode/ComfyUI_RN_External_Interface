# Billing Configuration Guide

计费配置文件说明，用于配置 ComfyUI_RN_External_Interface 节点的预估价格显示。

## 文件位置

```
config/billing_config.json
```

## Docker 挂载更新

```bash
docker run -v /path/to/billing_config.json:/app/custom_nodes/ComfyUI_RN_External_Interface/config/billing_config.json ...
```

## 配置结构

### 1. 计费类型 (billing_types)

定义支持的计费方式：

| 类型 | 说明 | 所需参数 |
|------|------|----------|
| `token` | 按 token 计费 | `input_price_per_1k`, `output_price_per_1k` |
| `per_use` | 按次计费（生成次数） | `price_per_use` |
| `per_second` | 按秒计费（视频/音频时长） | `price_per_second` |
| `per_second_with_conditions` | 按秒计费（按条件字段区分单价） | `billing_conditions`（可选 `price_per_second` 作为兜底） |
| `per_model` | 按模型计费（固定费用） | `price_per_model` |

### 2. 模型配置 (models)

每个模型是一个 key，配置示例：

```json
{
  "模型名称": {
    "billing_type": "计费类型",
    "价格参数": 数值
  }
}
```

#### 按秒计费 (per_second)

```json
{
  "sora-2": {
    "billing_type": "per_second",
    "price_per_second": 0.005
  }
}
```

**价格计算**：`duration × price_per_second`

#### 按秒计费（条件分价）(per_second_with_conditions)

用于同一个模型在不同 `mode` / `resolution` / 未来新增字段下单价不同的情况。

`billing_conditions` 是一个数组；每个元素里，除了 `price_per_second`（以及可选的 `multiplier`）等价格字段外，其余字段都视为“匹配条件”。当多个条件都能匹配时，会优先选择“匹配字段更多”的那条（更精确）。

推荐直接在每条条件里写 `price_per_second`（单价）；此时模型顶层的 `price_per_second` 可以省略。

如果想用倍率写法，也可以在模型顶层提供 `price_per_second` 作为基础单价，然后在条件里用 `multiplier` 做倍数调整。

```json
{
  "kling-v2-1-master": {
    "billing_type": "per_second_with_conditions",
    "billing_conditions": [
      {"mode": "std", "price_per_second": 0.001},
      {"mode": "pro", "price_per_second": 0.004}
    ]
  }
}
```

也可以用 `resolution` 作为条件（字段名需与节点 widget 名一致）：

```json
{
  "MiniMax-Hailuo-2.3": {
    "billing_type": "per_second_with_conditions",
    "billing_conditions": [
      {"resolution": "720P", "price_per_second": 0.001},
      {"resolution": "768P", "price_per_second": 0.0012},
      {"resolution": "1080P", "price_per_second": 0.0018}
    ]
  }
}
```

#### 按次计费 (per_use)

```json
{
  "midjourney": {
    "billing_type": "per_use",
    "price_per_use": 0.035
  }
}
```

**价格计算**：`1 × price_per_use`

#### 按 Token 计费 (token)

```json
{
  "gemini": {
    "billing_type": "token",
    "input_price_per_1k": 0.000125,
    "output_price_per_1k": 0.0005
  }
}
```

**价格计算**：基于输入/输出 token 数量

#### 按模型计费 (per_model)

```json
{
  "ollama": {
    "billing_type": "per_model",
    "price_per_model": 0
  }
}
```

**价格计算**：`1 × price_per_model`（通常为 0 表示免费）

### 3. 显示设置 (display_settings)

```json
{
  "display_settings": {
    "show_estimate_badge": true,      // 显示预估价格 badge
    "show_actual_price": true,        // 显示实际价格
    "show_workflow_total": true,     // 显示工作流总费用
    "currency": "USD",               // 货币单位
    "credits_conversion_rate": 211   // 积分兑换比例
  }
}
```

## 节点价格计算规则

### 普通节点

根据节点类型自动匹配模型配置，例如：
- `RunNode_sora2` → 查找 `sora-2` 模型
- `RunNode_kling_text2video` → 查找 `kling-*` 模型
- `RunNode_mj` → 查找 `midjourney` 模型

### 批量节点

批量运行节点会自动乘以数量：

| 节点 | 数量 |
|------|------|
| `RunNode_sora2_run_4` | ×4 |
| `RunNode_sora2_run_8` | ×8 |
| `RunNode_banana2_edit_run_4` | ×4 |
| `RunNode_banana2_edit_S2A_run_8` | ×8 |

## 价格显示格式

Badge 显示格式：`[图标] [价格][计费方式] [×数量]`

示例：
- `⏱️ $0.05 (10s)` - Sora 2（按时长计费）
- `⏱️ $0.10 (10s) ×4` - Sora 2 批量运行 4 次
- `📌 $0.035/use` - Midjourney
- `💰 $0.001/token` - Gemini token 计费

## 添加自定义模型

1. 找到对应的计费类型
2. 在 `models` 中添加新条目
3. 设置正确的计费参数

```json
{
  "my-custom-model": {
    "billing_type": "per_second",
    "price_per_second": 0.01
  }
}
```

## 注意事项

1. **价格单位**：所有价格均为 USD 美元
2. **精度**：建议使用 4 位小数精度
3. **默认模型**：如果节点选择的模型不在配置中，会尝试模糊匹配
4. **免费模型**：设置 `price_per_use: 0` 或 `price_per_model: 0` 表示免费
