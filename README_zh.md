# ComfyUI_RN_External_Interface

ComfyUI 自定义节点扩展，集成了 Sora、Kling、Midjourney、Suno、Gemini、Veo、Flux 等多种 AI 媒体生成 API。

**English Documentation**: [README.md](./README.md)

---

## 目录

- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [日志系统](#日志系统)
- [模型名称映射配置](#模型名称映射配置)
- [请求处理](#请求处理)
- [计费系统](#计费系统)
- [常见问题](#常见问题)

---

## 快速开始

### 安装

1. 克隆到 ComfyUI 的 `custom_nodes` 目录：
```bash
cd ComfyUI/custom_nodes
git clone <仓库地址> ComfyUI_RN_External_Interface
```

2. 安装依赖（如有）

3. 在 `config/ComfyUI_RN_External_Interface-config.json` 中配置 API 密钥

### 基本用法

1. 重启 ComfyUI
2. 在节点浏览器的 **RunNode/** 分类下找到节点
3. 连接节点构建 AI 媒体生成工作流

---

## 项目结构

```
ComfyUI_RN_External_Interface/
├── __init__.py           # 主入口，注册所有节点
├── AiHelper.py           # 辅助函数和工具
├── Tools.py              # API 客户端实现
├── billing_engine.py     # 计费引擎
├── billing_helpers.py    # 节点计费集成辅助
├── comfly_config.py      # 配置加载器
├── utils.py              # 核心工具（日志、进度等）
├── config/
│   ├── billing_config.json      # 价格配置
│   └── models_config.json       # 模型名称映射
├── nodes/
│   ├── nodes_openai.py          # Sora、GPT Image 节点
│   ├── nodes_google.py          # Gemini、Veo、Nano Banana 节点
│   ├── nodes_kling.py           # Kling 视频节点
│   ├── nodes_midjourney.py      # Midjourney 节点
│   ├── nodes_suno.py            # Suno 音乐节点
│   └── ...                      # 其他提供商节点
└── web/js/
    ├── Comfly_BillingBadge.js   # 价格标签显示
    └── Comfly_WorkflowBilling.js # 工作流总费用显示
```

---

## 日志系统

项目使用结构化日志系统追踪请求生命周期和调试。

### 日志函数

位于 [`utils.py`](./utils.py)：

| 函数 | 用途 |
|------|------|
| `log_prepare()` | 记录任务开始准备 |
| `log_complete()` | 记录任务完成 |
| `log_error()` | 记录错误详情 |
| `log_backend()` | 记录后端事件（心跳、任务状态） |
| `ProgressBar` | 在 ComfyUI UI 中显示进度 |

### 请求 ID 格式

每个请求都有唯一 ID：`rn_{provider}_{task_type}_{uuid}`

示例：`rn_openai_video_gen_a1b2c3d4`

### 日志输出示例

```
[RunNode] RunNode/OpenAI- [视频生成] rn_openai_video_gen_a1b2c3d4 准备中... model_version=sora-2
[RunNode] [任务信息] Task ID: task_12345
[RunNode] RunNode/OpenAI- [视频生成] rn_openvideo_gen_a1b2c3d4 完成。 elapsed_ms=15000
```

### 在代码中添加日志

```python
from .utils import generate_request_id, log_prepare, log_complete, log_error

def my_node_process(self, prompt, model):
    request_id = generate_request_id("my_task", "myprovider")
    log_prepare("我的任务", request_id, "RunNode/MyProvider-", "MyProvider", model_version=model)

    try:
        # 你的处理逻辑
        result = do_something()

        log_complete("我的任务", request_id, "RunNode/MyProvider-", "MyProvider")
        return result
    except Exception as e:
        log_error(str(e), request_id, "错误详情", "RunNode")
        raise
```

### 控制台输出示例

```
[RunNode] ▶️ [MyProvider] 我的任务 rn_myprovider_my_task_a1b2c3d4 准备中... model_version=sora-2
[RunNode] ✔️ [MyProvider] 我的任务 rn_myprovider_my_task_a1b2c3d4 完成。
```

---

## 模型名称映射配置

此功能允许在 ComfyUI 下拉菜单中显示**友好名称**，而在内部使用 **API 名称**进行实际 API 调用。

### 为什么需要映射？

- **界面友好**：用户看到 "Sora 2 Pro" 而不是 "sora-2-pro"
- **一致性**：同一模型在不同提供商可以有不同名称
- **灵活性**：可以随时更改 API 名称而不影响界面

### 配置文件

位置：[`config/models_config.json`](./config/models_config.json)

```json
{
  "display_name_mapping": {
    "sora-2": "Sora 2",
    "sora-2-pro": "Sora 2 Pro"
  },
  "api_name_mapping": {
    "Sora 2": "sora-2",
    "Sora 2 Pro": "sora-2-pro"
  }
}
```

### 工作原理

1. **用户看到**下拉菜单中的友好名称：`["Sora 2", "Sora 2 Pro"]`
2. **用户选择**：`Sora 2 Pro`
3. **节点函数**在调用 API 前转换回来：

```python
# 在节点的 process 函数中
model = get_api_model_name(model)  # "Sora 2 Pro" → "sora-2-pro"

# 现在 API 调用使用正确的内部名称
response = api.call(model="sora-2-pro", ...)
```

### 加载配置

在 [`comfly_config.py`](./comfly_config.py) 中：

```python
def load_models_config():
    """从配置文件加载模型名称映射"""
    config_path = os.path.join(os.path.dirname(__file__), "config", "models_config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_api_model_name(friendly_name):
    """将友好显示名称转换为 API 模型名称"""
    mapping = load_models_config().get("api_name_mapping", {})
    return mapping.get(friendly_name, friendly_name)  # 未找到时返回原值

def get_display_name(internal_name):
    """将内部名称转换为友好显示名称"""
    mapping = load_models_config().get("display_name_mapping", {})
    return mapping.get(internal_name, internal_name)
```

### 添加新的模型映射

1. 打开 [`config/models_config.json`](./config/models_config.json)
2. 添加到 `display_name_mapping`（用于界面下拉）：
   ```json
   "my-model": "我的模型显示名"
   ```
3. 添加到 `api_name_mapping`（用于 API 调用）：
   ```json
   "我的模型显示名": "my-model"
   ```

### 节点集成示例

```python
class Comfly_my_video_node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["我的模型", "我的模型 Pro"], {"default": "我的模型"}),
            },
            # ...
        }

    def process(self, model, ...):
        # 将友好名称转换为 API 名称
        api_model = get_api_model_name(model)

        # 使用 api_model 进行 API 调用
        response = api.call(model=api_model, ...)
```

---

## 请求处理

### 请求生命周期

```
用户点击"队列" →
  Node.process() 被调用 →
    log_prepare() →
      API 请求 →
        log_backend() (心跳/状态) →
          响应 →
            log_complete() →
              返回结果
```

### 进度追踪

`ProgressBar` 类在 ComfyUI UI 中提供实时进度更新：

```python
from .utils import ProgressBar

def process(self, prompt, model):
    request_id = generate_request_id("video_gen", "provider")

    # 创建进度条
    rn_pbar = ProgressBar(
        request_id,
        "提供商名称",
        extra_info=f"模型:{model}",
        streaming=True,
        task_type="视频生成",
        source="RunNode/Provider-"
    )

    # 更新进度
    rn_pbar.set_generating()  # 显示"生成中..."
    rn_pbar.update(50)       # 更新到 50%

    # 完成
    rn_pbar.done(char_count=1000, elapsed_ms=5000)
```

### 错误处理模式

```python
def process(self, prompt, model):
    request_id = generate_request_id("task", "provider")

    try:
        # 验证输入
        if not prompt:
            raise ValueError("提示词不能为空")

        # 发起 API 调用
        result = api.call(model=model, prompt=prompt)

        return result

    except Exception as e:
        # 记录错误
        log_error(
            message=str(e),
            request_id=request_id,
            detail="额外的错误上下文",
            source="RunNode/Provider",
            service_name="ProviderName"
        )

        # 在界面上显示错误
        rn_pbar.error(str(e))

        raise
```

### 安全 URL 日志

使用 `safe_public_url()` 防止日志泄露敏感信息：

```python
from .utils import safe_public_url

# 直接记录含 API 密钥的完整 URL（有风险！）
print(f"调用 API {baseurl}/v1/endpoint")

# 使用安全版本，会隐藏密钥
print(f"调用 API {safe_public_url(baseurl)}")
# 输出: https://api.example.com/v1/endpoint (密钥已隐藏)
```

---

## 计费系统

计费系统计算并显示每个节点的预估费用。

### 计费类型

| 类型 | 说明 | 计算方式 |
|------|------|----------|
| `per_second` | 按视频/音频时长 | `秒数 × price_per_second` |
| `per_use` | 按生成次数 | `1 × price_per_use` |
| `token` | 按 Token 数量 | `输入Token × 输入单价 + 输出Token × 输出单价` |
| `per_model` | 按模型固定费用 | `1 × price_per_model` |

### 配置说明

详细配置文档：[`config/billing_config_README.md`](./config/billing_config_README.md)

### 价格显示

节点显示预估价格标签：

```
Sora 2 ⏱️ ¥0.05/s      （按秒计费）
Midjourney 📌 $0.035/use  （按次计费）
Gemini 💰 $0.001/token   （Token 计费）
```

### 计费辅助函数

在 [`billing_helpers.py`](./billing_helpers.py) 中：

```python
from .billing_helpers import setup_node_billing, record_node_execution

# 在节点的 process 函数中
def process(self, prompt, model, ...):
    # 为此节点设置计费
    billing_data = setup_node_billing(
        node_id=self.id,
        node_type="RunNode_my_node",
        model_key=model,
        duration=10,  # 按秒计费时使用
        widgets=self.widgets
    )

    # ... 你的处理逻辑 ...

    # 完成后记录实际使用量
    record_node_execution(
        node_id=self.id,
        billing_data=billing_data,
        actual_duration=actual_time,
        actual_count=actual_generations,
        input_tokens=token_count,
        output_tokens=output_count
    )
```

---

## 配置文件位置

| 文件 | 用途 |
|------|------|
| [`config/ComfyUI_RN_External_Interface-config.json`](./config/) | API 密钥、基础 URL、提供商设置 |
| [`config/billing_config.json`](./config/billing_config.json) | 模型价格和计费规则 |
| [`config/models_config.json`](./config/models_config.json) | 模型名称映射（友好名称 ↔ API 名称） |

### 环境变量

也可以通过环境变量配置：

| 变量 | 说明 |
|------|------|
| `COMFLY_API_KEY` | 主 API 密钥 |
| `COMFLY_BASE_URL` | 主基础 URL |
| `BILLING_CONFIG_PATH` | 自定义计费配置路径 |
| `RUNNODE_HEARTBEAT_LOG` | 启用/禁用心跳日志 |

---

## 添加新节点

### 步骤 1：定义节点类

```python
# 在 nodes/nodes_myprovider.py 中
from ..comfly_config import *

class Comfly_my_node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["我的模型", "我的模型 Pro"], {"default": "我的模型"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "process"
    CATEGORY = "RunNode/MyProvider"

    def process(self, prompt, model, api_key=""):
        # 将友好名称转换为 API 名称
        model = get_api_model_name(model)

        # 你的处理逻辑
        result = do_something(model, prompt)

        return (result, "success")
```

### 步骤 2：注册节点

在 [`__init__.py`](./__init__.py) 中：

```python
from .nodes.nodes_myprovider import Comfly_my_node

NODE_CLASS_MAPPINGS = {
    "RunNode_my_node": Comfly_my_node,
    # ... 现有节点 ...
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunNode_my_node": "RunNode 我的节点",
    # ... 现有映射 ...
}
```

### 步骤 3：添加计费支持（可选）

```python
from .billing_helpers import setup_node_billing, record_node_execution

def process(self, prompt, model, ...):
    billing_data = setup_node_billing(
        node_id=getattr(self, 'id', 'unknown'),
        node_type="RunNode_my_node",
        model_key=model,
        widgets=self.widgets
    )

    # ... 处理逻辑 ...

    record_node_execution(
        node_id=getattr(self, 'id', 'unknown'),
        billing_data=billing_data
    )
```

---

## 常见问题

### 节点不显示价格

1. 检查 [`config/billing_config.json`](./config/billing_config.json) 存在且模型已配置
2. 确认 `display_settings` 中 `show_estimate_badge: true`
3. 查看浏览器控制台是否有 JavaScript 错误

### 模型下拉显示内部名称

1. 确保 [`config/models_config.json`](./config/models_config.json) 格式正确
2. 验证 `display_name_mapping` 有正确的条目
3. 配置修改后重启 ComfyUI

### API 错误

1. 检查配置文件中或环境变量中的 API 密钥
2. 验证基础 URL 是否正确
3. 开启调试日志：`export RUNNODE_HEARTBEAT_LOG=true`

---

## 相关文档

- [计费配置说明](./config/billing_config_README.md) - 详细计费配置指南
- [英文文档](./README.md) - English documentation

---

## 许可证

本项目按原样提供，用于 ComfyUI。
