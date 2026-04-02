# ComfyUI_RN_External_Interface

A ComfyUI custom nodes extension that provides integrations with various AI media generation APIs including Sora, Kling, Midjourney, Suno, Gemini, Veo, Flux, and more.

**中文文档**: [README_zh.md](./README_zh.md)

---

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Logging System](#logging-system)
- [Model Name Mapping Configuration](#model-name-mapping-configuration)
- [Request Handling](#request-handling)
- [Billing System](#billing-system)

---

## Quick Start

### Installation

1. Clone into ComfyUI's `custom_nodes` directory:
```bash
cd ComfyUI/custom_nodes
git clone <repository-url> ComfyUI_RN_External_Interface
```

2. Install dependencies (if any)

3. Configure API keys in `config/ComfyUI_RN_External_Interface-config.json`

### Basic Usage

1. Restart ComfyUI
2. Find nodes under the **RunNode/** category in the node browser
3. Connect nodes to build workflows for AI media generation

---

## Project Structure

```
ComfyUI_RN_External_Interface/
├── __init__.py           # Main entry point, registers all nodes
├── AiHelper.py           # Helper functions and utilities
├── Tools.py              # API client implementations
├── billing_engine.py     # Billing calculation engine
├── billing_helpers.py    # Node billing integration helpers
├── comfly_config.py     # Configuration loader
├── utils.py              # Core utilities (logging, progress, etc.)
├── config/
│   ├── billing_config.json      # Price configuration
│   └── models_config.json       # Model name mapping
├── nodes/
│   ├── nodes_openai.py          # Sora, GPT Image nodes
│   ├── nodes_google.py          # Gemini, Veo, Nano Banana nodes
│   ├── nodes_kling.py          # Kling video nodes
│   ├── nodes_midjourney.py     # Midjourney nodes
│   ├── nodes_suno.py           # Suno music nodes
│   └── ...                     # Other provider nodes
└── web/js/
    ├── Comfly_BillingBadge.js  # Price badge display
    └── Comfly_WorkflowBilling.js # Workflow total display
```

---

## Logging System

The project uses a structured logging system to track request lifecycle and debugging.

### Log Functions

Located in `utils.py`:

| Function | Purpose |
|----------|---------|
| `log_prepare()` | Log when a task starts preparing |
| `log_complete()` | Log when a task completes |
| `log_error()` | Log errors with details |
| `log_backend()` | Log backend events (heartbeats, task info) |
| `ProgressBar` | Show progress in ComfyUI UI |

### Request ID Format

Every request gets a unique ID: `rn_{provider}_{task_type}_{uuid}`

Example: `rn_openai_video_gen_a1b2c3d4`

### Log Output Example

```
[RunNode] RunNode/OpenAI- [视频生成] rn_openai_video_gen_a1b2c3d4 Preparing... model_version=sora-2
[RunNode] [Task Info] Task ID: task_12345
[RunNode] RunNode/OpenAI- [视频生成] rn_openai_video_gen_a1b2c3d4 Completed. elapsed_ms=15000
```

### Adding Logs to Your Code

```python
from .utils import generate_request_id, log_prepare, log_complete, log_error

def my_node_process(self, prompt, model):
    request_id = generate_request_id("my_task", "myprovider")
    log_prepare("My Task", request_id, "RunNode/MyProvider-", "MyProvider", model_version=model)

    try:
        # Your processing logic
        result = do_something()

        log_complete("My Task", request_id, "RunNode/MyProvider-", "MyProvider")
        return result
    except Exception as e:
        log_error(str(e), request_id, "Error details here", "RunNode")
        raise
```

### Console Output

```
[RunNode] ▶️ [MyProvider] My Task rn_myprovider_my_task_a1b2c3d4 Preparing... model_version=sora-2
[RunNode] ✔️ [MyProvider] My Task rn_myprovider_my_task_a1b2c3d4 Completed.
```

---

## Model Name Mapping Configuration

This feature allows showing **friendly names** in the ComfyUI dropdown while using **API names** internally for actual API calls.

### Why Mapping?

- **UI-Friendly**: Users see "Sora 2 Pro" instead of "sora-2-pro"
- **Consistency**: Same model can have different names across providers
- **Flexibility**: Change API names without changing UI

### Configuration File

Location: `config/models_config.json`

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

### How It Works

1. **User sees** friendly name in dropdown: `["Sora 2", "Sora 2 Pro"]`
2. **User selects**: `Sora 2 Pro`
3. **Node function** converts back before API call:

```python
# In your node's process function
model = get_api_model_name(model)  # "Sora 2 Pro" → "sora-2-pro"

# Now API call uses the correct internal name
response = api.call(model="sora-2-pro", ...)
```

### Loading the Configuration

```python
# In comfly_config.py
def load_models_config():
    """Load model name mapping from config file"""
    config_path = os.path.join(os.path.dirname(__file__), "config", "models_config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_api_model_name(friendly_name):
    """Convert friendly display name to API model name"""
    mapping = load_models_config().get("api_name_mapping", {})
    return mapping.get(friendly_name, friendly_name)  # Returns original if not found

def get_display_name(internal_name):
    """Convert internal name to friendly display name"""
    mapping = load_models_config().get("display_name_mapping", {})
    return mapping.get(internal_name, internal_name)
```

### Adding a New Model Mapping

1. Open `config/models_config.json`
2. Add to `display_name_mapping` (for UI dropdown):
   ```json
   "my-model": "My Model Display Name"
   ```
3. Add to `api_name_mapping` (for API calls):
   ```json
   "My Model Display Name": "my-model"
   ```

### Node Integration Example

```python
class Comfly_my_video_node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["My Model", "My Model Pro"], {"default": "My Model"}),
            },
            ...
        }

    def process(self, model, ...):
        # Convert friendly name to API name
        api_model = get_api_model_name(model)

        # Use api_model for API calls
        response = api.call(model=api_model, ...)
```

---

## Request Handling

### Request Lifecycle

```
User clicks "Queue" →
  Node.process() called →
    log_prepare() →
      API Request →
        log_backend() (heartbeat/status) →
          Response →
            log_complete() →
              Return results
```

### Progress Tracking

The `ProgressBar` class provides real-time progress updates in the ComfyUI UI:

```python
from .utils import ProgressBar

def process(self, prompt, model):
    request_id = generate_request_id("video_gen", "provider")

    # Create progress bar
    rn_pbar = ProgressBar(
        request_id,
        "ProviderName",
        extra_info=f"模型:{model}",
        streaming=True,
        task_type="视频生成",
        source="RunNode/Provider-"
    )

    # Update progress
    rn_pbar.set_generating()  # Show "Generating..."
    rn_pbar.update(50)        # Update to 50%

    # Complete
    rn_pbar.done(char_count=1000, elapsed_ms=5000)
```

### Error Handling Pattern

```python
def process(self, prompt, model):
    request_id = generate_request_id("task", "provider")

    try:
        # Validate inputs
        if not prompt:
            raise ValueError("Prompt is required")

        # Make API call
        result = api.call(model=model, prompt=prompt)

        return result

    except Exception as e:
        # Log the error
        log_error(
            message=str(e),
            request_id=request_id,
            detail="Additional error context",
            source="RunNode/Provider",
            service_name="ProviderName"
        )

        # Show error in UI
        rn_pbar.error(str(e))

        raise
```

### Safe URL Logging

Use `safe_public_url()` to prevent logging sensitive information:

```python
from .utils import safe_public_url

# Instead of logging full URL with API key
print(f"Calling API at {baseurl}/v1/endpoint")  # Dangerous!

# Use safe version that redacts keys
print(f"Calling API at {safe_public_url(baseurl)}")  # Safe!
# Output: https://api.example.com/v1/endpoint (key redacted)
```

---

## Billing System

The billing system calculates and displays estimated costs for each node.

### Billing Types

| Type | Description | Calculation |
|------|-------------|-------------|
| `per_second` | Video/audio duration | `seconds × price_per_second` |
| `per_use` | Per generation | `1 × price_per_use` |
| `token` | Based on token count | `input_tokens × input_price + output_tokens × output_price` |
| `per_model` | Fixed per model | `1 × price_per_model` |

### Configuration

See `config/billing_config_README.md` for detailed billing configuration.

### Price Display

Nodes show estimated prices as badges:

```
Sora 2 ⏱️ ¥0.05/s      (per second billing)
Midjourney 📌 $0.035/use  (per use billing)
Gemini 💰 $0.001/token   (token billing)
```

### Billing Helper Functions

In `billing_helpers.py`:

```python
from .billing_helpers import setup_node_billing, record_node_execution

# In your node's process function
def process(self, prompt, model, ...):
    # Set up billing for this node
    billing_data = setup_node_billing(
        node_id=self.id,
        node_type="RunNode_my_node",
        model_key=model,
        duration=10,  # For per_second billing
        widgets=self.widgets
    )

    # ... your processing ...

    # Record actual usage after completion
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

## Configuration File Locations

| File | Purpose |
|------|---------|
| `config/ComfyUI_RN_External_Interface-config.json` | API keys, base URLs, provider settings |
| `config/billing_config.json` | Model prices and billing rules |
| `config/models_config.json` | Model name mappings (friendly ↔ API names) |

### Environment Variables

You can also configure via environment variables:

| Variable | Description |
|----------|-------------|
| `COMFLY_API_KEY` | Main API key |
| `COMFLY_BASE_URL` | Main base URL |
| `BILLING_CONFIG_PATH` | Path to custom billing config |
| `RUNNODE_HEARTBEAT_LOG` | Enable/disable heartbeat logging |

---

## Adding New Nodes

### Step 1: Define the Node Class

```python
# In nodes/nodes_myprovider.py
from ..comfly_config import *

class Comfly_my_node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["My Model", "My Model Pro"], {"default": "My Model"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "process"
    CATEGORY = "RunNode/MyProvider"

    def process(self, prompt, model, api_key=""):
        # Convert friendly name to API name
        model = get_api_model_name(model)

        # Your processing logic
        result = do_something(model, prompt)

        return (result, "success")
```

### Step 2: Register the Node

In `__init__.py`:

```python
from .nodes.nodes_myprovider import Comfly_my_node

NODE_CLASS_MAPPINGS = {
    "RunNode_my_node": Comfly_my_node,
    # ... existing nodes ...
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunNode_my_node": "RunNode My Node",
    # ... existing mappings ...
}
```

### Step 3: Add Billing Support (Optional)

```python
from .billing_helpers import setup_node_billing, record_node_execution

def process(self, prompt, model, ...):
    billing_data = setup_node_billing(
        node_id=getattr(self, 'id', 'unknown'),
        node_type="RunNode_my_node",
        model_key=model,
        widgets=self.widgets
    )

    # ... processing ...

    record_node_execution(
        node_id=getattr(self, 'id', 'unknown'),
        billing_data=billing_data
    )
```

---

## Troubleshooting

### Nodes Not Showing Prices

1. Check `config/billing_config.json` exists and has the model configured
2. Verify `show_estimate_badge: true` in display_settings
3. Check browser console for JavaScript errors

### Model Dropdown Shows Internal Names

1. Ensure `config/models_config.json` is properly formatted
2. Verify `display_name_mapping` has the correct entries
3. Restart ComfyUI after config changes

### API Errors

1. Check API keys in config file or environment variables
2. Verify base URLs are correct
3. Enable debug logging: `export RUNNODE_HEARTBEAT_LOG=true`

---

## License

This project is provided as-is for use with ComfyUI.
