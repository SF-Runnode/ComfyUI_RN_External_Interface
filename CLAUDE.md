# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**ComfyUI_RN_External_Interface** is a ComfyUI custom nodes extension (v1.21.2) that provides integrations with various AI media generation APIs including Sora, Kling, Midjourney, Suno, Gemini, Veo, Flux, Qwen, Vidu, MiniMax, Ollama, and more. It is published under `SF-Runnode` in the ComfyUI registry. There is no test suite.

The extension is placed under `ComfyUI/custom_nodes/` and loaded automatically when ComfyUI starts.

## Project Structure

```
ComfyUI_RN_External_Interface/
├── __init__.py              # Entry point: registers nodes, sets WEB_DIRECTORY, initializes AiHelper server
├── Tools.py                 # API client classes (Comfly_api_set, Comfly_LLm_API)
├── comfly_config.py         # Configuration loader (API keys, base URLs, billing config, model name mapping)
├── billing_engine.py        # Billing calculation engine with pluggable strategy pattern
├── billing_helpers.py       # Node billing integration helpers (price display, ContextVar node ID tracking)
├── utils.py                 # Core utilities: logging, ProgressBar, error formatting, pil2tensor/tensor2pil, video/audio adapters, asset bundle parsers
├── AiHelper.py              # HTTP route handlers and init_server() for registering /api/ endpoints
├── config/
│   ├── billing_config.json          # Model prices and billing rules
│   ├── billing_config_README.md     # Billing config documentation
│   ├── models_config.json           # Model name mappings (friendly ↔ API names)
│   └── ComfyUI_RN_External_Interface-config-mock.json  # Mock API config template
├── nodes/
│   ├── nodes_openai.py          # Sora, GPT Image nodes
│   ├── nodes_google.py          # Gemini, Veo, Nano Banana nodes
│   ├── nodes_kling.py           # Kling video nodes
│   ├── nodes_midjourney.py      # Midjourney nodes
│   ├── nodes_suno.py            # Suno music nodes
│   ├── nodes_vidu.py            # Vidu video nodes
│   ├── nodes_qwen.py            # Qwen image nodes
│   ├── nodes_MiniMax.py         # MiniMax video nodes
│   ├── nodes_bytedance.py       # ByteDance (Jimeng) nodes
│   ├── nodes_blackforestlabs.py # Flux nodes
│   ├── nodes_ollama.py          # Ollama connectivity nodes
│   └── nodes_xai.py             # Grok video nodes
└── web/js/
    ├── Comfly_BillingBadge.js   # Price badge display on nodes
    ├── Comfly_WorkflowBilling.js # Workflow total consumption panel
    ├── Comfly_mjstyle.js        # Midjourney style loader
    └── Comfly_OllamaNode.js     # Ollama node frontend
```

## Architecture

### Node Registration

`__init__.py` imports all node modules and registers them via `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`. New nodes must:
1. Be defined in the appropriate `nodes/nodes_<provider>.py` file
2. Be imported and registered in `__init__.py`

### Configuration System

- **API Config**: `comfly_config.get_config()` reads from `config/ComfyUI_RN_External_Interface-config.json` with environment variable overrides (`COMFLY_API_KEY`, `COMFLY_BASE_URL`, etc.)
- **`save_config()` is disabled** (raises `PermissionError`) — API keys must be set via env vars or config file manually
- **Model Name Mapping**: `config/models_config.json` provides `display_name_mapping` (UI → internal) and `api_name_mapping` (internal → API)
- **Billing Config**: `config/billing_config.json` with hot-reload support via `reload_billing_config()`

### Billing System

The billing engine (`billing_engine.py`) uses a **strategy pattern** for extensible pricing:

| Billing Type | Description | Required Config Keys |
|---|---|---|
| `token` | Per-token (input/output) | `input_price_per_1k`, `output_price_per_1k` |
| `per_use` | Fixed per generation | `price_per_use` |
| `per_second` | Duration-based | `price_per_second` |
| `per_second_with_conditions` | Duration-based with param matching | `price_per_second`, `billing_conditions` |
| `per_model` | Fixed per model call | `price_per_model` |

New billing strategies can be registered via:
```python
@BillingCalculator.register("custom_type")
def calc_custom(model_config, data, is_estimate):
    return calculated_price
```

**Credit conversion**: USD prices are multiplied by `211` to get credits (hardcoded in `billing_engine.py`, `billing_helpers.py`, and frontend JS).

### Logging & Progress

`utils.py` provides structured logging:
- `generate_request_id(task_type, provider)` — creates `rn_{provider}_{task_type}_{uuid}` IDs
- `log_prepare()`, `log_complete()`, `log_error()` — lifecycle logging
- `log_backend()`, `log_backend_exception()` — heartbeat/status logging
- `ProgressBar` class — UI progress tracking
- `sanitize_sensitive_network_info()` — redacts IPs, URLs, hosts from logs

### HTTP Routes

`AiHelper.py` registers these routes via `init_server(app)`:
- `/api/get_config` — API configuration
- `/api/billing_config` — Billing config with model name mappings
- `/api/model_mapping` — Model name mapping only
- `/lib/marked.min.js`, `/lib/purify.min.js` — Library serving
- `/mjstyle/{name}.json` — Midjourney style templates

### Node Lifecycle Pattern

Every generation node follows the same sequence (see any `nodes/nodes_*.py`):

1. `get_api_model_name(model)` — convert friendly display name to API name
2. `generate_request_id(task_type, provider)` — create unique `rn_*` ID
3. `log_prepare()` — log task start
4. `ProgressBar(request_id, ...)` — create progress tracker, call `set_generating()`
5. Validate inputs, check API key (fall back to `get_config().get('api_key')`)
6. Make HTTP request via `requests` library
7. `log_complete()` / `log_error()` — log outcome
8. Return ComfyUI tensors (use `pil2tensor()` for images, `ComflyVideoAdapter` for video)

### Batch Execution Nodes

Some providers (Sora2, Nano Banana) have "Group" + "Run_N" paired nodes:

- **Group nodes** (e.g., `RunNode_sora2_group`) split work into batches of N items
- **Run_N nodes** (e.g., `RunNode_sora2_run_4`) execute N parallel requests within a batch
- The Group node serializes data as JSON strings that Run_N nodes deserialize — this pattern is necessary because ComfyUI doesn't natively support batch iteration

### Disabling Nodes

Nodes are disabled by commenting out their registration in `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` in `__init__.py`. Never delete the class definition from the node file — keep it for potential re-enable.

## Key Conventions

- **Error handling**: Always use `format_runnode_error()` for API errors, `log_error()` for console logging, and `ProgressBar.error()` for UI
- **Video handling**: Use `ComflyVideoAdapter` for video I/O (supports URL and local paths)
- **Image handling**: Use `pil2tensor()` / `tensor2pil()` for ComfyUI tensor conversion
- **Category naming**: Node categories follow `RunNode/<Provider>` pattern
- **Widget callbacks**: Billing badges hook into widget callbacks to update prices dynamically

## Environment Variables

| Variable | Purpose |
|---|---|
| `COMFLY_API_KEY` | Main API key |
| `COMFLY_BASE_URL` | Main base URL |
| `COMFLY_MODEL`, `COMFLY_TEMPERATURE`, `COMFLY_MAX_TOKENS`, `COMFLY_TOP_P` | LLM defaults |
| `BILLING_CONFIG_PATH` | Custom billing config path |
| `RUNNODE_LOG_LEVEL` | Log level (default: INFO) |
| `RUNNODE_HEARTBEAT_LOG` | Enable heartbeat logging (default: true) |
| `RUNNODE_HEARTBEAT_INTERVAL_SEC` | Heartbeat interval (default: 15) |
| `RUNNODE_STREAMING_PROGRESS` | Enable streaming progress (default: true) |
| `SORA2_BASE_URL`, `SORA2_API_KEY` | Sora2-specific overrides |
| `RUNNODE_HIDE_URL_HOSTS` | Hide URL hosts in logs (default: true) |
