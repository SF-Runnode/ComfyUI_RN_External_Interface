---
name: logging-system
description: The structured logging and progress system in utils.py used by all backend nodes. Covers the RunNode logger, log lifecycle functions (log_prepare/log_complete/log_error/log_backend), ProgressBar class, error formatting (format_runnode_error), and sensitive info sanitization. Use when adding or modifying logging in any node file, debugging log output, changing progress reporting, or adjusting sanitization rules.
---

# Logging System (utils.py)

The centralized logging infrastructure in `utils.py` used by every node in `nodes/`. Provides structured console output, progress reporting to the ComfyUI frontend, error message formatting, and sanitization of sensitive data before it reaches logs or user-facing messages.

## Logger Setup

```python
_RN_LOGGER = logging.getLogger("RunNode")
_RN_LOGGER.setLevel(getattr(logging, _RN_LOG_LEVEL, logging.INFO))
_RN_LOGGER.propagate = False
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
```

Key points:
- Logger name is `"RunNode"` — accessible via `logging.getLogger("RunNode")`
- Output goes to **stderr**, not stdout (separates logs from data output)
- Format: `2026-05-09 12:34:56,789 | INFO | message here`
- Level controlled by `RUNNODE_LOG_LEVEL` env var (default `"INFO"`)
- `propagate = False` — logs only go to this handler, not to root logger

## Log Prefix Constants

Used as visual markers in console output:

| Constant | Value | Usage |
|---|---|---|
| `PREFIX` | `"✨"` | General/info logs |
| `ERROR_PREFIX` | `"✨-❌"` | Error messages |
| `PROCESS_PREFIX` | `"✨"` | Progress updates |
| `REQUEST_PREFIX` | `"✨"` | Request-level logs |
| `WARN_PREFIX` | `"✨-⚠️"` | Warnings (defined but used sparingly) |

These are prepended to `print()` calls, NOT to the `logging` module output.

## Key Logging Functions

### `generate_request_id(task_type, provider) -> str`
Generates a unique request ID: `rn_{provider}_{task_type}_{uuid[:8]}`
Example: `rn_openai_sora_abc123de`

### `log_prepare(task_name, request_id, prefix, service_name, model_version=None, speed=None, **kwargs)`
Called when a request is about to be sent. Prints to stdout:
```
✨ {prefix} [{task_name}] {request_id} Preparing... model_version=X speed=Y
```

### `log_complete(task_name, request_id, prefix, service_name, image_url=None, **kwargs)`
Called when a request finishes successfully:
```
✨ {prefix} [{task_name}] {request_id} Completed. image_url=...
```

### `log_error(message, request_id=None, detail=None, source="RunNode", service_name=None)`
Called on errors. Prints to stdout with ERROR_PREFIX:
```
✨-❌ [{source}] {service_name} Error: {message} - {detail}
```
Designed to be called with the **already-sanitized** detail string from `format_runnode_error()`.

### `log_backend(event_type, **kwargs)`
Heartbeat/status logging with **debounce**. Only prints if `RUNNODE_HEARTBEAT_LOG` env var is not `"false"` AND the interval since last heartbeat for the same `(event_type, request_id, task_id)` key exceeds `RUNNODE_HEARTBEAT_INTERVAL_SEC` (default 15s).

Output format:
```
✨ 💓 Heartbeat {event_type} request_id=... task_id=...
```

### `log_backend_exception(event_type, **kwargs)`
Placeholder for structured exception logging. If `error`/`exception` not in kwargs, it captures `traceback.format_exc()`. Currently does not persist to any external system.

---

## ProgressBar Class

Reports execution progress to the ComfyUI frontend via `PromptServer` messages.

### Constructor
```python
ProgressBar(request_id, service_name, extra_info="", streaming=True, task_type="Task", source="RunNode")
```

### Methods

| Method | Behavior |
|---|---|
| `update_absolute(value, message=None)` | Sends absolute progress %. Throttled to max once per 0.5s to avoid flooding. |
| `update(value, message=None)` | Alias for `update_absolute`. |
| `set_generating()` | Prints `"✨ Generating..."` if streaming is enabled. |
| `error(message)` | Prints error. Message is sanitized via `sanitize_sensitive_network_info()` before output. |
| `done(char_count=0, elapsed_ms=0)` | Prints `"✨ Done in {elapsed_ms}ms"`. |

### Usage Pattern in Nodes

```python
request_id = generate_request_id("text2video", "kling")
rn_pbar = ProgressBar(request_id, "Kling", task_type="text2video")
log_prepare("Kling Text2Video", request_id, REQUEST_PREFIX, "Kling", model_version="v1.6")

try:
    rn_pbar.set_generating()
    # ... API call with polling ...
    rn_pbar.update_absolute(50, "Processing...")
    # ... complete ...
    elapsed = int((time.time() - start) * 1000)
    rn_pbar.done(elapsed_ms=elapsed)
    log_complete("Kling Text2Video", request_id, REQUEST_PREFIX, "Kling")
except Exception as e:
    error_msg = format_runnode_error(e)   # sanitized for frontend
    rn_pbar.error(error_msg)               # shows in UI
    log_error("API call failed", request_id, error_msg, "Kling", "Kling")
```

---

## Error Formatting: `format_runnode_error(response, sanitize=True)`

Recursively extracts human-readable error messages from API responses. Handles three input types:

1. **requests.Response object**: calls `.json()`, falls back to `.text`
2. **dict**: extracts `status` for status code
3. **string**: tries to parse as JSON if it starts with `{`

### Extraction Priority (Recursive)

The function recursively descends into nested structures looking for these keys in order:
1. `"error"`
2. `"message"`
3. `"fail_reason"`
4. `"err_code"`
5. `"detail"`
6. `"base_resp"` → `"status_msg"` (nested inside base_resp)
7. `"status_msg"`

At each level, if a key's value is a JSON string, it's parsed and the recursion continues. This handles APIs that nest error info inside JSON-encoded string fields.

### Output Format
- With status code: `"API Error: {status_code} - {message}"`
- Without status code: just the message string

### Sanitization (default on)

When `sanitize=True`, the output is run through `sanitize_sensitive_network_info()`. The node pattern is:
- `format_runnode_error(e)` → sanitized → sent to **frontend** (ProgressBar.error)
- The **same sanitized string** goes to `log_error()` → printed to console

To show raw errors in console, nodes would need to call `format_runnode_error(e, sanitize=False)` for the log and `format_runnode_error(e, sanitize=True)` for the frontend. Currently most nodes use the default `sanitize=True` for both paths.

---

## Sanitization: `sanitize_sensitive_network_info(text)`

Redacts sensitive network details from strings before they reach logs or the frontend:

| Pattern | Replacement |
|---|---|
| `host='...'` | `host='<hidden-host>'` |
| `host="..."` | `host="<hidden-host>"` |
| `port=NNN` | `port=<hidden-port>` |
| IPv4 addresses | `<hidden-ip>` |
| IPv6 addresses | `<hidden-ipv6>` |
| `Connection to {host}` | `Connection to <hidden-host>` |
| URL authority `http(s)://host:port` | `http(s)://<hidden-host>` |

The URL host hiding is controlled by `RUNNODE_HIDE_URL_HOSTS` env var (default `"true"`). Set to `"false"` to see full URLs in logs.

### `safe_public_url(url)`
Strips `user:password@` from URLs before display. Uses `urllib.parse.urlsplit` / `urlunsplit`.

---

## Streaming Progress Control

```python
_streaming_progress_enabled = os.environ.get("RUNNODE_STREAMING_PROGRESS", "true").lower() != "false"

def is_streaming_progress_enabled() -> bool
def set_streaming_progress_enabled(enabled: bool)
```

When disabled, `ProgressBar` and related functions suppress some output.

---

## Heartbeat System

Thread-safe debounced heartbeat logging via `_heartbeat_lock` and `_last_heartbeat` dict.

- Key format: `"{event_type}:{request_id}:{task_id}"`
- Default interval: 15 seconds (`RUNNODE_HEARTBEAT_INTERVAL_SEC`)
- Heartbeat logging toggle: `RUNNODE_HEARTBEAT_LOG` (default `"true"`)
- If `"check"` is not in event_type, the task_id is also printed as a `[Task Info]` line

---

## Windows VT Support

`_enable_windows_vt()` enables ANSI escape code processing on Windows consoles via `kernel32.SetConsoleMode` with `ENABLE_VIRTUAL_TERMINAL_PROCESSING` (0x0004). Called at module import time. Also sets stdout/stderr to UTF-8 on Windows.

---

## CJK Display Width

`get_display_width(text)` returns the display width of a string, counting CJK characters (U+4E00–U+9FFF, U+3000–U+303F) as width 2. Used for alignment calculations.

---

## When Adding a New Node

When adding a new node that calls an external API, follow this pattern:

1. Generate a `request_id` via `generate_request_id(task_type, provider)`
2. Create a `ProgressBar` instance
3. Call `log_prepare()` before the API request
4. Use `rn_pbar.set_generating()` / `update_absolute()` / `done()` for progress
5. Wrap the API call in try/except, using `format_runnode_error(e)` for error messages
6. Call `log_complete()` on success, `log_error()` on failure
7. Consider calling `log_backend()` for long-running polling loops
