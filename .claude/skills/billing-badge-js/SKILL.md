---
name: billing-badge-js
description: Frontend billing UI code in web/js/. Covers Comfly_BillingBadge.js (per-node price badges showing estimated cost with billing type icons) and Comfly_WorkflowBilling.js (bottom-right collapsible summary panel for total workflow cost). Use when modifying node price display, adding new provider badges, changing billing type rendering, fixing badge positioning/lifecycle, or working on the workflow billing panel. Also use when the billing_config API response format changes and frontend parsing needs updating.
---

# Billing Badge JS

The two JavaScript files in `web/js/` that render price information on the ComfyUI frontend. They are ComfyUI extensions registered via `app.registerExtension()`.

## Comfly_BillingBadge.js — Per-Node Price Badge

Injects a colored price badge at the top-right of every Comfly node. The badge shows the estimated cost based on the current widget values, updating in real time as the user changes parameters.

### Config Loading

On setup, fetches `/api/billing_config` (served by `AiHelper.py`) and caches the result. Config provides:
- `models` — price definitions keyed by API model name
- `display_settings.currency` — display currency code (default `'USD'`)
- `display_settings.base_currency` — base currency for internal math (default `'USD'`)
- `display_settings.currency_rates` — exchange rates from base to other currencies
- `display_settings.credits_conversion_rate` — how many credits per USD (default `211`)
- `display_settings.models_currency` — what currency the model prices are in
- `display_settings.model_display_names` — friendly names for the badge label
- `display_settings.model_api_names` — mapping from display names to API names (for price lookup)

### Node Detection

`isComflyNode(nodeName)` matches three naming patterns:
- `RunNode[A-Z]*` or `RunNode_*` — most provider nodes
- `OpenAI_Sora_API*` — Sora API nodes
- `Comfly_*` — legacy prefix nodes

### Price Estimation

`estimatePrice(nodeType, widgets)` determines the model key and billing type by inspecting the node type string and widget values:

| Node type contains | Model key default | Notes |
|---|---|---|
| `sora2` / `sora` | `sora-2` | reads `model` widget |
| `kling` | `kling-v1-6` | reads `model` widget |
| `mj` / `midjourney` | `midjourney` | hardcoded `per_use` at $0.035 |
| `suno` | `Suno 4.5` | reads `model` widget |
| `doubao` / `seedream` / `seededit` | `doubao-seedream` | reads `model` widget |
| `jimeng` | `jimeng-video` or `jimeng-image` | video variant reads duration |
| `grok` | `grok-video-3` | |
| `vidu` | `viduq2-pro` | reads `model` widget |
| `minimax` | `minimax-video` | reads `model` widget |
| `flux` | `flux-kontext` | reads `model` widget |
| `qwen` / `z-image` | `qwen-image` | reads `model` widget |
| `gemini` / `veo` | `gemini` | reads `model` widget |
| `nano` / `banana` | `nano-banana` | reads `model` widget |
| `wan` | `wan2.6-video` | |
| `ollama` | `ollama` | duration forced to 0 |
| `lip_sync` | `lip_sync` | duration from widget or 10s |
| (has `model` widget) | model widget value | fallback |
| (none) | `null` | badge hidden |

After determining the model key, it converts the display name to API name via `modelApiNames` mapping, then looks up the billing config. If no exact match, falls back to substring matching.

### Billing Type Calculation

| Billing type | Calculation |
|---|---|
| `per_second` | `duration × price_per_second` |
| `per_second_with_conditions` | Matches `billing_conditions` against widget params (highest-score match wins), uses matched `price_per_second` or `base × multiplier`, then `duration × unit_price` |
| `per_use` | `price_per_use` directly |
| `token` | `(prompt_len / 4) / 1000 × input_price_per_1k + (prompt_len / 8) / 1000 × output_price_per_1k` |
| `per_model` | `price_per_model` directly |

The result price is converted from `models_currency` to base currency via `toBaseFrom()`. The `details` object carries `unit_price_per_second`, `matched_condition`, and `duration_seconds` for tooltip display.

### Condition Matching

`matchBillingCondition(conditions, params)` iterates all conditions arrays, scores each by how many non-reserved keys match (via `String()` comparison, with array-valued conditions matching if any element matches). Highest-score matching condition wins. Reserved keys (pricing fields, label, name, description) are excluded from matching.

### Batch Node Detection

`detectBatchNode(nodeName)` uses regex to identify batch variants:
- `sora2_run_(\d+)` — e.g. `sora2_run_4` → count = 4
- `banana2_edit(?:_s2a)?_run_(\d+)` — e.g. `banana2_edit_run_8` → count = 8

Batch total = unit price × count, displayed as `×N` suffix.

### Badge Rendering

Uses `node.addDOMWidget("__comfly_price_badge", "pb", badgeContainer)` to inject a DOM element. The badge is absolutely positioned at top-right of the node. Color scheme:
- Green gradient (`#11998e → #38ef7d`) for USD-range prices
- Purple gradient (`#667eea → #764ba2`) for sub-cent credit-range prices

Badge text format: `{display_name} {icon} {price}{billing_label}` (or `{total_price} ×{count}` for batch).

### Widget Hooks

On first attach, wraps each existing widget's `callback` to trigger badge refresh via `requestAnimationFrame`. Each widget is hooked only once (`__comfly_price_badge_hooked` flag). When a widget value changes, `attachBadge` re-runs — if the price becomes zero/unavailable, the badge widget is removed.

### Global API

`window.__COMFLY_BILLING__` exposes `{ loadBillingConfig, estimatePrice, formatPrice }` for debugging.

---

## Comfly_WorkflowBilling.js — Bottom-Right Summary Panel

A fixed-position collapsible panel at `bottom: 20px; right: 20px`. Shows per-node and total costs, toggling between "estimated" (before run) and "actual" (after run, parsed from progress text).

### Panel Structure
- **Header**: shows title ("预估消费" / "实际消费") and total, click to expand/collapse
- **Content**: scrollable node list (name + price per node) + grand total row
- Hidden initially (`display: none`), shown when `updatePanel()` is called

### Lifecycle Hooks
- `setup()`: creates panel DOM, loads billing config
- `workflowStarted(workflowId)`: resets panel, scans graph nodes with `RunNode_` / `Comfly` prefix for `_price_estimate` data
- `nodeExecuted(node)`: logs execution
- `setNodeProgress(nodeId, progress, text)`: parses `"Price: X credits"` format to extract actual cost, updates node price in panel
- `workflowFinished()`: keeps panel visible with actual costs

### Progress Text Parsing

`handleProgressText(nodeId, text)` matches the regex `/Price:\s*([\d.]+)\s*(cr|credits|[\$])?/i`. Converts credits to USD (dividing by `creditsRate`), then to base currency.

### Currency Handling

Same `fromBaseTo` / `toBaseFrom` / `getRate` pattern as the badge. `formatPrice` with `showCredits=true` shows credit format for values < $0.01.

### Global API

`window.__COMFLY_BILLING_PANEL__` exposes `{ updateNodePrice, updatePanel, hidePanel, resetPanel }`.

---

## Extension Pattern (Common to Both)

Both files follow the same structure:
1. Wrapped in an IIFE `(function() { ... })()`
2. Import `app` from `"../../../scripts/app.js"`
3. Register via `app.registerExtension({ name, setup, ...lifecycleHooks })`
4. Use `requestAnimationFrame` for safe DOM access after node creation
5. Expose debug APIs on `window`

## Backend Counterparts

- `billing_engine.py` — server-side billing calculation engine with the same billing types
- `billing_helpers.py` — `send_price_to_ui()` sends `"Price: X credits"` progress text that the workflow panel parses
- `AiHelper.py` — serves `/api/billing_config` endpoint consumed by both JS files

## When Editing These Files

- The `__comfly_price_badge` widget name is a sentinel used in both badge and billing helpers (`buildParamsFromWidgets` skips it). Don't rename without updating all references.
- Widget callback wrapping must call the original callback and then `requestAnimationFrame` — synchronous re-attach can cause infinite loops.
- When adding a new provider node type, update both `estimatePrice()` in the badge and `_infer_model_key()` in `billing_helpers.py` simultaneously.
- Badge removal on zero price is important — otherwise stale badges persist after model change.
- The DOM widget fallback (`node.element.appendChild`) handles older ComfyUI versions that lack `addDOMWidget`.
