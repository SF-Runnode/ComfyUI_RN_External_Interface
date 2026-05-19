/**
 * ComfyUI_RN_External_Interface - Model Display Name Mapper
 * Injects getOptionLabel on model/version/speed combo widgets so the UI
 * shows friendly display names while the actual values (API names) flow
 * through the workflow JSON and into node execution.
 */
import { app } from "../../../scripts/app.js";

(function () {
    const extensionId = "Comfly.ModelDisplay";

    let displayNames = {};

    function flattenStringMap(input) {
        if (!input || typeof input !== 'object') return {};
        const out = {};
        for (const [k, v] of Object.entries(input)) {
            if (typeof v === 'string') {
                out[k] = v;
                continue;
            }
            if (v && typeof v === 'object') {
                for (const [ik, iv] of Object.entries(v)) {
                    if (typeof iv === 'string') out[ik] = iv;
                }
            }
        }
        return out;
    }

    async function loadDisplayNames() {
        try {
            const resp = await fetch(`/api/billing_config?_=${Date.now()}`, { cache: 'no-store' });
            if (resp.ok) {
                const config = await resp.json();
                displayNames = flattenStringMap(
                    config?.display_settings?.model_display_names || {}
                );
                console.log(`[${extensionId}] Loaded ${Object.keys(displayNames).length} display name mappings`);
            }
        } catch (e) {
            console.error(`[${extensionId}] Failed to load display names:`, e);
        }
    }

    app.registerExtension({
        name: extensionId,

        async setup() {
            await loadDisplayNames();
        },

        async nodeCreated(node) {
            for (const widget of node.widgets || []) {
                if (!['model', 'model_name', 'version', 'speed'].includes(widget.name)) continue;
                if (widget.type !== 'combo') continue;

                widget.options.getOptionLabel = (value) => {
                    if (value === undefined || value === null) return '';
                    const key = String(value);
                    return displayNames[key] || key;
                };
            }
        }
    });
})();
