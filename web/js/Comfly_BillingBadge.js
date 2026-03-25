/**
 * ComfyUI_RN_External_Interface - Price Badge Display
 * 在节点底部显示预估费用 badge
 */

import { app } from "../../../scripts/app.js";

(function () {
    const extensionId = "Comfly.PriceBadge";

    let billingConfig = null;

    /**
     * 加载计费配置
     */
    async function loadBillingConfig() {
        if (billingConfig) return billingConfig;

        try {
            const resp = await fetch('/api/billing_config');
            if (resp.ok) {
                billingConfig = await resp.json();
                console.log(`[${extensionId}] Loaded billing config`);
                return billingConfig;
            }
        } catch (e) {
            console.error(`[${extensionId}] Failed to load config:`, e);
        }

        billingConfig = { models: {} };
        return billingConfig;
    }

    /**
     * 根据节点类型和 widgets 估算价格
     */
    function estimatePrice(nodeType, widgets) {
        if (!billingConfig?.models) return null;

        const type = (nodeType || '').toLowerCase();
        const getWidgetValue = (name) => {
            const w = widgets?.find(w => w.name === name);
            return w?.value ?? w;
        };

        let modelKey = null;
        let duration = 10;

        const model = getWidgetValue('model') || getWidgetValue('model_name');
        const durVal = getWidgetValue('duration');
        if (durVal) duration = parseInt(durVal) || 10;

        if (type.includes('sora2') || type.includes('sora')) {
            modelKey = model || 'sora-2';
        }
        else if (type.includes('kling')) {
            modelKey = model || 'kling-v1-6';
        }
        else if (type.includes('mj') || type.includes('midjourney')) {
            return billingConfig.models['midjourney']?.price_per_use || 0.035;
        }
        else if (type.includes('suno')) {
            modelKey = model ? `suno-${model}` : 'suno-v4.5';
        }
        else if (type.includes('doubao') || type.includes('seedream') || type.includes('seededit')) {
            modelKey = model || 'doubao-seedream';
        }
        else if (type.includes('jimeng')) {
            modelKey = type.includes('video') ? 'jimeng-video' : 'jimeng-image';
            if (type.includes('video')) {
                duration = parseInt(getWidgetValue('duration')) || 5;
            }
        }
        else if (type.includes('grok')) {
            modelKey = 'grok-video-3';
        }
        else if (type.includes('vidu')) {
            modelKey = model || 'viduq2-pro';
        }
        else if (type.includes('minimax')) {
            modelKey = model || 'minimax-video';
        }
        else if (type.includes('flux')) {
            modelKey = model || 'flux-kontext';
        }
        else if (type.includes('qwen') || type.includes('z-image')) {
            modelKey = model || 'qwen-image';
        }
        else if (type.includes('gemini') || type.includes('veo')) {
            modelKey = model || 'gemini';
        }
        else if (type.includes('nano') || type.includes('banana')) {
            modelKey = model || 'nano-banana';
        }
        else if (type.includes('wan')) {
            modelKey = 'wan2.6-video';
        }
        else if (type.includes('ollama')) {
            modelKey = 'ollama';
            duration = 0;
        }
        else if (model) {
            modelKey = model;
        }
        else {
            return null;
        }

        let config = billingConfig.models[modelKey];

        if (!config) {
            for (const [key, val] of Object.entries(billingConfig.models)) {
                if (modelKey.includes(key) || key.includes(modelKey)) {
                    config = val;
                    break;
                }
            }
        }

        if (!config) return null;

        switch (config.billing_type) {
            case 'per_second':
                return (duration || 10) * (config.price_per_second || 0);
            case 'per_use':
                return config.price_per_use || 0;
            case 'token': {
                const promptWidget = widgets?.find(w => w.name === 'prompt');
                const promptLen = (promptWidget?.value || '').length || 1000;
                const inputTokens = Math.ceil(promptLen / 4);
                const outputTokens = Math.ceil(promptLen / 8);
                return (inputTokens / 1000 * (config.input_price_per_1k || 0)) +
                       (outputTokens / 1000 * (config.output_price_per_1k || 0));
            }
            case 'per_model':
                return config.price_per_model || 0;
            default:
                return null;
        }
    }

    /**
     * 格式化价格显示
     */
    function formatPrice(price) {
        if (!price || price <= 0) return null;
        if (price < 0.01) {
            const credits = Math.round(price * 211 * 10) / 10;
            return { text: `${credits.toFixed(1)} cr`, small: true };
        }
        return { text: `$${price.toFixed(4)}`, small: false };
    }

    /**
     * 检查是否是 Comfly 节点（仅 RunNode_* 前缀）
     */
    function isComflyNode(nodeName) {
        return nodeName.startsWith('RunNode_');
    }

    /**
     * 为节点添加价格 badge
     */
    function attachBadge(node, nodeName) {
        if (!isComflyNode(nodeName)) return;

        const price = estimatePrice(nodeName, node.widgets);
        if (!price) return;

        const formatted = formatPrice(price);
        if (!formatted) return;

        // 创建 badge 容器
        const badgeContainer = document.createElement('div');
        badgeContainer.style.cssText = 'position:absolute;top:2px;right:2px;z-index:1000;pointer-events:none;';

        const badge = document.createElement('div');
        badge.className = 'comfly-price-badge';
        badge.textContent = formatted.text;

        const baseStyle = 'padding:2px 8px;border-radius:10px;' +
            'font-size:11px;font-weight:600;' +
            'font-family:system-ui,sans-serif;box-shadow:0 1px 3px rgba(0,0,0,0.3);';

        badge.style.cssText = baseStyle +
            'background:linear-gradient(135deg,#11998e,#38ef7d);color:white;';

        if (formatted.small) {
            badge.style.cssText = baseStyle +
                'background:linear-gradient(135deg,#667eea,#764ba2);color:white;';
        }

        badgeContainer.appendChild(badge);

        // 移除已存在的 badge widget
        const existingWidget = node.widgets?.find(w => w.name === '__comfly_price_badge');
        if (existingWidget) {
            node.removeWidget(existingWidget);
        }

        // 使用 addDOMWidget 添加 badge
        try {
            const widget = node.addDOMWidget("__comfly_price_badge", "pb", badgeContainer);
            widget.getValue = () => "";
            widget.callback = () => {};
        } catch (e) {
            // 如果 addDOMWidget 失败，直接添加到节点元素
            if (node.element) {
                node.element.style.position = 'relative';
                const existing = node.element.querySelector('.comfly-price-badge');
                if (existing) existing.remove();
                node.element.appendChild(badgeContainer);
            }
        }
    }

    // ============== ComfyUI Extension ==============

    app.registerExtension({
        name: extensionId,

        async setup() {
            console.log(`[${extensionId}] Setup`);
            await loadBillingConfig();
        },

        async beforeRegisterNodeDef(nodeType, nodeData, app) {
            const nodeName = nodeData.name || '';
            if (!isComflyNode(nodeName)) return;

            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const result = originalOnNodeCreated?.apply(this, arguments);
                requestAnimationFrame(() => {
                    attachBadge(this, nodeName);
                });
                return result;
            };
        },

        async nodeCreated(node, app) {
            const nodeName = node.type || '';
            if (isComflyNode(nodeName)) {
                requestAnimationFrame(() => {
                    attachBadge(node, nodeName);
                });
            }
        }
    });

    // 暴露 API 方便调试
    window.__COMFLY_BILLING__ = {
        loadBillingConfig,
        estimatePrice,
        formatPrice
    };

})();
