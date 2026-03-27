/**
 * ComfyUI_RN_External_Interface - Price Badge Display
 * 在节点显示预估费用 badge（价格 + 计费方式）
 * 支持单节点和批量并发节点
 */

import { app } from "../../../scripts/app.js";

(function () {
    const extensionId = "Comfly.PriceBadge";

    let billingConfig = null;
    let currencyCode = 'USD';
    let baseCurrency = 'USD';
    let creditsRate = 211;
    let rates = {};
    let modelsCurrency = 'USD';
    let modelDisplayNames = {};
    let modelApiNames = {};

    /**
     * 加载计费配置
     */
    async function loadBillingConfig() {
        if (billingConfig) return billingConfig;

        try {
            const resp = await fetch('/api/billing_config');
            if (resp.ok) {
                billingConfig = await resp.json();
                const ds = billingConfig?.display_settings || {};
                currencyCode = ds.currency || 'USD';
                baseCurrency = ds.base_currency || 'USD';
                rates = ds.currency_rates || {};
                creditsRate = ds.credits_conversion_rate || 211;
                modelsCurrency = ds.models_currency || 'USD';
                modelDisplayNames = ds.model_display_names || {};
                modelApiNames = ds.model_api_names || {};
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
     * 获取模型的友好显示名称
     */
    function getModelDisplayName(modelKey) {
        return modelDisplayNames[modelKey] || modelKey;
    }

    /**
     * 将友好显示名称转换为 API 名称（用于价格查找）
     */
    function getApiModelName(displayName) {
        if (!displayName) return displayName;
        // 如果已经是 API 名称（原名），直接返回
        if (billingConfig?.models?.hasOwnProperty(displayName)) {
            return displayName;
        }
        // 尝试从映射表中查找
        return modelApiNames[displayName] || displayName;
    }

    /**
     * 计费方式显示信息
     */
    const BILLING_TYPE_INFO = {
        'per_second': { icon: '⏱️', label: 'per sec', shortLabel: '/s' },
        'per_use': { icon: '📌', label: 'per use', shortLabel: '/use' },
        'token': { icon: '💰', label: 'per token', shortLabel: '/token' },
        'per_model': { icon: '📦', label: 'per model', shortLabel: '/model' }
    };

    /**
     * 检测批量节点并获取数量
     * 返回 { isBatch, count } - count 为批次数量（单节点为1）
     */
    function detectBatchNode(nodeName) {
        const type = nodeName.toLowerCase();

        // sora2_run_X 模式
        const soraMatch = type.match(/sora2_run_(\d+)/);
        if (soraMatch) {
            return { isBatch: true, count: parseInt(soraMatch[1]) };
        }

        // banana2_edit_run_X 模式
        const bananaMatch = type.match(/banana2_edit(?:_s2a)?_run_(\d+)/);
        if (bananaMatch) {
            return { isBatch: true, count: parseInt(bananaMatch[1]) };
        }

        return { isBatch: false, count: 1 };
    }

    /**
     * 根据节点类型和 widgets 估算价格
     * @returns { price: number, billingType: string, billingTypeLabel: string, billingTypeIcon: string } 或 null
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
            const price = billingConfig.models['midjourney']?.price_per_use || 0.035;
            const info = BILLING_TYPE_INFO['per_use'];
            return { price, billingType: 'per_use', billingTypeIcon: info.icon, billingTypeLabel: info.shortLabel, modelKey: 'midjourney' };
        }
        else if (type.includes('suno')) {
            modelKey = model ? model : 'Suno 4.5';
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
        else if (type.includes('lip_sync')) {
            modelKey = 'lip_sync';
            // lip_sync uses duration from widget or default 10s
            duration = parseInt(getWidgetValue('duration')) || 10;
        }
        else if (model) {
            modelKey = model;
        }
        else {
            return null;
        }

        // 将友好名称转换为 API 名称（用于价格查找）
        modelKey = getApiModelName(modelKey);

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

        const info = BILLING_TYPE_INFO[config.billing_type] || { icon: '💳', label: '', shortLabel: '' };
        let price = 0;

        switch (config.billing_type) {
            case 'per_second':
                price = (duration || 10) * (config.price_per_second || 0);
                break;
            case 'per_use':
                price = config.price_per_use || 0;
                break;
            case 'token': {
                const promptWidget = widgets?.find(w => w.name === 'prompt');
                const promptLen = (promptWidget?.value || '').length || 1000;
                const inputTokens = Math.ceil(promptLen / 4);
                const outputTokens = Math.ceil(promptLen / 8);
                price = (inputTokens / 1000 * (config.input_price_per_1k || 0)) +
                       (outputTokens / 1000 * (config.output_price_per_1k || 0));
                break;
            }
            case 'per_model':
                price = config.price_per_model || 0;
                break;
            default:
                return null;
        }

        const priceBase = toBaseFrom(price, modelsCurrency || 'USD');
        return { price: priceBase, billingType: config.billing_type, billingTypeIcon: info.icon, billingTypeLabel: info.shortLabel, modelKey };
    }

    /**
     * 格式化价格显示
     */
    function getRate(code){
        if (code === baseCurrency) return 1;
        if (!rates || Object.keys(rates).length === 0) return undefined;
        return rates[code] || undefined;
    }

    function fromBaseTo(amount, code){
        const r = getRate(code);
        if (!r) return amount;
        return amount * r;
    }

    function toBaseFrom(amount, code){
        if (code === baseCurrency) return amount;
        const r = getRate(code);
        if (!r) return amount;
        return amount / r;
    }

    function formatPrice(price) {
        if (!price || price <= 0) return null;
        if (price < 0.01) {
            const credits = Math.round(price * creditsRate * 10) / 10;
            return { text: `${credits.toFixed(1)} cr`, small: true };
        }
        const converted = fromBaseTo((price || 0), currencyCode);
        const text = new Intl.NumberFormat(undefined, { style: 'currency', currency: currencyCode, maximumFractionDigits: 4 }).format(converted);
        return { text, small: false };
    }

    /**
     * 检查是否是 Comfly 节点
     * 支持多种命名模式：
     * - RunNode_* (RunNode_api_set, RunNode_mj, RunNode_sora2, etc.)
     * - RunNode[A-Z]* (RunNodeJimengApi, RunNodeJimengVideoApi)
     * - OpenAI_Sora_API* (OpenAI_Sora_API, OpenAI_Sora_API_Plus)
     * - Comfly_* (Comfly_*)
     */
    function isComflyNode(nodeName) {
        if (!nodeName) return false;
        // 匹配 RunNode_* 或 RunNode + 大写字母开头 (camelCase)
        if (/^RunNode[A-Z]/.test(nodeName) || nodeName.startsWith('RunNode_')) return true;
        // OpenAI Sora API 节点
        if (nodeName.startsWith('OpenAI_Sora_API')) return true;
        // Comfly_* 节点
        if (nodeName.startsWith('Comfly_')) return true;
        return false;
    }

    /**
     * 为节点添加价格 badge
     */
    function attachBadge(node, nodeName) {
        if (!isComflyNode(nodeName)) return;

        const result = estimatePrice(nodeName, node.widgets);
        if (!result || !result.price) return;

        const formatted = formatPrice(result.price);
        if (!formatted) return;

        // 检测是否是批量节点
        const { isBatch, count } = detectBatchNode(nodeName);
        const totalPrice = result.price * count;

        // 获取模型显示名称
        const modelDisplayName = getModelDisplayName(result.modelKey || '');

        // 格式化显示文本：模型显示名称 + 图标 + 价格 + 计费方式
        // 格式示例：Sora 2 ⏱️ ¥0.05/s 或 Sora 2 ⏱️ ¥0.20/s ×4
        let displayText;
        if (isBatch && count > 1) {
            const totalFormatted = formatPrice(totalPrice);
            displayText = `${modelDisplayName} ${result.billingTypeIcon} ${totalFormatted.text}${result.billingTypeLabel}${count > 1 ? ` ×${count}` : ''}`;
        } else {
            displayText = `${modelDisplayName} ${result.billingTypeIcon} ${formatted.text}${result.billingTypeLabel}`;
        }

        // 创建 badge 容器
        const badgeContainer = document.createElement('div');
        badgeContainer.style.cssText = 'position:absolute;top:2px;right:2px;z-index:1000;pointer-events:none;';

        const badge = document.createElement('div');
        badge.className = 'comfly-price-badge';
        badge.textContent = displayText;
        badge.title = `${modelDisplayName} - 计费方式: ${result.billingType}`;

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
