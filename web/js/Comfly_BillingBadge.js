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
    let modelBillingNames = {};
    let displayToBillingKey = {};
    let badgeStyleInstalled = false;

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

    /**
     * 加载计费配置
     */
    async function loadBillingConfig() {
        if (billingConfig) return billingConfig;

        try {
            const resp = await fetch(`/api/billing_config?_=${Date.now()}`, { cache: 'no-store' });
            if (resp.ok) {
                billingConfig = await resp.json();
                const ds = billingConfig?.display_settings || {};
                currencyCode = ds.currency || 'USD';
                baseCurrency = ds.base_currency || 'USD';
                rates = ds.currency_rates || {};
                creditsRate = ds.credits_conversion_rate || 211;
                modelsCurrency = ds.models_currency || 'USD';
                modelDisplayNames = flattenStringMap(ds.model_display_names || {});
                modelApiNames = flattenStringMap(ds.model_api_names || {});
                modelBillingNames = flattenStringMap(ds.model_billing_names || {});
                rebuildDisplayToBillingKey();
                console.log(`[${extensionId}] Loaded billing config`);
                return billingConfig;
            }
        } catch (e) {
            console.error(`[${extensionId}] Failed to load config:`, e);
        }

        billingConfig = { models: {} };
        return billingConfig;
    }

    function rebuildDisplayToBillingKey() {
        displayToBillingKey = {};
        const models = billingConfig?.models || {};

        for (const key of Object.keys(models)) {
            const disp = modelDisplayNames?.[key];
            if (typeof disp !== 'string') continue;
            const t = disp.trim();
            if (!t) continue;
            displayToBillingKey[t] = key;
            displayToBillingKey[t.toLowerCase()] = key;
        }

        for (const [k, v] of Object.entries(modelBillingNames || {})) {
            if (typeof k !== 'string') continue;
            if (typeof v !== 'string') continue;
            const t = k.trim();
            if (!t) continue;
            if (models.hasOwnProperty(v)) {
                displayToBillingKey[t] = v;
                displayToBillingKey[t.toLowerCase()] = v;
            }
        }

        for (const [k, v] of Object.entries(modelApiNames || {})) {
            if (typeof k !== 'string') continue;
            if (typeof v !== 'string') continue;
            const t = k.trim();
            if (!t) continue;
            if (models.hasOwnProperty(v)) {
                displayToBillingKey[t] = v;
                displayToBillingKey[t.toLowerCase()] = v;
            }
        }
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
    function getBillingModelKey(displayName) {
        if (!displayName) return displayName;
        const raw = displayName;
        const name = (typeof raw === 'string') ? raw.trim() : raw;

        if (billingConfig?.models?.hasOwnProperty(raw)) return raw;
        if (name !== raw && billingConfig?.models?.hasOwnProperty(name)) return name;

        if (typeof raw === 'string') {
            const t = raw.trim();
            if (t && displayToBillingKey?.[t]) return displayToBillingKey[t];
            const lower = t ? t.toLowerCase() : '';
            if (lower && displayToBillingKey?.[lower]) return displayToBillingKey[lower];
        }

        const billingNameRaw = modelBillingNames?.[raw];
        const billingName = (typeof billingNameRaw === 'string') ? billingNameRaw.trim() : billingNameRaw;
        if (billingNameRaw && billingConfig?.models?.hasOwnProperty(billingNameRaw)) return billingNameRaw;
        if (billingName && billingConfig?.models?.hasOwnProperty(billingName)) return billingName;

        const apiNameRaw = modelApiNames?.[raw];
        const apiName = (typeof apiNameRaw === 'string') ? apiNameRaw.trim() : apiNameRaw;
        if (apiNameRaw && billingConfig?.models?.hasOwnProperty(apiNameRaw)) return apiNameRaw;
        if (apiName && billingConfig?.models?.hasOwnProperty(apiName)) return apiName;

        if (typeof name === 'string' && name) {
            const lower = name.toLowerCase();
            for (const [k, v] of Object.entries(modelBillingNames || {})) {
                if (typeof k === 'string' && k.trim().toLowerCase() === lower) return v;
            }
            for (const [k, v] of Object.entries(modelApiNames || {})) {
                if (typeof k === 'string' && k.trim().toLowerCase() === lower) return v;
            }
        }

        return billingName || apiName || name || raw;
    }

    /**
     * 计费方式显示信息
     */
    const BILLING_TYPE_INFO = {
        'per_second': { icon: '⏱️', label: 'per sec', shortLabel: '/s' },
        'per_second_with_conditions': { icon: '⏱️', label: 'per sec', shortLabel: '/s' },
        'per_use': { icon: '📌', label: 'per use', shortLabel: '/use' },
        'token': { icon: '💰', label: 'per token', shortLabel: '/token' },
        'per_model': { icon: '📦', label: 'per model', shortLabel: '/model' }
    };

    function normalizeParamValue(value) {
        if (value === undefined || value === null) return undefined;
        if (typeof value === 'string') return value;
        if (typeof value === 'number') return value;
        if (typeof value === 'boolean') return value;
        if (Array.isArray(value)) return value.map(normalizeParamValue);
        return String(value);
    }

    function extractWidgetValue(widget) {
        if (!widget) return undefined;
        const raw = (widget.value !== undefined) ? widget.value : widget;

        if (typeof raw === 'number') {
            const values = widget.options?.values;
            if (Array.isArray(values)) {
                const opt = values[raw];
                if (typeof opt === 'string') return opt;
                if (Array.isArray(opt) && opt.length) return opt[opt.length - 1];
                if (opt && typeof opt === 'object') {
                    if (opt.value !== undefined) return opt.value;
                    if (opt.content !== undefined) return opt.content;
                    if (opt.label !== undefined) return opt.label;
                    if (opt.text !== undefined) return opt.text;
                }
            }
        }

        if (raw && typeof raw === 'object') {
            if (raw.value !== undefined) return raw.value;
            if (raw.content !== undefined) return raw.content;
            if (raw.label !== undefined) return raw.label;
            if (raw.text !== undefined) return raw.text;
        }

        return raw;
    }

    function buildParamsFromWidgets(widgets) {
        const params = {};
        for (const w of (widgets || [])) {
            if (!w || !w.name) continue;
            if (w.name === '__comfly_price_badge') continue;
            const v = normalizeParamValue(extractWidgetValue(w));
            if (v !== undefined) params[w.name] = v;
        }
        return params;
    }

    function matchBillingCondition(conditions, params) {
        if (!Array.isArray(conditions) || conditions.length === 0) return null;
        const reservedKeys = new Set([
            'price_per_second',
            'price_per_use',
            'price_per_model',
            'input_price_per_1k',
            'output_price_per_1k',
            'multiplier',
            'label',
            'name',
            'description'
        ]);

        let best = null;
        let bestScore = -1;
        for (const cond of conditions) {
            if (!cond || typeof cond !== 'object') continue;
            let ok = true;
            let score = 0;
            for (const [k, expectedRaw] of Object.entries(cond)) {
                if (reservedKeys.has(k)) continue;
                const actual = params?.[k];
                if (actual === undefined) {
                    ok = false;
                    break;
                }
                const expected = normalizeParamValue(expectedRaw);
                if (Array.isArray(expected)) {
                    if (!expected.map(String).includes(String(actual))) {
                        ok = false;
                        break;
                    }
                } else {
                    if (String(actual) !== String(expected)) {
                        ok = false;
                        break;
                    }
                }
                score += 1;
            }
            if (!ok) continue;
            if (score > bestScore) {
                bestScore = score;
                best = cond;
            }
        }
        return best;
    }

    function resolvePerSecondUnitPrice(config, params) {
        if (!config) return { unitPrice: 0, matched: null };
        const baseUnit = Number(config.price_per_second || 0);
        if (config.billing_type !== 'per_second_with_conditions') {
            return { unitPrice: baseUnit, matched: null };
        }

        const matched = matchBillingCondition(config.billing_conditions, params);
        if (matched && matched.price_per_second !== undefined && matched.price_per_second !== null) {
            return { unitPrice: Number(matched.price_per_second || 0), matched };
        }
        if (matched && matched.multiplier !== undefined && matched.multiplier !== null) {
            return { unitPrice: baseUnit * Number(matched.multiplier || 0), matched };
        }
        return { unitPrice: baseUnit, matched };
    }

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
        const getWidgetValue = (name) => extractWidgetValue(widgets?.find(w => w.name === name));

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
            modelKey = getWidgetValue('speed') || 'midjourney-fast';
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
        modelKey = getBillingModelKey(modelKey);

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

        const params = buildParamsFromWidgets(widgets);
        const info = BILLING_TYPE_INFO[config.billing_type] || { icon: '💳', label: '', shortLabel: '' };
        let price = 0;
        let billingTypeLabel = info.shortLabel;
        let details = undefined;

        switch (config.billing_type) {
            case 'per_second':
            case 'per_second_with_conditions': {
                const dur = (duration || 10);
                const { unitPrice, matched } = resolvePerSecondUnitPrice(config, params);
                price = dur * unitPrice;
                billingTypeLabel = dur ? ` (${dur}s)` : '';
                details = { unit_price_per_second: unitPrice, matched_condition: matched || null, duration_seconds: dur };
                break;
            }
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
        return { price: priceBase, billingType: config.billing_type, billingTypeIcon: info.icon, billingTypeLabel, modelKey, details };
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
        // 匹配 RunNode_* / RunNode + 大写字母开头 (camelCase) / 以及友好名（如 "RunNode Doubao Seedance 2.0"）
        if (/^RunNode[A-Z]/.test(nodeName) || nodeName.startsWith('RunNode_') || nodeName.startsWith('RunNode ')) return true;
        // OpenAI Sora API 节点
        if (nodeName.startsWith('OpenAI_Sora_API')) return true;
        // Comfly_* 节点
        if (nodeName.startsWith('Comfly_')) return true;
        return false;
    }

    // 价格 badge 的可见文本高度，本身不随节点纵向拉伸变化。
    const BADGE_TEXT_HEIGHT = 18;
    // badge 到节点底边的最终留白。当前调到 4 时视觉效果最贴近需求。
    const BADGE_BOTTOM_INSET = 4;
    // badge 到节点左边框、右边框的最终留白。
    const BADGE_LEFT_INSET = 14;
    const BADGE_RIGHT_INSET = 14;
    // 预留的“上一栏到 badge”额外间距。当前最终方案里由向上位移控制，因此这里保持 0。
    const BADGE_TOP_GAP = 0;
    // 额外的纵向补偿量：
    // 1. 用来微调 badge 与上一栏的距离；
    // 2. 当前为 0，表示不再额外补偿，直接使用底部留白作为上移基准。
    const BADGE_GAP_COMPENSATION_Y = 0;
    // badge 实际向上位移的总量。当前等于底部留白，因此上下视觉关系能同时满足。
    const BADGE_RAISE_Y = BADGE_BOTTOM_INSET + BADGE_GAP_COMPENSATION_Y;
    // addDOMWidget 的 margin 会同时作用在四周：
    // - 底部：决定 badge 外层离底边的安全距离
    // - 左右：会先吃掉一部分可见宽度，后面再用 INNER_LEFT / INNER_RIGHT 补回到目标留白
    const BADGE_WIDGET_MARGIN = BADGE_BOTTOM_INSET;
    // 由于外层 DOMWidget 已经有 margin，这里只补“目标左右留白 - 外层 margin”的差值。
    const BADGE_INNER_LEFT = Math.max(0, BADGE_LEFT_INSET - BADGE_WIDGET_MARGIN);
    const BADGE_INNER_RIGHT = Math.max(0, BADGE_RIGHT_INSET - BADGE_WIDGET_MARGIN);
    // DOMWidget 的可见区域会被 margin 从四周各缩掉一圈，因此总高度必须包含：
    // 外层上下 margin + badge 文本高度 + 额外顶部空隙。
    const BADGE_WIDGET_HEIGHT = BADGE_TEXT_HEIGHT + BADGE_TOP_GAP + BADGE_WIDGET_MARGIN * 2;

    function ensureBadgeStylesInstalled() {
        if (badgeStyleInstalled || typeof document === 'undefined') return;
        const style = document.createElement('style');
        style.id = 'comfly-price-badge-style';
        style.textContent = `
.dom-widget:has(.comfly-price-badge-container) {
    pointer-events: none !important;
    opacity: 1 !important;
}
.dom-widget:has(.comfly-price-badge-container) .comfly-price-badge-container {
    pointer-events: none !important;
}
`;
        document.head.appendChild(style);
        badgeStyleInstalled = true;
    }

    function removeExistingBadge(node) {
        const existingWidget = node.widgets?.find(w => w.name === '__comfly_price_badge');
        if (existingWidget) {
            node.removeWidget(existingWidget);
        }

        if (node.element) {
            const existingContainer = node.element.querySelector('.comfly-price-badge-container');
            if (existingContainer) existingContainer.remove();
        }
    }

    function requestBadgeRedraw(node) {
        node?.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas?.(true, true);
    }

    function syncBadgeWidgetPosition(node, widget) {
        if (!node || !widget || !Array.isArray(node.size)) return;
        const nodeHeight = Number(node.size[1]) || 0;
        const y = Math.max(0, nodeHeight - BADGE_WIDGET_HEIGHT);
        widget.y = y;
        widget.computedHeight = BADGE_WIDGET_HEIGHT;
        widget.height = BADGE_WIDGET_HEIGHT;
        widget.last_y = y;
    }

    function ensureBadgeWidget(node, text, title, small) {
        ensureBadgeStylesInstalled();

        let widget = node.widgets?.find(w => w.name === '__comfly_price_badge');
        let badgeContainer = widget?.element;
        let badge = badgeContainer?.querySelector?.('.comfly-price-badge');

        if (!widget || !badgeContainer || !badge) {
            removeExistingBadge(node);

            badgeContainer = document.createElement('div');
            badgeContainer.className = 'comfly-price-badge-container';
            badgeContainer.style.cssText = [
                'box-sizing:border-box',
                'width:100%',
                'height:100%',
                'position:relative',
                'pointer-events:none'
            ].join(';') + ';';

            badge = document.createElement('div');
            badge.className = 'comfly-price-badge';
            badgeContainer.appendChild(badge);

            widget = node.addDOMWidget("__comfly_price_badge", "pb", badgeContainer, {
                hideOnZoom: false,
                margin: BADGE_WIDGET_MARGIN,
                getMinHeight: () => BADGE_WIDGET_HEIGHT,
                getMaxHeight: () => BADGE_WIDGET_HEIGHT,
                getHeight: () => BADGE_WIDGET_HEIGHT,
                afterResize() {
                    syncBadgeWidgetPosition(node, widget);
                },
                onDraw() {
                    syncBadgeWidgetPosition(node, widget);
                }
            });
            widget.getValue = () => "";
            widget.callback = () => {};
        }

        const baseStyle = 'padding:2px 8px;border-radius:10px;' +
            'font-size:11px;font-weight:600;' +
            'font-family:system-ui,sans-serif;box-shadow:0 1px 3px rgba(0,0,0,0.3);' +
            `position:absolute;left:${BADGE_INNER_LEFT}px;right:${BADGE_INNER_RIGHT}px;bottom:0;` +
            `display:block;box-sizing:border-box;height:${BADGE_TEXT_HEIGHT}px;line-height:14px;` +
            `transform:translateY(-${BADGE_RAISE_Y}px);` +
            'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;';

        badge.textContent = text;
        badge.title = title;
        badge.style.cssText = baseStyle +
            (small
                ? 'background:linear-gradient(135deg,#667eea,#764ba2);color:white;'
                : 'background:linear-gradient(135deg,#11998e,#38ef7d);color:white;');

        syncBadgeWidgetPosition(node, widget);
        return widget;
    }

    /**
     * 为节点添加价格 badge
     */
    function attachBadge(node, nodeName) {
        if (!isComflyNode(nodeName)) return;

        if (!node.__comfly_price_badge_hooked) {
            node.__comfly_price_badge_hooked = true;
            for (const w of (node.widgets || [])) {
                if (!w || !w.name) continue;
                if (w.name === '__comfly_price_badge') continue;
                if (w.__comfly_price_badge_hooked) continue;
                w.__comfly_price_badge_hooked = true;
                const original = w.callback;
                w.callback = function () {
                    const r = original?.apply(this, arguments);
                    requestAnimationFrame(() => attachBadge(node, nodeName));
                    return r;
                };
            }
        }

        const result = estimatePrice(nodeName, node.widgets);
        if (!result || !result.price) {
            removeExistingBadge(node);
            return;
        }

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

        let badgeTitle = `${modelDisplayName} - 计费方式: ${result.billingType}`;
        if (result.details?.matched_condition) {
            const cond = result.details.matched_condition;
            const reserved = new Set(['price_per_second', 'price_per_use', 'price_per_model', 'input_price_per_1k', 'output_price_per_1k', 'multiplier', 'label', 'name', 'description']);
            const parts = Object.entries(cond)
                .filter(([k]) => !reserved.has(k))
                .map(([k, v]) => `${k}=${v}`);
            if (parts.length) {
                badgeTitle += `\n匹配条件: ${parts.join(', ')}`;
            }
        }

        ensureBadgeWidget(node, displayText, badgeTitle, formatted.small);
        requestBadgeRedraw(node);
    }

    // ============== ComfyUI Extension ==============

    app.registerExtension({
        name: extensionId,

        async setup() {
            console.log(`[${extensionId}] Setup`);
            await loadBillingConfig();
        },

        async beforeRegisterNodeDef(nodeType, nodeData, app) {
            const nodeTypeName = nodeData?.name || '';
            const nodeDisplayName = nodeData?.display_name || nodeData?.displayName || '';
            if (!isComflyNode(nodeTypeName) && !isComflyNode(nodeDisplayName)) return;

            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const result = originalOnNodeCreated?.apply(this, arguments);
                requestAnimationFrame(() => {
                    attachBadge(this, this.type || nodeTypeName || nodeDisplayName);
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
