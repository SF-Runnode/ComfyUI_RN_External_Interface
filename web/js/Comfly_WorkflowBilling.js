/**
 * ComfyUI_RN_External_Interface - Workflow Billing Panel
 * 在界面右下角显示整个 Workflow 的总消费统计
 *
 * 功能：
 * 1. 点击运行时统计所有节点的总预估费用
 * 2. 执行完成后显示各节点的实际费用
 * 3. 显示总消费汇总
 */

(function () {
    const extensionId = "Comfly.WorkflowBilling";
    const PANEL_ID = "comfly-workflow-billing-panel";

    // 面板样式
    const PANEL_STYLE = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: rgba(30, 30, 40, 0.95);
        color: white;
        padding: 0;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        z-index: 10000;
        min-width: 220px;
        max-width: 350px;
        font-family: system-ui, -apple-system, sans-serif;
        font-size: 13px;
        overflow: hidden;
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.1);
    `;

    const HEADER_STYLE = `
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 14px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        cursor: pointer;
        user-select: none;
    `;

    const CONTENT_STYLE = `
        max-height: 0;
        overflow: hidden;
        transition: max-height 0.3s ease;
    `;

    const NODE_ROW_STYLE = `
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 4px 14px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        font-size: 12px;
    `;

    const TOTAL_ROW_STYLE = `
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 14px;
        background: rgba(255,255,255,0.05);
        font-weight: 600;
        font-size: 13px;
    `;

    // 状态存储
    let billingData = {
        workflow_id: null,
        nodes: {},
        total_estimated: 0,
        total_actual: 0,
        is_expanded: false,
    };

    let panelElement = null;
    let contentElement = null;

    /**
     * 创建/获取面板元素
     */
    function getOrCreatePanel() {
        if (panelElement) return panelElement;

        panelElement = document.createElement("div");
        panelElement.id = PANEL_ID;
        panelElement.style.cssText = PANEL_STYLE;

        panelElement.innerHTML = `
            <div class="billing-header" style="${HEADER_STYLE}">
                <span>💰 <span id="comfly-billing-title">预估消费</span></span>
                <span id="comfly-billing-total">$0.00</span>
            </div>
            <div class="billing-content" id="comfly-billing-content" style="${CONTENT_STYLE}">
                <div id="comfly-billing-nodes"></div>
                <div style="${TOTAL_ROW_STYLE}">
                    <span>总计</span>
                    <span id="comfly-billing-grand-total">$0.00</span>
                </div>
            </div>
        `;

        // 点击头部展开/收起
        const header = panelElement.querySelector(".billing-header");
        header.onclick = togglePanel;

        // 初始隐藏
        panelElement.style.display = "none";

        document.body.appendChild(panelElement);
        contentElement = panelElement.querySelector("#comfly-billing-content");

        return panelElement;
    }

    /**
     * 切换面板展开/收起
     */
    function togglePanel() {
        billingData.is_expanded = !billingData.is_expanded;
        if (contentElement) {
            contentElement.style.maxHeight = billingData.is_expanded ? "300px" : "0";
        }
    }

    /**
     * 显示/更新面板
     */
    function updatePanel(data) {
        const panel = getOrCreatePanel();
        panel.style.display = "block";

        billingData = { ...billingData, ...data };
        const { total_estimated = 0, total_actual = 0, nodes = {} } = billingData;

        // 更新总计显示
        const displayTotal = total_actual > 0 ? total_actual : total_estimated;
        const totalEl = panel.querySelector("#comfly-billing-total");
        const grandTotalEl = panel.querySelector("#comfly-billing-grand-total");
        const titleEl = panel.querySelector("#comfly-billing-title");

        if (total_actual > 0) {
            titleEl.textContent = "实际消费";
            totalEl.textContent = formatPrice(total_actual, true);
            totalEl.style.color = "#38ef7d";
        } else {
            titleEl.textContent = "预估消费";
            totalEl.textContent = formatPrice(total_estimated, true);
            totalEl.style.color = "white";
        }

        grandTotalEl.textContent = formatPrice(displayTotal, true);

        // 更新节点列表
        const nodesContainer = panel.querySelector("#comfly-billing-nodes");
        nodesContainer.innerHTML = "";

        Object.entries(nodes).forEach(([nodeId, info]) => {
            if (!info.name) return;
            const row = document.createElement("div");
            row.style.cssText = NODE_ROW_STYLE;
            row.innerHTML = `
                <span style="color: rgba(255,255,255,0.8); max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${info.name}">${info.name}</span>
                <span style="color: ${info.actual > 0 ? '#38ef7d' : '#f0ad4e'};">${formatPrice(info.actual > 0 ? info.actual : info.estimated)}</span>
            `;
            nodesContainer.appendChild(row);
        });
    }

    /**
     * 格式化价格显示
     */
    function formatPrice(price, showCredits = false) {
        if (price === undefined || price === null || price <= 0) {
            return "$0.00";
        }

        if (showCredits && price < 0.01) {
            const credits = Math.round(price * 211 * 10) / 10;
            return `${credits.toFixed(1)} cr`;
        }

        if (price < 0.01) {
            return `${(price * 211).toFixed(1)} cr`;
        }

        return `$${price.toFixed(4)}`;
    }

    /**
     * 隐藏面板
     */
    function hidePanel() {
        if (panelElement) {
            panelElement.style.display = "none";
        }
    }

    /**
     * 重置面板数据
     */
    function resetPanel() {
        billingData = {
            workflow_id: null,
            nodes: {},
            total_estimated: 0,
            total_actual: 0,
            is_expanded: false,
        };
        hidePanel();
    }

    /**
     * 更新单个节点的费用
     */
    function updateNodePrice(nodeId, nodeName, priceInfo) {
        const { estimated = 0, actual = 0 } = priceInfo || {};
        billingData.nodes[nodeId] = {
            name: nodeName,
            estimated,
            actual,
        };

        // 重新计算总计
        let totalEst = 0;
        let totalAct = 0;
        Object.values(billingData.nodes).forEach(n => {
            totalEst += n.estimated || 0;
            totalAct += n.actual || 0;
        });
        billingData.total_estimated = totalEst;
        billingData.total_actual = totalAct;

        updatePanel(billingData);
    }

    /**
     * 处理来自后端的进度文本（包含价格信息）
     */
    function handleProgressText(nodeId, text) {
        // 格式: "Price: 2.11 credits" 或 "Price: $0.01"
        if (!text || !text.startsWith("Price:")) return null;

        const match = text.match(/Price:\s*([\d.]+)\s*(cr|credits|[\$])?/i);
        if (!match) return null;

        const value = parseFloat(match[1]);
        const unit = (match[2] || "$").toLowerCase();

        let priceUSD;
        if (unit === "cr" || unit === "credits") {
            priceUSD = value / 211;
        } else {
            priceUSD = value;
        }

        return priceUSD;
    }

    // ============== ComfyUI 事件监听 ==============

    app.registerExtension({
        name: extensionId,

        async setup() {
            console.log(`[${extensionId}] Setup`);
            getOrCreatePanel();
        },

        /**
         * 监听节点执行
         */
        async beforeRun(node) {
            // 节点运行前，可以记录预估费用
            if (node.type && node.type.startsWith("RunNode_")) {
                console.log(`[${extensionId}] Before run: ${node.id} (${node.type})`);
            }
        },

        /**
         * 监听执行完成（需要拦截 PromptServer 的 progress）
         */
        async nodeExecuted(node) {
            console.log(`[${extensionId}] Node executed: ${node.id} (${node.type})`);
        },

        /**
         * 监听进度文本（实际费用通过此方式传递）
         */
        async setNodeProgress(nodeId, progress, text) {
            if (text && text.startsWith("Price:")) {
                const priceUSD = handleProgressText(nodeId, text);
                if (priceUSD !== null) {
                    const node = app.graph.getNodeById(parseInt(nodeId));
                    updateNodePrice(nodeId, node?.type || "Unknown", { estimated: 0, actual: priceUSD });
                }
            }
        },

        /**
         * 监听 Workflow 开始
         */
        async workflowStarted(workflowId) {
            console.log(`[${extensionId}] Workflow started: ${workflowId}`);
            resetPanel();
            billingData.workflow_id = workflowId;

            // 获取工作流中的计费节点并显示预估
            const nodes = app.graph._nodes || app.graph.nodes;
            if (nodes) {
                nodes.forEach(node => {
                    if (node.type && (node.type.startsWith("RunNode_") || node.type.startsWith("Comfly"))) {
                        // 尝试从节点数据获取预估
                        const priceInfo = node._price_estimate;
                        if (priceInfo) {
                            updateNodePrice(node.id, node.type, priceInfo);
                        }
                    }
                });
            }
        },

        /**
         * 监听 Workflow 完成
         */
        async workflowFinished() {
            console.log(`[${extensionId}] Workflow finished`);
            // 保持显示实际费用
        },

        // 暴露 API
        getExtensionAPI() {
            return {
                updateNodePrice,
                updatePanel,
                hidePanel,
                resetPanel,
                formatPrice,
            };
        }
    });

    // 暴露给全局
    if (typeof window !== "undefined") {
        window.__COMFLY_BILLING_PANEL__ = {
            updateNodePrice,
            updatePanel,
            hidePanel,
            resetPanel,
        };
    }

})();
