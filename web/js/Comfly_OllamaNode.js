import { app } from "../../../scripts/app.js";

app.registerExtension({
  name: "Comfly_OllamaNode",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (["OllamaConnectivityV2", "RunNode_ollama_connectivity"].includes(nodeData.name)) {
      const originalNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = async function () {
        if (originalNodeCreated) {
          originalNodeCreated.apply(this, arguments);
        }

        const urlWidget = this.widgets.find((w) => w.name === "url");
        const modelWidget = this.widgets.find((w) => w.name === "model");
        const apiKeyWidget = this.widgets.find((w) => w.name === "api_key");
        let refreshButtonWidget = this.addWidget("button", "🔄 刷新模型");

        const fetchModels = async (url) => {
          const response = await fetch("/runnode_ollama/get_models", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              url,
              api_key: apiKeyWidget ? apiKeyWidget.value : "",
            }),
          });

          if (response.ok) {
            const models = await response.json();
            return models;
          } else {
            throw new Error(await response.text());
          }
        };

        const updateModels = async () => {
          refreshButtonWidget.name = "⏳ 获取中...";
          const url = urlWidget ? urlWidget.value : "";

          let models = [];
          try {
            models = await fetchModels(url);
          } catch (error) {
            app.extensionManager.toast.add({
              severity: "error",
              summary: "Ollama连接错误",
              detail: "请确认服务器可用并已安装模型",
              life: 5000,
            });
            refreshButtonWidget.name = "🔄 刷新模型";
            this.setDirtyCanvas(true);
            return;
          }

          const prevValue = modelWidget.value;
          if (modelWidget && modelWidget.options && Array.isArray(modelWidget.options.values)) {
            modelWidget.options.values = models;
          }

          if (models.includes(prevValue)) {
            modelWidget.value = prevValue;
          } else if (models.length > 0) {
            modelWidget.value = models[0];
          }

          refreshButtonWidget.name = "🔄 刷新模型";
          this.setDirtyCanvas(true);
        };

        if (urlWidget) urlWidget.callback = updateModels;
        refreshButtonWidget.callback = updateModels;

        await updateModels();
      };
    }
  },
});
