import { app } from "../../../scripts/app.js";
import { $el } from "../../../scripts/ui.js";

let pb_cache = {};

/**
 * 获取指定类型的风格数据（从本地服务）
 * @param {string} e - 风格类型
 * @returns {Promise<Object>} 风格数据
 */
async function getStyles(e) {
    // 优先从缓存获取，避免重复请求
    if (pb_cache[e]) {
        return pb_cache[e];
    }
    // 从本地8080服务请求风格数据
    const t = await fetch(`http://localhost:8080/mjstyle/${e}.json`);
    if (t.status === 200) {
        let s = await t.json();
        pb_cache[e] = s; // 存入缓存
        return s;
    }
}

/**
 * 创建风格标签元素（带复选框）
 * @param {Object} e - 风格数据（包含name、negative_prompt等）
 * @param {boolean} t - 是否默认选中
 * @param {Function} s - 选中状态变化的回调函数
 * @returns {HTMLElement} 标签元素
 */
function createTagElement(e, t, s) {
    return $el("label.Comfly_style-model-tag.comfy-btn", {
        dataset: {
            tag: e.name,
            name: e.name,
            negativePrompt: e.negative_prompt
        },
        style: {
            margin: "5px",
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "space-between",
            fontSize: "16px"
        },
        onclick: (t) => {
            t.preventDefault();
            t.stopPropagation();
            let o = t.currentTarget.querySelector("input[type='checkbox']");
            o.checked = !o.checked;
            // 切换选中样式
            t.currentTarget.classList.toggle("Comfly_style-model-tag--selected", o.checked);
            // 触发回调
            s(e, o.checked);
        }
    }, [
        $el("span", { textContent: e.name }),
        $el("input", {
            type: "checkbox",
            checked: t,
            style: { accentColor: "var(--comfy-menu-bg)" }
        })
    ]);
}

// 注册ComfyUI扩展
app.registerExtension({
    name: "Comfly_mjstyle",
    async beforeRegisterNodeDef(e, t, s) {
        // 只处理名为"Comfly_mjstyle"的节点
        if (["Comfly_mjstyle"].indexOf(t.name) >= 0) {
            // 保存原始的onNodeCreated方法
            const originalOnNodeCreated = e.prototype.onNodeCreated;
            
            // 重写节点创建方法
            e.prototype.onNodeCreated = function () {
                // 调用原始方法
                const result = originalOnNodeCreated?.apply(this, arguments);
                
                // 初始化节点属性
                this.properties.values = this.properties.values || []; // 存储选中的风格名称
                this.properties.currentStyleType = ""; // 当前选中的风格类型
                
                // 找到"styles_type"类型的widget
                const styleTypeWidget = this.widgets.find(widget => widget.name === "styles_type");
                
                // 创建"已选中风格"展示区域
                const selectedStylesContainer = $el("div", {
                    style: { display: "none" }
                }, [
                    $el("div", {
                        textContent: "Selected Styles:",
                        style: { marginBottom: "5px", fontWeight: "bold", fontSize: "18px" }
                    }),
                    $el("div.Comfly_selected-tags-list", {
                        style: {
                            marginBottom: "10px",
                            padding: "5px",
                            border: "1px solid var(--border-color)",
                            borderRadius: "5px",
                            maxHeight: "100px",
                            overflowY: "auto"
                        }
                    })
                ]);
                
                // 创建"可用风格"列表区域
                const availableStylesContainer = $el("div.Comfly_style-model-tags-list", {
                    style: {
                        height: "200px",
                        overflowY: "auto",
                        backgroundColor: "var(--comfy-menu-bg)",
                        color: "var(--fg-color)",
                        border: "1px solid var(--border-color)",
                        borderRadius: "5px",
                        padding: "5px",
                        display: "none"
                    },
                    onwheel: (e) => {
                        e.stopPropagation(); // 阻止滚动事件冒泡
                    }
                });
                
                // 创建"清除所有"按钮
                const clearAllBtn = $el("button.comfy-btn", {
                    textContent: "Clear all",
                    style: {
                        marginTop: "10px",
                        alignSelf: "flex-end",
                        fontSize: "16px",
                        padding: "4px 10px"
                    },
                    onclick: () => {
                        this.properties.values = []; // 清空选中列表
                        updateSelectedTags(); // 更新已选标签展示
                        renderAvailableTags(); // 重新渲染可用标签
                    }
                });
                
                // 组合整个节点的UI结构
                const stylePreviewContainer = $el("div.Comfly_style-preview", {
                    style: {
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "stretch",
                        position: "relative",
                        height: "100%"
                    }
                }, [
                    selectedStylesContainer,
                    $el("div", {
                        textContent: "Available Styles:",
                        style: { marginTop: "10px", marginBottom: "5px", fontWeight: "bold", fontSize: "18px" }
                    }),
                    availableStylesContainer,
                    $el("div", { style: { display: "flex", justifyContent: "flex-end" } }, [clearAllBtn])
                ]);
                
                // 更新已选中标签的展示
                const updateSelectedTags = () => {
                    const selectedTagsList = selectedStylesContainer.querySelector(".Comfly_selected-tags-list");
                    selectedTagsList.innerHTML = ""; // 清空现有内容
                    
                    if (this.properties.values.length > 0) {
                        selectedStylesContainer.style.display = "block"; // 显示已选区域
                        this.properties.values.forEach(styleName => {
                            // 创建已选标签（点击取消时从列表移除）
                            selectedTagsList.appendChild(createTagElement(
                                { name: styleName, negative_prompt: "" },
                                true,
                                (e, isChecked) => {
                                    if (!isChecked) {
                                        // 从选中列表中移除
                                        this.properties.values = this.properties.values.filter(name => name !== e.name);
                                        updateSelectedTags();
                                        renderAvailableTags();
                                    }
                                }
                            ));
                        });
                    } else {
                        selectedStylesContainer.style.display = "none"; // 隐藏已选区域
                    }
                };
                
                // 渲染可用标签列表
                const renderAvailableTags = () => {
                    const currentStyles = pb_cache[this.properties.currentStyleType];
                    if (currentStyles) {
                        availableStylesContainer.innerHTML = ""; // 清空现有内容
                        currentStyles.forEach(style => {
                            const isSelected = this.properties.values.includes(style.name);
                            // 创建可用标签（点击时切换选中状态）
                            availableStylesContainer.appendChild(createTagElement(
                                style,
                                isSelected,
                                (e, isChecked) => {
                                    if (isChecked && !this.properties.values.includes(e.name)) {
                                        this.properties.values.push(e.name); // 添加到选中列表
                                    } else if (!isChecked) {
                                        this.properties.values = this.properties.values.filter(name => name !== e.name); // 从选中列表移除
                                    }
                                    updateSelectedTags(); // 更新已选标签展示
                                }
                            ));
                        });
                    }
                };
                
                // 处理风格类型切换（当选择不同的风格类型时）
                if (styleTypeWidget) {
                    const originalCallback = styleTypeWidget.callback;
                    styleTypeWidget.callback = (newType) => {
                        // 调用原始回调
                        if (originalCallback) {
                            originalCallback(newType);
                        }
                        this.properties.currentStyleType = newType;
                        
                        if (newType) {
                            // 获取该类型的风格数据并渲染
                            getStyles(newType).then(styles => {
                                if (styles) {
                                    pb_cache[newType] = styles;
                                    renderAvailableTags();
                                    updateSelectedTags();
                                    availableStylesContainer.style.display = "block"; // 显示可用风格区域
                                }
                            });
                        } else {
                            // 清空并隐藏可用风格区域
                            availableStylesContainer.innerHTML = "";
                            availableStylesContainer.style.display = "none";
                        }
                    };
                }
                
                // 向节点添加DOM组件
                this.addDOMWidget("button", "btn", stylePreviewContainer).getValue = () => {
                    const selectedNames = this.properties.values;
                    const positivePrompt = selectedNames.join(", "); // 正向提示词（风格名称拼接）
                    this.properties.style_positive = positivePrompt;
                    
                    // 负向提示词（拼接选中风格的negative_prompt）
                    const negativePrompt = selectedNames
                        .map(name => {
                            const style = pb_cache[this.properties.currentStyleType]?.find(s => s.name === name);
                            return style?.negative_prompt || "";
                        })
                        .filter(p => p) // 过滤空值
                        .join(", ");
                    this.properties.style_negative = negativePrompt;
                    
                    return selectedNames.join(",");
                };
                
                // 设置节点大小
                this.setSize([500, 550]);
                
                return result;
            };
        }
    }
});
