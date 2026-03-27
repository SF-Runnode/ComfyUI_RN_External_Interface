"""
ComfyUI_RN_External_Interface - Billing Helpers for Nodes
节点集成计费功能的辅助工具
"""

from typing import Dict, Any, Optional, Callable
from contextvars import ContextVar
from .billing_engine import (
    BillingCalculator,
    BillingResult,
    estimate_price,
    calculate_actual_price,
    format_price_for_display,
    get_workflow_billing,
)

# 使用 ContextVar 实现线程/协程安全的节点 ID 存储
_current_node_id_var: ContextVar[Optional[str]] = ContextVar('current_node_id', default=None)


def set_current_node_id(node_id: str):
    """设置当前执行的节点 ID"""
    _current_node_id_var.set(node_id)


def get_current_node_id() -> Optional[str]:
    """获取当前执行的节点 ID"""
    return _current_node_id_var.get()


def send_progress_text(node_id: str, text: str):
    """
    发送进度文本到前端显示

    Args:
        node_id: 节点 ID
        text: 要显示的文本
    """
    try:
        from server import PromptServer
        PromptServer.instance.send_progress_text(node_id, text)
    except Exception as e:
        print(f"[Billing] Warning: Failed to send progress text: {e}")


def send_price_to_ui(node_id: str, price_usd: float):
    """
    将实际价格发送到前端 UI 显示

    Args:
        node_id: 节点 ID
        price_usd: 实际费用（美元）
    """
    if price_usd <= 0:
        return

    # 转换为 credits 格式显示
    credits = price_usd * 211
    credits_str = f"{credits:,.1f}".rstrip("0").rstrip(".")
    send_progress_text(node_id, f"Price: {credits_str} credits")


def get_price_estimate_for_node(node_type: str, params: Dict) -> Optional[BillingResult]:
    """
    根据节点类型和参数获取价格预估

    Args:
        node_type: 节点类型名（如 "RunNode_LLm_API"）
        params: 节点参数字典

    Returns:
        BillingResult 或 None（如果未找到配置）
    """
    # 从节点类型推断模型
    model_key = _infer_model_key(node_type, params)
    if not model_key:
        return None

    return estimate_price(model_key, **params)


def _infer_model_key(node_type: str, params: Dict) -> Optional[str]:
    """
    从节点类型和参数推断模型标识符

    Args:
        node_type: 节点类型名
        params: 节点参数字典

    Returns:
        模型标识符字符串
    """
    # 节点类型到模型前缀的映射
    type_to_prefix = {
        "RunNode_LLm_API": "openai/",
        "RunNodeChatGPTApi": "openai/",
        "Comfly_gpt_image_1_edit": "openai/gpt-image-1",
        "Comfly_gpt_image_1": "openai/gpt-image-1",
        "Comfly_GeminiAPI": "gemini/",
        "ComflyGeminiTextOnly": "gemini/",
        "Comfly_Googel_Veo3": "gemini/",
        "RunNodeGeminiAPI": "gemini/",
        "RunNode_Gemini_TextOnly": "gemini/",
        "Comfly_Mj": "midjourney-",
        "RunNode_Mj": "midjourney-",
        "Comfly_Mju": "midjourney-",
        "Comfly_Mjv": "midjourney-",
        "Comfly_Mj_swap_face": "midjourney-",
        "Comfly_mj_video": "midjourney-",
        "Comfly_mj_video_extend": "midjourney-",
        "Comfly_suno_description": "suno-",
        "Comfly_suno_lyrics": "suno-",
        "Comfly_suno_custom": "suno-",
        "Comfly_suno_upload": "suno-",
        "Comfly_suno_upload_extend": "suno-",
        "Comfly_suno_cover": "suno-",
        "Comfly_kling_text2video": "kling-video",
        "Comfly_kling_image2video": "kling-video",
        "Comfly_kling_multi_image2video": "kling-video",
        "Comfly_video_extend": "kling-video",
        "Comfly_lip_sync": "kling-video",
        "RunNode_kling_text2video": "kling-video",
        "RunNode_kling_image2video": "kling-video",
        "RunNode_kling_multi_image2video": "kling-video",
        "RunNode_video_extend": "kling-video",
        "RunNode_lip_sync": "kling-video",
        "Comfly_sora2_openai": "sora2",
        "Comfly_sora2": "sora2",
        "Comfly_sora2_chat": "sora2",
        "Comfly_sora2_character": "sora2",
        "Comfly_sora2_batch_32": "sora2",
        "Comfly_sora2_group": "sora2",
        "Comfly_sora2_run_4": "sora2",
        "Comfly_sora2_run_8": "sora2",
        "Comfly_sora2_run_16": "sora2",
        "Comfly_sora2_run_32": "sora2",
        "OpenAI_Sora_API": "sora2",
        "OpenAI_Sora_API_Plus": "sora2",
        "Comfly_vidu_img2video": "vidu-video",
        "Comfly_vidu_text2video": "vidu-video",
        "Comfly_vidu_ref2video": "vidu-video",
        "Comfly_vidu_start-end2video": "vidu-video",
        "RunNode_vidu_img2video": "vidu-video",
        "RunNode_vidu_text2video": "vidu-video",
        "RunNode_vidu_ref2video": "vidu-video",
        "RunNode_vidu_start-end2video": "vidu-video",
        "Comfly_MiniMax_video": "minimax-video",
        "RunNode_MiniMax_video": "minimax-video",
        "Comfly_Flux_Kontext": "flux-kontext",
        "Comfly_Flux_Kontext_Edit": "flux-kontext",
        "Comfly_Flux_Kontext_bfl": "flux-kontext",
        "Comfly_Flux_2_Max": "flux-2-max",
        "Comfly_Flux_2_Pro": "flux-2-pro",
        "Comfly_Flux_2_Flex": "flux-2-pro",
        "RunNode_Flux_Kontext": "flux-kontext",
        "RunNode_Flux_Kontext_Edit": "flux-kontext",
        "RunNode_Flux_Kontext_bfl": "flux-kontext",
        "RunNode_Flux_2_Max": "flux-2-max",
        "RunNode_Flux_2_Pro": "flux-2-pro",
        "RunNode_Flux_2_Flex": "flux-2-pro",
        "Comfly_qwen_image": "qwen-image",
        "Comfly_qwen_image_edit": "qwen-image",
        "RunNode_qwen_image": "qwen-image",
        "RunNode_qwen_image_edit": "qwen-image",
        "Comfly_Doubao_Seedream": "doubao-seedream",
        "Comfly_Doubao_Seedream_4": "doubao-seedream",
        "Comfly_Doubao_Seedream_4_5": "doubao-seedream",
        "Comfly_Doubao_Seededit": "doubao-seededit",
        "ComflySeededit": "doubao-seededit",
        "RunNode_Doubao_Seedream": "doubao-seedream",
        "RunNode_Doubao_Seedream_4": "doubao-seedream",
        "RunNode_Doubao_Seedream_4_5": "doubao-seedream",
        "RunNode_Doubao_Seededit": "doubao-seededit",
        "RunNodeSeededit": "doubao-seededit",
        "ComflyJimengVideoApi": "jimeng-video",
        "ComflyJimengApi": "jimeng-image",
        "RunNodeJimengVideoApi": "jimeng-video",
        "RunNodeJimengApi": "jimeng-image",
        "Comfly_wan2_6_API": "wan2.6-video",
        "RunNode_wan2_6_API": "wan2.6-video",
        "Comfly_nano_banana": "nano-banana",
        "Comfly_nano_banana_fal": "nano-banana",
        "Comfly_nano_banana_edit": "nano-banana",
        "Comfly_nano_banana2_edit": "nano-banana",
        "Comfly_nano_banana2_edit_S2A": "nano-banana",
        "RunNode_nano_banana": "nano-banana",
        "RunNode_nano_banana_fal": "nano-banana",
        "RunNode_nano_banana_edit": "nano-banana",
        "RunNode_nano_banana2_edit": "nano-banana",
        "RunNode_nano_banana2_edit_S2A": "nano-banana",
        "Comfly_Grok3VideoApi": "grok3-video",
        "RunNode_Grok3VideoApi": "grok3-video",
    }

    # 尝试直接匹配
    prefix = type_to_prefix.get(node_type)
    if not prefix:
        return None

    # 如果参数中有 model，使用完整 model key
    model_param = params.get("model")
    if model_param:
        # 有些节点直接传递完整 model 名
        if "/" in str(model_param):
            return str(model_param)
        return f"{prefix}{model_param}"

    # 有些节点使用 mode 或 quality 作为子类型
    mode_param = params.get("mode")
    if mode_param:
        return f"{prefix}{mode_param}"

    quality_param = params.get("quality")
    if quality_param:
        return f"{prefix}{quality_param}"

    # 返回基础前缀
    return prefix.rstrip("-")


def log_billing_event(event_type: str, node_id: str, node_name: str, result: BillingResult):
    """
    记录计费事件（可扩展用于日志/监控）

    Args:
        event_type: 事件类型（"estimate", "actual"）
        node_id: 节点 ID
        node_name: 节点名称
        result: 计费结果
    """
    if result.billing_type == "unknown":
        return

    print(f"[Billing] {event_type.upper()} | Node: {node_name} | "
          f"Type: {result.billing_type} | "
          f"Est: ${result.estimated:.6f} | "
          f"Act: ${result.actual:.6f}")


# 导出
__all__ = [
    "BillingCalculator",
    "BillingResult",
    "estimate_price",
    "calculate_actual_price",
    "format_price_for_display",
    "get_workflow_billing",
    "send_progress_text",
    "send_price_to_ui",
    "set_current_node_id",
    "get_current_node_id",
    "get_price_estimate_for_node",
    "log_billing_event",
]
