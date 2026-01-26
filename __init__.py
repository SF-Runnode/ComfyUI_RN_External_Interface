from .comfly_config import *
from .Tools import *
from .nodes.nodes_midjourney import *
from .nodes.nodes_bytedance import *
from .nodes.nodes_google import *
from .nodes.nodes_openai import *
from .nodes.nodes_blackforestlabs import *
from .nodes.nodes_suno import *
from .nodes.nodes_vidu import *
from .nodes.nodes_qwen import *
from .nodes.nodes_MiniMax import *
from .nodes.nodes_kling import *
from .nodes.nodes_ollama import *
from .nodes.nodes_xai import *


WEB_DIRECTORY = "./web"

__version__ = "1.18.3"

NODE_CLASS_MAPPINGS = {
    "RunNode_api_set": Comfly_api_set,
    "RunNode_ollama_connectivity": OllamaConnectivityV2,
    "RunNode_ollama_chat": OllamaChat,
    "OpenAI_Sora_API_Plus": OpenAISoraAPIPlus,
    "OpenAI_Sora_API": OpenAISoraAPI,
    "RunNode_Mj": Comfly_Mj,
    "RunNode_mjstyle": Comfly_mjstyle,
    "RunNode_upload": Comfly_upload,
    "RunNode_Mju": Comfly_Mju,
    "RunNode_Mjv": Comfly_Mjv,
    "RunNode_Mj_swap_face": Comfly_Mj_swap_face,
    "RunNode_kling_text2video": Comfly_kling_text2video,
    "RunNode_kling_image2video": Comfly_kling_image2video,
    "RunNode_kling_multi_image2video": Comfly_kling_multi_image2video,
    "RunNode_video_extend": Comfly_video_extend,
    "RunNode_lip_sync": Comfly_lip_sync,
    "RunNodeGeminiAPI": ComflyGeminiAPI,
    "RunNodeSeededit": ComflySeededit,
    "RunNodeChatGPTApi": ComflyChatGPTApi,
    "RunNode_sora2_openai": Comfly_sora2_openai,
    "RunNode_sora2": Comfly_sora2,
    "RunNode_sora2_chat": Comfly_sora2_chat,
    "RunNode_sora2_character": Comfly_sora2_character,
    "RunNode_sora2_batch_32": Comfly_sora2_batch_32,
    "RunNode_sora2_group": Comfly_sora2_group,
    "RunNode_sora2_run_4": Comfly_sora2_run_4,
    "RunNode_sora2_run_8": Comfly_sora2_run_8,
    "RunNode_sora2_run_16": Comfly_sora2_run_16,
    "RunNode_sora2_run_32": Comfly_sora2_run_32,
    "RunNodeJimengApi": ComflyJimengApi,
    "RunNode_gpt_image_1_edit": Comfly_gpt_image_1_edit,
    "RunNode_gpt_image_1": Comfly_gpt_image_1,
    "RunNodeJimengVideoApi": ComflyJimengVideoApi,
    "RunNode_Flux_Kontext": Comfly_Flux_Kontext,
    "RunNode_Flux_Kontext_Edit": Comfly_Flux_Kontext_Edit,
    "RunNode_Flux_Kontext_bfl": Comfly_Flux_Kontext_bfl,
    "RunNode_Flux_2_Max": Comfly_Flux_2_Max,
    "RunNode_Flux_2_Pro": Comfly_Flux_2_Pro,
    "RunNode_Flux_2_Flex": Comfly_Flux_2_Flex,
    "RunNode_Gemini_TextOnly": ComflyGeminiTextOnly,
    "RunNode_Googel_Veo3": Comfly_Googel_Veo3,
    "RunNode_mj_video": Comfly_mj_video,
    "RunNode_mj_video_extend": Comfly_mj_video_extend,
    "RunNode_qwen_image": Comfly_qwen_image,
    "RunNode_qwen_image_edit": Comfly_qwen_image_edit,
    "RunNode_Doubao_Seedream": Comfly_Doubao_Seedream,
    "RunNode_Doubao_Seedream_4": Comfly_Doubao_Seedream_4,
    "RunNode_Doubao_Seedream_4_5": Comfly_Doubao_Seedream_4_5,
    "RunNode_Doubao_Seededit": Comfly_Doubao_Seededit,
    "RunNode_MiniMax_video": Comfly_MiniMax_video,
    "RunNode_suno_description": Comfly_suno_description,
    "RunNode_suno_lyrics": Comfly_suno_lyrics,
    "RunNode_suno_custom": Comfly_suno_custom,
    "RunNode_suno_upload": Comfly_suno_upload,
    "RunNode_suno_upload_extend": Comfly_suno_upload_extend,
    "RunNode_suno_cover": Comfly_suno_cover,
    "RunNode_vidu_img2video": Comfly_vidu_img2video,
    "RunNode_vidu_text2video": Comfly_vidu_text2video,
    "RunNode_vidu_ref2video": Comfly_vidu_ref2video,
    "RunNode_vidu_start-end2video": Comfly_vidu_start_end2video,
    "RunNode_nano_banana": Comfly_nano_banana,
    "RunNode_nano_banana_fal": Comfly_nano_banana_fal,
    "RunNode_nano_banana_edit": Comfly_nano_banana_edit,
    "RunNode_nano_banana2_edit": Comfly_nano_banana2_edit,
    "RunNode_nano_banana2_edit_S2A": Comfly_nano_banana2_edit_S2A,
    "RunNode_banana2_edit_group": Comfly_banana2_edit_group,
    "RunNode_banana2_edit_run_4": Comfly_banana2_edit_run_4,
    "RunNode_banana2_edit_run_8": Comfly_banana2_edit_run_8,
    "RunNode_banana2_edit_run_16": Comfly_banana2_edit_run_16,
    "RunNode_banana2_edit_run_32": Comfly_banana2_edit_run_32,
    "RunNode_banana2_edit_S2A_group": Comfly_banana2_edit_S2A_group,
    "RunNode_banana2_edit_S2A_run_4": Comfly_banana2_edit_S2A_run_4,
    "RunNode_banana2_edit_S2A_run_8": Comfly_banana2_edit_S2A_run_8,
    "RunNode_banana2_edit_S2A_run_16": Comfly_banana2_edit_S2A_run_16,
    "RunNode_banana2_edit_S2A_run_32": Comfly_banana2_edit_S2A_run_32,
    "RunNode_Z_image_turbo": Comfly_Z_image_turbo,
    "RunNode_wan2_6_API": Comfly_wan2_6_API,
    "RunNode_LLm_API": Comfly_LLm_API,
    "RunNode_Grok3VideoApi": ComflyGrok3VideoApi,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunNode_api_set": "RunNode API Settings",
    "RunNode_ollama_connectivity": "RunNode Ollama ConnectivityV2",
    "RunNode_ollama_chat": "RunNode Ollama Chat",
    "OpenAI_Sora_API_Plus": "RunNode Sora API Plus节点",
    "OpenAI_Sora_API": "RunNode Sora API节点",
    "RunNode_Mj": "RunNode Mj",
    "RunNode_mjstyle": "RunNode mjstyle",
    "RunNode_upload": "RunNode upload",
    "RunNode_Mju": "RunNode Mju",
    "RunNode_Mjv": "RunNode Mjv",
    "RunNode_Mj_swap_face": "RunNode MJ Face Swap",
    "RunNode_kling_text2video": "RunNode kling_text2video",
    "RunNode_kling_image2video": "RunNode kling_image2video",
    "RunNode_kling_multi_image2video": "RunNode kling_multi_image2video",
    "RunNode_video_extend": "RunNode video_extend",
    "RunNode_lip_sync": "RunNode lip_sync",
    "RunNodeGeminiAPI": "RunNode Gemini API",
    "RunNodeSeededit": "RunNode Doubao SeedEdit2.0",
    "RunNodeChatGPTApi": "RunNode ChatGPT Api",
    "RunNode_sora2_openai": "RunNode Sora2 OpenAI",
    "RunNode_sora2": "RunNode Sora2",
    "RunNode_sora2_chat": "RunNode Sora2 Chat",
    "RunNode_sora2_character": "RunNode Sora2 Character",
    "RunNode_sora2_batch_32": "RunNode Sora2 batch_32",
    "RunNode_sora2_group": "RunNode Sora2 Group",
    "RunNode_sora2_run_4": "RunNode Sora2 Run 4",
    "RunNode_sora2_run_8": "RunNode Sora2 Run 8",
    "RunNode_sora2_run_16": "RunNode Sora2 Run 16",
    "RunNode_sora2_run_32": "RunNode Sora2 Run 32",
    "RunNodeJimengApi": "RunNode Jimeng API",
    "RunNode_gpt_image_1_edit": "RunNode gpt_image_1_edit",
    "RunNode_gpt_image_1": "RunNode gpt_image_1",
    "RunNodeJimengVideoApi": "RunNode Jimeng Video API",
    "RunNode_Flux_Kontext": "RunNode Flux Kontext",
    "RunNode_Flux_Kontext_Edit": "RunNode Flux Kontext Edit",
    "RunNode_Flux_Kontext_bfl": "RunNode Flux Kontext bfl",
    "RunNode_Flux_2_Max": "RunNode Flux 2 Max",
    "RunNode_Flux_2_Pro": "RunNode Flux 2 Pro",
    "RunNode_Flux_2_Flex": "RunNode Flux 2 Flex",
    "RunNode_Gemini_TextOnly": "RunNode Gemini TextOnly",
    "RunNode_Googel_Veo3": "RunNode Google Veo3",
    "RunNode_mj_video": "RunNode MJ Video",
    "RunNode_mj_video_extend": "RunNode MJ Video Extend",
    "RunNode_qwen_image": "RunNode qwen_image",
    "RunNode_qwen_image_edit": "RunNode qwen_image_edit",
    "RunNode_Doubao_Seedream": "RunNode Doubao Seedream3.0",
    "RunNode_Doubao_Seedream_4": "RunNode Doubao Seedream4.0",
    "RunNode_Doubao_Seedream_4_5": "RunNode Doubao Seedream4.5",
    "RunNode_Doubao_Seededit": "RunNode Doubao Seededit3.0",
    "RunNode_MiniMax_video": "RunNode MiniMax Hailuo Video",
    "RunNode_suno_description": "RunNode Suno Description",
    "RunNode_suno_lyrics": "RunNode Suno Lyrics",
    "RunNode_suno_custom": "RunNode Suno Custom",
    "RunNode_suno_upload": "RunNode Suno Upload",
    "RunNode_suno_upload_extend": "RunNode Suno Upload Extend",
    "RunNode_suno_cover": "RunNode Suno Cover",
    "RunNode_vidu_img2video": "RunNode Vidu Image2Video",
    "RunNode_vidu_text2video": "RunNode Vidu Text2Video",
    "RunNode_vidu_ref2video": "RunNode Vidu Ref2Video",
    "RunNode_vidu_start-end2video": "RunNode Vidu Start-End2Video",
    "RunNode_nano_banana": "RunNode nano_banana",
    "RunNode_nano_banana_fal": "RunNode nano_banana_fal",
    "RunNode_nano_banana_edit": "RunNode nano_banana_edit",
    "RunNode_nano_banana2_edit": "RunNode nano_banana2_edit",
    "RunNode_nano_banana2_edit_S2A": "RunNode nano_banana2_edit_S2A",
    "RunNode_banana2_edit_group": "RunNode banana2_edit Group",
    "RunNode_banana2_edit_run_4": "RunNode banana2_edit Run 4",
    "RunNode_banana2_edit_run_8": "RunNode banana2_edit Run 8",
    "RunNode_banana2_edit_run_16": "RunNode banana2_edit Run 16",
    "RunNode_banana2_edit_run_32": "RunNode banana2_edit Run 32",
    "RunNode_banana2_edit_S2A_group": "RunNode banana2_edit S2A Group",
    "RunNode_banana2_edit_S2A_run_4": "RunNode banana2_edit S2A Run 4",
    "RunNode_banana2_edit_S2A_run_8": "RunNode banana2_edit S2A Run 8",
    "RunNode_banana2_edit_S2A_run_16": "RunNode banana2_edit S2A Run 16",
    "RunNode_banana2_edit_S2A_run_32": "RunNode banana2_edit S2A Run 32",
    "RunNode_Z_image_turbo": "RunNode Z Image Turbo",
    "RunNode_wan2_6_API": "RunNode wan2.6 video",
    "RunNode_LLm_API": "RunNode LLM API",
    "RunNode_Grok3VideoApi": "RunNode Grok3 Video",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

def start_ai_helper():
    import threading
    import subprocess
    import os
    import sys

    def run_ai_helper():
        ai_helper_path = os.path.join(os.path.dirname(__file__), "AiHelper.py")
        subprocess.run([sys.executable, ai_helper_path])

    ai_helper_thread = threading.Thread(target=run_ai_helper)
    ai_helper_thread.start()

start_ai_helper()
