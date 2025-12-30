import urllib.request
import urllib.error
import json
import ssl
import base64
from io import BytesIO
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
from pydantic.json_schema import JsonSchemaValue
from pprint import pprint
import numpy as np
from PIL import Image
from server import PromptServer
from aiohttp import web
from ollama import Client
from .comfly_config import get_config, save_config, baseurl
import requests
import os
from io import BytesIO
import copy
import json



class Comfly_api_set:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_base": (["RunNode", "ip", "hk", "us"], {"default": "RunNode"}),
                "apikey": ("STRING", {"default": ""}),
            },
            "optional": {
                "custom_ip": ("STRING", {"default": "", "placeholder": "Enter IP when using 'ip' option (e.g. http://104.194.8.112:9088)"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("apikey",)
    FUNCTION = "set_api_base"
    CATEGORY = "RunNode"

    def set_api_base(self, api_base, apikey="", custom_ip=""):
        global baseurl
        base_url_mapping = {
            "RunNode": "https://ai.t8star.cn",
            "ip": custom_ip,
            "hk": "https://hk-api.gptbest.vip",
            "us": "https://api.gptbest.vip"
        }
        if api_base == "ip" and not custom_ip.strip():
            raise ValueError("When selecting 'ip' option, you must provide a custom IP address in the 'custom_ip' field")
        if api_base in base_url_mapping:
            baseurl = base_url_mapping[api_base]
        if apikey.strip():
            cfg = get_config()
            cfg['api_key'] = apikey
            save_config(cfg)
        print(f"API Base URL set to: {baseurl}")
        return (apikey,)


@dataclass
class ChatSession:
    messages: list[dict] = field(default_factory=list)
    model: str = ""


CHAT_SESSIONS: dict[str, ChatSession] = {}


def _filter_enabled_options(options: dict[str, Any] | None) -> dict[str, Any] | None:
    if not options:
        return None
    enablers = [
        "enable_mirostat",
        "enable_mirostat_eta",
        "enable_mirostat_tau",
        "enable_num_ctx",
        "enable_repeat_last_n",
        "enable_repeat_penalty",
        "enable_temperature",
        "enable_seed",
        "enable_stop",
        "enable_tfs_z",
        "enable_num_predict",
        "enable_top_k",
        "enable_top_p",
        "enable_min_p",
    ]
    out: dict[str, Any] = {}
    for enabler in enablers:
        if options.get(enabler, False):
            key = enabler.replace("enable_", "")
            out[key] = options[key]
    return out or None


@PromptServer.instance.routes.post("/runnode_ollama/get_models")
async def get_models_endpoint(request):
    data = await request.json()
    url = data.get("url")
    api_key = data.get("api_key", "")
    models = []
    if api_key:
        try:
            req = urllib.request.Request(url + "/api/tags")
            req.add_header("Content-Type", "application/json")
            req.add_header("Authorization", f"Bearer {api_key}")
            with urllib.request.urlopen(req) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            models_list = payload.get("models", [])
            try:
                models = [m["model"] for m in models_list]
            except Exception:
                models = [m.get("name", "") for m in models_list]
        except Exception:
            models = []
    else:
        client = Client(host=url)
        models_list = client.list().get('models', [])
        try:
            models = [model['model'] for model in models_list]
        except Exception:
            models = [model.get('name', '') for model in models_list]
    return web.json_response(models)


class OllamaConnectivityV2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:11434",
                    "tooltip": "The URL of the Ollama server. Default value points to a local instance with ollama's default port configuration."
                }),
                "model": ((), {"tooltip": "Select a model for inference. This is a list of available models on the Ollama server. If you don't see any, make sure the Ollama server is running on the url and there are models installed."}),
                "keep_alive": ("INT", {"default": 5, "min": -1, "max": 120, "step": 1, "tooltip": "Configures how long ollama keeps the model loaded in memory after inference. -1 = keep alive indefinitely, 0 = unload model immediately after inference"}),
                "keep_alive_unit": (["minutes", "hours"],),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "tooltip": "API key for remote providers (optional)"}),
            },
        }

    RETURN_TYPES = ("OLLAMA_CONNECTIVITY",)
    RETURN_NAMES = ("connection",)
    FUNCTION = "ollama_connectivity"
    CATEGORY = "Ollama"
    DESCRIPTION = "Provides connection to an Ollama server. Use the refresh button to load the model list in case of connection error or after installing a new model."

    def ollama_connectivity(self, url, model, keep_alive, keep_alive_unit, api_key=""):
        data = {
            "url": url,
            "model": model,
            "keep_alive": keep_alive,
            "keep_alive_unit": keep_alive_unit,
            "api_key": api_key,
        }
        return (data,)


class OllamaChat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "You are an AI artist.",
                        "tooltip": "System prompt - use this to set the role and general behavior of the model.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "What is art?",
                        "tooltip": "User prompt - a question or task you want the model to answer or perform. For vision tasks, you can refer to the input image as 'this image', 'photo' etc. like 'Describe this image in detail'",
                    },
                ),
                "think": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If enabled, the model will do a thinking process before answering. This can result in more accurate results. The thinking is then available as a separate output for debugging or understanding how the model arrived at its answer. Some models don't support this feature and the generation will fail.",
                    },
                ),
                "format": (
                    ["text", "json"],
                    {
                        "tooltip": "Output format of the response. 'text' will return a plain text response, while 'json' will return a structured response in JSON format. This is useful when the model is part of a larger pipeline and you need additional processing on the response. In this case I recommend showing the model example outputs in the system prompt. Some models are not trained to perform well in structured output.",
                    },
                ),
            },
            "optional": {
                "connectivity": (
                    "OLLAMA_CONNECTIVITY",
                    {
                        "forceInput": False,
                        "tooltip": "Set an ollama provider for the generation. If this input is empty, the 'meta' input must be set.",
                    },
                ),
                "options": (
                    "OLLAMA_OPTIONS",
                    {
                        "forceInput": False,
                        "tooltip": "Connect an Ollama Options node for advanced inference configuration.",
                    },
                ),
                "images": (
                    "IMAGE",
                    {
                        "forceInput": False,
                        "tooltip": "Provide an image or a batch of images for vision tasks. Make sure that the selected model supports vision, otherwise it may hallucinate the response.",
                    },
                ),
                "meta": (
                    "OLLAMA_META",
                    {
                        "forceInput": False,
                        "tooltip": "Use this input to chain multiple 'Ollama Generate' nodes. In this case the connectivity and options inputs are passed along.",
                    },
                ),
                "history": (
                    "OLLAMA_HISTORY",
                    {
                        "forceInput": False,
                        "tooltip": "Optionally set an existing model history, useful for multi-turn conversations, follow-up questions.",
                    },
                ),
                "reset_session": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Clear the conversation history. WARNING: If using shared history, this will affect all nodes using the same history ID.",
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "OLLAMA_META",
        "OLLAMA_HISTORY",
    )
    RETURN_NAMES = (
        "result",
        "thinking",
        "meta",
        "history",
    )
    FUNCTION = "ollama_chat"
    CATEGORY = "Ollama"
    DESCRIPTION = "Text generation with Ollama Chat. Supports vision tasks, multi-turn conversations, and advanced inference options. Connect an Ollama Connectivity node to set the server URL and model."

    def ollama_chat(
        self,
        system: str,
        prompt: str,
        think: bool,
        unique_id: str,
        format: str,
        options: dict[str, Any] | None = None,
        connectivity: dict[str, Any] | None = None,
        images: list[Any] | None = None,
        meta: dict[str, Any] | None = None,
        history: str | None = None,
        reset_session: bool = False,
    ) -> tuple[str | None, str | None, dict[str, Any], str | None]:

        if meta is None:
            if connectivity is None:
                raise ValueError("Either 'connectivity' or 'meta' must be provided.")
            meta = {}

        if connectivity is not None:
            meta["connectivity"] = connectivity
        if options is not None:
            meta["options"] = options
        else:
            meta["options"] = None

        if "connectivity" not in meta or meta["connectivity"] is None:
            raise ValueError("'connectivity' must be present in meta.")

        url = meta["connectivity"]["url"]
        model = meta["connectivity"]["model"]
        api_key = meta["connectivity"].get("api_key", "")

        debug_print = (
            True if meta["options"] is not None and meta["options"].get("debug", False) else False
        )

        ollama_format: Literal["", "json"] | JsonSchemaValue | None = None
        if format == "json":
            ollama_format = "json"
        elif format == "text":
            ollama_format = ""

        keep_alive_unit = (
            "m" if meta["connectivity"]["keep_alive_unit"] == "minutes" else "h"
        )
        request_keep_alive = str(meta["connectivity"]["keep_alive"]) + keep_alive_unit

        request_options = _filter_enabled_options(options)

        images_b64: list[str] | None = None
        if images is not None:
            images_b64 = []
            for batch_number, image in enumerate(images):
                i = 255.0 * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_bytes = base64.b64encode(buffered.getvalue()).decode("utf-8")
                images_b64.append(img_bytes)

        if debug_print:
            print(
                f"""
                    --- ollama chat request: 

                    url: {url}
                    model: {model}
                    system: {system}
                    prompt: {prompt}
                    images: {0 if images_b64 is None else len(images_b64)}
                    think: {think}
                    options: {request_options}
                    keep alive: {request_keep_alive}
                    format: {format}
                    ---------------------------------------------------------
                    """
            )

        session_key = history if history is not None else unique_id
        if reset_session:
            CHAT_SESSIONS[session_key] = ChatSession()
            if debug_print:
                print(f"Session {session_key} has been reset")
        if session_key not in CHAT_SESSIONS:
            CHAT_SESSIONS[session_key] = ChatSession()
        session = CHAT_SESSIONS[session_key]
        history = session_key

        if system:
            if session.messages and session.messages[0].get("role") == "system":
                session.messages[0] = {"role": "system", "content": system}
            else:
                session.messages.insert(0, {"role": "system", "content": system})

        user_message_for_history: dict[str, Any] = {
            "role": "user",
            "content": prompt,
        }
        session.messages.append(user_message_for_history)

        if debug_print:
            print("\n--- ollama chat session:")
            for message in session.messages:
                pprint(f"{message['role']}> {message['content'][:50]}...")
                if "images" in message:
                    for image in message["images"]:
                        pprint(f"Image: {image[:50]}...")
            print("---------------------------------------------------------")

        messages_for_api = [m.copy() for m in session.messages]
        if images_b64 is not None:
            messages_for_api[-1]["images"] = images_b64

        if api_key:
            req = urllib.request.Request(
                url + "/api/chat",
                data=json.dumps({
                    "model": model,
                    "messages": messages_for_api,
                    "options": request_options,
                    "keep_alive": request_keep_alive,
                    "format": ollama_format,
                    "stream": False,
                }).encode("utf-8")
            )
            req.add_header("Content-Type", "application/json")
            req.add_header("Authorization", f"Bearer {api_key}")
            with urllib.request.urlopen(req) as resp:
                response = json.loads(resp.read().decode("utf-8"))
        else:
            client = Client(host=url)
            response = client.chat(
                model=model,
                messages=messages_for_api,
                options=request_options,
                keep_alive=request_keep_alive,
                format=ollama_format,
            )

        if debug_print:
            print("\n--- ollama chat response:")
            pprint(response)
            print("---------------------------------------------------------")

        ollama_response_text = response.message.content if hasattr(response, 'message') else response.get('message', {}).get('content')
        ollama_response_thinking = response.message.thinking if hasattr(response, 'message') and think else None

        session.messages.append(
            {
                "role": "assistant",
                "content": ollama_response_text,
            }
        )

        return (
            ollama_response_text,
            ollama_response_thinking,
            meta,
            history,
        )


class Comfly_LLm_API:
    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.api_baseurl = baseurl
        self.timeout = 300
        # self.api_endpoint = f"{baseurl}/v1/chat/completions"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_baseurl": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": "gemini-3-pro-preview"}),
                "role": ("STRING", {"multiline": True, "default": "You are a helpful assistant"}),
                "prompt": ("STRING", {"multiline": True, "default": "describe the image"}),
                "temperature": ("FLOAT", {"default": 0.6}),
                "seed": ("INT", {"default": 100}),
            },
            "optional": {
                "ref_image": ("IMAGE",),
                "video": ("VIDEO",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("describe",)
    FUNCTION = "run_llmapi"
    CATEGORY = "RunNode/LLM"

    def _encode_image_b64(self, image_tensor):
        try:
            if hasattr(image_tensor, "cpu") and hasattr(image_tensor, "numpy"):
                if len(image_tensor.shape) == 4:
                    image_tensor = image_tensor[0]
                if image_tensor.shape[0] == 3:
                    image_tensor = image_tensor.permute(1, 2, 0)
                image_np = image_tensor.cpu().numpy()
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype("uint8")
                image = Image.fromarray(image_np)
            else:
                image = image_tensor
            original_size = image.size
            max_dimension = 1536
            if max(original_size) > max_dimension:
                ratio = max_dimension / max(original_size)
                new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format="JPEG", quality=75, optimize=True)
            img_byte_arr.seek(0)
            image_bytes = img_byte_arr.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            return f"data:image/jpeg;base64,{image_base64}"
        except Exception as e:
            return None

    def _encode_video_b64(self, video):
        try:
            video_bytes = None
            if isinstance(video, str) and os.path.exists(video):
                with open(video, "rb") as f:
                    video_bytes = f.read()
            elif hasattr(video, "video_path") and video.video_path and os.path.exists(video.video_path):
                with open(video.video_path, "rb") as f:
                    video_bytes = f.read()
            elif hasattr(video, "video_url") and video.video_url:
                resp = requests.get(video.video_url, stream=True, timeout=60)
                resp.raise_for_status()
                video_bytes = resp.content
            elif hasattr(video, "save_to"):
                temp_dir = os.path.join(os.getcwd(), "temp_llm_video")
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, "input.mp4")
                ok = video.save_to(temp_path)
                if ok and os.path.exists(temp_path):
                    with open(temp_path, "rb") as f:
                        video_bytes = f.read()
                    try:
                        os.remove(temp_path)
                    except:
                        pass
            if not video_bytes:
                return None
            return base64.b64encode(video_bytes).decode("utf-8")
        except Exception as e:
            return None

    def run_llmapi(self, api_baseurl, api_key, model, role, prompt, temperature, seed, ref_image=None, video=None):
        if api_key.strip():
            self.api_key = api_key
            # config = get_config()
            # config['api_key'] = api_key
            # save_config(config)
        else:
            self.api_key = get_config().get('api_key', '')

        if api_baseurl.strip():
             self.api_baseurl = api_baseurl
        else:
             self.api_baseurl = baseurl
        
        base = self.api_baseurl.strip()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        messages = []
        if video is not None:
            b64v = self._encode_video_b64(video)
            if not b64v:
                return ("Error: failed to encode video",)
            messages = [
                {"role": "system", "content": f"{role}"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{prompt}"},
                        {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64v}"}},
                    ],
                },
            ]
        elif ref_image is None:
            messages = [
                {"role": "system", "content": f"{role}"},
                {"role": "user", "content": f"{prompt}"},
            ]
        else:
            b64i = self._encode_image_b64(ref_image)
            if not b64i:
                return ("Error: failed to encode image",)
            messages = [
                {"role": "system", "content": f"{role}"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{prompt}"},
                        {"type": "image_url", "image_url": {"url": b64i}},
                    ],
                },
            ]
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if isinstance(seed, int):
            payload["seed"] = seed
        try:
            url = f"{base.rstrip('/')}/chat/completions"
            resp = requests.post(url, headers=headers, json=payload, timeout=300)
            if resp.status_code != 200:
                return (f"Error: {resp.status_code} {resp.text}",)
            data = resp.json()
            if data and "choices" in data and data["choices"]:
                content = data["choices"][0].get("message", {}).get("content", "")
                return (content or "",)
            return ("Error: empty response",)
        except Exception as e:
            return (f"Error calling API: {str(e)}",)
