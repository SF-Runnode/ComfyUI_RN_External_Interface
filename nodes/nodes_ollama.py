from ..comfly_config import *
from .__init__ import *


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
    req_id = generate_request_id("models", "ollama")
    start = time.perf_counter()
    models: list[str] = []
    try:
        data = await request.json()
        raw_url = (data.get("url") or "").strip()
        raw_api_key = (data.get("api_key", "") or "").strip()

        url = raw_url or os.environ.get("COMFYUI_RN_BASE_URL") or get_config().get("base_url", "http://127.0.0.1:11434")
        api_key = raw_api_key or os.environ.get("COMFYUI_RN_API_KEY") or get_config().get("api_key", "")

        if api_key:
            req = urllib.request.Request(url.rstrip("/") + "/api/tags")
            req.add_header("Content-Type", "application/json")
            req.add_header("Authorization", f"Bearer {api_key}")
            with urllib.request.urlopen(req) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            models_list = payload.get("models", [])
            models = [m.get("model") or m.get("name", "") for m in models_list if isinstance(m, dict)]
        else:
            client = Client(host=url)
            models_list = client.list().get("models", [])
            models = [m.get("model") or m.get("name", "") for m in models_list if isinstance(m, dict)]
    except Exception:
        log_backend_exception(
            "ollama_get_models_failed",
            request_id=req_id,
            url=safe_public_url(locals().get("url", "")),
        )
        models = []
    finally:
        log_backend(
            "ollama_get_models_done",
            request_id=req_id,
            url=safe_public_url(locals().get("url", "")),
            model_count=len(models),
            elapsed_ms=int((time.perf_counter() - start) * 1000),
        )
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
                    "default": "",
                    "tooltip": "服务地址：本地 Ollama 用 http://127.0.0.1:11434；第三方服务无需填写“/v1”，系统会自动处理。"
                }),
                "model": ("STRING", {"multiline": False, "default": "", "tooltip": "Model name. Supports manual input. Use installed model names for local Ollama or third-party provider naming."}),
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
    CATEGORY = "RunNode/Ollama"
    DESCRIPTION = "Provides connection to an Ollama server. Use the refresh button to load the model list in case of connection error or after installing a new model."

    def ollama_connectivity(self, url, model, keep_alive, keep_alive_unit, api_key=""):
        if not url or str(url).strip() == "":
            url = os.environ.get('COMFYUI_RN_BASE_URL') or get_config().get('base_url', "http://127.0.0.1:11434")
        if not api_key or str(api_key).strip() == "":
            api_key = os.environ.get('COMFYUI_RN_API_KEY') or get_config().get('api_key', "")
        if not model or str(model).strip() == "":
            model = os.environ.get('COMFYUI_RN_MODEL') or get_config().get('model', "")
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
    CATEGORY = "RunNode/Ollama"
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

        url = meta["connectivity"]["url"] or os.environ.get("COMFYUI_RN_BASE_URL") or get_config().get("base_url", "http://127.0.0.1:11434")
        model = meta["connectivity"]["model"] or os.environ.get("COMFYUI_RN_MODEL") or get_config().get("model", "")
        api_key = meta["connectivity"].get("api_key", "") or os.environ.get("COMFYUI_RN_API_KEY") or get_config().get("api_key", "")

        req_id = generate_request_id("chat", "ollama")
        task_type = "图文对话" if images is not None else "文本生成"
        service_label = format_service_label("Ollama", url, bool(api_key))
        log_prepare(task_type, req_id, "RunNode/Ollama-", service_label, model_name=model)
        rn_pbar = ProgressBar(req_id, service_label, streaming=True, task_type=task_type, source="RunNode/Ollama-")
        rn_pbar.set_generating()

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

        options_summary = None
        if request_options:
            options_summary = {}
            for k, v in request_options.items():
                if isinstance(v, (int, float, bool)) or v is None:
                    options_summary[k] = v
                elif isinstance(v, (list, tuple)):
                    options_summary[k] = f"list({len(v)})"
                else:
                    options_summary[k] = "set"

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
            log_backend(
                "ollama_chat_request_debug",
                level="DEBUG",
                request_id=req_id,
                url=safe_public_url(url),
                model=model,
                prompt_len=len(prompt or ""),
                system_len=len(system or ""),
                image_count=(0 if images_b64 is None else len(images_b64)),
                think=think,
                options=options_summary,
                keep_alive=request_keep_alive,
                format=format,
            )

        session_key = history if history is not None else unique_id
        if reset_session:
            CHAT_SESSIONS[session_key] = ChatSession()
            if debug_print:
                log_backend(
                    "ollama_chat_session_reset",
                    level="DEBUG",
                    request_id=req_id,
                    session_key=session_key,
                )
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
            log_backend(
                "ollama_chat_session_debug",
                level="DEBUG",
                request_id=req_id,
                session_key=session_key,
                message_count=len(session.messages),
                image_count=(0 if images_b64 is None else len(images_b64)),
            )

        messages_for_api = [m.copy() for m in session.messages]
        if images_b64 is not None:
            messages_for_api[-1]["images"] = images_b64
        ollama_response_text = None
        ollama_response_thinking = None
        if api_key:
            remote_start = time.perf_counter()
            try:
                log_backend(
                    "ollama_chat_remote_start",
                    request_id=req_id,
                    url=safe_public_url(url),
                    model=model,
                )
                openai_messages = []
                for msg in messages_for_api:
                    if "images" in msg and msg["images"]:
                        vision_content = []
                        if msg.get("content"):
                            vision_content.append({"type": "text", "text": msg["content"]})
                        for img_b64 in msg["images"]:
                            vision_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
                        openai_messages.append({"role": msg["role"], "content": vision_content})
                    else:
                        openai_messages.append({"role": msg["role"], "content": msg.get("content", "")})
                request_body = {"model": model, "messages": openai_messages, "stream": False}
                if request_options:
                    if "temperature" in request_options:
                        request_body["temperature"] = request_options["temperature"]
                    if "max_tokens" in request_options:
                        request_body["max_tokens"] = request_options["max_tokens"]
                    elif "num_predict" in request_options:
                        request_body["max_tokens"] = request_options["num_predict"]
                base = url.rstrip("/")
                endpoint = (base + "/chat/completions") if (base.endswith("/v1") or "/v1/" in base) else (base + "/v1/chat/completions")
                req = urllib.request.Request(
                    endpoint,
                    data=json.dumps(request_body).encode("utf-8"),
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
                )
                with urllib.request.urlopen(req) as resp:
                    response_data = json.loads(resp.read().decode("utf-8"))
                ollama_response_text = response_data["choices"][0]["message"]["content"]
                ollama_response_thinking = None
                log_backend(
                    "ollama_chat_remote_done",
                    request_id=req_id,
                    url=safe_public_url(url),
                    model=model,
                    mode="openai_compat",
                    response_len=len(ollama_response_text or ""),
                    elapsed_ms=int((time.perf_counter() - remote_start) * 1000),
                )
            except Exception as e:
                log_backend_exception(
                    "ollama_chat_remote_openai_compat_failed",
                    request_id=req_id,
                    url=safe_public_url(url),
                    model=model,
                )
                try:
                    req = urllib.request.Request(
                        url.rstrip("/") + "/api/chat",
                        data=json.dumps({
                            "model": model,
                            "messages": messages_for_api,
                            "options": request_options,
                            "keep_alive": request_keep_alive,
                            "format": ollama_format,
                            "stream": False,
                        }).encode("utf-8"),
                        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                    )
                    with urllib.request.urlopen(req) as resp:
                        response_json = json.loads(resp.read().decode("utf-8"))
                    ollama_response_text = response_json.get("message", {}).get("content")
                    ollama_response_thinking = None
                    log_backend(
                        "ollama_chat_remote_done",
                        request_id=req_id,
                        url=safe_public_url(url),
                        model=model,
                        mode="ollama_api_chat",
                        response_len=len(ollama_response_text or ""),
                        elapsed_ms=int((time.perf_counter() - remote_start) * 1000),
                    )
                except Exception as e2:
                    error_msg = f"远程服务调用失败: {format_runnode_error(str(e2))}"
                    if isinstance(e2, urllib.error.HTTPError):
                        try:
                            err_body = e2.read().decode('utf-8')
                            error_msg = f"远程服务调用失败: {format_runnode_error(err_body)}"
                        except:
                            pass

                    rn_pbar.error(f"{error_msg} (建议: 检查 Base URL 是否可访问，API Key 是否正确)")
                    log_error("远程服务调用失败", req_id, error_msg, "RunNode/Ollama-", service_label)
                    log_backend_exception(
                        "ollama_chat_remote_failed",
                        request_id=req_id,
                        url=safe_public_url(url),
                        model=model,
                    )
                    raise RuntimeError(
                        f"{error_msg} (建议: 检查 Base URL 是否可访问，API Key 是否正确)"
                    )
        else:
            try:
                log_backend(
                    "ollama_chat_local_start",
                    request_id=req_id,
                    url=safe_public_url(url),
                    model=model,
                )
                client = Client(host=url)
                response_native = client.chat(
                    model=model,
                    messages=messages_for_api,
                    options=request_options,
                    keep_alive=request_keep_alive,
                    format=ollama_format,
                )
                ollama_response_text = response_native.message.content if hasattr(response_native, "message") else None
                ollama_response_thinking = response_native.message.thinking if hasattr(response_native, "message") and think else None
            except Exception as e:
                log_backend_exception(
                    "ollama_chat_local_failed",
                    request_id=req_id,
                    url=safe_public_url(url),
                    model=model,
                )
                fallback_url = get_config().get("base_url", "http://127.0.0.1:11434")
                try:
                    client = Client(host=fallback_url)
                    response_native = client.chat(
                        model=model,
                        messages=messages_for_api,
                        options=request_options,
                        keep_alive=request_keep_alive,
                        format=ollama_format,
                    )
                    ollama_response_text = response_native.message.content if hasattr(response_native, "message") else None
                    ollama_response_thinking = response_native.message.thinking if hasattr(response_native, "message") and think else None
                except Exception:
                    rn_pbar.error(
                        format_user_error(
                            "本地 Ollama 调用失败",
                            request_id=req_id,
                            suggestion="检查服务是否启动、URL 是否正确、模型是否已拉取",
                        )
                    )
                    log_error(
                        "本地服务调用失败",
                        req_id,
                        format_user_error(
                            "本地 Ollama 调用失败",
                            request_id=req_id,
                            suggestion="检查服务是否启动、URL 是否正确、模型是否已拉取",
                        ),
                        "RunNode/Ollama-",
                        service_label,
                    )
                    log_backend_exception(
                        "ollama_chat_local_fallback_failed",
                        request_id=req_id,
                        url=safe_public_url(fallback_url),
                        model=model,
                    )
                    raise RuntimeError(
                        format_user_error(
                            "本地 Ollama 调用失败",
                            request_id=req_id,
                            suggestion="检查服务是否启动、URL 是否正确、模型是否已拉取",
                        )
                    )

        try:
            rn_pbar.done(char_count=len(ollama_response_text or ""))
        except Exception:
            pass

        if debug_print:
            log_backend(
                "ollama_chat_response_debug",
                level="DEBUG",
                request_id=req_id,
                text_len=len(ollama_response_text or ""),
                has_thinking=bool(ollama_response_thinking),
            )

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
