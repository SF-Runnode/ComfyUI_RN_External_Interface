from .comfly_config import get_config, save_config, baseurl


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


class RN_OllamaConnectivityV2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Model name. For third-party APIs, use their model naming (e.g., gpt-4-vision-preview). For local Ollama, use installed model names."
                }),
                "keep_alive": ("INT", {"default": 5, "min": -1, "max": 120, "step": 1, "tooltip": "How long to keep the model loaded after inference. -1 = indefinitely, 0 = unload immediately"}),
                "keep_alive_unit": (["minutes", "hours"],),
            },
            "optional": {
                "url": ("STRING", {"default": "", "tooltip": "Ollama server URL. If empty, default from config is used."}),
                "api_key": ("STRING", {"default": "", "tooltip": "API key for third-party Ollama API. If empty, default from config is used."}),
            },
        }

    RETURN_TYPES = ("OLLAMA_CONNECTIVITY",)
    RETURN_NAMES = ("connection",)
    FUNCTION = "ollama_connectivity"
    CATEGORY = "RunNode"
    DESCRIPTION = "Provides connection to an Ollama server or third-party API. If no URL or API key is provided as inputs, defaults are loaded from configuration."

    def ollama_connectivity(self, model, keep_alive, keep_alive_unit, url="", api_key=""):
        cfg = get_config() or {}
        resolved_url = url if str(url or "").strip() else cfg.get("base_url", "")
        resolved_api_key = api_key if str(api_key or "").strip() else cfg.get("api_key", "")

        data = {
            "url": resolved_url,
            "model": model,
            "keep_alive": keep_alive,
            "keep_alive_unit": keep_alive_unit,
            "api_key": resolved_api_key,
        }

        return (data,)
