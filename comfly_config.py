import os
import json
import glob



def get_config():
    try:
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config", "ComfyUI_RN_External_Interface-config.json")

        # 如果文件不存在，尝试查找大小写不敏感的文件名
        if not os.path.exists(config_path):
            config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config")
            pattern = os.path.join(config_dir, "*[Cc]onfig*[.]json")
            matching_files = [f for f in glob.glob(pattern) if "rn_external_interface" in f.lower()]
            if matching_files:
                config_path = matching_files[0]
        cfg = {}
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
        llm = cfg.get('llm', {})
        current_provider = llm.get('current_provider', '')
        providers = llm.get('providers', {})
        provider_cfg = providers.get(current_provider, {})

        def _env(names):
            for n in names:
                v = os.environ.get(n)
                if v is not None and str(v).strip() != "":
                    return v
            return None

        api_key = _env(["COMFLY_API_KEY", "COMFYUI_RN_API_KEY", "RUNNODE_API_KEY", "RN_API_KEY"]) or provider_cfg.get('api_key', '')
        base_url = _env(["COMFLY_BASE_URL", "COMFYUI_RN_BASE_URL", "RUNNODE_BASE_URL", "RN_BASE_URL"]) or provider_cfg.get('base_url', '')
        model = _env(["COMFLY_MODEL", "COMFYUI_RN_MODEL", "RUNNODE_MODEL", "RN_MODEL"]) or provider_cfg.get('model', '')

        def _env_float(names, default):
            v = _env(names)
            if v is None:
                return default
            try:
                return float(v)
            except Exception:
                return default

        def _env_int(names, default):
            v = _env(names)
            if v is None:
                return default
            try:
                return int(v)
            except Exception:
                return default

        temperature = _env_float(["COMFLY_TEMPERATURE", "COMFYUI_RN_TEMPERATURE"], provider_cfg.get('temperature', 0.7))
        max_tokens = _env_int(["COMFLY_MAX_TOKENS", "COMFYUI_RN_MAX_TOKENS"], provider_cfg.get('max_tokens', 1000))
        top_p = _env_float(["COMFLY_TOP_P", "COMFYUI_RN_TOP_P"], provider_cfg.get('top_p', 0.9))

        def _env_bool(names, default):
            v = _env(names)
            if v is None:
                return default
            return str(v).lower() in ("true", "1", "yes", "on")

        sora2_prioritize_v1 = _env_bool(
            ["SORA2_PRIORITIZE_V1", "COMFLY_SORA2_PRIORITIZE_V1", "COMFYUI_RN_SORA2_PRIORITIZE_V1", 
             "SORA2_V1_ENABLE", "COMFLY_SORA2_V1_ENABLE", "COMFYUI_RN_SORA2_V1_ENABLE"], 
            provider_cfg.get('sora2_prioritize_v1', provider_cfg.get('sora2_v1_enable', False))
        )
        
        # Sora2 独立 API 配置 (默认回退到通用配置)
        # 优先读取环境变量 -> 其次读取配置文件中 sora2 专属配置 -> 若为空字符串则回退到通用配置
        _sora2_base = _env(["SORA2_BASE_URL", "COMFLY_SORA2_BASE_URL"])
        if _sora2_base is None:
             _sora2_base = provider_cfg.get('sora2_base_url')
        
        sora2_base_url = _sora2_base if _sora2_base and str(_sora2_base).strip() else base_url

        _sora2_key = _env(["SORA2_API_KEY", "COMFLY_SORA2_API_KEY"])
        if _sora2_key is None:
            _sora2_key = provider_cfg.get('sora2_api_key')

        sora2_api_key = _sora2_key if _sora2_key and str(_sora2_key).strip() else api_key

        return {
            'api_key': api_key,
            'model': model,
            'base_url': base_url,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'top_p': top_p,
            'sora2_prioritize_v1': sora2_prioritize_v1,
            'sora2_base_url': sora2_base_url,
            'sora2_api_key': sora2_api_key,
        }
    except Exception:
        return {}

def save_config(config):
    raise PermissionError("save_config is disabled")


baseurl = get_config().get('base_url', '')
