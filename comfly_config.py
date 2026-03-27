import os
import json
import glob
from typing import Dict



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

        _sora2_new_base = _env(["SORA2_NEW_BASE_URL", "COMFLY_SORA2_NEW_BASE_URL"])
        if _sora2_new_base is None:
             _sora2_new_base = provider_cfg.get('sora2_new_base_url')
        
        sora2_new_base_url = _sora2_new_base if _sora2_new_base and str(_sora2_new_base).strip() else base_url

        sora2_new_api_key = _env(["SORA2_NEW_API_KEY", "COMFLY_SORA2_NEW_API_KEY"]) or provider_cfg.get('sora2_new_api_key', '')

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
            'sora2_new_base_url': sora2_new_base_url,
            'sora2_new_api_key': sora2_new_api_key,
        }
    except Exception:
        return {}

def save_config(config):
    raise PermissionError("save_config is disabled")


baseurl = get_config().get('base_url', '')


# ============== 计费配置 ==============

_billing_config_cache = None


def get_billing_config() -> Dict:
    """
    获取计费配置

    支持从以下位置加载（按优先级）:
    1. 环境变量 BILLING_CONFIG_PATH 指定的路径
    2. 挂载路径 config/billing_config.json
    3. 默认配置 config/billing_config.json

    Docker/K8s 部署时，可通过挂载新的 billing_config.json 实现价格更新，
    无需重新构建镜像。
    """
    global _billing_config_cache

    if _billing_config_cache is not None:
        return _billing_config_cache

    # 优先从环境变量读取
    env_path = os.environ.get("BILLING_CONFIG_PATH")
    if env_path and os.path.exists(env_path):
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                _billing_config_cache = json.load(f)
                return _billing_config_cache
        except Exception as e:
            print(f"[Billing] Warning: Failed to load billing config from env path {env_path}: {e}")

    # 默认配置路径
    default_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "config",
        "billing_config.json"
    )

    # 尝试多个可能的文件名（兼容大小写和不同命名）
    config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config")
    config_patterns = [
        "billing_config.json",
        "Billing_Config.json",
        "billing-config.json",
        "*billing*config*.json",
        "*Billing*Config*.json",
    ]

    config_path = None
    if os.path.exists(default_path):
        config_path = default_path
    else:
        import fnmatch
        for pattern in config_patterns:
            full_pattern = os.path.join(config_dir, pattern)
            matches = [f for f in glob.glob(full_pattern) if os.path.isfile(f)]
            if matches:
                config_path = matches[0]
                break

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                _billing_config_cache = json.load(f)
                print(f"[Billing] Loaded billing config from: {config_path}")
                return _billing_config_cache
        except Exception as e:
            print(f"[Billing] Warning: Failed to load billing config from {config_path}: {e}")

    # 返回空配置（禁用计费）
    print("[Billing] Warning: No billing config found, billing disabled")
    _billing_config_cache = {"models": {}, "display_settings": {"show_estimate_badge": False}}
    return _billing_config_cache


def reload_billing_config():
    """重新加载计费配置（用于热更新）"""
    global _billing_config_cache
    _billing_config_cache = None
    return get_billing_config()


# ============== 模型名称映射配置 ==============

_models_config_cache = None


def load_models_config() -> Dict:
    """
    加载模型名称映射配置（统一的 display_name_mapping 和 api_name_mapping）
    """
    global _models_config_cache

    if _models_config_cache is not None:
        return _models_config_cache

    try:
        config_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "config",
            "models_config.json"
        )

        if not os.path.exists(config_path):
            print(f"[Models] Config file not found: {config_path}")
            _models_config_cache = {"display_name_mapping": {}, "api_name_mapping": {}}
            return _models_config_cache

        with open(config_path, 'r', encoding='utf-8') as f:
            _models_config_cache = json.load(f)
            print(f"[Models] Loaded config from: {config_path}")
            return _models_config_cache
    except Exception as e:
        print(f"[Models] Error loading models config: {e}")
        _models_config_cache = {"display_name_mapping": {}, "api_name_mapping": {}}
        return _models_config_cache


def get_api_model_name(friendly_name: str) -> str:
    """
    根据友好显示名称获取实际 API 模型名称
    """
    mapping = load_models_config().get("api_name_mapping", {})
    return mapping.get(friendly_name, friendly_name)


def get_display_name(internal_name: str) -> str:
    """
    根据内部模型名称获取友好显示名称
    """
    mapping = load_models_config().get("display_name_mapping", {})
    return mapping.get(internal_name, internal_name)

