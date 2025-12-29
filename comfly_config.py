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
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        llm = cfg.get('llm', {})
        current_provider = llm.get('current_provider', '')
        providers = llm.get('providers', {})
        provider_cfg = providers.get(current_provider, {})
        return {
            'api_key': provider_cfg.get('api_key', ''),
            'model': provider_cfg.get('model', ''),
            'base_url': provider_cfg.get('base_url', ''),
            'temperature': provider_cfg.get('temperature', 0.7),
            'max_tokens': provider_cfg.get('max_tokens', 1000),
            'top_p': provider_cfg.get('top_p', 0.9)
        }
    except Exception:
        return {}

def save_config(config):
    raise PermissionError("save_config is disabled")


baseurl = get_config().get('base_url', '')
