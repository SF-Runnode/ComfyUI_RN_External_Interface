import base64
from io import BytesIO
from PIL import Image
from .comfly_config import get_config, save_config, baseurl
from .utils import generate_request_id, log_prepare, log_complete, log_error, ProgressBar
import requests
import os



class Comfly_api_set:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_base": (["RunNode", "ip"], {"default": "RunNode"}),
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
        request_id = generate_request_id("api_cfg", "runnode")
        log_prepare("API设置", request_id, "RunNode-", "Config")
        global baseurl
        base_url_mapping = {
            "RunNode": baseurl,
            "ip": custom_ip
        }
        if api_base == "ip" and not custom_ip.strip():
            raise ValueError("When selecting 'ip' option, you must provide a custom IP address in the 'custom_ip' field")
        if api_base in base_url_mapping:
            baseurl = base_url_mapping[api_base]
        if apikey.strip():
            cfg = get_config()
            cfg['api_key'] = apikey
            save_config(cfg)
        log_complete("API设置", request_id, "Config", char_count=len(baseurl), elapsed_ms=0, source="RunNode-")
        return (apikey,)

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
        request_id = generate_request_id("llm_chat", "runnode")
        log_prepare("LLM对话", request_id, "RunNode/LLM-", "LLM", model_name=model)
        rn_pbar = ProgressBar(request_id, "LLM", streaming=True, task_type="LLM对话", source="RunNode/LLM-")
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
                rn_pbar.error(f"Error: {resp.status_code} {resp.text}")
                return (f"Error: {resp.status_code} {resp.text}",)
            data = resp.json()
            if data and "choices" in data and data["choices"]:
                content = data["choices"][0].get("message", {}).get("content", "")
                rn_pbar.done(char_count=len(content or ""))
                return (content or "",)
            rn_pbar.error("Error: empty response")
            return ("Error: empty response",)
        except Exception as e:
            rn_pbar.error(f"Error calling API: {str(e)}")
            return (f"Error calling API: {str(e)}",)
