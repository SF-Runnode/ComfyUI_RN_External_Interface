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
            existing = json.load(f)
        llm = existing.get('llm', {})
        current_provider = llm.get('current_provider', '')
        providers = llm.get('providers', {})
        if current_provider in providers:
            for k in ['api_key', 'model', 'base_url', 'temperature', 'max_tokens', 'top_p']:
                if k in config:
                    providers[current_provider][k] = config[k]
            existing['llm']['providers'] = providers
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

baseurl = get_config().get('base_url', '')

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


class ComflyVideoAdapter:
    def __init__(self, video_path_or_url):
        if video_path_or_url.startswith('http'):
            self.is_url = True
            self.video_url = video_path_or_url
            self.video_path = None
        else:
            self.is_url = False
            self.video_path = video_path_or_url
            self.video_url = None
        
    def get_dimensions(self):
        if self.is_url:
            return 1280, 720
        else:
            try: 
                cap = cv2.VideoCapture(self.video_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                return width, height
            except Exception as e:
                print(f"Error getting video dimensions: {str(e)}")
                return 1280, 720
            
    def save_to(self, output_path, format="auto", codec="auto", metadata=None):
        if self.is_url:
            try:
                response = requests.get(self.video_url, stream=True)
                response.raise_for_status()
                
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            except Exception as e:
                print(f"Error downloading video from URL: {str(e)}")
                return False
        else:
            try:
                shutil.copyfile(self.video_path, output_path)
                return True
            except Exception as e:
                print(f"Error saving video: {str(e)}")
                return False


def create_audio_object(audio_url):
    """Create an audio object compatible with ComfyUI's audio nodes"""
    if not audio_url:
        return {
            "waveform": torch.zeros((1, 1, 44100)),  
            "sample_rate": 44100
        }
        
    try:
        temp_dir = os.path.join(folder_paths.get_temp_directory(), "suno_audio")
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, f"suno_{str(uuid.uuid4())[:8]}.mp3")
        
        response = requests.get(audio_url, stream=True)
        response.raise_for_status()
        
        with open(temp_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        try:
            waveform, sample_rate = torchaudio.load(temp_file)
            if len(waveform.shape) == 2:  
                waveform = waveform.unsqueeze(0)  
            
            return {
                "waveform": waveform,
                "sample_rate": sample_rate
            }
        except Exception as e:
            print(f"Error loading audio with torchaudio: {str(e)}")

            try:
                if hasattr(folder_paths, "get_ffmpeg_path"):
                    ffmpeg_path = folder_paths.get_ffmpeg_path()
                else:
                    ffmpeg_path = shutil.which("ffmpeg")
                
                if ffmpeg_path:
                    temp_wav = temp_file.replace(".mp3", ".wav")
                    subprocess.run([ffmpeg_path, "-y", "-i", temp_file, temp_wav], 
                                  check=True, capture_output=True)

                    waveform, sample_rate = torchaudio.load(temp_wav)
                    if len(waveform.shape) == 2:  
                        waveform = waveform.unsqueeze(0)  

                    try:
                        os.remove(temp_wav)
                    except:
                        pass
                        
                    return {
                        "waveform": waveform,
                        "sample_rate": sample_rate
                    }
                else:
                    raise Exception("ffmpeg not found, can't process audio")
            except Exception as ffmpeg_error:
                print(f"Error with ffmpeg conversion: {str(ffmpeg_error)}")

                return {
                    "waveform": torch.zeros((1, 1, 44100)),  
                    "sample_rate": 44100,
                    "url": audio_url  
                }
        
    except Exception as e:
        print(f"Error downloading or processing audio: {str(e)}")

    return {
        "waveform": torch.zeros((1, 1, 44100)),
        "sample_rate": 44100
    }


baseurl = get_config().get('base_url', '')