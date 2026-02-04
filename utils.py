import numpy as np
import torch
import os
import requests
import shutil
import cv2
import uuid
import subprocess
import torchaudio
import folder_paths
from PIL import Image
from typing import List, Union

def pil2tensor(image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    if isinstance(image, list):
        if len(image) == 0:
            return torch.empty(0)
        return torch.cat([pil2tensor(img) for img in image], dim=0)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    img_array = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(img_array)[None,]

def tensor2pil(image: torch.Tensor) -> List[Image.Image]:
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2pil(image[i]))
        return out
    numpy_image = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    return [Image.fromarray(numpy_image)]

 

def format_runnode_error(response):
    """
    Format API error response into a human-readable string.
    Handles recursive JSON parsing to extract the actual error message.
    """
    try:
        data = None
        status_code = 'Unknown'

        # If it's a requests.Response object
        if hasattr(response, 'json') and callable(response.json):
            try:
                data = response.json()
            except:
                text = getattr(response, 'text', '')
                status_code = getattr(response, 'status_code', 'Unknown')
                if not text.strip():
                    return f"API Error: {status_code}"
                return f"API Error: {status_code} - {text}"
            
            status_code = getattr(response, 'status_code', 'Unknown')
        
        # If it's a dict (already parsed JSON)
        elif isinstance(response, dict):
            data = response
            status_code = data.get('status', 'Unknown')
            
        # If it's a string, try to parse it as JSON
        elif isinstance(response, str):
            try:
                if response.strip().startswith('{'):
                    data = json.loads(response)
            except:
                pass
            
            if data is None:
                return response
        
        else:
            return str(response)

        def recursive_extract(err_data):
            if isinstance(err_data, str):
                try:
                    # Only attempt to parse if it looks like a JSON object
                    if err_data.strip().startswith('{'):
                        parsed = json.loads(err_data)
                        return recursive_extract(parsed)
                except:
                    pass
                return err_data
            
            if isinstance(err_data, dict):
                # Prioritize 'message' or 'error' keys
                if "error" in err_data:
                    return recursive_extract(err_data["error"])
                if "message" in err_data:
                    return recursive_extract(err_data["message"])
                if "fail_reason" in err_data:
                    return recursive_extract(err_data["fail_reason"])
                if "err_code" in err_data:
                    return recursive_extract(err_data["err_code"])
                if "detail" in err_data:
                    return recursive_extract(err_data["detail"])
                if "base_resp" in err_data:
                    base_resp = err_data["base_resp"]
                    if isinstance(base_resp, dict) and "status_msg" in base_resp:
                        return recursive_extract(base_resp["status_msg"])
                    return recursive_extract(base_resp)
                if "status_msg" in err_data:
                    return recursive_extract(err_data["status_msg"])
                # If no known keys, return string representation of dict
            
            return str(err_data)
            
        message = recursive_extract(data)
        if status_code != 'Unknown':
             return f"API Error: {status_code} - {message}"
        else:
             return str(message)

    except Exception as e:
        return f"Error formatting error: {str(e)}"


class EmptyVideoAdapter:
    """Empty video adapter for error cases"""
    def __init__(self):
        self.is_empty = True
        
    def get_dimensions(self):
        return 1, 1  # Minimal dimensions
    
    def save_to(self, output_path, format="auto", codec="auto", metadata=None):
        # Create a minimal black video file
        try:
            import numpy as np
            # Create a 1x1 black frame
            frame = np.zeros((1, 1, 3), dtype=np.uint8)
            # Write a minimal video using opencv
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 1.0, (1, 1))
            out.write(frame)
            out.release()
            return True
        except:
            return False


def create_audio_object(audio_url):
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
                    subprocess.run([ffmpeg_path, "-y", "-i", temp_file, temp_wav], check=True, capture_output=True)
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
        self._cached_path = None

    def _ensure_local_file(self):
        if not self.is_url and self.video_path:
             return self.video_path
        
        if self._cached_path and os.path.exists(self._cached_path):
             return self._cached_path
             
        if self.is_url:
             try:
                 import folder_paths
                 temp_dir = os.path.join(folder_paths.get_temp_directory(), "runnode_video_cache")
                 os.makedirs(temp_dir, exist_ok=True)
                 ext = self.video_url.split('.')[-1] if '.' in self.video_url else 'mp4'
                 if len(ext) > 4: ext = 'mp4'
                 filename = f"cached_{str(uuid.uuid4())[:8]}.{ext}"
                 temp_path = os.path.join(temp_dir, filename)
                 
                 response = requests.get(self.video_url, stream=True)
                 response.raise_for_status()
                 with open(temp_path, "wb") as f:
                     for chunk in response.iter_content(chunk_size=8192):
                         f.write(chunk)
                 self._cached_path = temp_path
                 return temp_path
             except Exception as e:
                 print(f"Error downloading video for components: {e}")
                 return None
        return None

    def get_components(self):
        local_path = self._ensure_local_file()
        if not local_path:
            return EmptyVideoAdapter().get_components()
            
        try:
            import comfy_api.latest as comfy_io
            return comfy_io.InputImpl.VideoFromFile(local_path).get_components()
        except ImportError:
            # Fallback for when comfy_api is not available (e.g. older ComfyUI)
            # We try to use load_video from nodes_video if possible or manual
            print("Warning: comfy_api not found, falling back to basic component extraction")
            # Minimal fallback
            cap = cv2.VideoCapture(local_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0: fps = 30.0
            
            # Read all frames (memory intensive but consistent with get_components)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
            cap.release()
            
            if not frames:
                return EmptyVideoAdapter().get_components()
                
            images = pil2tensor(frames)
            
            # Simple dummy object
            class ManualVideoComponents:
                def __init__(self, img, fps):
                    self.images = img
                    self.frame_rate = fps
                    self.audio = None
            
            return ManualVideoComponents(images, fps)

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

import sys
import io
import time
import threading
import json
import logging
from urllib.parse import urlsplit, urlunsplit

if sys.platform == 'win32' and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass


_RN_LOGGER = logging.getLogger("RunNode")
_RN_LOG_LEVEL = os.environ.get("RUNNODE_LOG_LEVEL", "INFO").upper()
if not _RN_LOGGER.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    _RN_LOGGER.addHandler(_handler)
_RN_LOGGER.setLevel(getattr(logging, _RN_LOG_LEVEL, logging.INFO))
_RN_LOGGER.propagate = False

PREFIX = "✨"
ERROR_PREFIX = "✨-❌"
PROCESS_PREFIX = "✨"
REQUEST_PREFIX = "✨"
WARN_PREFIX = "✨-⚠️"

_ANSI_CLEAR_EOL = "\033[K"
_global_last_output_len = 0
_progress_lock = threading.Lock()
_streaming_progress_enabled = (os.environ.get("RUNNODE_STREAMING_PROGRESS", "true").lower() != "false")

def _enable_windows_vt():
    if os.name == 'nt':
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_ulong()
            kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:
            pass

_enable_windows_vt()

def is_streaming_progress_enabled():
    return _streaming_progress_enabled

def set_streaming_progress_enabled(enabled: bool):
    global _streaming_progress_enabled
    _streaming_progress_enabled = bool(enabled)

def get_display_width(text: str) -> int:
    w = 0
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff' or '\u3000' <= ch <= '\u303f':
            w += 2
        else:
            w += 1
    return w

def safe_public_url(url: str) -> str:
    if not url:
        return url
    try:
        parts = urlsplit(url)
        if '@' in parts.netloc:
            # Hide auth info
            netloc = parts.netloc.split('@')[-1]
            parts = parts._replace(netloc=netloc)
            return urlunsplit(parts)
    except:
        pass
    return url

def generate_request_id(task_type: str, provider: str) -> str:
    import uuid
    short_uuid = str(uuid.uuid4())[:8]
    return f"rn_{provider}_{task_type}_{short_uuid}"

def log_backend(event_type: str, **kwargs):
    # This is a placeholder for backend logging integration
    # In a real scenario, this might send data to a telemetry server
    pass

def log_backend_exception(event_type: str, **kwargs):
    # Placeholder for exception logging
    if "error" not in kwargs and "exception" not in kwargs:
        import traceback
        kwargs["traceback"] = traceback.format_exc()
    pass

def log_prepare(task_name, request_id, prefix, service_name, model_version=None, speed=None, **kwargs):
    info = f" {kwargs}" if kwargs else ""
    if model_version: info += f" model_version={model_version}"
    if speed: info += f" speed={speed}"
    print(f"{prefix} [{task_name}] {request_id} Preparing...{info}")

def log_complete(task_name, request_id, prefix, service_name, image_url=None, **kwargs):
    info = f" {kwargs}" if kwargs else ""
    if image_url: info += f" image_url={image_url}"
    print(f"{prefix} [{task_name}] {request_id} Completed.{info}")

def log_error(task_name, request_id, message, prefix, service_name):
    print(f"{prefix} [{task_name}] {request_id} Error: {message}")

class ProgressBar:
    def __init__(self, request_id, service_name, extra_info="", streaming=True, task_type="Task", source="RunNode"):
        self.request_id = request_id
        self.service_name = service_name
        self.streaming = streaming
        self.last_update = time.time()
    
    def update_absolute(self, value):
        if time.time() - self.last_update > 0.5:
            print(f"Progress: {value}%")
            self.last_update = time.time()
            
    def update(self, value):
        self.update_absolute(value)

    def set_generating(self):
        if self.streaming:
            print(f"Generating...")

    def error(self, message):
        print(f"Error: {message}")
        
    def done(self, char_count=0, elapsed_ms=0):
        print(f"Done in {elapsed_ms}ms")
