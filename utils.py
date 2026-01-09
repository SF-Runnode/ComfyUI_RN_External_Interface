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
        w += 2 if ord(ch) > 0x7F else 1
    return w

def format_elapsed_time(elapsed_ms: int) -> str:
    return f"{elapsed_ms/1000:.1f}s"

def generate_request_id(req_type: str, service_type: str = None, node_id: str = "0") -> str:
    ts = str(int(time.time()))[-4:]
    parts = [req_type]
    if service_type:
        parts.append(service_type)
    parts.append(str(node_id))
    parts.append(ts)
    return "_".join(parts)


def safe_public_url(url: str) -> str:
    if not url:
        return ""
    try:
        s = urlsplit(url)
        host = s.hostname or ""
        if not host:
            return url
        netloc = host
        if s.port:
            netloc = f"{host}:{s.port}"
        return urlunsplit((s.scheme, netloc, "", "", ""))
    except Exception:
        return url


def url_hostport(url: str) -> str:
    if not url:
        return ""
    try:
        s = urlsplit(url)
        host = s.hostname or ""
        if not host:
            return ""
        if s.port:
            return f"{host}:{s.port}"
        return host
    except Exception:
        return ""


def format_service_label(base: str, url: str, is_remote: bool) -> str:
    base = (base or "").strip() or "服务"
    hp = url_hostport(url)
    if is_remote:
        return f"{base}(远程API {hp})" if hp else f"{base}(远程API)"
    return f"{base}(本地 {hp})" if hp else f"{base}(本地)"


def _to_json(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


def log_backend(event: str, level: str = "INFO", **fields) -> None:
    payload = {"event": event, "ts_ms": int(time.time() * 1000)}
    for k, v in fields.items():
        if v is None:
            continue
        payload[k] = v
    msg = _to_json(payload)
    lvl = (level or "INFO").upper()
    if lvl == "DEBUG":
        _RN_LOGGER.debug(msg)
    elif lvl == "WARNING" or lvl == "WARN":
        _RN_LOGGER.warning(msg)
    elif lvl == "ERROR":
        _RN_LOGGER.error(msg)
    else:
        _RN_LOGGER.info(msg)


def log_backend_exception(event: str, **fields) -> None:
    payload = {"event": event, "ts_ms": int(time.time() * 1000)}
    for k, v in fields.items():
        if v is None:
            continue
        payload[k] = v
    _RN_LOGGER.exception(_to_json(payload))


def format_user_error(message: str, request_id: str | None = None, suggestion: str | None = None) -> str:
    msg = (message or "").strip() or "发生错误"
    sug = (suggestion or "").strip()
    rid = (request_id or "").strip()
    parts = []
    if rid:
        parts.append(f"[{rid}]")
    parts.append(msg)
    if sug:
        parts.append(f"建议：{sug}")
    return " ".join(parts)

def log_prepare(task_type: str, request_id: str, source: str, service_name: str, model_name: str = None, rule_name: str = None, extra: dict = None) -> None:
    if bool(getattr(sys.stdout, "isatty", lambda: False)()):
        print(f"\r{_ANSI_CLEAR_EOL}", end="")
    parts = [f"{PREFIX} 🟡 {source}{task_type}准备", f"服务:{service_name}"]
    if model_name:
        parts.append(f"模型:{model_name}")
    if rule_name:
        parts.append(f"规则:{rule_name}")
    parts.append(f"ID:{request_id}")
    if extra:
        for k, v in extra.items():
            parts.append(f"{k}:{v}")
    print(f"{parts[0]} | {' | '.join(parts[1:])}", flush=True)
    log_backend(
        "task_prepare",
        level="INFO",
        task_type=task_type,
        request_id=request_id,
        source=source,
        service=service_name,
        model=model_name,
        rule=rule_name,
        extra=extra,
    )

def log_complete(task_type: str, request_id: str, service_name: str, char_count: int, elapsed_ms: int, source: str = None) -> None:
    if bool(getattr(sys.stdout, "isatty", lambda: False)()):
        print(f"\r{_ANSI_CLEAR_EOL}", end="")
    src = source if source else ""
    parts = [f"{PREFIX} ✅ {src}{task_type}完成", f"服务:{service_name}", f"ID:{request_id}", f"字符:{char_count}", f"耗时:{format_elapsed_time(elapsed_ms)}"]
    print(f"{parts[0]} | {' | '.join(parts[1:])}", flush=True)
    log_backend(
        "task_complete",
        level="INFO",
        task_type=task_type,
        request_id=request_id,
        source=source,
        service=service_name,
        char_count=char_count,
        elapsed_ms=elapsed_ms,
    )

def log_error(task_type: str, request_id: str, error_msg: str, source: str = None) -> None:
    if bool(getattr(sys.stdout, "isatty", lambda: False)()):
        print(f"\r{_ANSI_CLEAR_EOL}", end="")
    src = source if source else ""
    print(f"{PREFIX} ❌ {src}{task_type}失败 | ID:{request_id} | 错误:{error_msg}", flush=True)
    log_backend(
        "task_error",
        level="ERROR",
        task_type=task_type,
        request_id=request_id,
        source=source,
        error=error_msg,
    )

class ProgressBar:
    STATE_WAITING = "waiting"
    STATE_GENERATING = "generating"
    STATE_DONE = "done"
    def __init__(self, request_id: str, service_name: str, extra_info: str = None, streaming: bool = True, task_type: str = None, source: str = None):
        self._request_id = request_id
        self._service_name = service_name
        self._extra_info = extra_info
        enabled = is_streaming_progress_enabled()
        isatty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self._interactive = bool(enabled and isatty)
        self._streaming = bool(streaming and self._interactive)
        self._heartbeat = bool(enabled and (not isatty) and bool(streaming))
        self._heartbeat_interval_sec = 15.0
        self._last_heartbeat = 0.0
        self._task_type = task_type
        self._source = source
        self._state = self.STATE_WAITING
        self._char_count = 0
        self._start_time = time.perf_counter()
        self._closed = False
        self._stop_event = threading.Event()
        self._timer_thread = None
        with _progress_lock:
            global _global_last_output_len
            _global_last_output_len = 0
        self._refresh()
        if self._streaming or self._heartbeat:
            self._timer_thread = threading.Thread(target=self._timer_loop, daemon=True)
            self._timer_thread.start()
    def _format_elapsed(self) -> str:
        s = time.perf_counter() - self._start_time
        if s < 60:
            return f"{s:.1f}s"
        m = int(s // 60)
        sec = int(s % 60)
        return f"{m}m{sec}s"
    def _render(self) -> str:
        e = self._format_elapsed()
        if self._state == self.STATE_WAITING:
            base = f"{PREFIX} 🟠 等待{self._service_name}响应..."
            if not self._streaming:
                if self._heartbeat:
                    if self._extra_info:
                        return f"{base} | {self._extra_info} | {e}"
                    return f"{base} | {e}"
                return base
            if self._extra_info:
                return f"{base} | {self._extra_info} | {e}"
            return f"{base} | {e}"
        if self._state == self.STATE_GENERATING:
            if self._streaming:
                return f"{PREFIX} 🔵 生成中 | {self._char_count}字符 | {e}"
            if self._heartbeat:
                return f"{PREFIX} 🔵 生成中 | {self._char_count}字符 | {e}"
            return f"{PREFIX} 🔵 生成中..."
        return ""
    def _refresh(self) -> None:
        if not self._interactive:
            return
        if self._closed:
            return
        out = self._render()
        if not out:
            return
        with _progress_lock:
            global _global_last_output_len
            w = get_display_width(out)
            pad = ""
            if _global_last_output_len > w:
                pad = " " * (_global_last_output_len - w)
            print(f"\r{_ANSI_CLEAR_EOL}{out}{pad}{_ANSI_CLEAR_EOL}  ", end='', flush=True)
            _global_last_output_len = w + len(pad)
    def _timer_loop(self):
        while not self._stop_event.is_set() and not self._closed:
            if self._interactive:
                self._refresh()
                if self._stop_event.wait(0.1):
                    break
                continue

            if self._heartbeat:
                now = time.perf_counter()
                if (now - self._last_heartbeat) >= self._heartbeat_interval_sec:
                    self._last_heartbeat = now
                    out = self._render()
                    if out:
                        print(out, flush=True)
                if self._stop_event.wait(1.0):
                    break
                continue

            if self._stop_event.wait(0.5):
                break
    def set_generating(self, char_count: int = 0) -> None:
        if self._closed or self._state == self.STATE_GENERATING:
            return
        self._state = self.STATE_GENERATING
        self._char_count = char_count
        self._refresh()
    def update(self, char_count: int) -> None:
        if self._closed:
            return
        self._char_count = char_count
        if self._streaming:
            self._refresh()
    def done(self, message: str = None, char_count: int = None, elapsed_ms: int = None) -> None:
        if self._closed:
            return
        self._closed = True
        self._state = self.STATE_DONE
        self._stop_event.set()
        with _progress_lock:
            global _global_last_output_len
            _global_last_output_len = 0
        if self._task_type:
            log_complete(self._task_type, self._request_id, self._service_name, char_count if char_count is not None else self._char_count, elapsed_ms if elapsed_ms is not None else int((time.perf_counter() - self._start_time) * 1000), source=self._source)
            return
        elapsed = self._format_elapsed()
        final_count = char_count if char_count is not None else self._char_count
        msg = message or f"{PREFIX} ✅ 完成 | 服务:{self._service_name} | ID:{self._request_id} | 字符:{final_count} | 耗时:{elapsed}"
        print(f"\r{_ANSI_CLEAR_EOL}{msg}", flush=True)
    def error(self, message: str) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        with _progress_lock:
            global _global_last_output_len
            _global_last_output_len = 0
        if self._task_type:
            log_error(self._task_type, self._request_id, message, source=self._source)
            return
        print(f"\r{_ANSI_CLEAR_EOL}{message}", flush=True)
    def cancel(self, message: str = None) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        with _progress_lock:
            global _global_last_output_len
            _global_last_output_len = 0
        m = message or "任务被取消"
        if self._task_type:
            log_error(self._task_type, self._request_id, m, source=self._source)
            return
        print(f"\r{_ANSI_CLEAR_EOL}{WARN_PREFIX} {m} | ID:{self._request_id}", flush=True)
