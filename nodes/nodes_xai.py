from ..comfly_config import *
from .__init__ import *

class ComflyGrok3VideoApi:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["grok-video-3"], {"default": "grok-video-3"}),
                "ratio": (["2:3", "3:2", "16:9", "9:16", "1:1"], {"default": "1:1"}),
                "duration": ([6, 10], {"default": 10}),
                "resolution": (["480P", "720P", "1080P"], {"default": "1080P"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            },
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "task_id", "response", "video_url")
    FUNCTION = "generate_video"
    CATEGORY = "RunNode/xAI"

    def __init__(self):
        config = get_config()
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "")
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _process_image(self, image_tensor, request_id=None):
        """
        Process image tensor: resize if needed and compress to JPEG.
        Returns: (file_content, mime_type, filename)
        """
        try:
            # 1. Convert tensor to PIL
            img = tensor2pil(image_tensor)[0]
            
            # 2. Resize if too large (max 1536px)
            original_size = img.size
            max_dimension = 1536
            if max(original_size) > max_dimension:
                ratio = max_dimension / max(original_size)
                new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
                resampling = getattr(Image, 'Resampling', None)
                if resampling and hasattr(resampling, 'LANCZOS'):
                    img = img.resize(new_size, resampling.LANCZOS)
                else:
                    img = img.resize(new_size, getattr(Image, 'LANCZOS', Image.BICUBIC))
            
            # 3. Compress to JPEG with size limit check
            # Convert to RGB for JPEG compatibility
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
                
            formats_to_try = [
                ('JPEG', {'quality': 75, 'optimize': True}),
                ('JPEG', {'quality': 60, 'optimize': True}),
                ('JPEG', {'quality': 50, 'optimize': True}),
            ]
            
            best_bytes = None
            smallest_size = float('inf')
            
            for format_name, save_kwargs in formats_to_try:
                buf = BytesIO()
                img.save(buf, format=format_name, **save_kwargs)
                img_bytes = buf.getvalue()
                
                # Check size (aim for < 2MB, similar to base64 check in Comfly.py)
                current_size = len(img_bytes)
                if current_size < smallest_size:
                    smallest_size = current_size
                    best_bytes = img_bytes
                    
                    if current_size < 2 * 1024 * 1024: # 2MB
                        break
            
            if best_bytes:
                return best_bytes, "image/jpeg", "image.jpg"
                
        except Exception as e:
            log_backend_exception("image_process_failed", request_id=request_id, error=str(e))
            
        # Fallback to original PNG behavior
        pil_image = tensor2pil(image_tensor)[0]
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return buffered.getvalue(), "image/png", "image.png"

    def upload_image(self, image_tensor, request_id=None, rn_pbar=None):
        try:
            file_content, mime_type, filename = self._process_image(image_tensor, request_id=request_id)

            files = {"file": (filename, file_content, mime_type)}

            response = requests.post(
                f"{self.base_url}/v1/files",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files=files,
                timeout=self.timeout,
            )

            response.raise_for_status()
            result = response.json()

            if "url" in result:
                return result["url"]
            else:
                msg = f"Unexpected response from file upload API: {format_runnode_error(result)}"
                log_backend(
                    "xai_grok_upload_unexpected_response",
                    level="ERROR",
                    request_id=request_id,
                    url=safe_public_url(self.base_url),
                )
                if rn_pbar is not None:
                    rn_pbar.error("文件上传返回异常，请稍后重试")
                return None

        except Exception as e:
            msg = f"Error uploading image: {format_runnode_error(str(e))}"
            log_backend_exception(
                "xai_grok_upload_exception",
                request_id=request_id,
                url=safe_public_url(self.base_url),
            )
            if rn_pbar is not None:
                rn_pbar.error("上传参考图像失败，请检查网络或图像格式")
            return None

    def generate_video(self, prompt, model, ratio, duration, resolution, api_key="", image=None, seed=0):
        request_id = generate_request_id("video_gen", "xai")
        log_prepare("视频生成", request_id, "RunNode/xAI-", "xAI", model_name=model)
        rn_pbar = ProgressBar(
            request_id,
            "xAI",
            extra_info=f"模型:{model}",
            streaming=True,
            task_type="视频生成",
            source="RunNode/xAI-",
        )
        _rn_start = time.perf_counter()

        if api_key.strip():
            self.api_key = api_key
        else:
            self.api_key = get_config().get("api_key", "")

        if not self.base_url:
            self.base_url = get_config().get("base_url", "")

        if not self.api_key:
            error_message = "API key not found in Comflyapi.json"
            rn_pbar.error(error_message)
            log_backend(
                "xai_video_generate_failed",
                level="ERROR",
                request_id=request_id,
                stage="missing_api_key",
                model=model,
                elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
            )
            log_error("配置缺失", request_id, error_message, "RunNode/xAI-", "xAI")
            raise Exception(error_message)

        if not self.base_url:
            error_message = "Base URL not configured"
            rn_pbar.error(error_message)
            log_backend(
                "xai_video_generate_failed",
                level="ERROR",
                request_id=request_id,
                stage="missing_base_url",
                model=model,
                elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
            )
            log_error("配置缺失", request_id, error_message, "RunNode/xAI-", "xAI")
            raise Exception(error_message)

        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            payload = {
                "prompt": prompt,
                "model": model,
                "ratio": ratio,
                "duration": duration,
                "resolution": resolution,
            }

            if seed > 0:
                payload["seed"] = seed

            image_url = None
            if image is not None:
                pbar.update_absolute(20)
                image_url = self.upload_image(image, request_id=request_id, rn_pbar=rn_pbar)
                if image_url:
                    payload["images"] = [image_url]
                else:
                    error_message = "Failed to upload image. Please check your image and try again."
                    rn_pbar.error(error_message)
                    log_backend(
                        "xai_video_generate_failed",
                        level="ERROR",
                        request_id=request_id,
                        stage="upload_image_failed",
                        model=model,
                        elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                    )
                    raise Exception(error_message)

            pbar.update_absolute(30)

            log_backend(
                "xai_video_generate_start",
                request_id=request_id,
                url=safe_public_url(self.base_url),
                model=model,
                prompt_len=len(prompt or ""),
                ratio=ratio,
                resolution=resolution,
                has_image=bool(image is not None),
                seed=(int(seed) if int(seed) > 0 else None),
            )

            response = requests.post(
                f"{self.base_url}/v2/videos/generations",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout,
            )

            if response.status_code != 200:
                error_message = format_runnode_error(response)
                rn_pbar.error(error_message)
                log_backend(
                    "xai_video_generate_failed",
                    level="ERROR",
                    request_id=request_id,
                    stage="http_error",
                    model=model,
                    status_code=int(response.status_code),
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )
                log_error("请求失败", request_id, error_message, "RunNode/xAI-", "xAI")
                raise Exception(error_message)

            result = response.json()

            task_id = result.get("task_id")
            if not task_id:
                error_message = "No task ID returned from API"
                rn_pbar.error(error_message)
                log_backend(
                    "xai_video_generate_failed",
                    level="ERROR",
                    request_id=request_id,
                    stage="missing_task_id",
                    model=model,
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )
                log_error("任务ID缺失", request_id, error_message, "RunNode/xAI-", "xAI")
                raise Exception(error_message)

            pbar.update_absolute(40)

            video_url = None
            attempts = 0
            max_attempts = 200
            start_time = time.time()
            max_wait_time = 600

            while attempts < max_attempts:
                current_time = time.time()
                elapsed_time = current_time - start_time

                if elapsed_time > max_wait_time:
                    error_message = f"Video generation timeout after {elapsed_time:.1f} seconds (max: {max_wait_time}s)"
                    rn_pbar.error(error_message)
                    log_backend(
                        "xai_video_generate_failed",
                        level="ERROR",
                        request_id=request_id,
                        stage="task_timeout",
                        model=model,
                        task_id=task_id,
                        attempts=int(attempts),
                        elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                    )
                    log_error("任务超时", request_id, error_message, "RunNode/xAI-", "xAI")
                    raise Exception(error_message)

                time.sleep(5)
                attempts += 1

                try:
                    status_response = requests.get(
                        f"{self.base_url}/v2/videos/generations/{task_id}",
                        headers=self.get_headers(),
                        timeout=30,
                    )

                    if status_response.status_code != 200:
                        continue

                    status_result = status_response.json()
                    status = status_result.get("status", "UNKNOWN")

                    if status == "IN_PROGRESS":
                        progress = status_result.get("progress", "0%")
                        try:
                            if isinstance(progress, str) and progress.endswith("%"):
                                progress_num = int(progress.rstrip("%"))
                                pbar_value = min(90, 40 + progress_num * 50 / 100)
                                pbar.update_absolute(pbar_value)
                        except (ValueError, AttributeError):
                            progress_value = min(80, 40 + (attempts * 40 // max_attempts))
                            pbar.update_absolute(progress_value)

                    if status == "SUCCESS":
                        data = status_result.get("data", {})
                        if "output" in data:
                            video_url = data["output"]
                            break
                        else:
                            continue
                    elif status == "FAILURE":
                        fail_reason = status_result.get("fail_reason", "Unknown error")
                        error_message = f"Video generation failed: {format_runnode_error(fail_reason)}"
                        rn_pbar.error(error_message)
                        log_backend(
                            "xai_video_generate_failed",
                            level="ERROR",
                            request_id=request_id,
                            stage="task_failed",
                            model=model,
                            task_id=task_id,
                            fail_reason=str(fail_reason),
                            elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                        )
                        log_error("任务失败", request_id, error_message, "RunNode/xAI-", "xAI")
                        raise Exception(error_message)
                    elif status in ["NOT_START", "IN_PROGRESS"]:
                        continue
                    else:
                        continue

                except requests.exceptions.Timeout:
                    continue
                except Exception:
                    continue

            if not video_url:
                elapsed_time = time.time() - start_time
                error_message = f"Video generation timeout or failed to retrieve video URL after {attempts} attempts, elapsed time: {elapsed_time:.1f}s"
                rn_pbar.error(error_message)
                log_backend(
                    "xai_video_generate_failed",
                    level="ERROR",
                    request_id=request_id,
                    stage="task_timeout_final",
                    model=model,
                    task_id=task_id,
                    attempts=int(attempts),
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )
                log_error("任务超时", request_id, error_message, "RunNode/xAI-", "xAI")
                raise Exception(error_message)

            if video_url:
                pbar.update_absolute(95)
                video_adapter = ComflyVideoAdapter(video_url)

                response_data = {
                    "code": "success",
                    "url": video_url,
                    "task_id": task_id,
                    "model": model,
                    "ratio": ratio,
                    "resolution": resolution,
                }

                rn_pbar.done(char_count=len(json.dumps(response_data)))
                log_backend(
                    "xai_video_generate_done",
                    request_id=request_id,
                    url=safe_public_url(self.base_url),
                    model=model,
                    task_id=task_id,
                    video_url=safe_public_url(video_url),
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )
                return (video_adapter, task_id, json.dumps(response_data), video_url)

        except Exception as e:
            error_message = f"Error generating video: {format_runnode_error(str(e))}"
            rn_pbar.error(error_message)
            log_backend_exception(
                "xai_video_generate_exception",
                request_id=request_id,
                url=safe_public_url(self.base_url),
                model=model,
            )
            log_error("异常", request_id, error_message, "RunNode/xAI-", "xAI")
            raise Exception(error_message)


