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
                "duration": (["6", "10", "15"], {"default": "15"}),
                "resolution": (["480P", "720P", "1080P"], {"default": "1080P"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
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
        try:
            pil_image = tensor2pil(image_tensor)[0]
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            return buffered.getvalue(), "image/png", "image.png"
        except Exception as e:
            log_backend_exception("image_process_failed", request_id=request_id, error=str(e))
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
                log_backend(
                    "xai_grok_upload_unexpected_response",
                    level="ERROR",
                    request_id=request_id,
                    url=safe_public_url(self.base_url),
                )
                if rn_pbar is not None:
                    rn_pbar.error("文件上传返回异常，请稍后重试")
                return None

        except Exception:
            log_backend_exception(
                "xai_grok_upload_exception",
                request_id=request_id,
                url=safe_public_url(self.base_url),
            )
            if rn_pbar is not None:
                rn_pbar.error("上传参考图像失败，请检查网络或图像格式")
            return None

    def generate_video(self, prompt, model, ratio, duration, resolution, api_key="", image1=None, image2=None, image3=None, image4=None, image5=None, image6=None, image7=None, seed=0):
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
            error_message = "API key not found in configuration file or environment variables."
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

            try:
                duration_value = int(duration)
            except Exception:
                duration_value = duration

            payload = {
                "prompt": prompt,
                "model": model,
                "ratio": ratio,
                "duration": duration_value,
                "resolution": resolution,
            }

            if seed > 0:
                payload["seed"] = seed

            all_images = [image1, image2, image3, image4, image5, image6, image7]
            image_urls = []

            for i, img in enumerate(all_images):
                if img is None:
                    continue

                pbar.update_absolute(min(29, 15 + i * 2))
                uploaded_url = self.upload_image(img, request_id=request_id, rn_pbar=rn_pbar)
                if uploaded_url:
                    image_urls.append(uploaded_url)
                else:
                    error_message = f"Failed to upload image {i+1}. Please check your image and try again."
                    rn_pbar.error(error_message)
                    log_backend(
                        "xai_video_generate_failed",
                        level="ERROR",
                        request_id=request_id,
                        stage="upload_image_failed",
                        model=model,
                        image_index=int(i + 1),
                        elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                    )
                    raise Exception(error_message)

            if image_urls:
                payload["images"] = image_urls

            pbar.update_absolute(30)

            log_backend(
                "xai_video_generate_start",
                request_id=request_id,
                url=safe_public_url(self.base_url),
                model=model,
                prompt_len=len(prompt or ""),
                ratio=ratio,
                resolution=resolution,
                has_image=bool(image_urls),
                image_count=int(len(image_urls)),
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


class ComflyGrok3VideoApi30S:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["grok-video-3"], {"default": "grok-video-3"}),
                "ratio": (["2:3", "3:2", "16:9", "9:16", "1:1"], {"default": "1:1"}),
                "duration": ([str(i) for i in range(6, 31)], {"default": "15"}),
                "resolution": (["720P"], {"default": "720P"}),
            },
            "optional": {
                # "api_key": ("STRING", {"default": ""}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "skip_error": ("BOOLEAN", {"default": False, "tooltip": "开启后，节点失败时不报错、按旧行为返回默认空结果；关闭时（默认）失败直接抛出错误。"})
            }
        }
    
    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "task_id", "response", "video_url")
    FUNCTION = "generate_video"
    CATEGORY = "RunNode/Grok"

    def __init__(self):
        config = get_config()
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "") or baseurl
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def upload_image(self, image_tensor, request_id=None, rn_pbar=None):
        """Upload image to the file endpoint and return the URL"""
        try:
            pil_image = tensor2pil(image_tensor)[0]

            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            file_content = buffered.getvalue()

            files = {'file': ('image.png', file_content, 'image/png')}

            response = requests.post(
                f"{self.base_url}/v1/files",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files=files,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            if 'url' in result:
                return result['url']
            else:
                log_backend(
                    "xai_grok_upload_unexpected_response",
                    level="ERROR",
                    request_id=request_id,
                    url=safe_public_url(self.base_url),
                )
                if rn_pbar is not None:
                    rn_pbar.error("文件上传返回异常，请稍后重试")
                return None
                
        except Exception:
            log_backend_exception(
                "xai_grok_upload_exception",
                request_id=request_id,
                url=safe_public_url(self.base_url),
            )
            if rn_pbar is not None:
                rn_pbar.error("上传参考图像失败，请检查网络或图像格式")
            return None
    
    def generate_video(self, prompt, model, ratio, duration, resolution, api_key="", image1=None, image2=None, image3=None, image4=None, image5=None, image6=None, image7=None, seed=0, skip_error=False):
        request_id = generate_request_id("video_gen", "xai")
        log_prepare("视频生成", request_id, "RunNode/Grok-", "xAI", model_name=model)
        rn_pbar = ProgressBar(
            request_id,
            "xAI",
            extra_info=f"模型:{model}",
            streaming=True,
            task_type="视频生成",
            source="RunNode/Grok-",
        )
        _rn_start = time.perf_counter()

        if api_key.strip():
            self.api_key = api_key
        else:
            self.api_key = get_config().get("api_key", "")

        if not self.base_url:
            self.base_url = get_config().get("base_url", "") or baseurl
            
        if not self.api_key:
            error_message = "API key not found in configuration file or environment variables."
            error_response = {"code": "error", "message": error_message}
            rn_pbar.error(error_message)
            log_backend(
                "xai_video_generate_failed",
                level="ERROR",
                request_id=request_id,
                stage="missing_api_key",
                model=model,
                skip_error=bool(skip_error),
                elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
            )
            log_error("配置缺失", request_id, error_message, "RunNode/Grok-", "xAI")
            if not skip_error:
                raise RuntimeError(f"[ComflyGrok3VideoApi30S] {error_response}")
            return ("", "", json.dumps(error_response), "")

        if not self.base_url:
            error_message = "Base URL not configured"
            error_response = {"code": "error", "message": error_message}
            rn_pbar.error(error_message)
            log_backend(
                "xai_video_generate_failed",
                level="ERROR",
                request_id=request_id,
                stage="missing_base_url",
                model=model,
                skip_error=bool(skip_error),
                elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
            )
            log_error("配置缺失", request_id, error_message, "RunNode/Grok-", "xAI")
            if not skip_error:
                raise RuntimeError(f"[ComflyGrok3VideoApi30S] {error_response}")
            return ("", "", json.dumps(error_response), "")
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)
            rn_pbar.update_absolute(10, "准备请求...")

            payload = {
                "prompt": prompt,
                "model": model,
                "ratio": ratio,
                "duration": int(duration),
                "resolution": resolution
            }

            if seed > 0:
                payload["seed"] = seed

            # Handle image inputs (up to 7 reference images)
            all_images = [image1, image2, image3, image4, image5, image6, image7]
            image_urls = []
            
            for i, img in enumerate(all_images):
                if img is not None:
                    pbar.update_absolute(15 + i * 2)
                    uploaded_url = self.upload_image(img, request_id=request_id, rn_pbar=rn_pbar)
                    if uploaded_url:
                        image_urls.append(uploaded_url)
                    else:
                        error_message = f"Failed to upload image {i+1}. Please check your image and try again."
                        rn_pbar.error(error_message)
                        log_backend(
                            "xai_video_generate_failed",
                            level="ERROR",
                            request_id=request_id,
                            stage="upload_image_failed",
                            model=model,
                            image_index=int(i + 1),
                            skip_error=bool(skip_error),
                            elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                        )
                        log_error("上传失败", request_id, error_message, "RunNode/Grok-", "xAI")
                        if not skip_error:
                            raise RuntimeError(f"[ComflyGrok3VideoApi30S] {error_message}")
                        return ("", "", json.dumps({"code": "error", "message": error_message}), "")
            
            if image_urls:
                payload["images"] = image_urls

            pbar.update_absolute(30)
            rn_pbar.update_absolute(30, "提交任务...")

            log_backend(
                "xai_video_generate_start",
                request_id=request_id,
                url=safe_public_url(self.base_url),
                model=model,
                prompt_len=len(prompt or ""),
                ratio=ratio,
                resolution=resolution,
                has_image=bool(image_urls),
                image_count=int(len(image_urls)),
                seed=(int(seed) if int(seed) > 0 else None),
            )
            
            # Submit video generation request
            response = requests.post(
                f"{self.base_url}/v2/videos/generations",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
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
                    skip_error=bool(skip_error),
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )
                log_error("请求失败", request_id, error_message, "RunNode/Grok-", "xAI")
                if not skip_error:
                    raise RuntimeError(f"[ComflyGrok3VideoApi30S] {error_message}")
                return ("", "", json.dumps({"code": "error", "message": error_message}), "")
                
            result = response.json()
            
            # Extract task_id from response
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
                    skip_error=bool(skip_error),
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )
                log_error("任务ID缺失", request_id, error_message, "RunNode/Grok-", "xAI")
                if not skip_error:
                    raise RuntimeError(f"[ComflyGrok3VideoApi30S] {error_message}")
                return ("", "", json.dumps({"code": "error", "message": error_message}), "")
            
            pbar.update_absolute(40)
            rn_pbar.update_absolute(40, "排队中...")
            
            # Poll for video generation completion
            video_url = None
            attempts = 0
            max_attempts = 200  # Wait up to 3 minutes (36 * 5 seconds)
            start_time = time.time()
            max_wait_time = 600  # 5 minutes
        
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
                        skip_error=bool(skip_error),
                        elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                    )
                    log_error("任务超时", request_id, error_message, "RunNode/Grok-", "xAI")
                    if not skip_error:
                        raise RuntimeError(f"[ComflyGrok3VideoApi30S] {error_message}")
                    return ("", task_id, json.dumps({"code": "error", "message": error_message}), "")
                
                time.sleep(5)  
                attempts += 1
                
                try:
                    # Query task status
                    status_response = requests.get(
                        f"{self.base_url}/v2/videos/generations/{task_id}",
                        headers=self.get_headers(),
                        timeout=30
                    )
                    
                    if status_response.status_code != 200:
                        continue
                        
                    status_result = status_response.json()
                    
                    # Check task status
                    status = status_result.get("status", "UNKNOWN")
                    log_backend(
                        "xai_video_generate_check",
                        request_id=request_id,
                        task_id=task_id,
                        model=model,
                        status=str(status),
                        attempts=int(attempts),
                    )
                    
                    # Update progress bar based on status
                    if status == "IN_PROGRESS":
                        progress = status_result.get("progress", "0%")
                        try:
                            if progress.endswith('%'):
                                progress_num = int(progress.rstrip('%'))
                                pbar_value = min(90, 40 + progress_num * 50 / 100)
                                pbar.update_absolute(pbar_value)
                                rn_pbar.update_absolute(int(pbar_value), f"处理中 {progress}...")
                        except (ValueError, AttributeError):
                            progress_value = min(80, 40 + (attempts * 40 // max_attempts))
                            pbar.update_absolute(progress_value)
                            rn_pbar.update_absolute(int(progress_value), "处理中...")
                    
                    # Handle different statuses
                    if status == "SUCCESS":
                        # Extract video URL from successful response
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
                            skip_error=bool(skip_error),
                            elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                        )
                        log_error("任务失败", request_id, error_message, "RunNode/Grok-", "xAI")
                        if not skip_error:
                            raise RuntimeError(f"[ComflyGrok3VideoApi30S] {error_message}")
                        return ("", task_id, json.dumps({"code": "error", "message": error_message}), "")
                    
                    elif status in ["NOT_START", "IN_PROGRESS"]:
                        continue
                    else:
                        continue
                    
                except requests.exceptions.Timeout:
                    continue
                except Exception as e:
                    continue
            
            if not video_url:
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
                    skip_error=bool(skip_error),
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )
                log_error("任务超时", request_id, error_message, "RunNode/Grok-", "xAI")
                if not skip_error:
                    raise RuntimeError(f"[ComflyGrok3VideoApi30S] {error_message}")
                return ("", task_id, json.dumps({"code": "error", "message": error_message}), "")

            if video_url:
                pbar.update_absolute(95)
                rn_pbar.update_absolute(95, "生成完成")
                
                # Return video adapter
                video_adapter = ComflyVideoAdapter(video_url)
                response_data = {"code": "success", "url": video_url, "task_id": task_id, "model": model}
                rn_pbar.done(char_count=len(json.dumps(response_data)), elapsed_ms=int((time.perf_counter() - _rn_start) * 1000))
                log_backend(
                    "xai_video_generate_done",
                    request_id=request_id,
                    url=safe_public_url(self.base_url),
                    model=model,
                    task_id=task_id,
                    video_url=safe_public_url(video_url),
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )
                log_complete("视频生成", request_id, "RunNode/Grok-", "xAI", video_url=safe_public_url(video_url))
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
            log_error("异常", request_id, error_message, "RunNode/Grok-", "xAI")
            if not skip_error:
                raise
            return ("", "", json.dumps({"code": "error", "message": error_message}), "")


class Comfly_veo_omini:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "基于这张图片生成视频"}),
                "mode": (["img2video", "video_edit"], {"default": "img2video"}),
                "model": (["omni_flash-10s"], {"default": "omni_flash-10s"}),
                "size": (["1280x720", "720x1280"], {"default": "1280x720"}),
                "seconds": (["4", "5", "6", "8", "10"], {"default": "10"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_url": ("STRING", {"default": "", "tooltip": "img2video 模式下可选的参考图片 URL。若已连接 image 输入，则忽略此项。"}),
                "video": (IO.VIDEO, {"tooltip": "video_edit 模式下的参考视频输入。"}),
                "video_url": ("STRING", {"default": "", "tooltip": "video_edit 模式下可选的参考视频 URL。若视频输入已提供 URL，则忽略此项。"}),
                "video_way": (["upload", "video_url"], {"default": "upload", "tooltip": "upload：上传已连接的 IO.VIDEO 到 /v1/files 并提交其 URL；video_url：若已提供 video_url，则优先使用它。"}),
                # "api_key": ("STRING", {"default": ""}),
                "watermark": ("BOOLEAN", {"default": False}),
                "poll_interval": ("INT", {"default": 6, "min": 1, "max": 60}),
                "max_poll_attempts": ("INT", {"default": 600, "min": 1, "max": 10000}),
                "skip_error": ("BOOLEAN", {"default": False, "tooltip": "开启后，节点失败时不报错、返回空视频；关闭时（默认）失败直接抛出错误。"})
            }
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "response", "video_url", "task_id")
    FUNCTION = "process"
    CATEGORY = "RunNode/Veo"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 900

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}"}

    def _empty_result(self, response_data=None, video_url="", task_id=""):
        response_data = response_data or {}
        return (
            ComflyVideoAdapter(video_url or ""),
            json.dumps(response_data, ensure_ascii=False),
            video_url or "",
            task_id or ""
        )

    def _decode_response(self, response):
        try:
            return response.json()
        except Exception:
            return {"message": response.text}

    def _extract_error_message(self, payload):
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload.strip()
        if isinstance(payload, list):
            parts = [self._extract_error_message(item) for item in payload]
            return "; ".join(part for part in parts if part)
        if not isinstance(payload, dict):
            return str(payload)

        status = str(payload.get("status", "")).strip().lower()
        code = str(payload.get("code", "")).strip().lower()
        if status in {"queued", "pending", "processing", "running", "in_progress", "in-progress"}:
            return ""
        if status in {"failed", "failure", "error", "cancelled", "canceled"}:
            for key in ("fail_reason", "failure_details", "message", "error", "detail"):
                if payload.get(key):
                    return self._extract_error_message(payload.get(key))
            return status
        if code and any(token in code for token in ("fail", "error", "unauthorized", "forbidden", "not_found", "not-found", "invalid")):
            msg = self._extract_error_message(payload.get("message") or payload.get("data") or payload.get("detail"))
            return f"{code}: {msg}" if msg else code

        for key in ("failure_details", "fail_reason", "error", "errors"):
            value = payload.get(key)
            if value:
                return self._extract_error_message(value)
        if not status and payload.get("detail"):
            return self._extract_error_message(payload.get("detail"))
        return ""

    def _extract_video_url(self, payload):
        if payload is None:
            return ""
        if isinstance(payload, str):
            text = payload.strip()
            return text if text.startswith(("http://", "https://")) else ""
        if isinstance(payload, list):
            for item in payload:
                found = self._extract_video_url(item)
                if found:
                    return found
            return ""
        if not isinstance(payload, dict):
            return ""

        for key in ("video_url", "url"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip().startswith(("http://", "https://")):
                return value.strip()

        for key in ("video", "videos", "output", "outputs", "result", "data"):
            found = self._extract_video_url(payload.get(key))
            if found:
                return found
        return ""

    def _extract_file_url(self, payload):
        if payload is None:
            return ""
        if isinstance(payload, str):
            text = payload.strip()
            return text if text.startswith(("http://", "https://")) else ""
        if isinstance(payload, list):
            for item in payload:
                found = self._extract_file_url(item)
                if found:
                    return found
            return ""
        if not isinstance(payload, dict):
            return ""

        for key in ("url", "download_url", "file_url"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip().startswith(("http://", "https://")):
                return value.strip()

        for key in ("file", "data", "result", "output"):
            found = self._extract_file_url(payload.get(key))
            if found:
                return found
        return ""

    def _upload_file_to_get_url(self, file_bytes, filename, content_type):
        if not file_bytes:
            return ""
        files = {"file": (filename, file_bytes, content_type)}
        response = requests.post(
            f"{baseurl.rstrip('/')}/v1/files",
            headers=self._headers(),
            files=files,
            timeout=self.timeout
        )
        result = self._decode_response(response)
        if response.status_code != 200:
            message = self._extract_error_message(result) or response.text
            raise RuntimeError(f"File upload API Error: {response.status_code} - {message}")

        error_message = self._extract_error_message(result)
        if error_message:
            raise RuntimeError(f"File upload API Error: {error_message}")

        file_url = self._extract_file_url(result)
        if not file_url:
            raise RuntimeError(f"No file URL in upload response: {json.dumps(result, ensure_ascii=False)}")
        return file_url

    def _direct_video_url(self, video_input):
        if video_input is None:
            return ""
        if isinstance(video_input, str) and video_input.strip().startswith(("http://", "https://")):
            return video_input.strip()
        for attr in ("video_url", "url"):
            value = getattr(video_input, attr, "")
            if isinstance(value, str) and value.strip().startswith(("http://", "https://")):
                return value.strip()
        if isinstance(video_input, dict):
            for key in ("video_url", "url"):
                value = video_input.get(key)
                if isinstance(value, str) and value.strip().startswith(("http://", "https://")):
                    return value.strip()
        return ""

    def _prepare_video_reference_url(self, video=None, video_url="", video_way="upload"):
        explicit_url = str(video_url or "").strip()
        if video_way == "video_url" and explicit_url:
            return explicit_url

        direct_url = self._direct_video_url(video)
        if direct_url:
            return direct_url

        file_bytes, filename = _doubao_seedance_video_input_to_bytes(video)
        if file_bytes:
            if not filename:
                filename = f"reference_video_{abs(hash(file_bytes)) % 10**10}.mp4"
            content_type = mimetypes.guess_type(filename)[0] or "video/mp4"
            return self._upload_file_to_get_url(file_bytes, filename, content_type)

        if explicit_url:
            return explicit_url
        return ""

    def _submit_task(self, prompt, mode, image, image_url, video, video_url, video_way, model, size, seconds, watermark):
        data = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "seconds": str(seconds),
            "watermark": str(bool(watermark)).lower()
        }

        files = []
        mode = str(mode or "img2video").strip().lower()
        if mode == "video_edit":
            reference_url = self._prepare_video_reference_url(video, video_url, video_way)
            if not reference_url:
                raise RuntimeError("video_edit mode requires a video input or video_url.")
            files.append(("input_reference", (None, reference_url)))
        else:
            if image is not None:
                pil_image = tensor2pil(image)[0]
                buffered = BytesIO()
                pil_image.save(buffered, format="PNG")
                buffered.seek(0)
                files.append(("input_reference", ("input_reference.png", buffered, "image/png")))
            elif image_url and str(image_url).strip():
                files.append(("input_reference", (None, str(image_url).strip())))
            else:
                raise RuntimeError("img2video mode requires an image input or image_url.")

        submit_url = f"{baseurl.rstrip('/')}/v1/videos"
        response = requests.post(
            submit_url,
            headers=self._headers(),
            data=data,
            files=files,
            timeout=self.timeout
        )
        result = self._decode_response(response)
        if response.status_code != 200:
            message = self._extract_error_message(result) or response.text
            raise RuntimeError(f"API Error: {response.status_code} - {message}")

        error_message = self._extract_error_message(result)
        if error_message:
            raise RuntimeError(f"API Error: {error_message}")

        task_id = result.get("task_id") or result.get("id")
        if not task_id:
            raise RuntimeError(f"No task_id in API response: {json.dumps(result, ensure_ascii=False)}")
        return str(task_id), result

    def _poll_task(self, task_id, poll_interval, max_poll_attempts, pbar):
        poll_url = f"{baseurl.rstrip('/')}/v1/videos/{task_id}"
        last_status_data = {}

        for attempt in range(1, max_poll_attempts + 1):
            time.sleep(max(1, int(poll_interval)))
            response = requests.get(
                poll_url,
                headers=self._headers(),
                timeout=self.timeout
            )
            status_data = self._decode_response(response)
            last_status_data = status_data if isinstance(status_data, dict) else {"response": status_data}

            if response.status_code != 200:
                message = self._extract_error_message(status_data) or response.text
                raise RuntimeError(f"Status API Error: {response.status_code} - {message}")

            error_message = self._extract_error_message(status_data)
            if error_message:
                raise RuntimeError(
                    f"Video generation failed for task_id={task_id}: {error_message}. "
                    f"Last response: {json.dumps(status_data, ensure_ascii=False)}"
                )

            progress = status_data.get("progress", 0) if isinstance(status_data, dict) else 0
            try:
                pbar.update_absolute(min(95, 30 + int(float(progress) * 0.6)))
            except (TypeError, ValueError):
                pbar.update_absolute(min(90, 30 + (attempt * 60 // max_poll_attempts)))

            status = str(status_data.get("status", "")).strip().lower() if isinstance(status_data, dict) else ""
            if status in {"completed", "complete", "succeeded", "success", "done"}:
                video_url = self._extract_video_url(status_data)
                if not video_url:
                    raise RuntimeError(f"Task completed but no video URL found: {json.dumps(status_data, ensure_ascii=False)}")
                return video_url, status_data

        raise RuntimeError(
            f"Video generation timed out after {max_poll_attempts} attempts. "
            f"Last response: {json.dumps(last_status_data, ensure_ascii=False)}"
        )

    def process(
        self,
        prompt,
        mode="img2video",
        model="omni_flash-10s",
        size="1280x720",
        seconds="10",
        image=None,
        image_url="",
        video=None,
        video_url="",
        video_way="upload",
        api_key="",
        watermark=False,
        poll_interval=6,
        max_poll_attempts=600,
        skip_error=False
    ):
        request_id = generate_request_id("video_gen", "xai")
        log_prepare("视频生成", request_id, "RunNode/Grok-", "xAI", model_name=model)
        rn_pbar = ProgressBar(
            request_id,
            "xAI",
            extra_info=f"模型:{model}",
            streaming=True,
            task_type="视频生成",
            source="RunNode/Grok-",
        )
        _rn_start = time.perf_counter()

        if api_key.strip():
            self.api_key = api_key.strip()
        else:
            self.api_key = get_config().get("api_key", "")

        if not self.api_key:
            error_response = {"status": "error", "message": "API key not provided or not found in config"}
            error_message = error_response["message"]
            rn_pbar.error(error_message)
            log_error("配置缺失", request_id, error_message, "RunNode/Grok-", "xAI")
            if not skip_error:
                raise RuntimeError(f"[Comfly_veo_omini] {error_message}")
            return self._empty_result(error_response)

        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(10)
        rn_pbar.set_generating()
        log_backend(
            "xai_veo_omini_start",
            request_id=request_id,
            model=model,
            mode=mode,
            size=size,
            seconds=str(seconds),
            elapsed_ms=0,
        )

        try:
            task_id, submit_response = self._submit_task(prompt, mode, image, image_url, video, video_url, video_way, model, size, seconds, watermark)
            log_backend(
                "xai_veo_omini_submitted",
                request_id=request_id,
                model=model,
                mode=mode,
                task_id=task_id,
                elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
            )
            pbar.update_absolute(30)

            video_url, status_response = self._poll_task(task_id, poll_interval, max_poll_attempts, pbar)
            pbar.update_absolute(100)

            response_data = {
                "status": "success",
                "task_id": task_id,
                "mode": mode,
                "model": model,
                "prompt": prompt,
                "size": size,
                "seconds": str(seconds),
                "watermark": bool(watermark),
                "video_url": video_url,
                "submit_response": submit_response,
                "status_response": status_response
            }
            rn_pbar.done(
                char_count=len(json.dumps(response_data, ensure_ascii=False)),
                elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
            )
            log_complete("视频生成", request_id, "RunNode/Grok-", "xAI", video_url=safe_public_url(video_url))
            log_backend(
                "xai_veo_omini_done",
                request_id=request_id,
                model=model,
                mode=mode,
                task_id=task_id,
                video_url=safe_public_url(video_url),
                elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
            )
            return (ComflyVideoAdapter(video_url), json.dumps(response_data, ensure_ascii=False), video_url, task_id)

        except Exception as e:
            error_message = f"Error in veo omini video generation: {format_runnode_error(e)}"
            rn_pbar.error(error_message)
            log_error("异常", request_id, error_message, "RunNode/Grok-", "xAI")
            log_backend_exception(
                "xai_veo_omini_exception",
                request_id=request_id,
                model=model,
                mode=mode,
                error=str(e),
            )
            if not skip_error:
                raise RuntimeError(f"[Comfly_veo_omini] {error_message}") from e
            return self._empty_result({"status": "error", "message": error_message})


class Comfly_grok_video_1_5(Comfly_veo_omini):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "基于这张图片生成视频"}),
                "model": ([
                    "grok-1.5-video-6s",
                    "grok-1.5-video-10s",
                    "grok-1.5-video-15s"
                ], {"default": "grok-1.5-video-6s"}),
                "size": (["1280x720", "720x1280"], {"default": "1280x720"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_url": ("STRING", {"default": "", "tooltip": "可选的参考图片 URL。若已连接 image 输入，则忽略此项。"}),
                # "api_key": ("STRING", {"default": ""}),
                "poll_interval": ("INT", {"default": 6, "min": 1, "max": 60}),
                "max_poll_attempts": ("INT", {"default": 600, "min": 1, "max": 10000}),
                "skip_error": ("BOOLEAN", {"default": False, "tooltip": "开启后，节点失败时不报错、返回空视频；关闭时（默认）失败直接抛出错误。"})
            }
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "response", "video_url", "task_id")
    FUNCTION = "process"
    CATEGORY = "RunNode/Grok"

    def _submit_grok_task(self, prompt, model, size, image, image_url):
        files = [
            ("model", (None, str(model))),
            ("prompt", (None, str(prompt))),
            ("size", (None, str(size)))
        ]

        if image is not None:
            pil_image = tensor2pil(image)[0]
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            buffered.seek(0)
            files.append(("input_reference", ("input_reference.png", buffered, "image/png")))
        elif image_url and str(image_url).strip():
            files.append(("input_reference", (None, str(image_url).strip())))
        else:
            raise RuntimeError("grok-video-1.5 requires an image input or image_url as input_reference.")

        response = requests.post(
            f"{baseurl.rstrip('/')}/v1/videos",
            headers=self._headers(),
            files=files,
            timeout=self.timeout
        )
        result = self._decode_response(response)
        if response.status_code != 200:
            message = self._extract_error_message(result) or response.text
            raise RuntimeError(f"API Error: {response.status_code} - {message}")

        error_message = self._extract_error_message(result)
        if error_message:
            raise RuntimeError(f"API Error: {error_message}")

        task_id = result.get("task_id") or result.get("id")
        if not task_id:
            raise RuntimeError(f"No task_id in API response: {json.dumps(result, ensure_ascii=False)}")
        return str(task_id), result

    def process(
        self,
        prompt,
        model="grok-1.5-video-6s",
        size="1280x720",
        image=None,
        image_url="",
        api_key="",
        poll_interval=6,
        max_poll_attempts=600,
        skip_error=False
    ):
        request_id = generate_request_id("video_gen", "xai")
        log_prepare("视频生成", request_id, "RunNode/Grok-", "xAI", model_name=model)
        rn_pbar = ProgressBar(
            request_id,
            "xAI",
            extra_info=f"模型:{model}",
            streaming=True,
            task_type="视频生成",
            source="RunNode/Grok-",
        )
        _rn_start = time.perf_counter()

        if api_key.strip():
            self.api_key = api_key.strip()
        else:
            self.api_key = get_config().get("api_key", "")

        if not self.api_key:
            error_response = {"status": "error", "message": "API key not provided or not found in config"}
            error_message = error_response["message"]
            rn_pbar.error(error_message)
            log_error("配置缺失", request_id, error_message, "RunNode/Grok-", "xAI")
            if not skip_error:
                raise RuntimeError(f"[Comfly_grok_video_1_5] {error_message}")
            return self._empty_result(error_response)

        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(10)
        rn_pbar.set_generating()
        log_backend(
            "xai_grok_video15_start",
            request_id=request_id,
            model=model,
            size=size,
            elapsed_ms=0,
        )

        try:
            task_id, submit_response = self._submit_grok_task(prompt, model, size, image, image_url)
            log_backend(
                "xai_grok_video15_submitted",
                request_id=request_id,
                model=model,
                task_id=task_id,
                elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
            )
            pbar.update_absolute(30)

            video_url, status_response = self._poll_task(task_id, poll_interval, max_poll_attempts, pbar)
            pbar.update_absolute(100)

            response_data = {
                "status": "success",
                "task_id": task_id,
                "model": model,
                "prompt": prompt,
                "size": size,
                "video_url": video_url,
                "submit_response": submit_response,
                "status_response": status_response
            }
            rn_pbar.done(
                char_count=len(json.dumps(response_data, ensure_ascii=False)),
                elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
            )
            log_complete("视频生成", request_id, "RunNode/Grok-", "xAI", video_url=safe_public_url(video_url))
            log_backend(
                "xai_grok_video15_done",
                request_id=request_id,
                model=model,
                task_id=task_id,
                video_url=safe_public_url(video_url),
                elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
            )
            return (ComflyVideoAdapter(video_url), json.dumps(response_data, ensure_ascii=False), video_url, task_id)

        except Exception as e:
            error_message = f"Error in grok video 1.5 generation: {format_runnode_error(e)}"
            rn_pbar.error(error_message)
            log_error("异常", request_id, error_message, "RunNode/Grok-", "xAI")
            log_backend_exception(
                "xai_grok_video15_exception",
                request_id=request_id,
                model=model,
                error=str(e),
            )
            if not skip_error:
                raise RuntimeError(f"[Comfly_grok_video_1_5] {error_message}") from e
            return self._empty_result({"status": "error", "message": error_message})


class Comfly_grok_image:
    """Grok Image generation via OpenAI Dall-e compatible endpoint."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                # "api_key": ("STRING", {"default": ""}),
                "model": (["grok-4.2-image"], {"default": "grok-4.2-image"}),
                "aspect_ratio": ("STRING", {"default": "1:1", "multiline": False}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image_urls": ("STRING", {"default": "", "multiline": True, "tooltip": "Optional reference image URLs, separated by newline or comma."}),
                "image_way": (["image_url", "base64"], {"default": "image_url"}),
                "async_mode": ("BOOLEAN", {"default": True, "tooltip": "Use async task submission and poll until the result is ready."}),
                "task_id": ("STRING", {"default": "", "tooltip": "Optional existing async task id to query instead of submitting a new request."}),
                "poll_interval": ("INT", {"default": 5, "min": 2, "max": 60, "step": 1}),
                "max_poll_attempts": ("INT", {"default": 180, "min": 10, "max": 1200, "step": 10}),
                "skip_error": ("BOOLEAN", {"default": False, "tooltip": "开启后，节点失败时不报错、按旧行为返回默认空结果；关闭时（默认）失败直接抛出错误。"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_urls", "response")
    FUNCTION = "generate_image"
    CATEGORY = "RunNode/Grok"
    OUTPUT_NODE = True

    _SUCCESS_STATUSES = {"SUCCESS", "COMPLETED", "COMPLETE", "DONE", "FINISHED"}
    _FAILED_STATUSES = {"FAILED", "FAILURE", "ERROR", "CANCELLED", "CANCELED"}
    _WAIT_STATUSES = {"NOT_START", "PENDING", "QUEUED", "IN_QUEUE", "IN_PROGRESS", "RUNNING", "PROCESSING"}

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _blank_image(self, color="white"):
        return pil2tensor(Image.new('RGB', (1024, 1024), color=color))

    def _tensor_to_data_uri(self, image_tensor):
        pil_image = tensor2pil(image_tensor)[0]
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{b64}"

    def _upload_image(self, image_tensor):
        pil_image = tensor2pil(image_tensor)[0]
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        buffered.seek(0)
        response = requests.post(
            f"{baseurl}/v1/files",
            headers={"Authorization": f"Bearer {self.api_key}"},
            files={'file': ('image.png', buffered.getvalue(), 'image/png')},
            timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()
        image_url = result.get("url")
        if not image_url:
            raise RuntimeError(f"Unexpected upload response: {result}")
        return image_url

    def _split_image_urls(self, image_urls):
        if not image_urls or not str(image_urls).strip():
            return []
        return [u.strip() for u in re.split(r"[\n,]+", str(image_urls)) if u.strip()]

    def _extract_task_id(self, result):
        if not isinstance(result, dict):
            return ""
        for key in ("task_id", "id", "request_id"):
            value = result.get(key)
            if value:
                return str(value)
        data = result.get("data")
        if isinstance(data, str) and data.strip() and not data.startswith(("http://", "https://", "data:image")):
            return data.strip()
        if isinstance(data, dict):
            for key in ("task_id", "id", "request_id"):
                value = data.get(key)
                if value:
                    return str(value)
        return ""

    def _extract_status(self, result):
        if not isinstance(result, dict):
            return ""
        for key in ("status", "task_status", "state"):
            value = result.get(key)
            if value:
                return str(value).upper()
        data = result.get("data")
        if isinstance(data, dict):
            for key in ("status", "task_status", "state"):
                value = data.get(key)
                if value:
                    return str(value).upper()
        return ""

    def _raise_api_body_error(self, result, prefix):
        if isinstance(result, list):
            raise RuntimeError(f"[Comfly_grok_image] {prefix}: {json.dumps(result[:3], ensure_ascii=False)}")
        if isinstance(result, dict) and "detail" in result:
            raise RuntimeError(f"[Comfly_grok_image] {prefix}: {result['detail']}")

    def _extract_result_items(self, result):
        if isinstance(result, list):
            return result
        if isinstance(result, str):
            return [result] if result.strip() else []
        if not isinstance(result, dict):
            return []

        if any(result.get(k) for k in ("url", "image_url", "b64_json", "base64", "image_base64")):
            return [result]

        for key in ("data", "images", "result", "output", "image", "url"):
            value = result.get(key)
            if not value:
                continue
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                nested = self._extract_result_items(value)
                if nested:
                    return nested
                continue
            if isinstance(value, str):
                return [value]
        return []

    def _item_to_tensor(self, item):
        image_url = ""
        b64_data = ""

        if isinstance(item, str):
            if item.startswith("http://") or item.startswith("https://"):
                image_url = item
            else:
                b64_data = item
        elif isinstance(item, dict):
            image_url = item.get("url") or item.get("image_url") or ""
            b64_data = item.get("b64_json") or item.get("base64") or item.get("image_base64") or ""
            nested_image = item.get("image")
            if not image_url and isinstance(nested_image, str) and nested_image.startswith(("http://", "https://")):
                image_url = nested_image
            elif not b64_data and isinstance(nested_image, str):
                b64_data = nested_image

        if b64_data:
            if b64_data.startswith("data:image"):
                b64_data = b64_data.split(",", 1)[-1]
            image_data = base64.b64decode(b64_data)
            pil_image = Image.open(BytesIO(image_data))
            return pil2tensor(pil_image), ""

        if image_url:
            response = requests.get(image_url, timeout=self.timeout)
            response.raise_for_status()
            pil_image = Image.open(BytesIO(response.content))
            return pil2tensor(pil_image), image_url

        return None, ""

    def _poll_task(self, task_id, pbar, poll_interval, max_poll_attempts):
        query_url = f"{baseurl}/v1/images/tasks/{task_id}"
        print(f"[Comfly_grok_image] Queued, task_id={task_id}, polling {query_url}")

        for attempt in range(max_poll_attempts):
            time.sleep(poll_interval)
            pbar.update_absolute(35 + min(55, int((attempt + 1) / max_poll_attempts * 55)))

            response = requests.get(query_url, headers=self.get_headers(), timeout=self.timeout)
            if response.status_code != 200:
                err_body = response.text[:500]
                try:
                    err_json = json.loads(err_body) if err_body.strip().startswith("{") else None
                    poll_status = self._extract_status(err_json) if isinstance(err_json, dict) else ""
                    if poll_status in self._WAIT_STATUSES:
                        if attempt % 10 == 0:
                            print(f"[Comfly_grok_image] Poll #{attempt+1}: HTTP {response.status_code}, status={poll_status} (waiting)")
                        continue
                except (json.JSONDecodeError, Exception):
                    pass
                raise RuntimeError(f"[Comfly_grok_image] Poll error HTTP {response.status_code}: {err_body[:300]}")

            poll_result = response.json()
            self._raise_api_body_error(poll_result, "Poll API error")

            status = self._extract_status(poll_result)
            items = self._extract_result_items(poll_result)
            if items:
                print(f"[Comfly_grok_image] Async task completed, task_id={task_id}")
                return poll_result

            if status in self._FAILED_STATUSES:
                fail_reason = (
                    poll_result.get("fail_reason")
                    or poll_result.get("error")
                    or poll_result.get("message")
                    or "unknown error"
                )
                raise RuntimeError(f"[Comfly_grok_image] Task {status}: {fail_reason}")

            if status in self._SUCCESS_STATUSES:
                raise RuntimeError(f"[Comfly_grok_image] Task {status} but no image data returned: {str(poll_result)[:300]}")

            if attempt % 10 == 0:
                print(f"[Comfly_grok_image] Polling... attempt {attempt+1}/{max_poll_attempts}, status={status or 'UNKNOWN'}")

        raise RuntimeError(f"[Comfly_grok_image] Timeout: no result after {max_poll_attempts * poll_interval}s, task_id={task_id}")

    def _process_images(self, result, pbar):
        items = self._extract_result_items(result)
        if not items:
            raise RuntimeError(f"[Comfly_grok_image] No generated image found in response: {str(result)[:500]}")

        generated_tensors = []
        result_urls = []
        for i, item in enumerate(items):
            try:
                tensor, image_url = self._item_to_tensor(item)
                if tensor is not None:
                    generated_tensors.append(tensor)
                if image_url:
                    result_urls.append(image_url)
            except Exception as e:
                print(f"[Comfly_grok_image] Error processing image item {i}: {str(e)}")
            pbar.update_absolute(90 + int((i + 1) / len(items) * 8))

        if not generated_tensors:
            raise RuntimeError("[Comfly_grok_image] No images were successfully processed")

        return torch.cat(generated_tensors, dim=0), result_urls

    def generate_image(self, prompt, api_key="", model="grok-4.2-image", aspect_ratio="1:1",
                       image1=None, image2=None, image3=None, image4=None,
                       image_urls="", image_way="image_url", async_mode=True,
                       task_id="", poll_interval=5, max_poll_attempts=180, skip_error=False):
        request_id = generate_request_id("image_gen", "xai")
        log_prepare("图像生成", request_id, "RunNode/Grok-", "xAI", model_name=model)
        rn_pbar = ProgressBar(
            request_id,
            "xAI",
            extra_info=f"模型:{model}",
            streaming=True,
            task_type="图像生成",
            source="RunNode/Grok-",
        )
        _rn_start = time.perf_counter()

        if api_key.strip():
            self.api_key = api_key.strip()
        else:
            self.api_key = get_config().get("api_key", "")

        default_image = self._blank_image()
        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(10)

        try:
            if not self.api_key:
                err = "API key not found in Comflyapi.json"
                rn_pbar.error(err)
                log_error("配置缺失", request_id, err, "RunNode/Grok-", "xAI")
                if not skip_error:
                    raise RuntimeError(f"[Comfly_grok_image] {err}")
                return (default_image, "", err)

            rn_pbar.set_generating()
            log_backend(
                "xai_grok_image_start",
                request_id=request_id,
                model=model,
                async_mode=bool(async_mode),
                task_id=task_id.strip() or None,
                elapsed_ms=0,
            )

            active_task_id = task_id.strip()
            image_refs = self._split_image_urls(image_urls)
            input_images = [img for img in (image1, image2, image3, image4) if img is not None]

            result = None
            if active_task_id:
                log_backend(
                    "xai_grok_image_query_existing_task",
                    request_id=request_id,
                    model=model,
                    task_id=active_task_id,
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )
                result = self._poll_task(active_task_id, pbar, poll_interval, max_poll_attempts)
            else:
                for index, image_tensor in enumerate(input_images):
                    if image_way == "base64":
                        image_refs.append(self._tensor_to_data_uri(image_tensor))
                    else:
                        uploaded_url = self._upload_image(image_tensor)
                        if not uploaded_url:
                            raise RuntimeError("上传参考图像失败，未获取到可用 URL")
                        image_refs.append(uploaded_url)
                    pbar.update_absolute(10 + int((index + 1) / max(len(input_images), 1) * 20))

                payload = {
                    "model": model,
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio.strip() or "1:1",
                }
                if image_refs:
                    payload["image"] = image_refs

                pbar.update_absolute(35)
                params = {"async": "true"} if async_mode else None
                mode_label = "async" if async_mode else "sync"
                log_backend(
                    "xai_grok_image_submit",
                    request_id=request_id,
                    model=model,
                    task_mode=mode_label,
                    refs_count=len(image_refs),
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )

                response = requests.post(
                    f"{baseurl}/v1/images/generations",
                    headers=self.get_headers(),
                    params=params,
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code != 200:
                    err = format_runnode_error(response)
                    rn_pbar.error(err)
                    log_error("请求失败", request_id, err, "RunNode/Grok-", "xAI")
                    if not skip_error:
                        raise RuntimeError(f"[Comfly_grok_image] {err}")
                    return (default_image, "", err)

                result = response.json()
                self._raise_api_body_error(result, "API error")

                returned_task_id = self._extract_task_id(result)
                if returned_task_id:
                    active_task_id = returned_task_id
                    log_backend(
                        "xai_grok_image_submitted",
                        request_id=request_id,
                        model=model,
                        task_id=active_task_id,
                        elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                    )
                    result = self._poll_task(active_task_id, pbar, poll_interval, max_poll_attempts)

            pbar.update_absolute(90)
            output_tensor, result_urls = self._process_images(result, pbar)

            response_info = {
                "status": "success",
                "model": model,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio.strip() or "1:1",
                "async_mode": bool(async_mode or active_task_id),
                "task_id": active_task_id,
                "input_images": len(image_refs),
                "images_count": int(output_tensor.shape[0]),
                "image_urls": result_urls,
            }
            pbar.update_absolute(100)
            rn_pbar.done(
                char_count=len(json.dumps(response_info, ensure_ascii=False)),
                elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
            )
            log_complete(
                "图像生成",
                request_id,
                "RunNode/Grok-",
                "xAI",
                image_url=safe_public_url(result_urls[0]) if result_urls else None,
            )
            log_backend(
                "xai_grok_image_done",
                request_id=request_id,
                model=model,
                task_id=active_task_id or None,
                images_count=int(output_tensor.shape[0]),
                image_url=safe_public_url(result_urls[0]) if result_urls else None,
                elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
            )
            return (output_tensor, ", ".join(result_urls), json.dumps(response_info, ensure_ascii=False, indent=2))

        except Exception as e:
            error_message = f"Error generating Grok image: {format_runnode_error(e)}"
            rn_pbar.error(error_message)
            log_error("异常", request_id, error_message, "RunNode/Grok-", "xAI")
            log_backend_exception(
                "xai_grok_image_exception",
                request_id=request_id,
                model=model,
                error=str(e),
            )
            if not skip_error:
                raise RuntimeError(f"[Comfly_grok_image] {error_message}") from e
            return (default_image, "", error_message)
