from ..comfly_config import *
from .__init__ import *


class ComflyGrok3VideoApi:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["grok-video-3"], {"default": "grok-video-3"}),
                "ratio": (["2:3", "3:2", "1:1"], {"default": "1:1"}),
                "resolution": (["720P", "1080P"], {"default": "720P"}),
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
        self.api_key = get_config().get("api_key", "")
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def upload_image(self, image_tensor, request_id=None, rn_pbar=None):
        try:
            pil_image = tensor2pil(image_tensor)[0]
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            file_content = buffered.getvalue()

            files = {"file": ("image.png", file_content, "image/png")}

            response = requests.post(
                f"{baseurl}/v1/files",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files=files,
                timeout=self.timeout,
            )

            response.raise_for_status()
            result = response.json()

            if "url" in result:
                return result["url"]
            else:
                msg = f"Unexpected response from file upload API: {result}"
                print(msg)
                log_backend(
                    "xai_grok_upload_unexpected_response",
                    level="ERROR",
                    request_id=request_id,
                    url=safe_public_url(baseurl),
                )
                if rn_pbar is not None:
                    rn_pbar.error("文件上传返回异常，请稍后重试")
                return None

        except Exception as e:
            msg = f"Error uploading image: {str(e)}"
            print(msg)
            log_backend_exception(
                "xai_grok_upload_exception",
                request_id=request_id,
                url=safe_public_url(baseurl),
            )
            if rn_pbar is not None:
                rn_pbar.error("上传参考图像失败，请检查网络或图像格式")
            return None

    def generate_video(self, prompt, model, ratio, resolution, api_key="", image=None, seed=0):
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
            error_response = {"code": "error", "message": error_message}
            return ("", "", json.dumps(error_response), "")

        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            payload = {
                "prompt": prompt,
                "model": model,
                "ratio": ratio,
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
                    error_response = {"code": "error", "message": error_message}
                    return ("", "", json.dumps(error_response), "")

            pbar.update_absolute(30)

            log_backend(
                "xai_video_generate_start",
                request_id=request_id,
                url=safe_public_url(baseurl),
                model=model,
                prompt_len=len(prompt or ""),
                ratio=ratio,
                resolution=resolution,
                has_image=bool(image is not None),
                seed=(int(seed) if int(seed) > 0 else None),
            )

            response = requests.post(
                f"{baseurl}/v2/videos/generations",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout,
            )

            if response.status_code != 200:
                error_message = f"API error: {response.status_code} - {response.text}"
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
                error_response = {"code": "error", "message": error_message}
                return ("", "", json.dumps(error_response), "")

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
                error_response = {"code": "error", "message": error_message}
                return ("", "", json.dumps(error_response), "")

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
                    error_response = {"code": "error", "message": error_message}
                    return ("", task_id, json.dumps(error_response), "")

                time.sleep(5)
                attempts += 1

                try:
                    status_response = requests.get(
                        f"{baseurl}/v2/videos/generations/{task_id}",
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
                        error_message = f"Video generation failed: {fail_reason}"
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
                        error_response = {"code": "error", "message": error_message}
                        return ("", task_id, json.dumps(error_response), "")
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
                error_response = {"code": "error", "message": error_message}
                return ("", task_id, json.dumps(error_response), "")

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
                    url=safe_public_url(baseurl),
                    model=model,
                    task_id=task_id,
                    video_url=safe_public_url(video_url),
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )
                return (video_adapter, task_id, json.dumps(response_data), video_url)

        except Exception as e:
            error_message = f"Error generating video: {str(e)}"
            rn_pbar.error(error_message)
            traceback.print_exc()
            log_backend_exception(
                "xai_video_generate_exception",
                request_id=request_id,
                url=safe_public_url(baseurl),
                model=model,
            )
            error_response = {"code": "error", "message": error_message}
            return ("", "", json.dumps(error_response), "")


