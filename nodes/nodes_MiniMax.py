from ..comfly_config import *
from .__init__ import *


class Comfly_MiniMax_video:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "tooltip": "描述视频内容的提示词"}),
                "model": (["MiniMax-Hailuo-02", "T2V-01", "T2V-01-Director", "I2V-01-Director", 
                            "I2V-01-live", "I2V-01", "S2V-01", "MiniMax-Hailuo-2.3-Fast",
                            "MiniMax-Hailuo-2.3"], 
                            {"default": "MiniMax-Hailuo-02", "tooltip": "使用的模型版本"}),
                "duration": (["6", "10"], {"default": "6", "tooltip": "生成视频的时长（秒）"}),
                "resolution": (["720P","768P", "1080P"], {"default": "768P", "tooltip": "视频分辨率"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "tooltip": "MiniMax API 密钥，留空则使用全局配置"}),
                # "api_key": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "prompt_optimizer": ("BOOLEAN", {"default": True, "tooltip": "是否启用提示词优化"}),
                "fast_pretreatment": ("BOOLEAN", {"default": False, "tooltip": "是否启用快速预处理"}),
                "first_frame_image": ("IMAGE", {"tooltip": "首帧参考图像"}),
                "last_frame_image": ("IMAGE", {"tooltip": "尾帧参考图像"}),
                "subject_reference": ("IMAGE", {"tooltip": "主体参考图像"}),  
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "tooltip": "随机种子"}),
            }
        }
    
    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video", "task_id", "response")
    FUNCTION = "generate_video"
    CATEGORY = "RunNode/MiniMax"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 600
        self.api_endpoint = f"{baseurl}/minimax/v1/video_generation"
        self.query_endpoint = f"{baseurl}/minimax/v1/query/video_generation"

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def image_to_base64(self, image_tensor):
        """Convert tensor to base64 string"""
        if image_tensor is None:
            return None
            
        pil_image = tensor2pil(image_tensor)[0]
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def generate_video(self, prompt, model="MiniMax-Hailuo-02", duration="6", resolution="768P", 
               prompt_optimizer=True, fast_pretreatment=False, first_frame_image=None, last_frame_image=None,
               subject_reference=None, api_key="", seed=0):
        request_id = generate_request_id("video_gen", "minimax")
        
        log_params = {
            "model_name": model,
            "duration": duration,
            "resolution": resolution,
            "prompt_optimizer": prompt_optimizer,
        }
        if seed != 0: log_params["seed"] = seed
        if fast_pretreatment: log_params["fast_pretreatment"] = True
        
        log_prepare("视频生成", request_id, "RunNode/MiniMax-", "MiniMax", **log_params)
        rn_pbar = ProgressBar(request_id, "MiniMax", streaming=True, task_type="视频生成", source="RunNode/MiniMax-")
        rn_pbar.set_generating(0)
        
        if api_key.strip():
            self.api_key = api_key
            # config = get_config()
            # config['api_key'] = api_key
            # save_config(config)
        else:
            self.api_key = get_config().get('api_key', '')
            
        if not self.api_key:
            error_message = "API key not provided or not found in config"
            log_error("配置错误", request_id, error_message, "RunNode/MiniMax-", "MiniMax")
            raise Exception(error_message)
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            payload = {
                "model": model,
                "prompt": prompt,
                "duration": int(duration),
                "resolution": resolution,
                "prompt_optimizer": prompt_optimizer
            }

            if seed > 0:
                payload["seed"] = seed
                
            log_backend("minimax_start", request_id=request_id, model=model)

            if model == "MiniMax-Hailuo-02":
                payload["fast_pretreatment"] = fast_pretreatment

                if fast_pretreatment and last_frame_image is not None:
                    rn_pbar.error("fast_pretreatment is disabled when last_frame_image is provided")
                    payload["fast_pretreatment"] = False

            if model in ["T2V-01", "T2V-01-Director"]:
                if first_frame_image is not None or last_frame_image is not None:
                    rn_pbar.error(f"Model {model} only supports text-to-video. Image inputs will be ignored.")
                
            elif model in ["I2V-01-Director", "I2V-01-live", "I2V-01"]:
                if first_frame_image is None:
                    rn_pbar.error(f"Model {model} requires first_frame_image for image-to-video generation.")
                if last_frame_image is not None:
                    rn_pbar.error(f"Model {model} doesn't support last_frame_image. It will be ignored.")

            if first_frame_image is not None and model != "T2V-01" and model != "T2V-01-Director":
                image_base64 = self.image_to_base64(first_frame_image)
                if image_base64:
                    payload["first_frame_image"] = f"data:image/png;base64,{image_base64}"

            if last_frame_image is not None and model == "MiniMax-Hailuo-02":
                image_base64 = self.image_to_base64(last_frame_image)
                if image_base64:
                    payload["last_frame_image"] = f"data:image/png;base64,{image_base64}"

            if model == "S2V-01" and subject_reference is not None:
                image_base64 = self.image_to_base64(subject_reference)
                if image_base64:
                    payload["subject_reference"] = {
                        "type": "character",
                        "image": [f"data:image/png;base64,{image_base64}"]
                    }
            
            response = requests.post(
                self.api_endpoint,
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(30)
            
            if response.status_code != 200:
                error_message = format_runnode_error(response)
                rn_pbar.error(error_message)
                log_error("API请求失败", request_id, error_message, "RunNode/MiniMax-", "MiniMax")
                return (None, "", json.dumps({"status": "error", "message": error_message}))
                
            result = response.json()
            
            if "base_resp" not in result or result["base_resp"]["status_code"] != 0:
                error_message = f"API returned error: {result.get('base_resp', {}).get('status_msg', 'Unknown error')}"
                rn_pbar.error(error_message)
                log_error("API返回错误", request_id, error_message, "RunNode/MiniMax-", "MiniMax")
                return (None, "", json.dumps({"status": "error", "message": error_message}))
                
            task_id = result.get("task_id")
            if not task_id:
                error_message = "No task ID returned from API"
                rn_pbar.error(error_message)
                log_error("缺失TaskID", request_id, error_message, "RunNode/MiniMax-", "MiniMax")
                return (None, "", json.dumps({"status": "error", "message": error_message}))
            
            pbar.update_absolute(40)
            

            max_attempts = 120  
            attempts = 0
            file_id = None
            video_url = None
            
            log_backend("minimax_poll_start", request_id=request_id, task_id=task_id)

            while attempts < max_attempts:
                time.sleep(10)  
                attempts += 1
                
                try:
                    log_backend("minimax_poll_check", request_id=request_id, task_id=task_id, attempt=attempts)
                    
                    status_response = requests.get(
                        f"{self.query_endpoint}?task_id={task_id}",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )
                    
                    if status_response.status_code != 200:
                        msg = f"Error checking status: {format_runnode_error(status_response)}"
                        rn_pbar.error(msg)
                        log_backend_exception(msg, request_id=request_id)
                        continue
                        
                    status_result = status_response.json()
                    
                    if "base_resp" not in status_result or status_result["base_resp"]["status_code"] != 0:
                        msg = f"Error in status response: {status_result.get('base_resp', {}).get('status_msg', 'Unknown error')}"
                        rn_pbar.error(msg)
                        log_backend_exception(msg, request_id=request_id)
                        continue
                    
                    status = status_result.get("status", "")
                    
                    progress_value = min(80, 40 + (attempts * 40 // max_attempts))
                    pbar.update_absolute(progress_value)
                    
                    if status == "Success":
                        file_id = status_result.get("file_id")
                        if file_id:
                            video_retrieval_url = f"{baseurl}/minimax/v1/files/retrieve?file_id={file_id}"
                            file_response = requests.get(
                                video_retrieval_url,
                                headers=self.get_headers(),
                                timeout=self.timeout
                            )
                            
                            if file_response.status_code == 200:
                                file_data = file_response.json()
                                if "file" in file_data and "download_url" in file_data["file"]:
                                    video_url = file_data["file"]["download_url"]
                                else:
                                    video_url = f"{baseurl}/minimax/v1/file?file_id={file_id}"
                            else:
                                video_url = f"{baseurl}/minimax/v1/file?file_id={file_id}"
                                
                            log_backend("minimax_poll_success", request_id=request_id, video_url=safe_public_url(video_url))
                            break
                        else:
                            # Success but no file_id? weird.
                            log_backend("minimax_poll_success_no_fileid", request_id=request_id)
                            break
                            
                    elif status == "Failed":
                        error_message = f"Video generation failed: {status_result.get('base_resp', {}).get('status_msg', 'Unknown error')}"
                        rn_pbar.error(error_message)
                        log_error("视频生成失败", request_id, error_message, "RunNode/MiniMax-", "MiniMax")
                        raise Exception(error_message)
                    
                except Exception as e:
                    rn_pbar.error(f"Error checking generation status: {format_runnode_error(str(e))}")
                    log_backend_exception(f"Error checking generation status: {format_runnode_error(str(e))}", request_id=request_id)
            
            if not file_id and not video_url:
                error_message = "Failed to retrieve file_id after multiple attempts"
                rn_pbar.error(error_message)
                log_error("视频生成超时", request_id, error_message, "RunNode/MiniMax-", "MiniMax")
                raise Exception(error_message)
                
            if not video_url:
                video_url = f"{baseurl}/minimax/v1/file?file_id={file_id}"
            
            pbar.update_absolute(90)
            
            
            video_adapter = ComflyVideoAdapter(video_url)
            
            response_data = {
                "status": "success",
                "task_id": task_id,
                "file_id": file_id,
                "video_url": safe_public_url(video_url),
                "width": status_result.get("video_width", 0),
                "height": status_result.get("video_height", 0)
            }
            
            pbar.update_absolute(100)
            log_complete("视频生成完成", request_id, "RunNode/MiniMax-", "MiniMax", video_url=safe_public_url(video_url))
            rn_pbar.done(char_count=len(json.dumps(response_data)))
            return (video_adapter, task_id, json.dumps(response_data))
            
        except Exception as e:
            error_message = f"Error generating video: {format_runnode_error(str(e))}"
            rn_pbar.error(error_message)
            log_error("未捕获异常", request_id, error_message, "RunNode/MiniMax-", "MiniMax")
            raise
