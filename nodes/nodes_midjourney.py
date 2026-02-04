from ..comfly_config import *
from .__init__ import *


class ComflyBaseNode:
    def __init__(self):
        self.midjourney_api_url = {
            "turbo mode": f"{baseurl}/mj-turbo",
            "fast mode": f"{baseurl}/mj-fast",
            "relax mode": f"{baseurl}/mj-relax"
        }
        self.api_key = get_config().get('api_key', '') 
        self.speed = "fast mode"
        self.timeout = 800

    def set_speed(self, speed):
        self.speed = speed

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key
        }

    def generateUUID(self):
        return str(uuid.uuid4())

    async def midjourney_submit_action(self, action, taskId, index, custom_id, request_id=None):
        headers = self.get_headers()
        payload = {
            "customId": custom_id,
            "taskId": taskId
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.midjourney_api_url[self.speed]}/mj/submit/action", headers=headers, json=payload, timeout=self.timeout) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()

                            if data.get("status") == "FAILURE":
                                error_message = format_runnode_error(data)
                                log_error("操作提交失败", request_id, error_message, "RunNode/Midjourney-", "Midjourney")
                                raise Exception(error_message)
                                
                            return data
                        except aiohttp.client_exceptions.ContentTypeError:
                            text_response = await response.text()
                            
                            try:
                                import json
                                data = json.loads(text_response)

                                if data.get("status") == "FAILURE":
                                    error_message = format_runnode_error(data)
                                    log_error("操作提交失败", request_id, error_message, "RunNode/Midjourney-", "Midjourney")
                                    raise Exception(error_message)
                                    
                                return data
                            except json.JSONDecodeError:
                                if text_response and len(text_response) < 100:
                                    return {"result": text_response.strip()}
                                log_error("API响应解析失败", request_id, f"Invalid response: {text_response}", "RunNode/Midjourney-", "Midjourney")
                                raise Exception(f"Invalid response format: {text_response}")
                    else:
                        try:
                            error_text = await response.text()
                        except:
                            error_text = ""
                        
                        error_message = format_runnode_error(error_text)
                        if not error_message.startswith("API Error") and str(response.status) not in error_message:
                             error_message = f"API Error: {response.status} - {error_message}"
                             
                        log_error("API请求失败", request_id, error_message, "RunNode/Midjourney-", "Midjourney")
                        raise Exception(error_message)
        except asyncio.TimeoutError:
            error_message = f"Timeout error: Request to submit action timed out after {self.timeout} seconds"
            log_error("请求超时", request_id, error_message, "RunNode/Midjourney-", "Midjourney")
            raise Exception(error_message)
        except Exception as e:
            if "Action submission failed" in str(e):
                raise 
            # print(f"Exception in midjourney_submit_action: {format_runnode_error(str(e))}")
            log_backend_exception(f"Exception in submit_action: {format_runnode_error(str(e))}", request_id=request_id)
            raise e


    def midjourney_upload_image_sync(self, image, request_id=None):
        pil_image = tensor2pil(image)[0]
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        payload = {
            "base64Array": [f"data:image/png;base64,{image_base64}"],
            "instanceId": "",
            "notifyHook": ""
        }
        
        try:
            response = requests.post(
                f"{self.midjourney_api_url[self.speed]}/mj/submit/upload-discord-images",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "FAILURE":
                    error_message = format_runnode_error(result)
                    if request_id:
                        log_error("图片上传失败", request_id, error_message, "RunNode/Midjourney-", "Midjourney")
                    raise Exception(error_message)
                    
                if "result" in result and result["result"]:
                    return result["result"][0]
                else:
                    error_message = f"Unexpected response: {format_runnode_error(result)}"
                    if request_id:
                        log_error("图片上传异常", request_id, error_message, "RunNode/Midjourney-", "Midjourney")
                    raise Exception(error_message)
            else:
                error_message = format_runnode_error(response)
                if request_id:
                    log_error("图片上传请求失败", request_id, error_message, "RunNode/Midjourney-", "Midjourney")
                raise Exception(error_message)
        except Exception as e:
            if request_id:
                log_backend_exception(f"Exception in upload_image: {str(e)}", request_id=request_id)
            raise e

    def extract_taskId(self, U, action, index):
        pattern = fr'"customId": "MJ::JOB::{action}::{index}::(.*?)"'
        match = re.search(pattern, U)
        if match:
            return match.group(1)
        else:
            raise ValueError(f"Invalid custom ID format in U: {U}")

    async def midjourney_submit_imagine_task(self, prompt, ar, no, c, s, iw, tile, r, video, sw, cw, sv, seed, request_id=None):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key
        }
        payload = {
            "base64Array": [],
            "instanceId": "",
            "modes": [],
            "notifyHook": "",
            "prompt": prompt,
            "remix": True,
            "state": "",
            "ar": ar,
            "no": no,
            "c": c,
            "s": s,
            "iw": iw,
            "tile": tile,
            "r": r,
            "video": video,
            "sw": sw,
            "cw": cw,
            "sv": sv,
            "seed": seed
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.midjourney_api_url[self.speed]}/mj/submit/imagine", headers=headers, json=payload, timeout=self.timeout) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()

                            if data.get("status") == "FAILURE":
                                error_message = format_runnode_error(data)
                                log_error("任务提交被拒", request_id, error_message, "RunNode/Midjourney-", "Midjourney")
                                raise Exception(error_message)
                                
                            return data["result"]
                        except aiohttp.client_exceptions.ContentTypeError:
                            text_response = await response.text()
                            
                            try:
                                import json
                                data = json.loads(text_response)
 
                                if data.get("status") == "FAILURE":
                                    error_message = format_runnode_error(data)
                                    log_error("任务提交被拒", request_id, error_message, "RunNode/Midjourney-", "Midjourney")
                                    raise Exception(error_message)
                                    
                                return data["result"]
                            except (json.JSONDecodeError, KeyError):
                                if text_response and len(text_response) < 100:
                                    return text_response.strip()
                                log_error("API响应解析失败", request_id, f"Invalid response: {text_response}", "RunNode/Midjourney-", "Midjourney")
                                raise Exception(f"Server returned invalid response: {text_response}")
                    else:
                        try:
                            error_text = await response.text()
                        except:
                            error_text = ""
                        
                        error_message = format_runnode_error(error_text)
                        if not error_message.startswith("API Error") and str(response.status) not in error_message:
                             error_message = f"API Error: {response.status} - {error_message}"

                        log_error("API请求失败", request_id, error_message, "RunNode/Midjourney-", "Midjourney")
                        raise Exception(error_message)
        except asyncio.TimeoutError:
            error_message = f"Timeout error: Request to submit imagine task timed out after {self.timeout} seconds"
            log_error("请求超时", request_id, error_message, "RunNode/Midjourney-", "Midjourney")
            raise Exception(error_message)
        except Exception as e:
            if "Midjourney task failed" in str(e):
                raise  
            # print(f"Exception in midjourney_submit_imagine_task: {format_runnode_error(str(e))}")
            log_backend_exception(f"Exception in submit_imagine: {format_runnode_error(str(e))}", request_id=request_id)
            raise e


    async def midjourney_fetch_task_result(self, taskId, request_id=None):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.midjourney_api_url[self.speed]}/mj/task/{taskId}/fetch", headers=headers, timeout=self.timeout) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()
                            return data
                        except aiohttp.client_exceptions.ContentTypeError:
                            text_response = await response.text()
                            try:
                                import json
                                data = json.loads(text_response)
                                return data
                            except json.JSONDecodeError:
                                if text_response and len(text_response) < 100:
                                    return {"status": "SUCCESS", "progress": "100%", "imageUrl": text_response.strip()}
                                log_error("API响应解析失败", request_id, f"Invalid response: {text_response}", "RunNode/Midjourney-", "Midjourney")
                                raise Exception(f"Server returned invalid response: {text_response}")
                    else:
                        try:
                            error_text = await response.text()
                        except:
                            error_text = ""
                        
                        error_message = format_runnode_error(error_text)
                        if not error_message.startswith("API Error") and str(response.status) not in error_message:
                             error_message = f"API Error: {response.status} - {error_message}"
                        # Transient error, let caller handle
                        raise Exception(error_message)
        except asyncio.TimeoutError:
            error_message = f"Timeout error: Request to fetch task result timed out after {self.timeout} seconds"
            # log_error("请求超时", request_id, error_message, "RunNode/Midjourney-", "Midjourney")
            raise Exception(error_message)


class Comfly_upload(ComflyBaseNode):
    """
    Comfly_upload node
    Uploads an image to Midjourney and returns the URL link.
    Inputs:
        image (IMAGE): Input image to be uploaded.
    Outputs:
        url (STRING): URL link of the uploaded image.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"default": ""}),
                # "api_key": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("url",)
    FUNCTION = "upload_image"
    CATEGORY = "RunNode/Midjourney"

    def upload_image(self, image, api_key=""):
        request_id = generate_request_id("mj_upload", "midjourney")
        log_prepare("Midjourney图片上传", request_id, "RunNode/Midjourney-", "Midjourney")
        rn_pbar = ProgressBar(request_id, "Midjourney", streaming=True, task_type="图片上传", source="RunNode/Midjourney-")

        if api_key.strip():
            self.api_key = api_key
        else:
            self.api_key = get_config().get('api_key', '')
            
        try:
            log_backend("mj_upload_start", request_id=request_id)
            
            image_url = self.midjourney_upload_image_sync(image, request_id)
            
            log_complete("图片上传成功", request_id, "RunNode/Midjourney-", "Midjourney", image_url=safe_public_url(image_url))
            rn_pbar.done(char_count=len(image_url))
            return (image_url,)
                
        except Exception as e:
            rn_pbar.error(str(e))
            # Error already logged in midjourney_upload_image_sync
            raise e
        

class Comfly_Mj(ComflyBaseNode):
    """
    Comfly_Mj node
    Processes text or image inputs using Midjourney AI model and returns the processed results.
    Inputs:
        text (STRING, optional): Input text.
        api_key (STRING): API key for Midjourney.
        model_version (STRING): Selected Midjourney model version (v 6.1, v 6.0, v 5.2, v 5.1, niji 6, niji 5, niji 4).
        speed (STRING): Selected speed mode (turbo mode, fast mode, relax mode).
        ar (STRING): Aspect ratio.
        no (STRING): Number of images.
        c (STRING): Chaos value (0-100).
        s (STRING): Stylize value (0-1000).
        iw (STRING): Image weight (0-2).
        tile (BOOL): Enable/disable tile.
        r (STRING): Repeat value (1-40).
        video (BOOL): Enable/disable video.
        sw (STRING): Style weight (0-1000).
        cw (STRING): Color weight (0-100).
        sv (STRING): Style variation (1-4).
        seed (INT): Random seed.
        cref (STRING): Creative reference.
        oref (STRING): Object reference.
        sref (STRING): Style reference.
        positive (STRING): Additional positive prompt to be appended to the main prompt.
    Outputs:
        image_url (STRING): URL of the processed image.
        text (STRING): Processed text output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "tooltip": "主提示词。会与下方参数一起拼成最终 prompt 并提交生成。"}),
                "speed": (["turbo mode", "fast mode", "relax mode"], {"default": "fast mode", "tooltip": "速度模式。仅影响请求路由：/mj-turbo、/mj-fast、/mj-relax（由后端决定排队/计费/速度差异）。"}), 
            },
            "optional": {
                "text_en": ("STRING", {"multiline": True, "default": "", "tooltip": "可选英文提示词。非空时会替代 text 作为基础 prompt。"}),
                "api_key": ("STRING", {"default": "", "tooltip": "可选。用于 Authorization Bearer；填写后会写入本插件配置并覆盖默认 key。"}),  
                # "api_key": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "ar": ("STRING", {"default": "1:1", "tooltip": "画幅比例参数：--ar，例如 1:1、16:9、2:3。"}),
                "model_version": (["v 7", "v 6.1", "v 6.0", "v 5.2", "v 5.1", "niji 7", "niji 6", "niji 5", "niji 4"], {"default": "v 7", "tooltip": "模型版本。会追加到 prompt：--v 6.1 / --niji 6 等。"}),
                "no": ("STRING", {"default": "", "forceInput": True, "tooltip": "排除内容：--no，例如 '--no text' 中的 text。"}),
                "c": ("INT", {"default": 0, "min": 0, "max": 100, "forceInput": True, "tooltip": "混沌度：--c，范围 0-100。越高越随机。"}),
                "s": ("INT", {"default": 0, "min": 0, "max": 1000, "forceInput": True, "tooltip": "风格化强度：--s，范围 0-1000。"}),
                "iw": ("FLOAT", {"default": 0, "min": 0, "max": 2, "forceInput": True, "tooltip": "图片权重：--iw，范围 0-2（仅在有图参考时有意义）。"}),
                "r": ("INT", {"default": 1, "min": 1, "max": 40, "forceInput": True, "tooltip": "重复次数：--r，范围 1-40。"}),
                "sw": ("INT", {"default": 0, "min": 0, "max": 1000, "forceInput": True, "tooltip": "风格权重：--sw，范围 0-1000。"}),
                "cw": ("INT", {"default": 0, "min": 0, "max": 100, "forceInput": True, "tooltip": "色彩权重：--cw，范围 0-100。"}),
                "sv": (["1", "2", "3", "4"], {"default": "1", "forceInput": True, "tooltip": "风格变化：--sv，可选 1-4。"}),
                "oref": ("STRING", {"default": "none", "forceInput": True, "tooltip": "对象参考：--oref。填入引用信息；为 none 时不追加。"}),
                "cref": ("STRING", {"default": "none", "forceInput": True, "tooltip": "创意参考：--cref。填入引用信息；为 none 时不追加。"}),
                "sref": ("STRING", {"default": "none", "forceInput": True, "tooltip": "风格参考：--sref。填入引用信息；为 none 时不追加。"}),
                "positive": ("STRING", {"default": "", "forceInput": True, "tooltip": "附加正向提示词。会直接拼到主提示词后面。"}),                
                "video": ("BOOLEAN", {"default": False, "tooltip": "是否追加 --video 参数。"}),
                "tile": ("BOOLEAN", {"default": False, "tooltip": "是否追加 --tile 参数（平铺纹理）。"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "tooltip": "随机种子：seed。0 表示不指定。"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")  
    RETURN_NAMES = ("image", "text", "taskId")    
    OUTPUT_NODE = True

    FUNCTION = "process_input"

    CATEGORY = "RunNode/Midjourney"

    def __init__(self):
        super().__init__()
        self.image = None
        self.text = ""

    def process_input(self, speed, text, text_en="", image=None, model_version=None, ar=None, no=None, c=None, s=None, iw=None, r=None, sw=None, cw=None, sv=None, video=False, tile=False, seed=0, cref="none", oref="none", sref="none", positive="", api_key=""):
        request_id = generate_request_id("mj_generate", "midjourney")
        
        log_prepare("Midjourney生成", request_id, "RunNode/Midjourney-", "Midjourney", model_version=model_version, speed=speed)
        
        rn_pbar = ProgressBar(request_id, "Midjourney", streaming=True, task_type="图像生成", source="RunNode/Midjourney-")

        if api_key.strip():
            self.api_key = api_key
        else:
            self.api_key = get_config().get('api_key', '')
            
        self.image = image
        self.speed = speed

        prompt = text_en if text_en else text

        if positive:
            prompt += f" {positive}"

        if self.image is not None:
            try:
                log_backend("mj_upload_start_internal", request_id=request_id)
                image_url = self.midjourney_upload_image_sync(self.image, request_id)
                prompt = f"{image_url} {prompt}"
                log_backend("mj_upload_success_internal", request_id=request_id, url=image_url)
            except Exception as e:
                rn_pbar.error(f"Image upload failed: {str(e)}")
                raise e

        if model_version:
            prompt += f" --{model_version}"
        if ar:
            prompt += f" --ar {ar}"
        if no:
            prompt += f" --no {no}"
        if c:
            prompt += f" --c {c}"
        if s:
            prompt += f" --s {s}"
        if iw:
            prompt += f" --iw {iw}"
        if r:
            prompt += f" --r {r}"
        if sw:
            prompt += f" --sw {sw}"
        if cw:
            prompt += f" --cw {cw}"
        if sv:
            prompt += f" --sv {sv}"
        if video:
            prompt += " --video"
        if tile:
            prompt += " --tile"
        if oref != "none":
            prompt += f" --oref {oref}"    
        if cref != "none":
            prompt += f" --cref {cref}"
        if sref != "none":
            prompt += f" --sref {sref}"

        self.text = prompt
        
        log_backend("mj_prompt_ready", request_id=request_id, prompt=prompt)

        try:
            if self.text:
                pbar = comfy.utils.ProgressBar(100)
                image_url, text, taskId = self.process_text(pbar, ar, no, c, s, iw, tile, r, video, sw, cw, sv, seed, request_id, rn_pbar)
                
                response = requests.get(image_url)
                image = Image.open(BytesIO(response.content))
                tensor_image = pil2tensor(image)
                
                log_complete("Midjourney生成完成", request_id, "RunNode/Midjourney-", "Midjourney", image_url=safe_public_url(image_url))
                rn_pbar.done(char_count=len(image_url))
                
                return tensor_image, text, taskId
            else:
                error_message = "Either image or text input must be provided for Midjourney model."
                rn_pbar.error(error_message)
                log_error("输入缺失", request_id, error_message, "RunNode/Midjourney-", "Midjourney")
                raise ValueError(error_message)
        except Exception as e:
            if "Midjourney task failed" in str(e):
                raise e
            rn_pbar.error(f"Process failed: {format_runnode_error(str(e))}")
            log_error("处理失败", request_id, format_runnode_error(str(e)), "RunNode/Midjourney-", "Midjourney")
            raise e
        
        return image_url, text, taskId

    def process_text_midjourney_sync(self, text, pbar, ar, no, c, s, iw, tile, r, video, sw, cw, sv, seed, request_id, rn_pbar):
        try:
            log_backend("mj_submit_start", request_id=request_id)
            taskId = self.midjourney_submit_imagine_task_sync(text, ar, no, c, s, iw, tile, r, video, sw, cw, sv, seed, request_id)
            log_backend("mj_submit_success", request_id=request_id, task_id=taskId)
            
            task_result = None
            attempts = 0

            while True:
                time.sleep(2) # Increased from 1 to 2 to reduce spam
                attempts += 1
                try:
                    # Log heartbeat every 15 seconds (approx every 7-8 loops)
                    if attempts % 8 == 0:
                        log_backend("mj_poll_check", request_id=request_id, task_id=taskId)
                        
                    task_result = self.midjourney_fetch_task_result_sync(taskId, request_id)
 
                    if task_result.get("status") == "FAILURE":
                        fail_reason = task_result.get("fail_reason", "Unknown failure reason")
                        error_message = f"Midjourney task failed: {format_runnode_error(fail_reason)}"
                        rn_pbar.error(error_message)
                        log_error("任务执行失败", request_id, error_message, "RunNode/Midjourney-", "Midjourney")
                        raise Exception(error_message)  
                    if task_result.get("status") == "SUCCESS":
                        log_backend("mj_poll_success", request_id=request_id, task_id=taskId)
                        break
                        
                    progress = task_result.get("progress", 0)
                    try:
                        progress_int = int(str(progress).replace('%', '')) # Handle "50%" or 50
                    except (ValueError, TypeError):
                        progress_int = 0
                    pbar.update_absolute(progress_int)
                    # rn_pbar.set_progress(progress_int) # Optional: Update RN pbar if supported

                except Exception as e:
                    if "Midjourney task failed" in str(e):
                        raise  
                    rn_pbar.error(f"Error fetching task result: {format_runnode_error(str(e))}")
                    log_backend_exception(f"Error fetching task result: {format_runnode_error(str(e))}", request_id=request_id)
                    time.sleep(2)
                    continue
                
            image_url = task_result.get("imageUrl", "")
            prompt = task_result.get("prompt", text)

            U1 = self.generate_custom_id("upsample", 1, task_result.get("id", taskId))
            U2 = self.generate_custom_id("upsample", 2, task_result.get("id", taskId))
            U3 = self.generate_custom_id("upsample", 3, task_result.get("id", taskId))
            U4 = self.generate_custom_id("upsample", 4, task_result.get("id", taskId))

            return image_url, prompt, U1, U2, U3, U4, taskId 
        
        except Exception as e:
            # print(f"Error in process_text_midjourney_sync: {str(e)}")
            # Already logged in outer block or inner catch
            raise e

    def midjourney_submit_imagine_task_sync(self, prompt, ar, no, c, s, iw, tile, r, video, sw, cw, sv, seed, request_id=None):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key
        }
        payload = {
            "base64Array": [],
            "instanceId": "",
            "modes": [],
            "notifyHook": "",
            "prompt": prompt,
            "remix": True,
            "state": "",
            "ar": ar,
            "no": no,
            "c": c,
            "s": s,
            "iw": iw,
            "tile": tile,
            "r": r,
            "video": video,
            "sw": sw,
            "cw": cw,
            "sv": sv,
            "seed": seed
        }
        try:
            response = requests.post(
                f"{self.midjourney_api_url[self.speed]}/mj/submit/imagine", 
                headers=headers, 
                json=payload, 
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                error_message = f"Error submitting Midjourney task: {response.status_code}"
                try:
                    error_details = response.text
                    error_message += f" - {format_runnode_error(error_details)}"
                except:
                    pass
                log_error("API提交失败", request_id, error_message, "RunNode/Midjourney-", "Midjourney")
                raise Exception(error_message)
                
            try:
                data = response.json()
                
                if data.get("status") == "FAILURE":
                    fail_reason = data.get("fail_reason", "Unknown failure reason")
                    error_message = f"Midjourney task failed: {format_runnode_error(fail_reason)}"
                    log_error("任务提交被拒", request_id, error_message, "RunNode/Midjourney-", "Midjourney")
                    raise Exception(error_message)
                    
                return data["result"]
            except (json.JSONDecodeError, KeyError):
                text_response = response.text
                if text_response and len(text_response) < 100:
                    return text_response.strip()
                log_error("API响应解析失败", request_id, f"Invalid response: {text_response}", "RunNode/Midjourney-", "Midjourney")
                raise Exception(f"Server returned invalid response: {text_response}")
        except Exception as e:
            if "Midjourney task failed" in str(e):
                raise  
            # print(f"Exception in midjourney_submit_imagine_task_sync: {str(e)}")
            log_backend_exception(f"Exception in submit: {str(e)}", request_id=request_id)
            raise e

    def midjourney_fetch_task_result_sync(self, taskId, request_id=None):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key
        }
        try:
            response = requests.get(
                f"{self.midjourney_api_url[self.speed]}/mj/task/{taskId}/fetch",
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                error_message = f"Error fetching Midjourney task result: {response.status_code}"
                try:
                    error_details = response.text
                    error_message += f" - {format_runnode_error(error_details)}"
                except:
                    pass
                # Don't log error here as it might be transient, caller handles retry or logging
                raise Exception(error_message)
                
            try:
                data = response.json()
                return data
            except json.JSONDecodeError:
                text_response = response.text
                if text_response and len(text_response) < 100:
                    return {"status": "SUCCESS", "progress": "100%", "imageUrl": text_response.strip()}
                raise Exception(f"Server returned invalid response: {text_response}")
        except Exception as e:
            # print(f"Error in midjourney_fetch_task_result_sync: {format_runnode_error(str(e))}")
            raise e
            
    def process_text(self, pbar, ar, no, c, s, iw, tile, r, video, sw, cw, sv, seed, request_id, rn_pbar):

        image_url, text, U1, U2, U3, U4, taskId = self.process_text_midjourney_sync(self.text, pbar, ar, no, c, s, iw, tile, r, video, sw, cw, sv, seed, request_id, rn_pbar)
        
        U = json.dumps({"U1": U1, "U2": U2, "U3": U3, "U4": U4})

        self.taskId = taskId
    
        return image_url, text, self.taskId

    def generate_custom_id(self, action, index, job_id=None):
        uuid_val = job_id if job_id else self.generateUUID()
        if action in ["upsample_v6_2x_subtle", "upsample_v6_2x_creative", "pan_left", "pan_right", "pan_up", "pan_down"]:
            return f"MJ::JOB::{action}::{index}::{uuid_val}::SOLO"
        elif action == "Outpaint::50":
            return f"MJ::Outpaint::50::{index}::{uuid_val}::SOLO"
        else:
            return f"MJ::JOB::{action}::{index}::{uuid_val}"
    
    
class Comfly_Mju(ComflyBaseNode):
    class MidjourneyError(Exception):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "taskId": ("STRING", {"default": "", "forceInput": True}),
                "U1": ("BOOLEAN", {"default": False}),
                "U2": ("BOOLEAN", {"default": False}),
                "U3": ("BOOLEAN", {"default": False}),
                "U4": ("BOOLEAN", {"default": False})
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                # "api_key": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")  
    RETURN_NAMES = ("image", "taskId")  
    FUNCTION = "run"
    CATEGORY = "RunNode/Midjourney"

    def run(self, taskId, U1=False, U2=False, U3=False, U4=False, api_key=""):
        request_id = generate_request_id("mj_upscale", "midjourney")
        log_prepare("Midjourney操作", request_id, "RunNode/Midjourney-", "Midjourney", taskId=taskId)

        if api_key.strip():
            self.api_key = api_key
        else:
            self.api_key = get_config().get('api_key', '')
        
        try:
            try:
                current_loop = asyncio.get_running_loop()
            
                def run_async():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(self.process_input(taskId, request_id, U1, U2, U3, U4))
                    finally:
                        new_loop.close()

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async)
                    result = future.result()
                    log_complete("Midjourney操作成功", request_id, "RunNode/Midjourney-", "Midjourney")
                    return result
                    
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    results = loop.run_until_complete(self.process_input(taskId, request_id, U1, U2, U3, U4))
                    log_complete("Midjourney操作成功", request_id, "RunNode/Midjourney-", "Midjourney")
                    return results
                finally:
                    loop.close()
                
        except Exception as e:
            error_message = f"Error in run method: {format_runnode_error(str(e))}"
            log_error("操作失败", request_id, error_message, "RunNode/Midjourney-", "Midjourney")
            
            blank_image = Image.new('RGB', (512, 512), color='white')
            blank_tensor = pil2tensor(blank_image)
            return (blank_tensor, "")


    async def process_input(self, taskId, request_id, U1=False, U2=False, U3=False, U4=False):
        try:
            if not any([U1, U2, U3, U4]):
                raise self.MidjourneyError("NO_ACTION_SELECTED")

            action = None
            index = None
            if U1:
                action = "upsample"
                index = 1
            elif U2:
                action = "upsample"
                index = 2
            elif U3:
                action = "upsample"
                index = 3
            elif U4:
                action = "upsample"
                index = 4

            task_result = await self.midjourney_fetch_task_result(taskId, request_id)

            if task_result.get("status") == "FAILURE":
                fail_reason = task_result.get("fail_reason", "Unknown failure reason")
                error_message = f"Original task failed: {format_runnode_error(fail_reason)}"
                log_error("原始任务失败", request_id, error_message, "RunNode/Midjourney-", "Midjourney")
                raise self.MidjourneyError(error_message)

            messageId = None

            if "properties" in task_result and task_result["properties"]:
                properties = task_result["properties"]

                if isinstance(properties, str):
                    try:
                        properties = json.loads(properties)
                    except json.JSONDecodeError as e:
                        # log_error("JSON解析失败", request_id, f"Failed to parse properties JSON: {e}", "RunNode/Midjourney-", "Midjourney")
                        properties = {}

                if isinstance(properties, dict):
                    messageId = properties.get("messageId") or properties.get("message_id")

            if not messageId and "buttons" in task_result and task_result["buttons"]:
                buttons = task_result["buttons"]
                
                try:
                    if isinstance(buttons, str):
                        buttons = json.loads(buttons)

                    if isinstance(buttons, list) and buttons:
                        for button in buttons:
                            if isinstance(button, dict):
                                custom_id = button.get("customId", "")
                                if custom_id:
                                    parts = custom_id.split("::")
                                    if len(parts) >= 5:
                                        messageId = parts[4]
                                        break
                                
                except (json.JSONDecodeError, TypeError, AttributeError) as e:
                    log_error("按钮解析失败", request_id, f"Error processing buttons: {format_runnode_error(str(e))}", "RunNode/Midjourney-", "Midjourney")

            if not messageId:
                possible_fields = ['messageId', 'message_id', 'id', 'task_id']
                for field in possible_fields:
                    if field in task_result and task_result[field]:
                        messageId = task_result[field]
                        # print(f"Found messageId in field '{field}': {messageId}")
                        break

            if not messageId:
                response_str = str(task_result)

                id_patterns = [
                    r'"(?:messageId|message_id|id)"\s*:\s*"([a-zA-Z0-9\-_]{16,})"',
                    r'MJ::JOB::\w+::\d+::([a-zA-Z0-9\-_]{16,})',
                    r'"([a-zA-Z0-9\-_]{20,})"'
                ]
                
                for pattern in id_patterns:
                    match = re.search(pattern, response_str)
                    if match:
                        messageId = match.group(1)
                        # print(f"Found messageId using pattern: {messageId}")
                        break

            if not messageId:
                messageId = taskId
                # print(f"Using taskId as messageId: {messageId}")

            if not messageId:
                error_message = "Could not find messageId in task result"
                log_error("ID未找到", request_id, error_message, "RunNode/Midjourney-", "Midjourney")
                # print(f"Task result structure: {json.dumps(task_result, indent=2)}")
                raise self.MidjourneyError(error_message)

            custom_id = self.generate_custom_id(action, index, messageId)

            response = await self.midjourney_submit_action(action, taskId, index, custom_id, request_id)

            if isinstance(response, str) and response.startswith("Error"):
                raise self.MidjourneyError(response)

            if "result" not in response:
                raise self.MidjourneyError(f"Unexpected response from Midjourney API: {response}")

            new_task_id = response["result"]
            while True:
                await asyncio.sleep(1)
                task_result = await self.midjourney_fetch_task_result(new_task_id, request_id)

                if task_result.get("status") == "FAILURE":
                    fail_reason = task_result.get("fail_reason", "Unknown failure reason")
                    error_message = f"Task failed: {format_runnode_error(fail_reason)}"
                    log_error("任务失败", request_id, error_message, "RunNode/Midjourney-", "Midjourney")
                    raise self.MidjourneyError(error_message)

                if task_result.get("status") == "SUCCESS":
                    break

            if task_result.get("code") == 5 and task_result.get("description") == "task_no_found":
                raise self.MidjourneyError(f"Task not found for taskId: {new_task_id}")

            response = requests.get(task_result["imageUrl"])
            image = Image.open(BytesIO(response.content))
            tensor_image = pil2tensor(image)
            return tensor_image, new_task_id 

        except self.MidjourneyError as e:
            print(f"Midjourney API error: {format_runnode_error(str(e))}")
            raise e
        except Exception as e:
            print(f"An error occurred while processing input: {format_runnode_error(str(e))}")
            raise e
       

    def generate_custom_id(self, action, index, message_id):
        if action == "zoom":
            return f"MJ::CustomZoom::{message_id}"
        elif action in ["upsample_v6_2x_subtle", "upsample_v6_2x_creative", "upsample_v5_2x", "upsample_v5_4x",
                "pan_left", "pan_right", "pan_up", "pan_down", "reroll"]:
            return f"MJ::JOB::{action}::{index}::{message_id}::SOLO"
        elif action.startswith("Outpaint"):
            return f"MJ::{action}::{index}::{message_id}::SOLO"
        elif action == "Inpaint":
            return f"MJ::{action}::1::{index}::{message_id}::SOLO"
        elif action == "CustomZoom":
            return f"MJ::CustomZoom::{index}::{message_id}"
        else:
            return f"MJ::JOB::{action}::{index}::{message_id}"

    def generateUUID(self):
        return str(uuid.uuid4()) 

    async def process_custom_id(self, custom_id):
        if custom_id:
            try:
                action, index, uuid = self.parse_custom_id(custom_id)
                taskId = self.extract_taskId(custom_id)

                
                response = await self.midjourney_submit_action(action, taskId, index, custom_id)
                taskId = response["result"]

                
                task_result = None
                while not task_result or task_result.get("status") != "SUCCESS":
                    await asyncio.sleep(1)
                    task_result = await self.midjourney_fetch_task_result(taskId)
                
                image_url = task_result["imageUrl"]
                return image_url
                
            except Exception as e:
                print(f"Error processing custom_id: {custom_id}. Error: {str(e)}")
                raise ValueError(f"Error processing image from custom ID: {custom_id}. Error: {str(e)}")
        else:
            print("Custom ID is empty, returning empty string")
            return ""

    def parse_custom_id(self, custom_id):
        parts = custom_id.split("::")
        action = parts[2]
        index = int(parts[3])
        uuid = parts[4]
        return action, index, uuid

    def extract_custom_id(self, U, action, index):
        pattern = fr"MJ::JOB::{action}::{index}::(\w+)"
        match = re.search(pattern, U)
        if match:
            return f"MJ::JOB::{action}::{index}::{match.group(1)}"
        else:
            return None

    def extract_taskId(self, U, action, index):
        pattern = fr'"customId": "MJ::JOB::{action}::{index}::(.*?)"'
        match = re.search(pattern, U)
        if match:
            return match.group(1)
        else:
            raise ValueError(f"Invalid custom ID format in U: {U}")

    async def midjourney_submit_action(self, action, taskId, index, custom_id):
        headers = self.get_headers()
        payload = {
            "customId": custom_id,
            "taskId": taskId
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.midjourney_api_url[self.speed]}/mj/submit/action", headers=headers, json=payload, timeout=self.timeout) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()

                            if data.get("status") == "FAILURE":
                                fail_reason = data.get("fail_reason", "Unknown failure reason")
                                error_message = f"Action submission failed: {format_runnode_error(fail_reason)}"
                                print(error_message)
                                raise Exception(error_message)
                                
                            return data
                        except aiohttp.client_exceptions.ContentTypeError:
                            text_response = await response.text()
                            
                            try:
                                import json
                                data = json.loads(text_response)
 
                                if data.get("status") == "FAILURE":
                                    fail_reason = data.get("fail_reason", "Unknown failure reason")
                                    error_message = f"Action submission failed: {format_runnode_error(fail_reason)}"
                                    print(error_message)
                                    raise Exception(error_message)
                                    
                                return data
                            except json.JSONDecodeError:
                                if text_response and len(text_response) < 100:
                                    return {"result": text_response.strip()}
                                raise Exception(f"Invalid response format: {text_response}")
                    else:
                        error_message = f"Error submitting Midjourney action: {response.status}"
                        try:
                            error_details = await response.text()
                            error_message += f" - {format_runnode_error(error_details)}"
                        except:
                            pass
                        raise Exception(error_message)
        except asyncio.TimeoutError:
            error_message = f"Timeout error: Request to submit action timed out after {self.timeout} seconds"
            raise Exception(error_message)
        except Exception as e:
            if "Action submission failed" in str(e):
                raise 
            print(f"Exception in midjourney_submit_action: {format_runnode_error(str(e))}")
            raise e


    async def midjourney_fetch_task_result(self, taskId):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.midjourney_api_url[self.speed]}/mj/task/{taskId}/fetch", headers=headers, timeout=self.timeout) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()

                            return data
                        except aiohttp.client_exceptions.ContentTypeError:
                            text_response = await response.text()
                            try:
                                import json
                                data = json.loads(text_response)
                                return data
                            except json.JSONDecodeError:
                                if text_response and len(text_response) < 100:
                                    return {"status": "SUCCESS", "progress": "100%", "imageUrl": text_response.strip()}
                                raise Exception(f"Server returned invalid response: {text_response}")
                    else:
                        error_message = f"Error fetching Midjourney task result: {response.status}"
                        try:
                            error_details = await response.text()
                            error_message += f" - {format_runnode_error(error_details)}"
                        except:
                            pass
                        raise Exception(error_message)
        except asyncio.TimeoutError:
            error_message = f"Timeout error: Request to fetch task result timed out after {self.timeout} seconds"
            raise Exception(error_message)

                

    async def midjourney_submit_imagine_task(self, prompt, ar, no, c, s, iw, tile, r, video, sw, cw, sv, seed):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key
        }
        payload = {
            "base64Array": [],
            "instanceId": "",
            "modes": [],
            "notifyHook": "",
            "prompt": prompt,
            "remix": True,
            "state": "",
            "ar": ar,
            "no": no,
            "c": c,
            "s": s,
            "iw": iw,
            "tile": tile,
            "r": r,
            "video": video,
            "sw": sw,
            "cw": cw,
            "sv": sv,
            "seed": seed
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.midjourney_api_url[self.speed]}/mj/submit/imagine", headers=headers, json=payload, timeout=self.timeout) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()

                            if data.get("status") == "FAILURE":
                                fail_reason = data.get("fail_reason", "Unknown failure reason")
                                error_message = f"Midjourney task failed: {format_runnode_error(fail_reason)}"
                                print(error_message)
                                raise Exception(error_message)
                                
                            return data["result"]
                        except aiohttp.client_exceptions.ContentTypeError:
                            text_response = await response.text()
                            
                            try:
                                import json
                                data = json.loads(text_response)

                                if data.get("status") == "FAILURE":
                                    fail_reason = data.get("fail_reason", "Unknown failure reason")
                                    error_message = f"Midjourney task failed: {format_runnode_error(fail_reason)}"
                                    print(error_message)
                                    raise Exception(error_message)
                                    
                                return data["result"]
                            except (json.JSONDecodeError, KeyError):
                                if text_response and len(text_response) < 100:
                                    return text_response.strip()
                                raise Exception(f"Server returned invalid response: {text_response}")
                    else:
                        error_message = f"Error submitting Midjourney task: {response.status}"
                        try:
                            error_details = await response.text()
                            error_message += f" - {format_runnode_error(error_details)}"
                        except:
                            pass
                        raise Exception(error_message)
        except asyncio.TimeoutError:
            error_message = f"Timeout error: Request to submit imagine task timed out after {self.timeout} seconds"
            raise Exception(error_message)
        except Exception as e:
            if "Midjourney task failed" in str(e):
                raise  
            print(f"Exception in midjourney_submit_imagine_task: {format_runnode_error(str(e))}")
            raise e


    async def midjourney_fetch_task_result(self, taskId):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.midjourney_api_url[self.speed]}/mj/task/{taskId}/fetch", headers=headers, timeout=self.timeout) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()

                            return data
                        except aiohttp.client_exceptions.ContentTypeError:
                            text_response = await response.text()
                            try:
                                import json
                                data = json.loads(text_response)
                                return data
                            except json.JSONDecodeError:
                                if text_response and len(text_response) < 100:
                                    return {"status": "SUCCESS", "progress": "100%", "imageUrl": text_response.strip()}
                                raise Exception(f"Server returned invalid response: {text_response}")
                    else:
                        error_message = f"Error fetching Midjourney task result: {response.status}"
                        try:
                            error_details = await response.text()
                            error_message += f" - {error_details}"
                        except:
                            pass
                        raise Exception(error_message)
        except asyncio.TimeoutError:
            error_message = f"Timeout error: Request to fetch task result timed out after {self.timeout} seconds"
            raise Exception(error_message)


class Comfly_Mjv(ComflyBaseNode):
    class MidjourneyError(Exception):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "taskId": ("STRING", {"default": "", "forceInput": True}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                # "api_key": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "upsample_v6_2x_subtle": ("BOOLEAN", {"default": False}),
                "upsample_v6_2x_creative": ("BOOLEAN", {"default": False}),
                "costume_zoom": ("BOOLEAN", {"default": False}),
                "zoom": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.1}),
                "pan_left": ("BOOLEAN", {"default": False}),
                "pan_right": ("BOOLEAN", {"default": False}),
                "pan_up": ("BOOLEAN", {"default": False}),
                "pan_down": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",) 
    RETURN_NAMES = ("image",) 
    FUNCTION = "run"
    CATEGORY = "RunNode/Midjourney"

    def run(self, taskId, upsample_v6_2x_subtle=False, upsample_v6_2x_creative=False, costume_zoom=False, zoom=1.0, pan_left=False, pan_right=False, pan_up=False, pan_down=False, api_key=""):
        if api_key.strip():
            self.api_key = api_key
            # config = get_config()
            # config['api_key'] = api_key
            # save_config(config)
        else:
            self.api_key = get_config().get('api_key', '')
        
        try:
            try:
                current_loop = asyncio.get_running_loop()
            
                def run_async():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(self.process_input(taskId, upsample_v6_2x_subtle, upsample_v6_2x_creative, costume_zoom, zoom, pan_left, pan_right, pan_up, pan_down))
                    finally:
                        new_loop.close()

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async)
                    result = future.result()

                    if isinstance(result, tuple) and len(result) > 0:
                        return result
                    else:
                        blank_image = Image.new('RGB', (512, 512), color='white')
                        blank_tensor = pil2tensor(blank_image)
                        return (blank_tensor,)
                    
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(self.process_input(taskId, upsample_v6_2x_subtle, upsample_v6_2x_creative, costume_zoom, zoom, pan_left, pan_right, pan_up, pan_down))

                    if isinstance(result, tuple) and len(result) > 0:
                        return result
                    else:
                        blank_image = Image.new('RGB', (512, 512), color='white')
                        blank_tensor = pil2tensor(blank_image)
                        return (blank_tensor,)
                finally:
                    loop.close()
                
        except Exception as e:
            print(f"Error in run method: {format_runnode_error(str(e))}")
            blank_image = Image.new('RGB', (512, 512), color='white')
            blank_tensor = pil2tensor(blank_image)
            return (blank_tensor,)

    async def process_input(self, taskId, upsample_v6_2x_subtle=False, upsample_v6_2x_creative=False, costume_zoom=False, zoom=1.0, pan_left=False, pan_right=False, pan_up=False, pan_down=False):
        if taskId:
            try:
                task_result = await self.midjourney_fetch_task_result(taskId)
    
                if task_result.get("status") == "FAILURE":
                    fail_reason = task_result.get("fail_reason", "Unknown failure reason")
                    error_message = f"Original task failed: {format_runnode_error(fail_reason)}"
                    print(error_message)
                    raise Exception(error_message)

                messageId = None
                properties = {}
                if "properties" in task_result:
                    if isinstance(task_result["properties"], str):
                        try:
                            properties = json.loads(task_result["properties"])
                        except json.JSONDecodeError as e:
                            error_message = f"Failed to parse properties JSON: {format_runnode_error(str(e))}"
                            print(error_message)
                            raise Exception(error_message)
                    else:
                        properties = task_result["properties"]
                    
                    messageId = properties.get("messageId")

                if not messageId and "buttons" in task_result and task_result["buttons"]:
                    try:
                        buttons = task_result["buttons"]
                        if isinstance(buttons, str):
                            buttons = json.loads(buttons)
                        elif isinstance(buttons, list):
                            pass
                        else:
                            print(f"Unexpected buttons type: {type(buttons)}")
                            buttons = []
                        
                        if buttons and len(buttons) > 0 and isinstance(buttons[0], dict):
                            first_button_id = buttons[0].get("customId", "")
                            parts = first_button_id.split("::")
                            if len(parts) >= 5:
                                messageId = parts[4]
                    except Exception as e:
                        print(f"Failed to extract messageId from buttons: {format_runnode_error(str(e))}")

                if not messageId:
                    error_message = "Could not find messageId in task result"
                    print(error_message)
                    raise Exception(error_message)
                
                prompt = task_result.get("prompt", "")

                customId = None
                if upsample_v6_2x_subtle:
                    customId = f"MJ::JOB::upsample_v6_2x_subtle::2::{messageId}::SOLO"
                elif upsample_v6_2x_creative:
                    customId = f"MJ::JOB::upsample_v6_2x_creative::2::{messageId}::SOLO"
                elif costume_zoom:
                    customId = f"MJ::CustomZoom::{messageId}::2"
                elif pan_left:
                    customId = f"MJ::JOB::pan_left::2::{messageId}::SOLO"
                elif pan_right:
                    customId = f"MJ::JOB::pan_right::2::{messageId}::SOLO"
                elif pan_up:
                    customId = f"MJ::JOB::pan_up::2::{messageId}::SOLO"
                elif pan_down:
                    customId = f"MJ::JOB::pan_down::2::{messageId}::SOLO"
                else:
                    raise Exception("No action selected")
                
                print(f"Generated customId: {customId}")

                if costume_zoom and zoom > 1.0:
                    zoom_param = f" --zoom {zoom}"
                    final_prompt = prompt + zoom_param
 
                    action_response = await self.submit_action(customId, taskId)
                    new_task_id = action_response.get("result", "")
                    
                    if not new_task_id:
                        raise Exception("Failed to get new task ID from action response")

                    modal_response = await self.submit_modal(final_prompt, new_task_id)
                    final_task_id = modal_response.get("result", "")
                    
                    if not final_task_id:
                        raise Exception("Failed to get task ID from modal response")

                    image_url = await self.process_task(final_task_id)
                    
                else:  
                    action_response = await self.submit_action(customId, taskId)
                    new_task_id = action_response.get("result", "")
                    
                    if not new_task_id:
                        raise Exception("Failed to get new task ID from action response")

                    image_url = await self.process_task(new_task_id)

                if image_url:
                    response = requests.get(image_url)
                    image = Image.open(BytesIO(response.content))
                    tensor_image = pil2tensor(image)
                    return (tensor_image,)
                else:
                    blank_image = Image.new('RGB', (512, 512), color='white')
                    blank_tensor = pil2tensor(blank_image)
                    return (blank_tensor,)

            except Exception as e:
                error_message = f"Error processing action: {format_runnode_error(str(e))}"
                print(error_message)
                blank_image = Image.new('RGB', (512, 512), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor,)
        else:
            print("No taskId provided.")
            blank_image = Image.new('RGB', (512, 512), color='white')
            blank_tensor = pil2tensor(blank_image)
            return (blank_tensor,)

    async def process_task(self, taskId):
        while True:
            await asyncio.sleep(1)
            try:
                task_result = await self.midjourney_fetch_task_result(taskId)

                if task_result.get("status") == "FAILURE":
                    fail_reason = task_result.get("fail_reason", "Unknown failure reason")
                    error_message = f"Task failed: {format_runnode_error(fail_reason)}"
                    print(error_message)
                    raise Exception(error_message)

                if task_result.get("status") == "SUCCESS":
                    return task_result.get("imageUrl", task_result.get("image_url", ""))
                    
            except Exception as e:
                if "Task failed:" in str(e):
                    raise  
                error_message = f"Error fetching task result: {format_runnode_error(str(e))}"
                print(error_message)
                raise Exception(error_message)

    async def submit_action(self, customId, taskId):
        headers = self.get_headers()
        payload = {
            "customId": customId,
            "taskId": taskId
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.midjourney_api_url[self.speed]}/mj/submit/action", headers=headers, json=payload) as response: 
                if response.status == 200:
                    try:
                        data = await response.json()
                        return data
                    except aiohttp.client_exceptions.ContentTypeError:
                        text_response = await response.text()
                        print(f"API returned non-JSON response: {text_response}")
                        try:
                            import json
                            data = json.loads(text_response)
                            return data
                        except json.JSONDecodeError:
                            if text_response and len(text_response) < 100:
                                return {"result": text_response.strip()}
                            raise Exception(f"Invalid response format: {text_response}")
                else:
                    error_message = f"Error submitting Midjourney action: {response.status}"
                    print(error_message)
                    try:
                        error_details = await response.text()
                        error_message += f" - {format_runnode_error(error_details)}"
                    except:
                        pass
                    raise Exception(error_message)
    
    async def submit_modal(self, prompt, taskId, maskBase64=None):
        headers = self.get_headers()
        payload = {
            "prompt": prompt,
            "taskId": taskId
        }
        
        if maskBase64:
            payload["maskBase64"] = maskBase64

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.midjourney_api_url[self.speed]}/mj/submit/modal", headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    error_message = f"Error submitting Midjourney modal: {response.status}"
                    print(error_message)
                    raise Exception(error_message)

    async def midjourney_fetch_task_result(self, taskId):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.midjourney_api_url[self.speed]}/mj/task/{taskId}/fetch", headers=headers, timeout=self.timeout) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()
                            return data
                        except aiohttp.client_exceptions.ContentTypeError:
                            text_response = await response.text()
                            try:
                                import json
                                data = json.loads(text_response)
                                return data
                            except json.JSONDecodeError:
                                if text_response and len(text_response) < 100:
                                    return {"status": "SUCCESS", "progress": "100%", "imageUrl": text_response.strip()}
                                raise Exception(f"Server returned invalid response: {text_response}")
                    else:
                        error_message = f"Error fetching Midjourney task result: {response.status}"
                        try:
                            error_details = await response.text()
                            error_message += f" - {error_details}"
                        except:
                            pass
                        raise Exception(error_message)
        except asyncio.TimeoutError:
            error_message = f"Timeout error: Request to fetch task result timed out after {self.timeout} seconds"
            raise Exception(error_message)


class Comfly_Mj_swap_face(ComflyBaseNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "target_image": ("IMAGE",),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                # "api_key": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "image_url")
    FUNCTION = "swap_face"
    CATEGORY = "RunNode/Midjourney"

    def __init__(self):
        super().__init__()
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300
        self.poll_interval = 3 

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key
        }
        
    def image_to_base64(self, image_tensor):
        """Convert tensor to base64 string with data URI prefix"""
        if image_tensor is None:
            return None
            
        pil_image = tensor2pil(image_tensor)[0]
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{base64_str}"

    def fetch_task_result(self, task_id):
        """Fetch task result by task_id"""
        try:
            response = requests.get(
                f"{baseurl}/mj/task/{task_id}/fetch",
                headers=self.get_headers(),
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                return None, f"API Error: {response.status_code} - {response.text}"
                
            result = response.json()
            return result, None
            
        except Exception as e:
            return None, f"Error fetching task result: {format_runnode_error(str(e))}"

    def swap_face(self, source_image, target_image, api_key="", seed=0):
        if api_key.strip():
            self.api_key = api_key
            # config = get_config()
            # config['api_key'] = api_key
            # save_config(config)
        else:
            self.api_key = get_config().get('api_key', '')
            
        if not self.api_key:
            return (source_image, "API key not provided or not found in config")
            
        try:
            source_base64 = self.image_to_base64(source_image)
            target_base64 = self.image_to_base64(target_image)
            
            if not source_base64 or not target_base64:
                return (source_image, "Failed to convert images to base64 format")
            
            payload = {
                "sourceBase64": source_base64,
                "targetBase64": target_base64
            }

            if seed > 0:
                payload["seed"] = seed
            
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(5)

            print("Sending request to face swap API...")
            response = requests.post(
                f"{baseurl}/mj/insight-face/swap",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(10)
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                return (source_image, error_message)
                
            result = response.json()

            if "result" not in result:
                error_message = "No task ID (result field) in API response"
                print(error_message)
                return (source_image, error_message)
                
            task_id = result["result"]
            print(f"Got task ID: {task_id}")

            start_time = time.time()
            last_progress = 0
            
            while time.time() - start_time < self.timeout:
                task_result, error = self.fetch_task_result(task_id)
                
                if error:
                    print(f"Error fetching task result: {error}")
                    time.sleep(self.poll_interval)
                    continue
                
                status = task_result.get("status")
                progress_str = task_result.get("progress", "0%").rstrip("%")
                
                try:
                    progress = int(progress_str)
                except ValueError:
                    progress = last_progress
                
                if progress > last_progress:
                    last_progress = progress
                    pbar.update_absolute(10 + int(progress * 0.8))
                
                if status == "SUCCESS":
                    image_url = None
                    for field in ["imageUrl", "image_url", "url"]:
                        if field in task_result:
                            image_url = task_result.get(field)
                            break
                    
                    if not image_url:
                        error_message = "No image URL in completed task result"
                        print(error_message)
                        return (source_image, error_message)
                    
                    print(f"Found image URL: {image_url}")
                    
                    try:
                        img_response = requests.get(image_url, timeout=self.timeout)
                        img_response.raise_for_status()
                        
                        swapped_image = Image.open(BytesIO(img_response.content))
                        swapped_tensor = pil2tensor(swapped_image)
                        
                        pbar.update_absolute(100)
                        print(f"Face swap completed successfully")
                        return (swapped_tensor, image_url)
                        
                    except Exception as e:
                        error_message = f"Error downloading swapped face image: {format_runnode_error(str(e))}"
                        print(error_message)
                        return (source_image, error_message)
                
                elif status == "FAILURE":
                    fail_reason = task_result.get("fail_reason", "Unknown failure")
                    error_message = f"Task failed: {format_runnode_error(fail_reason)}"
                    print(error_message)
                    return (source_image, error_message)
                
                time.sleep(self.poll_interval)
            
            error_message = f"Task timed out after {self.timeout} seconds"
            print(error_message)
            return (source_image, error_message)
            
        except Exception as e:
            error_message = f"Error in face swapping: {format_runnode_error(str(e))}"
            print(error_message)
            import traceback
            traceback.print_exc()
            return (source_image, error_message)


class Comfly_mjstyle:
    styleAll = {}

    def __init__(self):
        dir = os.path.join(os.path.dirname(__file__), "docs", "mjstyle")
        if not os.path.exists(dir):
            os.mkdir(dir)
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(".json"):
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for d in data:
                            Comfly_mjstyle.styleAll[d['name']] = d

    @classmethod
    def INPUT_TYPES(cls):
        dir = os.path.join(os.path.dirname(__file__), "docs", "mjstyle")
        files_name = []
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(".json"):
                    files_name.append(file.split(".")[0])

        return {
            "required": {
                "styles_type": (files_name,),
            },
            "optional": {
                "positive": ("STRING", {"forceInput": True}),
                "negative": ("STRING", {"forceInput": True}),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("style_positive", "style_negative",)

    FUNCTION = "run"

    OUTPUT_NODE = True

    CATEGORY = "RunNode/Midjourney"

    def replace_repeat(self, prompt):
        prompt = prompt.replace("，", ",")
        arr = prompt.split(",")
        if len(arr) != len(set(arr)):
            all_weight_prompt = re.findall(re.compile(r'[(](.*?)[)]', re.S), prompt)
            if len(all_weight_prompt) > 0:
                return prompt
            else:
                arr = [item.strip() for item in arr]
                arr = list(set(arr))
                return ", ".join(arr)
        else:
            return prompt

    def run(self, extra_pnginfo, unique_id, styles_type, positive="", negative=""):
        values = []
        for node in extra_pnginfo["workflow"]["nodes"]:
            if node["id"] == int(unique_id):
                values = node["properties"]["values"]
                break

        style_positive = ""
        style_negative = ""
        has_prompt = False

        for val in values:
            if "prompt" in Comfly_mjstyle.styleAll[val]:
                if "{positive}" in Comfly_mjstyle.styleAll[val]["prompt"] and not has_prompt:
                    style_positive = Comfly_mjstyle.styleAll[val]["prompt"].format(positive=positive)
                    has_prompt = True
                else:
                    style_positive += ", " + Comfly_mjstyle.styleAll[val]["prompt"].replace(", {positive}", "").replace("{positive}", "")
            if "negative_prompt" in Comfly_mjstyle.styleAll[val]:
                style_negative += ', ' + Comfly_mjstyle.styleAll[val]['negative_prompt'] if negative else Comfly_mjstyle.styleAll[val]['negative_prompt']

        style_positive = self.replace_repeat(style_positive) if style_positive else ""
        style_negative = self.replace_repeat(style_negative) if style_negative else ""

        return (style_positive, style_negative)


class Comfly_mj_video(ComflyBaseNode):
    """
    Comfly_mj_video node
    Generates videos using Midjourney's video API based on text prompts or text+image combination.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "motion": (["Low", "high"], {"default": "Low"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                # "api_key": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "image": ("IMAGE",),
                "notify_hook": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = (IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video1", "video2", "video3", "video4", "task_id", "response")
    FUNCTION = "generate_video"
    CATEGORY = "RunNode/Midjourney"

    def __init__(self):
        super().__init__()
        self.api_key = get_config().get('api_key', '')

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def image_to_base64(self, image_tensor):
        """Convert tensor to base64 string with data URI prefix"""
        if image_tensor is None:
            return None
            
        pil_image = tensor2pil(image_tensor)[0]
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return base64_str  

    def extract_video_urls(self, response_data):
        video_urls = []
 
        if "video_urls" in response_data and response_data["video_urls"]:
            print(f"Found video_urls field: {response_data['video_urls']}")
            try:
                if isinstance(response_data["video_urls"], str):
                    video_urls_json = json.loads(response_data["video_urls"])
                    print(f"Parsed video_urls JSON: {json.dumps(video_urls_json, indent=2)}")
                    
                    if isinstance(video_urls_json, list):
                        urls = [item["url"] for item in video_urls_json if "url" in item]
                        if urls:
                            video_urls = urls
                            print(f"Extracted {len(urls)} URLs from video_urls list")
                        else:
                            print("No 'url' keys found in video_urls items")
                    else:
                        print(f"video_urls_json is not a list: {type(video_urls_json)}")
                elif isinstance(response_data["video_urls"], list):
                    urls = [item["url"] for item in response_data["video_urls"] if "url" in item]
                    if urls:
                        video_urls = urls
                        print(f"Extracted {len(urls)} URLs directly from video_urls list")
                    else:
                        print("No 'url' keys found in video_urls items")
                else:
                    print(f"video_urls is neither string nor list: {type(response_data['video_urls'])}")
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError when parsing video_urls: {format_runnode_error(str(e))}")

                if isinstance(response_data["video_urls"], str) and response_data["video_urls"].startswith("http"):
                    video_urls = [response_data["video_urls"]]
                    print(f"Using video_urls as a direct URL: {response_data['video_urls']}")
            except Exception as e:
                print(f"Unexpected error when processing video_urls: {format_runnode_error(str(e))}")

        if not video_urls:
            if "videoUrl" in response_data and response_data["videoUrl"]:
                video_urls = [response_data["videoUrl"]]
                print(f"Using videoUrl field: {response_data['videoUrl']}")
            elif "video_url" in response_data and response_data["video_url"]:
                video_urls = [response_data["video_url"]]
                print(f"Using video_url field: {response_data['video_url']}")

            if "properties" in response_data and response_data["properties"]:
                try:
                    if isinstance(response_data["properties"], str):
                        props = json.loads(response_data["properties"])
                        if "messageHash" in props:
                            msg_hash = props["messageHash"]
                            print(f"Found messageHash in properties: {msg_hash}")
                except Exception as e:
                    print(f"Error processing properties field: {str(e)}")

        if not video_urls:
            response_str = json.dumps(response_data)
            video_url_pattern = r'https?://\S+\.mp4'
            found_urls = re.findall(video_url_pattern, response_str)
            if found_urls:
                video_urls = found_urls
                print(f"Found {len(found_urls)} video URLs using regex: {found_urls}")
        
        print(f"Final extracted video URLs: {video_urls}")
        return video_urls

    def generate_video(self, prompt, motion="Low", api_key="", image=None, notify_hook="", seed=0):
        request_id = generate_request_id("video_gen", "midjourney")
        log_prepare("视频生成", request_id, "RunNode/Midjourney-", "Midjourney")
        rn_pbar = ProgressBar(request_id, "Midjourney", streaming=True, task_type="视频生成", source="RunNode/Midjourney-")
        if api_key.strip():
            self.api_key = api_key
            # config = get_config()
            # config['api_key'] = api_key
            # save_config(config)
        else:
            self.api_key = get_config().get('api_key', '')
            
        if not self.api_key:
            error_response = json.dumps({
                "status": "error",
                "message": "API key not provided. Please set your API key."
            })
            empty_adapters = [ComflyVideoAdapter("") for _ in range(4)]
            rn_pbar.error("API key not provided. Please set your API key.")
            return (*empty_adapters, "", error_response)
            
        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(10)
        
        try:
            payload = {
                "prompt": prompt,
                "motion": motion,
                "notifyHook": notify_hook
            }

            if seed > 0:
                payload["seed"] = seed
 
            if image is not None:
                pbar.update_absolute(20)
                
                image_base64 = self.image_to_base64(image)
                if image_base64:
                    payload["image"] = image_base64
                else:
                    rn_pbar.error("Failed to convert image to base64")

            try:
                response = requests.post(
                    f"{baseurl}/mj/submit/video", 
                    headers=self.get_headers(),
                    json=payload,
                    timeout=(30, 90)  
                )
                response.raise_for_status()
            except requests.exceptions.Timeout:
                error_message = "API request timed out during submission"
                rn_pbar.error(error_message)
                error_response = json.dumps({"status": "error", "message": error_message})
                empty_adapters = [ComflyVideoAdapter("") for _ in range(4)]
                return (*empty_adapters, "", error_response)
            except requests.exceptions.RequestException as e:
                error_message = f"API request error during submission: {format_runnode_error(str(e))}"
                rn_pbar.error(error_message)
                error_response = json.dumps({"status": "error", "message": error_message})
                empty_adapters = [ComflyVideoAdapter("") for _ in range(4)]
                return (*empty_adapters, "", error_response)
            
            result = response.json()
            
            if "result" not in result:
                error_message = f"No task ID in response: {result}"
                rn_pbar.error(error_message)
                error_response = json.dumps({
                    "status": "error", 
                    "message": error_message
                })
                empty_adapters = [ComflyVideoAdapter("") for _ in range(4)]
                return (*empty_adapters, "", error_response)
                
            task_id = result["result"]
            
            pbar.update_absolute(40)

            task_result = {"status": "PENDING", "progress": "0%"}
            max_retries = 120  
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    status_response = requests.get(
                        f"{baseurl}/mj/task/{task_id}/fetch",
                        headers=self.get_headers(),
                        timeout=(10, 30)  
                    )
                    status_response.raise_for_status()
                    
                    task_result = status_response.json()

                    if task_result.get("status") == "FAILURE":
                        fail_reason = task_result.get("fail_reason", "Unknown failure reason")
                        error_message = f"Video generation failed: {fail_reason}"
                        rn_pbar.error(error_message)
                        error_response = json.dumps({
                            "status": "error", 
                            "message": error_message
                        })
                        raise Exception(error_message)

                    progress = task_result.get("progress", "0%")
                    try:
                        if isinstance(progress, str) and progress.endswith('%'):
                            progress_int = int(progress[:-1])  
                            pbar.update_absolute(progress_int)
                    except (ValueError, TypeError):

                        progress_estimate = min(80, 40 + (retry_count * 40 // max_retries))
                        pbar.update_absolute(progress_estimate)
                    
                    if task_result.get("status") == "SUCCESS":
                        print("Video generation completed. Processing results...")
                        break
                        
                except requests.exceptions.Timeout:
                    rn_pbar.error(f"Timeout while checking task status (retry {retry_count+1}/{max_retries})")
                except requests.exceptions.RequestException as e:
                    rn_pbar.error(f"Error checking task status (retry {retry_count+1}/{max_retries}): {format_runnode_error(str(e))}")
                except Exception as e:
                    rn_pbar.error(f"Unexpected error (retry {retry_count+1}/{max_retries}): {format_runnode_error(str(e))}")
                
                time.sleep(5)  
                retry_count += 1

            if retry_count >= max_retries and task_result.get("status") != "SUCCESS":
                print(f"Warning: Maximum retries ({max_retries}) reached. Proceeding with current result.")
            
            pbar.update_absolute(90)
            video_urls = self.extract_video_urls(task_result)
        
            if not video_urls:
                error_message = "No video URLs found in response"
                rn_pbar.error(error_message)
                error_response = json.dumps({
                    "status": "error",
                    "message": error_message,
                    "task_id": task_id,
                    "response_data": task_result
                })
                raise Exception(error_message)

            video_adapters = []
            for url in video_urls:
                video_adapters.append(ComflyVideoAdapter(url))

            while len(video_adapters) < 4:
                video_adapters.append(ComflyVideoAdapter(""))

            success_response = json.dumps({
                "status": "success",
                "message": f"Generated {len(video_urls)} videos successfully",
                "task_id": task_id,
                "video_urls": video_urls
            })

            pbar.update_absolute(100)
            rn_pbar.done(char_count=len(success_response))
            return (video_adapters[0], video_adapters[1], video_adapters[2], video_adapters[3], 
                   task_id, success_response)
                
        except Exception as e:
            error_message = f"Error in video generation: {format_runnode_error(str(e))}"
            rn_pbar.error(error_message)
            error_response = json.dumps({
                "status": "error",
                "message": error_message
            })

            empty_adapters = [ComflyVideoAdapter("") for _ in range(4)]
            return (*empty_adapters, "", error_response)


class Comfly_mj_video_extend(ComflyBaseNode):
    """
    Comfly_mj_video_extend node
    Extends a Midjourney video based on a task ID.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_id": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "index": ([0, 1, 2, 3], {"default": 0}),   
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                # "api_key": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
            }
        }
    
    RETURN_TYPES = (IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video1", "video2", "video3", "video4", "task_id", "response")
    FUNCTION = "extend_video"
    CATEGORY = "RunNode/Midjourney"

    def __init__(self):
        super().__init__()
        self.api_key = get_config().get('api_key', '')

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def extend_video(self, task_id, index=0, api_key=""):
        request_id = generate_request_id("video_extend", "midjourney")
        log_prepare("视频扩展", request_id, "RunNode/Midjourney-", "Midjourney")
        rn_pbar = ProgressBar(request_id, "Midjourney", streaming=True, task_type="视频扩展", source="RunNode/Midjourney-")
        if api_key.strip():
            self.api_key = api_key
            # config = get_config()
            # config['api_key'] = api_key
            # save_config(config)
        else:
            self.api_key = get_config().get('api_key', '')
            
        if not self.api_key:
            error_message = "API key not provided. Please set your API key."
            rn_pbar.error(error_message)
            empty_adapters = [ComflyVideoAdapter("") for _ in range(4)]
            return (*empty_adapters, "", error_message)
            
        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(10)
        
        try:
            payload = {
                "action": "extend",
                "index": index,
                "taskId": task_id
            }
                
            pbar.update_absolute(30)

            response = requests.post(
                f"{baseurl}/mj/submit/video",
                headers=self.get_headers(),
                json=payload
            )
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                rn_pbar.error(error_message)
                empty_adapters = [ComflyVideoAdapter("") for _ in range(4)]
                return (*empty_adapters, "", error_message)
                
            result = response.json()
            if result.get("code") != 1:
                error_message = format_runnode_error(result)
                rn_pbar.error(error_message)
                empty_adapters = [ComflyVideoAdapter("") for _ in range(4)]
                return (*empty_adapters, "", error_message)
            
            new_task_id = result["result"]
            
            pbar.update_absolute(40)

            video_urls = []
            
            while True:
                try:
                    status_response = requests.get(
                        f"{baseurl}/mj/task/{new_task_id}/fetch",
                        headers=self.get_headers()
                    )
                    
                    if status_response.status_code != 200:
                        time.sleep(2)
                        continue
                        
                    status_result = status_response.json()

                    if status_result.get("status") == "FAILURE":
                        fail_reason = status_result.get("fail_reason", "Unknown failure reason")
                        error_message = f"Video extension failed: {format_runnode_error(fail_reason)}"
                        rn_pbar.error(error_message)
                        empty_adapters = [ComflyVideoAdapter("") for _ in range(4)]
                        return (*empty_adapters, new_task_id, error_message)

                    progress = status_result.get("progress", "0%")
                    try:
                        progress_int = int(progress[:-1])  
                        pbar.update_absolute(progress_int)
                    except (ValueError, TypeError):

                        pass
                    
                    if status_result.get("status") == "SUCCESS":
                        if "video_urls" in status_result:
                            try:
                                if isinstance(status_result["video_urls"], str):
                                    video_urls_data = json.loads(status_result["video_urls"])
                                    if isinstance(video_urls_data, list):
                                        for item in video_urls_data:
                                            if isinstance(item, dict) and "url" in item:
                                                video_urls.append(item["url"])
                                elif isinstance(status_result["video_urls"], list):
                                    # Direct list
                                    for item in status_result["video_urls"]:
                                        if isinstance(item, dict) and "url" in item:
                                            video_urls.append(item["url"])
                            except Exception as e:
                                rn_pbar.error(f"Error parsing video_urls: {format_runnode_error(str(e))}")

                        if not video_urls:
                            if "videoUrl" in status_result and status_result["videoUrl"]:
                                video_urls.append(status_result["videoUrl"])
                            elif "video_url" in status_result and status_result["video_url"]:
                                video_urls.append(status_result["video_url"])

                        if video_urls:
                            break

                        if "properties" in status_result:
                            try:
                                props = status_result["properties"]
                                if isinstance(props, str):
                                    props_data = json.loads(props)
                                    if "videoUrl" in props_data:
                                        video_urls.append(props_data["videoUrl"])
                                elif isinstance(props, dict) and "videoUrl" in props:
                                    video_urls.append(props["videoUrl"])
                            except Exception as e:
                                rn_pbar.error(f"Error parsing properties: {format_runnode_error(str(e))}")

                        if video_urls:
                            break

                        response_str = json.dumps(status_result)
                        video_url_pattern = r'https?://\S+\.mp4'
                        found_urls = re.findall(video_url_pattern, response_str)
                        if found_urls:
                            video_urls.extend(found_urls)
                            break

                        if status_result.get("status") == "SUCCESS":
                            rn_pbar.error("Task completed successfully but no video URLs found in response")
                            break
                            
                    time.sleep(2) 
                    
                except Exception as e:
                    rn_pbar.error(f"Error checking task status: {format_runnode_error(str(e))}")
                    time.sleep(2)
            
            if not video_urls:
                error_message = "Failed to retrieve video URLs from the response"
                rn_pbar.error(error_message)
                empty_adapters = [ComflyVideoAdapter("") for _ in range(4)]
                return (*empty_adapters, new_task_id, error_message)

            video_adapters = []
            for url in video_urls:
                video_adapters.append(ComflyVideoAdapter(url))

            while len(video_adapters) < 4:
                video_adapters.append(ComflyVideoAdapter(""))
                
            pbar.update_absolute(100)
            
            success_response = json.dumps({
                "status": "success",
                "message": f"Extended {len(video_urls)} videos successfully",
                "task_id": new_task_id,
                "video_urls": video_urls
            })
            
            rn_pbar.done(char_count=len(success_response))
            return (video_adapters[0], video_adapters[1], video_adapters[2], video_adapters[3], 
                   new_task_id, success_response)
            
        except Exception as e:
            error_message = f"Error in video extension: {format_runnode_error(str(e))}"
            rn_pbar.error(error_message)
            empty_adapters = [ComflyVideoAdapter("") for _ in range(4)]
            return (*empty_adapters, "", error_message)
