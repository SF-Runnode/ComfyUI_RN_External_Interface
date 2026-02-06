from ..comfly_config import *
from .__init__ import *


class Comfly_Flux_Kontext:
    _last_image_url = ""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "input_image": ("IMAGE",),
                "model": (["flux-kontext-dev", "flux-kontext-pro", "flux-kontext-max"], {"default": "flux-kontext-pro"}),
                "apikey": ("STRING", {"default": ""}),
                # "apikey": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "aspect_ratio": (["Default", "21:9", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16", "9:21"], 
                         {"default": "Default"}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 20.0, "step": 0.5}),
                "num_of_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "clear_image": ("BOOLEAN", {"default": True})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "image_url")
    FUNCTION = "generate_image"
    CATEGORY = "RunNode/Flux"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def upload_image(self, image_tensor):
        """Upload image to the file endpoint and return the URL"""
        try:
            pil_image = tensor2pil(image_tensor)[0]
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            file_content = buffered.getvalue()
            
            files = {'file': ('image.png', file_content, 'image/png')}
            
            response = requests.post(
                f"{baseurl}/v1/files",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files=files,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            if 'url' in result:
                return result['url']
            else:
                raise Exception(f"Unexpected response from file upload API: {result}")
                
        except Exception as e:
            error_msg = f"Error uploading image: {format_runnode_error(str(e))}"
            print(error_msg)
            raise Exception(error_msg)
    
    def generate_image(self, prompt, input_image=None, model="flux-kontext-pro", 
                  apikey="", aspect_ratio="Default", guidance=3.5, num_of_images=1,
                  seed=-1, clear_image=True):
        request_id = generate_request_id("image_gen", "flux")
        log_prepare("图像生成", request_id, "RunNode/Flux-", "Flux", model_name=model)
        rn_pbar = ProgressBar(request_id, "Flux", streaming=True, task_type="图像生成", source="RunNode/Flux-")
        if apikey.strip():
            self.api_key = apikey
            # config = get_config()
            # config['api_key'] = apikey
            # save_config(config)
        else:
            self.api_key = get_config().get('api_key', '')
            
        if not self.api_key:
            error_message = "API key not found in Comflyapi.json"
            rn_pbar.error(error_message)
            log_error("配置缺失", request_id, error_message, "RunNode/Flux-", "Flux")
            raise Exception(error_message)
        
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            final_prompt = prompt
            custom_dimensions = None

            if input_image is not None:
                batch_size = input_image.shape[0]
                all_image_urls = []
                
                for i in range(batch_size):
                    single_image = input_image[i:i+1]
                    pbar.update_absolute(10 + (i * 10) // batch_size)
                    image_url = self.upload_image(single_image)
                    if image_url:
                        all_image_urls.append(image_url)

                if all_image_urls:
                    image_urls_text = " ".join(all_image_urls)
                    final_prompt = f"{image_urls_text} {prompt}"
                    if aspect_ratio == "match_input_image" and batch_size > 0:
                        pil_image = tensor2pil(input_image)[0]
                        width, height = pil_image.size
                        custom_dimensions = {"width": width, "height": height}
                else:
                    error_message = "Failed to upload any images"
                    rn_pbar.error(error_message)
                    raise Exception(error_message)
 
            elif not clear_image and Comfly_Flux_Kontext._last_image_url:
                final_prompt = f"{Comfly_Flux_Kontext._last_image_url} {prompt}"

            payload = {
                "prompt": final_prompt,
                "model": model,
                "n": num_of_images,  
                "guidance_scale": guidance  
            }

            if custom_dimensions and aspect_ratio == "match_input_image":
                payload.update(custom_dimensions)
            else:
                payload["aspect_ratio"] = aspect_ratio

            if seed != -1:
                payload["seed"] = seed

            response = requests.post(
                f"{baseurl}/v1/images/generations",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(50)
            
            if response.status_code != 200:
                error_message = format_runnode_error(response)
                rn_pbar.error(error_message)
                log_error("API请求失败", request_id, error_message, "RunNode/Flux-", "Flux")
                raise Exception(error_message)
                
            result = response.json()

            if not result.get("data") or not result["data"]:
                error_message = "No image data in response"
                rn_pbar.error(error_message)
                log_error("响应异常", request_id, error_message, "RunNode/Flux-", "Flux")
                raise Exception(error_message)

            generated_tensors = []
            image_urls = []
            
            for i, item in enumerate(result["data"]):
                pbar.update_absolute(50 + (i+1) * 40 // len(result["data"]))
                
                if "url" in item:
                    image_url = item["url"]
                    image_urls.append(image_url)
                    
                    try:
                        img_response = requests.get(image_url, timeout=self.timeout)
                        img_response.raise_for_status()
                        generated_image = Image.open(BytesIO(img_response.content))
                        generated_tensor = pil2tensor(generated_image)
                        generated_tensors.append(generated_tensor)
                    except Exception as e:
                        error_msg = f"Error downloading image from URL: {format_runnode_error(str(e))}"
                        rn_pbar.error(error_msg)
                        log_error("下载失败", request_id, error_msg, "RunNode/Flux-", "Flux")
                        raise Exception(error_msg)
                        
                elif "b64_json" in item:
                    image_data = base64.b64decode(item["b64_json"])
                    generated_image = Image.open(BytesIO(image_data))
                    generated_tensor = pil2tensor(generated_image)
                    generated_tensors.append(generated_tensor)
            
            pbar.update_absolute(100)
            
            if generated_tensors:
                combined_tensor = torch.cat(generated_tensors, dim=0)

                if image_urls:
                    Comfly_Flux_Kontext._last_image_url = image_urls[0]
                
                safe_urls = [safe_public_url(url) for url in image_urls]
                log_complete("图像生成", request_id, "RunNode/Flux-", "Flux", 
                            image_urls=safe_urls,
                            count=len(generated_tensors))

                rn_pbar.done(char_count=len("\n".join(image_urls)))
                return (combined_tensor, "\n".join(image_urls))
            else:
                error_message = "Failed to process any images"
                rn_pbar.error(error_message)
                raise Exception(error_message)
            
        except Exception as e:
            error_message = f"Error in image generation: {format_runnode_error(str(e))}"
            rn_pbar.error(error_message)
            log_error("运行异常", request_id, error_message, "RunNode/Flux-", "Flux")
            raise Exception(error_message)
         

class Comfly_Flux_Kontext_Edit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "image": ("IMAGE",),
                "model": (["flux-kontext-dev", "flux-kontext-pro", "flux-kontext-max"], {"default": "flux-kontext-pro"}),
                "apikey": ("STRING", {"default": ""}),
                # "apikey": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "aspect_ratio": (["21:9", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16", "9:21"], 
                         {"default": "1:1"}),
                "num_of_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "image_url")
    FUNCTION = "generate_image"
    CATEGORY = "RunNode/Flux"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def generate_image(self, prompt, image=None, model="flux-kontext-pro", 
                  apikey="", aspect_ratio="1:1", num_of_images=1,
                  seed=-1):
        request_id = generate_request_id("image_edit", "flux")
        log_prepare("图像生成", request_id, "RunNode/Flux-", "Flux", model_name=model)
        rn_pbar = ProgressBar(request_id, "Flux", streaming=True, task_type="图像生成", source="RunNode/Flux-")
        if apikey.strip():
            self.api_key = apikey
            # config = get_config()
            # config['api_key'] = apikey
            # save_config(config)
        else:
            self.api_key = get_config().get('api_key', '')
            
        if not self.api_key:
            error_message = "API key not found in Comflyapi.json"
            rn_pbar.error(error_message)
            log_error("配置缺失", request_id, error_message, "RunNode/Flux-", "Flux")
            raise Exception(error_message)
        
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            if image is not None:
                pil_image = tensor2pil(image)[0]

                buffered = BytesIO()
                pil_image.save(buffered, format="PNG")
                buffered.seek(0)

                files = {
                    'image': ('image.png', buffered, 'image/png')
                }
                
                data = {
                    'prompt': prompt,
                    'model': model
                }

                if aspect_ratio != "Default":
                    data["aspect_ratio"] = aspect_ratio

                if seed != -1:
                    data["seed"] = str(seed)
                    
                if num_of_images > 1:
                    data["n"] = str(num_of_images)

                pbar.update_absolute(30)
                response = requests.post(
                    f"{baseurl}/v1/images/edits",
                    headers=self.get_headers(),
                    data=data,
                    files=files,
                    timeout=self.timeout
                )
            else:
                payload = {
                    "prompt": prompt,
                    "model": model,
                    "n": num_of_images
                }
                
                if aspect_ratio != "Default":
                    payload["aspect_ratio"] = aspect_ratio
                
                if seed != -1:
                    payload["seed"] = seed
                    
                headers = self.get_headers()
                headers["Content-Type"] = "application/json"
                
                response = requests.post(
                    f"{baseurl}/v1/images/generations",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
            
            pbar.update_absolute(50)
            
            if response.status_code != 200:
                error_message = format_runnode_error(response)
                rn_pbar.error(error_message)
                log_error("API请求失败", request_id, error_message, "RunNode/Flux-", "Flux")
                raise Exception(error_message)
                
            result = response.json()

            if not result.get("data") or not result["data"]:
                error_message = "No image data in response"
                rn_pbar.error(error_message)
                log_error("响应异常", request_id, error_message, "RunNode/Flux-", "Flux")
                raise Exception(error_message)

            generated_tensors = []
            image_urls = []
            
            for i, item in enumerate(result["data"]):
                pbar.update_absolute(50 + (i+1) * 40 // len(result["data"]))
                
                if "url" in item:
                    image_url = item["url"]
                    image_urls.append(image_url)
                    
                    try:
                        img_response = requests.get(image_url, timeout=self.timeout)
                        img_response.raise_for_status()
                        generated_image = Image.open(BytesIO(img_response.content))
                        generated_tensor = pil2tensor(generated_image)
                        generated_tensors.append(generated_tensor)
                    except Exception as e:
                        error_msg = f"Error downloading image from URL: {format_runnode_error(str(e))}"
                        rn_pbar.error(error_msg)
                        log_error("下载失败", request_id, error_msg, "RunNode/Flux-", "Flux")
                        raise Exception(error_msg)
                        
                elif "b64_json" in item:
                    image_data = base64.b64decode(item["b64_json"])
                    generated_image = Image.open(BytesIO(image_data))
                    generated_tensor = pil2tensor(generated_image)
                    generated_tensors.append(generated_tensor)
            
            pbar.update_absolute(100)
            
            if generated_tensors:
                combined_tensor = torch.cat(generated_tensors, dim=0)
                
                safe_urls = [safe_public_url(url) for url in image_urls]
                log_complete("图像生成", request_id, "RunNode/Flux-", "Flux", 
                            image_urls=safe_urls,
                            count=len(generated_tensors))

                rn_pbar.done(char_count=len("\n".join(image_urls)))
                return (combined_tensor, "\n".join(image_urls))
            else:
                error_message = "Failed to process any images"
                rn_pbar.error(error_message)
                raise Exception(error_message)
            
        except Exception as e:
            error_message = f"Error in image generation: {format_runnode_error(str(e))}"
            rn_pbar.error(error_message)
            log_error("运行异常", request_id, error_message, "RunNode/Flux-", "Flux")
            raise Exception(error_message)


class Comfly_Flux_Kontext_bfl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["flux-kontext-pro", "flux-kontext-max"], {"default": "flux-kontext-pro"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                # "api_key": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "input_image": ("IMAGE",),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "aspect_ratio": (["21:9", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16", "9:21"], 
                         {"default": "1:1"}),
                "output_format": (["png", "jpeg"], {"default": "png"}),
                "prompt_upsampling": ("BOOLEAN", {"default": False}),
                "safety_tolerance": ("INT", {"default": 2, "min": 0, "max": 6})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "task_id", "response")
    FUNCTION = "generate_image"
    CATEGORY = "RunNode/Flux"
    OUTPUT_NODE = True

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

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
    
    def generate_image(self, prompt, model="flux-kontext-pro", input_image=None, 
                      seed=-1, aspect_ratio="1:1", output_format="png", 
                      prompt_upsampling=False, safety_tolerance=2, api_key=""):
        request_id = generate_request_id("image_gen", "flux")
        log_prepare("图像生成", request_id, "RunNode/Flux-", "Flux", model_name=model)
        rn_pbar = ProgressBar(request_id, "Flux", streaming=True, task_type="图像生成", source="RunNode/Flux-")
        if api_key.strip():
            self.api_key = api_key
            # config = get_config()
            # config['api_key'] = api_key
            # save_config(config)
        else:
            self.api_key = get_config().get('api_key', '')

        if not self.api_key:
            error_message = "API key not found"
            rn_pbar.error(error_message)
            log_error("配置缺失", request_id, error_message, "RunNode/Flux-", "Flux")
            raise Exception(error_message)
            
        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(10)

        api_endpoint = f"{baseurl}/bfl/v1/{model}"
        
        try:
            payload = {
                "prompt": prompt,
                "output_format": output_format,
                "prompt_upsampling": prompt_upsampling,
                "safety_tolerance": safety_tolerance
            }

            if input_image is not None:
                input_image_base64 = self.image_to_base64(input_image)
                if input_image_base64:
                    payload["input_image"] = input_image_base64
            
            if seed != -1:
                payload["seed"] = seed
                
            if aspect_ratio:
                payload["aspect_ratio"] = aspect_ratio

            response = requests.post(
                api_endpoint,
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(30)
            
            if response.status_code != 200:
                error_message = format_runnode_error(response)
                rn_pbar.error(error_message)
                log_error("API请求失败", request_id, error_message, "RunNode/Flux-", "Flux")
                raise Exception(error_message)
                
            result = response.json()
            
            if "id" not in result or "polling_url" not in result:
                error_message = "Invalid response format from API"
                rn_pbar.error(error_message)
                log_error("响应异常", request_id, error_message, "RunNode/Flux-", "Flux")
                raise Exception(error_message)
                
            task_id = result["id"]
            polling_url = result["polling_url"]

            pbar.update_absolute(40)

            max_attempts = 300  # 300 * 2s = 600s
            attempts = 0
            image_url = ""
            start_time = time.time()
            last_log_time = start_time
            
            while attempts < max_attempts:
                time.sleep(2)
                attempts += 1
                current_time = time.time()
                if current_time - last_log_time >= 15:
                    log_backend("flux_poll_heartbeat", request_id=request_id, attempts=attempts, elapsed=int(current_time - start_time))
                    last_log_time = current_time
                
                try:
                    result_response = requests.get(
                        f"{baseurl}/bfl/v1/get_result?id={task_id}",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )
                    
                    if result_response.status_code != 200:
                        continue
                        
                    result_data = result_response.json()
                    status = result_data.get("status")
                    
                    if status == "Ready":
                        if "result" in result_data and "sample" in result_data["result"]:
                            image_url = result_data["result"]["sample"]
                            break

                    progress = min(80, 40 + (attempts * 40 // max_attempts))
                    pbar.update_absolute(progress)
                        
                except Exception as e:
                    error_msg = f"Error checking generation status: {format_runnode_error(str(e))}"
                    rn_pbar.error(error_msg)
                    log_error("轮询异常", request_id, error_msg, "RunNode/Flux-", "Flux")
            
            if not image_url:
                error_message = "Failed to retrieve generated image URL after multiple attempts"
                rn_pbar.error(error_message)
                log_error("超时失败", request_id, error_message, "RunNode/Flux-", "Flux")
                raise Exception(error_message)

            pbar.update_absolute(90)
            
            try:
                img_response = requests.get(image_url, timeout=self.timeout)
                img_response.raise_for_status()
                
                generated_image = Image.open(BytesIO(img_response.content))
                generated_tensor = pil2tensor(generated_image)
                
                pbar.update_absolute(100)
                
                result_info = {
                    "status": "success",
                    "task_id": task_id,
                    "prompt": prompt,
                    "model": model,
                    "seed": seed if seed > 0 else "random",
                    "aspect_ratio": aspect_ratio
                }
                
                safe_url = safe_public_url(image_url)
                log_complete("图像生成", request_id, "RunNode/Flux-", "Flux", 
                            image_url=safe_url,
                            task_id=task_id)
                rn_pbar.done(char_count=len(json.dumps(result_info)))
                return (generated_tensor, image_url, json.dumps(result_info))
                
            except Exception as e:
                error_message = f"Error downloading generated image: {format_runnode_error(str(e))}"
                rn_pbar.error(error_message)
                log_error("下载失败", request_id, error_message, "RunNode/Flux-", "Flux")
                raise Exception(error_message)
            
        except Exception as e:
            error_message = f"Error in image generation: {format_runnode_error(str(e))}"
            rn_pbar.error(error_message)
            log_error("运行异常", request_id, error_message, "RunNode/Flux-", "Flux")
            raise Exception(error_message)


class Comfly_Flux_2_Max:
    """
    Comfly Flux 2 Max node
    Generates images using the Flux 2 Max API with support for multiple input images.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "input_image": ("IMAGE",),
                "input_image_2": ("IMAGE",),
                "input_image_3": ("IMAGE",),
                "input_image_4": ("IMAGE",),
                "input_image_5": ("IMAGE",),
                "input_image_6": ("IMAGE",),
                "input_image_7": ("IMAGE",),
                "input_image_8": ("IMAGE",),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 6000, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 6000, "step": 8}),
                "safety_tolerance": ("INT", {"default": 2, "min": 0, "max": 5}),
                "output_format": (["jpeg", "png"], {"default": "jpeg"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "response")
    FUNCTION = "generate_image"
    CATEGORY = "RunNode/Flux"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 600

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
    
    def generate_image(self, prompt, api_key="", input_image=None, input_image_2=None,
                      input_image_3=None, input_image_4=None, input_image_5=None,
                      input_image_6=None, input_image_7=None, input_image_8=None,
                      seed=-1, width=1024, height=1024, safety_tolerance=2, 
                      output_format="jpeg"):
        
        request_id = generate_request_id("image_gen", "flux-2-max")
        log_prepare("图像生成", request_id, "RunNode/Flux-", "Flux", model_name="flux-2-max")
        rn_pbar = ProgressBar(request_id, "Flux", extra_info="模型:flux-2-max", streaming=True, task_type="图像生成", source="RunNode/Flux-")
        _rn_start = time.perf_counter()
        
        if api_key.strip():
            self.api_key = api_key
        else:
            self.api_key = get_config().get('api_key', '')
            
        if not self.api_key:
            error_message = "API key not found"
            rn_pbar.error(error_message)
            log_error("配置缺失", request_id, error_message, "RunNode/Flux-", "Flux")
            raise Exception(error_message)
            
        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(10)
        rn_pbar.update(10)
        
        try:
            payload = {
                "prompt": prompt,
                "safety_tolerance": safety_tolerance,
                "output_format": output_format
            }

            if width > 0:
                payload["width"] = width
            if height > 0:
                payload["height"] = height

            if seed != -1:
                payload["seed"] = seed

            image_inputs = [
                ("input_image", input_image),
                ("input_image_2", input_image_2),
                ("input_image_3", input_image_3),
                ("input_image_4", input_image_4),
                ("input_image_5", input_image_5),
                ("input_image_6", input_image_6),
                ("input_image_7", input_image_7),
                ("input_image_8", input_image_8),
            ]
            
            for field_name, img in image_inputs:
                if img is not None:
                    img_base64 = self.image_to_base64(img)
                    if img_base64:
                        payload[field_name] = img_base64
            
            pbar.update_absolute(20)
            rn_pbar.update(20)

            response = requests.post(
                f"{baseurl}/bfl/v1/flux-2-max",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(30)
            rn_pbar.update(30)
            
            if response.status_code != 200:
                error_message = format_runnode_error(response)
                rn_pbar.error(error_message)
                log_error("API请求失败", request_id, error_message, "RunNode/Flux-", "Flux")
                raise Exception(error_message)
                
            result = response.json()
            
            if "id" not in result:
                error_message = "No task ID in response"
                rn_pbar.error(error_message)
                log_error("响应异常", request_id, error_message, "RunNode/Flux-", "Flux")
                raise Exception(error_message)
                
            task_id = result["id"]
            polling_url = result.get("polling_url", "")
            
            pbar.update_absolute(40)
            rn_pbar.update(40)

            max_attempts = 120
            attempts = 0
            image_url = ""
            final_result_data = None
            
            while attempts < max_attempts:
                time.sleep(5)
                attempts += 1
                
                try:
                    result_response = requests.get(
                        f"{baseurl}/bfl/v1/get_result?id={task_id}",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )
                    
                    if result_response.status_code != 200:
                        continue
                        
                    result_data = result_response.json()
                    status = result_data.get("status", "")

                    progress = min(90, 40 + (attempts * 50 // max_attempts))
                    pbar.update_absolute(progress)
                    rn_pbar.update(progress)
                    
                    if status == "Ready":
                        if "result" in result_data and "sample" in result_data["result"]:
                            image_url = result_data["result"]["sample"]
                            final_result_data = result_data
                            break
                    elif status in ["Failed", "Error"]:
                        error_message = f"Task failed: {result_data.get('details', 'Unknown error')}"
                        rn_pbar.error(error_message)
                        log_error("任务失败", request_id, error_message, "RunNode/Flux-", "Flux")
                        raise Exception(error_message)
                        
                except Exception as e:
                    print(f"Error checking generation status: {str(e)}")
            
            if not image_url:
                error_message = "Failed to retrieve generated image URL after multiple attempts"
                rn_pbar.error(error_message)
                log_error("超时失败", request_id, error_message, "RunNode/Flux-", "Flux")
                raise Exception(error_message)
            
            pbar.update_absolute(90)
            rn_pbar.update(90)

            try:
                img_response = requests.get(image_url, timeout=self.timeout)
                img_response.raise_for_status()
                
                generated_image = Image.open(BytesIO(img_response.content))
                generated_tensor = pil2tensor(generated_image)
                
                pbar.update_absolute(100)
                rn_pbar.update(100)
                
                safe_url = safe_public_url(image_url)
                log_complete("图像生成", request_id, "RunNode/Flux-", "Flux",
                            image_url=safe_url,
                            task_id=task_id)

                result_info = {
                    "status": "success",
                    "id": task_id,
                    "result": final_result_data.get("result", {}) if final_result_data else {},
                    "prompt": prompt,
                    "seed": final_result_data.get("result", {}).get("seed", seed) if final_result_data else seed,
                    "width": width,
                    "height": height,
                    "image_url": image_url,
                    "duration": final_result_data.get("result", {}).get("duration") if final_result_data else None,
                    "start_time": final_result_data.get("result", {}).get("start_time") if final_result_data else None,
                    "end_time": final_result_data.get("result", {}).get("end_time") if final_result_data else None,
                }
                
                rn_pbar.done(char_count=len(json.dumps(result_info)))
                return (generated_tensor, image_url, json.dumps(result_info, indent=2))
                
            except Exception as e:
                error_message = f"Error downloading generated image: {format_runnode_error(str(e))}"
                rn_pbar.error(error_message)
                log_error("下载失败", request_id, error_message, "RunNode/Flux-", "Flux")
                raise Exception(error_message)
            
        except Exception as e:
            error_message = f"Error in image generation: {format_runnode_error(str(e))}"
            rn_pbar.error(error_message)
            log_error("运行异常", request_id, error_message, "RunNode/Flux-", "Flux")
            raise Exception(error_message)



class Comfly_Flux_2_Pro:
    """
    Comfly Flux 2 Pro node
    Generates images using the Flux 2 Pro API with support for multiple input images.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "tooltip": "描述图像内容的提示词"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "tooltip": "BFL API 密钥，留空则使用全局配置"}),
                # "api_key": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "input_image": ("IMAGE", {"tooltip": "输入图像 1"}),
                "input_image_2": ("IMAGE", {"tooltip": "输入图像 2"}),
                "input_image_3": ("IMAGE", {"tooltip": "输入图像 3"}),
                "input_image_4": ("IMAGE", {"tooltip": "输入图像 4"}),
                "input_image_5": ("IMAGE", {"tooltip": "输入图像 5"}),
                "input_image_6": ("IMAGE", {"tooltip": "输入图像 6"}),
                "input_image_7": ("IMAGE", {"tooltip": "输入图像 7"}),
                "input_image_8": ("IMAGE", {"tooltip": "输入图像 8"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647, "tooltip": "随机种子，-1 表示随机"}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 6000, "step": 8, "tooltip": "图像宽度"}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 6000, "step": 8, "tooltip": "图像高度"}),
                "safety_tolerance": ("INT", {"default": 2, "min": 0, "max": 5, "tooltip": "安全容忍度 (0-5)"}),
                "output_format": (["jpeg", "png"], {"default": "png", "tooltip": "输出图片格式"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "task_id", "response")
    FUNCTION = "generate_image"
    CATEGORY = "RunNode/Flux"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 600

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
    
    def generate_image(self, prompt, api_key="", input_image=None, input_image_2=None,
                      input_image_3=None, input_image_4=None, input_image_5=None,
                      input_image_6=None, input_image_7=None, input_image_8=None,
                      seed=-1, width=1024, height=1024, safety_tolerance=2, 
                      output_format="png"):
        request_id = generate_request_id("image_gen", "flux")
        log_prepare("图像生成", request_id, "RunNode/Flux-", "Flux")
        rn_pbar = ProgressBar(request_id, "Flux", streaming=True, task_type="图像生成", source="RunNode/Flux-")
        
        if api_key.strip():
            self.api_key = api_key
            # config = get_config()
            # config['api_key'] = api_key
            # save_config(config)
        else:
            self.api_key = get_config().get('api_key', '')

        if not self.api_key:
            error_message = "API key not found"
            rn_pbar.error(error_message)
            log_error("配置缺失", request_id, error_message, "RunNode/Flux-", "Flux")
            raise Exception(error_message)
            
        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(10)
        
        try:
            payload = {
                "prompt": prompt,
                "safety_tolerance": safety_tolerance,
                "output_format": output_format
            }

            if width > 0:
                payload["width"] = width
            if height > 0:
                payload["height"] = height

            if seed != -1:
                payload["seed"] = seed

            image_inputs = [
                ("input_image", input_image),
                ("input_image_2", input_image_2),
                ("input_image_3", input_image_3),
                ("input_image_4", input_image_4),
                ("input_image_5", input_image_5),
                ("input_image_6", input_image_6),
                ("input_image_7", input_image_7),
                ("input_image_8", input_image_8),
            ]
            
            for field_name, img in image_inputs:
                if img is not None:
                    img_base64 = self.image_to_base64(img)
                    if img_base64:
                        payload[field_name] = img_base64
            
            pbar.update_absolute(20)

            response = requests.post(
                f"{baseurl}/bfl/v1/flux-2-pro",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(30)
            
            if response.status_code != 200:
                error_message = format_runnode_error(response)
                rn_pbar.error(error_message)
                log_error("API请求失败", request_id, error_message, "RunNode/Flux-", "Flux")
                raise Exception(error_message)
                
            result = response.json()
            
            if "id" not in result:
                error_message = "No task ID in response"
                rn_pbar.error(error_message)
                log_error("响应异常", request_id, error_message, "RunNode/Flux-", "Flux")
                raise Exception(error_message)
                
            task_id = result["id"]
            polling_url = result.get("polling_url", "")
            
            pbar.update_absolute(40)

            max_attempts = 300  # 300 * 2s = 600s
            attempts = 0
            image_url = ""
            start_time = time.time()
            last_log_time = start_time
            
            while attempts < max_attempts:
                time.sleep(2)
                attempts += 1
                current_time = time.time()
                if current_time - last_log_time >= 15:
                    log_backend("flux_poll_heartbeat", request_id=request_id, attempts=attempts, elapsed=int(current_time - start_time))
                    last_log_time = current_time
                
                try:
                    result_response = requests.get(
                        f"{baseurl}/bfl/v1/get_result?id={task_id}",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )
                    
                    if result_response.status_code != 200:
                        continue
                        
                    result_data = result_response.json()
                    status = result_data.get("status", "")

                    progress = min(90, 40 + (attempts * 50 // max_attempts))
                    pbar.update_absolute(progress)
                    
                    if status == "Ready":
                        if "result" in result_data and "sample" in result_data["result"]:
                            image_url = result_data["result"]["sample"]
                            break
                    elif status in ["Failed", "Error"]:
                        error_message = f"Task failed: {result_data.get('details', 'Unknown error')}"
                        rn_pbar.error(error_message)
                        log_error("任务失败", request_id, error_message, "RunNode/Flux-", "Flux")
                        raise Exception(error_message)
                        
                except Exception as e:
                    error_msg = f"Error checking generation status: {format_runnode_error(str(e))}"
                    rn_pbar.error(error_msg)
                    log_error("轮询异常", request_id, error_msg, "RunNode/Flux-", "Flux")
            
            if not image_url:
                error_message = "Failed to retrieve generated image URL after multiple attempts"
                rn_pbar.error(error_message)
                log_error("超时失败", request_id, error_message, "RunNode/Flux-", "Flux")
                raise Exception(error_message)
            
            pbar.update_absolute(90)

            try:
                img_response = requests.get(image_url, timeout=self.timeout)
                img_response.raise_for_status()
                
                generated_image = Image.open(BytesIO(img_response.content))
                generated_tensor = pil2tensor(generated_image)
                
                pbar.update_absolute(100)
                
                result_info = {
                    "status": "success",
                    "task_id": task_id,
                    "prompt": prompt,
                    "seed": result_data.get("result", {}).get("seed", seed),
                    "width": width,
                    "height": height,
                    "image_url": image_url
                }
                
                safe_url = safe_public_url(image_url)
                log_complete("图像生成", request_id, "RunNode/Flux-", "Flux", 
                            image_url=safe_url,
                            task_id=task_id)
                rn_pbar.done(char_count=len(json.dumps(result_info)))
                return (generated_tensor, image_url, json.dumps(result_info))
                
            except Exception as e:
                error_message = f"Error downloading generated image: {format_runnode_error(str(e))}"
                rn_pbar.error(error_message)
                log_error("下载失败", request_id, error_message, "RunNode/Flux-", "Flux")
                raise Exception(error_message)
            
        except Exception as e:
            error_message = f"Error in image generation: {format_runnode_error(str(e))}"
            rn_pbar.error(error_message)
            log_error("运行异常", request_id, error_message, "RunNode/Flux-", "Flux")
            raise Exception(error_message)


class Comfly_Flux_2_Flex:
    """
    Comfly Flux 2 Flex node
    Generates images using the Flux 2 Flex API with support for multiple input images,
    prompt upsampling, guidance, and steps parameters.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                # "api_key": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "input_image": ("IMAGE",),
                "input_image_2": ("IMAGE",),
                "input_image_3": ("IMAGE",),
                "input_image_4": ("IMAGE",),
                "input_image_5": ("IMAGE",),
                "input_image_6": ("IMAGE",),
                "input_image_7": ("IMAGE",),
                "input_image_8": ("IMAGE",),
                "prompt_upsampling": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 6000, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 6000, "step": 8}),
                "guidance": ("FLOAT", {"default": 5.0, "min": 1.5, "max": 10.0, "step": 0.1}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 50}),
                "safety_tolerance": ("INT", {"default": 2, "min": 0, "max": 5}),
                "output_format": (["jpeg", "png"], {"default": "png"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "task_id", "response")
    FUNCTION = "generate_image"
    CATEGORY = "RunNode/Flux"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 600

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
    
    def generate_image(self, prompt, api_key="", input_image=None, input_image_2=None,
                      input_image_3=None, input_image_4=None, input_image_5=None,
                      input_image_6=None, input_image_7=None, input_image_8=None,
                      prompt_upsampling=True, seed=-1, width=1024, height=1024,
                      guidance=5.0, steps=50, safety_tolerance=2, output_format="png"):
        
        request_id = generate_request_id("image_gen", "flux")
        log_prepare("图像生成", request_id, "RunNode/Flux-", "Flux")
        rn_pbar = ProgressBar(request_id, "Flux", streaming=True, task_type="图像生成", source="RunNode/Flux-")
        
        if api_key.strip():
            self.api_key = api_key
            # config = get_config()
            # config['api_key'] = api_key
            # save_config(config)
        else:
            self.api_key = get_config().get('api_key', '')

        if not self.api_key:
            error_message = "API key not found"
            rn_pbar.error(error_message)
            log_error("配置缺失", request_id, error_message, "RunNode/Flux-", "Flux")
            raise Exception(error_message)
            
        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(10)
        
        try:
            payload = {
                "prompt": prompt,
                "prompt_upsampling": prompt_upsampling,
                "guidance": guidance,
                "steps": steps,
                "safety_tolerance": safety_tolerance,
                "output_format": output_format
            }

            if width > 0:
                payload["width"] = width
            if height > 0:
                payload["height"] = height

            if seed != -1:
                payload["seed"] = seed

            image_inputs = [
                ("input_image", input_image),
                ("input_image_2", input_image_2),
                ("input_image_3", input_image_3),
                ("input_image_4", input_image_4),
                ("input_image_5", input_image_5),
                ("input_image_6", input_image_6),
                ("input_image_7", input_image_7),
                ("input_image_8", input_image_8),
            ]
            
            for field_name, img in image_inputs:
                if img is not None:
                    img_base64 = self.image_to_base64(img)
                    if img_base64:
                        payload[field_name] = img_base64
            
            pbar.update_absolute(20)

            response = requests.post(
                f"{baseurl}/bfl/v1/flux-2-flex",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(30)
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                rn_pbar.error(error_message)
                log_error("API请求失败", request_id, error_message, "RunNode/Flux-", "Flux")
                raise Exception(error_message)
                
            result = response.json()
            
            if "id" not in result:
                error_message = "No task ID in response"
                rn_pbar.error(error_message)
                log_error("响应异常", request_id, error_message, "RunNode/Flux-", "Flux")
                raise Exception(error_message)
                
            task_id = result["id"]
            polling_url = result.get("polling_url", "")
            
            pbar.update_absolute(40)

            max_attempts = 300  # 300 * 2s = 600s
            attempts = 0
            image_url = ""
            start_time = time.time()
            last_log_time = start_time
            
            while attempts < max_attempts:
                time.sleep(2)
                attempts += 1
                current_time = time.time()
                if current_time - last_log_time >= 15:
                    log_backend("flux_poll_heartbeat", request_id=request_id, attempts=attempts, elapsed=int(current_time - start_time))
                    last_log_time = current_time
                
                try:
                    result_response = requests.get(
                        f"{baseurl}/bfl/v1/get_result?id={task_id}",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )
                    
                    if result_response.status_code != 200:
                        continue
                        
                    result_data = result_response.json()
                    status = result_data.get("status", "")

                    progress = min(90, 40 + (attempts * 50 // max_attempts))
                    pbar.update_absolute(progress)
                    
                    if status == "Ready":
                        if "result" in result_data and "sample" in result_data["result"]:
                            image_url = result_data["result"]["sample"]
                            break
                    elif status in ["Failed", "Error"]:
                        error_message = f"Task failed: {result_data.get('details', 'Unknown error')}"
                        rn_pbar.error(error_message)
                        log_error("任务失败", request_id, error_message, "RunNode/Flux-", "Flux")
                        raise Exception(error_message)
                        
                except Exception as e:
                    error_msg = f"Error checking generation status: {format_runnode_error(str(e))}"
                    rn_pbar.error(error_msg)
                    log_error("轮询异常", request_id, error_msg, "RunNode/Flux-", "Flux")
            
            if not image_url:
                error_message = "Failed to retrieve generated image URL after multiple attempts"
                rn_pbar.error(error_message)
                log_error("超时失败", request_id, error_message, "RunNode/Flux-", "Flux")
                raise Exception(error_message)
            
            pbar.update_absolute(90)

            try:
                img_response = requests.get(image_url, timeout=self.timeout)
                img_response.raise_for_status()
                
                generated_image = Image.open(BytesIO(img_response.content))
                generated_tensor = pil2tensor(generated_image)
                
                pbar.update_absolute(100)
                
                result_info = {
                    "status": "success",
                    "task_id": task_id,
                    "prompt": prompt,
                    "seed": result_data.get("result", {}).get("seed", seed),
                    "width": width,
                    "height": height,
                    "guidance": guidance,
                    "steps": steps,
                    "prompt_upsampling": prompt_upsampling,
                    "image_url": image_url
                }
                
                safe_url = safe_public_url(image_url)
                log_complete("图像生成", request_id, "RunNode/Flux-", "Flux", 
                            image_url=safe_url,
                            task_id=task_id)
                rn_pbar.done(char_count=len(json.dumps(result_info)))
                return (generated_tensor, image_url, json.dumps(result_info))
                
            except Exception as e:
                error_message = f"Error downloading generated image: {format_runnode_error(str(e))}"
                rn_pbar.error(error_message)
                log_error("下载失败", request_id, error_message, "RunNode/Flux-", "Flux")
                raise Exception(error_message)
            
        except Exception as e:
            error_message = f"Error in image generation: {format_runnode_error(str(e))}"
            rn_pbar.error(error_message)
            log_error("运行异常", request_id, error_message, "RunNode/Flux-", "Flux")
            raise Exception(error_message)
