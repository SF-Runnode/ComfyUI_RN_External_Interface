from ..comfly_config import *
from .__init__ import *
from ..utils import *
from ..utils import _parse_asset_bundle_only, _comfly_split_asset_ids, _comfly_asset_id_to_url, _doubao_seedance_video_input_to_bytes, _doubao_seedance_io_file_to_bytes, _comfy_waveform_to_wav_bytes


class Comfly_Doubao_Seedream:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["doubao-seedream-3-0-t2i-250415"], {"default": "doubao-seedream-3-0-t2i-250415"}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "size": (["1024x1024", "864x1152", "1152x864", "1280x720", "720x1280", "832x1248", 
                         "1248x832", "1512x648", "Custom"], {"default": "1024x1024"}),
                "Custom_size": ("STRING", {"default": "1536x1024", "multiline": False}),
                "guidance_scale": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "apikey": ("STRING", {"default": ""}),
                # "apikey": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "watermark": ("BOOLEAN", {"default": True})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "response")
    FUNCTION = "generate_image"
    CATEGORY = "RunNode/Doubao"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def validate_custom_size(self, size_str):
        """Validate a custom size string to ensure it's in the correct format and within allowed range."""
        try:
            if 'x' not in size_str:
                return False, "Custom size must be in format 'widthxheight'"
            
            width, height = map(int, size_str.split('x'))

            if width < 512 or width > 2048 or height < 512 or height > 2048:
                return False, f"Custom size dimensions must be between 512 and 2048 pixels. Got {width}x{height}."
            
            return True, f"{width}x{height}"
        except ValueError:
            return False, "Custom size must contain valid integers in format 'widthxheight'"
    
    def generate_image(self, prompt, model, response_format="url", size="1024x1024",
                       Custom_size="1536x1024", guidance_scale=2.5, apikey="",
                       seed=-1, watermark=True):
        request_id = generate_request_id("img_gen", "doubao")
        log_prepare("图像生成", request_id, "RunNode/Doubao-", "Doubao", model_name=model)
        rn_pbar = ProgressBar(request_id, "Doubao", streaming=True, task_type="图像生成", source="RunNode/Doubao-")
        rn_pbar.set_generating()
        if apikey.strip():
            self.api_key = apikey
        else:
            self.api_key = get_config().get('api_key', '')
            
        if not self.api_key:
            error_message = "API key not found in configuration file or environment variables."
            rn_pbar.error(error_message)
            log_error(error_message, request_id, "RunNode/Doubao-", "Doubao")
            raise Exception(error_message)
            
        try:
            rn_pbar.update(10, "Initializing...")

            final_size = size
            if size == "Custom":
                is_valid, result = self.validate_custom_size(Custom_size)
                if not is_valid:
                    error_message = result
                    rn_pbar.error(error_message)
                    log_error(error_message, request_id, "RunNode/Doubao-", "Doubao")
                    raise Exception(error_message)
                final_size = result
            
            payload = {
                "model": model,
                "prompt": prompt,
                "response_format": response_format,
                "size": final_size,
                "guidance_scale": guidance_scale,
                "watermark": watermark
            }
            
            if seed != -1:
                payload["seed"] = seed
            
            rn_pbar.update(30, "Submitting task...")
            response = requests.post(
                f"{baseurl}/v1/images/generations",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                error_message = format_runnode_error(response)
                rn_pbar.error(error_message)
                log_error(error_message, request_id, "RunNode/Doubao-", "Doubao")
                raise Exception(error_message)
                
            result = response.json()
            
            rn_pbar.update(50, "Processing response...")
            
            if "data" not in result or not result["data"]:
                error_message = "No image data in response"
                rn_pbar.error(error_message)
                log_error(error_message, request_id, "RunNode/Doubao-", "Doubao")
                raise Exception(error_message)
            
            image_url = None
            image_data = None

            if response_format == "url":
                image_url = result["data"][0].get("url")
                if not image_url:
                    error_message = "No image URL in response"
                    rn_pbar.error(error_message)
                    log_error(error_message, request_id, "RunNode/Doubao-", "Doubao")
                    raise Exception(error_message)
                    
                try:
                    img_response = requests.get(image_url, timeout=self.timeout)
                    img_response.raise_for_status()
                    image_data = BytesIO(img_response.content)
                except Exception as e:
                    error_message = f"Error downloading image: {format_runnode_error(str(e))}"
                    rn_pbar.error(error_message)
                    log_error(error_message, request_id, "RunNode/Doubao-", "Doubao")
                    raise Exception(error_message)
            else:
                b64_data = result["data"][0].get("b64_json")
                if not b64_data:
                    error_message = "No base64 data in response"
                    rn_pbar.error(error_message)
                    log_error(error_message, request_id, "RunNode/Doubao-", "Doubao")
                    raise Exception(error_message)
                    
                image_data = BytesIO(base64.b64decode(b64_data))
            
            rn_pbar.update(80, "Converting image...")

            try:
                pil_image = Image.open(image_data)
                tensor_image = pil2tensor(pil_image)

                response_info = {
                    "prompt": prompt,
                    "model": model,
                    "size": final_size,
                    "guidance_scale": guidance_scale,
                    "seed": seed if seed != -1 else "auto",
                    "url": image_url if image_url else "base64 data"
                }
                
                rn_pbar.update(100, "Done")
                rn_pbar.done(char_count=len(json.dumps(response_info, ensure_ascii=False)))
                
                safe_url = safe_public_url(image_url) if image_url else ""
                log_complete("Doubao任务完成", request_id, "RunNode/Doubao-", "Doubao", 
                             image_count=1, first_url=safe_url)
                             
                return (tensor_image, json.dumps(response_info, indent=2))
                
            except Exception as e:
                error_message = f"Error processing image: {format_runnode_error(str(e))}"
                rn_pbar.error(error_message)
                log_error(error_message, request_id, "RunNode/Doubao-", "Doubao")
                raise Exception(error_message)
                
        except Exception as e:
            error_message = f"Error generating image: {format_runnode_error(str(e))}"
            rn_pbar.error(error_message)
            log_error(error_message, request_id, "RunNode/Doubao-", "Doubao")
            raise Exception(error_message)


class Comfly_Doubao_Seedream_4:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["doubao-seedream-4-0-250828"], {"default": "doubao-seedream-4-0-250828"}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "resolution": (["1K", "2K", "4K"], {"default": "1K"}),
            },
            "optional": {
                "aspect_ratio": (["1:1", "4:3", "3:4", "16:9", "9:16", "2:3", "3:2", "21:9", "9:21", "Custom"], {"default": "1:1"}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 1}),
                "apikey": ("STRING", {"default": ""}),
                # "apikey": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "sequential_image_generation": (["disabled", "auto"], {"default": "disabled"}),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 15, "step": 1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "watermark": ("BOOLEAN", {"default": True}),
                "stream": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "response", "image_url")
    FUNCTION = "generate_image"
    CATEGORY = "RunNode/Doubao"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 900
        self.size_mapping = {
            
            "1K": {
                "1:1": "1024x1024",
                "4:3": "1152x864",
                "3:4": "864x1152",
                "16:9": "1280x720",
                "9:16": "720x1280",
                "2:3": "832x1248",
                "3:2": "1248x832",
                "21:9": "1512x648",
                "9:21": "648x1512"
            },

            "2K": {
                "1:1": "2048x2048",
                "4:3": "2048x1536",
                "3:4": "1536x2048",
                "16:9": "2048x1152",
                "9:16": "1152x2048",
                "2:3": "1536x2048",
                "3:2": "2048x1536",
                "21:9": "2048x864",
                "9:21": "864x2048"
            },

            "4K": {
                "1:1": "4096x4096",
                "4:3": "4096x3072",
                "3:4": "3072x4096",
                "16:9": "4096x2304",
                "9:16": "2304x4096",
                "2:3": "3072x4096",
                "3:2": "4096x3072",
                "21:9": "4096x1728",
                "9:21": "1728x4096"
            }
        }

        self.resolution_factors = {
            "1K": 1,
            "2K": 2,
            "4K": 4
        }

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
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"
    
    def generate_image(self, prompt, model, response_format="url", resolution="1K",
                  aspect_ratio="1:1", width=1024, height=1024, apikey="",
                  image1=None, image2=None, image3=None, image4=None, image5=None,
                  sequential_image_generation="disabled", max_images=1, seed=-1,
                  watermark=True, stream=False):
        request_id = generate_request_id("img_gen", "doubao")
        log_prepare("图像生成", request_id, "RunNode/Doubao-", "Doubao", model_name=model)
        rn_pbar = ProgressBar(request_id, "Doubao", streaming=True, task_type="图像生成", source="RunNode/Doubao-")
        rn_pbar.set_generating()
        if apikey.strip():
            self.api_key = apikey
            # config = get_config()
            # config['api_key'] = apikey
            # save_config(config)
        else:
            self.api_key = get_config().get('api_key', '')
            
        if not self.api_key:
            error_message = "API key not found in configuration file or environment variables."
            rn_pbar.error(error_message)
            log_error("配置缺失", request_id, error_message, "RunNode/Doubao-", "Doubao")
            raise Exception(error_message)
            
        try:
            rn_pbar.update(10, "Initializing...")

            if aspect_ratio == "Custom":

                scale_factor = self.resolution_factors.get(resolution, 1)
                scaled_width = int(width * scale_factor)
                scaled_height = int(height * scale_factor)
    
                final_size = f"{scaled_width}x{scaled_height}"
                pass
            else:
                if resolution in self.size_mapping and aspect_ratio in self.size_mapping[resolution]:
                    final_size = self.size_mapping[resolution][aspect_ratio]
                else:
                    final_size = "1024x1024"
                    rn_pbar.error("不支持的分辨率与宽高比组合，已回退为 1024x1024")
            
            payload = {
                "model": model,
                "prompt": prompt,
                "response_format": response_format,
                "size": final_size,
                "watermark": watermark,
                "stream": stream
            }

            if sequential_image_generation == "auto":
                payload["sequential_image_generation"] = sequential_image_generation
                payload["n"] = max_images
                
            if seed != -1:
                payload["seed"] = seed

            image_urls = []
            for img in [image1, image2, image3, image4, image5]:
                if img is not None:
                    batch_size = img.shape[0]
                    for i in range(batch_size):
                        single_image = img[i:i+1]
                        image_base64 = self.image_to_base64(single_image)
                        if image_base64:
                            image_urls.append(image_base64)
            
            if image_urls:
                payload["image"] = image_urls
            
            rn_pbar.update(30, "Submitting task to API...")
            
            response = requests.post(
                f"{baseurl}/v1/images/generations",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                error_message = format_runnode_error(response)
                rn_pbar.error(error_message)
                log_error("API提交失败", request_id, error_message, "RunNode/Doubao-", "Doubao")
                raise Exception(error_message)
                
            result = response.json()
            
            rn_pbar.update(50, "Processing response...")
            
            if "data" not in result or not result["data"]:
                error_message = "No image data in response"
                rn_pbar.error(error_message)
                log_error("响应异常", request_id, error_message, "RunNode/Doubao-", "Doubao")
                raise Exception(error_message)
            
            image_url = None
            image_data = None
            generated_images = []
            image_urls = []
            for item in result["data"]:
                if response_format == "url":
                    image_url = item.get("url")
                    if not image_url:
                        continue
                    
                    image_urls.append(image_url)
                    
                    try:
                        img_response = requests.get(image_url, timeout=self.timeout)
                        img_response.raise_for_status()
                        image_data = BytesIO(img_response.content)
                        
                        pil_image = Image.open(image_data)
                        tensor_image = pil2tensor(pil_image)
                        generated_images.append(tensor_image)
                    except Exception as e:
                        err_msg = f"下载图片失败: {format_runnode_error(str(e))}"
                        rn_pbar.error(err_msg)
                        log_error(err_msg, request_id, "RunNode/Doubao-", "Doubao")
                        raise Exception(err_msg)
                else:
                    b64_data = item.get("b64_json")
                    if not b64_data:
                        continue
                        
                    image_data = BytesIO(base64.b64decode(b64_data))
                    
                    pil_image = Image.open(image_data)
                    tensor_image = pil2tensor(pil_image)
                    generated_images.append(tensor_image)
            
            rn_pbar.update(80, "Converting images...")
            if not generated_images:
                error_message = "Failed to process any images"
                rn_pbar.error(error_message)
                log_error("生成异常", request_id, error_message, "RunNode/Doubao-", "Doubao")
                raise Exception(error_message)
            
            combined_tensor = torch.cat(generated_images, dim=0)
                
            response_info = {
                "prompt": prompt,
                "model": model,
                "resolution": resolution,
                "size": final_size,
                "seed": seed if seed != -1 else "auto",
                "urls": image_urls if image_urls else [],
                "sequential_image_generation": sequential_image_generation,
                "aspect_ratio": aspect_ratio
            }

            if aspect_ratio == "Custom":
                response_info["original_dimensions"] = f"{width}x{height}"
                response_info["scaled_dimensions"] = final_size
            
            if sequential_image_generation == "auto":
                response_info["max_images"] = max_images
            
            response_info["images_generated"] = len(generated_images)
            
            rn_pbar.update(100, "Done")
            rn_pbar.done(char_count=len(json.dumps(response_info, ensure_ascii=False)))
            first_image_url = image_urls[0] if image_urls else ""
            log_complete("Doubao任务完成", request_id, "RunNode/Doubao-", "Doubao", 
                       image_count=len(generated_images), first_url=safe_public_url(first_image_url))
            return (combined_tensor, json.dumps(response_info, indent=2), first_image_url)
                
        except Exception as e:
            error_message = f"Error generating image: {format_runnode_error(str(e))}"
            rn_pbar.error(error_message)
            log_error("生成失败", request_id, error_message, "RunNode/Doubao-", "Doubao")
            raise Exception(error_message)


class Comfly_Doubao_Seedream_4_5:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["doubao-seedream-4-5-251128"], {"default": "doubao-seedream-4-5-251128"}),
                "response_format": (["url", "b64_json"], {"default": "b64_json"}),
                "resolution": (["2K", "4K"], {"default": "2K"}),
            },
            "optional": {
                "aspect_ratio": (["1:1", "4:3", "3:4", "16:9", "9:16", "2:3", "3:2", "21:9", "9:21", "Custom"], {"default": "16:9"}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 1}),
                "apikey": ("STRING", {"default": ""}),
                # "apikey": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "sequential_image_generation": (["disabled", "auto"], {"default": "disabled"}),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 15, "step": 1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "watermark": ("BOOLEAN", {"default": False}),
                "stream": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "response", "image_url")
    FUNCTION = "generate_image"
    CATEGORY = "RunNode/Doubao"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 900
        self.size_mapping = {
            
            "2K": {
                "1:1": "2048x2048",
                "4:3": "2304x1728",
                "3:4": "1728x2304",
                "16:9": "2560x1440",
                "9:16": "1440x2560",
                "2:3": "1664x2496",
                "3:2": "2496x1664",
                "21:9": "3024x1296",
                "9:21": "1296x3024"
            },

            "4K": {
                "1:1": "4096x4096",
                "4:3": "4096x3072",
                "3:4": "3072x4096",
                "16:9": "4096x2304",
                "9:16": "2304x4096",
                "2:3": "3072x4096",
                "3:2": "4096x3072",
                "21:9": "4096x1728",
                "9:21": "1728x4096"
            }
        }

        self.resolution_factors = {
            "2K": 2,
            "4K": 4
        }

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
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"
    
    def generate_image(self, prompt, model, response_format="url", resolution="2K", 
                  aspect_ratio="1:1", width=1024, height=1024, apikey="", 
                  image1=None, image2=None, image3=None, image4=None, image5=None, 
                  sequential_image_generation="disabled", max_images=1, seed=-1, 
                  watermark=True, stream=False):
        request_id = generate_request_id("img_gen", "doubao")
        log_prepare("图像生成", request_id, "RunNode/Doubao-", "Doubao", model_name=model)
        rn_pbar = ProgressBar(request_id, "Doubao", streaming=True, task_type="图像生成", source="RunNode/Doubao-")
        rn_pbar.set_generating()
        if apikey.strip():
            self.api_key = apikey
            # config = get_config()
            # config['api_key'] = apikey
            # save_config(config)
        else:
            self.api_key = get_config().get('api_key', '')
            
        if not self.api_key:
            error_message = "API key not found in configuration file or environment variables."
            rn_pbar.error(error_message)
            log_error("配置缺失", request_id, error_message, "RunNode/Doubao-", "Doubao")
            raise Exception(error_message)
            
        try:
            rn_pbar.update(10, "Initializing...")

            if aspect_ratio == "Custom":

                scale_factor = self.resolution_factors.get(resolution, 1)
                scaled_width = int(width * scale_factor)
                scaled_height = int(height * scale_factor)
    
                final_size = f"{scaled_width}x{scaled_height}"
            else:
                if resolution in self.size_mapping and aspect_ratio in self.size_mapping[resolution]:
                    final_size = self.size_mapping[resolution][aspect_ratio]
                else:
                    final_size = "2048x2048"
            
            payload = {
                "model": model,
                "prompt": prompt,
                "response_format": response_format,
                "size": final_size,
                "watermark": watermark,
                "stream": stream
            }

            if sequential_image_generation == "auto":
                payload["sequential_image_generation"] = sequential_image_generation
                payload["n"] = max_images
                
            if seed != -1:
                payload["seed"] = seed

            image_urls = []
            for img in [image1, image2, image3, image4, image5]:
                if img is not None:
                    batch_size = img.shape[0]
                    for i in range(batch_size):
                        single_image = img[i:i+1]
                        image_base64 = self.image_to_base64(single_image)
                        if image_base64:
                            image_urls.append(image_base64)
            
            if image_urls:
                payload["image"] = image_urls
            
            response = requests.post(
                f"{baseurl}/v1/images/generations",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            rn_pbar.update(30, "Submitting task to API...")
            
            if response.status_code != 200:
                error_message = format_runnode_error(response)
                rn_pbar.error(error_message)
                log_error("API请求失败", request_id, error_message, "RunNode/Doubao-", "Doubao")
                raise Exception(error_message)
                
            result = response.json()
            
            rn_pbar.update(50, "Processing response...")
            
            if "data" not in result or not result["data"]:
                error_message = "No image data in response"
                rn_pbar.error(error_message)
                log_error("数据缺失", request_id, error_message, "RunNode/Doubao-", "Doubao")
                raise Exception(error_message)
            
            image_url = None
            image_data = None
            generated_images = []
            image_urls = []
            for item in result["data"]:
                if response_format == "url":
                    image_url = item.get("url")
                    if not image_url:
                        continue
                    
                    image_urls.append(image_url)
                    
                    try:
                        img_response = requests.get(image_url, timeout=self.timeout)
                        img_response.raise_for_status()
                        image_data = BytesIO(img_response.content)
                        
                        pil_image = Image.open(image_data)
                        tensor_image = pil2tensor(pil_image)
                        generated_images.append(tensor_image)
                    except Exception as e:
                        error_msg = f"Error downloading image: {format_runnode_error(str(e))}"
                        rn_pbar.error(error_msg)
                        log_error("下载失败", request_id, error_msg, "RunNode/Doubao-", "Doubao")
                        raise Exception(error_msg)
                else:
                    b64_data = item.get("b64_json")
                    if not b64_data:
                        continue
                        
                    image_data = BytesIO(base64.b64decode(b64_data))
                    
                    pil_image = Image.open(image_data)
                    tensor_image = pil2tensor(pil_image)
                    generated_images.append(tensor_image)
            
            rn_pbar.update(80, "Converting images...")
            if not generated_images:
                error_message = "Failed to process any images"
                rn_pbar.error(error_message)
                log_error("处理失败", request_id, error_message, "RunNode/Doubao-", "Doubao")
                raise Exception(error_message)
            
            combined_tensor = torch.cat(generated_images, dim=0)
                
            response_info = {
                "prompt": prompt,
                "model": model,
                "resolution": resolution,
                "size": final_size,
                "seed": seed if seed != -1 else "auto",
                "urls": image_urls if image_urls else [],
                "sequential_image_generation": sequential_image_generation,
                "aspect_ratio": aspect_ratio
            }

            if aspect_ratio == "Custom":
                response_info["original_dimensions"] = f"{width}x{height}"
                response_info["scaled_dimensions"] = final_size
            
            if sequential_image_generation == "auto":
                response_info["max_images"] = max_images
            
            response_info["images_generated"] = len(generated_images)
            
            rn_pbar.update(100, "Done")
            first_image_url = image_urls[0] if image_urls else ""
            safe_url = safe_public_url(first_image_url)
            log_complete("Doubao任务完成", request_id, "RunNode/Doubao-", "Doubao", 
                         image_count=len(generated_images), first_url=safe_url)
            rn_pbar.done(char_count=len(json.dumps(response_info, ensure_ascii=False)))
            
            return (combined_tensor, json.dumps(response_info, indent=2), first_image_url)
                
        except Exception as e:
            error_message = f"Error generating image: {format_runnode_error(str(e))}"
            rn_pbar.error(error_message)
            log_error("系统异常", request_id, error_message, "RunNode/Doubao-", "Doubao")
            raise Exception(error_message)


class Comfly_Doubao_Seededit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "model": (["doubao-seededit-3-0-i2i-250628"], {"default": "doubao-seededit-3-0-i2i-250628"}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "size": ("STRING", {"default": "adaptive"}),
                "guidance_scale": ("FLOAT", {"default": 5.5, "min": 1.0, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "apikey": ("STRING", {"default": ""}),
                # "apikey": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "watermark": ("BOOLEAN", {"default": True})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "response")
    FUNCTION = "edit_image"
    CATEGORY = "RunNode/Doubao"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def edit_image(self, image, prompt, model, response_format="url", size="adaptive", 
                guidance_scale=5.5, apikey="", seed=-1, watermark=True):
        request_id = generate_request_id("img_edit", "doubao")
        log_prepare("图像编辑", request_id, "RunNode/Doubao-", "SeedEdit", model_name=model)
        rn_pbar = ProgressBar(request_id, "SeedEdit", streaming=True, task_type="图像编辑", source="RunNode/Doubao-")
        rn_pbar.set_generating()

        if apikey.strip():
            self.api_key = apikey
            # config = get_config()
            # config['api_key'] = apikey
            # save_config(config)
        else:
            self.api_key = get_config().get('api_key', '')
            
        if not self.api_key:
            error_message = "API key not found in configuration file or environment variables."
            rn_pbar.error(error_message)
            log_error("配置缺失", request_id, error_message, "RunNode/Doubao-", "SeedEdit")
            raise Exception(error_message)
            
        try:
            rn_pbar.update(10, "Initializing...")

            pil_image = tensor2pil(image)[0]

            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
            
            payload = {
                "model": model,
                "prompt": prompt,
                "image": img_base64,
                "response_format": response_format,
                "size": size,
                "guidance_scale": guidance_scale,
                "watermark": watermark
            }
            
            if seed != -1:
                payload["seed"] = seed
            
            response = requests.post(
                f"{baseurl}/v1/images/generations",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            rn_pbar.update(30, "Submitting task to API...")
            
            if response.status_code != 200:
                error_message = format_runnode_error(response)
                rn_pbar.error(error_message)
                log_error("API请求失败", request_id, error_message, "RunNode/Doubao-", "SeedEdit")
                raise Exception(error_message)
                
            result = response.json()
            
            rn_pbar.update(50, "Processing response...")
            
            if "data" not in result or not result["data"]:
                error_message = "No image data in response"
                rn_pbar.error(error_message)
                log_error("数据缺失", request_id, error_message, "RunNode/Doubao-", "SeedEdit")
                raise Exception(error_message)
            
            image_url = None
            image_data = None

            if response_format == "url":
                image_url = result["data"][0].get("url")
                if not image_url:
                    error_message = "No image URL in response"
                    rn_pbar.error(error_message)
                    log_error("数据缺失", request_id, error_message, "RunNode/Doubao-", "SeedEdit")
                    raise Exception(error_message)
                    
                try:
                    img_response = requests.get(image_url, timeout=self.timeout)
                    img_response.raise_for_status()
                    image_data = BytesIO(img_response.content)
                except Exception as e:
                    error_message = f"Error downloading image: {format_runnode_error(str(e))}"
                    rn_pbar.error(error_message)
                    log_error("下载失败", request_id, error_message, "RunNode/Doubao-", "SeedEdit")
                    raise Exception(error_message)
            else:
                b64_data = result["data"][0].get("b64_json")
                if not b64_data:
                    error_message = "No base64 data in response"
                    rn_pbar.error(error_message)
                    log_error("数据缺失", request_id, error_message, "RunNode/Doubao-", "SeedEdit")
                    raise Exception(error_message)
                    
                image_data = BytesIO(base64.b64decode(b64_data))
            
            rn_pbar.update(80, "Converting image...")

            try:
                edited_pil_image = Image.open(image_data)
                edited_tensor = pil2tensor(edited_pil_image)

                response_info = {
                    "prompt": prompt,
                    "model": model,
                    "size": size,
                    "guidance_scale": guidance_scale,
                    "seed": seed if seed != -1 else "auto",
                    "url": image_url if image_url else "base64 data"
                }
                
                rn_pbar.update(100, "Done")
                rn_pbar.done(char_count=len(json.dumps(response_info, ensure_ascii=False)))
                
                safe_url = safe_public_url(image_url) if image_url else ""
                log_complete("Doubao图像编辑完成", request_id, "RunNode/Doubao-", "SeedEdit", 
                             image_count=1, first_url=safe_url)
                             
                return (edited_tensor, json.dumps(response_info, indent=2))
                
            except Exception as e:
                error_message = f"Error processing edited image: {format_runnode_error(str(e))}"
                rn_pbar.error(error_message)
                log_error("处理失败", request_id, error_message, "RunNode/Doubao-", "SeedEdit")
                raise Exception(error_message)
                
        except Exception as e:
            error_message = f"Error editing image: {format_runnode_error(str(e))}"
            rn_pbar.error(error_message)
            log_error("系统异常", request_id, error_message, "RunNode/Doubao-", "SeedEdit")
            raise Exception(error_message)


class ComflyJimengApi:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "scale": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647, "step": 1}),
                "width": ("INT", {"default": 1328, "min": 512, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1328, "min": 512, "max": 2048, "step": 8}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                # "api_key": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "use_pre_llm": ("BOOLEAN", {"default": False}),
                "add_logo": ("BOOLEAN", {"default": False}),
                "logo_position": (["右下角", "左下角", "左上角", "右上角"], {"default": "右下角"}),
                "logo_language": (["中文", "英文"], {"default": "中文"}),
                "logo_text": ("STRING", {"default": "", "multiline": False}),
                "logo_opacity": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1}),
                "image": ("IMAGE",),  
                "image_url": ("STRING", {"default": "", "multiline": False}),  
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("generated_image", "response", "image_url")
    FUNCTION = "generate_image"
    CATEGORY = "RunNode/Doubao"
    
    def __init__(self):
        super().__init__()
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
    def get_logo_position_value(self, position_str):
        position_map = {
            "右下角": 0,
            "左下角": 1,
            "左上角": 2,
            "右上角": 3
        }
        return position_map.get(position_str, 0)
        
    def get_logo_language_value(self, language_str):
        language_map = {
            "中文": 0,
            "英文": 1
        }
        return language_map.get(language_str, 0)
    
    def upload_image(self, image_tensor, request_id=None):
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
                msg = f"Unexpected response from file upload API: {result}"
                if request_id:
                    log_error("上传异常", request_id, msg, "RunNode/Doubao-", "Jimeng")
                raise Exception(msg)
                
        except Exception as e:
            msg = f"Error uploading image: {format_runnode_error(str(e))}"
            if request_id:
                log_error("上传失败", request_id, msg, "RunNode/Doubao-", "Jimeng")
            raise Exception(msg)
    
    def generate_image(self, prompt, scale=2.5, seed=-1, width=1328, height=1328, use_pre_llm=False, 
                      add_logo=False, logo_position="右下角", logo_language="中文", 
                      logo_text="", logo_opacity=0.3, api_key="", image=None, image_url=""):
        request_id = generate_request_id("img_gen", "jimeng")
        log_prepare("Jimeng图像生成", request_id, "RunNode/Doubao-", "Jimeng")
        rn_pbar = ProgressBar(request_id, "Jimeng", streaming=True, task_type="图像生成", source="RunNode/Doubao-")
        rn_pbar.set_generating()

        if api_key.strip():
            self.api_key = api_key
            # config = get_config()
            # config['api_key'] = api_key
            # save_config(config)
        else:
            self.api_key = get_config().get('api_key', '')
            
        try:
            if not self.api_key:
                error_message = "API key not found in configuration file or environment variables."
                rn_pbar.error(error_message)
                log_error("配置缺失", request_id, error_message, "RunNode/Doubao-", "Jimeng")
                raise Exception(error_message)

            rn_pbar.update(10, "Initializing...")

            uploaded_image_url = None
            if image is not None:
                rn_pbar.update(20, "Uploading image...")
                # This will raise Exception if it fails, which is what we want
                uploaded_image_url = self.upload_image(image, request_id)
                if uploaded_image_url:
                    if not prompt.strip():
                        prompt = "Generate a 3D style version of this image"
                else:
                    # Should not be reached if upload_image raises exception, but safe to keep logic
                    rn_pbar.error("Image upload failed")
                    raise Exception("Image upload failed")

            final_image_url = uploaded_image_url if uploaded_image_url else image_url

            model_name = "seedream-3.0"  
            
            position_value = self.get_logo_position_value(logo_position)
            language_value = self.get_logo_language_value(logo_language)
            
            logo_info = {
                "add_logo": add_logo,
                "position": position_value,
                "language": language_value,
                "opacity": logo_opacity
            }
 
            if logo_text:
                logo_info["logo_text_content"] = logo_text

            payload = {
                "req_key": "high_aes_general_v30l_zt2i",
                "prompt": prompt,
                "scale": scale,
                "seed": seed,
                "width": width,
                "height": height,
                "use_pre_llm": use_pre_llm,
                "return_url": True,
                "logo_info": logo_info
            }

            if final_image_url:
                combined_prompt = f"{final_image_url} {prompt}"
                
                messages = [
                    {
                        "role": "user",
                        "content": combined_prompt
                    }
                ]
                
                chat_payload = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": 0.5,
                    "top_p": 1,
                    "presence_penalty": 0,
                    "max_tokens": 8192,
                    "stream": True
                }

                api_url = f"{baseurl}/v1/chat/completions"
                headers = self.get_headers()

                rn_pbar.update(30, "Submitting task...")

                response = requests.post(
                    api_url,
                    headers=headers,
                    json=chat_payload,
                    timeout=self.timeout,
                    stream=True
                )

                full_response = ""
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        if line_text.startswith('data: '):
                            data = line_text[6:]
                            if data == '[DONE]':
                                break
                            try:
                                chunk = json.loads(data)
                                if 'choices' in chunk and chunk['choices']:
                                    delta = chunk['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        content = delta['content']
                                        full_response += content
                            except json.JSONDecodeError:
                                continue

                image_url = ""
                image_urls = self.extract_image_urls(full_response)
                if image_urls:
                    image_url = image_urls[0]

                if image_url:
                    response_info = f"**Image Generation with {model_name}**\n\n"
                    response_info += f"Prompt: {prompt}\n"
                    response_info += f"Generated image URL: {image_url}\n\n"
                    response_info += f"Model response: {full_response}"

                    try:
                        img_response = requests.get(image_url, timeout=self.timeout)
                        img_response.raise_for_status()
                        generated_image = Image.open(BytesIO(img_response.content))
                        generated_tensor = pil2tensor(generated_image)
                        rn_pbar.update(100, "Done")
                        
                        safe_url = safe_public_url(image_url)
                        log_complete("Jimeng任务完成", request_id, "RunNode/Doubao-", "Jimeng", 
                                     image_count=1, first_url=safe_url)
                        rn_pbar.done(char_count=len(full_response))
                        
                        return (generated_tensor, response_info, image_url)
                    except Exception as e:
                        error_message = f"Error downloading result image: {format_runnode_error(str(e))}"
                        rn_pbar.error(error_message)
                        log_error("下载失败", request_id, error_message, "RunNode/Doubao-", "Jimeng")
                        raise Exception(error_message)
                else:
                    error_message = "No image URL found in response"
                    rn_pbar.error(error_message)
                    log_error("数据缺失", request_id, error_message, "RunNode/Doubao-", "Jimeng")
                    raise Exception(error_message)
            
            else:
                api_url = f"{baseurl}/volcv/v1?Action=CVProcess&Version=2022-08-31"
                
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                response_info = f"**Jimeng Image Generation Request**\n\n"
                response_info += f"Prompt: {prompt}\n"
                response_info += f"Scale: {scale}\n"
                response_info += f"Seed: {seed}\n"
                response_info += f"Dimensions: {width}x{height}\n"
                response_info += f"Time: {timestamp}\n\n"
                
                try:
                    response = requests.post(
                        api_url,
                        headers=self.get_headers(),
                        json=payload,
                        timeout=self.timeout
                    )
                except requests.exceptions.Timeout:
                    error_message = f"API request timed out after {self.timeout} seconds"
                    rn_pbar.error(error_message)
                    log_error("API超时", request_id, error_message, "RunNode/Doubao-", "Jimeng")
                    raise Exception(error_message)

                if response.status_code != 200:
                    error_message = format_runnode_error(response)
                    rn_pbar.error(error_message)
                    log_error("API请求失败", request_id, error_message, "RunNode/Doubao-", "Jimeng")
                    raise Exception(error_message)
                    
                result = response.json()
                
                rn_pbar.update(70, "Processing result...")

                if result.get("code") != 10000:
                    error_message = format_runnode_error(result)
                    rn_pbar.error(error_message)
                    log_error("API返回错误", request_id, error_message, "RunNode/Doubao-", "Jimeng")
                    raise Exception(error_message)

                image_url = ""
                if "image_urls" in result["data"] and result["data"]["image_urls"]:
                    image_url = result["data"]["image_urls"][0]
                    response_info += f"Success!\n\nImage URL: {image_url}\n\n"
                    
                    if "vlm_result" in result["data"] and result["data"]["vlm_result"]:
                        response_info += f"VLM Description: {result['data']['vlm_result']}\n"
                else:
                    error_message = "No image URL found in response"
                    rn_pbar.error(error_message)
                    log_error("数据缺失", request_id, error_message, "RunNode/Doubao-", "Jimeng")
                    raise Exception(error_message)
                
                # print(f"Found image URL: {image_url}")

                try:
                    img_response = requests.get(image_url, timeout=self.timeout)
                    img_response.raise_for_status()
                except requests.exceptions.Timeout:
                    error_message = f"Timeout while downloading result image after {self.timeout} seconds"
                    rn_pbar.error(error_message)
                    log_error("下载超时", request_id, error_message, "RunNode/Doubao-", "Jimeng")
                    raise Exception(error_message)
                except Exception as e:
                    error_message = f"Error downloading result image: {format_runnode_error(str(e))}"
                    rn_pbar.error(error_message)
                    log_error("下载失败", request_id, error_message, "RunNode/Doubao-", "Jimeng")
                    raise Exception(error_message)
                    
                generated_image = Image.open(BytesIO(img_response.content))
                
                generated_tensor = pil2tensor(generated_image)
                
                rn_pbar.update(100, "Done")
            
                if "request_id" in result:
                    response_info += f"Request ID: {result['request_id']}\n"
                
                if "time_elapsed" in result:
                    response_info += f"Processing Time: {result['time_elapsed']}\n"
                
                safe_url = safe_public_url(image_url)
                log_complete("Jimeng任务完成", request_id, "RunNode/Doubao-", "Jimeng", 
                             image_count=1, first_url=safe_url)
                rn_pbar.done(char_count=len(json.dumps(response_info, ensure_ascii=False)))
                
                return (generated_tensor, response_info, image_url)
                
        except Exception as e:
            error_message = f"Error in image generation: {format_runnode_error(str(e))}"
            rn_pbar.error(error_message)
            log_error("系统异常", request_id, error_message, "RunNode/Doubao-", "Jimeng")
            raise Exception(error_message)
            
    def extract_image_urls(self, response_text):
        """Extract image URLs from markdown format in response"""
        image_pattern = r'!\[.*?\]\((.*?)\)'
        matches = re.findall(image_pattern, response_text)

        if not matches:
            url_pattern = r'https?://\S+\.(?:jpg|jpeg|png|gif|webp)'
            matches = re.findall(url_pattern, response_text)
            
        return matches if matches else []


class ComflyJimengVideoApi:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "duration": ([5, 10], {"default": 5}),
                "aspect_ratio": (["1:1", "21:9", "16:9", "9:16", "4:3", "3:4"], {"default": "16:9"}),
                "cfg_scale": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                # "api_key": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647})
            }
        }
    
    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "task_id", "response", "video_url")
    FUNCTION = "generate_video"
    CATEGORY = "RunNode/Doubao"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def upload_image(self, image_tensor, request_id=None):
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
                msg = f"Unexpected response from file upload API: {result}"
                if request_id:
                    log_error(msg, request_id, "RunNode/Doubao-", "JimengVideo")
                raise Exception(msg)
                
        except Exception as e:
            msg = f"Error uploading image: {format_runnode_error(str(e))}"
            if request_id:
                log_error(msg, request_id, "RunNode/Doubao-", "JimengVideo")
            raise Exception(msg)
    
    def generate_video(self, prompt, duration, aspect_ratio, cfg_scale, api_key="", image=None, seed=0):
        if api_key.strip():
            self.api_key = api_key
        else:
            self.api_key = get_config().get('api_key', '')
            
        request_id = generate_request_id("video_gen", "jimeng")
        
        if not self.api_key:
            error_message = "API key not found in configuration file or environment variables."
            log_error(error_message, request_id, "RunNode/Doubao-", "JimengVideo")
            raise Exception(error_message)
            
        log_prepare("Jimeng视频生成", request_id, "RunNode/Doubao-", "JimengVideo")
        rn_pbar = ProgressBar(request_id, "JimengVideo", streaming=True, task_type="视频生成", source="RunNode/Doubao-")
        rn_pbar.set_generating()
            
        try:
            rn_pbar.update(10, "Initializing...")

            payload = {
                "prompt": prompt,
                "duration": int(duration),
                "aspect_ratio": aspect_ratio,
                "cfg_scale": cfg_scale
            }

            if seed > 0:
                payload["seed"] = seed

            image_url = None
            if image is not None:
                rn_pbar.update(20, "Uploading reference image...")
                # Will raise Exception if fails
                image_url = self.upload_image(image, request_id)
                if image_url:
                    payload["image_url"] = image_url
                else:
                    error_message = "Failed to upload image. Please check your image and try again."
                    log_error(error_message, request_id, "RunNode/Doubao-", "JimengVideo")
                    raise Exception(error_message)

            rn_pbar.update(30, "Submitting task...")
            response = requests.post(
                f"{baseurl}/jimeng/submit/videos",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                error_message = format_runnode_error(response)
                log_error(error_message, request_id, "RunNode/Doubao-", "JimengVideo")
                raise Exception(error_message)
                
            result = response.json()
            
            if result.get("code") != "success":
                error_message = f"API returned error: {result.get('message', 'Unknown error')}"
                log_error(error_message, request_id, "RunNode/Doubao-", "JimengVideo")
                raise Exception(error_message)
                
            task_id = result.get("data")
            if not task_id:
                error_message = "No task ID returned from API"
                log_error(error_message, request_id, "RunNode/Doubao-", "JimengVideo")
                raise Exception(error_message)
            
            rn_pbar.update(40, f"Task submitted (ID: {task_id}), waiting for generation...")
            video_url = None
            attempts = 0
            max_attempts = 18  
            start_time = time.time()
            max_wait_time = 300 
        
            while attempts < max_attempts:
                current_time = time.time()
                elapsed_time = current_time - start_time

                if elapsed_time > max_wait_time:
                    error_message = f"Video generation timeout after {elapsed_time:.1f} seconds (max: {max_wait_time}s)"
                    log_error(error_message, request_id, "RunNode/Doubao-", "JimengVideo")
                    raise Exception(error_message)
                
                time.sleep(5)  
                attempts += 1
                
                try:
                    status_response = requests.get(
                        f"{baseurl}/jimeng/fetch/{task_id}",
                        headers=self.get_headers(),
                        timeout=30
                    )
                    
                    if status_response.status_code != 200:
                        continue
                        
                    status_result = status_response.json()

                    if status_result.get("code") != "success":
                        continue

                    data = status_result.get("data", {})
                    progress = data.get("progress", "0%")
                    status = data.get("status", "")

                    try:
                        if isinstance(progress, str) and progress.endswith('%'):
                            progress_num = int(progress.rstrip('%'))
                            pbar_value = min(90, 40 + progress_num * 50 / 100)
                            rn_pbar.update(pbar_value, f"Generating: {progress}")
                        else:
                            # Fallback progress calculation
                            progress_value = min(80, 40 + (attempts * 40 // max_attempts))
                            rn_pbar.update(progress_value, f"Generating... ({attempts}/{max_attempts})")
                    except (ValueError, AttributeError):
                        progress_value = min(80, 40 + (attempts * 40 // max_attempts))
                        rn_pbar.update(progress_value, f"Generating... ({attempts}/{max_attempts})")

                    if status == "SUCCESS":
                        video_url = None

                        if "video" in data:
                            video_url = data["video"]

                        elif "data" in data and isinstance(data["data"], dict):
                            nested_data = data["data"]
                            if "video" in nested_data:
                                video_url = nested_data["video"]
                            elif "videos" in nested_data and isinstance(nested_data["videos"], list) and len(nested_data["videos"]) > 0:
                                if "url" in nested_data["videos"][0]:
                                    video_url = nested_data["videos"][0]["url"]

                        elif "task_result" in data:
                            task_result = data["task_result"]
                            if "videos" in task_result and isinstance(task_result["videos"], list) and len(task_result["videos"]) > 0:
                                if "url" in task_result["videos"][0]:
                                    video_url = task_result["videos"][0]["url"]
                        
                        if video_url:
                            break
                        else:
                            continue

                    elif status == "FAILED":
                        fail_reason = data.get("fail_reason", "Unknown error")
                        error_message = f"Video generation failed: {fail_reason}"
                        log_error(error_message, request_id, "RunNode/Doubao-", "JimengVideo")
                        raise Exception(error_message)
                    
                    elif status in ["PENDING", "PROCESSING", "RUNNING"]:
                        continue
                    else:
                        continue
                    
                except requests.exceptions.Timeout:
                    continue
                except Exception as e:
                    # Don't fail the loop on transient errors
                    continue
            
            if not video_url:
                error_message = f"Video generation timeout or failed to retrieve video URL after {attempts} attempts, elapsed time: {elapsed_time:.1f}s"
                log_error(error_message, request_id, "RunNode/Doubao-", "JimengVideo")
                raise Exception(error_message)

            if video_url:
                rn_pbar.update(95, "Video generated, finalizing...")
                
                safe_url = safe_public_url(video_url)
                log_complete("Jimeng视频生成完成", request_id, "RunNode/Doubao-", "JimengVideo", 
                             video_url=safe_url)
                rn_pbar.done(char_count=len(json.dumps({"code": "success", "url": video_url})))
                
                video_adapter = ComflyVideoAdapter(video_url)
                return (video_adapter, task_id, json.dumps({"code": "success", "url": video_url}), video_url)
            
        except Exception as e:
            error_message = f"Error generating video: {format_runnode_error(str(e))}"
            log_error(error_message, request_id, "RunNode/Doubao-", "JimengVideo")
            import traceback
            traceback.print_exc()
            raise


class ComflySeededit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647, "step": 1}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                # "api_key": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "add_logo": ("BOOLEAN", {"default": False}),
                "logo_position": (["右下角", "左下角", "左上角", "右上角"], {"default": "右下角"}),
                "logo_language": (["中文", "英文"], {"default": "中文"}),
                "logo_text": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("edited_image", "response", "image_url")
    FUNCTION = "edit_image"
    CATEGORY = "RunNode/Doubao"
    
    def __init__(self):
        super().__init__()
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
    def get_logo_position_value(self, position_str):
        position_map = {
            "右下角": 0,
            "左下角": 1,
            "左上角": 2,
            "右上角": 3
        }
        return position_map.get(position_str, 0)
        
    def get_logo_language_value(self, language_str):
        language_map = {
            "中文": 0,
            "英文": 1
        }
        return language_map.get(language_str, 0)
    
    def edit_image(self, image, prompt, scale=0.5, seed=-1, add_logo=False, logo_position="右下角", 
                   logo_language="中文", logo_text="", api_key=""):
        if api_key.strip():
            self.api_key = api_key
        else:
            self.api_key = get_config().get('api_key', '')
            
        request_id = generate_request_id("img_edit", "doubao")
        log_prepare("Doubao图像编辑", request_id, "RunNode/Doubao-", "SeedEdit")
        rn_pbar = ProgressBar(request_id, "SeedEdit", streaming=True, task_type="图像编辑", source="RunNode/Doubao-")
        rn_pbar.set_generating()
            
        try:
            if not self.api_key:
                error_message = "API key not found in configuration file or environment variables."
                log_error(error_message, request_id, "RunNode/Doubao-", "SeedEdit")
                raise Exception(error_message)
                
            rn_pbar.update(10, "Processing input image...")
            # Convert tensor to PIL image
            pil_image = tensor2pil(image)[0]
            
            # Convert image to base64
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            position_value = self.get_logo_position_value(logo_position)
            language_value = self.get_logo_language_value(logo_language)
            
            logo_info = {
                "add_logo": add_logo,
                "position": position_value,
                "language": language_value
            }
 
            if logo_text:
                logo_info["logo_text_content"] = logo_text
            
            # Prepare the API request
            payload = {
                "req_key": "byteedit_v2.0",
                "binary_data_base64": [img_base64],
                "prompt": prompt,
                "scale": scale,
                "seed": seed,
                "return_url": True,
                "logo_info": logo_info
            }
            
            # Call the API
            api_url = f"{baseurl}/volcv/v1?Action=CVProcess&Version=2022-08-31"
            
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            response_info = f"**SeedEdit Request**\n\n"
            response_info += f"Prompt: {prompt}\n"
            response_info += f"Scale: {scale}\n"
            response_info += f"Seed: {seed}\n"
            response_info += f"Time: {timestamp}\n\n"
            
            rn_pbar.update(30, "Submitting edit request...")
            try:
                response = requests.post(
                    api_url,
                    headers=self.get_headers(),
                    json=payload,
                    timeout=self.timeout
                )
            except requests.exceptions.Timeout:
                error_message = f"API request timed out after {self.timeout} seconds"
                log_error(error_message, request_id, "RunNode/Doubao-", "SeedEdit")
                raise Exception(error_message)
            
            # Check for status code
            if response.status_code != 200:
                error_message = format_runnode_error(response)
                log_error(error_message, request_id, "RunNode/Doubao-", "SeedEdit")
                raise Exception(error_message)
                
            result = response.json()
            
            rn_pbar.update(70, "Processing result...")
            
            # Check for API errors
            if result.get("code") != 10000:
                error_message = format_runnode_error(result)
                log_error(error_message, request_id, "RunNode/Doubao-", "SeedEdit")
                raise Exception(error_message)
            
            # Get the result image URL
            image_url = ""
            if "image_urls" in result["data"] and result["data"]["image_urls"]:
                image_url = result["data"]["image_urls"][0]
                response_info += f"Success!\n\nImage URL: {image_url}\n\n"
                
                if "vlm_result" in result["data"] and result["data"]["vlm_result"]:
                    response_info += f"VLM Description: {result['data']['vlm_result']}\n"
            else:
                error_message = "No image URL found in response"
                log_error(error_message, request_id, "RunNode/Doubao-", "SeedEdit")
                raise Exception(error_message)
            
            rn_pbar.update(80, "Downloading result image...")
            
            # Download the image
            try:
                img_response = requests.get(image_url, timeout=self.timeout)
                img_response.raise_for_status()
            except requests.exceptions.Timeout:
                error_message = f"Timeout while downloading result image after {self.timeout} seconds"
                log_error(error_message, request_id, "RunNode/Doubao-", "SeedEdit")
                raise Exception(error_message)
            except Exception as e:
                error_message = f"Error downloading result image: {format_runnode_error(str(e))}"
                log_error(error_message, request_id, "RunNode/Doubao-", "SeedEdit")
                raise Exception(error_message)
                
            edited_image = Image.open(BytesIO(img_response.content))
            
            # Convert back to tensor
            edited_tensor = pil2tensor(edited_image)
            
            rn_pbar.update(100, "Done")
        
            if "request_id" in result:
                response_info += f"Request ID: {result['request_id']}\n"
            
            if "time_elapsed" in result:
                response_info += f"Processing Time: {result['time_elapsed']}\n"
            
            safe_url = safe_public_url(image_url)
            log_complete("Doubao图像编辑完成", request_id, "RunNode/Doubao-", "SeedEdit", first_url=safe_url)
            rn_pbar.done(char_count=len(json.dumps(response_info, ensure_ascii=False)))
            
            return (edited_tensor, response_info, image_url)
            
        except Exception as e:
            error_message = f"Error in image editing: {format_runnode_error(str(e))}"
            log_error(error_message, request_id, "RunNode/Doubao-", "SeedEdit")
            raise Exception(error_message)


class Comfly_Doubao_Seedance_2_0:
    @classmethod
    def INPUT_TYPES(cls):
        return{
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["doubao-seedance-2-0-260128", "doubao-seedance-2-0-fast-260128", "doubao-seedance-2.0-mini"], {"default": "doubao-seedance-2-0-260128"}),
                "duration": ("INT", {"default": 5, "min": 4, "max": 15, "step": 1, "tooltip": "视频时长，单位秒"}),
                "ratio": (["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "9:21", "adaptive"], {"default": "16:9", "tooltip": "视频比例"}),
                "resolution": (["720p", "480p", "native1080p", "1080p", "2k", "4k", "native4K"], {"default": "720p", "tooltip": "视频分辨率"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "first_frame": ("IMAGE", {"tooltip": "第一帧"}),
                "last_frame": ("IMAGE", {"tooltip": "最后一帧"}),
                "ref_image1": ("IMAGE", {"tooltip": "参考图片1"}),
                "ref_image2": ("IMAGE", {"tooltip": "参考图片2"}),
                "ref_image3": ("IMAGE", {"tooltip": "参考图片3"}),
                "ref_image4": ("IMAGE", {"tooltip": "参考图片4"}),
                "ref_image5": ("IMAGE", {"tooltip": "参考图片5"}),
                "ref_image6": ("IMAGE", {"tooltip": "参考图片6"}),
                "ref_image7": ("IMAGE", {"tooltip": "参考图片7"}),
                "ref_image8": ("IMAGE", {"tooltip": "参考图片8"}),
                "ref_image9": ("IMAGE", {"tooltip": "参考图片9"}),  
                "video1": (IO.VIDEO, {"tooltip": "参考视频1"}),
                "video2": (IO.VIDEO, {"tooltip": "参考视频2"}),
                "video3": (IO.VIDEO, {"tooltip": "参考视频3"}),
                "audio1": (IO.AUDIO, {"tooltip": "参考音频1"}),
                "audio2": (IO.AUDIO, {"tooltip": "参考音频2"}),
                "audio3": (IO.AUDIO, {"tooltip": "参考音频3"}),
                "generate_audio": ("BOOLEAN", {"default": True}),
                "return_last_frame": ("BOOLEAN", {"default": False}),
                "web_search": ("BOOLEAN", {"default": False}),
                "watermark": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "asset_bundle": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "请连接「Asset ID Bundle」的输出。 它会生成一个 JSON 数据，把每个槽位里那张资产的编号（asset_id）整理到一起，方便后续节点使用。",
                    },
                ),
                "skip_error": ("BOOLEAN", {"default": False, "tooltip": "开启后，节点失败时不报错、按旧行为返回默认空结果；关闭时（默认）失败直接抛出错误。"})
            }
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("video", "task_id", "response", "video_url", "last_frame_image")
    FUNCTION = "generate_video"
    CATEGORY = "RunNode/Doubao"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 3600
        self.poll_interval = 10
        self.max_wait_time = 3600

    def get_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def image_tensor_to_base64(self, image_tensor):
        if image_tensor is None:
            return None
        try:
            pil_image = tensor2pil(image_tensor)[0]
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            b64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{b64_str}"
        except Exception as e:
            print(f"Image to base64 error: {str(e)}")
            return None

    def upload_file(self, file_content, filename, content_type):
        try:
            files = {'file': (filename, file_content, content_type)}
            response = requests.post(
                f"{baseurl}/v1/files",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files=files,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            if isinstance(result, dict) and result.get("url"):
                return result["url"]
            msg = f"Unexpected response from file upload API: {result}"
            raise Exception(msg)
                
        except Exception as e:
            msg = f"File upload error: {format_runnode_error(str(e))}"
            raise Exception(msg)

    def upload_image_get_url(self, image_tensor):
        """Upload IMAGE tensor to /v1/files as PNG; return HTTPS URL for assets/create."""
        if image_tensor is None:
            return None
        try:
            pil_image = tensor2pil(image_tensor)[0]
            buf = BytesIO()
            pil_image.save(buf, format="PNG")
            data = buf.getvalue()
            fn = f"upload_{abs(hash(data)) % 10**10}.png"
            url = self.upload_file(data, fn, "image/png")
            if url:
                print(f"Image uploaded successfully: {url}")
            return url
        except Exception as e:
            raise Exception(f"Image upload error: {format_runnode_error(str(e))}")

    def upload_video_get_url(self, video_input):
        if video_input is None:
            return None
        try:
            file_content, filename = _doubao_seedance_video_input_to_bytes(video_input)
            if not file_content:
                raise Exception("empty file content")
            if not filename:
                filename = f"reference_video_{abs(hash(file_content)) % 10**10}.mp4"

            mime_type, _ = mimetypes.guess_type(filename)
            if not mime_type:
                mime_type = "video/mp4"

            url = self.upload_file(file_content, filename, mime_type)
            if url:
                print(f"Video uploaded successfully: {url}")
            return url
        except Exception as e:
            raise Exception(f"Video upload error: {format_runnode_error(str(e))}")

    def upload_audio_get_url(self, audio_input):
        """
        POST /v1/files: prefer reading an on-disk / stream file as raw bytes (no re-encode).
        Only when the input is pure Comfy AUDIO {waveform, sample_rate} with no file path,
        encode to WAV via _comfy_waveform_to_wav_bytes.
        """
        if audio_input is None:
            return None
        try:
            # 1) Path, dict.path, get_stream_source(), etc. — upload original bytes
            file_content, filename = _doubao_seedance_io_file_to_bytes(audio_input, ".wav", "audio")
            if file_content:
                if not filename:
                    filename = f"reference_audio_{abs(hash(file_content)) % 10**10}.wav"
                mime_type, _ = mimetypes.guess_type(filename)
                if not mime_type:
                    mime_type = "audio/wav"
                url = self.upload_file(file_content, filename, mime_type)
                if url:
                    print(f"Audio uploaded successfully (from file/stream): {url}")
                return url

            # 2) Standard Comfy AUDIO: waveform tensor only — must encode to WAV bytes
            if isinstance(audio_input, dict) and audio_input.get("waveform") is not None:
                waveform = audio_input["waveform"]
                if torch.is_tensor(waveform):
                    sample_rate = int(audio_input.get("sample_rate", 44100))
                    if waveform.dim() == 3:
                        waveform = waveform.squeeze(0)
                    if waveform.dim() == 1:
                        waveform = waveform.unsqueeze(0)
                    file_content = _comfy_waveform_to_wav_bytes(waveform, sample_rate)
                    url = self.upload_file(file_content, "audio.wav", "audio/wav")
                    if url:
                        print(f"Audio uploaded successfully (from waveform): {url}")
                    return url

            return None
        except Exception as e:
            raise Exception(f"Audio upload error: {format_runnode_error(str(e))}")

    def download_image_from_url(self, url):
        try:
            img_response = requests.get(url, timeout=60)
            img_response.raise_for_status()
            pil_image = Image.open(BytesIO(img_response.content))
            return pil2tensor(pil_image)
        except Exception as e:
            return None

    def generate_video(self, prompt, model, duration, ratio, resolution,
                       api_key="",
                       first_frame=None, last_frame=None,
                       ref_image1=None, ref_image2=None, ref_image3=None,
                       ref_image4=None, ref_image5=None, ref_image6=None,
                       ref_image7=None, ref_image8=None, ref_image9=None,
                       video1=None, video2=None, video3=None,
                       audio1=None, audio2=None, audio3=None,
                       generate_audio=True, return_last_frame=False,
                       web_search=False,
                       watermark=False, seed=-1,
                       asset_bundle="", skip_error=False):
        
        blank_image = Image.new('RGB', (1, 1), color='black')
        blank_tensor = pil2tensor(blank_image)
        empty_video = ComflyVideoAdapter("")
        task_id = ""

        request_id = generate_request_id("video_gen", "doubao")
        log_prepare("Seedance视频生成", request_id, "RunNode/Doubao-", "Seedance2")
        rn_pbar = ProgressBar(request_id, "Seedance2", streaming=True, task_type="视频生成", source="RunNode/Doubao-")
        rn_pbar.set_generating()

        if api_key.strip():
            self.api_key = api_key
        else:
            self.api_key = get_config().get('api_key', '')

        if not self.api_key:
            error_message = "API key not found in configuration file or environment variables."
            rn_pbar.error(error_message)
            log_error("配置缺失", request_id, error_message, "RunNode/Doubao-", "Seedance2")
            if skip_error:
                return (empty_video, "", json.dumps({"error": error_message}, ensure_ascii=False), "", blank_tensor)
            raise RuntimeError(error_message)

        try:
            _rn_start = time.perf_counter()
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(5)

            asset_id_first_frame, asset_id_last_frame, asset_ids_ref_images, asset_ids_ref_videos, asset_ids_ref_audios = _parse_asset_bundle_only(
                asset_bundle
            )

            content = []
            content.append({"type": "text", "text": prompt})

            frame_count = 0
            has_first_tensor = first_frame is not None
            has_last_tensor = last_frame is not None

            if has_first_tensor:
                b64 = self.image_tensor_to_base64(first_frame)
                if b64:
                    entry = {'type': 'image_url', 'image_url': {'url':b64}}
                    if has_last_tensor:
                        entry['role'] = "first_frame"
                    content.append(entry)
                    frame_count += 1
            else:
                u = _comfly_asset_id_to_url(asset_id_first_frame)
                if u:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": u},
                            "role": "first_frame",
                        }
                    )
                    frame_count += 1

            has_first_effective = has_first_tensor or bool(_comfly_asset_id_to_url(asset_id_first_frame))

            if has_last_tensor:
                if not has_first_effective:
                    print("Warning: last_frame without first_frame, skipping.")
                else:
                    b64 = self.image_tensor_to_base64(last_frame)
                    if b64:
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": b64},
                                "role": "last_frame"
                            }
                        )
                        frame_count += 1

            if not has_last_tensor and has_first_effective:
                u = _comfly_asset_id_to_url(asset_id_last_frame)
                if u:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": u},
                        "role": "last_frame"
                    })
                    frame_count += 1
                
            ref_images = [
                ref_image1, ref_image2, ref_image3, 
                ref_image4, ref_image5, ref_image6, 
                ref_image7, ref_image8, ref_image9
            ]
            ref_count = 0

            for img in ref_images:
                if img is not None:
                    b64 = self.image_tensor_to_base64(img)
                    if b64:
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": b64},
                                "role": "reference_image"
                            }
                        )
                        ref_count += 1

            for aid in _comfly_split_asset_ids(asset_ids_ref_images):
                u = _comfly_asset_id_to_url(aid)
                if u:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": u},
                            "role": "reference_image"
                        }
                    )
                    ref_count += 1
            
            video_inputs = [video1, video2, video3]
            video_count = 0

            for vid in video_inputs:
                url = self.upload_video_get_url(vid)
                if url:
                    content.append(
                        {
                            "type": "video_url",
                            "video_url": {"url": url},
                            "role": "reference_video"
                        }
                    )
                    video_count += 1

            for aid in _comfly_split_asset_ids(asset_ids_ref_videos):
                u = _comfly_asset_id_to_url(aid)
                if u:
                    content.append(
                        {
                            "type": "video_url",
                            "video_url": {"url": u},
                            "role": "reference_video"
                        }
                    )
                    video_count += 1

            pbar.update_absolute(25)

            audio_inputs = [audio1, audio2, audio3]
            audio_count = 0

            for aud in audio_inputs:
                url = self.upload_audio_get_url(aud)
                if url:
                    content.append(
                        {
                            "type": "audio_url",
                            "audio_url": {"url": url},
                            "role": "reference_audio"
                        }
                    )
                    audio_count += 1

            for aud in _comfly_split_asset_ids(asset_ids_ref_audios):
                u = _comfly_asset_id_to_url(aud)
                if u:
                    content.append(
                        {
                            "type": "audio_url",
                            "audio_url": {"url": u},
                            "role": "reference_audio"
                        }
                    )
                    audio_count += 1

            pbar.update_absolute(30)

            payload = {
                "model": model,
                "content": content,
                "duration": int(duration),
                "ratio": ratio,
                "resolution": resolution,
                "generate_audio": generate_audio,
                "return_last_frame": return_last_frame,
                "watermark": watermark
            }
            
            if web_search:
                payload["tools"] = [{"type": "web_search"}]
            if seed != -1:
                payload["seed"] = seed

            log_backend(
                "doubao_seedance_submit",
                request_id=request_id,
                model=model,
                duration=int(duration),
                ratio=ratio,
                resolution=resolution,
                frames=frame_count,
                ref_images=ref_count,
                videos=video_count,
                audios=audio_count,
            )

            response = requests.post(
                f"{baseurl}/seedance/v3/contents/generations/tasks",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )

            pbar.update_absolute(35)

            if response.status_code != 200:
                error_message = format_runnode_error(response)
                rn_pbar.error(error_message)
                log_error("API请求失败", request_id, error_message, "RunNode/Doubao-", "Seedance2")
                if not skip_error:
                    raise RuntimeError(f"[Comfly_Doubao_Seedance2_0] {error_message}")
                return (empty_video, "", json.dumps({"error": error_message}, ensure_ascii=False), "", blank_tensor)

            result = response.json()
            task_id = result.get('id', '') or result.get('task_id', '')
            if not task_id:
                error_message = f"No task ID. Response: {json.dumps(result, ensure_ascii=False)}"
                rn_pbar.error(error_message)
                log_error("任务ID缺失", request_id, error_message, "RunNode/Doubao-", "Seedance2")
                if not skip_error:
                    raise RuntimeError(f"[Comfly_Doubao_Seedance2_0] {error_message}")
                return (empty_video, "", json.dumps({"error": error_message}, ensure_ascii=False), "", blank_tensor)

            log_backend("doubao_seedance_poll_start", request_id=request_id, task_id=task_id)

            pbar.update_absolute(40)
            start_time = time.time()
            video_url = None
            last_frame_url = None
            final_status_data = None
            last_log_time = start_time

            while True:
                elapsed = time.time() - start_time
                if elapsed > self.max_wait_time:
                    if not skip_error:
                        raise RuntimeError(f"[Comfly_Doubao_Seedance2_0] Timeout {elapsed:.1f}s")
                    error_message = f"Timeout {elapsed:.1f}s"
                    rn_pbar.error(error_message)
                    log_error("生成超时", request_id, error_message, "RunNode/Doubao-", "Seedance2")
                    return (empty_video, task_id, json.dumps({"error": error_message, "task_id": task_id}, ensure_ascii=False), "", blank_tensor)

                time.sleep(self.poll_interval)

                try:
                    now = time.time()
                    if now - last_log_time >= 15:
                        log_backend(
                            "doubao_seedance_poll_heartbeat",
                            request_id=request_id,
                            task_id=task_id,
                            elapsed=int(now - start_time),
                        )
                        last_log_time = now

                    status_response = requests.get(
                        f"{baseurl}/seedance/v3/contents/generations/tasks/{task_id}",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        timeout=30
                    )
                    if status_response.status_code != 200:
                        continue

                    status_data = status_response.json()
                    final_status_data = status_data

                    raw_status = status_data.get("status", "")
                    status = raw_status.lower()
                    if status == "success":
                        status = "succeeded"
                    elif status in ("fail", "failure"):
                        status = "failed"

                    progress_str = status_data.get("progress", "")
                    progress = min(90, 40 + int((elapsed / self.max_wait_time) * 50))
                    pbar.update_absolute(progress)

                    if status == "succeeded":
                        # Root { "content": { "video_url": "..." }, "status": "succeeded" } — common shape
                        root_content = status_data.get("content")
                        if isinstance(root_content, dict):
                            video_url = root_content.get("video_url") or root_content.get("videoUrl")

                        # Nested: data.content.video_url
                        data = status_data.get("data")
                        if isinstance(data, dict):
                            data_content = data.get("content")
                            if isinstance(data_content, dict):
                                if not video_url:
                                    video_url = data_content.get("video_url") or data_content.get("videoUrl")
                                if video_url:
                                    print("Found video in data.content.video_url")
                            if not video_url:
                                video_url = data.get("video_url") or data.get("videoUrl")

                        if not video_url:
                            results = status_data.get("results", [])
                            if isinstance(results, list):
                                for r in results:
                                    if isinstance(r, dict):
                                        r_url = r.get("url", "")
                                        r_type = r.get("outputType", "")
                                        if r_type in ("mp4", "video") or r_url.endswith(".mp4"):
                                            video_url = r_url
                                            break
                                        elif r_url and not video_url:
                                            video_url = r_url

                        if not video_url:
                            content_list = status_data.get("content")
                            if isinstance(content_list, list):
                                for item in content_list:
                                    if not isinstance(item, dict):
                                        continue
                                    item_type = item.get("type", "")
                                    item_role = item.get("role", "")
                                    if item_type == "video_url":
                                        vu = item.get("video_url")
                                        if isinstance(vu, dict):
                                            video_url = vu.get("url", "")
                                        elif isinstance(vu, str):
                                            video_url = vu
                                        if video_url:
                                            break
                                    if item_type == "image_url" and item_role == "last_frame":
                                        iu = item.get("image_url")
                                        if isinstance(iu, dict):
                                            last_frame_url = iu.get("url", "")
                                        elif isinstance(iu, str):
                                            last_frame_url = iu

                        if not video_url:
                            video_url = status_data.get("video_url") or status_data.get("videoUrl")

                        if not last_frame_url and return_last_frame:
                            last_frame_url = (
                                status_data.get("last_frame_url")
                                or status_data.get("lastFrameUrl")
                                or status_data.get("last_frame_image_url")
                            )
                            if not last_frame_url:
                                lf = status_data.get("last_frame") or status_data.get("lastFrame")
                                if isinstance(lf, dict):
                                    last_frame_url = lf.get("url", "")
                                elif isinstance(lf, str):
                                    last_frame_url = lf

                        if video_url:
                            log_backend(
                                "doubao_seedance_poll_success",
                                request_id=request_id,
                                task_id=task_id,
                                video_url=safe_public_url(video_url),
                            )
                            break
                        else:
                            error_message = f"Succeeded but no video URL found: {json.dumps(status_data, ensure_ascii=False)}"
                            rn_pbar.error(error_message)
                            log_error("结果缺失", request_id, error_message, "RunNode/Doubao-", "Seedance2")
                            if not skip_error:
                                raise RuntimeError(f"[Comfly_Doubao_Seedance2_0] {error_message}")
                            return (empty_video, task_id, json.dumps({"error": error_message, "task_id": task_id}, ensure_ascii=False), "", blank_tensor)

                    elif status == "failed":
                        fail_reason = status_data.get("fail_reason", "") or status_data.get("failReason", "")
                        error_message = f"Task failed: {fail_reason}"
                        rn_pbar.error(error_message)
                        log_error("生成失败", request_id, fail_reason, "RunNode/Doubao-", "Seedance2")
                        if not skip_error:
                            raise RuntimeError(f"[Comfly_Doubao_Seedance2_0] {fail_reason}")
                        return (empty_video, task_id, json.dumps(status_data, indent=2, ensure_ascii=False), "", blank_tensor)
                
                except requests.exceptions.Timeout:
                    continue
                except Exception as e:
                    continue

            if video_url:
                pbar.update_absolute(95)

                last_frame_tensor = blank_tensor
                if return_last_frame and last_frame_url:
                    downloaded_frame = self.download_image_from_url(last_frame_url)
                    if downloaded_frame is not None:
                        last_frame_tensor = downloaded_frame

                response_info = {
                    "task_id": task_id,
                    "model": model,
                    "status": "succeeded",
                    "video_url": video_url,
                    "duration": duration,
                    "ratio": ratio,
                    "resolution": resolution,
                    "generate_audio": generate_audio,
                    "return_last_frame": return_last_frame,
                    "seed": seed if seed != -1 else "auto",
                    "first_frame": has_first_effective,
                    "last_frame_input": has_first_effective
                    and (has_last_tensor or bool(_comfly_asset_id_to_url(asset_id_last_frame))),
                    "reference_images": ref_count,
                    "reference_videos": video_count,
                    "reference_audios": audio_count,
                }

                if last_frame_url:
                    response_info["last_frame_image_url"] = last_frame_url
                if final_status_data and isinstance(final_status_data, dict):
                    data = final_status_data.get("data")
                    if isinstance(data, dict):
                        if "duration" in data:
                            response_info["actual_duration"] = data["duration"]
                        if "usage" in data:
                            response_info["usage"] = data["usage"]
                    if "usage" in final_status_data:
                        response_info["usage"] = final_status_data["usage"]
                    if "duration" in final_status_data and "actual_duration" not in response_info:
                        response_info["actual_duration"] = final_status_data["duration"]

                # Prefer comfy_api VideoFromFile so Save Video (IO.VIDEO) can preview/save reliably
                video_out = ComflyVideoAdapter(video_url)
                try:
                    from comfy_api.latest import VideoFromFile as CFVideoFromFile

                    fd, tmp_path = tempfile.mkstemp(suffix=".mp4", prefix="comfly_seedance_")
                    os.close(fd)
                    if video_out.save_to(tmp_path):
                        video_out = CFVideoFromFile(tmp_path)
                except Exception as e:
                    print(f"[Comfly Seedance] Using ComflyVideoAdapter (VideoFromFile unavailable): {e}")

                pbar.update_absolute(100)
                response_text = json.dumps(response_info, indent=2, ensure_ascii=False)
                elapsed_ms = int((time.perf_counter() - _rn_start) * 1000)
                rn_pbar.done(char_count=len(response_text), elapsed_ms=elapsed_ms)
                log_complete(
                    "Seedance视频生成完成",
                    request_id,
                    "RunNode/Doubao-",
                    "Seedance2",
                    video_url=safe_public_url(video_url),
                    char_count=len(response_text),
                    elapsed_ms=elapsed_ms,
                )
                return (video_out, task_id, response_text, video_url, last_frame_tensor)
            else:
                if not skip_error:
                    raise RuntimeError("[Comfly_Doubao_Seedance2_0] Video adapter init failed (see terminal log for details)")
                error_message = "No video URL"
                rn_pbar.error(error_message)
                log_error("结果缺失", request_id, error_message, "RunNode/Doubao-", "Seedance2")
                return (empty_video, task_id, json.dumps({"error": error_message, "task_id": task_id}, ensure_ascii=False), "", blank_tensor)

        except Exception as e:
            error_message = f"Seedance 2.0 generate_video failed: {format_runnode_error(str(e))}"
            rn_pbar.error(error_message)
            log_error("系统异常", request_id, error_message, "RunNode/Doubao-", "Seedance2")
            import traceback
            traceback.print_exc()
            if skip_error:
                return (empty_video, "", json.dumps({"error": error_message, "task_id": task_id}, ensure_ascii=False), "", blank_tensor)
            raise Exception(error_message)


class Comfly_Doubao_Seedance_2_0_AssetIdBundle:
    """
    Collect asset_id strings from «Comfly Doubao Seedance 2.0 Asset Upload» (one slot per wire),
    same layout as Seedance 2.0, into one JSON for the main node's asset_bundle input.
    """

    @classmethod
    def INPUT_TYPES(cls):
        opt = {
            "apikey": ("STRING", {"default": ""}),
            "first_frame": ("STRING", {"default": "", "tooltip": "来自「RunNode Doubao Seedance 2.0 Asset」节点的 asset_id"}),
            "last_frame": ("STRING", {"default": "", "tooltip": "来自「RunNode Doubao Seedance 2.0 Asset」节点的 asset_id"}),
        }
        for i in range(1, 10):
            opt[f"ref_image{i}"] = ("STRING", {"default": "", "tooltip": "来自「RunNode Doubao Seedance 2.0 Asset」节点的 asset_id"})
        for i in range(1, 4):
            opt[f"video{i}"] = ("STRING", {"default": "", "tooltip": "来自「RunNode Doubao Seedance 2.0 Asset」节点的 asset_id"})
        for i in range(1, 4):
            opt[f"audio{i}"] = ("STRING", {"default": "", "tooltip": "来自「RunNode Doubao Seedance 2.0 Asset」节点的 asset_id"})
        return {"required": {}, "optional": opt}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("asset_bundle",)
    FUNCTION = "bundle"
    CATEGORY = "RunNode/Doubao"

    def bundle(
        self,
        apikey="",
        first_frame="",
        last_frame="",
        ref_image1="",
        ref_image2="",
        ref_image3="",
        ref_image4="",
        ref_image5="",
        ref_image6="",
        ref_image7="",
        ref_image8="",
        ref_image9="",
        video1="",
        video2="",
        video3="",
        audio1="",
        audio2="",
        audio3="",
    ):
        request_id = generate_request_id("asset_bundle", "doubao")
        log_prepare("Seedance资产Bundle", request_id, "RunNode/Doubao-", "Seedance2AssetBundle")
        rn_pbar = ProgressBar(request_id, "Seedance2AssetBundle", streaming=True, task_type="资产Bundle", source="RunNode/Doubao-")
        rn_pbar.set_generating()

        try:
            def sid(x):
                return (x or "").strip() if x is not None else ""

            ff = sid(first_frame)
            lf = sid(last_frame)
            ref_images = []
            for i in range(1, 10):
                t = sid(locals().get(f"ref_image{i}"))
                if t:
                    ref_images.append(t)
            videos = []
            for i in range(1, 4):
                t = sid(locals().get(f"video{i}"))
                if t:
                    videos.append(t)
            audios = []
            for i in range(1, 4):
                t = sid(locals().get(f"audio{i}"))
                if t:
                    audios.append(t)

            payload = {
                "first_frame": ff,
                "last_frame": lf,
                "ref_images": ref_images,
                "videos": videos,
                "audios": audios,
            }

            response_text = json.dumps(payload, ensure_ascii=False)
            rn_pbar.done(char_count=len(response_text))
            log_complete(
                "Seedance资产Bundle完成",
                request_id,
                "RunNode/Doubao-",
                "Seedance2AssetBundle",
                char_count=len(response_text),
            )
            return (response_text,)
        except Exception as e:
            error_message = f"Seedance资产Bundle失败: {format_runnode_error(str(e))}"
            rn_pbar.error(error_message)
            log_error("系统异常", request_id, error_message, "RunNode/Doubao-", "Seedance2AssetBundle")
            raise


class Comfly_Doubao_Seedance_2_0_Asset:
    """
    Create Seedance asset from Comfy IMAGE / VIDEO / AUDIO.
    Image → /v1/files (HTTPS URL); video/audio → same upload path as Seedance 2.0. assetType inferred; no url widget.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                # "apikey": ("STRING", {"default": ""}),
                "name": ("STRING", {"default": "", "multiline": False}),
                "image": ("IMAGE",),
                "video": (IO.VIDEO, {"tooltip": "参考视频；上传方式与 Seedance 2.0 主节点一致。"}),
                "audio": (IO.AUDIO, {"tooltip": "参考音频；上传方式与 Seedance 2.0 主节点一致。"}),
                "skip_error": ("BOOLEAN", {"default": False, "tooltip": "开启后，节点失败时不报错、按旧行为返回默认空结果；关闭时（默认）失败直接抛出错误。"})
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("asset_id", "status", "response")
    FUNCTION = "upload_asset"
    CATEGORY = "RunNode/Doubao"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 120
        self.poll_interval = 3
        self.max_wait_time = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def query_asset_status(self, asset_id):
        payload = {"assetId": asset_id}
        response = requests.post(
            f"{baseurl}/seedance/v3/assets/query",
            headers=self.get_headers(),
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def upload_asset(self, apikey="", name="", image=None, video=None, audio=None, skip_error=False):
        request_id = generate_request_id("asset_upload", "doubao")
        log_prepare("Seedance资产上传", request_id, "RunNode/Doubao-", "Seedance2Asset")
        rn_pbar = ProgressBar(request_id, "Seedance2Asset", streaming=True, task_type="资产上传", source="RunNode/Doubao-")
        rn_pbar.set_generating()

        if apikey and str(apikey).strip():
            self.api_key = str(apikey).strip()
        else:
            self.api_key = get_config().get("api_key", "")

        if not self.api_key:
            error_message = "API key not found in configuration file or environment variables."
            rn_pbar.error(error_message)
            log_error("配置缺失", request_id, error_message, "RunNode/Doubao-", "Seedance2Asset")
            if not skip_error:
                raise RuntimeError(f"[Comfly_Doubao_Seedance2_0_Asset] {error_message}")
            return ("", "", json.dumps({"error": error_message}, ensure_ascii=False))

        seed = Comfly_Doubao_Seedance_2_0()
        seed.api_key = self.api_key
        seed.timeout = self.timeout

        wired = []
        if image is not None:
            wired.append("image")
        if video is not None:
            wired.append("video")
        if audio is not None:
            wired.append("audio")
        if len(wired) > 1:
            log_backend(
                "doubao_seedance_asset_multiple_inputs",
                request_id=request_id,
                wired=",".join(wired),
            )

        try:
            _rn_start = time.perf_counter()
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(5)

            asset_type = None
            media_url = None
            if image is not None:
                asset_type = "Image"
                media_url = seed.upload_image_get_url(image)
            elif video is not None:
                asset_type = "Video"
                media_url = seed.upload_video_get_url(video)
            elif audio is not None:
                asset_type = "Audio"
                media_url = seed.upload_audio_get_url(audio)
            else:
                error_message = "Connect image, video, or audio."
                rn_pbar.error(error_message)
                log_error("参数缺失", request_id, error_message, "RunNode/Doubao-", "Seedance2Asset")
                if not skip_error:
                    raise RuntimeError(f"[Comfly_Doubao_Seedance2_0_Asset] {error_message}")
                return ("", "", json.dumps({"error": error_message}, ensure_ascii=False))

            if not media_url:
                error_message = "Could not obtain HTTPS URL for asset (upload failed or empty media)."
                rn_pbar.error(error_message)
                log_error("上传失败", request_id, error_message, "RunNode/Doubao-", "Seedance2Asset")
                if not skip_error:
                    raise RuntimeError(f"[Comfly_Doubao_Seedance2_0_Asset] {error_message}")
                return ("", "", json.dumps({"error": error_message}, ensure_ascii=False))

            display_name = (name or "").strip() or f"asset_{uuid.uuid4().hex[:12]}"

            log_backend(
                "doubao_seedance_asset_create_submit",
                request_id=request_id,
                asset_type=asset_type,
                name=display_name,
            )

            payload = {"url": media_url, "assetType": asset_type, "name": display_name}
            response = requests.post(
                f"{baseurl}/seedance/v3/assets/create",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout,
            )

            if response.status_code != 200:
                error_message = format_runnode_error(response)
                rn_pbar.error(error_message)
                log_error("API请求失败", request_id, error_message, "RunNode/Doubao-", "Seedance2Asset")
                if not skip_error:
                    raise RuntimeError(f"[Comfly_Doubao_Seedance2_0_Asset] {error_message}")
                return ("", "", json.dumps({"error": error_message}, ensure_ascii=False))

            result = response.json()
            if result.get("code") != 0:
                error_message = format_runnode_error(result)
                rn_pbar.error(error_message)
                log_error("API返回错误", request_id, error_message, "RunNode/Doubao-", "Seedance2Asset")
                if not skip_error:
                    raise RuntimeError(f"[Comfly_Doubao_Seedance2_0_Asset] {error_message}")
                return ("", "", json.dumps(result, indent=2, ensure_ascii=False))

            data = result.get("data", {}) if isinstance(result, dict) else {}
            asset_id = data.get("assetId", "")
            status = data.get("status", "")

            pbar.update_absolute(20)

            if status == "Active":
                response_text = json.dumps(result, indent=2, ensure_ascii=False)
                elapsed_ms = int((time.perf_counter() - _rn_start) * 1000)
                rn_pbar.done(char_count=len(response_text), elapsed_ms=elapsed_ms)
                log_complete(
                    "Seedance资产上传完成",
                    request_id,
                    "RunNode/Doubao-",
                    "Seedance2Asset",
                    asset_id=asset_id,
                    status=status,
                    char_count=len(response_text),
                    elapsed_ms=elapsed_ms,
                )
                return (asset_id, status, response_text)

            start_time = time.time()
            last_log_time = start_time
            while True:
                elapsed = time.time() - start_time
                if elapsed > self.max_wait_time:
                    error_message = f"Asset processing timeout after {elapsed:.1f}s. Last status: {status}"
                    rn_pbar.error(error_message)
                    log_error("生成超时", request_id, error_message, "RunNode/Doubao-", "Seedance2Asset")
                    if not skip_error:
                        raise RuntimeError(f"[Comfly_Doubao_Seedance2_0_Asset] {error_message}")
                    return ("", status, json.dumps({"error": error_message, "asset_id": asset_id, "last_status": status}, ensure_ascii=False))

                now = time.time()
                if now - last_log_time >= 15:
                    log_backend(
                        "doubao_seedance_asset_poll_heartbeat",
                        request_id=request_id,
                        asset_id=asset_id,
                        elapsed=int(now - start_time),
                        status=status,
                    )
                    last_log_time = now

                time.sleep(self.poll_interval)
                progress = min(90, 20 + int((elapsed / self.max_wait_time) * 70))
                pbar.update_absolute(progress)

                try:
                    query_result = self.query_asset_status(asset_id)
                    if query_result.get("code") != 0:
                        continue

                    query_data = query_result.get("data", {})
                    status = query_data.get("status", "")

                    if status == "Active":
                        response_text = json.dumps(query_result, indent=2, ensure_ascii=False)
                        elapsed_ms = int((time.perf_counter() - _rn_start) * 1000)
                        rn_pbar.done(char_count=len(response_text), elapsed_ms=elapsed_ms)
                        log_complete(
                            "Seedance资产上传完成",
                            request_id,
                            "RunNode/Doubao-",
                            "Seedance2Asset",
                            asset_id=asset_id,
                            status=status,
                            char_count=len(response_text),
                            elapsed_ms=elapsed_ms,
                        )
                        pbar.update_absolute(100)
                        return (asset_id, status, response_text)

                    if status in ("Failed", "Error", "Deleted"):
                        error_message = f"Asset processing failed with status: {status}"
                        rn_pbar.error(error_message)
                        log_error("生成失败", request_id, error_message, "RunNode/Doubao-", "Seedance2Asset")
                        if not skip_error:
                            raise RuntimeError(f"[Comfly_Doubao_Seedance2_0_Asset] {error_message}")
                        return ("", status, json.dumps(query_result, indent=2, ensure_ascii=False))

                except requests.exceptions.Timeout:
                    continue
                except Exception as e:
                    log_backend_exception(
                        "doubao_seedance_asset_poll_exception",
                        request_id=request_id,
                        asset_id=asset_id,
                        error=format_runnode_error(str(e)),
                    )
                    continue

        except Exception as e:
            error_message = f"Asset upload error: {format_runnode_error(str(e))}"
            rn_pbar.error(error_message)
            log_error("系统异常", request_id, error_message, "RunNode/Doubao-", "Seedance2Asset")
            if not skip_error:
                raise
            return ("", "", json.dumps({"error": error_message}, ensure_ascii=False))
