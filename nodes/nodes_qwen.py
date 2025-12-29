from ..comfly_config import *
from .__init__ import *


class Comfly_qwen_image:
    
    """
    A node that generates images using Qwen AI service
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "size": (["512x512", "1024x1024", "768x1024", "576x1024", "1024x768", "1024x576", "Custom"], {"default": "1024x768"}),
                "Custom_size": ("STRING", {"default": "Enter custom size (e.g. 1280x720)", "multiline": False}),
                "model": (["qwen-image"], {"default": "qwen-image"}),
                "num_images": ([1, 2, 3, 4], {"default": 1}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                # "api_key": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "num_inference_steps": ("INT", {"default": 30, "min": 2, "max": 50, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "guidance_scale": ("FLOAT", {"default": 2.5, "min": 0, "max": 20, "step": 0.5}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "output_format": (["jpeg", "png"], {"default": "png"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "response", "image_url")
    FUNCTION = "generate_image"
    CATEGORY = "RunNode/Qwen"
       
    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
 
    def generate_image(self, prompt, size, Custom_size, model, num_images=1,
                       api_key="", num_inference_steps=30, seed=0, guidance_scale=2.5, 
                       enable_safety_checker=True, negative_prompt="", output_format="png"):
        if api_key.strip():
            self.api_key = api_key
            # config = get_config()
            # config['api_key'] = api_key
            # save_config(config)
            
        try:
            if not self.api_key:
                error_message = "API key not found in Comflyapi.json"
                print(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, error_message, "")
                
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)
            
            actual_size = Custom_size if size == "Custom" else size

            if size == "Custom" and (Custom_size == "Enter custom size (e.g. 1280x720)" or "x" not in Custom_size):
                error_message = "Please enter a valid custom size in the format 'widthxheight' (e.g. 1280x720)"
                print(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, error_message, "")

            payload = {
                "prompt": prompt,
                "size": actual_size,  
                "model": model,
                "n": num_images,  
            }

            if num_inference_steps != 30:
                payload["num_inference_steps"] = num_inference_steps
                
            if seed != 0:
                payload["seed"] = seed
                
            if guidance_scale != 2.5:
                payload["guidance_scale"] = guidance_scale
                
            if not enable_safety_checker:
                payload["enable_safety_checker"] = enable_safety_checker
                
            if negative_prompt.strip():
                payload["negative_prompt"] = negative_prompt
                
            if output_format != "png":
                payload["output_format"] = output_format
            
            pbar.update_absolute(30)
            
            response = requests.post(
                f"{baseurl}/v1/images/generations", 
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(50)
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, error_message, "")
                
            result = response.json()

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            response_info = f"**Qwen Image Generation ({timestamp})**\n\n"
            response_info += f"Prompt: {prompt}\n"
            response_info += f"Model: {model}\n"
            response_info += f"Size: {actual_size}\n"
            response_info += f"Number of Images: {num_images}\n"
            response_info += f"Seed: {seed}\n\n"
            
            generated_images = []
            image_urls = []
            
            if "data" in result and result["data"]:
                for i, item in enumerate(result["data"]):
                    pbar.update_absolute(50 + (i+1) * 40 // len(result["data"]))
                    
                    if "b64_json" in item:
                        image_data = base64.b64decode(item["b64_json"])
                        generated_image = Image.open(BytesIO(image_data))
                        generated_tensor = pil2tensor(generated_image)
                        generated_images.append(generated_tensor)
                    elif "url" in item:
                        image_url = item["url"]
                        image_urls.append(image_url)
                        try:
                            img_response = requests.get(image_url, timeout=self.timeout)
                            img_response.raise_for_status()
                            generated_image = Image.open(BytesIO(img_response.content))
                            generated_tensor = pil2tensor(generated_image)
                            generated_images.append(generated_tensor)
                        except Exception as e:
                            print(f"Error downloading image from URL: {str(e)}")
            else:
                error_message = "No generated images in response"
                print(error_message)
                response_info += f"Error: {error_message}\n"
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, response_info, "")
                
            if generated_images:
                combined_tensor = torch.cat(generated_images, dim=0)
                
                pbar.update_absolute(100)
                return (combined_tensor, response_info, image_urls[0] if image_urls else "")
            else:
                error_message = "No images were successfully processed"
                print(error_message)
                response_info += f"Error: {error_message}\n"
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, response_info, "")
                
        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(error_message)
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            blank_tensor = pil2tensor(blank_image)
            return (blank_tensor, error_message, "")


class Comfly_qwen_image_edit:
    
    """
    A node that edits images using Qwen AI service
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "image": ("IMAGE",),
                "size": (["512x512", "1024x1024", "768x1024", "576x1024", "1024x768", "1024x576", "Custom"], {"default": "1024x768"}),
                "Custom_size": ("STRING", {"default": "Enter custom size (e.g. 1280x720)", "multiline": False}),
                "model": (["qwen-image-edit"], {"default": "qwen-image-edit"}),
            },
            "optional": {
                "apikey": ("STRING", {"default": ""}),
                # "apikey": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "num_inference_steps": ("INT", {"default": 30, "min": 2, "max": 50, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "guidance_scale": ("FLOAT", {"default": 4.0, "min": 0, "max": 20, "step": 0.5}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "output_format": (["jpeg", "png"], {"default": "png"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "acceleration": (["none", "regular", "high"], {"default": "none"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "response", "image_url")
    FUNCTION = "edit_image"
    CATEGORY = "RunNode/Qwen"
       
    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
 
    def edit_image(self, prompt, image, size, Custom_size, model,
                  apikey="", num_inference_steps=30, seed=0, guidance_scale=4.0, 
                  enable_safety_checker=True, negative_prompt="", output_format="png",
                  num_images=1, acceleration="none"):
        if apikey.strip():
            self.api_key = apikey
            # config = get_config()
            # config['api_key'] = apikey
            # save_config(config)
            
        try:
            if not self.api_key:
                error_message = "API key not found in Comflyapi.json"
                print(error_message)
                return (image, error_message, "")
                
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)
            
            actual_size = Custom_size if size == "Custom" else size

            if size == "Custom" and (Custom_size == "Enter custom size (e.g. 1280x720)" or "x" not in Custom_size):
                error_message = "Please enter a valid custom size in the format 'widthxheight' (e.g. 1280x720)"
                print(error_message)
                return (image, error_message, "")

            pil_image = tensor2pil(image)[0]

            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            buffered.seek(0) 

            files = {
                'image': ('image.png', buffered, 'image/png')
            }
            
            data = {
                "prompt": prompt,
                "size": actual_size,  
                "model": model,
                "n": str(num_images),
            }

            if num_inference_steps != 30:
                data["num_inference_steps"] = str(num_inference_steps)
                
            if seed != 0:
                data["seed"] = str(seed)
                
            if guidance_scale != 4.0:
                data["guidance_scale"] = str(guidance_scale)
                
            if not enable_safety_checker:
                data["enable_safety_checker"] = str(enable_safety_checker).lower()
                
            if negative_prompt.strip():
                data["negative_prompt"] = negative_prompt
                
            if output_format != "png":
                data["output_format"] = output_format
                
            if acceleration != "none":
                data["acceleration"] = acceleration
            
            pbar.update_absolute(30)

            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            response = requests.post(
                f"{baseurl}/v1/images/edits", 
                headers=headers,
                files=files,
                data=data,
                timeout=self.timeout
            )
            
            pbar.update_absolute(50)
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                return (image, error_message, "")
                
            result = response.json()

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            response_info = f"**Qwen Image Edit ({timestamp})**\n\n"
            response_info += f"Prompt: {prompt}\n"
            response_info += f"Model: {model}\n"
            response_info += f"Size: {actual_size}\n"
            response_info += f"Number of Images: {num_images}\n"
            response_info += f"Acceleration: {acceleration}\n"
            response_info += f"Seed: {seed}\n\n"
            
            edited_images = []
            image_urls = []
            
            if "data" in result and result["data"]:
                for i, item in enumerate(result["data"]):
                    pbar.update_absolute(50 + (i+1) * 40 // len(result["data"]))
                    
                    if "b64_json" in item:
                        image_data = base64.b64decode(item["b64_json"])
                        edited_image = Image.open(BytesIO(image_data))
                        edited_tensor = pil2tensor(edited_image)
                        edited_images.append(edited_tensor)
                    elif "url" in item:
                        image_url = item["url"]
                        image_urls.append(image_url)
                        try:
                            img_response = requests.get(image_url, timeout=self.timeout)
                            img_response.raise_for_status()
                            edited_image = Image.open(BytesIO(img_response.content))
                            edited_tensor = pil2tensor(edited_image)
                            edited_images.append(edited_tensor)
                        except Exception as e:
                            print(f"Error downloading image from URL: {str(e)}")
            else:
                error_message = "No edited images in response"
                print(error_message)
                response_info += f"Error: {error_message}\n"
                return (image, response_info, "")
                
            if edited_images:
                combined_tensor = torch.cat(edited_images, dim=0)
                
                pbar.update_absolute(100)
                return (combined_tensor, response_info, image_urls[0] if image_urls else "")
            else:
                error_message = "No images were successfully processed"
                print(error_message)
                response_info += f"Error: {error_message}\n"
                return (image, response_info, "")
                
        except Exception as e:
            error_message = f"Error in image editing: {str(e)}"
            print(error_message)
            return (image, error_message, "")


class Comfly_Z_image_turbo:
    """
    Comfly Z Image Turbo node
    Generates images using Z Image Turbo API
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["z-image-turbo"], {"default": "z-image-turbo"}),
                "size": (["512x512", "768x768", "1024x1024", "1280x720", "720x1280", "1536x1024", "1024x1536", "Custom"], {"default": "1024x1024"}),
                "output_format": (["jpeg", "png", "webp"], {"default": "jpeg"}),
            },
            "optional": {
                "custom_size": ("STRING", {"default": "1024x1024", "placeholder": "Enter custom size (e.g. 1280x720)"}),
                "apikey": ("STRING", {"default": ""}),
                # "apikey": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "guidance_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.04}),
                "num_inference_steps": ("INT", {"default": 8, "min": 1, "max": 50, "step": 1}),
                "output_quality": ("INT", {"default": 80, "min": 0, "max": 100, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "task_id", "response")
    FUNCTION = "generate_image"
    CATEGORY = "RunNode/Qwen"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def generate_image(self, prompt, model="z-image-turbo", size="1024x1024", output_format="jpg",
                      custom_size="1024x1024", apikey="", guidance_scale=0.0, num_inference_steps=8,
                      output_quality=80, seed=0):
        if apikey.strip():
            self.api_key = apikey
            # config = get_config()
            # config['api_key'] = apikey
            # save_config(config)
            
        if not self.api_key:
            error_message = "API key not found in Comflyapi.json"
            print(error_message)
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            blank_tensor = pil2tensor(blank_image)
            return (blank_tensor, "", error_message)
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            actual_size = custom_size if size == "Custom" else size

            if size == "Custom":
                if "x" not in custom_size:
                    error_message = "Custom size must be in format 'widthxheight' (e.g. 1280x720)"
                    print(error_message)
                    blank_image = Image.new('RGB', (1024, 1024), color='white')
                    blank_tensor = pil2tensor(blank_image)
                    return (blank_tensor, "", error_message)
                
                try:
                    width, height = map(int, custom_size.split('x'))
                    if width < 64 or width > 2048 or height < 64 or height > 2048:
                        error_message = "Width and height must be between 64 and 2048"
                        print(error_message)
                        blank_image = Image.new('RGB', (1024, 1024), color='white')
                        blank_tensor = pil2tensor(blank_image)
                        return (blank_tensor, "", error_message)
                except ValueError:
                    error_message = "Invalid custom size format. Use 'widthxheight' (e.g. 1280x720)"
                    print(error_message)
                    blank_image = Image.new('RGB', (1024, 1024), color='white')
                    blank_tensor = pil2tensor(blank_image)
                    return (blank_tensor, "", error_message)

            try:
                width, height = map(int, actual_size.split('x'))
            except:
                width, height = 1024, 1024

            payload = {
                "prompt": prompt,
                "model": model,
                "size": actual_size,
                "output_format": output_format,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "output_quality": output_quality
            }
            
            if seed > 0:
                payload["seed"] = seed
            
            pbar.update_absolute(30)

            response = requests.post(
                f"{baseurl}/v1/images/generations",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(50)
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                blank_image = Image.new('RGB', (width, height), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, "", error_message)
                
            result = response.json()

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            response_info = f"**Z Image Turbo Generation ({timestamp})**\n\n"
            response_info += f"Prompt: {prompt}\n"
            response_info += f"Model: {model}\n"
            response_info += f"Size: {actual_size}\n"
            response_info += f"Output Format: {output_format}\n"
            response_info += f"Guidance Scale: {guidance_scale}\n"
            response_info += f"Steps: {num_inference_steps}\n"
            response_info += f"Output Quality: {output_quality}\n"
            response_info += f"Seed: {seed if seed > 0 else 'auto'}\n\n"
            
            image_url = ""
            generated_image = None
            
            if "data" in result and result["data"]:
                item = result["data"][0]
                pbar.update_absolute(70)
                
                if "b64_json" in item:
                    image_data = base64.b64decode(item["b64_json"])
                    generated_image = Image.open(BytesIO(image_data))
                elif "url" in item:
                    image_url = item["url"]
                    response_info += f"Image URL: {image_url}\n"
                    try:
                        img_response = requests.get(image_url, timeout=self.timeout)
                        img_response.raise_for_status()
                        generated_image = Image.open(BytesIO(img_response.content))
                    except Exception as e:
                        print(f"Error downloading image from URL: {str(e)}")
                        response_info += f"Error: {str(e)}\n"
            else:
                error_message = "No image data in response"
                print(error_message)
                response_info += f"Error: {error_message}\n"
                blank_image = Image.new('RGB', (width, height), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, "", response_info)
            
            pbar.update_absolute(90)
            
            if generated_image:
                generated_tensor = pil2tensor(generated_image)
                pbar.update_absolute(100)
                return (generated_tensor, image_url, response_info)
            else:
                error_message = "Failed to process image"
                print(error_message)
                response_info += f"Error: {error_message}\n"
                blank_image = Image.new('RGB', (width, height), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, "", response_info)
                
        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            blank_tensor = pil2tensor(blank_image)
            return (blank_tensor, "", error_message)



class Comfly_wan2_6_API:
    def __init__(self):
        self.timeout = 300
        self.api_key = get_config().get('api_key', '')

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "一幅都市奇幻艺术的场景。一个充满动感的涂鸦艺术角色。一个由喷漆所画成的少年,正从一面混凝土墙上活过来。他一边用极快的语速演唱一首英文rap,一边摆着一个经典的、充满活力的说唱歌手姿势。场景设定在夜晚一个充满都市感的铁路桥下。灯光来自一盏孤零零的街灯,营造出电影般的氛围,充满高能量和惊人的细节。视频的音频部分完全由他的rap构成,没有其他对话或杂音。"}),
                "resolution": (["1080P", "720P"], {"default": "1080P"}),
                "duration": ([5, 10, 15], {"default": 5}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "image": ("IMAGE",),
                "audio_url": ("STRING", {"default": ""}),
                "prompt_extend": ("BOOLEAN", {"default": True}),
                "shot_type": (["single", "multi"], {"default": "multi"}),
                "audio_enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "task_id")
    FUNCTION = "generate_video"
    CATEGORY = "zhenzhen/wanx"

    def convert_image_to_base64(self, image_tensor):
        """Convert image tensor to base64 data URL - exactly as wan.py"""
        try:
            # Convert tensor to PIL Image
            if isinstance(image_tensor, torch.Tensor):
                # Handle batch dimension
                if len(image_tensor.shape) == 4:
                    image_tensor = image_tensor[0]
                
                # Convert from [C, H, W] to [H, W, C] if needed
                if image_tensor.shape[0] == 3:
                    image_tensor = image_tensor.permute(1, 2, 0)
                
                # Convert to numpy and ensure correct range
                image_np = image_tensor.cpu().numpy()
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype('uint8')
                
                image = Image.fromarray(image_np)
            else:
                image = image_tensor

            # Get original image info
            original_size = image.size
            print(f"[Zhenzhen_WanVideo INFO] Original image size: {original_size[0]}x{original_size[1]}")
            
            # Optimize image size to reduce Base64 length
            max_dimension = 1536
            
            if max(original_size) > max_dimension:
                # Calculate new size while maintaining aspect ratio
                ratio = max_dimension / max(original_size)
                new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                print(f"[Zhenzhen_WanVideo INFO] Resized image to: {new_size[0]}x{new_size[1]}")

            # Try JPEG format with quality optimization
            formats_to_try = [
                ('JPEG', 'image/jpeg', {'quality': 75, 'optimize': True}),
                ('JPEG', 'image/jpeg', {'quality': 60, 'optimize': True}),
                ('PNG', 'image/png', {'optimize': True})
            ]
            
            best_result = None
            smallest_size = float('inf')
            
            for format_name, mime_type, save_kwargs in formats_to_try:
                try:
                    img_byte_arr = BytesIO()
                    
                    # Handle JPEG format (doesn't support transparency)
                    if format_name == 'JPEG' and image.mode in ('RGBA', 'LA'):
                        # Convert RGBA to RGB with white background
                        jpeg_image = Image.new('RGB', image.size, 'white')
                        if image.mode == 'RGBA':
                            jpeg_image.paste(image, mask=image.split()[-1])
                        else:
                            jpeg_image.paste(image)
                        jpeg_image.save(img_byte_arr, format=format_name, **save_kwargs)
                    else:
                        image.save(img_byte_arr, format=format_name, **save_kwargs)
                    
                    img_byte_arr.seek(0)
                    image_bytes = img_byte_arr.read()
                    
                    # Check if this format gives smaller result
                    if len(image_bytes) < smallest_size:
                        smallest_size = len(image_bytes)
                        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                        best_result = f"data:{mime_type};base64,{image_base64}"
                        
                        # Calculate base64 size in MB
                        base64_size_mb = len(image_base64) / (1024 * 1024)
                        print(f"[Zhenzhen_WanVideo INFO] {format_name} format: {base64_size_mb:.2f}MB base64")
                        
                        # If JPEG is small enough, use it
                        if format_name == 'JPEG' and base64_size_mb < 2.0:
                            break
                            
                except Exception as format_error:
                    print(f"[Zhenzhen_WanVideo WARNING] Failed to save as {format_name}: {format_error}")
                    continue
            
            if best_result:
                final_size_mb = len(best_result.split(',')[1]) / (1024 * 1024)
                print(f"[Zhenzhen_WanVideo INFO] Final base64 size: {final_size_mb:.2f}MB")
                
                if final_size_mb > 3.0:
                    print(f"[Zhenzhen_WanVideo WARNING] Base64 data is large ({final_size_mb:.2f}MB), may cause API issues")
                
                return best_result
            else:
                raise Exception("Failed to encode image in any supported format")
                
        except Exception as e:
            print(f"[Zhenzhen_WanVideo ERROR] Image base64 conversion error: {str(e)}")
            return None

    def generate_video(self, prompt, api_key, resolution, duration, image=None, audio_url="", prompt_extend=True, shot_type="multi", audio_enabled=True):
        if api_key.strip():
            self.api_key = api_key
            # config = get_config()
            # config['api_key'] = api_key
            # save_config(config)
            
        if not self.api_key:
            error_msg = "API key not found. Please provide an API key."
            print(error_msg)
            return (EmptyVideoAdapter(), error_msg, "")
        
        try:
            # Validate prompt
            if not prompt or prompt.strip() == "":
                raise ValueError("Prompt cannot be empty")
            
            if len(prompt) > 1500:
                raise ValueError(f"Prompt too long ({len(prompt)} chars). Max 1500 characters")
            
            # Convert image to base64 (exactly as wan.py)
            image_url = None
            if image is not None:
                image_url = self.convert_image_to_base64(image)
                if not image_url:
                    raise ValueError("Failed to convert image to base64")
                print(f"[Zhenzhen_Wan26_I2V INFO] Image converted to base64 successfully")
            
            # Prepare request body (exactly matching wan.py structure)
            request_body = {
                "model": "wan2.6-i2v",
                "input": {
                    "prompt": prompt
                },
                "parameters": {
                    "prompt_extend": prompt_extend,
                    "resolution": resolution,
                    "duration": duration
                }
            }
            
            # Add image URL (inside input object)
            if image_url:
                request_body["input"]["img_url"] = image_url
            
            # Add shot_type
            if shot_type:
                request_body["parameters"]["shot_type"] = shot_type
            
            # Add audio
            if audio_url and audio_url.strip():
                request_body["input"]["audio_url"] = audio_url
                request_body["parameters"]["audio"] = True
            else:
                request_body["parameters"]["audio"] = audio_enabled
            
            # Submit task (exactly as wan.py)
            cfg = get_config()
            base_url = cfg.get('base_url', '').strip() or "https://dashscope.aliyuncs.com"
            url_primary = f"{base_url}/api/v1/services/aigc/video-generation/video-synthesis"
            url_fallback = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'X-DashScope-Async': 'enable'
            }
            
            # Log with truncated base64
            log_body = copy.deepcopy(request_body)
            if "input" in log_body and "img_url" in log_body["input"]:
                img_url = log_body["input"]["img_url"]
                if img_url and img_url.startswith("data:image/"):
                    base64_part = img_url.split(",", 1)
                    if len(base64_part) > 1:
                        truncated = base64_part[0] + "," + base64_part[1][:50] + f"...[{len(base64_part[1])} chars]"
                        log_body["input"]["img_url"] = truncated
            
            print(f"[Zhenzhen_WanVideo INFO] Request body: {json.dumps(log_body, indent=2, ensure_ascii=False)}")
            print(f"[Zhenzhen_WanVideo INFO] Sending request to: {url_primary}")
            
            response = None
            try:
                response = requests.post(url_primary, headers=headers, json=request_body, timeout=60)
            except Exception as e:
                print(f"[Zhenzhen_WanVideo WARNING] Primary request error: {str(e)}")
                response = None
            
            if (response is None or response.status_code != 200) and (base_url.rstrip('/') != "https://dashscope.aliyuncs.com"):
                print(f"[Zhenzhen_WanVideo INFO] Fallback to DashScope: {url_fallback}")
                try:
                    response = requests.post(url_fallback, headers=headers, json=request_body, timeout=60)
                except Exception as e:
                    print(f"[Zhenzhen_WanVideo ERROR] Fallback request error: {str(e)}")
                    response = None
            
            if response is not None and response.status_code == 200:
                result = response.json()
                task_id = result.get('output', {}).get('task_id')
                if task_id:
                    print(f"[Zhenzhen_WanVideo INFO] Task created. Task ID: {task_id}")
                    
                    # Poll for result
                    video_url = self.poll_task_status(task_id)
                    
                    if video_url:
                        print(f"[Zhenzhen_WanVideo INFO] Downloading video from: {video_url}")
                        video_adapter = ComflyVideoAdapter(video_url)
                        return (video_adapter, video_url, task_id)
                    else:
                        error_msg = "Failed to generate video"
                        print(error_msg)
                        return (EmptyVideoAdapter(), error_msg, task_id)
                else:
                    print(f"[Zhenzhen_WanVideo ERROR] No task ID: {result}")
                    return (EmptyVideoAdapter(), "No task ID in response", "")
            else:
                # Error handling
                status_code = response.status_code if response is not None else -1
                error_message = "Unknown error"
                
                try:
                    if response is not None and response.text:
                        response_json = response.json()
                        if isinstance(response_json, dict):
                            if 'message' in response_json:
                                error_message = str(response_json['message'])
                            elif 'error' in response_json:
                                error_obj = response_json['error']
                                if isinstance(error_obj, dict) and 'message' in error_obj:
                                    error_message = str(error_obj['message'])
                                else:
                                    error_message = str(error_obj)
                            
                            if len(error_message) > 200:
                                error_message = error_message[:200] + "..."
                except:
                    error_message = response.text[:200] if (response is not None and response.text) else "No details"
                
                print(f"[Zhenzhen_WanVideo ERROR] API error ({status_code}): {error_message}")
                return (EmptyVideoAdapter(), f"API error: {status_code} - {error_message}", "")
                
        except Exception as e:
            print(f"[Zhenzhen_WanVideo ERROR] Video generation error: {str(e)}")
            return (EmptyVideoAdapter(), str(e), "")

    def poll_task_status(self, task_id):
        """Poll task status (exactly as wan.py)"""
        cfg = get_config()
        base_url = cfg.get('base_url', '').strip() or "https://dashscope.aliyuncs.com"
        query_url_primary = f"{base_url}/api/v1/tasks/{task_id}"
        query_url_fallback = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
        headers = {'Authorization': f'Bearer {self.api_key}'}
        
        max_poll_time = 300
        poll_interval = 10
        start_time = time.time()
        
        while time.time() - start_time < max_poll_time:
            try:
                print(f"[Zhenzhen_WanVideo INFO] Querying task: {query_url_primary}")
                response = None
                try:
                    response = requests.get(query_url_primary, headers=headers, timeout=30)
                except Exception as e:
                    print(f"[Zhenzhen_WanVideo WARNING] Primary query error: {str(e)}")
                    response = None
                
                if (response is None or response.status_code != 200) and (base_url.rstrip('/') != "https://dashscope.aliyuncs.com"):
                    print(f"[Zhenzhen_WanVideo INFO] Fallback querying task: {query_url_fallback}")
                    try:
                        response = requests.get(query_url_fallback, headers=headers, timeout=30)
                    except Exception as e:
                        print(f"[Zhenzhen_WanVideo ERROR] Fallback query error: {str(e)}")
                        response = None
                
                if response is not None and response.status_code == 200:
                    result = response.json()
                    task_status = result.get('output', {}).get('task_status')
                    
                    if task_status == 'SUCCEEDED':
                        video_url = result.get('output', {}).get('video_url')
                        if video_url:
                            print(f"[Zhenzhen_WanVideo INFO] Video ready: {video_url}")
                            return video_url
                        else:
                            print(f"[Zhenzhen_WanVideo ERROR] No video URL: {result}")
                            return None
                    elif task_status == 'FAILED':
                        error_msg = result.get('output', {}).get('message', 'Unknown error')
                        print(f"[Zhenzhen_WanVideo ERROR] Task failed: {error_msg}")
                        return None
                    elif task_status in ['PENDING', 'RUNNING']:
                        elapsed = time.time() - start_time
                        remaining = max_poll_time - elapsed
                        print(f"[Zhenzhen_WanVideo INFO] Status: {task_status} ({elapsed:.1f}s elapsed, {remaining:.1f}s remaining)")
                        time.sleep(poll_interval)
                        continue
                    else:
                        print(f"[Zhenzhen_WanVideo WARNING] Unknown status: {task_status}")
                        time.sleep(poll_interval)
                        continue
                else:
                    status = response.status_code if response is not None else -1
                    print(f"[Zhenzhen_WanVideo ERROR] Query failed: {status}")
                    return None
                    
            except Exception as e:
                print(f"[Zhenzhen_WanVideo ERROR] Query error: {str(e)}")
                return None
        
        print("[Zhenzhen_WanVideo ERROR] Polling timeout")
        return None
