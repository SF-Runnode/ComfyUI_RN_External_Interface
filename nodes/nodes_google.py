from ..comfly_config import *
from .__init__ import *


class ComflyGeminiAPI:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["gemini-2.0-flash-exp-image"], {"default": "gemini-2.0-flash-exp-image"}),
                "resolution": (
                    [
                        "512x512", 
                        "768x768", 
                        "1024x1024", 
                        "1280x1280", 
                        "1536x1536", 
                        "2048x2048",
                        "object_image size",
                        "subject_image size",
                        "scene_image size"
                    ], 
                    {"default": "1024x1024"}
                ),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "timeout": ("INT", {"default": 120, "min": 10, "max": 600, "step": 10}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                # "api_key": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "object_image": ("IMAGE",),  
                "subject_image": ("IMAGE",),
                "scene_image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("generated_images", "response", "image_url")
    FUNCTION = "process"
    CATEGORY = "RunNode/Google"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = None  

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def image_to_base64(self, image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def extract_image_urls(self, response_text):
        image_pattern = r'!\[.*?\]\((.*?)\)'
        matches = re.findall(image_pattern, response_text)

        if not matches:
            url_pattern = r'https?://\S+\.(?:jpg|jpeg|png|gif|webp)'
            matches = re.findall(url_pattern, response_text)
            
        return matches if matches else []

    def resize_to_target_size(self, image, target_size):
        """Resize image to target size while preserving aspect ratio with padding"""

        img_width, img_height = image.size
        target_width, target_height = target_size

        width_ratio = target_width / img_width
        height_ratio = target_height / img_height
        scale = min(width_ratio, height_ratio)

        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        resized_img = image.resize((new_width, new_height), Image.LANCZOS)

        new_img = Image.new("RGB", (target_width, target_height), (255, 255, 255))

        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
 
        new_img.paste(resized_img, (paste_x, paste_y))
        
        return new_img

    def parse_resolution(self, resolution_str):
        """Parse resolution string (e.g., '1024x1024') to width and height"""
        width, height = map(int, resolution_str.split('x'))
        return (width, height)

    def process(self, prompt, model, resolution, num_images, temperature, top_p, seed, timeout=120, 
                object_image=None, subject_image=None, scene_image=None, api_key=""):

        if api_key.strip():
            self.api_key = api_key
            # config = get_config()
            # config['api_key'] = api_key
            # save_config(config)
        else:
            self.api_key = get_config().get('api_key', '')

        self.timeout = timeout
        
        try:

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            target_size = None

            if resolution == "object_image size" and object_image is not None:
                pil_image = tensor2pil(object_image)[0]
                target_size = pil_image.size
            elif resolution == "subject_image size" and subject_image is not None:
                pil_image = tensor2pil(subject_image)[0]
                target_size = pil_image.size
            elif resolution == "scene_image size" and scene_image is not None:
                pil_image = tensor2pil(scene_image)[0]
                target_size = pil_image.size
            else:
                target_size = self.parse_resolution(resolution)

            has_images = object_image is not None or subject_image is not None or scene_image is not None

            content = []

            if has_images:
                content.append({"type": "text", "text": prompt})
 
                image_descriptions = {
                    "object_image": "an object or item",
                    "subject_image": "a subject or character",
                    "scene_image": "a scene or environment"
                }
 
                for image_var, image_tensor in [("object_image", object_image), 
                                             ("subject_image", subject_image), 
                                             ("scene_image", scene_image)]:
                    if image_tensor is not None:
                        pil_image = tensor2pil(image_tensor)[0]
                        image_base64 = self.image_to_base64(pil_image)
                        content.append({
                            "type": "image_url", 
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                        })
            else:

                dimensions = f"{target_size[0]}x{target_size[1]}"
                aspect_ratio = "1:1" if target_size[0] == target_size[1] else f"{target_size[0]}:{target_size[1]}"
                
                if num_images == 1:
                    enhanced_prompt = f"Generate a high-quality, detailed image with dimensions {dimensions} and aspect ratio {aspect_ratio}. Based on this description: {prompt}"
                else:
                    enhanced_prompt = f"Generate {num_images} DIFFERENT high-quality images with VARIED content, each with unique and distinct visual elements, all having the exact same dimensions of {dimensions} and aspect ratio {aspect_ratio}. Important: make sure each image has different content but maintains the same technical dimensions. Based on this description: {prompt}"
                
                content.append({"type": "text", "text": enhanced_prompt})

            messages = [{
                "role": "user",
                "content": content
            }]

            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed if seed > 0 else None,
                "max_tokens": 8192
            }

            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            try:
                response = requests.post(
                    f"{baseurl}/v1/chat/completions",
                    headers=self.get_headers(),
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.Timeout:
                raise TimeoutError(f"API request timed out after {self.timeout} seconds")
            except requests.exceptions.RequestException as e:
                raise Exception(f"API request failed: {str(e)}")
            
            pbar.update_absolute(40)

            response_text = result["choices"][0]["message"]["content"]

            formatted_response = f"**User prompt**: {prompt}\n\n**Response** ({timestamp}):\n{response_text}"

            image_urls = self.extract_image_urls(response_text)
            
            if image_urls:
                try:
                    images = []
                    first_image_url = ""  
                    
                    for i, url in enumerate(image_urls):
                        pbar.update_absolute(40 + (i+1) * 50 // len(image_urls))
                        
                        if i == 0:
                            first_image_url = url  
                        
                        try:
                            img_response = requests.get(url, timeout=self.timeout)
                            img_response.raise_for_status()
                            pil_image = Image.open(BytesIO(img_response.content))

                            resized_image = self.resize_to_target_size(pil_image, target_size)

                            img_tensor = pil2tensor(resized_image)
                            images.append(img_tensor)
                            
                        except Exception as img_error:
                            print(f"Error processing image URL {i+1}: {str(img_error)}")
                            continue
                    
                    if images:
                        try:
                            combined_tensor = torch.cat(images, dim=0)
                        except RuntimeError:
                            print("Warning: Images have different sizes, returning first image")
                            combined_tensor = images[0] if images else None
                            
                        pbar.update_absolute(100)
                        return (combined_tensor, formatted_response, first_image_url)
                    else:
                        raise Exception("No images could be processed successfully")
                    
                except Exception as e:
                    print(f"Error processing image URLs: {str(e)}")

            pbar.update_absolute(100)

            reference_image = None
            if object_image is not None:
                reference_image = object_image
            elif subject_image is not None:
                reference_image = subject_image
            elif scene_image is not None:
                reference_image = scene_image
                
            if reference_image is not None:
                return (reference_image, formatted_response, "")
            else:
                default_image = Image.new('RGB', target_size, color='white')
                default_tensor = pil2tensor(default_image)
                return (default_tensor, formatted_response, "")
            
        except TimeoutError as e:
            error_message = f"API timeout error: {str(e)}"
            print(error_message)
            return self.handle_error(object_image, subject_image, scene_image, error_message, resolution)
            
        except Exception as e:
            error_message = f"Error calling Gemini API: {str(e)}"
            print(error_message)
            return self.handle_error(object_image, subject_image, scene_image, error_message, resolution)
    
    def handle_error(self, object_image, subject_image, scene_image, error_message, resolution="1024x1024"):
        """Handle errors with appropriate image output"""
        if object_image is not None:
            return (object_image, error_message, "")
        elif subject_image is not None:
            return (subject_image, error_message, "")
        elif scene_image is not None:
            return (scene_image, error_message, "")
        else:
            if resolution in ["object_image size", "subject_image size", "scene_image size"]:
                target_size = (1024, 1024)  
            else:
                target_size = self.parse_resolution(resolution)
                
            default_image = Image.new('RGB', target_size, color='white')
            default_tensor = pil2tensor(default_image)
            return (default_tensor, error_message, "")


class Comfly_Googel_Veo3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["veo3", "veo3-fast", "veo3-pro", "veo3-fast-frames", "veo3-pro-frames", "veo3.1", "veo3.1-pro", "veo3.1-components"], {"default": "veo3"}),
                "enhance_prompt": ("BOOLEAN", {"default": False}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
            },
            "optional": {
                "apikey": ("STRING", {"default": ""}),
                # "apikey": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "enable_upsample": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "response")
    FUNCTION = "generate_video"
    CATEGORY = "RunNode/Google"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = None  

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
    
    def generate_video(self, prompt, model="veo3", enhance_prompt=False, aspect_ratio="16:9", apikey="", 
                      image1=None, image2=None, image3=None, seed=0, enable_upsample=False):
        request_id = generate_request_id("video_gen", "google")
        log_prepare("视频生成", request_id, "RunNode/Google-", "Google", model_name=model)
        rn_pbar = ProgressBar(request_id, "Google", streaming=True, task_type="视频生成", source="RunNode/Google-")
        if apikey.strip():
            self.api_key = apikey
            # config = get_config()
            # config['api_key'] = apikey
            # save_config(config)
        else:
            self.api_key = get_config().get('api_key', '')
            
        if not self.api_key:
            error_response = {"code": "error", "message": "API key not found in Comflyapi.json"}
            return ("", "", json.dumps(error_response))
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            has_images = any(img is not None for img in [image1, image2, image3])
 
            payload = {
                "prompt": prompt,
                "model": model,
                "enhance_prompt": enhance_prompt
            }
 
            if seed > 0:
                payload["seed"] = seed

            if model in ["veo3", "veo3-fast", "veo3-pro", "veo3.1", "veo3.1-pro"] and aspect_ratio:
                payload["aspect_ratio"] = aspect_ratio

            if model in ["veo3", "veo3-fast", "veo3-pro", "veo3.1", "veo3.1-pro"] and enable_upsample:
                payload["enable_upsample"] = enable_upsample

            if has_images:
                images_base64 = []
                for img in [image1, image2, image3]:
                    if img is not None:
                        batch_size = img.shape[0]
                        for i in range(batch_size):
                            single_image = img[i:i+1]
                            image_base64 = self.image_to_base64(single_image)
                            if image_base64:
                                images_base64.append(f"data:image/png;base64,{image_base64}")
                
                if images_base64:
                    payload["images"] = images_base64

            response = requests.post(
                f"{baseurl}/google/v1/models/veo/videos",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                rn_pbar.error(error_message)
                return ("", "", json.dumps({"code": "error", "message": error_message}))
                
            result = response.json()
            
            if result.get("code") != "success":
                error_message = f"API returned error: {result.get('message', 'Unknown error')}"
                rn_pbar.error(error_message)
                return ("", "", json.dumps({"code": "error", "message": error_message}))
                
            task_id = result.get("data")
            if not task_id:
                error_message = "No task ID returned from API"
                rn_pbar.error(error_message)
                return ("", "", json.dumps({"code": "error", "message": error_message}))
            
            pbar.update_absolute(30)

            max_attempts = 60  
            attempts = 0
            video_url = None
            
            while attempts < max_attempts:
                time.sleep(10)
                attempts += 1
                
                try:
                    status_response = requests.get(
                        f"{baseurl}/google/v1/tasks/{task_id}",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )
                    
                    if status_response.status_code != 200:
                        continue
                        
                    status_result = status_response.json()
                    
                    if status_result.get("code") != "success":
                        continue
                    
                    data = status_result.get("data", {})
                    status = data.get("status", "")
                    progress = data.get("progress", "0%")
                    
                    try:
                        if progress.endswith('%'):
                            progress_num = int(progress.rstrip('%'))
                            pbar_value = min(90, 30 + progress_num * 60 / 100)
                            pbar.update_absolute(pbar_value)
                    except (ValueError, AttributeError):
                        progress_value = min(80, 30 + (attempts * 50 // max_attempts))
                        pbar.update_absolute(progress_value)
                    
                    if status == "SUCCESS":
                        if "data" in data and "video_url" in data["data"]:
                            video_url = data["data"]["video_url"]
                            break
                    elif status == "FAILURE":
                        fail_reason = data.get("fail_reason", "Unknown error")
                        error_message = f"Video generation failed: {fail_reason}"
                        rn_pbar.error(error_message)
                        return ("", "", json.dumps({"code": "error", "message": error_message}))
                        
                except Exception as e:
                    rn_pbar.error(f"Error checking generation status: {str(e)}")
            
            if not video_url:
                error_message = "Failed to retrieve video URL after multiple attempts"
                rn_pbar.error(error_message)
                return ("", "", json.dumps({"code": "error", "message": error_message}))
            
            if video_url:
                pbar.update_absolute(95)
                
                response_data = {
                    "code": "success",
                    "task_id": task_id,
                    "prompt": prompt,
                    "model": model,
                    "enhance_prompt": enhance_prompt,
                    "aspect_ratio": aspect_ratio if model in ["veo3", "veo3-fast", "veo3-pro"] else "default",
                    "enable_upsample": enable_upsample if model in ["veo3", "veo3-fast", "veo3-pro"] else False,
                    "video_url": video_url,
                    "images_count": len([img for img in [image1, image2, image3] if img is not None])
                }
                
                video_adapter = ComflyVideoAdapter(video_url)
                rn_pbar.done(char_count=len(json.dumps(response_data)))
                return (video_adapter, video_url, json.dumps(response_data))
            
        except Exception as e:
            error_message = f"Error generating video: {str(e)}"
            rn_pbar.error(error_message)
            return ("", "", json.dumps({"code": "error", "message": error_message}))


class Comfly_nano_banana:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "model": (["nano-banana-2","gemini-3-pro-image-preview", "gemini-2.5-flash-image", "nano-banana", "nano-banana-hd", "gemini-2.5-flash-image-preview"], {"default": "nano-banana"}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "apikey": ("STRING", {"default": ""}),
                # "apikey": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "max_tokens": ("INT", {"default": 32768, "min": 1, "max": 32768})
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "response", "image_url")
    FUNCTION = "process"
    CATEGORY = "RunNode/Google"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = None  

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

    def send_request_streaming(self, payload):
        """Send a streaming request to the API"""
        full_response = ""
        session = requests.Session()
        
        try:
            response = session.post(
                f"{baseurl}/v1/chat/completions",
                headers=self.get_headers(),
                json=payload, 
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8').strip()
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
            
            return full_response
            
        except requests.exceptions.Timeout:
            raise TimeoutError(f"API request timed out after {self.timeout} seconds")
        except Exception as e:
            raise Exception(f"Error in streaming response: {str(e)}")

    def process(self, text, model="gemini-2.5-flash-image-preview", 
                image1=None, image2=None, image3=None, image4=None,
                temperature=1.0, top_p=0.95, apikey="", seed=0, max_tokens=32768):
        if apikey.strip():
            self.api_key = apikey
            # config = get_config()
            # config['api_key'] = apikey
            # save_config(config)
        else:
            self.api_key = get_config().get('api_key', '')

        default_image = None
        for img in [image1, image2, image3, image4]:
            if img is not None:
                default_image = img
                break

        if default_image is None:
            blank_image = Image.new('RGB', (512, 512), color='white')
            default_image = pil2tensor(blank_image)

        try:
            if not self.api_key:
                return (default_image, "API key not provided. Please set your API key.", "")

            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            content = [{"type": "text", "text": text}]

            images_added = 0
            for idx, img in enumerate([image1, image2, image3, image4], 1):
                if img is not None:
                    batch_size = img.shape[0]
                    print(f"Processing image{idx} with {batch_size} batch size")
                    
                    for i in range(batch_size):
                        single_image = img[i:i+1]
                        image_base64 = self.image_to_base64(single_image)
                        if image_base64:
                            content.append({
                                "type": "image_url", 
                                "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                            })
                            images_added += 1

            print(f"Total of {images_added} images added to the request")

            messages = [{
                "role": "user",
                "content": content
            }]

            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "stream": True 
            }

            if seed > 0:
                payload["seed"] = seed

            pbar.update_absolute(30)

            try:
                response_text = self.send_request_streaming(payload)
                pbar.update_absolute(70)
            except Exception as e:
                error_message = f"API Error: {str(e)}"
                print(error_message)
                return (default_image, error_message, "")

            base64_pattern = r'data:image\/[^;]+;base64,([A-Za-z0-9+/=]+)'
            base64_matches = re.findall(base64_pattern, response_text)
            
            if base64_matches:
                try:
                    image_data = base64.b64decode(base64_matches[0])
                    generated_image = Image.open(BytesIO(image_data))
                    generated_tensor = pil2tensor(generated_image)
                    
                    pbar.update_absolute(100)
                    return (generated_tensor, response_text, f"data:image/png;base64,{base64_matches[0]}")
                except Exception as e:
                    print(f"Error processing base64 image data: {str(e)}")

            image_pattern = r'!\[.*?\]\((.*?)\)'
            matches = re.findall(image_pattern, response_text)
            
            if not matches:
                url_pattern = r'https?://\S+\.(?:jpg|jpeg|png|gif|webp)'
                matches = re.findall(url_pattern, response_text)
            
            if not matches:
                all_urls_pattern = r'https?://\S+'
                matches = re.findall(all_urls_pattern, response_text)
                
            if matches:
                image_url = matches[0]
                try:
                    img_response = requests.get(image_url, timeout=self.timeout)
                    img_response.raise_for_status()
                    
                    generated_image = Image.open(BytesIO(img_response.content))
                    generated_tensor = pil2tensor(generated_image)
                    
                    pbar.update_absolute(100)
                    return (generated_tensor, response_text, image_url)
                except Exception as e:
                    print(f"Error downloading image: {str(e)}")
                    return (default_image, f"{response_text}\n\nError downloading image: {str(e)}", image_url)
            else:
                pbar.update_absolute(100)
                return (default_image, response_text, "")
                
        except Exception as e:
            error_message = f"Error processing request: {str(e)}"
            print(error_message)
            return (default_image, error_message, "")


class Comfly_nano_banana_fal:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["nano-banana", "nano-banana/edit"], {"default": "nano-banana/edit"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "image_way": (["image", "image_url"], {"default": "image_url"})
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "apikey": ("STRING", {"default": ""})
                # "apikey": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "response")
    FUNCTION = "process"
    CATEGORY = "RunNode/Google"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = None  

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
        return f"data:image/png;base64,{base64_str}"

    def upload_image_to_get_url(self, image_tensor):
        """Upload image to files API and get URL"""
        if image_tensor is None:
            return None
            
        try:
            pil_image = tensor2pil(image_tensor)[0]
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            file_content = buffered.getvalue()
            
            files = {'file': ('image.png', file_content, 'image/png')}
            
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.post(
                f"{baseurl}/v1/files",
                headers=headers,
                files=files,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            if 'url' in result:
                return result['url']
            else:
                print(f"Unexpected response from file upload API: {result}")
                return None
                
        except Exception as e:
            print(f"Error uploading image: {str(e)}")
            return None

    def process(self, prompt, model, num_images=1, seed=0, image_way="image",
                image1=None, image2=None, image3=None, image4=None, apikey=""):
        if apikey.strip():
            self.api_key = apikey
            # config = get_config()
            # config['api_key'] = apikey
            # save_config(config)
        else:
            self.api_key = get_config().get('api_key', '')

        default_image = None
        for img in [image1, image2, image3, image4]:
            if img is not None:
                default_image = img
                break
                
        if default_image is None:
            blank_image = Image.new('RGB', (512, 512), color='white')
            default_image = pil2tensor(blank_image)
        
        try:
            if not self.api_key:
                return (default_image, "API key not provided. Please set your API key.")

            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            image_urls = []

            if image_way == "image":
                print("Using base64 encoded images")
                for idx, img in enumerate([image1, image2, image3, image4]):
                    if img is not None:
                        image_base64 = self.image_to_base64(img)
                        if image_base64:
                            image_urls.append(image_base64)
            else:  
                print("Uploading images to get URLs")
                input_images = [img for img in [image1, image2, image3, image4] if img is not None]
                total_images = len(input_images)
                
                for idx, img in enumerate(input_images):
                    progress = 10 + int((idx / max(1, total_images)) * 10)
                    pbar.update_absolute(progress)
                    
                    img_url = self.upload_image_to_get_url(img)
                    if img_url:
                        image_urls.append(img_url)
                        print(f"Image {idx+1}/{total_images} uploaded, URL: {img_url}")
                    else:
                        print(f"Failed to upload image {idx+1}/{total_images}")

            pbar.update_absolute(20)
            
            if model.endswith("/edit"):
                api_endpoint = f"{baseurl}/fal-ai/{model}"

                payload = {
                    "prompt": prompt,
                    "num_images": num_images
                }
                
                if seed > 0:
                    payload["seed"] = seed

                if image_urls and len(image_urls) > 0:
                    payload["image_urls"] = image_urls
                
                pbar.update_absolute(30)

                response = requests.post(
                    api_endpoint,
                    headers=self.get_headers(),
                    json=payload,
                    timeout=self.timeout
                )
            else:
                api_endpoint = f"{baseurl}/fal-ai/{model}"
                
                payload = {
                    "prompt": prompt,
                    "num_images": num_images
                }
                
                if seed > 0:
                    payload["seed"] = seed
                    
                if image_urls:
                    payload["image_urls"] = image_urls
                
                pbar.update_absolute(30)

                response = requests.post(
                    api_endpoint,
                    headers=self.get_headers(),
                    json=payload,
                    timeout=self.timeout
                )

            if response.status_code != 200:
                return (default_image, f"API Error: {response.status_code} - {response.text}")
                
            result = response.json()
 
            if "request_id" not in result:
                return (default_image, "No request_id in response: " + str(result))
            
            request_id = result.get("request_id")
            response_url = result.get("response_url", "")
 
            if "queue.fal.run" in response_url:
                response_url = response_url.replace("https://queue.fal.run", baseurl)

            if not response_url:
                response_url = f"{baseurl}/fal-ai/{model}/requests/{request_id}"
            
            pbar.update_absolute(50)

            max_retries = 30
            retry_count = 0
            result_data = None 
            
            while retry_count < max_retries:
                retry_count += 1
                pbar.update_absolute(50 + min(40, retry_count * 40 // max_retries))
                
                try:
                    result_response = requests.get(
                        response_url,
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )
                    
                    if result_response.status_code != 200:
                        time.sleep(1)
                        continue
                        
                    result_data = result_response.json()

                    if "images" in result_data and result_data["images"]:
                        break

                    time.sleep(1)
                    
                except Exception as e:
                    print(f"Error fetching result: {str(e)}")
                    time.sleep(1)

            if result_data is None:
                return (default_image, "Failed to retrieve results after multiple attempts")

            if "images" not in result_data or not result_data["images"]:
                return (default_image, "No images in response: " + str(result_data))

            generated_images = []
            
            for i, img_data in enumerate(result_data["images"]):
                if "url" in img_data:
                    img_url = img_data["url"]
                    if "queue.fal.run" in img_url:
                        img_url = img_url.replace("https://queue.fal.run", baseurl)
                    
                    try:
                        img_response = requests.get(img_url, timeout=self.timeout)
                        if img_response.status_code == 200:
                            generated_image = Image.open(BytesIO(img_response.content))
                            generated_tensor = pil2tensor(generated_image)
                            generated_images.append(generated_tensor)
                    except Exception as e:
                        print(f"Error downloading image: {str(e)}")
            
            if generated_images:
                combined_tensor = torch.cat(generated_images, dim=0)
                pbar.update_absolute(100)
                return (combined_tensor, f"Successfully generated {len(generated_images)} images using {model}")
            else:
                return (default_image, "Failed to process any images from the API response")
                
        except Exception as e:
            error_message = f"Error processing request: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            return (default_image, error_message)


class Comfly_nano_banana_edit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "mode": (["text2img", "img2img"], {"default": "img2img"}),
                "model": (["nano-banana", "nano-banana-hd"], {"default": "nano-banana"}),
                "aspect_ratio": (["16:9", "4:3", "4:5", "3:2", "1:1", "2:3", "3:4", "5:4", "9:16", "21:9"], {"default": "1:1"}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "apikey": ("STRING", {"default": ""}),
                # "apikey": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647})  
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "response")
    FUNCTION = "generate_image"
    CATEGORY = "RunNode/Google"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = None  

    def get_headers(self):
        return {
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
    
    def generate_image(self, prompt, mode="text2img", model="nano-banana", aspect_ratio="1:1", 
                      image1=None, image2=None, image3=None, image4=None,
                      apikey="", response_format="url", seed=0):  
        if apikey.strip():
            self.api_key = apikey
            # config = get_config()
            # config['api_key'] = apikey
            # save_config(config)
        else:
            self.api_key = get_config().get('api_key', '')
            
        if not self.api_key:
            error_message = "API key not found in Comflyapi.json"
            print(error_message)
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            blank_tensor = pil2tensor(blank_image)
            return (blank_tensor, error_message)
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            final_prompt = prompt
            
            if mode == "text2img":
                headers = self.get_headers()
                headers["Content-Type"] = "application/json"
                
                payload = {
                    "prompt": final_prompt,
                    "model": model,
                    "aspect_ratio": aspect_ratio  
                }
                    
                if response_format:
                    payload["response_format"] = response_format

                if seed > 0:
                    payload["seed"] = seed
                
                response = requests.post(
                    f"{baseurl}/v1/images/generations",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
            else:
                headers = self.get_headers()
                
                files = []
                for img in [image1, image2, image3, image4]:
                    if img is not None:
                        pil_img = tensor2pil(img)[0]
                        buffered = BytesIO()
                        pil_img.save(buffered, format="PNG")
                        buffered.seek(0)
                        files.append(('image', ('image.png', buffered, 'image/png')))
                
                data = {
                    "prompt": final_prompt,
                    "model": model,
                    "aspect_ratio": aspect_ratio 
                }
                
                if response_format:
                    data["response_format"] = response_format

                if seed > 0:
                    data["seed"] = str(seed)
                
                response = requests.post(
                    f"{baseurl}/v1/images/edits",
                    headers=headers,
                    data=data,
                    files=files,
                    timeout=self.timeout
                )
            
            pbar.update_absolute(50)
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, error_message)
                
            result = response.json()
            
            if "data" not in result or not result["data"]:
                error_message = "No image data in response"
                print(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, error_message)
            
            generated_tensors = []
            response_info = f"Generated {len(result['data'])} images using {model}\n"
            response_info += f"Aspect ratio: {aspect_ratio}\n"

            if seed > 0:
                response_info += f"Seed: {seed}\n"
            
            for i, item in enumerate(result["data"]):
                pbar.update_absolute(50 + (i+1) * 40 // len(result["data"]))
                
                if "b64_json" in item:
                    image_data = base64.b64decode(item["b64_json"])
                    generated_image = Image.open(BytesIO(image_data))
                    generated_tensor = pil2tensor(generated_image)
                    generated_tensors.append(generated_tensor)
                    response_info += f"Image {i+1}: Base64 data\n"
                elif "url" in item:
                    image_url = item["url"]
                    response_info += f"Image {i+1}: {image_url}\n"
                    try:
                        img_response = requests.get(image_url, timeout=self.timeout)
                        img_response.raise_for_status()
                        generated_image = Image.open(BytesIO(img_response.content))
                        generated_tensor = pil2tensor(generated_image)
                        generated_tensors.append(generated_tensor)
                    except Exception as e:
                        print(f"Error downloading image from URL: {str(e)}")
            
            pbar.update_absolute(100)
            
            if generated_tensors:
                combined_tensor = torch.cat(generated_tensors, dim=0)
                return (combined_tensor, response_info)
            else:
                error_message = "Failed to process any images"
                print(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, error_message)
            
        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(error_message)
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            blank_tensor = pil2tensor(blank_image)
            return (blank_tensor, error_message)


class Comfly_nano_banana2_edit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "mode": (["text2img", "img2img"], {"default": "img2img"}),
                "model": (["nano-banana-2", "nano-banana-2-2k", "nano-banana-2-4k"], {"default": "nano-banana-2"}),
                "aspect_ratio": (["auto", "16:9", "4:3", "4:5", "3:2", "1:1", "2:3", "3:4", "5:4", "9:16", "21:9"], {"default": "auto"}),
                "image_size": (["1K", "2K", "4K"], {"default": "2K"}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "image10": ("IMAGE",),
                "image11": ("IMAGE",),
                "image12": ("IMAGE",),
                "image13": ("IMAGE",),
                "image14": ("IMAGE",),
                "apikey": ("STRING", {"default": ""}),
                # "apikey": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647})  
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "response", "image_url")
    FUNCTION = "generate_image"
    CATEGORY = "RunNode/Google"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = None

    def get_headers(self):
        return {
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
    
    def generate_image(self, prompt, mode="text2img", model="nano-banana-2", aspect_ratio="auto", 
                      image_size="2K", image1=None, image2=None, image3=None, image4=None,
                      image5=None, image6=None, image7=None, image8=None, image9=None, 
                      image10=None, image11=None, image12=None, image13=None, image14=None,
                      apikey="", response_format="url", seed=0):
        request_id = generate_request_id("img_edit", "google")
        log_prepare("图像编辑", request_id, "RunNode/Google-", "Google", model_name=model)
        rn_pbar = ProgressBar(request_id, "Google", streaming=True, task_type="图像编辑", source="RunNode/Google-")
        rn_pbar.set_generating()
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
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            blank_tensor = pil2tensor(blank_image)
            return (blank_tensor, error_message, "")
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            final_prompt = prompt
            
            if mode == "text2img":
                headers = self.get_headers()
                headers["Content-Type"] = "application/json"
                
                payload = {
                    "prompt": final_prompt,
                    "model": model,
                    "aspect_ratio": aspect_ratio
                }

                if model == "nano-banana-2":
                    payload["image_size"] = image_size
                    
                if response_format:
                    payload["response_format"] = response_format

                if seed > 0:
                    payload["seed"] = seed
                           
                response = requests.post(
                    f"{baseurl}/v1/images/generations",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
            else:
                headers = self.get_headers()

                all_images = [image1, image2, image3, image4, image5, image6, image7, 
                             image8, image9, image10, image11, image12, image13, image14]
                
                files = []
                image_count = 0
                for img in all_images:
                    if img is not None:
                        pil_img = tensor2pil(img)[0]
                        buffered = BytesIO()
                        pil_img.save(buffered, format="PNG")
                        buffered.seek(0)
                        files.append(('image', (f'image_{image_count}.png', buffered, 'image/png')))
                        image_count += 1
                
                print(f"Processing {image_count} input images")
                
                data = {
                    "prompt": final_prompt,
                    "model": model,
                    "aspect_ratio": aspect_ratio
                }
                
                if model == "nano-banana-2":
                    data["image_size"] = image_size
                
                if response_format:
                    data["response_format"] = response_format

                if seed > 0:
                    data["seed"] = str(seed)
               
                response = requests.post(
                    f"{baseurl}/v1/images/edits",
                    headers=headers,
                    data=data,
                    files=files,
                    timeout=self.timeout
                )
            
            pbar.update_absolute(50)
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                rn_pbar.error(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, error_message, "")
                
            result = response.json()
            
            if "data" not in result or not result["data"]:
                error_message = "No image data in response"
                rn_pbar.error(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, error_message, "")
            
            generated_tensors = []
            image_urls = []
            response_info = f"Generated {len(result['data'])} images using {model}\n"

            if model == "nano-banana-2":
                response_info += f"Image size: {image_size}\n"
            
            response_info += f"Aspect ratio: {aspect_ratio}\n"
            
            if mode == "img2img":
                response_info += f"Input images: {image_count}\n"

            if seed > 0:
                response_info += f"Seed: {seed}\n"
            
            for i, item in enumerate(result["data"]):
                pbar.update_absolute(50 + (i+1) * 40 // len(result["data"]))
                
                if "b64_json" in item:
                    image_data = base64.b64decode(item["b64_json"])
                    generated_image = Image.open(BytesIO(image_data))
                    generated_tensor = pil2tensor(generated_image)
                    generated_tensors.append(generated_tensor)
                    response_info += f"Image {i+1}: Base64 data\n"
                elif "url" in item:
                    image_url = item["url"]
                    image_urls.append(image_url)
                    response_info += f"Image {i+1}: {image_url}\n"
                    try:
                        img_response = requests.get(image_url, timeout=self.timeout)
                        img_response.raise_for_status()
                        generated_image = Image.open(BytesIO(img_response.content))
                        generated_tensor = pil2tensor(generated_image)
                        generated_tensors.append(generated_tensor)
                    except Exception as e:
                        rn_pbar.error(f"下载图片失败: {str(e)}")
            
            pbar.update_absolute(100)
            
            if generated_tensors:
                combined_tensor = torch.cat(generated_tensors, dim=0)
                first_image_url = image_urls[0] if image_urls else ""
                try:
                    rn_pbar.done(char_count=len(response_info))
                except Exception:
                    pass
                return (combined_tensor, response_info, first_image_url)
            else:
                error_message = "Failed to process any images"
                rn_pbar.error(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, error_message, "")
            
        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            rn_pbar.error(error_message)
            import traceback
            traceback.print_exc()
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            blank_tensor = pil2tensor(blank_image)
            return (blank_tensor, error_message, "")


class Comfly_nano_banana2_edit_S2A:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "mode": (["text2img", "img2img"], {"default": "img2img"}),
                "model": (["nano-banana-2", "nano-banana-2-2k", "nano-banana-2-4k"], {"default": "nano-banana-2"}),
                "aspect_ratio": (["auto", "16:9", "4:3", "4:5", "3:2", "1:1", "2:3", "3:4", "5:4", "9:16", "21:9"], {"default": "auto"}),
                "image_size": (["1K", "2K", "4K"], {"default": "2K"}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "image10": ("IMAGE",),
                "image11": ("IMAGE",),
                "image12": ("IMAGE",),
                "image13": ("IMAGE",),
                "image14": ("IMAGE",),
                "apikey": ("STRING", {"default": ""}),
                # "apikey": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "task_id": ("STRING", {"default": ""}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647})  
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "task_id", "response")
    FUNCTION = "generate_image"
    CATEGORY = "RunNode/Google"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = None

    def get_headers(self):
        return {
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
    
    def generate_image(self, prompt, mode="text2img", model="nano-banana-2", aspect_ratio="auto", 
                      image_size="2K", image1=None, image2=None, image3=None, image4=None,
                      image5=None, image6=None, image7=None, image8=None, image9=None, 
                      image10=None, image11=None, image12=None, image13=None, image14=None,
                      apikey="", task_id="", response_format="url", seed=0):
        request_id = generate_request_id("img_edit", "google")
        log_prepare("图像编辑", request_id, "RunNode/Google-", "Google", model_name=model)
        rn_pbar = ProgressBar(request_id, "Google", streaming=True, task_type="图像编辑", source="RunNode/Google-")
        rn_pbar.set_generating()
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
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            blank_tensor = pil2tensor(blank_image)
            return (blank_tensor, "", "", json.dumps({"status": "failed", "message": error_message}))
        
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            # 如果提供了task_id，则查询任务状态
            if task_id.strip():
                print(f"Querying task status for task_id: {task_id}")
                result = self._query_task_status(task_id, pbar)
                try:
                    rn_pbar.done(char_count=len(str(result[-1])))
                except Exception:
                    pass
                return result
            
            # 否则创建新的异步任务
                print(f"Creating new async task with mode: {mode}")
            final_prompt = prompt
            
            if mode == "text2img":
                headers = self.get_headers()
                headers["Content-Type"] = "application/json"
                
                payload = {
                    "prompt": final_prompt,
                    "model": model,
                    "aspect_ratio": aspect_ratio
                }

                if model == "nano-banana-2":
                    payload["image_size"] = image_size
                    
                if response_format:
                    payload["response_format"] = response_format

                if seed > 0:
                    payload["seed"] = seed
                
                # 根据API文档，async应该是查询参数
                params = {"async": "true"}
                
                print(f"Submitting text2img async request: {payload}")
                response = requests.post(
                    f"{baseurl}/v1/images/generations",
                    headers=headers,
                    params=params,  # async作为查询参数
                    json=payload,
                    timeout=self.timeout
                )
            else:  # img2img mode
                headers = self.get_headers()

                all_images = [image1, image2, image3, image4, image5, image6, image7, 
                             image8, image9, image10, image11, image12, image13, image14]
                
                files = []
                image_count = 0
                for img in all_images:
                    if img is not None:
                        pil_img = tensor2pil(img)[0]
                        buffered = BytesIO()
                        pil_img.save(buffered, format="PNG")
                        buffered.seek(0)
                        files.append(('image', (f'image_{image_count}.png', buffered, 'image/png')))
                        image_count += 1
                
                print(f"Processing {image_count} input images")
                
                data = {
                    "prompt": final_prompt,
                    "model": model,
                    "aspect_ratio": aspect_ratio
                }
                
                if model == "nano-banana-2":
                    data["image_size"] = image_size
                
                if response_format:
                    data["response_format"] = response_format

                if seed > 0:
                    data["seed"] = str(seed)
                
                # 根据API文档，async应该是查询参数
                params = {"async": "true"}
                
                print(f"Submitting img2img async request with {image_count} images")
                response = requests.post(
                    f"{baseurl}/v1/images/edits",
                    headers=headers,
                    params=params,  # async作为查询参数
                    data=data,
                    files=files,
                    timeout=self.timeout
                )
            
            pbar.update_absolute(30)
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                rn_pbar.error(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, "", "", json.dumps({"status": "failed", "message": error_message}))
                
            result = response.json()
            print(f"API response: {result}")
            
            # 智能判断：API返回异步task_id还是同步data
            if "task_id" in result:
                # 异步模式：返回task_id
                returned_task_id = result["task_id"]
                
                # 构建结构化JSON响应
                result_info = {
                    "status": "pending",
                    "task_id": returned_task_id,
                    "model": model,
                    "mode": mode,
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "image_size": image_size if model == "nano-banana-2" else None,
                    "seed": seed if seed > 0 else None,
                    "message": "Async task created successfully. Please use this task_id to query the result."
                }
                
                # 打印调试信息
                print(f"[ASYNC_RESPONSE] {json.dumps(result_info, ensure_ascii=False)}")
                
                blank_image = Image.new('RGB', (512, 512), color='lightgray')
                blank_tensor = pil2tensor(blank_image)
                pbar.update_absolute(100)
                
                # 异步模式：轮询任务状态直到完成
                print(f"Waiting for task completion: {returned_task_id}")
                max_attempts = 60  # 最多等待10分钟(每10秒查询一次)
                attempt = 0
                
                while attempt < max_attempts:
                    time.sleep(10)  # 等待10秒
                    attempt += 1
                    
                    try:
                        # 查询任务状态
                        query_url = f"{baseurl}/v1/images/tasks/{returned_task_id}"
                        query_response = requests.get(
                            query_url,
                            headers=headers,
                            timeout=self.timeout
                        )
                        
                        if query_response.status_code == 200:
                            query_result = query_response.json()
                            # 处理嵌套响应结构
                            # API返回: {"data": {"status": "SUCCESS", "data": {...}}}
                            actual_status = "unknown"
                            actual_data = None
                            
                            if "data" in query_result and isinstance(query_result["data"], dict):
                                actual_status = query_result["data"].get("status", "unknown")
                                actual_data = query_result["data"].get("data")
                            
                            print(f"Task status (attempt {attempt}): {actual_status}")
                            
                            # 处理更多表示成功的状态
                            if actual_status == "completed" or actual_status == "success" or actual_status == "done" or actual_status == "finished" or actual_status == "SUCCESS" or (actual_status == "unknown" and actual_data):
                                # 任务完成，处理结果
                                # 使用实际的数据字段
                                if actual_data:
                                    generated_tensors = []
                                    image_urls = []
                                    
                                    # 安全处理图片数据
                                    data_items = actual_data.get("data", []) if isinstance(actual_data, dict) else actual_data
                                    if not isinstance(data_items, list):
                                        data_items = [data_items]
                                        
                                    for item in data_items:
                                        try:
                                            if "b64_json" in item and item["b64_json"]:
                                                # 处理Base64图片数据
                                                image_data = base64.b64decode(item["b64_json"])
                                                image_stream = BytesIO(image_data)
                                                generated_image = Image.open(image_stream)
                                                generated_image.verify()  # 验证图片完整性
                                                # 重新打开图片（verify后流位置改变）
                                                image_stream.seek(0)
                                                generated_image = Image.open(image_stream)
                                                # 确保RGB模式
                                                if generated_image.mode != 'RGB':
                                                    generated_image = generated_image.convert('RGB')
                                                generated_tensor = pil2tensor(generated_image)
                                                generated_tensors.append(generated_tensor)
                                            elif "url" in item and item["url"]:
                                                # 处理URL图片数据
                                                image_url = item["url"]
                                                image_urls.append(image_url)
                                                img_response = requests.get(image_url, timeout=self.timeout)
                                                img_response.raise_for_status()
                                                image_stream = BytesIO(img_response.content)
                                                generated_image = Image.open(image_stream)
                                                generated_image.verify()  # 验证图片完整性
                                                # 重新打开图片（verify后流位置改变）
                                                image_stream.seek(0)
                                                generated_image = Image.open(image_stream)
                                                # 确保RGB模式
                                                if generated_image.mode != 'RGB':
                                                    generated_image = generated_image.convert('RGB')
                                                generated_tensor = pil2tensor(generated_image)
                                                generated_tensors.append(generated_tensor)
                                        except Exception as e:
                                            print(f"Error processing image item: {str(e)}")
                                            continue
                                    
                                    if generated_tensors:
                                        combined_tensor = torch.cat(generated_tensors, dim=0)
                                        first_image_url = image_urls[0] if image_urls else ""
                                        final_result_info = {
                                            "status": "success",
                                            "task_id": returned_task_id,
                                            "model": model,
                                            "mode": mode,
                                            "prompt": prompt,
                                            "aspect_ratio": aspect_ratio,
                                            "image_size": image_size if model == "nano-banana-2" else None,
                                            "seed": seed if seed > 0 else None,
                                            "images_count": len(generated_tensors),
                                            "image_url": first_image_url,
                                            "all_urls": image_urls
                                        }
                                    pbar.update_absolute(100)
                                    try:
                                        rn_pbar.done(char_count=len(json.dumps(final_result_info, ensure_ascii=False)))
                                    except Exception:
                                        pass
                                    return (combined_tensor, first_image_url, returned_task_id, json.dumps(final_result_info))
                                
                            elif actual_status == "failed" or actual_status == "error" or actual_status == "FAILURE":
                                # 任务失败
                                error_msg = query_result.get("error", "Unknown error")
                                rn_pbar.error(f"任务失败: {error_msg}")
                                blank_image = Image.new('RGB', (1024, 1024), color='red')
                                blank_tensor = pil2tensor(blank_image)
                                pbar.update_absolute(100)
                                return (blank_tensor, "", "", json.dumps({"status": "failed", "task_id": returned_task_id, "message": error_msg}))
                                
                        else:
                            rn_pbar.error(f"查询失败: {query_response.status_code}")
                            
                    except Exception as e:
                        rn_pbar.error(f"查询任务状态异常: {str(e)}")
                
                # 超时未完成
                rn_pbar.error("任务轮询超时")
                blank_image = Image.new('RGB', (512, 512), color='yellow')
                blank_tensor = pil2tensor(blank_image)
                pbar.update_absolute(100)
                return (blank_tensor, "", returned_task_id, json.dumps({"status": "timeout", "task_id": returned_task_id, "message": "Task polling timed out. Please query manually."}))
                
            elif "data" in result and result["data"]:
                # 同步模式：直接返回图片数据
                print(f"Sync mode detected, processing {len(result['data'])} images directly")
                generated_tensors = []
                image_urls = []
                response_info = f"Generated {len(result['data'])} images using {model}\n"
                
                if model == "nano-banana-2":
                    response_info += f"Image size: {image_size}\n"
                
                response_info += f"Aspect ratio: {aspect_ratio}\n"
                
                if mode == "img2img":
                    response_info += f"Input images: {image_count}\n"
                
                if seed > 0:
                    response_info += f"Seed: {seed}\n"
                
                # 安全处理图片数据
                data_items = result.get("data", [])
                if not isinstance(data_items, list):
                    data_items = [data_items]
                    
                for i, item in enumerate(data_items):
                    try:
                        pbar.update_absolute(50 + (i+1) * 40 // len(data_items))
                        
                        if "b64_json" in item and item["b64_json"]:
                            # 处理Base64图片数据
                            image_data = base64.b64decode(item["b64_json"])
                            image_stream = BytesIO(image_data)
                            generated_image = Image.open(image_stream)
                            generated_image.verify()  # 验证图片完整性
                            # 重新打开图片（verify后流位置改变）
                            image_stream.seek(0)
                            generated_image = Image.open(image_stream)
                            # 确保RGB模式
                            if generated_image.mode != 'RGB':
                                generated_image = generated_image.convert('RGB')
                            generated_tensor = pil2tensor(generated_image)
                            generated_tensors.append(generated_tensor)
                            response_info += f"Image {i+1}: Base64 data\n"
                        elif "url" in item and item["url"]:
                            # 处理URL图片数据
                            image_url = item["url"]
                            image_urls.append(image_url)
                            response_info += f"Image {i+1}: {image_url}\n"
                            img_response = requests.get(image_url, timeout=self.timeout)
                            img_response.raise_for_status()
                            image_stream = BytesIO(img_response.content)
                            generated_image = Image.open(image_stream)
                            generated_image.verify()  # 验证图片完整性
                            # 重新打开图片（verify后流位置改变）
                            image_stream.seek(0)
                            generated_image = Image.open(image_stream)
                            # 确保RGB模式
                            if generated_image.mode != 'RGB':
                                generated_image = generated_image.convert('RGB')
                            generated_tensor = pil2tensor(generated_image)
                            generated_tensors.append(generated_tensor)
                    except Exception as e:
                        print(f"Error processing image item {i}: {str(e)}")
                        continue
                
                pbar.update_absolute(100)
                
                if generated_tensors:
                    combined_tensor = torch.cat(generated_tensors, dim=0)
                    first_image_url = image_urls[0] if image_urls else ""
                    
                    # 构建结构化JSON响应（添加task_id以便网站识别）
                    import uuid
                    sync_task_id = f"sync_{uuid.uuid4().hex[:16]}"
                    
                    result_info = {
                        "status": "success",
                        "task_id": sync_task_id,
                        "model": model,
                        "mode": mode,
                        "prompt": prompt,
                        "aspect_ratio": aspect_ratio,
                        "image_size": image_size if model == "nano-banana-2" else None,
                        "seed": result.get("seed", seed) if seed > 0 else None,
                        "images_count": len(generated_tensors),
                        "image_url": first_image_url,
                        "all_urls": image_urls
                    }
                    
                    # 打印调试信息
                    print(f"[SYNC_RESPONSE] {json.dumps(result_info, ensure_ascii=False)}")
                    try:
                        rn_pbar.done(char_count=len(json.dumps(result_info, ensure_ascii=False)))
                    except Exception:
                        pass
                    
                    return (combined_tensor, first_image_url, sync_task_id, json.dumps(result_info))
                else:
                    error_message = "Failed to process any images"
                    rn_pbar.error(error_message)
                    blank_image = Image.new('RGB', (1024, 1024), color='white')
                    blank_tensor = pil2tensor(blank_image)
                    return (blank_tensor, "", "", json.dumps({"status": "failed", "message": error_message}))
                    
            else:
                # 未知响应格式
                error_message = f"Unexpected API response format: {result}"
                rn_pbar.error(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, "", "", json.dumps({"status": "failed", "message": error_message}))
            
        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            rn_pbar.error(error_message)
            import traceback
            traceback.print_exc()
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            blank_tensor = pil2tensor(blank_image)
            return (blank_tensor, "", "", json.dumps({"status": "failed", "message": error_message}))
    
    def _query_task_status(self, task_id, pbar):
        """查询异步任务状态"""
        try:
            headers = self.get_headers()
            headers["Content-Type"] = "application/json"
            
            # 根据API文档，应该是GET请求，路径为/v1/images/tasks/{task_id}
            query_url = f"{baseurl}/v1/images/tasks/{task_id}"
            print(f"Querying task status: {query_url}")
            response = requests.get(
                query_url,
                headers=headers,
                timeout=self.timeout
            )
            
            pbar.update_absolute(50)
            
            if response.status_code != 200:
                error_message = f"Query Error: {response.status_code} - {response.text}"
                print(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, "", "", json.dumps({"status": "query_failed", "task_id": task_id, "message": error_message}))
            
            result = response.json()
            print(f"Task status response: {result}")
            
            # 处理嵌套响应结构
            # API返回: {"data": {"status": "SUCCESS", "data": {...}}}
            actual_status = "unknown"
            actual_data = None
            
            if "data" in result and isinstance(result["data"], dict):
                actual_status = result["data"].get("status", "unknown")
                actual_data = result["data"].get("data")
            
            # 处理更多表示成功的状态
            if actual_status == "completed" or actual_status == "success" or actual_status == "done" or actual_status == "finished" or actual_status == "SUCCESS" or (actual_status == "unknown" and actual_data):
                # 任务完成，处理结果
                if "data" in result and result["data"]:
                    generated_tensors = []
                    image_urls = []
                    response_info = f"Task completed successfully\n"
                    response_info += f"Task ID: {task_id}\n"
                    response_info += f"Generated {len(result['data'])} images\n"
                    
                    # 安全处理图片数据
                    data_items = result.get("data", [])
                    if not isinstance(data_items, list):
                        data_items = [data_items]
                        
                    for i, item in enumerate(data_items):
                        try:
                            pbar.update_absolute(50 + (i+1) * 40 // len(data_items))
                            
                            if "b64_json" in item and item["b64_json"]:
                                # 处理Base64图片数据
                                image_data = base64.b64decode(item["b64_json"])
                                image_stream = BytesIO(image_data)
                                generated_image = Image.open(image_stream)
                                generated_image.verify()  # 验证图片完整性
                                # 重新打开图片（verify后流位置改变）
                                image_stream.seek(0)
                                generated_image = Image.open(image_stream)
                                # 确保RGB模式
                                if generated_image.mode != 'RGB':
                                    generated_image = generated_image.convert('RGB')
                                generated_tensor = pil2tensor(generated_image)
                                generated_tensors.append(generated_tensor)
                                response_info += f"Image {i+1}: Base64 data\n"
                            elif "url" in item and item["url"]:
                                # 处理URL图片数据
                                image_url = item["url"]
                                image_urls.append(image_url)
                                response_info += f"Image {i+1}: {image_url}\n"
                                img_response = requests.get(image_url, timeout=self.timeout)
                                img_response.raise_for_status()
                                image_stream = BytesIO(img_response.content)
                                generated_image = Image.open(image_stream)
                                generated_image.verify()  # 验证图片完整性
                                # 重新打开图片（verify后流位置改变）
                                image_stream.seek(0)
                                generated_image = Image.open(image_stream)
                                # 确保RGB模式
                                if generated_image.mode != 'RGB':
                                    generated_image = generated_image.convert('RGB')
                                generated_tensor = pil2tensor(generated_image)
                                generated_tensors.append(generated_tensor)
                        except Exception as e:
                            print(f"Error processing image item {i}: {str(e)}")
                            continue
                    
                    pbar.update_absolute(100)
                    
                    if generated_tensors:
                        combined_tensor = torch.cat(generated_tensors, dim=0)
                        first_image_url = image_urls[0] if image_urls else ""
                        return (combined_tensor, first_image_url, task_id, json.dumps({"status": "success", "task_id": task_id, "images_count": len(generated_tensors), "image_url": first_image_url, "all_urls": image_urls}))
                    else:
                        error_message = "No valid images in completed task"
                        print(error_message)
                        blank_image = Image.new('RGB', (1024, 1024), color='white')
                        blank_tensor = pil2tensor(blank_image)
                        return (blank_tensor, "", "", json.dumps({"status": "failed", "task_id": task_id, "message": error_message}))
                else:
                    error_message = "Task completed but no image data"
                    print(error_message)
                    blank_image = Image.new('RGB', (1024, 1024), color='white')
                    blank_tensor = pil2tensor(blank_image)
                    return (blank_tensor, "", "", json.dumps({"status": "failed", "task_id": task_id, "message": error_message}))
            
            elif actual_status == "processing" or actual_status == "pending" or actual_status == "in_progress":
                # 任务还在处理中
                response_info = f"Task is still processing\n"
                response_info += f"Task ID: {task_id}\n"
                response_info += f"Status: {actual_status}\n"
                response_info += f"Please query again later"
                
                blank_image = Image.new('RGB', (512, 512), color='yellow')
                blank_tensor = pil2tensor(blank_image)
                pbar.update_absolute(100)
                return (blank_tensor, "", "", json.dumps({"status": actual_status, "task_id": task_id, "message": "Task is still processing. Please query again later."}))
            
            elif actual_status == "failed" or actual_status == "error":
                # 任务失败
                error_msg = result.get("error", "Unknown error")
                response_info = f"Task failed\n"
                response_info += f"Task ID: {task_id}\n"
                response_info += f"Error: {error_msg}"
                
                blank_image = Image.new('RGB', (512, 512), color='red')
                blank_tensor = pil2tensor(blank_image)
                pbar.update_absolute(100)
                return (blank_tensor, "", "", json.dumps({"status": "failed", "task_id": task_id, "message": error_msg}))
            
            else:
                # 未知状态
                response_info = f"Unknown task status\n"
                response_info += f"Task ID: {task_id}\n"
                response_info += f"Status: {actual_status}\n"
                response_info += f"Response: {result}"
                
                blank_image = Image.new('RGB', (512, 512), color='gray')
                blank_tensor = pil2tensor(blank_image)
                pbar.update_absolute(100)
                return (blank_tensor, "", "", json.dumps({"status": actual_status, "task_id": task_id, "message": f"Unknown task status: {actual_status}", "raw_response": result}))
                
        except Exception as e:
            error_message = f"Error querying task status: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            blank_tensor = pil2tensor(blank_image)
            return (blank_tensor, "", "", json.dumps({"status": "query_error", "task_id": task_id, "message": error_message}))


class Comfly_banana2_edit_group:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "mode": (["text2img", "img2img"], {"default": "img2img"}),
                "model": (["nano-banana-2", "nano-banana-2-2k", "nano-banana-2-4k"], {"default": "nano-banana-2"}),
                "aspect_ratio": (["auto", "16:9", "4:3", "4:5", "3:2", "1:1", "2:3", "3:4", "5:4", "9:16", "21:9"], {"default": "auto"}),
                "image_size": (["1K", "2K", "4K"], {"default": "2K"}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "image10": ("IMAGE",),
                "image11": ("IMAGE",),
                "image12": ("IMAGE",),
                "image13": ("IMAGE",),
                "image14": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "build"
    CATEGORY = "RunNode/Google"
    def __init__(self):
        self.api_key = get_config().get('api_key', '')
    def _image_to_b64(self, image_tensor):
        if image_tensor is None:
            return None
        pil_image = tensor2pil(image_tensor)[0]
        buf = BytesIO()
        pil_image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{b64}"
    def build(self, **kwargs):
        prompt = kwargs.get("prompt", "")
        mode = kwargs.get("mode", "text2img")
        model = kwargs.get("model", "nano-banana-2")
        aspect_ratio = kwargs.get("aspect_ratio", "auto")
        image_size = kwargs.get("image_size", "2K")
        response_format = kwargs.get("response_format", "url")
        seed = int(kwargs.get("seed", 0))
        api_key = kwargs.get("api_key", "").strip() or self.api_key
        images = []
        for i in range(1, 15):
            b64 = self._image_to_b64(kwargs.get(f"image{i}"))
            if b64:
                images.append(b64)
        group = {
            "prompt": prompt,
            "mode": mode,
            "model": model,
            "aspect_ratio": aspect_ratio,
            "image_size": image_size,
            "response_format": response_format,
            "seed": seed,
            "api_key": api_key,
            "images": images,
        }
        return (json.dumps(group, ensure_ascii=False),)


class Comfly_banana2_edit_S2A_group:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "mode": (["text2img", "img2img"], {"default": "img2img"}),
                "model": (["nano-banana-2", "nano-banana-2-2k", "nano-banana-2-4k"], {"default": "nano-banana-2"}),
                "aspect_ratio": (["auto", "16:9", "4:3", "4:5", "3:2", "1:1", "2:3", "3:4", "5:4", "9:16", "21:9"], {"default": "auto"}),
                "image_size": (["1K", "2K", "4K"], {"default": "2K"}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "image10": ("IMAGE",),
                "image11": ("IMAGE",),
                "image12": ("IMAGE",),
                "image13": ("IMAGE",),
                "image14": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("group",)
    FUNCTION = "build"
    CATEGORY = "RunNode/Google"
    def __init__(self):
        self.api_key = get_config().get('api_key', '')
    def _image_to_b64(self, image_tensor):
        if image_tensor is None:
            return None
        pil_image = tensor2pil(image_tensor)[0]
        buf = BytesIO()
        pil_image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{b64}"
    def build(self, **kwargs):
        prompt = kwargs.get("prompt", "")
        mode = kwargs.get("mode", "text2img")
        model = kwargs.get("model", "nano-banana-2")
        aspect_ratio = kwargs.get("aspect_ratio", "auto")
        image_size = kwargs.get("image_size", "2K")
        response_format = kwargs.get("response_format", "url")
        seed = int(kwargs.get("seed", 0))
        api_key = kwargs.get("api_key", "").strip() or self.api_key
        images = []
        for i in range(1, 15):
            b64 = self._image_to_b64(kwargs.get(f"image{i}"))
            if b64:
                images.append(b64)
        group = {
            "prompt": prompt,
            "mode": mode,
            "model": model,
            "aspect_ratio": aspect_ratio,
            "image_size": image_size,
            "response_format": response_format,
            "seed": seed,
            "api_key": api_key,
            "images": images,
        }
        return (json.dumps(group, ensure_ascii=False),)

class _ComflyBanana2ImageBatchRunner:
    def __init__(self):
        self.config = get_config()
        self.api_key = self.config.get('api_key', '')
        self.timeout = None
        self.task_progress = {}
        self.global_pbar = None
        self.rn_pbar = None
    def _headers(self, api_key):
        return {"Authorization": f"Bearer {api_key}"}
    def _blank(self, w=1024, h=1024, color='white'):
        img = Image.new('RGB', (w, h), color=color)
        return pil2tensor(img)
    def _decode_b64_images(self, images_b64):
        files = []
        count = 0
        for b64 in images_b64:
            try:
                if b64.startswith("data:image"):
                    header, data = b64.split(",", 1)
                    raw = base64.b64decode(data)
                else:
                    raw = base64.b64decode(b64)
                bio = BytesIO(raw)
                files.append(('image', (f'image_{count}.png', bio, 'image/png')))
                count += 1
            except Exception:
                continue
        return files, count
    def _process(self, idx, payload):
        res = {"index": idx, "status": "failed", "image": self._blank(), "image_url": "", "error": "", "response": ""}
        prompt = payload.get("prompt", "")
        mode = payload.get("mode", "img2img")
        model = payload.get("model", "nano-banana-2")
        aspect_ratio = payload.get("aspect_ratio", "auto")
        image_size = payload.get("image_size", "2K")
        response_format = payload.get("response_format", "url")
        seed = int(payload.get("seed", 0))
        api_key = payload.get("api_key", "")
        images_b64 = payload.get("images", [])
        headers = self._headers(api_key)
        try:
            if mode == "text2img":
                headers["Content-Type"] = "application/json"
                body = {"prompt": prompt, "model": model, "aspect_ratio": aspect_ratio}
                if model == "nano-banana-2":
                    body["image_size"] = image_size
                if response_format:
                    body["response_format"] = response_format
                if seed > 0:
                    body["seed"] = seed
                resp = requests.post(f"{baseurl}/v1/images/generations", headers=headers, json=body, timeout=self.timeout)
            else:
                files, count = self._decode_b64_images(images_b64)
                data = {"prompt": prompt, "model": model, "aspect_ratio": aspect_ratio}
                if model == "nano-banana-2":
                    data["image_size"] = image_size
                if response_format:
                    data["response_format"] = response_format
                if seed > 0:
                    data["seed"] = str(seed)
                resp = requests.post(f"{baseurl}/v1/images/edits", headers=headers, data=data, files=files, timeout=self.timeout)
            if resp.status_code != 200:
                res["error"] = f"HTTP {resp.status_code}: {resp.text}"
                return res
            result = resp.json()
            tensors = []
            urls = []
            items = result.get("data", [])
            if not isinstance(items, list):
                items = [items]
            for item in items:
                if "b64_json" in item and item["b64_json"]:
                    raw = base64.b64decode(item["b64_json"])
                    bio = BytesIO(raw)
                    im = Image.open(bio)
                    if im.mode != 'RGB':
                        im = im.convert('RGB')
                    tensors.append(pil2tensor(im))
                elif "url" in item and item["url"]:
                    url = item["url"]
                    urls.append(url)
                    r = requests.get(url, timeout=self.timeout)
                    r.raise_for_status()
                    im = Image.open(BytesIO(r.content))
                    if im.mode != 'RGB':
                        im = im.convert('RGB')
                    tensors.append(pil2tensor(im))
            if tensors:
                combined = torch.cat(tensors, dim=0)
                res["status"] = "success"
                res["image"] = combined
                res["image_url"] = urls[0] if urls else ""
                res["response"] = json.dumps({"count": len(tensors), "urls": urls}, ensure_ascii=False)
                return res
            res["error"] = "No image data"
            return res
        except Exception as e:
            res["error"] = str(e)
            return res
    def run(self, groups, max_workers, global_cfg):
        request_id = generate_request_id("image_batch", "google")
        model_name = (global_cfg.get("model") or "nano-banana-2") if isinstance(global_cfg, dict) else "nano-banana-2"
        log_prepare("图像批量生成", request_id, "RunNode/Google-", "Google", model_name=model_name)
        self.rn_pbar = ProgressBar(request_id, "Google", extra_info=f"并发:{max_workers}", streaming=True, task_type="图像批量生成", source="RunNode/Google-")
        self.rn_pbar.set_generating()
        self.global_pbar = comfy.utils.ProgressBar(100)
        api_key = global_cfg.get("api_key", "").strip() or self.api_key
        if api_key:
            self.api_key = api_key
        if not self.api_key:
            empty = [self._blank()] * max_workers
            log = json.dumps({"error": "API Key未配置"}, ensure_ascii=False, indent=2)
            try:
                self.rn_pbar.error("API Key未配置")
            except Exception:
                pass
            return (*empty, log)
        tasks = []
        for i, g in enumerate(groups, start=1):
            try:
                d = json.loads(g) if isinstance(g, str) else {}
            except Exception:
                d = {}
            prompt = d.get("prompt", "") or global_cfg.get("global_prompt", "")
            mode = d.get("mode", global_cfg.get("mode", "img2img"))
            model = d.get("model", global_cfg.get("model", "nano-banana-2"))
            aspect_ratio = d.get("aspect_ratio", global_cfg.get("aspect_ratio", "auto"))
            image_size = d.get("image_size", global_cfg.get("image_size", "2K"))
            response_format = d.get("response_format", global_cfg.get("response_format", "url"))
            seed = int(d.get("seed", global_cfg.get("seed", 0)))
            g_api_key = d.get("api_key", "").strip() or self.api_key
            images = d.get("images", [])
            tasks.append({
                "idx": i,
                "payload": {
                    "prompt": prompt,
                    "mode": mode,
                    "model": model,
                    "aspect_ratio": aspect_ratio,
                    "image_size": image_size,
                    "response_format": response_format,
                    "seed": seed,
                    "api_key": g_api_key,
                    "images": images,
                }
            })
        self.task_progress = {t["idx"]: 0 for t in tasks}
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futmap = {}
            for t in tasks:
                f = ex.submit(self._process, t["idx"], t["payload"])
                futmap[f] = t["idx"]
                if t["payload"]["prompt"]:
                    time.sleep(0.1)
            for f in as_completed(futmap):
                idx = futmap[f]
                try:
                    results[idx] = f.result()
                except Exception as e:
                    results[idx] = {"index": idx, "status": "failed", "image": self._blank(), "image_url": "", "error": str(e), "response": ""}
        output_images = []
        for i in range(1, max_workers + 1):
            output_images.append(results.get(i, {}).get("image", self._blank()))
        log_data = {"global": {"max_concurrent": max_workers, "api_key_configured": bool(self.api_key)}, "tasks": []}
        for i in range(1, max_workers + 1):
            r = results.get(i, {})
            log_data["tasks"].append({"task_index": i, "status": r.get("status", "idle"), "image_url": r.get("image_url", ""), "error": r.get("error", "")})
        log = json.dumps(log_data, ensure_ascii=False, indent=2)
        try:
            self.rn_pbar.done(char_count=len(log))
        except Exception:
            pass
        return (*output_images, log)


class Comfly_banana2_edit_run_4:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "promt_1": ("STRING", {"forceInput": True, "multiline": True}),
            },
            "optional": {
                "promt_2": ("STRING", {"forceInput": True, "multiline": True}),
                "promt_3": ("STRING", {"forceInput": True, "multiline": True}),
                "promt_4": ("STRING", {"forceInput": True, "multiline": True}),
                "max_concurrent": ("INT", {"default": 4, "min": 1, "max": 4}),
                "global_prompt": ("STRING", {"default": "", "multiline": True}),
                "mode": (["text2img", "img2img"], {"default": "img2img"}),
                "model": (["nano-banana-2", "nano-banana-2-2k", "nano-banana-2-4k"], {"default": "nano-banana-2"}),
                "aspect_ratio": (["auto", "16:9", "4:3", "4:5", "3:2", "1:1", "2:3", "3:4", "5:4", "9:16", "21:9"], {"default": "auto"}),
                "image_size": (["1K", "2K", "4K"], {"default": "2K"}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
            }
        }
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("image1", "image2", "image3", "image4", "log")
    FUNCTION = "run"
    CATEGORY = "RunNode/Google"
    def __init__(self):
        self.runner = _ComflyBanana2ImageBatchRunner()
    def run(self, promt_1, promt_2, promt_3, promt_4, **cfg):
        max_workers = int(cfg.get("max_concurrent", 4))
        max_workers = max(1, min(4, max_workers))
        groups = [promt_1, promt_2, promt_3, promt_4]
        return self.runner.run(groups, max_workers, cfg)


class Comfly_banana2_edit_run_8:
    @classmethod
    def INPUT_TYPES(cls):
        promt_ = {f"promt_{i}": ("STRING", {"forceInput": True, "multiline": True}) for i in range(2, 9)}
        return {
            "required": {
                "promt_1": ("STRING", {"forceInput": True, "multiline": True}),
            },
            "optional": {
                **promt_,
                "max_concurrent": ("INT", {"default": 8, "min": 1, "max": 8}),
                "global_prompt": ("STRING", {"default": "", "multiline": True}),
                "mode": (["text2img", "img2img"], {"default": "img2img"}),
                "model": (["nano-banana-2", "nano-banana-2-2k", "nano-banana-2-4k"], {"default": "nano-banana-2"}),
                "aspect_ratio": (["auto", "16:9", "4:3", "4:5", "3:2", "1:1", "2:3", "3:4", "5:4", "9:16", "21:9"], {"default": "auto"}),
                "image_size": (["1K", "2K", "4K"], {"default": "2K"}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
            }
        }
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("image1", "image2", "image3", "image4", "image5", "image6", "image7", "image8", "log")
    FUNCTION = "run"
    CATEGORY = "RunNode/Google"
    def __init__(self):
        self.runner = _ComflyBanana2ImageBatchRunner()
    def run(self, promt_1, promt_2, promt_3, promt_4, promt_5=None, promt_6=None, promt_7=None, promt_8=None, **cfg):
        max_workers = int(cfg.get("max_concurrent", 8))
        max_workers = max(1, min(8, max_workers))
        groups = [promt_1, promt_2, promt_3, promt_4, promt_5, promt_6, promt_7, promt_8]
        return self.runner.run(groups, max_workers, cfg)


class Comfly_banana2_edit_run_16:
    @classmethod
    def INPUT_TYPES(cls):
        promt_ = {f"promt_{i}": ("STRING", {"forceInput": True, "multiline": True}) for i in range(2, 17)}
        return {
            "required": {
                "promt_1": ("STRING", {"forceInput": True, "multiline": True}),
            },
            "optional": {
                **promt_,
                "max_concurrent": ("INT", {"default": 16, "min": 1, "max": 16}),
                "global_prompt": ("STRING", {"default": "", "multiline": True}),
                "mode": (["text2img", "img2img"], {"default": "img2img"}),
                "model": (["nano-banana-2", "nano-banana-2-2k", "nano-banana-2-4k"], {"default": "nano-banana-2"}),
                "aspect_ratio": (["auto", "16:9", "4:3", "4:5", "3:2", "1:1", "2:3", "3:4", "5:4", "9:16", "21:9"], {"default": "auto"}),
                "image_size": (["1K", "2K", "4K"], {"default": "2K"}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
            }
        }
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
                    "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("image1", "image2", "image3", "image4", "image5", "image6", "image7", "image8",
                    "image9", "image10", "image11", "image12", "image13", "image14", "image15", "image16", "log")
    FUNCTION = "run"
    CATEGORY = "RunNode/Google"
    def __init__(self):
        self.runner = _ComflyBanana2ImageBatchRunner()
    def run(self, **cfg):
        max_workers = int(cfg.get("max_concurrent", 16))
        max_workers = max(1, min(16, max_workers))
        groups = [cfg.get(f"promt_{i}", "") for i in range(1, 17)]
        return self.runner.run(groups, max_workers, cfg)


class Comfly_banana2_edit_run_32:
    @classmethod
    def INPUT_TYPES(cls):
        promt_ = {f"promt_{i}": ("STRING", {"forceInput": True, "multiline": True}) for i in range(2, 33)}
        return {
            "required": {
                "promt_1": ("STRING", {"forceInput": True, "multiline": True}),
            },
            "optional": {
                **promt_,
                "max_concurrent": ("INT", {"default": 32, "min": 1, "max": 32}),
                "global_prompt": ("STRING", {"default": "", "multiline": True}),
                "mode": (["text2img", "img2img"], {"default": "img2img"}),
                "model": (["nano-banana-2", "nano-banana-2-2k", "nano-banana-2-4k"], {"default": "nano-banana-2"}),
                "aspect_ratio": (["auto", "16:9", "4:3", "4:5", "3:2", "1:1", "2:3", "3:4", "5:4", "9:16", "21:9"], {"default": "auto"}),
                "image_size": (["1K", "2K", "4K"], {"default": "2K"}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
            }
        }
    RETURN_TYPES = (
        "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
        "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
        "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
        "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
        "STRING"
    )
    RETURN_NAMES = (
        "image1", "image2", "image3", "image4", "image5", "image6", "image7", "image8",
        "image9", "image10", "image11", "image12", "image13", "image14", "image15", "image16",
        "image17", "image18", "image19", "image20", "image21", "image22", "image23", "image24",
        "image25", "image26", "image27", "image28", "image29", "image30", "image31", "image32",
        "log"
    )
    FUNCTION = "run"
    CATEGORY = "RunNode/Google"
    def __init__(self):
        self.runner = _ComflyBanana2ImageBatchRunner()
    def run(self, **cfg):
        max_workers = int(cfg.get("max_concurrent", 32))
        max_workers = max(1, min(32, max_workers))
        groups = [cfg.get(f"promt_{i}", "") for i in range(1, 33)]
        return self.runner.run(groups, max_workers, cfg)


class _ComflyBanana2ImageBatchRunnerS2A:
    def __init__(self):
        self.config = get_config()
        self.api_key = self.config.get('api_key', '')
        self.timeout = None
    def _headers(self, api_key):
        return {"Authorization": f"Bearer {api_key}"}
    def _blank(self, w=512, h=512, color='lightgray'):
        img = Image.new('RGB', (w, h), color=color)
        return pil2tensor(img)
    def _decode_b64_images(self, images_b64):
        files = []
        count = 0
        for b64 in images_b64:
            try:
                if b64.startswith("data:image"):
                    header, data = b64.split(",", 1)
                    raw = base64.b64decode(data)
                else:
                    raw = base64.b64decode(b64)
                bio = BytesIO(raw)
                files.append(('image', (f'image_{count}.png', bio, 'image/png')))
                count += 1
            except Exception:
                continue
        return files, count
    def _submit_async(self, payload, headers):
        mode = payload.get("mode", "img2img")
        model = payload.get("model", "nano-banana-2")
        prompt = payload.get("prompt", "")
        aspect_ratio = payload.get("aspect_ratio", "auto")
        image_size = payload.get("image_size", "2K")
        response_format = payload.get("response_format", "url")
        seed = int(payload.get("seed", 0))
        if mode == "text2img":
            headers["Content-Type"] = "application/json"
            body = {"prompt": prompt, "model": model, "aspect_ratio": aspect_ratio}
            if model == "nano-banana-2":
                body["image_size"] = image_size
            if response_format:
                body["response_format"] = response_format
            if seed > 0:
                body["seed"] = seed
            params = {"async": "true"}
            resp = requests.post(f"{baseurl}/v1/images/generations", headers=headers, params=params, json=body, timeout=self.timeout)
        else:
            files, count = self._decode_b64_images(payload.get("images", []))
            data = {"prompt": prompt, "model": model, "aspect_ratio": aspect_ratio}
            if model == "nano-banana-2":
                data["image_size"] = image_size
            if response_format:
                data["response_format"] = response_format
            if seed > 0:
                data["seed"] = str(seed)
            params = {"async": "true"}
            resp = requests.post(f"{baseurl}/v1/images/edits", headers=headers, params=params, data=data, files=files, timeout=self.timeout)
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}: {resp.text}"
        data = resp.json()
        task_id = data.get("task_id")
        if not task_id:
            return None, "No task_id in async response"
        return task_id, None
    def _poll(self, task_id, headers, poll_interval_sec=15, max_attempts=40):
        attempt = 0
        while attempt < max_attempts:
            time.sleep(poll_interval_sec)
            attempt += 1
            q = requests.get(f"{baseurl}/v1/images/tasks/{task_id}", headers=headers, timeout=self.timeout)
            if q.status_code != 200:
                continue
            qr = q.json()
            actual_status = "unknown"
            actual_data = None
            if isinstance(qr.get("data"), dict):
                actual_status = qr["data"].get("status", "unknown")
                actual_data = qr["data"].get("data")
            if actual_status in ["completed", "success", "done", "finished", "SUCCESS"] or (actual_status == "unknown" and actual_data):
                items = actual_data.get("data", []) if isinstance(actual_data, dict) else (actual_data or [])
                if not isinstance(items, list):
                    items = [items]
                tensors = []
                urls = []
                for item in items:
                    if "b64_json" in item and item["b64_json"]:
                        raw = base64.b64decode(item["b64_json"])
                        im = Image.open(BytesIO(raw))
                        if im.mode != 'RGB':
                            im = im.convert('RGB')
                        tensors.append(pil2tensor(im))
                    elif "url" in item and item["url"]:
                        url = item["url"]
                        urls.append(url)
                        r = requests.get(url, timeout=self.timeout)
                        r.raise_for_status()
                        im = Image.open(BytesIO(r.content))
                        if im.mode != 'RGB':
                            im = im.convert('RGB')
                        tensors.append(pil2tensor(im))
                if tensors:
                    combined = torch.cat(tensors, dim=0)
                    return combined, (urls[0] if urls else ""), None
                return self._blank(), "", "No image data in completed task"
            if actual_status in ["failed", "error", "FAILURE"]:
                err = qr.get("error", "Unknown error")
                return self._blank(w=1024, h=1024, color='red'), "", err
        return self._blank(), "", "Task polling timed out"
    def _process(self, idx, payload, poll_interval_sec=15, max_attempts=40):
        res = {"index": idx, "status": "failed", "image": self._blank(), "image_url": "", "error": "", "response": ""}
        api_key = payload.get("api_key", "")
        headers = self._headers(api_key)
        task_id, err = self._submit_async(payload, headers)
        if err:
            res["error"] = err
            return res
        image, url, perr = self._poll(task_id, headers, poll_interval_sec=poll_interval_sec, max_attempts=max_attempts)
        if perr:
            res["error"] = perr
            return res
        res["status"] = "success"
        res["image"] = image
        res["image_url"] = url
        return res
    def run(self, groups, max_workers, global_cfg):
        request_id = generate_request_id("image_batch_s2a", "google")
        model_name = (global_cfg.get("model") or "nano-banana-2") if isinstance(global_cfg, dict) else "nano-banana-2"
        log_prepare("图像批量生成(异步轮询)", request_id, "RunNode/Google-", "Google", model_name=model_name)
        api_key = global_cfg.get("api_key", "").strip() or self.api_key
        if api_key:
            self.api_key = api_key
        if not self.api_key:
            empty = [self._blank()] * max_workers
            log = json.dumps({"error": "API Key未配置"}, ensure_ascii=False, indent=2)
            return (*empty, log)
        poll_interval_sec = int(global_cfg.get("poll_interval_sec", 15))
        max_attempts = int(global_cfg.get("max_attempts", 40))
        tasks = []
        for i, g in enumerate(groups, start=1):
            try:
                d = json.loads(g) if isinstance(g, str) else {}
            except Exception:
                d = {}
            prompt = d.get("prompt", "") or global_cfg.get("global_prompt", "")
            mode = d.get("mode", global_cfg.get("mode", "img2img"))
            model = d.get("model", global_cfg.get("model", "nano-banana-2"))
            aspect_ratio = d.get("aspect_ratio", global_cfg.get("aspect_ratio", "auto"))
            image_size = d.get("image_size", global_cfg.get("image_size", "2K"))
            response_format = d.get("response_format", global_cfg.get("response_format", "url"))
            seed = int(d.get("seed", global_cfg.get("seed", 0)))
            g_api_key = d.get("api_key", "").strip() or self.api_key
            images = d.get("images", [])
            tasks.append({
                "idx": i,
                "payload": {
                    "prompt": prompt,
                    "mode": mode,
                    "model": model,
                    "aspect_ratio": aspect_ratio,
                    "image_size": image_size,
                    "response_format": response_format,
                    "seed": seed,
                    "api_key": g_api_key,
                    "images": images,
                }
            })
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futmap = {}
            for t in tasks:
                f = ex.submit(self._process, t["idx"], t["payload"], poll_interval_sec, max_attempts)
                futmap[f] = t["idx"]
                if t["payload"]["prompt"]:
                    time.sleep(0.1)
            for f in as_completed(futmap):
                idx = futmap[f]
                try:
                    results[idx] = f.result()
                except Exception as e:
                    results[idx] = {"index": idx, "status": "failed", "image": self._blank(), "image_url": "", "error": str(e), "response": ""}
        output_images = []
        for i in range(1, max_workers + 1):
            output_images.append(results.get(i, {}).get("image", self._blank()))
        log_data = {"global": {"max_concurrent": max_workers, "api_key_configured": bool(self.api_key)}, "tasks": []}
        for i in range(1, max_workers + 1):
            r = results.get(i, {})
            log_data["tasks"].append({"task_index": i, "status": r.get("status", "idle"), "image_url": r.get("image_url", ""), "error": r.get("error", "")})
        log = json.dumps(log_data, ensure_ascii=False, indent=2)
        return (*output_images, log)


class Comfly_banana2_edit_S2A_run_4:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "promt_1": ("STRING", {"forceInput": True, "multiline": True}),
            },
            "optional": {
                "promt_2": ("STRING", {"forceInput": True, "multiline": True}),
                "promt_3": ("STRING", {"forceInput": True, "multiline": True}),
                "promt_4": ("STRING", {"forceInput": True, "multiline": True}),
                "max_concurrent": ("INT", {"default": 4, "min": 1, "max": 4}),
                "global_prompt": ("STRING", {"default": "", "multiline": True}),
                "mode": (["text2img", "img2img"], {"default": "img2img"}),
                "model": (["nano-banana-2", "nano-banana-2-2k", "nano-banana-2-4k"], {"default": "nano-banana-2"}),
                "aspect_ratio": (["auto", "16:9", "4:3", "4:5", "3:2", "1:1", "2:3", "3:4", "5:4", "9:16", "21:9"], {"default": "auto"}),
                "image_size": (["1K", "2K", "4K"], {"default": "2K"}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "poll_interval_sec": ("INT", {"default": 15, "min": 5, "max": 300}),
                "max_attempts": ("INT", {"default": 40, "min": 10, "max": 1000}),
            }
        }
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("image1", "image2", "image3", "image4", "log")
    FUNCTION = "run"
    CATEGORY = "RunNode/Google"
    def __init__(self):
        self.runner = _ComflyBanana2ImageBatchRunnerS2A()
    def run(self, promt_1, promt_2, promt_3, promt_4, **cfg):
        max_workers = int(cfg.get("max_concurrent", 4))
        max_workers = max(1, min(4, max_workers))
        groups = [promt_1, promt_2, promt_3, promt_4]
        return self.runner.run(groups, max_workers, cfg)


class Comfly_banana2_edit_S2A_run_8:
    @classmethod
    def INPUT_TYPES(cls):
        promt_ = {f"promt_{i}": ("STRING", {"forceInput": True, "multiline": True}) for i in range(2, 9)}
        return {
            "required": {
                "promt_1": ("STRING", {"forceInput": True, "multiline": True}),
            },
            "optional": {
                **promt_,
                "max_concurrent": ("INT", {"default": 8, "min": 1, "max": 8}),
                "global_prompt": ("STRING", {"default": "", "multiline": True}),
                "mode": (["text2img", "img2img"], {"default": "img2img"}),
                "model": (["nano-banana-2", "nano-banana-2-2k", "nano-banana-2-4k"], {"default": "nano-banana-2"}),
                "aspect_ratio": (["auto", "16:9", "4:3", "4:5", "3:2", "1:1", "2:3", "3:4", "5:4", "9:16", "21:9"], {"default": "auto"}),
                "image_size": (["1K", "2K", "4K"], {"default": "2K"}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "poll_interval_sec": ("INT", {"default": 15, "min": 5, "max": 300}),
                "max_attempts": ("INT", {"default": 40, "min": 10, "max": 1000}),
            }
        }
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("image1", "image2", "image3", "image4", "image5", "image6", "image7", "image8", "log")
    FUNCTION = "run"
    CATEGORY = "RunNode/Google"
    def __init__(self):
        self.runner = _ComflyBanana2ImageBatchRunnerS2A()
    def run(self, **cfg):
        max_workers = int(cfg.get("max_concurrent", 8))
        max_workers = max(1, min(8, max_workers))
        groups = [cfg.get(f"promt_{i}", "") for i in range(1, 9)]
        return self.runner.run(groups, max_workers, cfg)


class Comfly_banana2_edit_S2A_run_16:
    @classmethod
    def INPUT_TYPES(cls):
        promt_ = {f"promt_{i}": ("STRING", {"forceInput": True, "multiline": True}) for i in range(2, 17)}
        return {
            "required": {
                "promt_1": ("STRING", {"forceInput": True, "multiline": True}),
            },
            "optional": {
                **promt_,
                "max_concurrent": ("INT", {"default": 16, "min": 1, "max": 16}),
                "global_prompt": ("STRING", {"default": "", "multiline": True}),
                "mode": (["text2img", "img2img"], {"default": "img2img"}),
                "model": (["nano-banana-2", "nano-banana-2-2k", "nano-banana-2-4k"], {"default": "nano-banana-2"}),
                "aspect_ratio": (["auto", "16:9", "4:3", "4:5", "3:2", "1:1", "2:3", "3:4", "5:4", "9:16", "21:9"], {"default": "auto"}),
                "image_size": (["1K", "2K", "4K"], {"default": "2K"}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "poll_interval_sec": ("INT", {"default": 15, "min": 5, "max": 300}),
                "max_attempts": ("INT", {"default": 40, "min": 10, "max": 1000}),
            }
        }
    RETURN_TYPES = (
        "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
        "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
        "STRING"
    )
    RETURN_NAMES = (
        "image1", "image2", "image3", "image4", "image5", "image6", "image7", "image8",
        "image9", "image10", "image11", "image12", "image13", "image14", "image15", "image16",
        "log"
    )
    FUNCTION = "run"
    CATEGORY = "RunNode/Google"
    def __init__(self):
        self.runner = _ComflyBanana2ImageBatchRunnerS2A()
    def run(self, **cfg):
        max_workers = int(cfg.get("max_concurrent", 16))
        max_workers = max(1, min(16, max_workers))
        groups = [cfg.get(f"promt_{i}", "") for i in range(1, 17)]
        return self.runner.run(groups, max_workers, cfg)


class Comfly_banana2_edit_S2A_run_32:
    @classmethod
    def INPUT_TYPES(cls):
        promt_ = {f"promt_{i}": ("STRING", {"forceInput": True, "multiline": True}) for i in range(2, 33)}
        return {
            "required": {
                "promt_1": ("STRING", {"forceInput": True, "multiline": True}),
            },
            "optional": {
                **promt_,
                "max_concurrent": ("INT", {"default": 32, "min": 1, "max": 32}),
                "global_prompt": ("STRING", {"default": "", "multiline": True}),
                "mode": (["text2img", "img2img"], {"default": "text2img"}),
                "model": (["nano-banana-2", "nano-banana-2-2k", "nano-banana-2-4k"], {"default": "nano-banana-2"}),
                "aspect_ratio": (["auto", "16:9", "4:3", "4:5", "3:2", "1:1", "2:3", "3:4", "5:4", "9:16", "21:9"], {"default": "auto"}),
                "image_size": (["1K", "2K", "4K"], {"default": "2K"}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "poll_interval_sec": ("INT", {"default": 15, "min": 5, "max": 300}),
                "max_attempts": ("INT", {"default": 40, "min": 10, "max": 1000}),
            }
        }
    RETURN_TYPES = (
        "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
        "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
        "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
        "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
        "STRING"
    )
    RETURN_NAMES = (
        "image1", "image2", "image3", "image4", "image5", "image6", "image7", "image8",
        "image9", "image10", "image11", "image12", "image13", "image14", "image15", "image16",
        "image17", "image18", "image19", "image20", "image21", "image22", "image23", "image24",
        "image25", "image26", "image27", "image28", "image29", "image30", "image31", "image32",
        "log"
    )
    FUNCTION = "run"
    CATEGORY = "RunNode/Google"
    def __init__(self):
        self.runner = _ComflyBanana2ImageBatchRunnerS2A()
    def run(self, **cfg):
        max_workers = int(cfg.get("max_concurrent", 32))
        max_workers = max(1, min(32, max_workers))
        groups = [cfg.get(f"promt_{i}", "") for i in range(1, 33)]
        return self.runner.run(groups, max_workers, cfg)
