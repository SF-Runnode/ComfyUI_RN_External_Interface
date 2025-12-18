from ..comfly_config import *
from .__init__ import *


class ComflyGeminiAPI:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": ("STRING", {"default": "gemini-2.0-flash-exp-image", "placeholder": "Enter model name"}),
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
        self.timeout = 120 

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
            config = get_config()
            config['api_key'] = api_key
            save_config(config)

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
    
    def generate_video(self, prompt, model="veo3", enhance_prompt=False, aspect_ratio="16:9", apikey="", 
                      image1=None, image2=None, image3=None, seed=0, enable_upsample=False):
        if apikey.strip():
            self.api_key = apikey
            config = get_config()
            config['api_key'] = apikey
            save_config(config)
            
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
                print(error_message)
                return ("", "", json.dumps({"code": "error", "message": error_message}))
                
            result = response.json()
            
            if result.get("code") != "success":
                error_message = f"API returned error: {result.get('message', 'Unknown error')}"
                print(error_message)
                return ("", "", json.dumps({"code": "error", "message": error_message}))
                
            task_id = result.get("data")
            if not task_id:
                error_message = "No task ID returned from API"
                print(error_message)
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
                        print(error_message)
                        return ("", "", json.dumps({"code": "error", "message": error_message}))
                        
                except Exception as e:
                    print(f"Error checking generation status: {str(e)}")
            
            if not video_url:
                error_message = "Failed to retrieve video URL after multiple attempts"
                print(error_message)
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
                return (video_adapter, video_url, json.dumps(response_data))
            
        except Exception as e:
            error_message = f"Error generating video: {str(e)}"
            print(error_message)
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
        self.timeout = 300

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
            config = get_config()
            config['api_key'] = apikey
            save_config(config)

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
        self.timeout = 300

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
            config = get_config()
            config['api_key'] = apikey
            save_config(config)

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
                response_url = response_url.replace("https://queue.fal.run", "https://ai.t8star.cn")

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
                        img_url = img_url.replace("https://queue.fal.run", "https://ai.t8star.cn")
                    
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
                "mode": (["text2img", "img2img"], {"default": "text2img"}),
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
        self.timeout = 1200

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
            config = get_config()
            config['api_key'] = apikey
            save_config(config)
            
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
                "mode": (["text2img", "img2img"], {"default": "text2img"}),
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
        self.timeout = 600

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
        if apikey.strip():
            self.api_key = apikey
            config = get_config()
            config['api_key'] = apikey
            save_config(config)
            
        if not self.api_key:
            error_message = "API key not found in Comflyapi.json"
            print(error_message)
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
                print(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, error_message, "")
                
            result = response.json()
            
            if "data" not in result or not result["data"]:
                error_message = "No image data in response"
                print(error_message)
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
                        print(f"Error downloading image from URL: {str(e)}")
            
            pbar.update_absolute(100)
            
            if generated_tensors:
                combined_tensor = torch.cat(generated_tensors, dim=0)
                first_image_url = image_urls[0] if image_urls else ""
                return (combined_tensor, response_info, first_image_url)
            else:
                error_message = "Failed to process any images"
                print(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, error_message, "")
            
        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            blank_tensor = pil2tensor(blank_image)
            return (blank_tensor, error_message, "")
