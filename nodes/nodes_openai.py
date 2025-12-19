from ..comfly_config import *
from .__init__ import *


def downscale_input(image):
    samples = image.movedim(-1,1)

    total = int(1536 * 1024)
    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
    if scale_by >= 1:
        return image
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)

    s = common_upscale(samples, width, height, "lanczos", "disabled")
    s = s.movedim(1,-1)
    return s


class Comfly_gpt_image_1_edit:

    _last_edited_image = None
    _conversation_history = []
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "mask": ("MASK",),
                "api_key": ("STRING", {"default": ""}),
                # "api_key": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "model": ("STRING", {"default": "gpt-image-1"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10}),
                "quality": (["auto", "high", "medium", "low"], {"default": "auto"}),
                "size": (["auto", "1024x1024", "1536x1024", "1024x1536"], {"default": "auto"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "clear_chats": ("BOOLEAN", {"default": True}),
                "background": (["auto", "transparent", "opaque"], {"default": "auto"}),
                "output_compression": ("INT", {"default": 100, "min": 0, "max": 100}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "max_retries": ("INT", {"default": 5, "min": 1, "max": 10}),
                "initial_timeout": ("INT", {"default": 900, "min": 60, "max": 1200}),
                "input_fidelity": (["low", "high"], {"default": "low"}),
                "partial_images": ([0, 1, 2, 3], {"default": 0}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("edited_image", "response", "chats")
    FUNCTION = "edit_image"
    CATEGORY = "RunNode/OpenAI"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 900
        self.session = requests.Session()
        retry_strategy = requests.packages.urllib3.util.retry.Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def get_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def format_conversation_history(self):
        """Format the conversation history for display"""
        if not Comfly_gpt_image_1_edit._conversation_history:
            return ""
        formatted_history = ""
        for entry in Comfly_gpt_image_1_edit._conversation_history:
            formatted_history += f"**User**: {entry['user']}\n\n"
            formatted_history += f"**AI**: {entry['ai']}\n\n"
            formatted_history += "---\n\n"
        return formatted_history.strip()
    
    def make_request_with_retry(self, url, data=None, files=None, max_retries=5, initial_timeout=300):
        """Make a request with automatic retries and exponential backoff"""
        for attempt in range(1, max_retries + 1):
            current_timeout = min(initial_timeout * (1.5 ** (attempt - 1)), 1200)  
            
            try:
                if files:
                    response = self.session.post(
                        url,
                        headers=self.get_headers(),
                        data=data,
                        files=files,
                        timeout=current_timeout
                    )
                else:
                    response = self.session.post(
                        url,
                        headers=self.get_headers(),
                        json=data,
                        timeout=current_timeout
                    )
                
                response.raise_for_status()
                return response
            
            except requests.exceptions.Timeout as e:
                if attempt == max_retries:
                    raise TimeoutError(f"Request timed out after {max_retries} attempts. Last timeout: {current_timeout}s")
                wait_time = min(2 ** (attempt - 1), 60)  
                time.sleep(wait_time)
            
            except requests.exceptions.ConnectionError as e:
                if attempt == max_retries:
                    raise ConnectionError(f"Connection error after {max_retries} attempts: {str(e)}")
                wait_time = min(2 ** (attempt - 1), 60)
                time.sleep(wait_time)
            
            except requests.exceptions.HTTPError as e:
                if e.response.status_code in (400, 401, 403):
                    print(f"Client error: {str(e)}")
                    raise
                if attempt == max_retries:
                    raise
                wait_time = min(2 ** (attempt - 1), 60)
                time.sleep(wait_time)
            
            except Exception as e:
                if attempt == max_retries:
                    raise
                wait_time = min(2 ** (attempt - 1), 60)
                time.sleep(wait_time)
    
    def edit_image(self, image, prompt, model="gpt-image-1", n=1, quality="auto", 
              seed=0, mask=None, api_key="", size="auto", clear_chats=True,
              background="auto", output_compression=100, output_format="png",
              max_retries=5, initial_timeout=300, input_fidelity="low", partial_images=0):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            save_config(config)
 
        original_image = image
        original_batch_size = image.shape[0]
        use_saved_image = False

        if not clear_chats and Comfly_gpt_image_1_edit._last_edited_image is not None:
            if original_batch_size > 1:
                last_batch_size = Comfly_gpt_image_1_edit._last_edited_image.shape[0]
                last_image_first = Comfly_gpt_image_1_edit._last_edited_image[0:1]
                if last_image_first.shape[1:] == original_image.shape[1:]:
                    image = torch.cat([last_image_first, original_image[1:]], dim=0)
                    use_saved_image = True
            else:
                image = Comfly_gpt_image_1_edit._last_edited_image
                use_saved_image = True

        if clear_chats:
            Comfly_gpt_image_1_edit._conversation_history = []

            
        try:
            if not self.api_key:
                error_message = "API key not found in Comflyapi.json"
                print(error_message)
                return (original_image, error_message, self.format_conversation_history())
          
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)
            
            files = {}
 
            if image is not None:
                batch_size = image.shape[0]
                for i in range(batch_size):
                    single_image = image[i:i+1]
                    scaled_image = downscale_input(single_image).squeeze()
                    
                    image_np = (scaled_image.numpy() * 255).astype(np.uint8)
                    img = Image.fromarray(image_np)
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    
                    if batch_size == 1:
                        files['image'] = ('image.png', img_byte_arr, 'image/png')
                    else:
                        if 'image[]' not in files:
                            files['image[]'] = []
                        files['image[]'].append(('image_{}.png'.format(i), img_byte_arr, 'image/png'))
            
            if mask is not None:
                if image.shape[0] != 1:
                    raise Exception("Cannot use a mask with multiple images")
                if image is None:
                    raise Exception("Cannot use a mask without an input image")
                if mask.shape[1:] != image.shape[1:-1]:
                    raise Exception("Mask and Image must be the same size")
                
                batch, height, width = mask.shape
                rgba_mask = torch.zeros(height, width, 4, device="cpu")
                rgba_mask[:,:,3] = (1-mask.squeeze().cpu())
                scaled_mask = downscale_input(rgba_mask.unsqueeze(0)).squeeze()
                mask_np = (scaled_mask.numpy() * 255).astype(np.uint8)
                mask_img = Image.fromarray(mask_np)
                mask_byte_arr = io.BytesIO()
                mask_img.save(mask_byte_arr, format='PNG')
                mask_byte_arr.seek(0)
                files['mask'] = ('mask.png', mask_byte_arr, 'image/png')

            data = {
                'prompt': prompt,
                'model': model,
                'n': str(n),
                'quality': quality
            }
            
            if size != "auto":
                data['size'] = size
                
            if background != "auto":
                data['background'] = background
                
            if output_compression != 100:
                data['output_compression'] = str(output_compression)
                
            if output_format != "png":
                data['output_format'] = output_format

            if input_fidelity != "low":
                data['input_fidelity'] = input_fidelity
                
            if partial_images > 0:
                data['partial_images'] = str(partial_images)

            pbar.update_absolute(30)

            try:
                if 'image[]' in files:
                    image_files = []
                    for file_tuple in files['image[]']:
                        image_files.append(('image', file_tuple))

                    if 'mask' in files:
                        image_files.append(('mask', files['mask']))

                    response = self.make_request_with_retry(
                        f"{baseurl}/v1/images/edits",
                        data=data,
                        files=image_files,
                        max_retries=max_retries,
                        initial_timeout=initial_timeout
                    )
                else:
                    request_files = []
                    if 'image' in files:
                        request_files.append(('image', files['image']))
                    if 'mask' in files:
                        request_files.append(('mask', files['mask']))

                    response = self.make_request_with_retry(
                        f"{baseurl}/v1/images/edits",
                        data=data,
                        files=request_files,
                        max_retries=max_retries,
                        initial_timeout=initial_timeout
                    )

            except TimeoutError as e:
                error_message = f"API timeout error: {str(e)}"
                print(error_message)
                return (original_image, error_message, self.format_conversation_history())
            except Exception as e:
                error_message = f"API request error: {str(e)}"
                print(error_message)
                return (original_image, error_message, self.format_conversation_history())

            pbar.update_absolute(50)
            result = response.json()
            
            if "data" not in result or not result["data"]:
                error_message = "No image data in response"
                print(error_message)
                return (original_image, error_message, self.format_conversation_history())

            edited_images = []
            image_urls = []

            for item in result["data"]:
                if "b64_json" in item:
                    b64_data = item["b64_json"]
                    if b64_data.startswith("data:image/png;base64,"):
                        b64_data = b64_data[len("data:image/png;base64,"):]    
                    image_data = base64.b64decode(b64_data)
                    edited_image = Image.open(BytesIO(image_data))
                    edited_tensor = pil2tensor(edited_image)
                    edited_images.append(edited_tensor)
                elif "url" in item:
                    image_urls.append(item["url"])
                    try:
                        for download_attempt in range(1, max_retries + 1):
                            try:
                                img_response = requests.get(
                                    item["url"], 
                                    timeout=min(initial_timeout * (1.5 ** (download_attempt - 1)), 900)
                                )
                                img_response.raise_for_status()
                                
                                edited_image = Image.open(BytesIO(img_response.content))
                                edited_tensor = pil2tensor(edited_image)
                                edited_images.append(edited_tensor)
                                break
                            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                                if download_attempt == max_retries:
                                    print(f"Failed to download image after {max_retries} attempts: {str(e)}")
                                    continue
                                wait_time = min(2 ** (download_attempt - 1), 60)
                                print(f"Image download error (attempt {download_attempt}/{max_retries}). Retrying in {wait_time} seconds...")
                                time.sleep(wait_time)
                            except Exception as e:
                                print(f"Error downloading image from URL: {str(e)}")
                                break
                    except Exception as e:
                        print(f"Error processing image URL: {str(e)}")

            pbar.update_absolute(90)

            if edited_images:
                combined_tensor = torch.cat(edited_images, dim=0)
                response_info = f"Successfully edited {len(edited_images)} image(s)\n"
                response_info += f"Prompt: {prompt}\n"
                response_info += f"Model: {model}\n"
                response_info += f"Quality: {quality}\n"
                
                if use_saved_image:
                    response_info += "[Using previous edited image as input]\n"
                    
                if size != "auto":
                    response_info += f"Size: {size}\n"
                    
                if background != "auto":
                    response_info += f"Background: {background}\n"
                    
                if output_compression != 100:
                    response_info += f"Output Compression: {output_compression}%\n"
                    
                if output_format != "png":
                    response_info += f"Output Format: {output_format}\n"

                if input_fidelity != "low":
                    response_info += f"Input Fidelity: {input_fidelity}\n"
                
                if partial_images > 0:
                    response_info += f"Partial Images: {partial_images}\n"

                Comfly_gpt_image_1_edit._conversation_history.append({
                    "user": f"Edit image with prompt: {prompt}",
                    "ai": f"Generated edited image with {model}"
                })
 
                Comfly_gpt_image_1_edit._last_edited_image = combined_tensor
                
                pbar.update_absolute(100)
                return (combined_tensor, response_info, self.format_conversation_history())
            else:
                error_message = "No edited images in response"
                print(error_message)
                return (original_image, error_message, self.format_conversation_history())
            
        except Exception as e:
            error_message = f"Error in image editing: {str(e)}"
            import traceback
            print(traceback.format_exc())  
            print(error_message)
            return (original_image, error_message, self.format_conversation_history())
        

class Comfly_gpt_image_1:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                # "api_key": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "model": ("STRING", {"default": "gpt-image-1"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10}),
                "quality": (["auto", "high", "medium", "low"], {"default": "auto"}),
                "size": (["auto", "1024x1024", "1536x1024", "1024x1536"], {"default": "auto"}),
                "background": (["auto", "transparent", "opaque"], {"default": "auto"}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "moderation": (["auto", "low"], {"default": "auto"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("generated_image", "response")
    FUNCTION = "generate_image"
    CATEGORY = "RunNode/OpenAI"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def generate_image(self, prompt, model="gpt-image-1", n=1, quality="auto", 
                size="auto", background="auto", output_format="png", 
                moderation="auto", seed=0, api_key=""):
        
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            save_config(config)
            
        try:
            if not self.api_key:
                error_message = "API key not found in Comflyapi.json"
                print(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, error_message)
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)
            payload = {
                "prompt": prompt,
                "model": model,
                "n": n,
                "quality": quality,
                "background": background,
                "output_format": output_format,
                "moderation": moderation,
            }

            if size != "auto":
                payload["size"] = size
            
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
                return (blank_tensor, error_message)

            result = response.json()

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            response_info = f"**GPT-image-1 Generation ({timestamp})**\n\n"
            response_info += f"Prompt: {prompt}\n"
            response_info += f"Model: {model}\n"
            response_info += f"Quality: {quality}\n"
            if size != "auto":
                response_info += f"Size: {size}\n"
            response_info += f"Background: {background}\n"
            response_info += f"Seed: {seed} (Note: Seed not used by API)\n\n"

            generated_images = []
            image_urls = []
            
            if "data" in result and result["data"]:
                for i, item in enumerate(result["data"]):
                    pbar.update_absolute(50 + (i+1) * 50 // len(result["data"]))
                    
                    if "b64_json" in item:
                        b64_data = item["b64_json"]
                        if b64_data.startswith("data:image/png;base64,"):
                            b64_data = b64_data[len("data:image/png;base64,"):]    
                        image_data = base64.b64decode(b64_data)
                        generated_image = Image.open(BytesIO(image_data))
                        generated_tensor = pil2tensor(generated_image)
                        generated_images.append(generated_tensor)
                    elif "url" in item:
                        image_urls.append(item["url"])
                        try:
                            img_response = requests.get(item["url"])
                            if img_response.status_code == 200:
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
                return (blank_tensor, response_info)

            if "usage" in result:
                response_info += "Usage Information:\n"
                if "total_tokens" in result["usage"]:
                    response_info += f"Total Tokens: {result['usage']['total_tokens']}\n"
                if "input_tokens" in result["usage"]:
                    response_info += f"Input Tokens: {result['usage']['input_tokens']}\n"
                if "output_tokens" in result["usage"]:
                    response_info += f"Output Tokens: {result['usage']['output_tokens']}\n"

                if "input_tokens_details" in result["usage"]:
                    response_info += "Input Token Details:\n"
                    details = result["usage"]["input_tokens_details"]
                    if "text_tokens" in details:
                        response_info += f"  Text Tokens: {details['text_tokens']}\n"
                    if "image_tokens" in details:
                        response_info += f"  Image Tokens: {details['image_tokens']}\n"
            
            if generated_images:
                combined_tensor = torch.cat(generated_images, dim=0)
                
                pbar.update_absolute(100)
                return (combined_tensor, response_info)
            else:
                error_message = "No images were successfully processed"
                print(error_message)
                response_info += f"Error: {error_message}\n"
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, response_info)
                
        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(error_message)
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            blank_tensor = pil2tensor(blank_image)
            return (blank_tensor, error_message)


class ComflyChatGPTApi:
 
    _last_generated_image_urls = ""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": ("STRING", {"default": "gpt-4o-image", "multiline": False}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                # "api_key": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "files": ("FILES",), 
                "image_url": ("STRING", {"multiline": False, "default": ""}),
                "images": ("IMAGE", {"default": None}),  
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 16384, "step": 1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "frequency_penalty": ("FLOAT", {"default": -2.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "image_download_timeout": ("INT", {"default": 600, "min": 300, "max": 1200, "step": 10}),
                "clear_chats": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "response", "image_urls", "chats")
    FUNCTION = "process"
    CATEGORY = "RunNode/OpenAI"
    
    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 800
        self.image_download_timeout = 600
        self.api_endpoint = f"{baseurl}/v1/chat/completions"
        self.conversation_history = []
 
    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def image_to_base64(self, image):
        """Convert PIL image to base64 string"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def file_to_base64(self, file_path):
        """Convert file to base64 string and return appropriate MIME type"""
        try:
            with open(file_path, "rb") as file:
                file_content = file.read()
                encoded_content = base64.b64encode(file_content).decode('utf-8')
                mime_type, _ = mimetypes.guess_type(file_path)
                if not mime_type:
                    mime_type = "application/octet-stream"
                return encoded_content, mime_type
        except Exception as e:
            print(f"Error encoding file: {str(e)}")
            return None, None

    def extract_image_urls(self, response_text):
        """Extract image URLs from markdown format in response"""
      
        image_pattern = r'!\[.*?\]\((.*?)\)'
        matches = re.findall(image_pattern, response_text)
      
        if not matches:
            url_pattern = r'https?://\S+\.(?:jpg|jpeg|png|gif|webp)'
            matches = re.findall(url_pattern, response_text)
        
        if not matches:
            all_urls_pattern = r'https?://\S+'
            matches = re.findall(all_urls_pattern, response_text)
        return matches if matches else []

    def download_image(self, url, timeout=30):
        """Download image from URL and convert to tensor with improved error handling"""
        try:
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://comfyui.com/'
            }
           
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
          
            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                print(f"Warning: URL doesn't point to an image. Content-Type: {content_type}")
               
            image = Image.open(BytesIO(response.content))
            return pil2tensor(image)
        except requests.exceptions.Timeout:
            print(f"Timeout error downloading image from {url} (timeout: {timeout}s)")
            return None
        except requests.exceptions.SSLError as e:
            print(f"SSL Error downloading image from {url}: {str(e)}")
            return None
        except requests.exceptions.ConnectionError:
            print(f"Connection error downloading image from {url}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Request error downloading image from {url}: {str(e)}")
            return None
        except Exception as e:
            print(f"Error downloading image from {url}: {str(e)}")
            return None

    def format_conversation_history(self):
        """Format the conversation history for display"""
        if not self.conversation_history:
            return ""
        formatted_history = ""
        for entry in self.conversation_history:
            formatted_history += f"**User**: {entry['user']}\n\n"
            formatted_history += f"**AI**: {entry['ai']}\n\n"
            formatted_history += "---\n\n"
        return formatted_history.strip()

    def send_request_synchronous(self, payload, pbar):
        """Send a synchronous streaming request to the API"""
        full_response = ""
        session = requests.Session()
        
        try:
            response = session.post(
                self.api_endpoint,
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

                                    pbar.update_absolute(min(40, 20 + len(full_response) // 100))
                        except json.JSONDecodeError:
                            continue
            
            return full_response
            
        except requests.exceptions.Timeout:
            raise TimeoutError(f"API request timed out after {self.timeout} seconds")
        except Exception as e:
            raise Exception(f"Error in streaming response: {str(e)}")

    def process(self, prompt, model, clear_chats=True, files=None, image_url="", images=None, temperature=0.7, 
           max_tokens=4096, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0, seed=-1,
           image_download_timeout=100, api_key=""):

        if model.lower() == "gpt-image-1":
            error_message = "不支持此模型，请使用 gpt-4o-image，gpt-4o-image-vip，sora_image，sora_image-vip 这4个模型。"
            print(error_message)

            if images is not None:
                return (images, error_message, "", self.format_conversation_history())
            else:
                blank_img = Image.new('RGB', (512, 512), color='white')
                return (pil2tensor(blank_img), error_message, "", self.format_conversation_history())
            
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            save_config(config)

        try:
            self.image_download_timeout = image_download_timeout
          
            if clear_chats:
                self.conversation_history = []
                
            if not self.api_key:
                error_message = "API key not found in Comflyapi.json"
                print(error_message)
               
                blank_img = Image.new('RGB', (512, 512), color='white')
                return (pil2tensor(blank_img), error_message, "", self.format_conversation_history()) 
            
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
           
            if seed < 0:
                seed = random.randint(0, 2147483647)
                print(f"Using random seed: {seed}")
           
            content = []
            
            content.append({"type": "text", "text": prompt})
            
           
            if not clear_chats and ComflyChatGPTApi._last_generated_image_urls:
                prev_image_url = ComflyChatGPTApi._last_generated_image_urls.split('\n')[0].strip()
                if prev_image_url:
                    print(f"Using previous image URL: {prev_image_url}")
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": prev_image_url}
                    })
            
            elif clear_chats:
                if images is not None:
                    batch_size = images.shape[0]
                    max_images = min(batch_size, 4)  
                    for i in range(max_images):
                        pil_image = tensor2pil(images)[i]
                        image_base64 = self.image_to_base64(pil_image)
                        content.append({
                            "type": "image_url", 
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                        })
                    if batch_size > max_images:
                        content.append({
                            "type": "text",
                            "text": f"\n(Note: {batch_size-max_images} additional images were omitted due to API limitations)"
                        })
                
                elif image_url:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    })
        
            elif image_url:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
            
            if files:
                file_paths = files if isinstance(files, list) else [files]
                for file_path in file_paths:
                    encoded_content, mime_type = self.file_to_base64(file_path)
                    if encoded_content and mime_type:
                       
                        if mime_type.startswith('image/'):
                           
                            content.append({
                                "type": "image_url", 
                                "image_url": {"url": f"data:{mime_type};base64,{encoded_content}"}
                            })
                        else:
                            
                            content.append({
                                "type": "text", 
                                "text": f"\n\nI've attached a file ({os.path.basename(file_path)}) for analysis."
                            })
                            content.append({
                                "type": "file_url",
                                "file_url": {
                                    "url": f"data:{mime_type};base64,{encoded_content}",
                                    "name": os.path.basename(file_path)
                                }
                            })
        
            messages = []
        
            messages.append({
                "role": "user",
                "content": content
            })
        
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "seed": seed,
                "stream": True  
            }

            response_text = self.send_request_synchronous(payload, pbar)
        
            self.conversation_history.append({
                "user": prompt,
                "ai": response_text
            })
        
            technical_response = f"**Model**: {model}\n**Temperature**: {temperature}\n**Seed**: {seed}\n**Time**: {timestamp}"
        
            image_urls = self.extract_image_urls(response_text)
            image_urls_string = "\n".join(image_urls) if image_urls else ""
            
            if image_urls:
                ComflyChatGPTApi._last_generated_image_urls = image_urls_string
     
            chat_history = self.format_conversation_history()
            if image_urls:
                try:
                    
                    img_tensors = []
                    successful_downloads = 0
                    for i, url in enumerate(image_urls):
                        print(f"Attempting to download image {i+1}/{len(image_urls)} from: {url}")
                
                        pbar.update_absolute(min(80, 40 + (i+1) * 40 // len(image_urls)))
                        img_tensor = self.download_image(url, self.image_download_timeout)
                        if img_tensor is not None:
                            img_tensors.append(img_tensor)
                            successful_downloads += 1
                    print(f"Successfully downloaded {successful_downloads} out of {len(image_urls)} images")
                    if img_tensors:
                
                        combined_tensor = torch.cat(img_tensors, dim=0)
                        pbar.update_absolute(100)
                        return (combined_tensor, technical_response, image_urls_string, chat_history)
                except Exception as e:
                    print(f"Error processing image URLs: {str(e)}")
        
            if images is not None:
                pbar.update_absolute(100)
                return (images, technical_response, image_urls_string, chat_history)  
            else:
                blank_img = Image.new('RGB', (512, 512), color='white')
                blank_tensor = pil2tensor(blank_img)
                pbar.update_absolute(100)
                return (blank_tensor, technical_response, image_urls_string, chat_history)  
                
        except Exception as e:
            error_message = f"Error calling ChatGPT API: {str(e)}"
            print(error_message)
        
            if images is not None:
                return (images, error_message, "", self.format_conversation_history())  
            else:
                blank_img = Image.new('RGB', (512, 512), color='white')
                blank_tensor = pil2tensor(blank_img)
                return (blank_tensor, error_message, "", self.format_conversation_history())


class Comfly_sora2_openai:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["sora-2", "sora-2-pro"], {"default": "sora-2"}),
            },
            "optional": {
                "apikey": ("STRING", {"default": ""}),
                # "apikey": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "seconds": (["10", "15", "25"], {"default": "15"}),
                "size": (["1280x720", "720x1280", "1792x1024", "1024x1792"], {"default": "1280x720"}),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "private": ("BOOLEAN", {"default": True})
            }
        }
    
    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "response", "video_url", "seed")
    FUNCTION = "process"
    CATEGORY = "RunNode/OpenAI"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 900

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
    
    def process(self, prompt, model, apikey="", seconds="15", size="1280x720", image=None, seed=0, private=True):
        if apikey.strip():
            self.api_key = apikey
            config = get_config()
            config['api_key'] = apikey
            save_config(config)
            
        if not self.api_key:
            error_response = {"status": "error", "message": "API key not provided or not found in config"}
            return ("", json.dumps(error_response), "", "0")

        if model == "sora-2":
            if seconds == "25":  
                error_message = "The sora-2 model does not support 25 second videos. Please use sora-2-pro for 25 second videos."
                print(error_message)
                return ("", json.dumps({"status": "error", "message": error_message}), "", "0")
            if size in ["1792x1024", "1024x1792"]:
                error_message = "The sora-2 model does not support 1080P resolution. Please use sora-2-pro for 1080P videos."
                print(error_message)
                return ("", json.dumps({"status": "error", "message": error_message}), "", "0")
      
        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(10)
        
        try:
            data = {
                "model": model,
                "prompt": prompt,
                "seconds": seconds,
                "size": size,
                "private": private
            }
            
            if seed > 0:
                data["seed"] = str(seed)
                
            files = []

            if image is not None:
                pil_image = tensor2pil(image)[0]
                buffered = BytesIO()
                pil_image.save(buffered, format="PNG")
                buffered.seek(0)
                files.append(('input_reference', ('image.png', buffered, 'image/png')))
            
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            pbar.update_absolute(20)
            
            # 请求体 暂不修改
            response = requests.post(
                "https://ai.t8star.cn/v1/videos",
                headers=headers,
                data=data,
                files=files,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                return ("", json.dumps({"status": "error", "message": error_message}), "", "0")
                
            result = response.json()
            
            if "id" not in result:
                error_message = "No task ID in API response"
                print(error_message)
                return ("", json.dumps({"status": "error", "message": error_message}), "", "0")
            
            task_id = result["id"]
            print(f"Task ID: {task_id}")
            
            pbar.update_absolute(30)

            max_attempts = 120  
            attempts = 0
            video_url = None
            actual_seed = str(seed) if seed > 0 else "0"
            
            while attempts < max_attempts:
                time.sleep(10)
                attempts += 1
                
                try:
                    # 请求体，暂不修改
                    status_response = requests.get(
                        f"https://ai.t8star.cn/v1/videos/{task_id}",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )
                    
                    if status_response.status_code != 200:
                        continue
                        
                    status_data = status_response.json()

                    progress = status_data.get("progress", 0)
                    try:
                        progress_int = int(progress)
                        pbar_value = min(90, 30 + int(progress_int * 0.6))
                        pbar.update_absolute(pbar_value)
                    except (ValueError, TypeError):
                        progress_value = min(80, 30 + (attempts * 50 // max_attempts))
                        pbar.update_absolute(progress_value)
                    
                    status = status_data.get("status", "")
                    
                    if status == "completed":
                        video_url = status_data.get("video_url")
                        if not video_url and "url" in status_data:
                            video_url = status_data.get("url")

                        if "seed" in status_data:
                            actual_seed = str(status_data.get("seed", "0"))
                        
                        break
                    elif status == "failed":
                        fail_reason = status_data.get("fail_reason", "Unknown error")
                        error_message = f"Video generation failed: {fail_reason}"
                        print(error_message)
                        return ("", json.dumps({"status": "error", "message": error_message, "task_id": task_id}), "", actual_seed)
                        
                except Exception as e:
                    print(f"Error checking task status: {str(e)}")
            
            if not video_url:
                error_message = f"Failed to get video URL after {max_attempts} attempts"
                print(error_message)
                return ("", json.dumps({"status": "error", "message": error_message, "task_id": task_id}), "", actual_seed)
            
            video_adapter = ComflyVideoAdapter(video_url)
            
            pbar.update_absolute(100)
            
            response_data = {
                "status": "success",
                "task_id": task_id,
                "prompt": prompt,
                "model": model,
                "size": size,
                "seconds": seconds,
                "private": private,
                "video_url": video_url,
                "seed": actual_seed
            }
            
            return (video_adapter, json.dumps(response_data), video_url, actual_seed)
            
        except Exception as e:
            error_message = f"Error in video generation: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            return ("", json.dumps({"status": "error", "message": error_message}), "", "0")
           

class Comfly_sora2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["sora-2", "sora-2-pro"], {"default": "sora-2"}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
                "duration": (["10", "15", "25"], {"default": "15"}),
                "hd": ("BOOLEAN", {"default": False}),
                "apikey": ("STRING", {"default": ""})
                # "apikey": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "private": ("BOOLEAN", {"default": True})
            }
        }
    
    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "response")
    FUNCTION = "process"
    CATEGORY = "RunNode/OpenAI"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 900

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
    
    def process(self, prompt, model, aspect_ratio="16:9", duration="10", hd=False, apikey="", 
                image1=None, image2=None, image3=None, image4=None, seed=0, private=True):
        if apikey.strip():
            self.api_key = apikey
            config = get_config()
            config['api_key'] = apikey
            save_config(config)
            
        if not self.api_key:
            error_response = {"status": "error", "message": "API key not provided or not found in config"}
            return ("", "", json.dumps(error_response))

        if duration == "25" and hd == True:
            error_message = "25s and hd parameters cannot be used together. Please choose only one of them."
            print(error_message)
            return ("", "", json.dumps({"status": "error", "message": error_message}))
            
        if model == "sora-2":
            if duration == "25":  
                error_message = "The sora-2 model does not support 25 second videos. Please use sora-2-pro for 25 second videos."
                print(error_message)
                return ("", "", json.dumps({"status": "error", "message": error_message}))
            if hd:
                error_message = "The sora-2 model does not support HD mode. Please use sora-2-pro for HD videos or disable HD."
                print(error_message)
                return ("", "", json.dumps({"status": "error", "message": error_message}))
      
        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(10)
        
        try:
            has_image = any(img is not None for img in [image1, image2, image3, image4])
            
            if has_image:
                images = []
                for img in [image1, image2, image3, image4]:
                    if img is not None:
                        img_base64 = self.image_to_base64(img)
                        if img_base64:
                            images.append(img_base64)
                
                if not images:
                    error_message = "Failed to process any of the input images"
                    print(error_message)
                    return ("", "", json.dumps({"status": "error", "message": error_message}))
                
                payload = {
                    "prompt": prompt,
                    "model": model,
                    "images": images,
                    "aspect_ratio": aspect_ratio,
                    "duration": duration,
                    "hd": hd,
                    "private": private
                }
                
                if seed > 0:
                    payload["seed"] = seed
                
                endpoint = f"{baseurl}/v2/videos/generations"
            else:
                payload = {
                    "prompt": prompt,
                    "model": model,
                    "aspect_ratio": aspect_ratio,
                    "duration": duration,
                    "hd": hd,
                    "private": private
                }
                
                if seed > 0:
                    payload["seed"] = seed
                    
                endpoint = f"{baseurl}/v2/videos/generations"
            
            pbar.update_absolute(20)
            
            response = requests.post(
                endpoint,
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                return ("", "", json.dumps({"status": "error", "message": error_message}))
                
            result = response.json()
            
            if "task_id" not in result:
                error_message = "No task ID in API response"
                print(error_message)
                return ("", "", json.dumps({"status": "error", "message": error_message}))
            
            task_id = result["task_id"]
            print(f"Task ID: {task_id}")
            
            pbar.update_absolute(30)

            max_attempts = 300  
            attempts = 0
            video_url = None
            
            while attempts < max_attempts:
                time.sleep(10)
                attempts += 1
                
                try:
                    status_response = requests.get(
                        f"{baseurl}/v2/videos/generations/{task_id}",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )
                    
                    if status_response.status_code != 200:
                        continue
                        
                    status_data = status_response.json()

                    progress_text = status_data.get("progress", "0%")
                    try:
                        if progress_text.endswith('%'):
                            progress_value = int(progress_text[:-1])
                            pbar_value = min(90, 30 + int(progress_value * 0.6))
                            pbar.update_absolute(pbar_value)
                    except (ValueError, AttributeError):
                        progress_value = min(80, 30 + (attempts * 50 // max_attempts))
                        pbar.update_absolute(progress_value)
                    
                    status = status_data.get("status", "")
                    
                    if status == "SUCCESS":
                        if "data" in status_data and "output" in status_data["data"]:
                            video_url = status_data["data"]["output"]
                            break
                            
                    elif status == "FAILURE":
                        fail_reason = status_data.get("fail_reason", "Unknown error")
                        error_message = f"Video generation failed: {fail_reason}"
                        print(error_message)
                        return ("", "", json.dumps({"status": "error", "message": error_message, "task_id": task_id}))
                        
                except Exception as e:
                    print(f"Error checking task status: {str(e)}")
            
            if not video_url:
                error_message = f"Failed to get video URL after {max_attempts} attempts"
                print(error_message)
                return ("", "", json.dumps({"status": "error", "message": error_message, "task_id": task_id}))
            
            video_adapter = ComflyVideoAdapter(video_url)
            
            pbar.update_absolute(100)
            
            response_data = {
                "status": "success",
                "task_id": task_id,
                "prompt": prompt,
                "model": model,
                "aspect_ratio": aspect_ratio,
                "duration": duration,
                "hd": hd,
                "private": private,
                "video_url": video_url
            }
            
            return (video_adapter, video_url, json.dumps(response_data))
            
        except Exception as e:
            error_message = f"Error in video generation: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            return ("", "", json.dumps({"status": "error", "message": error_message}))


class Comfly_sora2_chat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["sora-2", "sora-2-pro"], {"default": "sora-2"}),
                "duration": (["10", "15", "25"], {"default": "15"}),
                "orientation": (["portrait", "landscape"], {"default": "portrait"})
            },
            "optional": {
                "image": ("IMAGE",),
                "hd": ("BOOLEAN", {"default": False}),
                "apikey": ("STRING", {"default": ""}),
                # "apikey": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647})
            }
        }
    
    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "gif_url", "response")
    FUNCTION = "generate_video"
    CATEGORY = "RunNode/OpenAI"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 900

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
    
    def generate_video(self, prompt, model="sora-2", duration="15", orientation="portrait", 
                      image=None, hd=False, apikey="", seed=0):
        if apikey.strip():
            self.api_key = apikey
            config = get_config()
            config['api_key'] = apikey
            save_config(config)
            
        if not self.api_key:
            error_response = {"status": "error", "message": "API key not provided or not found in config"}
            return ("", "", "", json.dumps(error_response))

        if duration == "25" and hd:
            error_message = "25s and hd parameters cannot be used together. Please choose only one of them."
            print(error_message)
            return ("", "", "", json.dumps({"status": "error", "message": error_message}))
 
        if model == "sora-2":
            if duration == "25":
                error_message = "The sora-2 model does not support 25 second videos. Please use sora-2-pro for 25 second videos."
                print(error_message)
                return ("", "", "", json.dumps({"status": "error", "message": error_message}))
            if hd:
                error_message = "The sora-2 model does not support HD mode. Please use sora-2-pro for HD videos or disable HD."
                print(error_message)
                return ("", "", "", json.dumps({"status": "error", "message": error_message}))
      
        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(10)
        
        try:
            final_prompt = f"高清，{prompt}" if hd else prompt

            content = [
                {"type": "text", "text": final_prompt}
            ]

            if image is not None:
                image_base64 = self.image_to_base64(image)
                if image_base64:
                    content.append({
                        "type": "image_url", 
                        "image_url": {"url": image_base64}
                    })
            
            messages = [{"role": "user", "content": content}]
            
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": 8192,
                "temperature": 0.5,
                "top_p": 1,
                "presence_penalty": 0,
                "stream": True
            }

            if seed > 0:
                payload["seed"] = seed
            
            pbar.update_absolute(20)

            response = requests.post(
                f"{baseurl}/v1/chat/completions",
                headers=self.get_headers(),
                json=payload,
                stream=True,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                return ("", "", "", json.dumps({"status": "error", "message": error_message}))

            full_response = ""
            task_id = None
            data_preview_url = None
            
            for line in response.iter_lines():
                if line:
                    line_text = line.decode("utf-8")
                    if line_text.startswith("data: "):
                        data_str = line_text[6:]
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and data['choices']:
                                delta = data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    content = delta['content']
                                    full_response += content

                                    if not task_id:
                                        task_id_match = re.search(r"ID: `(task_[a-zA-Z0-9]+)`", full_response)
                                        if task_id_match:
                                            task_id = task_id_match.group(1)
                                            print(f"Found task ID: {task_id}")

                                    if not data_preview_url and task_id:
                                        preview_match = re.search(r"\[数据预览\]\((https://asyncdata.net/web/[^)]+)\)", full_response)
                                        if preview_match:
                                            data_preview_url = preview_match.group(1)
                                            print(f"Found data preview URL: {data_preview_url}")
                                            pbar.update_absolute(30)
                                            break  
                        except json.JSONDecodeError:
                            continue
            
            if not task_id:
                error_message = "Failed to obtain task ID from the response"
                print(error_message)
                return ("", "", "", json.dumps({"status": "error", "message": error_message, "response": full_response}))
            
            pbar.update_absolute(40)

            max_attempts = 120  
            attempts = 0
            video_url = None
            gif_url = None
            
            while attempts < max_attempts:
                time.sleep(10)
                attempts += 1
                
                try:
                    status_response = requests.get(
                        f"https://asyncdata.net/api/share/{task_id}",
                        timeout=self.timeout
                    )
                    
                    if status_response.status_code != 200:
                        continue
                        
                    status_data = status_response.json()

                    content_data = status_data.get("content", {})
                    progress = content_data.get("progress", 0)
                    status = content_data.get("status", "")

                    if progress > 0:
                        pbar_value = min(90, 40 + int(progress * 0.5))
                        pbar.update_absolute(pbar_value)
                    else:
                        progress_value = min(80, 40 + (attempts * 40 // max_attempts))
                        pbar.update_absolute(progress_value)

                    if status == "completed" and "url" in content_data:
                        video_url = content_data.get("url")
                        gif_url = content_data.get("gif_url", "")
                        break
                        
                except Exception as e:
                    print(f"Error checking task status: {str(e)}")
            
            if not video_url:
                error_message = f"Failed to get video URL after {max_attempts} attempts"
                print(error_message)
                return ("", "", "", json.dumps({"status": "error", "message": error_message, "task_id": task_id}))

            video_adapter = ComflyVideoAdapter(video_url)
            
            pbar.update_absolute(100)

            response_data = {
                "status": "success",
                "task_id": task_id,
                "prompt": prompt,
                "model": model,
                "duration": duration,
                "orientation": orientation,
                "hd": hd,
                "seed": seed if seed > 0 else "auto",
                "video_url": video_url,
                "gif_url": gif_url
            }
            
            return (video_adapter, video_url, gif_url, json.dumps(response_data))
            
        except Exception as e:
            error_message = f"Error in video generation: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            return ("", "", "", json.dumps({"status": "error", "message": error_message}))


class Comfly_sora2_character:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_url": ("STRING", {"multiline": False}),
                "timestamps": ("STRING", {"default": "1,3", "multiline": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                # "api_key": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("character_id", "username", "permalink", "profile_picture_url", "response")
    FUNCTION = "create_character"
    CATEGORY = "RunNode/OpenAI"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def create_character(self, video_url, timestamps="1,3", seed=0, api_key=""):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            save_config(config)
            
        if not self.api_key:
            error_response = {"status": "error", "message": "API key not provided or not found in config"}
            return ("", "", "", "", json.dumps(error_response))
            
        try:
            if not timestamps or "," not in timestamps:
                error_message = "Timestamps must be in format 'start,end' (e.g. '1,3')"
                print(error_message)
                return ("", "", "", "", json.dumps({"status": "error", "message": error_message}))
            
            try:
                start_time, end_time = map(float, timestamps.split(","))
                duration = end_time - start_time

                if duration < 1:
                    error_message = "Duration must be at least 1 second (minimum difference between start and end)"
                    print(error_message)
                    return ("", "", "", "", json.dumps({"status": "error", "message": error_message}))
                    
                if duration > 3:
                    error_message = "Duration must be at most 3 seconds (maximum difference between start and end)"
                    print(error_message)
                    return ("", "", "", "", json.dumps({"status": "error", "message": error_message}))
                    
            except ValueError:
                error_message = "Invalid timestamps format. Use numbers separated by comma (e.g. '1.5,3.2')"
                print(error_message)
                return ("", "", "", "", json.dumps({"status": "error", "message": error_message}))

            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)
            
            payload = {
                "url": video_url,
                "timestamps": timestamps
            }
            
            if seed > 0:
                payload["seed"] = seed
                
            pbar.update_absolute(30)
            
            response = requests.post(
                f"{baseurl}/sora/v1/characters",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(60)
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                return ("", "", "", "", json.dumps({"status": "error", "message": error_message}))
                
            result = response.json()
            
            pbar.update_absolute(90)

            character_id = result.get("id", "")
            username = result.get("username", "")
            permalink = result.get("permalink", "")
            profile_picture_url = result.get("profile_picture_url", "")
            
            if not character_id:
                error_message = "No character ID returned from API"
                print(error_message)
                return ("", "", "", "", json.dumps({"status": "error", "message": error_message}))
            
            pbar.update_absolute(100)
            
            response_data = {
                "status": "success",
                "character_id": character_id,
                "username": username,
                "permalink": permalink,
                "profile_picture_url": profile_picture_url,
                "video_url": video_url,
                "timestamps": timestamps,
                "duration": f"{duration:.1f}s",
                "seed": seed if seed > 0 else "auto"
            }
            
            print(f"Character created successfully. ID: {character_id}, Username: {username}")
            
            return (character_id, username, permalink, profile_picture_url, json.dumps(response_data))
            
        except Exception as e:
            error_message = f"Error creating character: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            return ("", "", "", "", json.dumps({"status": "error", "message": error_message}))


class OpenAISoraAPIPlus:
    """
    ComfyUI自定义节点：ai.t8star.cn Sora-2 视频生成（OpenAI兼容流式接口）
    - 参考 openai_chat_api_node.py 的结构与风格
    - 通过 ai.t8star.cn 的 /chat/completions 接口，以 stream=True 获取流式增量内容
    - 适配示例返回：每行均为 JSON，字段为 choices[0].delta.content
    - 超时时间：600 秒（10 分钟）
    输入参数：
      - base_url: 默认 https://ai.t8star.cn/v1
      - model: 默认 sora_video2
      - api_key: 必填
      - system_prompt: 可选，用于设定系统指令
      - user_prompt: 必填，视频生成描述
    输出：
      - reasoning_content: 保留为空（""），与参考节点保持一致
      - answer: 汇总的增量内容（通常包含进度与最终信息）
      - tokens_usage: 由于返回中未提供 usage，这里一般为空字符串
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {"default": "sora_video2", "multiline": False}),
                # "api_key": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "user_prompt": ("STRING", {"multiline": True, "default": "请描述要生成的视频内容"}),
            },
            "optional": {
                "base_url": ("STRING", {"default": "", "multiline": False}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                # 可选图像输入：提供则走“图生视频（image-to-video）”，不提供则为“文生视频（text-to-video）”
                "aspect_ratio": ("STRING", {"default": "16:9", "multiline": False, "options": ["16:9", "9:16"]}),
                "hd": ("BOOLEAN", {"default": True}),
                "duration": ("INT", {"default": 15, "options": [10, 15]}),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "tokens_usage")
    FUNCTION = "generate"
    CATEGORY = "RunNode/OpenAI"

    def generate(self, base_url, model, api_key, user_prompt,image=None,hd=True,duration=15,aspect_ratio="16:9"):
        """
        调用 ai.t8star.cn 的 sora-2 模型进行视频生成（流式）。
        请求：
          POST {base_url}/chat/completions
          headers:
            - Authorization: Bearer {api_key}
            - Accept: application/json
            - Content-Type: application/json
          json:
            {
              "model": model,
              "messages": [{"role": "system","content": system_prompt}, {"role":"user","content": user_prompt}],
              "stream": true
            }
        解析：
          - 逐行读取，每行是 JSON，取 choices[0].delta.content 累加
          - 若流式无内容，降级为非流式请求（stream=false）再解析
        超时：
          - timeout=600 秒
        """
        if not base_url.strip():  
            base_url = "https://ai.t8star.cn/v1"
        if not api_key:
            return (None, "", "错误：未配置API Key，请在节点参数中设置 api_key")
        if not base_url:
            return (None, "", "错误：未配置 base_url，请在节点参数中设置 base_url")
        if not user_prompt.strip():
            return (None, "", "错误：user_prompt 为空，请提供视频描述")

        try:
            headers = self._build_headers(api_key)
            api_url = f"{base_url.rstrip('/')}/chat/completions"

            # 构建聊天内容：
            # - 若提供 image：按 OpenAI 多模态格式使用 content 数组，携带文本与图片
            # - 若不提供 image：保持纯文本 content 字符串，兼容各类兼容接口
            if image is not None:
                try:
                    from io import BytesIO
                    import base64
                    from PIL import Image as _PILImage  # 仅用于确保PIL可用
                    pil_image = self._convert_to_pil(image)
                    buf = BytesIO()
                    pil_image.save(buf, format="PNG")
                    buf.seek(0)
                    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    base64_url = f"data:image/png;base64,{image_base64}"
                    content = [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": base64_url,
                                "detail": "high"
                            }
                        }
                    ]
                    print(f"[OpenAISoraAPI] 图生视频模式: 已附带输入图像，尺寸={pil_image.size}, base64长度={len(image_base64)}")
                except Exception as e:
                    return (None, f"输入图像处理失败: {e}", "")
                messages = [{"role": "user", "content": content}]
            else:
                print(f"[OpenAISoraAPI] 文生视频模式: 纯文本提示词")
                messages = [{"role": "user", "content": user_prompt}]

            payload = {
                "model": model,
                "messages": messages,
                "stream": True
            }

            print(f"[OpenAISoraAPI] 请求: {api_url} (chat/completions, stream=True)")
            print(f"[OpenAISoraAPI] 模型: {model}")
            # 打印裁剪后的提示词，便于用户确认任务内容
            _preview = (user_prompt[:120] + "...") if len(user_prompt) > 120 else user_prompt
            print(f"[OpenAISoraAPI] 提交Sora任务 | 提示词: {_preview}")
            # 打印精简后的载荷（避免输出完整base64）
            try:
                print(f"[OpenAISoraAPI] 请求载荷(精简): {self._safe_json_dumps(payload)}")
            except Exception:
                pass
            resp = requests.post(api_url, headers=headers, json=payload, timeout=600, stream=True)
            print(f"[OpenAISoraAPI] 响应状态码: {resp.status_code}")

            if resp.status_code != 200:
                return (None, f"API错误 (状态码: {resp.status_code}): {resp.text}", "")

            reasoning_content, answer, tokens_usage = self._parse_302_stream(resp)

            # 若流式无内容，降级为非流式
            if not answer:
                try:
                    safe_payload = dict(payload)
                    safe_payload["stream"] = False
                    print(f"[OpenAISoraAPI] 流式无增量，降级为非流式请求")
                    resp2 = requests.post(api_url, headers=headers, json=safe_payload, timeout=600)
                    if resp2.status_code == 200:
                        rc2, answer2, tu2 = self._parse_non_stream(resp2)
                        video_url2 = self._extract_video_url(answer2)
                        video2 = self._download_and_convert_video(video_url2)
                        return (video2, video_url2 or "", tu2)
                    else:
                        return (None, f"非流式降级失败 (状态码: {resp2.status_code}): {resp2.text}", tokens_usage)
                except Exception as _e:
                    print(f"[OpenAISoraAPI] 非流式降级异常: {_e}")

            # 正常流式结果：提取视频URL并下载
            video_url = self._extract_video_url(answer)
            video_output = self._download_and_convert_video(video_url)
            return (video_output, video_url or "", tokens_usage)
        except requests.exceptions.ConnectTimeout as e:
            return (None, f"网络连接超时: 无法连接到API服务器。请检查网络连接或代理。错误: {e}", "")
        except requests.exceptions.Timeout as e:
            return (None, f"请求超时: API响应时间过长。请稍后重试。错误: {e}", "")
        except requests.exceptions.ConnectionError as e:
            return (None, f"网络连接错误: 无法建立到API的连接。请检查网络设置。错误: {e}", "")
        except requests.exceptions.RequestException as e:
            return (None, f"API请求失败: {e}", "")
        except Exception as e:
            return (None, f"处理失败: {e}", "")

    def _build_headers(self, api_key: str):
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _convert_to_pil(self, image):
        """
        将ComfyUI的IMAGE/常见输入转换为PIL Image（RGB）。
        - 支持 torch.Tensor (N,H,W,3) 或 (H,W,3)，数值范围[0,1]或[0,255]
        - 支持PIL Image
        - 支持numpy数组
        """
        try:
            from PIL import Image
            if hasattr(image, "cpu"):  # torch.Tensor
                import torch
                import numpy as np
                t = image
                if t.dim() == 4:
                    t = t[0]
                # 期望 (H,W,3)
                if t.shape[-1] == 3:
                    arr = t.detach().cpu().numpy()
                elif t.shape[0] == 3 and t.dim() == 3:
                    # 兼容 (3,H,W) -> (H,W,3)
                    arr = t.detach().cpu().numpy().transpose(1, 2, 0)
                else:
                    raise ValueError(f"不支持的Tensor形状: {tuple(t.shape)}")
                # 归一化
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).clip(0, 255).astype("uint8")
                else:
                    arr = arr.clip(0, 255).astype("uint8")
                img = Image.fromarray(arr)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                return img
            elif hasattr(image, "save"):  # PIL
                from PIL import Image as _Image
                img = image
                if img.mode != "RGB":
                    img = img.convert("RGB")
                return img
            else:
                import numpy as np
                if isinstance(image, np.ndarray):
                    arr = image
                    if arr.ndim == 3 and arr.shape[0] == 3:
                        arr = arr.transpose(1, 2, 0)
                    if arr.max() <= 1.0:
                        arr = (arr * 255.0).clip(0, 255).astype("uint8")
                    else:
                        arr = arr.clip(0, 255).astype("uint8")
                    from PIL import Image
                    img = Image.fromarray(arr)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    return img
                raise ValueError(f"不支持的图像类型: {type(image)}")
        except Exception as e:
            print(f"[OpenAISoraAPI] 图像转换失败: {e}")
            raise

    def _safe_json_dumps(self, obj, ensure_ascii=False, indent=2):
        """
        序列化JSON时截断超长/疑似base64字段，避免日志刷屏。
        """
        import json as _json

        def _truncate_base64(value: str):
            if not isinstance(value, str):
                return value
            if len(value) > 100 and (
                value.startswith("data:image/") or
                value[:8] in ("iVBORw0K", "/9j/")  # 常见PNG/JPEG开头
            ):
                return value[:50] + f"... (len={len(value)})"
            return value

        def _walk(v):
            if isinstance(v, dict):
                return {k: _walk(_truncate_base64(val)) for k, val in v.items()}
            if isinstance(v, list):
                return [_walk(_truncate_base64(x)) for x in v]
            return _truncate_base64(v)

        return _json.dumps(_walk(obj), ensure_ascii=ensure_ascii, indent=indent)

    def _parse_302_stream(self, resp):
        """
        解析 ai.t8star.cn 的流式响应。
        示例行：
          {"choices":[{"delta":{"content":"...","role":"assistant"},"index":0}],"id":"...","model":"sora-2","object":"chat.completion.chunk"}
        策略：
          - 逐行解析 JSON
          - 提取 choices[0].delta.content 累加
          - 无 usage 字段，tokens_usage 保持为空
        """
        answer_parts = []
        tokens_usage = ""
        # 进度与心跳跟踪
        last_progress = -1   # 最近一次打印的百分比进度（0-100）
        chunk_count = 0      # 已接收的增量块数量
        printed_url = False  # 是否已打印过URL
        try:
            for raw in resp.iter_lines(decode_unicode=True):
                if raw is None:
                    continue
                line = raw.strip()
                if not line:
                    continue

                # 可能伴随时间戳行，如 "21:56:01"，跳过非 JSON 行
                if not (line.startswith("{") and line.endswith("}")):
                    continue

                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "choices" in payload and isinstance(payload["choices"], list) and payload["choices"]:
                    delta = payload["choices"][0].get("delta", {})
                    if isinstance(delta, dict):
                        piece = delta.get("content")
                        if isinstance(piece, str) and piece:
                            # 进度日志：尽量识别诸如 "进度 36.."、"41.."、"60.." 等
                            text = piece.strip()
                            # 优先匹配包含“进度”的片段
                            prog_candidates = []
                            if "进度" in text or "progress" in text.lower():
                                prog_candidates = re.findall(r'(\d{1,3})(?=%|\.{2,})', text)
                                if not prog_candidates:
                                    prog_candidates = re.findall(r'进度[^0-9]*?(\d{1,3})', text)
                            else:
                                # 一般性匹配 "41.." 这类
                                prog_candidates = re.findall(r'(\d{1,3})(?=%|\.{2,})', text)

                            # 过滤到 0-100 的最大值作为当前进度
                            curr_prog = None
                            for p in prog_candidates:
                                try:
                                    v = int(p)
                                    if 0 <= v <= 100:
                                        curr_prog = v if (curr_prog is None or v > curr_prog) else curr_prog
                                except Exception:
                                    pass
                            if curr_prog is not None and curr_prog > last_progress:
                                last_progress = curr_prog
                                print(f"[OpenAISoraAPI][{time.strftime('%H:%M:%S')}] 任务进度: {last_progress}%")

                            # 首次发现 URL 则提示
                            if not printed_url and ("http://" in text or "https://" in text):
                                urls = re.findall(r'https?://\S+', text)
                                if urls:
                                    print(f"[OpenAISoraAPI] 可能的视频URL: {urls[0]}")
                                    printed_url = True

                            # 心跳：每收到一定数量块打印一次累计长度
                            chunk_count += 1
                            if chunk_count % 20 == 0:
                                total_len = sum(len(x) for x in answer_parts) + len(text)
                                print(f"[OpenAISoraAPI] 流式接收中... 已接收 {chunk_count} 块，累计字符 {total_len}")

                            answer_parts.append(piece)

            # 合并并做简单的编码清理
            answer = self._normalize_text("".join(answer_parts).strip())
            return ("", answer, tokens_usage)
        except Exception as e:
            return ("", f"流式解析失败: {e}", tokens_usage)

    def _parse_non_stream(self, resp):
        """
        非流式响应解析（兼容 OpenAI chat/completions）
        预期结构：
          {"choices":[{"message":{"role":"assistant","content":"..."},"finish_reason":"..." }], "usage": {...}}
        """
        try:
            if resp.status_code != 200:
                return ("", f"API错误 (状态码: {resp.status_code}): {resp.text}", "")
            if not resp.text.strip():
                return ("", "API返回空响应", "")

            try:
                data = resp.json()
            except json.JSONDecodeError as json_error:
                return ("", f"API响应格式错误: {str(json_error)}", "")

            # 错误字段
            if "error" in data and data["error"]:
                err = data["error"]
                msg = err.get("message", str(err))
                typ = err.get("type", "unknown_error")
                return ("", f"API错误 ({typ}): {msg}", "")

            usage = data.get("usage", {})
            tokens_usage = self._format_tokens_usage(usage)

            if "choices" in data and data["choices"]:
                message = data["choices"][0].get("message", {})
                content = message.get("content", "")
                if not content:
                    finish_reason = data["choices"][0].get("finish_reason", "")
                    return ("", f"未返回内容，finish_reason={finish_reason}", tokens_usage)
                reasoning_content, answer = self._parse_content_tags(content)
                return ("", answer, tokens_usage)

            return ("", "API未返回choices内容", tokens_usage)
        except Exception as e:
            return ("", f"响应解析失败: {e}", "")

    def _parse_content_tags(self, content: str):
        """
        复用与参考节点一致的标签解析逻辑：
        - <think>...</think> 抽取思考
        - <answer>...</answer> 抽取答案
        - <reasoning>...</reasoning> 抽取思考
        """
        try:
            think_pattern = r'<think>(.*?)</think>'
            think_match = re.search(think_pattern, content, re.DOTALL)
            if think_match:
                reasoning_content = think_match.group(1).strip()
                answer = content.replace(think_match.group(0), "").strip()
                return (reasoning_content, answer)

            answer_pattern = r'<answer>(.*?)</answer>'
            answer_match = re.search(answer_pattern, content, re.DOTALL)
            if answer_match:
                return ("", answer_match.group(1).strip())

            answer_pattern_open = r'<answer>(.*)'
            answer_match_open = re.search(answer_pattern_open, content, re.DOTALL)
            if answer_match_open:
                return ("", answer_match_open.group(1).strip())

            reasoning_pattern = r'<reasoning>(.*?)</reasoning>'
            reasoning_match = re.search(reasoning_pattern, content, re.DOTALL)
            if reasoning_match:
                reasoning_content = reasoning_match.group(1).strip()
                answer = content.replace(reasoning_match.group(0), "").strip()
                return (reasoning_content, answer)

            return ("", content.strip())
        except Exception:
            return ("", content.strip())

    def _format_tokens_usage(self, usage):
        if not usage:
            return ""
        total_tokens = usage.get('total_tokens') or usage.get('total') or usage.get('tokens') or '-'
        prompt_tokens = (
            usage.get('prompt_tokens')
            or usage.get('input_tokens')
            or (usage.get('input', {}) if isinstance(usage.get('input'), dict) else None)
            or usage.get('prompt')
            or '-'
        )
        if isinstance(prompt_tokens, dict):
            prompt_tokens = prompt_tokens.get('tokens') or prompt_tokens.get('count') or '-'
        completion_tokens = (
            usage.get('completion_tokens')
            or usage.get('output_tokens')
            or (usage.get('output', {}) if isinstance(usage.get('output'), dict) else None)
            or usage.get('completion')
            or '-'
        )
        if isinstance(completion_tokens, dict):
            completion_tokens = completion_tokens.get('tokens') or completion_tokens.get('count') or '-'
        return f"total_tokens={total_tokens}, input_tokens={prompt_tokens}, output_tokens={completion_tokens}"

    def _normalize_text(self, s: str) -> str:
        if not isinstance(s, str) or not s:
            return s or ""
        sample = s[:8]
        suspicious = ("Ã", "å", "æ", "ç", "ð", "þ")
        if any(ch in sample for ch in suspicious):
            try:
                return s.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
            except Exception:
                return s
        return s

    def _extract_video_url(self, text: str) -> Optional[str]:
        """
        从返回文本中提取视频URL。
        优先匹配指向 mp4/webm 等视频资源的URL；若未匹配到，则回退匹配任意 http/https 链接。
        """
        if not isinstance(text, str) or not text:
            return None
        try:
            # 优先匹配视频直链
            m = re.findall(r'(https?://[^\s)>\]]+\.(?:mp4|webm)(?:\?[^\s)>\]]*)?)', text, flags=re.IGNORECASE)
            if m:
                return m[0]
            # 其次匹配 markdown 的 [在线播放](url) 格式
            m2 = re.findall(r'\((https?://[^\s)]+)\)', text, flags=re.IGNORECASE)
            if m2:
                return m2[0]
            # 再次匹配任意 http/https 链接
            m3 = re.findall(r'(https?://[^\s)>\]]+)', text, flags=re.IGNORECASE)
            if m3:
                return m3[0]
            return None
        except Exception:
            return None

    def _download_and_convert_video(self, video_url: str) -> Optional[Any]:
        """
        下载视频URL并转换为VIDEO对象，参考 jimeng_video_node.py 的实现。
        - 校验URL合法性
        - 使用 download_url_to_video_output(video_url, timeout=120)
        - 出错返回 None，保证节点稳定
        """
        try:
            if not video_url or not isinstance(video_url, str):
                print(f"[OpenAISoraAPI] 无效的视频URL: {video_url}")
                return None
            if not video_url.startswith(("http://", "https://")):
                print(f"[OpenAISoraAPI] 不支持的URL格式: {video_url}")
                return None

            print(f"[OpenAISoraAPI] 🎬 开始下载视频: {video_url[:80]}...")
            try:
                video_output = download_url_to_video_output(video_url, timeout=120)
                print(f"[OpenAISoraAPI] ✅ 视频下载完成")
                return video_output
            except Exception as download_error:
                print(f"[OpenAISoraAPI] ❌ 视频下载失败: {download_error}")
                return None
        except Exception as e:
            print(f"[OpenAISoraAPI] 视频下载转换过程出错: {e}")
            return None


class OpenAISoraAPI:
    """
    ComfyUI自定义节点：ai.t8star.cn Sora-2 视频生成（OpenAI兼容流式接口）
    - 参考 openai_chat_api_node.py 的结构与风格
    - 通过 ai.t8star.cn 的 /chat/completions 接口，以 stream=True 获取流式增量内容
    - 适配示例返回：每行均为 JSON，字段为 choices[0].delta.content
    - 超时时间：600 秒（10 分钟）
    输入参数：
      - base_url: 默认 https://ai.t8star.cn/v1
      - model: 默认 sora_video2
      - api_key: 必填
      - system_prompt: 可选，用于设定系统指令
      - user_prompt: 必填，视频生成描述
    输出：
      - reasoning_content: 保留为空（""），与参考节点保持一致
      - answer: 汇总的增量内容（通常包含进度与最终信息）
      - tokens_usage: 由于返回中未提供 usage，这里一般为空字符串
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {"default": "sora_video2", "multiline": False}),
                # "api_key": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "user_prompt": ("STRING", {"multiline": True, "default": "请描述要生成的视频内容"}),
                #"hd": (["true", "false"], {"default": "false"}),
                #"duration": (["10", "15"], {"default": "15"}),
                #"aspect_ratio": (["16:9", "9:16"], {"default": "9:16"}),
            },
            "optional": {
                "base_url": ("STRING", {"default": "", "multiline": False}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                # 可选图像输入：提供则走“图生视频（image-to-video）”，不提供则为“文生视频（text-to-video）”
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "tokens_usage")
    FUNCTION = "generate"
    CATEGORY = "RunNode/OpenAI"

    def generate(self, base_url, model, api_key, user_prompt, image=None):
        """
        调用 ai.t8star.cn 的 sora-2 模型进行视频生成（流式）。
        请求：
          POST {base_url}/chat/completions
          headers:
            - Authorization: Bearer {api_key}
            - Accept: application/json
            - Content-Type: application/json
          json:
            {
              "model": model,
              "messages": [{"role": "system","content": system_prompt}, {"role":"user","content": user_prompt}],
              "stream": true
            }
        解析：
          - 逐行读取，每行是 JSON，取 choices[0].delta.content 累加
          - 若流式无内容，降级为非流式请求（stream=false）再解析
        超时：
          - timeout=600 秒
        """
        if not base_url.strip():  
            base_url = "https://ai.t8star.cn/v1"
        if not api_key:
            return (None, "", "错误：未配置API Key，请在节点参数中设置 api_key")
        if not base_url:
            return (None, "", "错误：未配置 base_url，请在节点参数中设置 base_url")
        if not user_prompt.strip():
            return (None, "", "错误：user_prompt 为空，请提供视频描述")

        try:
            headers = self._build_headers(api_key)
            api_url = f"{base_url.rstrip('/')}/chat/completions"

            # 构建聊天内容：
            # - 若提供 image：按 OpenAI 多模态格式使用 content 数组，携带文本与图片
            # - 若不提供 image：保持纯文本 content 字符串，兼容各类兼容接口
            if image is not None:
                try:
                    from io import BytesIO
                    import base64
                    from PIL import Image as _PILImage  # 仅用于确保PIL可用
                    pil_image = self._convert_to_pil(image)
                    buf = BytesIO()
                    pil_image.save(buf, format="PNG")
                    buf.seek(0)
                    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    base64_url = f"data:image/png;base64,{image_base64}"
                    content = [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": base64_url,
                                "detail": "high"
                            }
                        }
                    ]
                    print(f"[OpenAISoraAPI] 图生视频模式: 已附带输入图像，尺寸={pil_image.size}, base64长度={len(image_base64)}")
                except Exception as e:
                    return (None, f"输入图像处理失败: {e}", "")
                messages = [{"role": "user", "content": content}]
            else:
                print(f"[OpenAISoraAPI] 文生视频模式: 纯文本提示词")
                messages = [{"role": "user", "content": user_prompt}]

            payload = {
                "model": model,
                "messages": messages,
                "stream": True
            }

            print(f"[OpenAISoraAPI] 请求: {api_url} (chat/completions, stream=True)")
            print(f"[OpenAISoraAPI] 模型: {model}")
            # 打印裁剪后的提示词，便于用户确认任务内容
            _preview = (user_prompt[:120] + "...") if len(user_prompt) > 120 else user_prompt
            print(f"[OpenAISoraAPI] 提交Sora任务 | 提示词: {_preview}")
            # 打印精简后的载荷（避免输出完整base64）
            try:
                print(f"[OpenAISoraAPI] 请求载荷(精简): {self._safe_json_dumps(payload)}")
            except Exception:
                pass
            resp = requests.post(api_url, headers=headers, json=payload, timeout=600, stream=True)
            print(f"[OpenAISoraAPI] 响应状态码: {resp.status_code}")

            if resp.status_code != 200:
                return (None, f"API错误 (状态码: {resp.status_code}): {resp.text}", "")

            reasoning_content, answer, tokens_usage = self._parse_302_stream(resp)

            # 若流式无内容，降级为非流式
            if not answer:
                try:
                    safe_payload = dict(payload)
                    safe_payload["stream"] = False
                    print(f"[OpenAISoraAPI] 流式无增量，降级为非流式请求")
                    resp2 = requests.post(api_url, headers=headers, json=safe_payload, timeout=600)
                    if resp2.status_code == 200:
                        rc2, answer2, tu2 = self._parse_non_stream(resp2)
                        video_url2 = self._extract_video_url(answer2)
                        video2 = self._download_and_convert_video(video_url2)
                        return (video2, video_url2 or "", tu2)
                    else:
                        return (None, f"非流式降级失败 (状态码: {resp2.status_code}): {resp2.text}", tokens_usage)
                except Exception as _e:
                    print(f"[OpenAISoraAPI] 非流式降级异常: {_e}")

            # 正常流式结果：提取视频URL并下载
            video_url = self._extract_video_url(answer)
            video_output = self._download_and_convert_video(video_url)
            return (video_output, video_url or "", tokens_usage)
        except requests.exceptions.ConnectTimeout as e:
            return (None, f"网络连接超时: 无法连接到API服务器。请检查网络连接或代理。错误: {e}", "")
        except requests.exceptions.Timeout as e:
            return (None, f"请求超时: API响应时间过长。请稍后重试。错误: {e}", "")
        except requests.exceptions.ConnectionError as e:
            return (None, f"网络连接错误: 无法建立到API的连接。请检查网络设置。错误: {e}", "")
        except requests.exceptions.RequestException as e:
            return (None, f"API请求失败: {e}", "")
        except Exception as e:
            return (None, f"处理失败: {e}", "")

    def _build_headers(self, api_key: str):
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _convert_to_pil(self, image):
        """
        将ComfyUI的IMAGE/常见输入转换为PIL Image（RGB）。
        - 支持 torch.Tensor (N,H,W,3) 或 (H,W,3)，数值范围[0,1]或[0,255]
        - 支持PIL Image
        - 支持numpy数组
        """
        try:
            from PIL import Image
            if hasattr(image, "cpu"):  # torch.Tensor
                import torch
                import numpy as np
                t = image
                if t.dim() == 4:
                    t = t[0]
                # 期望 (H,W,3)
                if t.shape[-1] == 3:
                    arr = t.detach().cpu().numpy()
                elif t.shape[0] == 3 and t.dim() == 3:
                    # 兼容 (3,H,W) -> (H,W,3)
                    arr = t.detach().cpu().numpy().transpose(1, 2, 0)
                else:
                    raise ValueError(f"不支持的Tensor形状: {tuple(t.shape)}")
                # 归一化
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).clip(0, 255).astype("uint8")
                else:
                    arr = arr.clip(0, 255).astype("uint8")
                img = Image.fromarray(arr)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                return img
            elif hasattr(image, "save"):  # PIL
                from PIL import Image as _Image
                img = image
                if img.mode != "RGB":
                    img = img.convert("RGB")
                return img
            else:
                import numpy as np
                if isinstance(image, np.ndarray):
                    arr = image
                    if arr.ndim == 3 and arr.shape[0] == 3:
                        arr = arr.transpose(1, 2, 0)
                    if arr.max() <= 1.0:
                        arr = (arr * 255.0).clip(0, 255).astype("uint8")
                    else:
                        arr = arr.clip(0, 255).astype("uint8")
                    from PIL import Image
                    img = Image.fromarray(arr)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    return img
                raise ValueError(f"不支持的图像类型: {type(image)}")
        except Exception as e:
            print(f"[OpenAISoraAPI] 图像转换失败: {e}")
            raise

    def _safe_json_dumps(self, obj, ensure_ascii=False, indent=2):
        """
        序列化JSON时截断超长/疑似base64字段，避免日志刷屏。
        """
        import json as _json

        def _truncate_base64(value: str):
            if not isinstance(value, str):
                return value
            if len(value) > 100 and (
                value.startswith("data:image/") or
                value[:8] in ("iVBORw0K", "/9j/")  # 常见PNG/JPEG开头
            ):
                return value[:50] + f"... (len={len(value)})"
            return value

        def _walk(v):
            if isinstance(v, dict):
                return {k: _walk(_truncate_base64(val)) for k, val in v.items()}
            if isinstance(v, list):
                return [_walk(_truncate_base64(x)) for x in v]
            return _truncate_base64(v)

        return _json.dumps(_walk(obj), ensure_ascii=ensure_ascii, indent=indent)

    def _parse_302_stream(self, resp):
        """
        解析 ai.t8star.cn 的流式响应。
        示例行：
          {"choices":[{"delta":{"content":"...","role":"assistant"},"index":0}],"id":"...","model":"sora-2","object":"chat.completion.chunk"}
        策略：
          - 逐行解析 JSON
          - 提取 choices[0].delta.content 累加
          - 无 usage 字段，tokens_usage 保持为空
        """
        answer_parts = []
        tokens_usage = ""
        # 进度与心跳跟踪
        last_progress = -1   # 最近一次打印的百分比进度（0-100）
        chunk_count = 0      # 已接收的增量块数量
        printed_url = False  # 是否已打印过URL
        try:
            for raw in resp.iter_lines(decode_unicode=True):
                if raw is None:
                    continue
                line = raw.strip()
                if not line:
                    continue

                # 可能伴随时间戳行，如 "21:56:01"，跳过非 JSON 行
                if not (line.startswith("{") and line.endswith("}")):
                    continue

                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "choices" in payload and isinstance(payload["choices"], list) and payload["choices"]:
                    delta = payload["choices"][0].get("delta", {})
                    if isinstance(delta, dict):
                        piece = delta.get("content")
                        if isinstance(piece, str) and piece:
                            # 进度日志：尽量识别诸如 "进度 36.."、"41.."、"60.." 等
                            text = piece.strip()
                            # 优先匹配包含“进度”的片段
                            prog_candidates = []
                            if "进度" in text or "progress" in text.lower():
                                prog_candidates = re.findall(r'(\d{1,3})(?=%|\.{2,})', text)
                                if not prog_candidates:
                                    prog_candidates = re.findall(r'进度[^0-9]*?(\d{1,3})', text)
                            else:
                                # 一般性匹配 "41.." 这类
                                prog_candidates = re.findall(r'(\d{1,3})(?=%|\.{2,})', text)

                            # 过滤到 0-100 的最大值作为当前进度
                            curr_prog = None
                            for p in prog_candidates:
                                try:
                                    v = int(p)
                                    if 0 <= v <= 100:
                                        curr_prog = v if (curr_prog is None or v > curr_prog) else curr_prog
                                except Exception:
                                    pass
                            if curr_prog is not None and curr_prog > last_progress:
                                last_progress = curr_prog
                                print(f"[OpenAISoraAPI][{time.strftime('%H:%M:%S')}] 任务进度: {last_progress}%")

                            # 首次发现 URL 则提示
                            if not printed_url and ("http://" in text or "https://" in text):
                                urls = re.findall(r'https?://\S+', text)
                                if urls:
                                    print(f"[OpenAISoraAPI] 可能的视频URL: {urls[0]}")
                                    printed_url = True

                            # 心跳：每收到一定数量块打印一次累计长度
                            chunk_count += 1
                            if chunk_count % 20 == 0:
                                total_len = sum(len(x) for x in answer_parts) + len(text)
                                print(f"[OpenAISoraAPI] 流式接收中... 已接收 {chunk_count} 块，累计字符 {total_len}")

                            answer_parts.append(piece)

            # 合并并做简单的编码清理
            answer = self._normalize_text("".join(answer_parts).strip())
            return ("", answer, tokens_usage)
        except Exception as e:
            return ("", f"流式解析失败: {e}", tokens_usage)

    def _parse_non_stream(self, resp):
        """
        非流式响应解析（兼容 OpenAI chat/completions）
        预期结构：
          {"choices":[{"message":{"role":"assistant","content":"..."},"finish_reason":"..." }], "usage": {...}}
        """
        try:
            if resp.status_code != 200:
                return ("", f"API错误 (状态码: {resp.status_code}): {resp.text}", "")
            if not resp.text.strip():
                return ("", "API返回空响应", "")

            try:
                data = resp.json()
            except json.JSONDecodeError as json_error:
                return ("", f"API响应格式错误: {str(json_error)}", "")

            # 错误字段
            if "error" in data and data["error"]:
                err = data["error"]
                msg = err.get("message", str(err))
                typ = err.get("type", "unknown_error")
                return ("", f"API错误 ({typ}): {msg}", "")

            usage = data.get("usage", {})
            tokens_usage = self._format_tokens_usage(usage)

            if "choices" in data and data["choices"]:
                message = data["choices"][0].get("message", {})
                content = message.get("content", "")
                if not content:
                    finish_reason = data["choices"][0].get("finish_reason", "")
                    return ("", f"未返回内容，finish_reason={finish_reason}", tokens_usage)
                reasoning_content, answer = self._parse_content_tags(content)
                return ("", answer, tokens_usage)

            return ("", "API未返回choices内容", tokens_usage)
        except Exception as e:
            return ("", f"响应解析失败: {e}", "")

    def _parse_content_tags(self, content: str):
        """
        复用与参考节点一致的标签解析逻辑：
        - <think>...</think> 抽取思考
        - <answer>...</answer> 抽取答案
        - <reasoning>...</reasoning> 抽取思考
        """
        try:
            think_pattern = r'<think>(.*?)</think>'
            think_match = re.search(think_pattern, content, re.DOTALL)
            if think_match:
                reasoning_content = think_match.group(1).strip()
                answer = content.replace(think_match.group(0), "").strip()
                return (reasoning_content, answer)

            answer_pattern = r'<answer>(.*?)</answer>'
            answer_match = re.search(answer_pattern, content, re.DOTALL)
            if answer_match:
                return ("", answer_match.group(1).strip())

            answer_pattern_open = r'<answer>(.*)'
            answer_match_open = re.search(answer_pattern_open, content, re.DOTALL)
            if answer_match_open:
                return ("", answer_match_open.group(1).strip())

            reasoning_pattern = r'<reasoning>(.*?)</reasoning>'
            reasoning_match = re.search(reasoning_pattern, content, re.DOTALL)
            if reasoning_match:
                reasoning_content = reasoning_match.group(1).strip()
                answer = content.replace(reasoning_match.group(0), "").strip()
                return (reasoning_content, answer)

            return ("", content.strip())
        except Exception:
            return ("", content.strip())

    def _format_tokens_usage(self, usage):
        if not usage:
            return ""
        total_tokens = usage.get('total_tokens') or usage.get('total') or usage.get('tokens') or '-'
        prompt_tokens = (
            usage.get('prompt_tokens')
            or usage.get('input_tokens')
            or (usage.get('input', {}) if isinstance(usage.get('input'), dict) else None)
            or usage.get('prompt')
            or '-'
        )
        if isinstance(prompt_tokens, dict):
            prompt_tokens = prompt_tokens.get('tokens') or prompt_tokens.get('count') or '-'
        completion_tokens = (
            usage.get('completion_tokens')
            or usage.get('output_tokens')
            or (usage.get('output', {}) if isinstance(usage.get('output'), dict) else None)
            or usage.get('completion')
            or '-'
        )
        if isinstance(completion_tokens, dict):
            completion_tokens = completion_tokens.get('tokens') or completion_tokens.get('count') or '-'
        return f"total_tokens={total_tokens}, input_tokens={prompt_tokens}, output_tokens={completion_tokens}"

    def _normalize_text(self, s: str) -> str:
        if not isinstance(s, str) or not s:
            return s or ""
        sample = s[:8]
        suspicious = ("Ã", "å", "æ", "ç", "ð", "þ")
        if any(ch in sample for ch in suspicious):
            try:
                return s.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
            except Exception:
                return s
        return s

    def _extract_video_url(self, text: str) -> Optional[str]:
        """
        从返回文本中提取视频URL。
        优先匹配指向 mp4/webm 等视频资源的URL；若未匹配到，则回退匹配任意 http/https 链接。
        """
        if not isinstance(text, str) or not text:
            return None
        try:
            # 优先匹配视频直链
            m = re.findall(r'(https?://[^\s)>\]]+\.(?:mp4|webm)(?:\?[^\s)>\]]*)?)', text, flags=re.IGNORECASE)
            if m:
                return m[0]
            # 其次匹配 markdown 的 [在线播放](url) 格式
            m2 = re.findall(r'\((https?://[^\s)]+)\)', text, flags=re.IGNORECASE)
            if m2:
                return m2[0]
            # 再次匹配任意 http/https 链接
            m3 = re.findall(r'(https?://[^\s)>\]]+)', text, flags=re.IGNORECASE)
            if m3:
                return m3[0]
            return None
        except Exception:
            return None

    def _download_and_convert_video(self, video_url: str) -> Optional[Any]:
        """
        下载视频URL并转换为VIDEO对象，参考 jimeng_video_node.py 的实现。
        - 校验URL合法性
        - 使用 download_url_to_video_output(video_url, timeout=120)
        - 出错返回 None，保证节点稳定
        """
        try:
            if not video_url or not isinstance(video_url, str):
                print(f"[OpenAISoraAPI] 无效的视频URL: {video_url}")
                return None
            if not video_url.startswith(("http://", "https://")):
                print(f"[OpenAISoraAPI] 不支持的URL格式: {video_url}")
                return None

            print(f"[OpenAISoraAPI] 🎬 开始下载视频: {video_url[:80]}...")
            try:
                video_output = download_url_to_video_output(video_url, timeout=120)
                print(f"[OpenAISoraAPI] ✅ 视频下载完成")
                return video_output
            except Exception as download_error:
                print(f"[OpenAISoraAPI] ❌ 视频下载失败: {download_error}")
                return None
        except Exception as e:
            print(f"[OpenAISoraAPI] 视频下载转换过程出错: {e}")
            return None


class Comfly_sora2_batch_32:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["sora-2", "sora-2-pro"], {"default": "sora-2", "multiline": False}),
            },
            "optional": {
                # 1-32 组图片+Prompt输入项
                "image_1": ("IMAGE",),
                "prompt_1": ("STRING", {"forceInput": True, "multiline": True}),
                "image_2": ("IMAGE",),
                "prompt_2": ("STRING", {"forceInput": True, "multiline": True}),
                "image_3": ("IMAGE",),
                "prompt_3": ("STRING", {"forceInput": True, "multiline": True}),
                "image_4": ("IMAGE",),
                "prompt_4": ("STRING", {"forceInput": True, "multiline": True}),
                "image_5": ("IMAGE",),
                "prompt_5": ("STRING", {"forceInput": True, "multiline": True}),
                "image_6": ("IMAGE",),
                "prompt_6": ("STRING", {"forceInput": True, "multiline": True}),
                "image_7": ("IMAGE",),
                "prompt_7": ("STRING", {"forceInput": True, "multiline": True}),
                "image_8": ("IMAGE",),
                "prompt_8": ("STRING", {"forceInput": True, "multiline": True}),
                "image_9": ("IMAGE",),
                "prompt_9": ("STRING", {"forceInput": True, "multiline": True}), 
                "image_10": ("IMAGE",),
                "prompt_10": ("STRING", {"forceInput": True, "multiline": True}),
                "image_11": ("IMAGE",),
                "prompt_11": ("STRING", {"forceInput": True, "multiline": True}),
                "image_12": ("IMAGE",),
                "prompt_12": ("STRING", {"forceInput": True, "multiline": True}),
                "image_13": ("IMAGE",),
                "prompt_13": ("STRING", {"forceInput": True, "multiline": True}),
                "image_14": ("IMAGE",),
                "prompt_14": ("STRING", {"forceInput": True, "multiline": True}),
                "image_15": ("IMAGE",),
                "prompt_15": ("STRING", {"forceInput": True, "multiline": True}),
                "image_16": ("IMAGE",),
                "prompt_16": ("STRING", {"forceInput": True, "multiline": True}),
                "image_17": ("IMAGE",),
                "prompt_17": ("STRING", {"forceInput": True, "multiline": True}),
                "image_18": ("IMAGE",),
                "prompt_18": ("STRING", {"forceInput": True, "multiline": True}),
                "image_19": ("IMAGE",),
                "prompt_19": ("STRING", {"forceInput": True, "multiline": True}),
                "image_20": ("IMAGE",),
                "prompt_20": ("STRING", {"forceInput": True, "multiline": True}),
                "image_21": ("IMAGE",),
                "prompt_21": ("STRING", {"forceInput": True, "multiline": True}),
                "image_22": ("IMAGE",),
                "prompt_22": ("STRING", {"forceInput": True, "multiline": True}),
                "image_23": ("IMAGE",),
                "prompt_23": ("STRING", {"forceInput": True, "multiline": True}),
                "image_24": ("IMAGE",),
                "prompt_24": ("STRING", {"forceInput": True, "multiline": True}),
                "image_25": ("IMAGE",),
                "prompt_25": ("STRING", {"forceInput": True, "multiline": True}),
                "image_26": ("IMAGE",),
                "prompt_26": ("STRING", {"forceInput": True, "multiline": True}),
                "image_27": ("IMAGE",),
                "prompt_27": ("STRING", {"forceInput": True, "multiline": True}),
                "image_28": ("IMAGE",),
                "prompt_28": ("STRING", {"forceInput": True, "multiline": True}),
                "image_29": ("IMAGE",),
                "prompt_29": ("STRING", {"forceInput": True, "multiline": True}),
                "image_30": ("IMAGE",),
                "prompt_30": ("STRING", {"forceInput": True, "multiline": True}),
                "image_31": ("IMAGE",),
                "prompt_31": ("STRING", {"forceInput": True, "multiline": True}),
                "image_32": ("IMAGE",),
                "prompt_32": ("STRING", {"forceInput": True, "multiline": True}),
                
                # 全局配置项
                "aspect_ratio": (["16:9", "9:16"], {"default": "9:16"}),
                "duration": ("INT", {"default": 10, "min": 1, "max": 60}),
                "hd": ("BOOLEAN", {"default": False}),
                "max_concurrent": ("INT", {"default": 1, "min": 1, "max": 32}),
                "global_prompt": ("STRING", {"default": "", "multiline": True}),
                "base_url": ("STRING", {"default": "", "multiline": False}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
            },
        }

    # 返回32个视频 + 1个日志字符串
    RETURN_TYPES = (
        IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO, 
        IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO,
        IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO,
        IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO,
        IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO,
        IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO,
        IO.VIDEO, IO.VIDEO,
        "STRING"
    )
    RETURN_NAMES = (
        "video1", "video2", "video3", "video4", "video5", 
        "video6", "video7", "video8", "video9", "video10",
        "video11", "video12", "video13", "video14", "video15",
        "video16", "video17", "video18", "video19", "video20",
        "video21", "video22", "video23", "video24", "video25",
        "video26", "video27", "video28", "video29", "video30",
        "video31", "video32",
        "log"
    )
    FUNCTION = "generate_video"
    CATEGORY = "RunNode/OpenAI"

    def __init__(self):
        # 从配置文件初始化API Key和base_url
        self.config = get_config()
        self.api_key = self.config.get('api_key', '')
        self.base_url = self.config.get('base_url', 'https://ai.t8star.cn')
        
        self.task_progress: Dict[int, int] = {}  # 任务进度 {任务索引: 进度值}
        self.global_pbar = None

    def get_headers(self) -> Dict[str, str]:
        """获取API请求头"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def image_to_base64(self, image_tensor) -> Optional[str]:
        """将图片张量转换为base64字符串（带data URI前缀）"""
        if image_tensor is None:
            return None
            
        pil_image = tensor2pil(image_tensor)[0]
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{base64_str}"

    def validate_task_params(self, model: str, duration: int, hd: bool) -> Tuple[bool, str]:
        """校验任务参数合法性"""
        # 25秒和HD不能同时使用（适配原有逻辑）
        if duration == 25 and hd:
            return False, "25秒时长和HD模式不能同时启用"
        
        # sora-2模型限制
        if model == "sora-2":
            if duration > 15:
                return False, "sora-2模型仅支持最大15秒时长（25秒需使用sora-2-pro）"
            if hd:
                return False, "sora-2模型不支持HD模式（需使用sora-2-pro）"
        
        # 时长范围额外校验
        if duration < 1 or duration > 60:
            return False, f"时长必须在1-60秒之间（当前值：{duration}）"
        
        return True, ""

    def process_single_task(
        self, 
        task_idx: int, 
        model: str, 
        prompt: str, 
        image, 
        aspect_ratio: str, 
        duration: int, 
        hd: bool,
        current_base_url: str
    ) -> Dict[str, Any]:
        """处理单个视频生成任务（增加current_base_url参数适配动态base_url）"""
        task_result = {
            "index": task_idx,
            "status": "failed",
            "video": "",
            "video_url": "",
            "error": "",
            "response": "",
            "task_id": ""
        }

        # 1. 参数校验
        is_valid, err_msg = self.validate_task_params(model, duration, hd)
        if not is_valid:
            task_result["error"] = err_msg
            self.task_progress[task_idx] = 100
            return task_result

        # 2. 空Prompt处理
        if not prompt.strip():
            task_result["error"] = "Prompt不能为空"
            self.task_progress[task_idx] = 100
            return task_result

        try:
            self.task_progress[task_idx] = 10  # 初始化进度

            # 3. 构建请求体
            payload = {
                "prompt": prompt,
                "model": model,
                "aspect_ratio": aspect_ratio,
                "duration": duration,
                "hd": hd,
                "private": True
            }

            # 添加参考图片
            image_base64 = self.image_to_base64(image)
            if image_base64:
                payload["images"] = [image_base64]

            self.task_progress[task_idx] = 20  # 构建请求完成

            # 4. 提交生成请求（使用动态base_url）
            response = requests.post(
                f"{current_base_url}/v2/videos/generations",
                headers=self.get_headers(),
                json=payload,
                timeout=TIMEOUT
            )

            if response.status_code != 200:
                task_result["error"] = f"API请求失败：{response.status_code} - {response.text}"
                self.task_progress[task_idx] = 100
                return task_result

            result = response.json()
            if "task_id" not in result:
                task_result["error"] = "API响应无任务ID"
                self.task_progress[task_idx] = 100
                return task_result

            task_id = result["task_id"]
            task_result["task_id"] = task_id
            self.task_progress[task_idx] = 30  # 提交任务完成

            # 5. 轮询任务状态（使用动态base_url）
            max_attempts = int(TIMEOUT / 10)  # 每10秒轮询一次
            attempts = 0
            video_url = None

            while attempts < max_attempts:
                time.sleep(10)
                attempts += 1

                # 更新进度（保底）
                self.task_progress[task_idx] = 30 + min(60, int((attempts / max_attempts) * 60))

                try:
                    status_resp = requests.get(
                        f"{current_base_url}/v2/videos/generations/{task_id}",
                        headers=self.get_headers(),
                        timeout=TIMEOUT
                    )

                    if status_resp.status_code != 200:
                        continue

                    status_data = status_resp.json()
                    task_result["response"] = json.dumps(status_data, ensure_ascii=False)

                    # 解析进度
                    progress_text = status_data.get("progress", "0%")
                    if progress_text.endswith('%'):
                        try:
                            progress_val = int(progress_text[:-1])
                            self.task_progress[task_idx] = 30 + int(progress_val * 0.6)
                        except:
                            pass

                    # 解析状态
                    status = status_data.get("status", "")
                    if status == "SUCCESS":
                        video_url = status_data.get("data", {}).get("output", "")
                        if video_url:
                            task_result["status"] = "success"
                            task_result["video_url"] = video_url
                            task_result["video"] = ComflyVideoAdapter(video_url)
                            self.task_progress[task_idx] = 100
                            return task_result
                    elif status == "FAILURE":
                        fail_reason = status_data.get("fail_reason", "未知错误")
                        task_result["error"] = f"生成失败：{fail_reason}"
                        self.task_progress[task_idx] = 100
                        return task_result

                except Exception as e:
                    task_result["error"] = f"轮询状态异常：{str(e)}"
                    continue

            # 超时未完成
            task_result["error"] = f"任务超时（{TIMEOUT}秒）未生成完成"
            self.task_progress[task_idx] = 100
            return task_result

        except Exception as e:
            task_result["error"] = f"任务执行异常：{str(e)}\n{traceback.format_exc()}"
            self.task_progress[task_idx] = 100
            return task_result

    def update_global_progress(self):
        """更新全局进度条"""
        if not self.global_pbar or not self.task_progress:
            return
        
        total_progress = sum(self.task_progress.values())
        avg_progress = int(total_progress / len(self.task_progress))
        self.global_pbar.update_absolute(avg_progress)

    def generate_video(self, **kwargs):
        """核心批量生成逻辑"""
        # 1. 处理base_url优先级：手动覆盖 > 配置文件 > 默认值
        base_url = kwargs.get("base_url", "").strip()
        current_base_url = base_url if base_url else self.base_url
        if not current_base_url:
            current_base_url = "https://ai.t8star.cn"  # 最终兜底

        # 2. 初始化API Key
        api_key = kwargs.get("api_key", "").strip()
        if api_key:
            self.api_key = api_key
            # 调用用户的save_config更新配置文件
            save_config({'api_key': api_key})
            # 重新加载配置确保最新
            self.config = get_config()
                
        if not self.api_key:
            log = json.dumps({
                "error": "API Key未配置",
                "tips": "请在节点中输入api_key或配置文件中设置"
            }, ensure_ascii=False, indent=2)
            empty_videos = [""] * 32
            return (*empty_videos, log)

        # 3. 提取全局配置
        model = kwargs.get("model")
        aspect_ratio = kwargs.get("aspect_ratio")
        duration = kwargs.get("duration")
        hd = kwargs.get("hd")
        max_concurrent = kwargs.get("max_concurrent", 1)
        global_prompt = kwargs.get("global_prompt", "").strip()

        # 4. 收集任务列表（按max_concurrent限制，最多32个）
        tasks = []
        for idx in range(1, 33):
            if idx > max_concurrent:
                break  # 仅处理max_concurrent指定的任务数

            # 图片和Prompt获取
            image = kwargs.get(f"image_{idx}")
            if global_prompt:
                prompt = global_prompt  # 使用全局Prompt
            else:
                prompt = kwargs.get(f"prompt_{idx}", "").strip()

            tasks.append({
                "idx": idx,
                "image": image,
                "prompt": prompt
            })

        # 5. 初始化进度跟踪
        self.task_progress = {t["idx"]: 0 for t in tasks}
        self.global_pbar = comfy.utils.ProgressBar(100)

        # 6. 并发执行任务
        task_results = {}
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # 提交任务
            future_to_idx = {}
            for task in tasks:
                future = executor.submit(
                    self.process_single_task,
                    task_idx=task["idx"],
                    model=model,
                    prompt=task["prompt"],
                    image=task["image"],
                    aspect_ratio=aspect_ratio,
                    duration=duration,
                    hd=hd,
                    current_base_url=current_base_url  # 传递动态base_url
                )
                future_to_idx[future] = task["idx"]
                time.sleep(0.1)  # 避免API限流

            # 收集结果并更新进度
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    task_results[idx] = result
                except Exception as e:
                    task_results[idx] = {
                        "index": idx,
                        "status": "failed",
                        "video": "",
                        "video_url": "",
                        "error": f"任务执行异常：{str(e)}",
                        "response": "",
                        "task_id": ""
                    }
                self.update_global_progress()

        # 7. 整理输出结果（32个视频）
        output_videos = []
        for idx in range(1, 33):
            result = task_results.get(idx, {})
            output_videos.append(result.get("video", ""))

        # 8. 整理日志（按顺序合并所有任务信息）
        log_data = {
            "global_config": {
                "model": model,
                "aspect_ratio": aspect_ratio,
                "duration": duration,
                "hd": hd,
                "max_concurrent": max_concurrent,
                "global_prompt_used": bool(global_prompt),
                "base_url": current_base_url,
                "api_key_configured": bool(self.api_key)
            },
            "tasks": []
        }

        for idx in range(1, 33):
            result = task_results.get(idx, {})
            log_data["tasks"].append({
                "task_index": idx,
                "status": result.get("status", "idle"),
                "video_url": result.get("video_url", ""),
                "task_id": result.get("task_id", ""),
                "error": result.get("error", ""),
                "prompt": tasks[idx-1]["prompt"] if (idx-1) < len(tasks) else ""
            })

        # 格式化日志为易读的JSON字符串
        log = json.dumps(log_data, ensure_ascii=False, indent=2)

        # 9. 完成进度条
        self.global_pbar.update_absolute(100)

        # 返回32个视频 + 日志
        return (*output_videos, log)


class Comfly_sora2_batch_front:
    @classmethod
    def INPUT_TYPES(cls):
        optional_inputs = {}
        for i in range(1, 33):
            optional_inputs[f"image{i}"] = ("IMAGE",)
            optional_inputs[f"prompt{i}"] = ("STRING", {"forceInput": True, "multiline": True})
        optional_inputs.update({
            "aspect_ratio": (["16:9", "9:16"], {"default": "9:16"}),
            "duration": ("INT", {"default": 10, "min": 1, "max": 60}),
            "hd": ("BOOLEAN", {"default": False}),
            "global_prompt": ("STRING", {"default": "", "multiline": True}),
            "base_url": ("STRING", {"default": "", "multiline": False}),
            "api_key": ("STRING", {"default": "", "multiline": False}),
        })
        return {
            "required": {
                "model": (["sora-2", "sora-2-pro"], {"default": "sora-2", "multiline": False}),
            },
            "optional": optional_inputs,
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("plan",)
    FUNCTION = "build_plan"
    CATEGORY = "RunNode/OpenAI"
    def __init__(self):
        self.config = get_config()
        self.api_key = self.config.get('api_key', '')
        self.base_url = self.config.get('base_url', 'https://ai.t8star.cn')
    def image_to_base64(self, image_tensor):
        if image_tensor is None:
            return None
        pil_image = tensor2pil(image_tensor)[0]
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{base64_str}"
    def build_plan(self, **kwargs):
        base_url = kwargs.get("base_url", "").strip()
        current_base_url = base_url if base_url else self.base_url or "https://ai.t8star.cn"
        api_key = kwargs.get("api_key", "").strip()
        if api_key:
            self.api_key = api_key
            save_config({'api_key': api_key})
            self.config = get_config()
        model = kwargs.get("model")
        aspect_ratio = kwargs.get("aspect_ratio")
        duration = kwargs.get("duration")
        hd = kwargs.get("hd")
        global_prompt = kwargs.get("global_prompt", "").strip()
        tasks = []
        for idx in range(1, 33):
            img = kwargs.get(f"image{idx}")
            prompt = global_prompt if global_prompt else kwargs.get(f"prompt{idx}", "").strip()
            image_b64 = self.image_to_base64(img)
            tasks.append({
                "idx": idx,
                "prompt": prompt,
                "image_base64": image_b64 or ""
            })
        plan = {
            "global_config": {
                "model": model,
                "aspect_ratio": aspect_ratio,
                "duration": duration,
                "hd": hd,
                "base_url": current_base_url,
                "api_key_configured": bool(self.api_key)
            },
            "tasks": tasks
        }
        return (json.dumps(plan, ensure_ascii=False, indent=2),)


class _ComflySora2BatchPlanBase:
    MAX_OUTPUTS = 32
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": ("STRING", {"multiline": True}),
            },
            "optional": {
                "max_concurrent": ("INT", {"default": cls.MAX_OUTPUTS, "min": 1, "max": cls.MAX_OUTPUTS}),
                "base_url": ("STRING", {"default": "", "multiline": False}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
            },
        }
    FUNCTION = "generate_video_from_plan"
    CATEGORY = "RunNode/OpenAI"
    def __init__(self):
        self.config = get_config()
        self.api_key = self.config.get('api_key', '')
        self.base_url = self.config.get('base_url', 'https://ai.t8star.cn')
        self.task_progress = {}
        self.global_pbar = None
        self.timeout = 900
    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    def validate_task_params(self, model, duration, hd):
        if duration == 25 and hd:
            return False, "25秒时长和HD模式不能同时启用"
        if model == "sora-2":
            if duration > 15:
                return False, "sora-2模型仅支持最大15秒时长（25秒需使用sora-2-pro）"
            if hd:
                return False, "sora-2模型不支持HD模式（需使用sora-2-pro）"
        if duration < 1 or duration > 60:
            return False, f"时长必须在1-60秒之间（当前值：{duration}）"
        return True, ""
    def process_single_task(self, task_idx, model, prompt, image_base64, aspect_ratio, duration, hd, current_base_url):
        task_result = {
            "index": task_idx,
            "status": "failed",
            "video": "",
            "video_url": "",
            "error": "",
            "response": "",
            "task_id": ""
        }
        is_valid, err_msg = self.validate_task_params(model, duration, hd)
        if not is_valid:
            task_result["error"] = err_msg
            self.task_progress[task_idx] = 100
            return task_result
        if not prompt.strip():
            task_result["error"] = "Prompt不能为空"
            self.task_progress[task_idx] = 100
            return task_result
        try:
            self.task_progress[task_idx] = 10
            payload = {
                "prompt": prompt,
                "model": model,
                "aspect_ratio": aspect_ratio,
                "duration": duration,
                "hd": hd,
                "private": True
            }
            if image_base64:
                payload["images"] = [image_base64]
            self.task_progress[task_idx] = 20
            response = requests.post(
                f"{current_base_url}/v2/videos/generations",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            if response.status_code != 200:
                task_result["error"] = f"API请求失败：{response.status_code} - {response.text}"
                self.task_progress[task_idx] = 100
                return task_result
            result = response.json()
            if "task_id" not in result:
                task_result["error"] = "API响应无任务ID"
                self.task_progress[task_idx] = 100
                return task_result
            task_id = result["task_id"]
            task_result["task_id"] = task_id
            self.task_progress[task_idx] = 30
            max_attempts = int(self.timeout / 10)
            attempts = 0
            while attempts < max_attempts:
                time.sleep(10)
                attempts += 1
                self.task_progress[task_idx] = 30 + min(60, int((attempts / max_attempts) * 60))
                try:
                    status_resp = requests.get(
                        f"{current_base_url}/v2/videos/generations/{task_id}",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )
                    if status_resp.status_code != 200:
                        continue
                    status_data = status_resp.json()
                    task_result["response"] = json.dumps(status_data, ensure_ascii=False)
                    progress_text = status_data.get("progress", "0%")
                    if isinstance(progress_text, str) and progress_text.endswith('%'):
                        try:
                            progress_val = int(progress_text[:-1])
                            self.task_progress[task_idx] = 30 + int(progress_val * 0.6)
                        except Exception:
                            pass
                    status = status_data.get("status", "")
                    if status == "SUCCESS":
                        video_url = status_data.get("data", {}).get("output", "")
                        if video_url:
                            task_result["status"] = "success"
                            task_result["video_url"] = video_url
                            task_result["video"] = ComflyVideoAdapter(video_url)
                            self.task_progress[task_idx] = 100
                            return task_result
                    elif status == "FAILURE":
                        fail_reason = status_data.get("fail_reason", "未知错误")
                        task_result["error"] = f"生成失败：{fail_reason}"
                        self.task_progress[task_idx] = 100
                        return task_result
                except Exception as e:
                    task_result["error"] = f"轮询状态异常：{str(e)}"
                    continue
            task_result["error"] = f"任务超时（{self.timeout}秒）未生成完成"
            self.task_progress[task_idx] = 100
            return task_result
        except Exception as e:
            task_result["error"] = f"任务执行异常：{str(e)}\n{traceback.format_exc()}"
            self.task_progress[task_idx] = 100
            return task_result
    def update_global_progress(self):
        if not self.global_pbar or not self.task_progress:
            return
        total_progress = sum(self.task_progress.values())
        avg_progress = int(total_progress / len(self.task_progress))
        self.global_pbar.update_absolute(avg_progress)
    def generate_video_from_plan(self, plan, max_concurrent=None, base_url="", api_key=""):
        try:
            parsed = json.loads(plan) if isinstance(plan, str) else {}
        except Exception as e:
            empty_videos = [""] * self.MAX_OUTPUTS
            log = json.dumps({"error": f"计划解析失败：{str(e)}"}, ensure_ascii=False, indent=2)
            return (*empty_videos, log)
        cfg = parsed.get("global_config", {})
        model = cfg.get("model")
        aspect_ratio = cfg.get("aspect_ratio")
        duration = cfg.get("duration")
        hd = cfg.get("hd")
        plan_base_url = cfg.get("base_url", "")
        current_base_url = (base_url.strip() or plan_base_url or self.base_url or "https://ai.t8star.cn")
        if api_key.strip():
            self.api_key = api_key.strip()
            save_config({'api_key': self.api_key})
            self.config = get_config()
        if not self.api_key:
            log = json.dumps({"error": "API Key未配置"}, ensure_ascii=False, indent=2)
            empty_videos = [""] * self.MAX_OUTPUTS
            return (*empty_videos, log)
        tasks_src = parsed.get("tasks", []) or []
        limit = max_concurrent or self.MAX_OUTPUTS
        limit = max(1, min(self.MAX_OUTPUTS, int(limit)))
        tasks = []
        for t in tasks_src[:limit]:
            tasks.append({
                "idx": int(t.get("idx", 0)) or len(tasks) + 1,
                "prompt": t.get("prompt", "") or "",
                "image_base64": t.get("image_base64", "") or ""
            })
        self.task_progress = {t["idx"]: 0 for t in tasks}
        self.global_pbar = comfy.utils.ProgressBar(100)
        task_results = {}
        with ThreadPoolExecutor(max_workers=limit) as executor:
            future_to_idx = {}
            for t in tasks:
                future = executor.submit(
                    self.process_single_task,
                    task_idx=t["idx"],
                    model=model,
                    prompt=t["prompt"],
                    image_base64=t["image_base64"],
                    aspect_ratio=aspect_ratio,
                    duration=duration,
                    hd=hd,
                    current_base_url=current_base_url
                )
                future_to_idx[future] = t["idx"]
                time.sleep(0.1)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    task_results[idx] = result
                except Exception as e:
                    task_results[idx] = {
                        "index": idx,
                        "status": "failed",
                        "video": "",
                        "video_url": "",
                        "error": f"任务执行异常：{str(e)}",
                        "response": "",
                        "task_id": ""
                    }
                self.update_global_progress()
        output_videos = []
        for i in range(1, self.MAX_OUTPUTS + 1):
            result = task_results.get(i, {})
            output_videos.append(result.get("video", ""))
        log_data = {
            "global_config": {
                "model": model,
                "aspect_ratio": aspect_ratio,
                "duration": duration,
                "hd": hd,
                "max_concurrent": limit,
                "base_url": current_base_url,
                "api_key_configured": bool(self.api_key)
            },
            "tasks": []
        }
        for i in range(1, self.MAX_OUTPUTS + 1):
            result = task_results.get(i, {})
            src_prompt = ""
            if i - 1 < len(tasks):
                src_prompt = tasks[i - 1]["prompt"]
            log_data["tasks"].append({
                "task_index": i,
                "status": result.get("status", "idle"),
                "video_url": result.get("video_url", ""),
                "task_id": result.get("task_id", ""),
                "error": result.get("error", ""),
                "prompt": src_prompt
            })
        log = json.dumps(log_data, ensure_ascii=False, indent=2)
        self.global_pbar.update_absolute(100)
        return (*output_videos, log)


class Comfly_sora2_batch_plan_32(_ComflySora2BatchPlanBase):
    MAX_OUTPUTS = 32
    RETURN_TYPES = (
        IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO,
        IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO,
        IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO,
        IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO,
        IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO,
        IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO,
        IO.VIDEO, IO.VIDEO,
        "STRING"
    )
    RETURN_NAMES = (
        "video1", "video2", "video3", "video4", "video5",
        "video6", "video7", "video8", "video9", "video10",
        "video11", "video12", "video13", "video14", "video15",
        "video16", "video17", "video18", "video19", "video20",
        "video21", "video22", "video23", "video24", "video25",
        "video26", "video27", "video28", "video29", "video30",
        "video31", "video32",
        "log"
    )


class Comfly_sora2_batch_plan_16(_ComflySora2BatchPlanBase):
    MAX_OUTPUTS = 16
    RETURN_TYPES = (
        IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO,
        IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO,
        IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO,
        IO.VIDEO,
        "STRING"
    )
    RETURN_NAMES = (
        "video1", "video2", "video3", "video4", "video5",
        "video6", "video7", "video8", "video9", "video10",
        "video11", "video12", "video13", "video14", "video15",
        "video16",
        "log"
    )


class Comfly_sora2_batch_plan_8(_ComflySora2BatchPlanBase):
    MAX_OUTPUTS = 8
    RETURN_TYPES = (
        IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO,
        IO.VIDEO, IO.VIDEO, IO.VIDEO,
        "STRING"
    )
    RETURN_NAMES = (
        "video1", "video2", "video3", "video4", "video5",
        "video6", "video7", "video8",
        "log"
    )


class Comfly_sora2_batch_plan_4(_ComflySora2BatchPlanBase):
    MAX_OUTPUTS = 4
    RETURN_TYPES = (
        IO.VIDEO, IO.VIDEO, IO.VIDEO, IO.VIDEO,
        "STRING"
    )
    RETURN_NAMES = (
        "video1", "video2", "video3", "video4",
        "log"
    )
