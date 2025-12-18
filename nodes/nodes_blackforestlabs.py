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
                print(f"Unexpected response from file upload API: {result}")
                return None
                
        except Exception as e:
            print(f"Error uploading image: {str(e)}")
            return None
    
    def generate_image(self, prompt, input_image=None, model="flux-kontext-pro", 
                  apikey="", aspect_ratio="Default", guidance=3.5, num_of_images=1,
                  seed=-1, clear_image=True):
        if apikey.strip():
            self.api_key = apikey
            config = get_config()
            config['api_key'] = apikey
            save_config(config)
            
        if not self.api_key:
            error_message = "API key not found in Comflyapi.json"
            print(error_message)

            if input_image is None:
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, "")
            return (input_image, "")
        
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
                    print("Failed to upload any images")
                    if input_image is None:
                        blank_image = Image.new('RGB', (1024, 1024), color='white')
                        blank_tensor = pil2tensor(blank_image)
                        return (blank_tensor, "")
                    return (input_image, "")
 
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
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                if input_image is None:
                    blank_image = Image.new('RGB', (1024, 1024), color='white')
                    blank_tensor = pil2tensor(blank_image)
                    return (blank_tensor, "")
                return (input_image, "")
                
            result = response.json()

            if not result.get("data") or not result["data"]:
                error_message = "No image data in response"
                print(error_message)
                if input_image is None:
                    blank_image = Image.new('RGB', (1024, 1024), color='white')
                    blank_tensor = pil2tensor(blank_image)
                    return (blank_tensor, "")
                return (input_image, "")

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
                        print(f"Error downloading image from URL: {str(e)}")
                        
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
                
                return (combined_tensor, "\n".join(image_urls))
            else:
                error_message = "Failed to process any images"
                print(error_message)
                if input_image is None:
                    blank_image = Image.new('RGB', (1024, 1024), color='white')
                    blank_tensor = pil2tensor(blank_image)
                    return (blank_tensor, "")
                return (input_image, "")
            
        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(error_message)
            if input_image is None:
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, "")
            return (input_image, "")
         

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
        if apikey.strip():
            self.api_key = apikey
            config = get_config()
            config['api_key'] = apikey
            save_config(config)
            
        if not self.api_key:
            error_message = "API key not found in Comflyapi.json"
            print(error_message)

            if image is None:
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, "")
            return (image, "")
        
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
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                if image is None:
                    blank_image = Image.new('RGB', (1024, 1024), color='white')
                    blank_tensor = pil2tensor(blank_image)
                    return (blank_tensor, "")
                return (image, "")
                
            result = response.json()

            if not result.get("data") or not result["data"]:
                error_message = "No image data in response"
                print(error_message)
                if image is None:
                    blank_image = Image.new('RGB', (1024, 1024), color='white')
                    blank_tensor = pil2tensor(blank_image)
                    return (blank_tensor, "")
                return (image, "")

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
                        print(f"Error downloading image from URL: {str(e)}")
                        
                elif "b64_json" in item:
                    image_data = base64.b64decode(item["b64_json"])
                    generated_image = Image.open(BytesIO(image_data))
                    generated_tensor = pil2tensor(generated_image)
                    generated_tensors.append(generated_tensor)
            
            pbar.update_absolute(100)
            
            if generated_tensors:
                combined_tensor = torch.cat(generated_tensors, dim=0)
                return (combined_tensor, "\n".join(image_urls))
            else:
                error_message = "Failed to process any images"
                print(error_message)
                if image is None:
                    blank_image = Image.new('RGB', (1024, 1024), color='white')
                    blank_tensor = pil2tensor(blank_image)
                    return (blank_tensor, "")
                return (image, "")
            
        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(error_message)
            if image is None:
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, "")
            return (image, "")


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
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            save_config(config)

        if input_image is not None:
            default_tensor = input_image  
        else:
            blank_image = Image.new('RGB', (512, 512), color='white')
            default_tensor = pil2tensor(blank_image)
            
        if not self.api_key:
            error_response = {"status": "failed", "message": "API key not found"}
            return (default_tensor, "", json.dumps(error_response))
            
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
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                return (default_tensor, "", json.dumps({"status": "failed", "message": error_message}))
                
            result = response.json()
            
            if "id" not in result or "polling_url" not in result:
                error_message = "Invalid response format from API"
                print(error_message)
                return (default_tensor, "", json.dumps({"status": "failed", "message": error_message}))
                
            task_id = result["id"]
            polling_url = result["polling_url"]

            pbar.update_absolute(40)

            max_attempts = 60  
            attempts = 0
            image_url = ""
            
            while attempts < max_attempts:
                time.sleep(10)
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
                    status = result_data.get("status")
                    
                    if status == "Ready":
                        if "result" in result_data and "sample" in result_data["result"]:
                            image_url = result_data["result"]["sample"]
                            break

                    progress = min(80, 40 + (attempts * 40 // max_attempts))
                    pbar.update_absolute(progress)
                        
                except Exception as e:
                    print(f"Error checking generation status: {str(e)}")
            
            if not image_url:
                error_message = "Failed to retrieve generated image URL after multiple attempts"
                print(error_message)
                return (default_tensor, "", json.dumps({"status": "failed", "message": error_message}))

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
                
                return (generated_tensor, image_url, json.dumps(result_info))
                
            except Exception as e:
                error_message = f"Error downloading generated image: {str(e)}"
                print(error_message)
                return (default_tensor, image_url, json.dumps({"status": "partial_success", "message": error_message, "image_url": image_url}))
            
        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            return (default_tensor, "", json.dumps({"status": "failed", "message": error_message}))


class Comfly_Flux_2_Pro:
    """
    Comfly Flux 2 Pro node
    Generates images using the Flux 2 Pro API with support for multiple input images.
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
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
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
                      seed=-1, width=1024, height=1024, safety_tolerance=2, 
                      output_format="png"):
        
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            save_config(config)

        blank_image = Image.new('RGB', (width, height), color='white')
        default_tensor = pil2tensor(blank_image)
            
        if not self.api_key:
            error_response = {"status": "failed", "message": "API key not found"}
            return (default_tensor, "", json.dumps(error_response))
            
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
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                return (default_tensor, "", json.dumps({"status": "failed", "message": error_message}))
                
            result = response.json()
            
            if "id" not in result:
                error_message = "No task ID in response"
                print(error_message)
                return (default_tensor, "", json.dumps({"status": "failed", "message": error_message}))
                
            task_id = result["id"]
            polling_url = result.get("polling_url", "")
            
            pbar.update_absolute(40)

            max_attempts = 120
            attempts = 0
            image_url = ""
            
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
                    
                    if status == "Ready":
                        if "result" in result_data and "sample" in result_data["result"]:
                            image_url = result_data["result"]["sample"]
                            break
                    elif status in ["Failed", "Error"]:
                        error_message = f"Task failed: {result_data.get('details', 'Unknown error')}"
                        print(error_message)
                        return (default_tensor, "", json.dumps({"status": "failed", "message": error_message}))
                        
                except Exception as e:
                    print(f"Error checking generation status: {str(e)}")
            
            if not image_url:
                error_message = "Failed to retrieve generated image URL after multiple attempts"
                print(error_message)
                return (default_tensor, "", json.dumps({"status": "failed", "message": error_message}))
            
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
                
                return (generated_tensor, image_url, json.dumps(result_info))
                
            except Exception as e:
                error_message = f"Error downloading generated image: {str(e)}"
                print(error_message)
                return (default_tensor, image_url, json.dumps({"status": "partial_success", "message": error_message, "image_url": image_url}))
            
        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            return (default_tensor, "", json.dumps({"status": "failed", "message": error_message}))


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
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
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
        
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            save_config(config)

        blank_image = Image.new('RGB', (width, height), color='white')
        default_tensor = pil2tensor(blank_image)
            
        if not self.api_key:
            error_response = {"status": "failed", "message": "API key not found"}
            return (default_tensor, "", json.dumps(error_response))
            
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
                print(error_message)
                return (default_tensor, "", json.dumps({"status": "failed", "message": error_message}))
                
            result = response.json()
            
            if "id" not in result:
                error_message = "No task ID in response"
                print(error_message)
                return (default_tensor, "", json.dumps({"status": "failed", "message": error_message}))
                
            task_id = result["id"]
            polling_url = result.get("polling_url", "")
            
            pbar.update_absolute(40)

            max_attempts = 120
            attempts = 0
            image_url = ""
            
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
                    
                    if status == "Ready":
                        if "result" in result_data and "sample" in result_data["result"]:
                            image_url = result_data["result"]["sample"]
                            break
                    elif status in ["Failed", "Error"]:
                        error_message = f"Task failed: {result_data.get('details', 'Unknown error')}"
                        print(error_message)
                        return (default_tensor, "", json.dumps({"status": "failed", "message": error_message}))
                        
                except Exception as e:
                    print(f"Error checking generation status: {str(e)}")
            
            if not image_url:
                error_message = "Failed to retrieve generated image URL after multiple attempts"
                print(error_message)
                return (default_tensor, "", json.dumps({"status": "failed", "message": error_message}))
            
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
                
                return (generated_tensor, image_url, json.dumps(result_info))
                
            except Exception as e:
                error_message = f"Error downloading generated image: {str(e)}"
                print(error_message)
                return (default_tensor, image_url, json.dumps({"status": "partial_success", "message": error_message, "image_url": image_url}))
            
        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            return (default_tensor, "", json.dumps({"status": "failed", "message": error_message}))


class Comfly_nano_banana2_edit_S2A:
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
                      apikey="", task_id="", response_format="url", seed=0):
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
            return (blank_tensor, "", "", json.dumps({"status": "failed", "message": error_message}))
        
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            # 如果提供了task_id，则查询任务状态
            if task_id.strip():
                print(f"Querying task status for task_id: {task_id}")
                return self._query_task_status(task_id, pbar)
            
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
                print(error_message)
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
                                        return (combined_tensor, first_image_url, returned_task_id, json.dumps(final_result_info))
                                
                            elif actual_status == "failed" or actual_status == "error" or actual_status == "FAILURE":
                                # 任务失败
                                error_msg = query_result.get("error", "Unknown error")
                                print(f"Task failed: {error_msg}")
                                blank_image = Image.new('RGB', (1024, 1024), color='red')
                                blank_tensor = pil2tensor(blank_image)
                                pbar.update_absolute(100)
                                return (blank_tensor, "", "", json.dumps({"status": "failed", "task_id": returned_task_id, "message": error_msg}))
                                
                        else:
                            print(f"Query failed with status {query_response.status_code}")
                            
                    except Exception as e:
                        print(f"Error querying task status: {str(e)}")
                
                # 超时未完成
                print("Task polling timed out")
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
                    
                    return (combined_tensor, first_image_url, sync_task_id, json.dumps(result_info))
                else:
                    error_message = "Failed to process any images"
                    print(error_message)
                    blank_image = Image.new('RGB', (1024, 1024), color='white')
                    blank_tensor = pil2tensor(blank_image)
                    return (blank_tensor, "", "", json.dumps({"status": "failed", "message": error_message}))
                    
            else:
                # 未知响应格式
                error_message = f"Unexpected API response format: {result}"
                print(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor(blank_image)
                return (blank_tensor, "", "", json.dumps({"status": "failed", "message": error_message}))
            
        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(error_message)
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
