from ..comfly_config import *
from .__init__ import *


class Comfly_suno_description:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "title": ("STRING", {"default": ""}),
                "description_prompt": ("STRING", {"multiline": True}),
                "version": (["v3.0", "v3.5", "v4", "v4.5", "v4.5+", "v5"], {"default": "v4.5"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "make_instrumental": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "apikey": ("STRING", {"default": ""})
                # "apikey": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio1", "audio2", "audio_url1", "audio_url2", "prompt", "task_id", "response", "clip_id1", "clip_id2", "tags", "title")
    FUNCTION = "generate_music"
    CATEGORY = "RunNode/Suno"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300
        
    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key
        }
    
    def generate_music(self, title, description_prompt, version="v4.5", seed=0, make_instrumental=False, apikey=""):
        request_id = generate_request_id("music_gen", "suno")
        log_prepare("音乐生成", request_id, "RunNode/Suno-", "Suno", rule_name="description")
        rn_pbar = ProgressBar(request_id, "Suno", streaming=True, task_type="音乐生成", source="RunNode/Suno-")
        rn_pbar.set_generating(0)
        _rn_start = time.perf_counter()
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
            log_backend(
                "suno_music_description_failed",
                level="ERROR",
                request_id=request_id,
                stage="missing_api_key",
                url=safe_public_url(baseurl),
                elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
            )
            raise Exception(error_message)
        
        mv_mapping = {
            "v3.0": "chirp-v3.0",
            "v3.5": "chirp-v3.5", 
            "v4": "chirp-v4",
            "v4.5": "chirp-auk",
            "v4.5+": "chirp-bluejay",
            "v5": "chirp-crow"
        }
        
        mv = mv_mapping.get(version, "chirp-auk")
            
        try:
            payload = {
                "gpt_description_prompt": description_prompt,
                "make_instrumental": make_instrumental,
                "mv": mv,
                "prompt": ""
            }
            if seed > 0:
                payload["seed"] = seed

            log_backend(
                "suno_music_description_start",
                request_id=request_id,
                url=safe_public_url(f"{baseurl}/suno/generate/description-mode"),
                mv=str(mv),
                version=str(version),
                seed=(int(seed) if isinstance(seed, int) and seed > 0 else None),
                make_instrumental=bool(make_instrumental),
                title_len=len(title or ""),
                prompt_len=len(description_prompt or ""),
            )
            
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)
           
            response = requests.post(
                f"{baseurl}/suno/generate/description-mode",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(20)
           
            if response.status_code != 200:
                error_message = format_runnode_error(response)
                rn_pbar.error(error_message)
                log_backend(
                    "suno_music_description_failed",
                    level="ERROR",
                    request_id=request_id,
                    stage="http_error",
                    url=safe_public_url(f"{baseurl}/suno/generate/description-mode"),
                    status_code=int(response.status_code),
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )
                raise Exception(error_message)
                
            result = response.json()
           
            if "id" not in result:
                error_message = "No task ID in response"
                rn_pbar.error(error_message)
                log_backend(
                    "suno_music_description_failed",
                    level="ERROR",
                    request_id=request_id,
                    stage="missing_task_id",
                    url=safe_public_url(f"{baseurl}/suno/generate/description-mode"),
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )
                raise Exception(error_message)
                
            task_id = result.get("id")
            
            if "clips" not in result or len(result["clips"]) < 2:
                error_message = "Expected at least 2 clips in the response"
                rn_pbar.error(error_message)
                log_backend(
                    "suno_music_description_failed",
                    level="ERROR",
                    request_id=request_id,
                    stage="insufficient_clips",
                    task_id=str(task_id),
                    clips_count=(len(result.get("clips", [])) if isinstance(result.get("clips", []), list) else None),
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )
                empty_audio = create_audio_object("")
                return (empty_audio, empty_audio, "", "", "", task_id, error_message, "", "", "", "")
                
            clip_ids = [clip["id"] for clip in result["clips"]]
            if len(clip_ids) < 2:
                error_message = "Expected at least 2 clip IDs"
                rn_pbar.error(error_message)
                log_backend(
                    "suno_music_description_failed",
                    level="ERROR",
                    request_id=request_id,
                    stage="missing_clip_ids",
                    task_id=str(task_id),
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )
                empty_audio = create_audio_object("")
                return (empty_audio, empty_audio, "", "", "", task_id, error_message, "", "", "", "")
                
            pbar.update_absolute(30)
            max_attempts = 30
            attempts = 0
            final_clips = []
            generated_prompt = ""
            extracted_tags = ""
            generated_title = ""  

            log_backend(
                "suno_music_description_poll_start",
                request_id=request_id,
                task_id=str(task_id),
                clip_ids_count=int(len(clip_ids)),
                max_attempts=int(max_attempts),
            )
           
            while attempts < max_attempts and len(final_clips) < 2:
                time.sleep(5)
                attempts += 1
                
                try:
                    clip_response = requests.get(
                        f"{baseurl}/suno/feed/{','.join(clip_ids)}",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )
                    
                    if clip_response.status_code != 200:
                        continue
                        
                    clips_data = clip_response.json()
                   
                    progress = min(80, 30 + (attempts * 50 // max_attempts))
                    pbar.update_absolute(progress)
                    complete_clips = [
                        clip for clip in clips_data 
                        if clip.get("status") == "complete" and (clip.get("audio_url") or clip.get("state") == "succeeded")
                    ]
                    for clip in complete_clips:
                        if clip.get("id") in clip_ids and clip not in final_clips:
                            final_clips.append(clip)
                            if not generated_prompt and "prompt" in clip:
                                generated_prompt = clip["prompt"]
                            if not extracted_tags and "tags" in clip:
                                extracted_tags = clip["tags"]
                            if not generated_title and "title" in clip and clip["title"]:
                                generated_title = clip["title"]
                    
                    if len(final_clips) >= 2:
                        break
                        
                except Exception as e:
                    rn_pbar.error(f"Error checking clip status: {format_runnode_error(str(e))}")
                    log_backend_exception(
                        "suno_music_description_poll_exception",
                        request_id=request_id,
                        task_id=str(task_id),
                        attempt=int(attempts),
                    )
            
            if len(final_clips) < 2:
                error_message = f"Only received {len(final_clips)} complete clips after {max_attempts} attempts"
                rn_pbar.error(error_message)

                log_backend(
                    "suno_music_description_poll_failed",
                    level="ERROR",
                    request_id=request_id,
                    task_id=str(task_id),
                    attempts=int(attempts),
                    complete_count=int(len(final_clips)),
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )
                
                if not final_clips:
                    raise Exception(error_message)
                
            final_title = generated_title if generated_title else title
            for clip in final_clips:
                if "title" not in clip or not clip["title"]:
                    clip["title"] = final_title
                    
            audio_urls = []
            clip_id_values = []
            
            for clip in final_clips[:2]:  
                audio_url = ""
                if "audio_url" in clip and clip["audio_url"]:
                    audio_url = clip["audio_url"]
                elif "cdn1.suno.ai" in str(clip):
                    match = re.search(r'https://cdn1\.suno\.ai/[^"\']+\.mp3', str(clip))
                    if match:
                        audio_url = match.group(0)
                
                if audio_url:
                    audio_urls.append(audio_url)
                else:
                    rn_pbar.error("No audio URL found in clip")
                    audio_urls.append("")
                    
                clip_id_value = clip.get("id", "")
                if clip_id_value:
                    clip_id_values.append(clip_id_value)
                else:
                    clip_id_values.append("")
                
            while len(audio_urls) < 2:
                audio_urls.append("")
                
            while len(clip_id_values) < 2:
                clip_id_values.append("")
            audio_objects = [create_audio_object(url) for url in audio_urls[:2]]
            while len(audio_objects) < 2:
                audio_objects.append(create_audio_object(""))
            
            pbar.update_absolute(100)
            
            response_info = {
                "status": "success",
                "prompt": generated_prompt,
                "title": final_title, 
                "version": version,
                "seed": seed if seed > 0 else "auto",
                "make_instrumental": make_instrumental,
                "clips_generated": len(final_clips),
                "tags": extracted_tags
            }
            log_backend(
                "suno_music_description_done",
                request_id=request_id,
                task_id=str(task_id),
                url=safe_public_url(f"{baseurl}/suno/feed/{','.join(clip_ids)}"),
                clips_generated=int(len(final_clips)),
                has_audio_url1=bool(audio_urls[0] if len(audio_urls) > 0 else ""),
                has_audio_url2=bool(audio_urls[1] if len(audio_urls) > 1 else ""),
                elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
            )
            rn_pbar.done(char_count=len(json.dumps(response_info, ensure_ascii=False)), elapsed_ms=int((time.perf_counter() - _rn_start) * 1000))
            return (
                audio_objects[0],
                audio_objects[1],
                audio_urls[0],
                audio_urls[1],
                generated_prompt,
                task_id,
                json.dumps(response_info),
                clip_id_values[0],
                clip_id_values[1],
                extracted_tags,
                final_title  
            )
                
        except Exception as e:
            error_message = f"Error generating music: {str(e)}"
            rn_pbar.error(error_message)
            import traceback
            traceback.print_exc()
            log_backend_exception(
                "suno_music_description_exception",
                request_id=request_id,
                url=safe_public_url(f"{baseurl}/suno/generate/description-mode"),
                elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
            )
            raise Exception(error_message)


class Comfly_suno_lyrics:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            },
            "optional": {
                "apikey": ("STRING", {"default": ""})
                # "apikey": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("lyrics", "response", "title", "tags")
    FUNCTION = "generate_lyrics"
    CATEGORY = "RunNode/Suno"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300
        
    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key
        }
        
    def generate_lyrics(self, prompt, seed=0, apikey=""):
        request_id = generate_request_id("lyrics_gen", "suno")
        log_prepare("歌词生成", request_id, "RunNode/Suno-", "Suno", rule_name="lyrics")
        rn_pbar = ProgressBar(request_id, "Suno", streaming=True, task_type="歌词生成", source="RunNode/Suno-")
        rn_pbar.set_generating(0)
        _rn_start = time.perf_counter()
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
            log_backend(
                "suno_lyrics_failed",
                level="ERROR",
                request_id=request_id,
                stage="missing_api_key",
                url=safe_public_url(baseurl),
                elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
            )
            raise Exception(error_message)
            
        try:
            payload = {"prompt": prompt}
            
            if seed > 0:
                payload["seed"] = seed

            log_backend(
                "suno_lyrics_start",
                request_id=request_id,
                url=safe_public_url(f"{baseurl}/suno/generate/lyrics/"),
                prompt_len=len(prompt or ""),
                seed=(int(seed) if isinstance(seed, int) and seed > 0 else None),
            )

            response = requests.post(
                f"{baseurl}/suno/generate/lyrics/",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                error_message = format_runnode_error(response)
                rn_pbar.error(error_message)
                log_backend(
                    "suno_lyrics_failed",
                    level="ERROR",
                    request_id=request_id,
                    stage="http_error",
                    url=safe_public_url(f"{baseurl}/suno/generate/lyrics/"),
                    status_code=int(response.status_code),
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )
                raise Exception(error_message)
                
            result = response.json()
            
            if "id" not in result:
                error_message = "No task ID in response"
                rn_pbar.error(error_message)
                log_backend(
                    "suno_lyrics_failed",
                    level="ERROR",
                    request_id=request_id,
                    stage="missing_task_id",
                    url=safe_public_url(f"{baseurl}/suno/generate/lyrics/"),
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )
                raise Exception(error_message)
                
            task_id = result.get("id")

            max_attempts = 30
            attempts = 0
            lyrics_text = ""
            generated_title = ""  
            tags = ""

            log_backend(
                "suno_lyrics_poll_start",
                request_id=request_id,
                task_id=str(task_id),
                max_attempts=int(max_attempts),
            )
            
            while attempts < max_attempts:
                time.sleep(2)
                attempts += 1
                
                try:
                    lyrics_response = requests.get(
                        f"{baseurl}/suno/lyrics/{task_id}",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )
                    
                    if lyrics_response.status_code != 200:
                        continue
                        
                    lyrics_data = lyrics_response.json()
                    
                    if lyrics_data.get("status") == "complete" or lyrics_data.get("status") == "succeed":
                        lyrics_text = lyrics_data.get("text", "")
                        generated_title = lyrics_data.get("title", "")  
                        tags = lyrics_data.get("tags", "")
                        break
                        
                except Exception as e:
                    log_backend_exception(
                        "suno_lyrics_poll_exception",
                        request_id=request_id,
                        task_id=str(task_id),
                        attempt=int(attempts),
                    )
            
            if not lyrics_text:
                error_message = f"Failed to generate lyrics after {max_attempts} attempts"
                rn_pbar.error(error_message)
                log_backend(
                    "suno_lyrics_poll_failed",
                    level="ERROR",
                    request_id=request_id,
                    task_id=str(task_id),
                    attempts=int(attempts),
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )
                raise Exception(error_message)
            
            success_response = {
                "status": "success",
                "title": generated_title,  
                "prompt": prompt,
                "seed": seed if seed > 0 else "auto",
                "tags": tags
            }

            log_backend(
                "suno_lyrics_done",
                request_id=request_id,
                task_id=str(task_id),
                lyrics_len=int(len(lyrics_text or "")),
                title_len=int(len(generated_title or "")),
                tags_len=int(len(tags or "")),
                elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
            )
            rn_pbar.done(char_count=len(json.dumps(success_response, ensure_ascii=False)), elapsed_ms=int((time.perf_counter() - _rn_start) * 1000))
            return (lyrics_text, json.dumps(success_response), generated_title, tags)
                
        except Exception as e:
            error_message = f"Error generating lyrics: {format_runnode_error(str(e))}"
            rn_pbar.error(error_message)
            log_backend_exception(
                "suno_lyrics_exception",
                request_id=request_id,
                url=safe_public_url(f"{baseurl}/suno/generate/lyrics/"),
                elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
            )
            raise Exception(error_message)


class Comfly_suno_custom:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "title": ("STRING", {"default": ""}),
                "version": (["v3.0", "v3.5", "v4", "v4.5", "v4.5+", "v5"], {"default": "v4.5"}),
                "prompt": ("STRING", {"multiline": True}), 
                "tags": ("STRING", {"default": ""}),  
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            },
            "optional": {
                "apikey": ("STRING", {"default": ""})
                # "apikey": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO", "STRING", "STRING", "STRING", "STRING", 
                   "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio1", "audio2", "audio_url1", "audio_url2", "task_id", "response",
                   "clip_id1", "clip_id2", "image_large_url1", "image_large_url2", "tags", "title")
    FUNCTION = "generate_music"
    CATEGORY = "RunNode/Suno"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key
        }
    
    def generate_music(self, title, version="v4.5", prompt="", tags="", seed=0, apikey=""):
        request_id = generate_request_id("music_custom", "suno")
        log_prepare("音乐生成", request_id, "RunNode/Suno-", "Suno", rule_name="custom")
        rn_pbar = ProgressBar(request_id, "Suno", streaming=True, task_type="音乐生成", source="RunNode/Suno-")
        rn_pbar.set_generating(0)
        _rn_start = time.perf_counter()
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
            log_backend(
                "suno_music_custom_failed",
                level="ERROR",
                request_id=request_id,
                stage="missing_api_key",
                url=safe_public_url(baseurl),
                elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
            )
            raise Exception(error_message)
        
        mv_mapping = {
            "v3.0": "chirp-v3.0",
            "v3.5": "chirp-v3.5", 
            "v4": "chirp-v4",
            "v4.5": "chirp-auk",
            "v4.5+": "chirp-bluejay",
            "v5": "chirp-crow"
        }
        
        mv = mv_mapping.get(version, "chirp-auk")
            
        try:
            payload = {
                "prompt": prompt,
                "tags": tags,
                "mv": mv,
                "title": title  
            }
            if seed > 0:
                payload["seed"] = seed

            log_backend(
                "suno_music_custom_start",
                request_id=request_id,
                url=safe_public_url(f"{baseurl}/suno/generate"),
                mv=str(mv),
                version=str(version),
                seed=(int(seed) if isinstance(seed, int) and seed > 0 else None),
                title_len=len(title or ""),
                prompt_len=len(prompt or ""),
                tags_len=len(tags or ""),
            )
            
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)
           
            response = requests.post(
                f"{baseurl}/suno/generate",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(20)
            
            if response.status_code != 200:
                error_message = format_runnode_error(response)
                rn_pbar.error(error_message)
                log_backend(
                    "suno_music_custom_failed",
                    level="ERROR",
                    request_id=request_id,
                    stage="http_error",
                    url=safe_public_url(f"{baseurl}/suno/generate"),
                    status_code=int(response.status_code),
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )
                raise Exception(error_message)
                
            result = response.json()
            
            if "id" not in result:
                error_message = "No task ID in response"
                rn_pbar.error(error_message)
                log_backend(
                    "suno_music_custom_failed",
                    level="ERROR",
                    request_id=request_id,
                    stage="missing_task_id",
                    url=safe_public_url(f"{baseurl}/suno/generate"),
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )
                raise Exception(error_message)
                
            task_id = result.get("id")
            
            if "clips" not in result or len(result["clips"]) < 2:
                error_message = "Expected at least 2 clips in the response"
                rn_pbar.error(error_message)
                log_backend(
                    "suno_music_custom_failed",
                    level="ERROR",
                    request_id=request_id,
                    stage="insufficient_clips",
                    task_id=str(task_id),
                    clips_count=(len(result.get("clips", [])) if isinstance(result.get("clips", []), list) else None),
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )
                raise Exception(error_message)
                
            clip_ids = [clip["id"] for clip in result["clips"]]
            if len(clip_ids) < 2:
                error_message = "Expected at least 2 clip IDs"
                rn_pbar.error(error_message)
                log_backend(
                    "suno_music_custom_failed",
                    level="ERROR",
                    request_id=request_id,
                    stage="missing_clip_ids",
                    task_id=str(task_id),
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )
                empty_audio = create_audio_object("")
                return (empty_audio, empty_audio, "", "", task_id, error_message, 
                    "", "", "", "", "", "")
                
            pbar.update_absolute(30)
            max_attempts = 30
            attempts = 0
            final_clips = []
            final_tags = tags  
            generated_title = ""  

            log_backend(
                "suno_music_custom_poll_start",
                request_id=request_id,
                task_id=str(task_id),
                clip_ids_count=int(len(clip_ids)),
                max_attempts=int(max_attempts),
            )
            
            while attempts < max_attempts and len(final_clips) < 2:
                time.sleep(5)
                attempts += 1
                
                try:
                    clip_response = requests.get(
                        f"{baseurl}/suno/feed/{','.join(clip_ids)}",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )
                    
                    if clip_response.status_code != 200:
                        continue
                        
                    clips_data = clip_response.json()
                    
                    progress = min(80, 30 + (attempts * 50 // max_attempts))
                    pbar.update_absolute(progress)
                    complete_clips = [
                        clip for clip in clips_data 
                        if clip.get("status") == "complete" and (clip.get("audio_url") or clip.get("state") == "succeeded")
                    ]
                    
                    for clip in complete_clips:
                        if clip.get("id") in clip_ids and clip not in final_clips:
                            final_clips.append(clip)
                            if "tags" in clip and clip["tags"]:
                                final_tags = clip["tags"]
                            if "title" in clip and clip["title"]:
                                generated_title = clip["title"]
                    
                    if len(final_clips) >= 2:
                        break
                        
                except Exception as e:
                    log_backend_exception(
                        "suno_music_custom_poll_exception",
                        request_id=request_id,
                        task_id=str(task_id),
                        attempt=int(attempts),
                    )
            
            if len(final_clips) < 2:
                error_message = f"Only received {len(final_clips)} complete clips after {max_attempts} attempts"
                rn_pbar.error(error_message)
                log_backend(
                    "suno_music_custom_poll_failed",
                    level="ERROR",
                    request_id=request_id,
                    task_id=str(task_id),
                    attempts=int(attempts),
                    complete_count=int(len(final_clips)),
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )
                
                if not final_clips:
                    raise Exception(error_message)
            final_title = generated_title if generated_title else title
                    
            audio_urls = []
            clip_id_values = []
            image_large_urls = []
            
            for clip in final_clips[:2]:
                audio_url = ""
                if "audio_url" in clip and clip["audio_url"]:
                    audio_url = clip["audio_url"]
                elif "cdn1.suno.ai" in str(clip):
                    match = re.search(r'https://cdn1\.suno\.ai/[^"\']+\.mp3', str(clip))
                    if match:
                        audio_url = match.group(0)
                
                audio_urls.append(audio_url if audio_url else "")
                clip_id = clip.get("clip_id", clip.get("id", ""))
                clip_id_values.append(clip_id)
                image_large_url = clip.get("image_large_url", "")
                image_large_urls.append(image_large_url)
            while len(audio_urls) < 2:
                audio_urls.append("")
                
            while len(clip_id_values) < 2:
                clip_id_values.append("")
                
            while len(image_large_urls) < 2:
                image_large_urls.append("")
            audio_objects = [create_audio_object(url) for url in audio_urls[:2]]
            while len(audio_objects) < 2:
                audio_objects.append(create_audio_object(""))
            
            pbar.update_absolute(100)
            
            response_info = {
                "status": "success",
                "title": final_title,
                "version": version,
                "seed": seed if seed > 0 else "auto",
                "clips_generated": len(final_clips),
                "tags": final_tags
            }

            log_backend(
                "suno_music_custom_done",
                request_id=request_id,
                task_id=str(task_id),
                url=safe_public_url(f"{baseurl}/suno/feed/{','.join(clip_ids)}"),
                clips_generated=int(len(final_clips)),
                has_audio_url1=bool(audio_urls[0] if len(audio_urls) > 0 else ""),
                has_audio_url2=bool(audio_urls[1] if len(audio_urls) > 1 else ""),
                elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
            )
            rn_pbar.done(char_count=len(json.dumps(response_info, ensure_ascii=False)), elapsed_ms=int((time.perf_counter() - _rn_start) * 1000))
            return (
                audio_objects[0],  
                audio_objects[1],  
                audio_urls[0],     
                audio_urls[1],     
                task_id,
                json.dumps(response_info),
                clip_id_values[0],
                clip_id_values[1],
                image_large_urls[0],
                image_large_urls[1],
                final_tags,
                final_title  
            )
                
        except Exception as e:
            error_message = f"Error generating music: {str(e)}"
            rn_pbar.error(error_message)
            import traceback
            traceback.print_exc()
            log_backend_exception(
                "suno_music_custom_exception",
                request_id=request_id,
                url=safe_public_url(f"{baseurl}/suno/generate"),
                elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
            )
            raise Exception(error_message)
        

class Comfly_suno_upload:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                # "api_key": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "upload_filename": ("STRING", {"default": "audio.mp3"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("clip_id", "title", "tags", "lyrics", "response")
    FUNCTION = "upload_audio"
    CATEGORY = "RunNode/Suno"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def upload_audio(self, audio, api_key="", upload_filename="audio.mp3", seed=0):
        request_id = generate_request_id("audio_upload", "suno")
        log_prepare("音频上传", request_id, "RunNode/Suno-", "Suno", rule_name="upload")
        rn_pbar = ProgressBar(request_id, "Suno", streaming=True, task_type="音频上传", source="RunNode/Suno-")
        rn_pbar.set_generating(0)
        _rn_start = time.perf_counter()
        if api_key.strip():
            self.api_key = api_key
            # config = get_config()
            # config['api_key'] = api_key
            # save_config(config)
        else:
            self.api_key = get_config().get('api_key', '')
            
        if not self.api_key:
            error_message = "API key not found in Comflyapi.json"
            rn_pbar.error(error_message)
            log_backend(
                "suno_audio_upload_failed",
                level="ERROR",
                request_id=request_id,
                stage="missing_api_key",
                url=safe_public_url(baseurl),
                elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
            )
            raise Exception(error_message)
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)
 
            extension = upload_filename.split('.')[-1] if '.' in upload_filename else "mp3"
            payload = {"extension": extension}

            if seed > 0:
                payload["seed"] = seed

            waveform = audio.get("waveform") if isinstance(audio, dict) else None
            sample_rate = audio.get("sample_rate") if isinstance(audio, dict) else None
            audio_seconds = None
            try:
                if waveform is not None and sample_rate:
                    sr = int(sample_rate)
                    total_samples = int(getattr(waveform, "shape", [0])[-1])
                    if sr > 0 and total_samples > 0:
                        audio_seconds = round(total_samples / sr, 3)
            except Exception:
                audio_seconds = None

            log_backend(
                "suno_audio_upload_start",
                request_id=request_id,
                url=safe_public_url(f"{baseurl}/suno/uploads/audio"),
                upload_filename=str(upload_filename),
                extension=str(extension),
                seed=(int(seed) if isinstance(seed, int) and seed > 0 else None),
                audio_seconds=audio_seconds,
            )
            
            response = requests.post(
                f"{baseurl}/suno/uploads/audio",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                error_message = format_runnode_error(response)
                rn_pbar.error(error_message)
                log_backend(
                    "suno_audio_upload_failed",
                    level="ERROR",
                    request_id=request_id,
                    stage="get_upload_url_http_error",
                    url=safe_public_url(f"{baseurl}/suno/uploads/audio"),
                    status_code=int(response.status_code),
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )
                raise Exception(error_message)
                
            upload_data = response.json()
            upload_id = upload_data["id"]
            upload_url = upload_data["url"]
            fields = upload_data["fields"]

            log_backend(
                "suno_audio_upload_got_url",
                request_id=request_id,
                upload_id=str(upload_id),
                upload_url=safe_public_url(str(upload_url)),
            )
            
            pbar.update_absolute(30)

            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]

            if len(waveform.shape) == 3:
                waveform = waveform.squeeze(0)

            temp_file = io.BytesIO()
            torchaudio.save(temp_file, waveform, sample_rate, format="mp3")
            temp_file.seek(0)
            audio_data = temp_file.read()

            files = {
                'Content-Type': ('', fields['Content-Type']),
                'key': ('', fields['key']),
                'AWSAccessKeyId': ('', fields['AWSAccessKeyId']),
                'policy': ('', fields['policy']),
                'signature': ('', fields['signature']),
                'file': (upload_filename, audio_data, 'audio/mpeg')
            }
            
            upload_response = requests.post(upload_url, files=files, timeout=self.timeout)
            
            if upload_response.status_code != 204:
                error_message = format_runnode_error(upload_response)
                rn_pbar.error(error_message)
                log_backend(
                    "suno_audio_upload_failed",
                    level="ERROR",
                    request_id=request_id,
                    stage="upload_http_error",
                    upload_id=str(upload_id),
                    upload_url=safe_public_url(str(upload_url)),
                    status_code=int(upload_response.status_code),
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )
                raise Exception(error_message)
                
            pbar.update_absolute(50)

            finish_payload = {
                "upload_type": "file_upload",
                "upload_filename": upload_filename
            }
            
            finish_response = requests.post(
                f"{baseurl}/suno/uploads/audio/{upload_id}/upload-finish",
                headers=self.get_headers(),
                json=finish_payload,
                timeout=self.timeout
            )
            
            if finish_response.status_code != 200:
                error_message = f"Failed to finish upload: {finish_response.status_code}"
                rn_pbar.error(error_message)
                log_backend(
                    "suno_audio_upload_failed",
                    level="ERROR",
                    request_id=request_id,
                    stage="upload_finish_http_error",
                    upload_id=str(upload_id),
                    url=safe_public_url(f"{baseurl}/suno/uploads/audio/{upload_id}/upload-finish"),
                    status_code=int(finish_response.status_code),
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )
                raise Exception(error_message)
                
            pbar.update_absolute(60)

            max_attempts = 20
            attempts = 0
            clip_id = ""
            title = ""
            tags = ""
            lyrics = ""

            log_backend(
                "suno_audio_upload_poll_start",
                request_id=request_id,
                upload_id=str(upload_id),
                max_attempts=int(max_attempts),
            )
            
            while attempts < max_attempts:
                time.sleep(2)
                attempts += 1
                
                status_response = requests.get(
                    f"{baseurl}/suno/uploads/audio/{upload_id}",
                    headers=self.get_headers(),
                    timeout=self.timeout
                )
                
                if status_response.status_code != 200:
                    continue
                    
                status_data = status_response.json()
                status = status_data.get("status", "")
                
                pbar.update_absolute(60 + (attempts * 20 // max_attempts))
                
                if status == "complete":
                    init_response = requests.post(
                        f"{baseurl}/suno/uploads/audio/{upload_id}/initialize-clip",
                        headers=self.get_headers(),
                        json={},
                        timeout=self.timeout
                    )
                    
                    if init_response.status_code == 200:
                        init_data = init_response.json()
                        clip_id = init_data.get("clip_id", "")

                        if clip_id:
                            try:
                                clip_detail_response = requests.get(
                                    f"{baseurl}/suno/feed/{clip_id}",
                                    headers=self.get_headers(),
                                    timeout=self.timeout
                                )
                                
                                if clip_detail_response.status_code == 200:
                                    clip_details = clip_detail_response.json()
                                    if isinstance(clip_details, list) and len(clip_details) > 0:
                                        clip_info = clip_details[0]
                                    else:
                                        clip_info = clip_details
                                    
                                    title = clip_info.get("title", "")
                                    tags = clip_info.get("tags", "")
                                    lyrics = clip_info.get("metadata", {}).get("prompt", "") or clip_info.get("prompt", "")
                                    
                            except Exception as e:
                                log_backend_exception(
                                    "suno_audio_upload_clip_detail_exception",
                                    request_id=request_id,
                                    upload_id=str(upload_id),
                                    clip_id=str(clip_id),
                                )
                        
                        pbar.update_absolute(100)
                        
                        response_info = {
                            "status": "success",
                            "upload_id": upload_id,
                            "clip_id": clip_id,
                            "title": title,
                            "tags": tags,
                            "lyrics": lyrics,
                            "seed": seed if seed > 0 else "auto",
                            "upload_filename": upload_filename
                        }

                        log_backend(
                            "suno_audio_upload_done",
                            request_id=request_id,
                            upload_id=str(upload_id),
                            clip_id=str(clip_id),
                            title_len=int(len(title or "")),
                            tags_len=int(len(tags or "")),
                            lyrics_len=int(len(lyrics or "")),
                            elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                        )
                        rn_pbar.done(char_count=len(json.dumps(response_info, ensure_ascii=False)), elapsed_ms=int((time.perf_counter() - _rn_start) * 1000))
                        return (clip_id, title, tags, lyrics, json.dumps(response_info))
                    else:
                        error_message = format_runnode_error(init_response)
                        rn_pbar.error(error_message)
                        log_backend(
                            "suno_audio_upload_failed",
                            level="ERROR",
                            request_id=request_id,
                            stage="initialize_clip_http_error",
                            upload_id=str(upload_id),
                            url=safe_public_url(f"{baseurl}/suno/uploads/audio/{upload_id}/initialize-clip"),
                            status_code=int(init_response.status_code),
                            elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                        )
                        raise Exception(error_message)
                        
                elif status in ["failed", "error"]:
                    error_message = f"Upload failed with status: {status}"
                    rn_pbar.error(error_message)
                    log_backend(
                        "suno_audio_upload_failed",
                        level="ERROR",
                        request_id=request_id,
                        stage="upload_failed",
                        upload_id=str(upload_id),
                        status=str(status),
                        attempts=int(attempts),
                        elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                    )
                    raise Exception(error_message)
            
            error_message = "Upload timeout - status check exceeded maximum attempts"
            rn_pbar.error(error_message)
            log_backend(
                "suno_audio_upload_failed",
                level="ERROR",
                request_id=request_id,
                stage="timeout",
                upload_id=str(upload_id),
                attempts=int(attempts),
                elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
            )
            raise Exception(error_message)
            
        except Exception as e:
            error_message = f"Error uploading audio: {str(e)}"
            rn_pbar.error(error_message)
            log_backend_exception(
                "suno_audio_upload_exception",
                request_id=request_id,
                url=safe_public_url(f"{baseurl}/suno/uploads/audio"),
                elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
            )
            raise Exception(error_message)


class Comfly_suno_upload_extend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_id": ("STRING", {"forceInput": True}),
                "prompt": ("STRING", {"multiline": True}),
                "tags": ("STRING", {"default": ""}),
                "title": ("STRING", {"default": ""}),
                "continue_at": ("INT", {"default": 28, "min": 0, "max": 120}),
                "version": (["v3.0", "v3.5", "v4", "v4.5", "v4.5+", "v5"], {"default": "v5"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                # "api_key": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio1", "audio2", "audio_url1", "audio_url2", "task_id", "response", "clip_id1", "clip_id2", "duration")
    FUNCTION = "extend_audio"
    CATEGORY = "RunNode/Suno"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def extend_audio(self, clip_id, prompt, tags="", title="", continue_at=28, version="v5", api_key="", seed=0):
        request_id = generate_request_id("upload_extend", "suno")
        log_prepare("音频续写", request_id, "RunNode/Suno-", "Suno", rule_name="upload_extend")
        rn_pbar = ProgressBar(request_id, "Suno", streaming=True, task_type="音频续写", source="RunNode/Suno-")
        rn_pbar.set_generating(0)
        _rn_start = time.perf_counter()
        if api_key.strip():
            self.api_key = api_key
            # config = get_config()
            # config['api_key'] = api_key
            # save_config(config)
        else:
            self.api_key = get_config().get('api_key', '')
            
        if not self.api_key:
            error_message = "API key not found in Comflyapi.json"
            rn_pbar.error(error_message)
            log_backend(
                "suno_upload_extend_failed",
                level="ERROR",
                request_id=request_id,
                stage="missing_api_key",
                url=safe_public_url(baseurl),
                elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
            )
            raise Exception(error_message)

        mv_mapping = {
            "v3.0": "chirp-v3.0",
            "v3.5": "chirp-v3.5", 
            "v4": "chirp-v4",
            "v4.5": "chirp-auk",
            "v4.5+": "chirp-bluejay",
            "v5": "chirp-crow"
        }
        
        mv = mv_mapping.get(version, "chirp-crow")
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            payload = {
                "continue_at": continue_at,
                "continue_clip_id": clip_id,
                "mv": mv,
                "prompt": prompt,
                "tags": tags,
                "task": "upload_extend",
                "title": title
            }
            
            if seed > 0:
                payload["seed"] = seed

            log_backend(
                "suno_upload_extend_start",
                request_id=request_id,
                url=safe_public_url(f"{baseurl}/suno/generate"),
                mv=str(mv),
                version=str(version),
                seed=(int(seed) if isinstance(seed, int) and seed > 0 else None),
                continue_at=(int(continue_at) if isinstance(continue_at, int) else continue_at),
                original_clip_id=str(clip_id),
                title_len=len(title or ""),
                prompt_len=len(prompt or ""),
                tags_len=len(tags or ""),
            )

            response = requests.post(
                f"{baseurl}/suno/generate",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(20)
            
            if response.status_code != 200:
                error_message = format_runnode_error(response)
                rn_pbar.error(error_message)
                log_backend(
                    "suno_upload_extend_failed",
                    level="ERROR",
                    request_id=request_id,
                    stage="http_error",
                    url=safe_public_url(f"{baseurl}/suno/generate"),
                    status_code=int(response.status_code),
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )
                raise Exception(error_message)
                
            result = response.json()
            
            if "id" not in result:
                error_message = "No task ID in response"
                rn_pbar.error(error_message)
                log_backend(
                    "suno_upload_extend_failed",
                    level="ERROR",
                    request_id=request_id,
                    stage="missing_task_id",
                    url=safe_public_url(f"{baseurl}/suno/generate"),
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )
                raise Exception(error_message)
                
            task_id = result.get("id")
            
            if "clips" not in result or len(result["clips"]) < 2:
                error_message = "Expected at least 2 clips in the response"
                rn_pbar.error(error_message)
                log_backend(
                    "suno_upload_extend_failed",
                    level="ERROR",
                    request_id=request_id,
                    stage="insufficient_clips",
                    task_id=str(task_id),
                    clips_count=(len(result.get("clips", [])) if isinstance(result.get("clips", []), list) else None),
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )
                raise Exception(error_message)
                
            clip_ids = [clip["id"] for clip in result["clips"]]
            
            pbar.update_absolute(30)

            max_attempts = 40
            attempts = 0
            final_clips = []

            log_backend(
                "suno_upload_extend_poll_start",
                request_id=request_id,
                task_id=str(task_id),
                clip_ids_count=int(len(clip_ids)),
                max_attempts=int(max_attempts),
            )
            
            while attempts < max_attempts and len(final_clips) < 2:
                time.sleep(5)
                attempts += 1
                
                try:
                    clip_response = requests.get(
                        f"{baseurl}/suno/feed/{','.join(clip_ids)}",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )
                    
                    if clip_response.status_code != 200:
                        continue
                        
                    clips_data = clip_response.json()
                   
                    progress = min(80, 30 + (attempts * 50 // max_attempts))
                    pbar.update_absolute(progress)
                    
                    complete_clips = [
                        clip for clip in clips_data 
                        if clip.get("status") == "complete" and (
                            clip.get("audio_url") or 
                            clip.get("state") == "succeeded"
                        )
                    ]
                    
                    for clip in complete_clips:
                        if clip.get("id") in clip_ids and clip not in final_clips:
                            final_clips.append(clip)
                    
                    if len(final_clips) >= 2:
                        break
                        
                except Exception as e:
                    log_backend_exception(
                        "suno_upload_extend_poll_exception",
                        request_id=request_id,
                        task_id=str(task_id),
                        attempt=int(attempts),
                    )
            
            if len(final_clips) < 2:
                error_message = f"Only received {len(final_clips)} complete clips after {max_attempts} attempts"
                rn_pbar.error(error_message)
                log_backend(
                    "suno_upload_extend_poll_failed",
                    level="ERROR",
                    request_id=request_id,
                    task_id=str(task_id),
                    attempts=int(attempts),
                    complete_count=int(len(final_clips)),
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )
                
                if not final_clips:
                    raise Exception(error_message)

            audio_urls = []
            clip_id_values = []
            durations = []
            
            for clip in final_clips[:2]:
                audio_url = ""
                if "audio_url" in clip and clip["audio_url"]:
                    audio_url = clip["audio_url"]
                    print(f"Found audio URL: {audio_url}")
                elif "cdn1.suno.ai" in str(clip):
                    match = re.search(r'https://cdn1\.suno\.ai/[^"\']+\.mp3', str(clip))
                    if match:
                        audio_url = match.group(0)
                
                audio_urls.append(audio_url if audio_url else "")

                clip_id_value = clip.get("clip_id", clip.get("id", ""))
                clip_id_values.append(clip_id_value)

                duration = clip.get("duration", clip.get("metadata", {}).get("duration", 0))
                durations.append(str(duration))
                
            while len(audio_urls) < 2:
                audio_urls.append("")
                
            while len(clip_id_values) < 2:
                clip_id_values.append("")
                
            while len(durations) < 2:
                durations.append("0")

            audio_objects = [create_audio_object(url) for url in audio_urls[:2]]
            while len(audio_objects) < 2:
                audio_objects.append(create_audio_object(""))
            
            pbar.update_absolute(100)
            
            response_info = {
                "status": "success",
                "original_clip_id": clip_id,
                "continue_at": continue_at,
                "extended_clips": len(final_clips),
                "version": version,
                "title": title,
                "tags": tags
            }

            duration_info = durations[0] if durations[0] != "0" else (durations[1] if len(durations) > 1 else "0")

            log_backend(
                "suno_upload_extend_done",
                request_id=request_id,
                task_id=str(task_id),
                url=safe_public_url(f"{baseurl}/suno/feed/{','.join(clip_ids)}"),
                clips_generated=int(len(final_clips)),
                duration=str(duration_info),
                elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
            )
            rn_pbar.done(char_count=len(json.dumps(response_info, ensure_ascii=False)), elapsed_ms=int((time.perf_counter() - _rn_start) * 1000))
            
            return (
                audio_objects[0],  
                audio_objects[1],  
                audio_urls[0],     
                audio_urls[1],     
                task_id,
                json.dumps(response_info),
                clip_id_values[0],
                clip_id_values[1],
                duration_info
            )
                
        except Exception as e:
            error_message = f"Error extending audio: {str(e)}"
            rn_pbar.error(error_message)
            import traceback
            traceback.print_exc()
            log_backend_exception(
                "suno_upload_extend_exception",
                request_id=request_id,
                url=safe_public_url(f"{baseurl}/suno/generate"),
                elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
            )
            raise Exception(error_message)


class Comfly_suno_cover:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cover_clip_id": ("STRING", {"forceInput": True}),
                "prompt": ("STRING", {"multiline": True}),
                "title": ("STRING", {"default": ""}),
                "tags": ("STRING", {"default": ""}),
                "version": (["v3.0", "v3.5", "v4", "v4.5", "v4.5+", "v5"], {"default": "v5"}),
                "make_instrumental": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                # "api_key": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "negative_tags": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio1", "audio2", "audio_url1", "audio_url2", "task_id", "response", "clip_id1", "clip_id2", "image_url1", "image_url2")
    FUNCTION = "generate_cover"
    CATEGORY = "RunNode/Suno"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "accept": "*/*",
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json"
        }
    
    def generate_cover(self, cover_clip_id, prompt, title="", tags="", version="v5", 
                    make_instrumental=False, api_key="", negative_tags="", seed=0):
        request_id = generate_request_id("cover", "suno")
        log_prepare("翻唱生成", request_id, "RunNode/Suno-", "Suno", rule_name="cover")
        rn_pbar = ProgressBar(request_id, "Suno", streaming=True, task_type="翻唱生成", source="RunNode/Suno-")
        rn_pbar.set_generating(0)
        _rn_start = time.perf_counter()
        if api_key.strip():
            self.api_key = api_key
            # config = get_config()
            # config['api_key'] = api_key
            # save_config(config)
        else:
            self.api_key = get_config().get('api_key', '')
            
        if not self.api_key:
            error_message = "API key not found in Comflyapi.json"
            rn_pbar.error(error_message)
            log_backend(
                "suno_cover_failed",
                level="ERROR",
                request_id=request_id,
                stage="missing_api_key",
                url=safe_public_url(baseurl),
                elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
            )
            raise Exception(error_message)
        mv_mapping = {
            "v3.0": "chirp-v3.0",
            "v3.5": "chirp-v3.5", 
            "v4": "chirp-v4-tau",
            "v4.5": "chirp-auk",
            "v4.5+": "chirp-bluejay",
            "v5": "chirp-crow"
        }
        
        mv = mv_mapping.get(version, "chirp-v4-tau")
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)
            payload = {
                "prompt": prompt,
                "generation_type": "TEXT",
                "tags": tags,
                "negative_tags": negative_tags,
                "mv": mv,
                "title": title,
                "continue_clip_id": None,
                "continue_at": None,
                "continued_aligned_prompt": None,
                "infill_start_s": None,
                "infill_end_s": None,
                "task": "cover",
                "make_instrumental": make_instrumental,
                "cover_clip_id": cover_clip_id
            }
            
            if seed > 0:
                payload["seed"] = seed

            log_backend(
                "suno_cover_start",
                request_id=request_id,
                url=safe_public_url(f"{baseurl}/suno/submit/music"),
                mv=str(mv),
                version=str(version),
                seed=(int(seed) if isinstance(seed, int) and seed > 0 else None),
                make_instrumental=bool(make_instrumental),
                cover_clip_id=str(cover_clip_id),
                title_len=int(len(title or "")),
                prompt_len=int(len(prompt or "")),
                tags_len=int(len(tags or "")),
                negative_tags_len=int(len(negative_tags or "")),
            )
           
            response = requests.post(
                f"{baseurl}/suno/submit/music",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(30)
            
            if response.status_code != 200:
                error_message = format_runnode_error(response)
                rn_pbar.error(error_message)
                log_backend(
                    "suno_cover_failed",
                    level="ERROR",
                    request_id=request_id,
                    stage="http_error",
                    url=safe_public_url(f"{baseurl}/suno/submit/music"),
                    status_code=int(response.status_code),
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )
                raise Exception(error_message)
                
            result = response.json()
            clips = []
            task_id = ""
            
            if isinstance(result, dict):
                if result.get("code") == "success" and "data" in result:
                    task_id = result["data"]
                    if isinstance(task_id, str):
                        clips = self.wait_for_task_completion(task_id, pbar, request_id=request_id)
                    elif isinstance(task_id, list):
                        clips = task_id
                    elif isinstance(task_id, dict) and "clips" in task_id:
                        clips = task_id["clips"]
                    else:
                        clips = []
                elif "status" in result and result.get("status") == "SUCCESS" and "data" in result:
                    clips = result["data"]
                    task_id = result.get("task_id", "")
                    pbar.update_absolute(80)
                elif "clips" in result:
                    clips = result["clips"]
                    task_id = result.get("id", result.get("task_id", ""))
                elif "data" in result and isinstance(result["data"], list):
                    clips = result["data"]
                    task_id = result.get("id", result.get("task_id", ""))
                elif "id" in result:
                    task_id = result["id"]
                    clips = []
                    clips = self.wait_for_task_completion(task_id, pbar, request_id=request_id)
                else:
                    task_id = str(result)[:50]
                    clips = []
            elif isinstance(result, list):
                clips = result
                task_id = "direct_response"
            else:
                error_message = f"Unexpected response format: {result}"
                rn_pbar.error(error_message)
                log_backend(
                    "suno_cover_failed",
                    level="ERROR",
                    request_id=request_id,
                    stage="unexpected_response_format",
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )
                raise Exception(error_message)
            
            if len(clips) == 0:
                error_message = f"No clips found in response. Task ID: {task_id}"
                rn_pbar.error(error_message)
                log_backend(
                    "suno_cover_failed",
                    level="ERROR",
                    request_id=request_id,
                    stage="no_clips",
                    task_id=str(task_id),
                    elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
                )
                raise Exception(error_message)
            if len(clips) < 2:
                while len(clips) < 2:
                    clips.append(clips[0] if clips else {})
            audio_urls = []
            clip_id_values = []
            image_urls = []
            
            for i, clip in enumerate(clips[:2]):
               
                audio_url = ""
                if "audio_url" in clip and clip["audio_url"]:
                    audio_url = clip["audio_url"]
                
                audio_urls.append(audio_url)
                
                clip_id_value = clip.get("clip_id", clip.get("id", ""))
                clip_id_values.append(clip_id_value)
                image_url = clip.get("image_url", clip.get("image_large_url", ""))
                image_urls.append(image_url)
            while len(audio_urls) < 2:
                audio_urls.append("")
            while len(clip_id_values) < 2:
                clip_id_values.append("")
            while len(image_urls) < 2:
                image_urls.append("")
            pbar.update_absolute(90)
            audio_objects = [create_audio_object(url) for url in audio_urls[:2]]
            while len(audio_objects) < 2:
                audio_objects.append(create_audio_object(""))
            
            pbar.update_absolute(100)
            
            response_info = {
                "status": "success",
                "cover_clip_id": cover_clip_id,
                "version": version,
                "title": title,
                "tags": tags,
                "make_instrumental": make_instrumental,
                "clips_generated": len(clips),
                "audio_urls": audio_urls,
                "clip_ids": clip_id_values
            }

            log_backend(
                "suno_cover_done",
                request_id=request_id,
                task_id=str(task_id),
                clips_generated=int(len(clips)),
                has_audio_url1=bool(audio_urls[0] if len(audio_urls) > 0 else ""),
                has_audio_url2=bool(audio_urls[1] if len(audio_urls) > 1 else ""),
                elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
            )
            rn_pbar.done(char_count=len(json.dumps(response_info, ensure_ascii=False)), elapsed_ms=int((time.perf_counter() - _rn_start) * 1000))

            return (
                audio_objects[0],  
                audio_objects[1],  
                audio_urls[0],     
                audio_urls[1],     
                task_id,
                json.dumps(response_info),
                clip_id_values[0],
                clip_id_values[1],
                image_urls[0],
                image_urls[1]
            )
                
        except Exception as e:
            error_message = f"Error generating cover: {str(e)}"
            rn_pbar.error(error_message)
            import traceback
            traceback.print_exc()
            log_backend_exception(
                "suno_cover_exception",
                request_id=request_id,
                url=safe_public_url(f"{baseurl}/suno/submit/music"),
                elapsed_ms=int((time.perf_counter() - _rn_start) * 1000),
            )
            raise Exception(error_message)
        
    def wait_for_task_completion(self, task_id, pbar, request_id=None):
        max_attempts = 50
        attempts = 0

        log_backend(
            "suno_cover_poll_start",
            request_id=(str(request_id) if request_id else None),
            task_id=str(task_id),
            max_attempts=int(max_attempts),
        )
        
        while attempts < max_attempts:
            time.sleep(3)
            attempts += 1
            
            try:
                status_response = requests.get(
                    f"{baseurl}/suno/task/{task_id}",
                    headers=self.get_headers(),
                    timeout=self.timeout
                )
                
                if status_response.status_code == 200:
                    status_result = status_response.json()

                    if status_result.get("status") == "SUCCESS" and "data" in status_result:
                        clips = status_result["data"]
                        if isinstance(clips, list) and len(clips) > 0:
                            return clips
                    elif status_result.get("status") in ["FAILED", "ERROR"]:
                        break

                feed_response = requests.get(
                    f"{baseurl}/suno/feed/{task_id}",
                    headers=self.get_headers(),
                    timeout=self.timeout
                )
                
                if feed_response.status_code == 200:
                    feed_result = feed_response.json()
                    if isinstance(feed_result, list):
                        complete_clips = [
                            clip for clip in feed_result 
                            if clip.get("status") == "complete" and (
                                clip.get("audio_url") or clip.get("state") == "succeeded"
                            )
                        ]
                        if len(complete_clips) >= 1:
                            return complete_clips
                
                progress = min(85, 35 + (attempts * 50 // max_attempts))
                pbar.update_absolute(progress)
                
            except Exception as e:
                log_backend_exception(
                    "suno_cover_poll_exception",
                    request_id=(str(request_id) if request_id else None),
                    task_id=str(task_id),
                    attempt=int(attempts),
                )
        
        log_backend(
            "suno_cover_poll_failed",
            level="ERROR",
            request_id=(str(request_id) if request_id else None),
            task_id=str(task_id),
            attempts=int(attempts),
            stage="timeout",
        )
        return []
