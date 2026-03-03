"""OpenAI-compatible API backend implementation."""

from typing import List, Dict, Any
import json
import base64
import time

import cv2
import numpy as np
import requests

from .base import VLMBackend


def _encode_jpeg_b64(img_bgr: np.ndarray, target_width: int = 640, quality: int = 80) -> str:
    """Convert BGR numpy array to JPEG base64 data URL."""
    if img_bgr is None or img_bgr.size == 0:
        return ""
    
    h, w = img_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return ""
    
    # Resize if needed (maintain aspect ratio)
    if w != target_width:
        target_height = int(h * (target_width / w))
        img_bgr = cv2.resize(img_bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
    # Encode as JPEG
    _, buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not _:
        return ""
    
    # Convert to base64 and add data URL prefix
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from text, handling markdown code blocks."""
    if not text:
        return {}
    
    t = text.replace("", "").replace("```", "").strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object in text
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(t[start:end+1])
        except json.JSONDecodeError:
            pass
    
    return {}


class OpenAICompatBackend(VLMBackend):
    """OpenAI-compatible API backend for VLM inference."""
    
    def __init__(
        self,
        api_url: str,
        api_key: str,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        target_width: int = 640,
        jpeg_quality: int = 80,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout_sec: float = 60.0,
        headers: Dict[str, str] = None,
    ):
        self.api_url = api_url.rstrip("/")
        if not self.api_url.endswith("/v1/chat/completions"):
            self.api_url = f"{self.api_url}/v1/chat/completions"
        self.api_key = api_key
        self.model_id = model_id
        self.target_width = target_width
        self.jpeg_quality = jpeg_quality
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_sec = timeout_sec
        self.headers = headers or {}
    
    @property
    def name(self) -> str:
        return "openai_compat"
    
    def warmup(self) -> None:
        """Optional warmup - no model to load for API backend."""
        print(f"[OpenAI] Backend ready (model: {self.model_id}), api_url: {self.api_url}")
    
    def infer(self, images: List[np.ndarray], prompt: str) -> Dict[str, Any]:
        """Run inference via OpenAI-compatible API."""
        if not images:
            return {}
        
        # Encode images to base64 data URLs
        image_urls = []
        for img in images:
            data_url = _encode_jpeg_b64(img, self.target_width, self.jpeg_quality)
            if data_url:
                image_urls.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })
        
        if not image_urls:
            return {}
        
        # Build OpenAI-compatible payload
        content_list = [{"type": "text", "text": prompt}]
        content_list.extend(image_urls)
        
        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": content_list
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "thinking": {
                "type": "disabled"
            }
        }
        
        # Prepare headers
        headers = dict(self.headers)
        headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Send request
        t0 = time.time()
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=self.timeout_sec
            )
            latency_s = time.time() - t0
            
            response.raise_for_status()
            
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if not content:
                print(f"[OpenAI] Empty response (latency_s={latency_s:.3f})")
                print(f"[OpenAI] Response: {response.text}")
                return {}
            
            # Extract JSON from content string
            parsed = _extract_json(content)
            return parsed
            
        except requests.exceptions.RequestException as e:
            print(f"[OpenAI] Request failed: {e}")
            return {}
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"[OpenAI] Failed to parse response: {e}")
            return {}
    
    def cleanup(self) -> None:
        """Optional cleanup - no resources to free for API backend."""
        print("[OpenAI] Backend cleaned up.")
