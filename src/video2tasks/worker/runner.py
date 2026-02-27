"""Worker runner implementation."""

import time
import json
import base64
from io import BytesIO
from typing import Optional, Dict, Any, List

import requests
import numpy as np
from PIL import Image

from ..config import Config
from ..vlm import create_backend
from ..vlm.base import VLMBackend
from ..prompt import prompt_switch_detection, prompt_label_segment

MAX_LOCAL_RETRIES = 2
MAX_RELABEL_RETRIES = 2


def _is_empty_vlm_json(vlm_json: Optional[Dict[str, Any]]) -> bool:
    return (not isinstance(vlm_json, dict)) or (not vlm_json)


def decode_b64_to_numpy(b64_str: str) -> Optional[np.ndarray]:
    """Decode base64 string to numpy BGR array."""
    if not b64_str:
        return None
    
    try:
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        # Convert RGB to BGR for OpenCV compatibility
        rgb_array = np.array(img)
        bgr_array = rgb_array[:, :, ::-1]
        return bgr_array
    except Exception:
        return None


def build_thumbnail_b64(images: List[np.ndarray], max_height: int = 120) -> Optional[str]:
    """Build a horizontal strip thumbnail from BGR images and return base64 PNG."""
    if not images:
        return None
    try:
        # BGR -> RGB for PIL
        thumbs = []
        for img in images:
            rgb = img[:, :, ::-1]
            pil = Image.fromarray(rgb)
            w, h = pil.size
            new_h = min(max_height, h)
            new_w = int(w * new_h / h) if h else w
            pil = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            thumbs.append(pil)
        total_w = sum(p.size[0] for p in thumbs)
        out = Image.new("RGB", (total_w, thumbs[0].size[1]))
        x = 0
        for p in thumbs:
            out.paste(p, (x, 0))
            x += p.size[0]
        buf = BytesIO()
        out.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def _has_instruction_mismatch(vlm_json: Dict[str, Any]) -> bool:
    """Return True when len(instructions) != len(transitions) + 1."""
    transitions = vlm_json.get("transitions", [])
    instructions = vlm_json.get("instructions", [])
    return len(instructions) != len(transitions) + 1


def _relabel_segments(
    images: List[np.ndarray],
    vlm_json: Dict[str, Any],
    backend: VLMBackend,
    task_id: str,
) -> Dict[str, Any]:
    """Fix a mismatch by re-labeling each segment independently.

    The original transitions are kept as-is; only instructions are replaced.
    Each segment [boundaries[i], boundaries[i+1]) is fed to the VLM with a
    single-task labeling prompt to obtain one instruction per segment.
    """
    transitions = sorted(int(t) for t in vlm_json.get("transitions", []))
    n = len(images)

    # Build half-open segment boundaries: [0, t0, t1, ..., n]
    boundaries = sorted(set([0] + transitions + [n]))
    n_segments = len(boundaries) - 1

    print(
        f"[Fix] {task_id}: instruction mismatch "
        f"(transitions={transitions}, "
        f"instructions={vlm_json.get('instructions', [])}). "
        f"Relabeling {n_segments} segment(s)."
    )

    new_instructions: List[str] = []
    for i in range(n_segments):
        s, e = boundaries[i], boundaries[i + 1]
        seg_images = images[s:e]

        if not seg_images:
            new_instructions.append("Unknown")
            continue

        prompt = prompt_label_segment(len(seg_images))
        seg_result: Dict[str, Any] = {}

        for attempt in range(MAX_RELABEL_RETRIES):
            try:
                seg_result = backend.infer(seg_images, prompt)
            except Exception as ex:
                print(f"[Fix]   seg {i} inference error: {ex}")
                seg_result = {}

            if isinstance(seg_result, dict) and seg_result.get("instruction", "").strip():
                break

            print(f"[Fix]   seg {i} attempt {attempt + 1}/{MAX_RELABEL_RETRIES} empty, retrying...")

        instruction = (seg_result.get("instruction") or "").strip()
        if not instruction:
            instruction = "Unknown"

        new_instructions.append(instruction)
        print(f"[Fix]   seg {i} frames [{s}, {e}): '{instruction}'")

    fixed = dict(vlm_json)
    fixed["instructions"] = new_instructions
    fixed["_relabeled"] = True
    return fixed


def run_worker(config: Config) -> None:
    """Run the worker loop."""
    server_url = config.worker.server_url
    
    # Create and warmup backend
    backend_kwargs = {}
    if config.worker.backend == "qwen3vl":
        backend_kwargs = {
            "model_path": config.worker.qwen3vl.model_path,
            "device_map": config.worker.qwen3vl.device_map,
        }
    elif config.worker.backend == "siliconflow":
        backend_kwargs = {
            "api_url": config.worker.siliconflow.api_url,
            "api_key": config.worker.siliconflow.api_key,
            "model_id": config.worker.siliconflow.model_id,
            "target_width": config.worker.siliconflow.target_width,
            "jpeg_quality": config.worker.siliconflow.jpeg_quality,
            "temperature": config.worker.siliconflow.temperature,
            "max_tokens": config.worker.siliconflow.max_tokens,
            "timeout_sec": config.worker.siliconflow.timeout_sec,
            "headers": config.worker.siliconflow.headers,
        }
    elif config.worker.backend == "remote_api":
        backend_kwargs = {
            "url": config.worker.remote_api.api_url,
            "api_key": config.worker.remote_api.api_key,
            "headers": config.worker.remote_api.headers,
            "timeout_sec": config.worker.remote_api.timeout_sec,
        }
    
    backend = create_backend(config.worker.backend, **backend_kwargs)
    print(f"[Worker] Using backend: {backend.name}")
    backend.warmup()
    
    print(f"[Worker] Connecting to {server_url}")
    
    max_connection_retries = 30  # ~60 seconds total
    connection_retry_count = 0
    
    try:
        while True:
            try:
                # Get job
                try:
                    r = requests.get(f"{server_url}/get_job", timeout=60)
                    connection_retry_count = 0  # Reset on successful connection
                except requests.exceptions.RequestException as e:
                    connection_retry_count += 1
                    if connection_retry_count >= max_connection_retries:
                        print(f"[Worker] Failed to connect after {max_connection_retries} retries. Exiting.")
                        break
                    print(f"[Worker] Waiting for server at {server_url}... (attempt {connection_retry_count}/{max_connection_retries})")
                    time.sleep(2)
                    continue
                
                if r.status_code != 200:
                    time.sleep(0.5)
                    continue
                
                resp = r.json()
                if resp.get("status") == "empty":
                    time.sleep(0.5)
                    continue
                
                job = resp.get("data")
                if job is None:
                    print("[Worker] Invalid job data received")
                    time.sleep(1)
                    continue
                task_id = job.get("task_id", "unknown")
                
                # Decode images
                images_b64 = job.get("images", [])
                images = []
                for b64 in images_b64:
                    img = decode_b64_to_numpy(b64)
                    if img is not None:
                        images.append(img)
                    else:
                        # Create dummy image
                        images.append(np.zeros((224, 224, 3), dtype=np.uint8))
                
                # Run inference with proper prompt (local retry on empty output)
                prompt = prompt_switch_detection(len(images))
                vlm_json: Dict[str, Any] = {}
                
                for attempt in range(MAX_LOCAL_RETRIES):
                    try:
                        vlm_json = backend.infer(images, prompt)
                    except Exception as e:
                        print(f"[Err] Inference failed: {e}")
                        vlm_json = {}
                    
                    if not _is_empty_vlm_json(vlm_json):
                        break
                    
                    print(
                        f"[Warn] {task_id} Empty VLM JSON "
                        f"(attempt {attempt + 1}/{MAX_LOCAL_RETRIES})"
                    )
                
                if _is_empty_vlm_json(vlm_json):
                    print(f"[Fail] {task_id} Returning empty to trigger server retry")
                else:
                    # Fix instruction/transition count mismatch (len(instr) must be len(trans)+1)
                    if _has_instruction_mismatch(vlm_json):
                        print(f"[Fix] {task_id} Instruction/transition count mismatch")
                        print(f"[Fix] {task_id} Original transitions: {vlm_json.get('transitions', [])}")
                        vlm_json = _relabel_segments(images, vlm_json, backend, task_id)
                        print(f"[Fix] {task_id} Relabeled segments: {vlm_json.get('instructions', [])}")

                    print(
                        f"[Done] {task_id} ({len(images)}f) "
                        f"-> Cuts: {vlm_json.get('transitions', [])} "
                        f"Instructions: {vlm_json.get('instructions', [])}"
                        + (" [relabeled]" if vlm_json.get("_relabeled") else "")
                    )
                
                # Build thumbnail for window_images/{window_id}.png (saved by server)
                thumbnail_b64 = build_thumbnail_b64(images) if images else None
                payload = {
                    "task_id": task_id,
                    "vlm_json": vlm_json,
                    "meta": job["meta"]
                }
                if thumbnail_b64 is not None:
                    payload["thumbnail_b64"] = thumbnail_b64
                requests.post(f"{server_url}/submit_result", json=payload, timeout=30)
            
            except KeyboardInterrupt:
                print("[Worker] Stopping...")
                break
            except Exception as e:
                print(f"[Error] Loop crashed: {e}")
                time.sleep(1)
    
    finally:
        backend.cleanup()
