"""FastAPI server for job queue management."""

import os
import json
import time
import glob
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

from ..config import Config, DatasetConfig
from .windowing import (
    read_video_info, build_windows, FrameExtractor,
    build_segments_via_cuts, Window
)


class SubmitModel(BaseModel):
    """Model for job result submission."""
    task_id: str
    vlm_output: str = ""
    vlm_json: Dict[str, Any] = Field(default_factory=dict)
    latency_s: float = 0.0
    meta: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class DatasetCtx:
    """Dataset context for processing."""
    data_root: str
    subset: str
    data_dir: str
    run_dir: str
    samples_dir: str
    sample_ids: List[str]


def parse_datasets(config: Config) -> List[DatasetCtx]:
    """Parse dataset configurations into contexts."""
    ctxs = []
    for ds in config.datasets:
        data_dir = Path(ds.root) / ds.subset
        run_dir = Path(config.run.base_dir) / ds.subset / config.run.run_id
        samples_dir = run_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        # List sample IDs: subdirs (each with Frame_*.mp4) or flat Frame_*.mp4 in data_dir
        if data_dir.exists():
            subdirs = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
            if subdirs:
                sample_ids = subdirs
            else:
                sample_ids = sorted([p.stem for p in data_dir.glob("Frame_*.mp4")])
        else:
            sample_ids = []
        
        ctxs.append(DatasetCtx(
            data_root=ds.root,
            subset=ds.subset,
            data_dir=str(data_dir),
            run_dir=str(run_dir),
            samples_dir=str(samples_dir),
            sample_ids=sample_ids
        ))
    return ctxs


def create_app(config: Config) -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(title="Video2Tasks Server")
    
    # Initialize dataset contexts
    dataset_ctxs = parse_datasets(config)
    samples_dir_by_subset = {ctx.subset: ctx.samples_dir for ctx in dataset_ctxs}
    data_dir_by_subset = {ctx.subset: ctx.data_dir for ctx in dataset_ctxs}
    
    # Thread-safe job management
    queue_lock = threading.Lock()
    job_queue: List[Dict[str, Any]] = []
    inflight: Dict[str, Dict[str, Any]] = {}
    retry_counts: Dict[str, int] = {}
    
    # Per-sample locks
    _sample_locks: Dict[str, threading.Lock] = {}
    _sample_locks_lock = threading.Lock()
    
    def get_sample_lock(sample_key: str) -> threading.Lock:
        with _sample_locks_lock:
            if sample_key not in _sample_locks:
                _sample_locks[sample_key] = threading.Lock()
            return _sample_locks[sample_key]
    
    def sample_out_dir(samples_dir: str, sample_id: str) -> str:
        p = Path(samples_dir) / sample_id
        p.mkdir(parents=True, exist_ok=True)
        return str(p)
    
    def windows_jsonl_path(samples_dir: str, sample_id: str) -> str:
        return str(Path(sample_out_dir(samples_dir, sample_id)) / "windows.jsonl")
    
    def segments_path(samples_dir: str, sample_id: str) -> str:
        return str(Path(sample_out_dir(samples_dir, sample_id)) / "segments.json")
    
    def done_marker_path(samples_dir: str, sample_id: str) -> str:
        return str(Path(sample_out_dir(samples_dir, sample_id)) / ".DONE")
    
    @app.get("/get_job")
    def get_job() -> Dict[str, Any]:
        with queue_lock:
            if not job_queue:
                return {"status": "empty"}
            job = job_queue.pop(0)
            inflight[job["task_id"]] = {"ts": time.time(), "job": job}
            return {"status": "ok", "data": job}
    
    @app.post("/submit_result")
    def submit_result(res: SubmitModel) -> Dict[str, str]:
        tid = res.task_id
        job_info = None
        
        with queue_lock:
            if tid in inflight:
                job_info = inflight.pop(tid)
        
        # Empty result: trigger retry
        if not res.vlm_json:
            if job_info:
                with queue_lock:
                    retry_counts[tid] = retry_counts.get(tid, 0) + 1
                    if retry_counts[tid] <= config.server.max_retries_per_job:
                        job_queue.insert(0, job_info["job"])
                        print(f"[Warn] Task {tid} empty, re-queueing (attempt {retry_counts[tid]})")
                    else:
                        print(f"[Err] Task {tid} failed max retries, dropping")
            return {"status": "retry_triggered"}
        
        subset = str(res.meta.get("subset", dataset_ctxs[0].subset if dataset_ctxs else "default"))
        sid = str(res.meta.get("sample_id", "unknown"))
        w_id = res.meta.get("window_id")
        
        samples_dir = samples_dir_by_subset.get(subset)
        if not samples_dir:
            samples_dir = str(Path(config.run.base_dir) / subset / config.run.run_id / "samples")
            Path(samples_dir).mkdir(parents=True, exist_ok=True)
        
        rec = {"task_id": tid, "window_id": w_id, "vlm_json": res.vlm_json}
        
        sample_key = f"{subset}::{sid}"
        with get_sample_lock(sample_key):
            with open(windows_jsonl_path(samples_dir, sid), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        
        return {"status": "received"}
    
    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}
    
    # Producer loop
    def producer_loop():
        # Compute progress totals
        total = sum(len(ctx.sample_ids) for ctx in dataset_ctxs)
        progress_total = config.progress.total_override if config.progress.total_override > 0 else total
        
        done = 0
        for ctx in dataset_ctxs:
            for sid in ctx.sample_ids:
                if Path(done_marker_path(ctx.samples_dir, sid)).exists():
                    done += 1
        
        print(
            f"[Server] Started. IMG=PNG, "
            f"FIXED={config.windowing.target_width}x{config.windowing.target_height}, "
            f"FRAMES_PER_WINDOW={config.windowing.frames_per_window}\n"
            f"[Plan] DATASETS={[(c.data_dir, c.subset) for c in dataset_ctxs]}\n"
            f"[Resume] Already done: {done}/{progress_total} (computed_total={total})"
        )
        
        # Initialize states
        states = {}
        for ctx in dataset_ctxs:
            states[ctx.subset] = {
                "cur_idx": 0,
                "sample_status": {sid: 0 for sid in ctx.sample_ids},
            }
        
        dataset_idx = 0
        global_done = done
        
        while True:
            # Check inflight timeouts
            now = time.time()
            with queue_lock:
                expired = [
                    tid for tid, info in inflight.items()
                    if now - info["ts"] > config.server.inflight_timeout_sec
                ]
                for tid in expired:
                    job = inflight.pop(tid)["job"]
                    retry_counts[tid] = retry_counts.get(tid, 0) + 1
                    if retry_counts[tid] <= config.server.max_retries_per_job:
                        job_queue.append(job)
            
            # All datasets done
            if dataset_idx >= len(dataset_ctxs):
                if config.server.auto_exit_after_all_done:
                    print(f"[All Done] {global_done}/{progress_total}. Exiting.")
                    os._exit(0)
                time.sleep(1.0)
                continue
            
            ctx = dataset_ctxs[dataset_idx]
            st = states[ctx.subset]
            cur_idx = st["cur_idx"]
            sample_status = st["sample_status"]
            sample_ids = ctx.sample_ids
            
            # Current dataset done, wait for queue to clear
            if cur_idx >= len(sample_ids):
                with queue_lock:
                    if not job_queue and not inflight:
                        print(f"[Dataset] Completed {ctx.subset}. Switching to next...")
                        dataset_idx += 1
                time.sleep(0.2)
                continue
            
            # Produce jobs if queue not full
            with queue_lock:
                q_len = len(job_queue)
            
            if q_len < config.server.max_queue:
                sid = sample_ids[cur_idx]
                s_dir = Path(ctx.data_dir) / sid

                # Skip if already done
                if Path(done_marker_path(ctx.samples_dir, sid)).exists():
                    sample_status[sid] = 3
                    st["cur_idx"] += 1
                    time.sleep(0.01)
                    continue

                # Find video: subdir with Frame_*.mp4, or flat Frame_<sid>.mp4 in data_dir
                if s_dir.is_dir():
                    mp4s = list(s_dir.glob("Frame_*.mp4"))
                else:
                    flat_mp4 = Path(ctx.data_dir) / f"{sid}.mp4"
                    mp4s = [flat_mp4] if flat_mp4.exists() else []
                if not mp4s:
                    st["cur_idx"] += 1
                    time.sleep(0.01)
                    continue
                mp4 = str(mp4s[0])
                
                w_path = windows_jsonl_path(ctx.samples_dir, sid)
                
                # Step A: Generate window tasks
                if sample_status[sid] == 0:
                    try:
                        fps, nframes = read_video_info(mp4)
                        windows = build_windows(
                            fps, nframes,
                            config.windowing.window_sec,
                            config.windowing.step_sec,
                            config.windowing.frames_per_window
                        )
                        
                        # Load completed windows
                        done_wids = set()
                        if Path(w_path).exists():
                            with open(w_path, "r", encoding="utf-8") as f:
                                for line in f:
                                    try:
                                        done_wids.add(json.loads(line)["window_id"])
                                    except (json.JSONDecodeError, KeyError) as e:
                                        print(f"[Warn] Corrupted line in {w_path}: {e}")
                        
                        with FrameExtractor(mp4) as extractor:
                            cnt = 0
                            
                            for w in windows:
                                if w.window_id in done_wids:
                                    continue
                                
                                tid = f"{ctx.subset}::{sid}_w{w.window_id}"
                                
                                # Check if already active
                                active = False
                                with queue_lock:
                                    if any(j["task_id"] == tid for j in job_queue) or tid in inflight:
                                        active = True
                                
                                if active:
                                    continue
                                
                                job = {
                                    "task_id": tid,
                                    "images": extractor.get_many_b64(
                                        w.frame_ids,
                                        config.windowing.target_width,
                                        config.windowing.target_height,
                                        config.windowing.png_compression
                                    ),
                                    "meta": {
                                        "subset": ctx.subset,
                                        "sample_id": sid,
                                        "window_id": w.window_id,
                                        "frame_ids": w.frame_ids
                                    }
                                }
                                
                                with queue_lock:
                                    job_queue.append(job)
                                
                                cnt += 1
                                if cnt > 20:
                                    break
                        
                        if cnt == 0:
                            sample_status[sid] = 2
                    
                    except Exception as e:
                        print(f"[Err] {ctx.subset}/{sid}: {e}")
                        import traceback
                        traceback.print_exc()
                        st["cur_idx"] += 1
                
                # Step B: Finalize
                if sample_status[sid] == 2:
                    try:
                        fps, nframes = read_video_info(mp4)
                        windows = build_windows(
                            fps, nframes,
                            config.windowing.window_sec,
                            config.windowing.step_sec,
                            config.windowing.frames_per_window
                        )
                        
                        by_wid = {}
                        if Path(w_path).exists():
                            with open(w_path, "r", encoding="utf-8") as f:
                                for line in f:
                                    try:
                                        d = json.loads(line)
                                        by_wid[d["window_id"]] = d
                                    except (json.JSONDecodeError, KeyError):
                                        pass
                        
                        if len(by_wid) >= len(windows):
                            print(f"[Finalize] {ctx.subset}/{sid}...")
                            
                            final_res = build_segments_via_cuts(
                                sid, windows, by_wid, fps, nframes,
                                config.windowing.frames_per_window
                            )
                            
                            with open(segments_path(ctx.samples_dir, sid), "w", encoding="utf-8") as f:
                                json.dump(final_res, f, indent=2, ensure_ascii=False)
                            
                            done_path = done_marker_path(ctx.samples_dir, sid)
                            already_done = Path(done_path).exists()
                            Path(done_path).touch()
                            
                            sample_status[sid] = 3
                            st["cur_idx"] += 1
                            
                            if not already_done:
                                global_done += 1
                            print(f"[Progress] {global_done}/{progress_total} (finished: {ctx.subset}/{sid})")
                    
                    except Exception as e:
                        print(f"[Err-Finalize] {ctx.subset}/{sid}: {e}")
            
            time.sleep(0.1)
    
    # Start producer thread
    producer_thread = threading.Thread(target=producer_loop, daemon=True)
    producer_thread.start()
    
    return app


def run_server(config: Config) -> None:
    """Run the server with given configuration."""
    app = create_app(config)
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.logging.level.lower()
    )
