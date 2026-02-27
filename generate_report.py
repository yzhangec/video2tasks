"""
Generate a debug HTML report for a video2tasks sample run.

For each window:
  - Shows the thumbnail strip (window_images/{wid}.png)
  - Draws a red vertical highlight over transition frames
  - Shows thought, transitions, and instructions below

Usage:
    python generate_report.py <sample_dir> [--out report.html]

    # Example:
    python generate_report.py runs/test/siliconflow_test/samples/Arrange_Fruits_20250819_011_epi000000
"""

import argparse
import base64
import json
import os
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import List, Optional

try:
    import yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False

import cv2
import numpy as np
from PIL import Image, ImageDraw


# ─────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────

def find_config(sample_dir: Path) -> Optional[Path]:
    """Walk up from sample_dir to find the nearest config.yaml."""
    candidate = sample_dir.resolve()
    for _ in range(10):
        p = candidate / "config.yaml"
        if p.exists():
            return p
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    # Also check script directory
    script_cfg = Path(__file__).parent / "config.yaml"
    if script_cfg.exists():
        return script_cfg
    return None


def load_model_info(config_path: Optional[Path]) -> dict:
    """Extract backend and model info from config.yaml."""
    if config_path is None or not _YAML_OK:
        return {}
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        worker = cfg.get("worker", {})
        backend = worker.get("backend", "unknown")
        model = "unknown"
        if backend == "qwen3vl":
            model = worker.get("qwen3vl", {}).get("model_path", "unknown")
        elif backend == "siliconflow":
            sf = worker.get("siliconflow", {})
            model = sf.get("model_id", "unknown")
            api_url = sf.get("api_url", "")
        elif backend == "remote_api":
            model = worker.get("remote_api", {}).get("api_url", "unknown")
        windowing = cfg.get("windowing", {})
        run = cfg.get("run", {})
        return {
            "backend": backend,
            "model": model,
            "run_id": run.get("run_id", ""),
            "frames_per_window": windowing.get("frames_per_window", 16),
            "window_sec": windowing.get("window_sec", 16.0),
            "step_sec": windowing.get("step_sec", 8.0),
            "config_path": str(config_path),
        }
    except Exception as e:
        print(f"[Warn] Could not parse config: {e}", file=sys.stderr)
        return {}


# ─────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────

def annotate_strip(img: Image.Image, transitions: list[int], n_frames: int = 16) -> Image.Image:
    """Draw semi-transparent red columns over transition frames."""
    img = img.copy().convert("RGBA")
    w, h = img.size
    frame_w = w / n_frames

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for t in transitions:
        if 0 <= t < n_frames:
            x0 = int(t * frame_w)
            x1 = int((t + 1) * frame_w)
            # semi-transparent red fill
            draw.rectangle([x0, 0, x1, h - 1], fill=(220, 30, 30, 110))
            # solid red border
            draw.rectangle([x0, 0, x1, h - 1], outline=(220, 30, 30, 255), width=3)

    result = Image.alpha_composite(img, overlay)
    return result.convert("RGB")


def img_to_data_url(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def load_windows(windows_jsonl: Path) -> list[dict]:
    windows = []
    with open(windows_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                windows.append(json.loads(line))
    windows.sort(key=lambda x: x.get("window_id", 0))
    return windows


# ─────────────────────────────────────────────
# Video / segment helpers
# ─────────────────────────────────────────────

def find_video(sample_id: str, config_path: Optional[Path]) -> Optional[Path]:
    """Try to locate <sample_id>.mp4 from config datasets or by walking up the directory tree."""
    candidates: List[Path] = []

    # 1. Search dataset roots listed in config
    if config_path and _YAML_OK:
        try:
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            for ds in cfg.get("datasets", []):
                root = ds.get("root", "")
                subset = ds.get("subset", "")
                if root and subset:
                    candidates.append(Path(root) / subset / f"{sample_id}.mp4")
                if root:
                    candidates.append(Path(root) / f"{sample_id}.mp4")
        except Exception:
            pass

    # 2. Walk up from script dir
    base = Path(__file__).parent
    for sub in ("data", ".", "videos"):
        candidates.append(base / sub / f"{sample_id}.mp4")

    for p in candidates:
        if p.exists():
            return p
    return None


def extract_segment_frames(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
    sample_fps: float = 1.0,
    target_w: int = 320,
    jpeg_quality: int = 75,
) -> List[str]:
    """Extract frames at ~sample_fps rate from [start_frame, end_frame) and return as JPEG data URLs."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    step = max(1, int(round(fps / sample_fps)))
    frame_ids = list(range(start_frame, end_frame, step))
    if not frame_ids:
        frame_ids = [start_frame]

    data_urls: List[str] = []
    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue
        h, w = bgr.shape[:2]
        if w != target_w:
            target_h = int(h * target_w / w)
            bgr = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        b64 = base64.b64encode(buf).decode()
        data_urls.append(f"data:image/jpeg;base64,{b64}")

    cap.release()
    return data_urls


def load_segments_json(sample_dir: Path) -> Optional[dict]:
    p = sample_dir / "segments.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# HTML generation
# ─────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: "Segoe UI", system-ui, sans-serif;
    background: #0f1117;
    color: #e0e0e0;
    padding: 24px;
}
h1 {
    font-size: 1.4rem;
    color: #a0c4ff;
    margin-bottom: 8px;
    word-break: break-all;
}
.meta {
    font-size: 0.78rem;
    color: #666;
    margin-bottom: 10px;
}
.model-bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    margin-bottom: 28px;
    padding: 10px 14px;
    background: #131825;
    border: 1px solid #1e2840;
    border-radius: 8px;
}
.mb-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #445;
    margin-right: 2px;
}
.mb-badge {
    background: #0e1a2e;
    border: 1px solid #1e3a60;
    color: #7ab3f0;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.8rem;
    font-family: monospace;
}
.mb-badge.backend {
    border-color: #2a4a1e;
    background: #0e1e0e;
    color: #7ed97e;
}
.mb-sep {
    color: #2a2f45;
    font-size: 1rem;
}
.window-card {
    background: #1a1d2e;
    border: 1px solid #2a2f45;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 24px;
}
.window-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 12px;
}
.window-id {
    font-size: 1rem;
    font-weight: 700;
    color: #a0c4ff;
}
.task-id {
    font-size: 0.76rem;
    color: #556;
    font-family: monospace;
}
.strip-wrap {
    position: relative;
    margin-bottom: 14px;
}
.strip-wrap img {
    width: 100%;
    border-radius: 4px;
    display: block;
    image-rendering: auto;
}
/* frame index ruler */
.ruler {
    display: flex;
    margin-top: 3px;
    margin-bottom: 10px;
}
.ruler-cell {
    flex: 1;
    text-align: center;
    font-size: 0.6rem;
    color: #445;
    user-select: none;
}
.ruler-cell.transition {
    color: #ff6b6b;
    font-weight: 700;
}
.section-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #556;
    margin-bottom: 4px;
}
.thought {
    font-size: 0.82rem;
    line-height: 1.55;
    color: #b0b8c8;
    background: #111420;
    border-left: 3px solid #2a3560;
    padding: 10px 14px;
    border-radius: 0 6px 6px 0;
    margin-bottom: 12px;
    white-space: pre-wrap;
    word-break: break-word;
}
.instructions {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 6px;
}
.instr-chip {
    background: #1c2a1c;
    border: 1px solid #2d5a2d;
    color: #7ed97e;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
}
.transitions-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 6px;
}
.transitions-label {
    font-size: 0.72rem;
    color: #556;
}
.t-chip {
    background: #2a1010;
    border: 1px solid #7a2020;
    color: #ff8080;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.8rem;
    font-family: monospace;
}
.no-transitions {
    font-size: 0.78rem;
    color: #445;
    font-style: italic;
}
/* ── Section headings ── */
.section-heading {
    font-size: 1.1rem;
    font-weight: 700;
    color: #c0d8ff;
    margin: 36px 0 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid #2a2f45;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-heading .sh-badge {
    font-size: 0.72rem;
    font-weight: 400;
    background: #1e2840;
    color: #7ab3f0;
    padding: 2px 10px;
    border-radius: 10px;
}
/* ── Segment cards ── */
.seg-card {
    background: #181c28;
    border: 1px solid #252a3a;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 18px;
}
.seg-header {
    display: flex;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 12px;
}
.seg-id {
    font-size: 0.9rem;
    font-weight: 700;
    color: #a0c4ff;
    min-width: 52px;
}
.seg-instr {
    font-size: 0.92rem;
    color: #7ed97e;
    font-weight: 600;
    flex: 1;
}
.seg-meta {
    font-size: 0.72rem;
    color: #445;
    font-family: monospace;
    white-space: nowrap;
}
.seg-strip {
    display: flex;
    gap: 3px;
    overflow-x: auto;
    padding-bottom: 4px;
}
.seg-strip img {
    height: 100px;
    width: auto;
    border-radius: 3px;
    flex-shrink: 0;
    display: block;
}
.seg-no-video {
    font-size: 0.78rem;
    color: #445;
    font-style: italic;
    padding: 8px 0;
}
"""

WINDOW_TEMPLATE = """
<div class="window-card">
  <div class="window-header">
    <span class="window-id">Window {wid}</span>
    <span class="task-id">{task_id}</span>
  </div>
  <div class="strip-wrap">
    <img src="{img_data_url}" alt="window {wid} strip" />
  </div>
  <div class="ruler">{ruler_html}</div>
  <div class="section-label">Transitions</div>
  {transitions_html}
  <div class="section-label" style="margin-top:12px">Instructions</div>
  <div class="instructions">{instructions_html}</div>
  <div class="section-label" style="margin-top:12px">Thought</div>
  <div class="thought">{thought}</div>
</div>
"""


def build_ruler(n_frames: int, transitions: list[int]) -> str:
    cells = []
    for i in range(n_frames):
        cls = "ruler-cell transition" if i in transitions else "ruler-cell"
        cells.append(f'<div class="{cls}">{i}</div>')
    return "".join(cells)


def build_model_bar(info: dict) -> str:
    """Render the model/config info bar HTML."""
    if not info:
        return ""
    parts = [
        f'<span class="mb-label">Backend</span>'
        f'<span class="mb-badge backend">{info.get("backend", "?")}</span>',
        f'<span class="mb-sep">·</span>'
        f'<span class="mb-label">Model</span>'
        f'<span class="mb-badge">{info.get("model", "?")}</span>',
    ]
    if info.get("run_id"):
        parts.append(
            f'<span class="mb-sep">·</span>'
            f'<span class="mb-label">Run</span>'
            f'<span class="mb-badge">{info["run_id"]}</span>'
        )
    win_sec = info.get("window_sec", "")
    step_sec = info.get("step_sec", "")
    fpw = info.get("frames_per_window", "")
    if win_sec and step_sec and fpw:
        parts.append(
            f'<span class="mb-sep">·</span>'
            f'<span class="mb-label">Window</span>'
            f'<span class="mb-badge">{win_sec}s / step {step_sec}s / {fpw} frames</span>'
        )
    return f'<div class="model-bar">{"".join(parts)}</div>'


def build_segments_section(
    sample_dir: Path,
    video_path: Optional[Path],
) -> str:
    """Build the HTML for the Segments section from segments.json."""
    seg_data = load_segments_json(sample_dir)
    if seg_data is None:
        return '<p class="seg-no-video">segments.json not found.</p>'

    segments = seg_data.get("segments", [])
    nframes_total = seg_data.get("nframes", 0)

    # Read video fps once
    fps = 30.0
    if video_path:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

    # Metadata from segments.json (may override config)
    seg_backend = seg_data.get("backend", "")
    seg_model = seg_data.get("model", "")
    meta_parts = [f"{len(segments)} segments", f"{nframes_total} frames total"]
    if seg_backend:
        meta_parts.append(f"backend: {seg_backend}")
    if seg_model:
        meta_parts.append(f"model: {seg_model}")

    html_parts = [
        f'<h2 class="section-heading">Segments <span class="sh-badge">{" · ".join(meta_parts)}</span></h2>'
    ]

    for seg in segments:
        seg_id = seg["seg_id"]
        start_f = seg["start_frame"]
        end_f = seg["end_frame"]
        instruction = seg.get("instruction", "")
        duration_s = (end_f - start_f) / fps

        # Extract 1fps frames
        if video_path:
            data_urls = extract_segment_frames(
                video_path, start_f, end_f, fps,
                sample_fps=1.0, target_w=320, jpeg_quality=75
            )
            n_samples = len(data_urls)
            strip_html = "".join(
                f'<img src="{url}" alt="seg {seg_id} frame" />'
                for url in data_urls
            )
            strip_html = f'<div class="seg-strip">{strip_html}</div>'
        else:
            n_samples = 0
            strip_html = '<p class="seg-no-video">No video found — pass --video to enable frame sampling.</p>'

        meta_str = f"frames {start_f}–{end_f} · {duration_s:.1f}s · {n_samples} samples @ 1fps"

        html_parts.append(f"""
<div class="seg-card">
  <div class="seg-header">
    <span class="seg-id">Seg {seg_id}</span>
    <span class="seg-instr">{instruction}</span>
    <span class="seg-meta">{meta_str}</span>
  </div>
  {strip_html}
</div>""")

        print(f"  Seg {seg_id}: [{start_f}, {end_f}) {duration_s:.1f}s  '{instruction}'  ({n_samples} frames)")

    return "\n".join(html_parts)


def build_report(sample_dir: Path, out_path: Path, n_frames: int = 16,
                 model_info: Optional[dict] = None,
                 video_path: Optional[Path] = None) -> None:
    windows_jsonl = sample_dir / "windows.jsonl"
    img_dir = sample_dir / "window_images"

    if not windows_jsonl.exists():
        print(f"[Error] windows.jsonl not found: {windows_jsonl}", file=sys.stderr)
        sys.exit(1)

    windows = load_windows(windows_jsonl)
    print(f"Loaded {len(windows)} windows from {windows_jsonl}")

    cards_html = []
    for rec in windows:
        wid = rec.get("window_id", 0)
        task_id = rec.get("task_id", "")
        vlm = rec.get("vlm_json", {})

        transitions = [int(t) for t in vlm.get("transitions", [])]
        instructions = vlm.get("instructions", [])
        thought = vlm.get("thought", "").strip()

        # Load & annotate image
        img_path = img_dir / f"{wid}.png"
        if img_path.exists():
            img = Image.open(img_path)
            annotated = annotate_strip(img, transitions, n_frames=n_frames)
            img_data_url = img_to_data_url(annotated)
        else:
            # Placeholder if image missing
            placeholder = Image.new("RGB", (2560, 120), color=(30, 30, 40))
            img_data_url = img_to_data_url(placeholder)
            print(f"  [Warn] Image not found: {img_path}")

        ruler_html = build_ruler(n_frames, transitions)

        if transitions:
            t_chips = "".join(f'<span class="t-chip">frame {t}</span>' for t in transitions)
            transitions_html = f'<div class="transitions-row"><span class="transitions-label">Cut at:</span>{t_chips}</div>'
        else:
            transitions_html = '<div class="no-transitions">No transitions detected</div>'

        instr_chips = "".join(
            f'<span class="instr-chip">{i + 1}. {inst}</span>'
            for i, inst in enumerate(instructions)
        )

        card = WINDOW_TEMPLATE.format(
            wid=wid,
            task_id=task_id,
            img_data_url=img_data_url,
            ruler_html=ruler_html,
            transitions_html=transitions_html,
            instructions_html=instr_chips or '<span style="color:#445;font-style:italic">None</span>',
            thought=thought or "(no thought)",
        )
        cards_html.append(card)
        print(f"  Window {wid}: transitions={transitions}, instructions={len(instructions)}")

    # ── Section 2: Segments ──
    print("\nBuilding segments section...")
    segments_html = build_segments_section(sample_dir, video_path)

    sample_name = sample_dir.name
    model_bar_html = build_model_bar(model_info or {})
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Debug Report – {sample_name}</title>
  <style>{CSS}</style>
</head>
<body>
  <h1>{sample_name}</h1>
  <p class="meta">Source: {sample_dir.resolve()} &nbsp;|&nbsp; Windows: {len(windows)}</p>
  {model_bar_html}
  <h2 class="section-heading">Windows <span class="sh-badge">{len(windows)} windows</span></h2>
  {"".join(cards_html)}
  {segments_html}
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"\nReport written to: {out_path.resolve()}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate debug HTML report for a sample")
    parser.add_argument(
        "sample_dir",
        nargs="?",
        default="runs/test/siliconflow_test/samples/Arrange_Fruits_20250819_011_epi000000",
        help="Path to the sample directory containing windows.jsonl and window_images/",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output HTML file path (default: <sample_dir>/report.html)",
    )
    parser.add_argument(
        "--frames-per-window",
        type=int,
        default=None,
        dest="frames_per_window",
        help="Number of frames per window strip image (default: read from config, fallback 16)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml (auto-detected if omitted)",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Path to source .mp4 (auto-detected from config datasets if omitted)",
    )
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    if not sample_dir.exists():
        print(f"[Error] Sample dir not found: {sample_dir}", file=sys.stderr)
        sys.exit(1)

    # Load model info from config
    config_path = Path(args.config) if args.config else find_config(sample_dir)
    if config_path:
        print(f"[Config] {config_path}")
    model_info = load_model_info(config_path)

    # frames_per_window: CLI > config > default 16
    if args.frames_per_window is not None:
        n_frames = args.frames_per_window
    elif model_info.get("frames_per_window"):
        n_frames = model_info["frames_per_window"]
    else:
        n_frames = 16

    # Locate video for segment frame sampling
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"[Warn] Video not found: {video_path}", file=sys.stderr)
            video_path = None
    else:
        sample_id = sample_dir.name
        video_path = find_video(sample_id, config_path)
        if video_path:
            print(f"[Video] {video_path}")
        else:
            print("[Video] Not found — segments section will skip frame sampling.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else sample_dir / f"report_{ts}.html"
    build_report(sample_dir, out_path, n_frames=n_frames, model_info=model_info,
                 video_path=video_path)


if __name__ == "__main__":
    main()
