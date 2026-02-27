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
from io import BytesIO
from pathlib import Path
from typing import Optional

try:
    import yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False

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


def build_report(sample_dir: Path, out_path: Path, n_frames: int = 16,
                 model_info: Optional[dict] = None) -> None:
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
  {"".join(cards_html)}
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

    out_path = Path(args.out) if args.out else sample_dir / "report.html"
    build_report(sample_dir, out_path, n_frames=n_frames, model_info=model_info)


if __name__ == "__main__":
    main()
