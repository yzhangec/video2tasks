"""
Split a video into segments based on frame indices from segments.json.

Uses frame-accurate trimming via ffmpeg's video filter (re-encodes to ensure
exact frame boundaries, no overlap between segments).

Usage:
    python split_segments.py <video_path> <segments_json> [--output_dir <dir>]
"""

import argparse
import json
import os
import subprocess
import sys


def get_video_info(video_path: str) -> dict:
    """Get video fps and total frames via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,nb_frames,codec_name,width,height",
        "-of", "json",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    stream = info["streams"][0]

    # Parse fractional fps like "30000/1001"
    num, den = map(int, stream["r_frame_rate"].split("/"))
    fps = num / den

    return {
        "fps": fps,
        "r_frame_rate": stream["r_frame_rate"],
        "nb_frames": int(stream["nb_frames"]),
        "codec": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
    }


def split_video_by_frames(
    video_path: str,
    segments_json: str,
    output_dir: str | None = None,
) -> list[str]:
    """
    Split video into segments using exact frame indices.

    Each segment is cut using ffmpeg's `trim` video filter with
    start_frame / end_frame, which guarantees frame-accurate boundaries
    (requires re-encoding).
    """
    # Load segments
    with open(segments_json) as f:
        data = json.load(f)

    segments = data["segments"]
    if not segments:
        print("No segments found.")
        return []

    # Resolve output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(segments_json), "segments")
    os.makedirs(output_dir, exist_ok=True)

    # Get video info
    info = get_video_info(video_path)
    fps = info["fps"]
    r_frame_rate = info["r_frame_rate"]
    print(f"Video: {video_path}")
    print(f"  fps={fps:.4f}  frames={info['nb_frames']}  "
          f"codec={info['codec']}  {info['width']}x{info['height']}")
    print(f"Output: {output_dir}")
    print(f"Segments: {len(segments)}\n")

    output_files = []
    for seg in segments:
        seg_id = seg["seg_id"]
        start_frame = seg["start_frame"]
        end_frame = seg["end_frame"]
        instruction = seg.get("instruction", "")
        n_frames = end_frame - start_frame

        # Sanitize instruction for filename
        safe_name = instruction.replace(" ", "_").replace("/", "_")
        out_file = os.path.join(output_dir, f"seg_{seg_id:02d}_{safe_name}.mp4")

        # Use trim filter with frame numbers for exact cutting.
        # trim=start_frame=N:end_frame=M  selects frames [N, M)
        # setpts=PTS-STARTPTS  resets timestamps to start from 0
        # Same for audio: atrim + asetpts
        vf = f"trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS"
        af = f"atrim=start={start_frame / fps}:end={end_frame / fps},asetpts=PTS-STARTPTS"

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", vf,
            "-af", af,
            "-r", r_frame_rate,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "aac",
            out_file,
        ]

        print(f"[seg {seg_id:02d}] frames {start_frame:>5d} - {end_frame:>5d}  "
              f"({n_frames:>4d} frames, {n_frames / fps:>6.2f}s)  {instruction}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr[-500:]}")
        else:
            # Verify output frame count
            out_info = get_video_info(out_file)
            actual = out_info["nb_frames"]
            status = "OK" if actual == n_frames else f"WARN: got {actual} frames"
            print(f"  -> {out_file}  ({status})")

        output_files.append(out_file)

    print(f"\nDone! {len(output_files)} segments written to {output_dir}")
    return output_files


VIDEO_FILE = "/home/eason/workspace/video2tasks/data/OpenGalaxea/output.mp4"
SEGMENTS_JSON = "/home/eason/workspace/video2tasks/runs/OpenGalaxea/siliconflow_test/samples/output/segments.json"
OUTPUT_DIR = "/home/eason/workspace/video2tasks/runs/OpenGalaxea/siliconflow_test/samples/output/segments"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split video by frame indices from segments.json")
    parser.add_argument("-v", "--video", default=VIDEO_FILE, help="Path to the source video file")
    parser.add_argument("-s", "--segments_json", default=SEGMENTS_JSON, help="Path to segments.json")
    parser.add_argument("-o", "--output_dir", default=OUTPUT_DIR,
                        help="Output directory (default: <segments_json_dir>/segments)")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"Error: video not found: {args.video}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.segments_json):
        print(f"Error: segments.json not found: {args.segments_json}", file=sys.stderr)
        sys.exit(1)

    split_video_by_frames(args.video, args.segments_json, args.output_dir)
