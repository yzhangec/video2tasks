#!/usr/bin/env python3
"""
Generate segments.json for each episode in a LeRobot dataset.

For each episode, reads the per-frame task_index from the parquet file,
groups consecutive frames with the same task_index into segments, maps
each task_index to its instruction via meta/tasks.jsonl, and writes
the result to segments/episode_xxxxxx.json.

Usage:
    python generate_segments.py [dataset_dir ...]

If no arguments are given, processes the two default datasets.
"""

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq


DEFAULT_DATASETS = [
    "/mnt/nas-uav-sharespace/dataset/OpenGalaxea/lerobot/Arrange_Fruits_20250819_011",
    "/mnt/nas-uav-sharespace/dataset/OpenGalaxea/lerobot/Desktop_Garbage_Organizing20250721_008",
]


def load_task_map(meta_dir: Path) -> dict[int, str]:
    """Load task_index -> instruction mapping from tasks.jsonl."""
    task_map = {}
    tasks_file = meta_dir / "tasks.jsonl"
    with open(tasks_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            task_text = data["task"]
            # Tasks may be formatted as "chinese@english"; keep only english part
            if "@" in task_text:
                task_text = task_text.split("@", 1)[1].strip()
            task_map[data["task_index"]] = task_text
    return task_map


def build_segments(task_indices: list[int], task_map: dict[int, str]) -> list[dict]:
    """Group consecutive identical task_index values into segments."""
    if not task_indices:
        return []

    segments = []
    seg_id = 0
    current_idx = task_indices[0]
    start_frame = 0

    for frame, idx in enumerate(task_indices[1:], start=1):
        if idx != current_idx:
            segments.append({
                "seg_id": seg_id,
                "start_frame": start_frame,
                "end_frame": frame,
                "instruction": task_map.get(current_idx, f"task_{current_idx}"),
                "confidence": 1.0,
            })
            seg_id += 1
            current_idx = idx
            start_frame = frame

    # Last segment ends at nframes
    nframes = len(task_indices)
    segments.append({
        "seg_id": seg_id,
        "start_frame": start_frame,
        "end_frame": nframes,
        "instruction": task_map.get(current_idx, f"task_{current_idx}"),
        "confidence": 1.0,
    })

    return segments


def process_dataset(dataset_dir: str) -> None:
    root = Path(dataset_dir)
    dataset_name = root.name
    data_dir = root / "data" / "chunk-000"
    meta_dir = root / "meta"
    segments_dir = root / "segments"
    segments_dir.mkdir(exist_ok=True)

    task_map = load_task_map(meta_dir)
    parquet_files = sorted(data_dir.glob("episode_*.parquet"))

    print(f"[{dataset_name}] Found {len(parquet_files)} episodes")

    for parquet_path in parquet_files:
        episode_stem = parquet_path.stem  # e.g. "episode_000000"
        episode_num = episode_stem.split("_", 1)[1]  # e.g. "000000"

        table = pq.read_table(str(parquet_path), columns=["task_index"])
        task_indices = table.column("task_index").to_pylist()
        nframes = len(task_indices)

        segments = build_segments(task_indices, task_map)

        output = {
            "sample_id": f"{dataset_name}_epi{episode_num}",
            "nframes": nframes,
            "segments": segments,
        }

        out_path = segments_dir / f"{episode_stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"  Written: {out_path.name}  ({len(segments)} segments, {nframes} frames)")


def main() -> None:
    dataset_dirs = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_DATASETS
    for d in dataset_dirs:
        process_dataset(d)
    print("Done.")


if __name__ == "__main__":
    main()
