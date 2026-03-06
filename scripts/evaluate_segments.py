#!/usr/bin/env python3
"""
Evaluate VLM segment predictions against ground-truth segments.

Metrics
-------
frame_semantic_sim
    Frame-weighted cosine similarity of instruction sentence embeddings.
    Each frame gets the instruction of the segment it belongs to; similarity
    is averaged over all frames.  Handles granularity mismatch naturally.

boundary_precision / boundary_recall / boundary_f1
    How well the predicted segment boundaries match the GT boundaries,
    within a tolerance window of ±tau frames.

Usage
-----
# Single prediction file (GT auto-resolved from sample_id)
python evaluate_segments.py path/to/pred/segments.json

# All samples inside a run directory
python evaluate_segments.py --samples_dir runs/test/my_run/samples

# Override defaults
python evaluate_segments.py --samples_dir runs/test/my_run/samples \\
    --gt_base /mnt/.../OpenGalaxea/lerobot \\
    --tau 30 \\
    --model paraphrase-multilingual-mpnet-base-v2
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# ─── defaults ───────────────────────────────────────────────────────────────

DEFAULT_GT_BASE    = "/mnt/nas-uav-sharespace/dataset/OpenGalaxea/lerobot"
DEFAULT_MODEL      = "paraphrase-multilingual-mpnet-base-v2"
DEFAULT_TAU        = 50   # boundary tolerance in frames
DEFAULT_SAMPLE_DIR = "/home/eason/workspace/video2tasks/runs/OpenGalaxea/video2tasks/samples"

# ─── helpers ────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_gt_path(sample_id: str, gt_base: str) -> Path:
    """
    Derive GT segments.json path from a sample_id or directory name.

    Expected format:  {dataset_name}_epi{episode_num}[_optional_suffix]
    e.g.  Arrange_Fruits_20250819_011_epi000000
          Arrange_Fruits_20250819_011_epi000000_qwen30b   (suffix ignored)

    dataset_name : everything before the last '_epi'
    episode_num  : the leading digits immediately after '_epi'
                   (any trailing model-name suffix is ignored)
    """
    idx = sample_id.rfind("_epi")
    if idx == -1:
        raise ValueError(f"Cannot parse sample_id: {sample_id!r}  (expected '_epi' separator)")
    dataset_name = sample_id[:idx]
    after_epi    = sample_id[idx + len("_epi"):]
    m = re.match(r"(\d+)", after_epi)
    if not m:
        raise ValueError(f"No episode number found after '_epi' in: {sample_id!r}")
    episode_num = m.group(1)
    return Path(gt_base) / dataset_name / "segments" / f"episode_{episode_num}.json"


def segments_to_frame_labels(segments: list[dict], nframes: int) -> list[str]:
    """Expand segment list into a per-frame instruction array."""
    labels = [""] * nframes
    for seg in segments:
        for f in range(seg["start_frame"], min(seg["end_frame"], nframes)):
            labels[f] = seg["instruction"]
    return labels


# ─── metrics ────────────────────────────────────────────────────────────────

def frame_semantic_similarity(
    gt: dict,
    pred: dict,
    model: SentenceTransformer,
) -> float:
    """
    Compute average cosine similarity between GT and pred instruction
    embeddings, weighted equally by frame.
    """
    nframes = gt["nframes"]
    gt_labels   = segments_to_frame_labels(gt["segments"],   nframes)
    pred_labels = segments_to_frame_labels(pred["segments"], nframes)

    # Encode all unique texts in a single batch for efficiency
    unique_texts = list(dict.fromkeys(gt_labels + pred_labels))  # preserves order, deduplicates
    embeddings   = model.encode(unique_texts, normalize_embeddings=True, show_progress_bar=False)
    emb_map      = {t: e for t, e in zip(unique_texts, embeddings)}

    gt_embs   = np.array([emb_map[t] for t in gt_labels])
    pred_embs = np.array([emb_map[t] for t in pred_labels])

    # Cosine sim = dot product when vectors are L2-normalised
    frame_sims = (gt_embs * pred_embs).sum(axis=1)
    return float(frame_sims.mean())


def boundary_metrics(gt: dict, pred: dict, tau: int) -> dict:
    """
    Compute precision, recall, F1 for segment boundary detection.

    A boundary is defined as the end_frame of every segment except the last.
    A predicted boundary is a true positive if it falls within ±tau frames
    of any GT boundary (each GT boundary may only be matched once).
    """
    gt_bounds   = sorted(seg["end_frame"] for seg in gt["segments"][:-1])
    pred_bounds = sorted(seg["end_frame"] for seg in pred["segments"][:-1])

    if not gt_bounds and not pred_bounds:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    # Greedy matching: for each pred boundary find the nearest unmatched GT boundary
    matched_gt = set()
    tp = 0
    for p in pred_bounds:
        best_dist, best_idx = float("inf"), -1
        for i, g in enumerate(gt_bounds):
            if i in matched_gt:
                continue
            d = abs(p - g)
            if d < best_dist:
                best_dist, best_idx = d, i
        if best_dist <= tau:
            matched_gt.add(best_idx)
            tp += 1

    precision = tp / len(pred_bounds) if pred_bounds else 0.0
    recall    = tp / len(gt_bounds)   if gt_bounds   else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


# ─── per-sample evaluation ───────────────────────────────────────────────────

def evaluate_one(
    pred_path: Path,
    gt_base: str,
    model: SentenceTransformer,
    tau: int,
) -> dict | None:
    pred = load_json(pred_path)
    sample_id = pred.get("sample_id", pred_path.parent.name)

    gt_path = resolve_gt_path(sample_id, gt_base)
    if not gt_path.exists():
        print(f"  [WARN] GT not found: {gt_path}", file=sys.stderr)
        return None

    gt = load_json(gt_path)

    sem_sim  = frame_semantic_similarity(gt, pred, model)
    boundary = boundary_metrics(gt, pred, tau)

    return {
        "sample_id":          sample_id,
        "nframes":            gt["nframes"],
        "n_gt_segs":          len(gt["segments"]),
        "n_pred_segs":        len(pred["segments"]),
        "frame_semantic_sim": round(sem_sim, 4),
        "boundary_precision": round(boundary["precision"], 4),
        "boundary_recall":    round(boundary["recall"],    4),
        "boundary_f1":        round(boundary["f1"],        4),
    }


# ─── reporting ───────────────────────────────────────────────────────────────

def group_name(sample_id: str) -> str:
    """Return the dataset portion of a sample_id (everything before '_epi')."""
    idx = sample_id.rfind("_epi")
    return sample_id[:idx] if idx != -1 else sample_id


def print_results(results: list[dict], tau: int) -> None:
    if not results:
        print("No results.")
        return

    metric_keys = ["frame_semantic_sim", "boundary_precision", "boundary_recall", "boundary_f1"]

    def mean_row(label: str, rows: list[dict], col_w: int) -> str:
        avgs = {k: np.mean([r[k] for r in rows]) for k in metric_keys}
        return (
            f"{label:<{col_w}}"
            f"{'':>7}  {'':>7}  {'':>9}  "
            f"{avgs['frame_semantic_sim']:>7.4f}  "
            f"{avgs['boundary_precision']:>6.4f}  "
            f"{avgs['boundary_recall']:>6.4f}  "
            f"{avgs['boundary_f1']:>6.4f}"
        )

    col_w = max(len(r["sample_id"]) for r in results) + 2
    header = (
        f"{'sample_id':<{col_w}}"
        f"{'frames':>7}  {'GT segs':>7}  {'Pred segs':>9}  "
        f"{'sem_sim':>7}  {'B-prec':>6}  {'B-rec':>6}  {'B-F1':>6}"
    )
    sep  = "-" * len(header)
    thin = "·" * len(header)

    print(sep)
    print(f"Boundary tolerance τ = {tau} frames")
    print(sep)
    print(header)
    print(sep)

    # Group consecutive results by dataset name and print sub-means
    from itertools import groupby
    groups = groupby(results, key=lambda r: group_name(r["sample_id"]))
    all_rows = []
    for gname, group_iter in groups:
        group_rows = list(group_iter)
        all_rows.extend(group_rows)
        for r in group_rows:
            print(
                f"{r['sample_id']:<{col_w}}"
                f"{r['nframes']:>7}  {r['n_gt_segs']:>7}  {r['n_pred_segs']:>9}  "
                f"{r['frame_semantic_sim']:>7.4f}  "
                f"{r['boundary_precision']:>6.4f}  "
                f"{r['boundary_recall']:>6.4f}  "
                f"{r['boundary_f1']:>6.4f}"
            )
        if len(group_rows) > 1:
            label = f"  ↳ {gname} mean"
            print(thin)
            print(mean_row(label, group_rows, col_w))
        print(sep)

    if len(all_rows) > 1:
        print(mean_row("TOTAL MEAN", all_rows, col_w))
        print(sep)


# ─── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pred_files", nargs="*", help="Prediction segments.json file(s)")
    parser.add_argument("--samples_dir", default=DEFAULT_SAMPLE_DIR, help="Directory whose subdirectories each contain a segments.json")
    parser.add_argument("--gt_base",  default=DEFAULT_GT_BASE, help="Root directory of GT dataset (default: %(default)s)")
    parser.add_argument("--tau",      type=int, default=DEFAULT_TAU, help="Boundary tolerance in frames (default: %(default)s)")
    parser.add_argument("--model",    default=DEFAULT_MODEL, help="Sentence-transformers model name (default: %(default)s)")
    args = parser.parse_args()

    # Collect prediction paths
    pred_paths: list[Path] = []
    if args.samples_dir:
        samples_dir = Path(args.samples_dir)
        pred_paths += sorted(samples_dir.glob("*/segments.json"))
    for p in args.pred_files:
        pred_paths.append(Path(p))

    if not pred_paths:
        parser.print_help()
        sys.exit(1)

    print(f"Loading model '{args.model}' ...", flush=True)
    model = SentenceTransformer(args.model)

    results = []
    for pred_path in pred_paths:
        print(f"Evaluating: {pred_path}", flush=True)
        result = evaluate_one(pred_path, args.gt_base, model, args.tau)
        if result:
            results.append(result)

    print()
    print_results(results, args.tau)


if __name__ == "__main__":
    main()
