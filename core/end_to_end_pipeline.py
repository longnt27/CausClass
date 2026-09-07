#!/usr/bin/env python3
"""
End-to-end CausClass pipeline v5-fast-phase1:
video -> YOLOv8 + ByteTrack behavior time series -> LLM-guided AERCA graph search
-> observation report.

Phase 1 uses batched YOLO inference by default for faster extraction.
Final report uses deepseek-v4-pro by default and is structured as evidence-first output:
what happened, what changed over time, graph edges, interpretations, inspection checklist,
and conditional classroom experiments.

Drop this file into the project root next to:
    extract_video_dynamics.py, run_synthetic_search.py, aerca_verifier.py, helper.py, graph_edit_agent.py

Example:
    python -m core.end_to_end_pipeline \
        --video data/classroom.mp4 \
        --model data/best.pt \
        --output_dir output/runs/classroom_001

Required env:
    DEEPSEEK_API_KEY=...
"""

import argparse
import copy
import datetime as _dt
import json
import os
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Reduce third-party library console output.
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("TQDM_DISABLE", "1")

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from ultralytics import YOLO

try:
    from ultralytics.nn import modules, tasks
except Exception:
    modules = None
    tasks = None

from core.aerca_verifier import get_initial_graph_from_aerca, run_masked_aerca
from utils.helpers import apply_edit, calculate_bic, hash_graph, load_env_file
from core.graph_edit_agent import LLMGraphAgent
from utils.paths import OUTPUT_DIR


class MHSA(nn.Module):
    """Multi-Head Self-Attention module used by the custom YOLO26 topology."""

    def __init__(self, c1, c2, h=4):
        super().__init__()
        self.h = h
        self.dh = c2 // h
        self.qkv = nn.Linear(c1, c2 * 3)
        self.project = nn.Linear(c2, c2)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)
        qkv = self.qkv(x_flat).view(B, -1, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.dh ** -0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        out = self.project(out).permute(0, 2, 1).view(B, C, H, W)
        return out


def patch_ultralytics_mhsa() -> None:
    """Register MHSA so Ultralytics can deserialize YOLO26-MHSA weights/topologies."""
    if modules is None or tasks is None:
        raise RuntimeError(
            "Could not import ultralytics.nn.modules/tasks. "
            "Check your ultralytics installation inside this environment."
        )

    print("[INFO] Patching MHSA logic into core computation graph...")
    setattr(modules, "MHSA", MHSA)
    setattr(tasks, "MHSA", MHSA)
    # parse_model resolves layer names through module globals in some Ultralytics versions.
    modules.__dict__["MHSA"] = MHSA
    tasks.__dict__["MHSA"] = MHSA


def get_model_class_names(model: YOLO) -> List[str]:
    """Return class names in class-id order, preserving the training YAML names."""
    names = model.names
    if isinstance(names, dict):
        return [str(names[i]) for i in sorted(names.keys())]
    return [str(x) for x in list(names)]

DEFAULT_CONTEXT = """
Classroom behavior dynamics extracted from video. No lesson context is available.
The system only knows detected behavior labels and their time ordering.

Variables are percentages of detected behavior-events in each time bin, NOT exact
percentages of students:
- Read: students reading or focusing on visible material.
- Write: students writing notes or solving written work.
- Phone: visible phone use.
- Bow: head-down posture. This may indicate phone use, reading, writing, fatigue,
  or camera-angle effects. Do not interpret it as disengagement by default.
- Lean: leaning over the table. This may indicate close work, fatigue, posture, or
  camera-angle effects. Do not interpret it as disengagement by default.
- Talk: visible peer talking. This may be collaboration or off-task chatter; without
  lesson context, do not assume either.

Graph edges are directed temporal dependencies. They are observational signals,
not moral judgments and not guaranteed direct causality.
""".strip()


def _safe_stem(path: str) -> str:
    return Path(path).stem.replace(" ", "_")


def _ensure_dir(path: Union[str, Path]) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _event_to_jsonable(event: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "object_id": int(event["object_id"]),
        "action": str(event["action"]),
        "start_time": float(event["start_time"]),
        "end_time": float(event["end_time"]),
    }


def extract_dynamics_with_periodic_debug(
    video_path: str,
    model_path: str,
    output_dir: Union[str, Path],
    *,
    patience_sec: float = 2.0,
    bin_size_sec: float = 4.0,
    debug_every_sec: float = 30.0,
    conf: float = 0.1,
    iou: float = 0.4,
    yolo_batch: int = 30,
    device: Optional[str] = None,
    imgsz: int = 640,
    half: bool = True,
    vid_stride: int = 1,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], List[str], Dict[str, str]]:
    """
    Same extraction logic as data.extract_video_dynamics.process_video, except:
    - does not render a full demo video;
    - saves periodic annotated debug frames only;
    - returns the raw time-series DataFrame for the next phase.
    """
    output_dir = _ensure_dir(output_dir)
    debug_dir = _ensure_dir(output_dir / "debug_frames")

    print("[Phase 1] Loading YOLO model and opening video...")
    patch_ultralytics_mhsa()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    # FP16 is only enabled on CUDA-capable devices.
    use_half = bool(half and str(device).lower() != "cpu" and torch.cuda.is_available())

    try:
        model = YOLO(model_path)
    except FileNotFoundError:
        print(f"[WARN] Model not found: {model_path}. Falling back to yolov8n.pt.")
        model = YOLO("yolov8n.pt")

    try:
        model.fuse()
    except Exception:
        pass

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to read video source: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if fps <= 0:
        raise ValueError("Video FPS is invalid; OpenCV returned <= 0. The video may be corrupted.")

    video_duration = total_frames / fps if total_frames > 0 else 0.0
    patience_frames = int(patience_sec * fps)
    debug_every_frames = max(1, int(debug_every_sec * fps))

    print(
        f"[Phase 1] Video metadata: {width}x{height}, fps={fps:.2f}, "
        f"frames={total_frames}, duration={video_duration:.2f}s"
    )
    print(
        f"[Phase 1] YOLO acceleration: batch={yolo_batch}, device={device}, "
        f"imgsz={imgsz}, half={use_half}, vid_stride={vid_stride}"
    )

    active_tracks: Dict[int, Dict[str, Any]] = {}
    finished_events: List[Dict[str, Any]] = []

    # Ultralytics can batch detector inference while still emitting sequential tracking results.
    # ByteTrack remains sequential because tracking depends on frame order.
    results = model.track(
        source=video_path,
        tracker="bytetrack.yaml",
        save=False,
        save_conf=False,
        stream=True,
        verbose=False,
        iou=iou,
        conf=conf,
        batch=max(1, int(yolo_batch)),
        device=device,
        imgsz=imgsz,
        half=use_half,
        vid_stride=max(1, int(vid_stride)),
        persist=True,
    )

    frame_idx = 0
    for r in results:
        frame_idx += 1
        current_active_ids = set()

        should_debug = (
            frame_idx == 1
            or frame_idx % debug_every_frames == 0
            or (total_frames > 0 and frame_idx == total_frames)
        )
        if should_debug:
            debug_path = debug_dir / f"frame_{frame_idx:08d}.jpg"
            cv2.imwrite(str(debug_path), r.plot())

        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            track_ids = r.boxes.id.int().cpu().numpy()
            class_ids = r.boxes.cls.int().cpu().numpy()

            for _, track_id, cls_id in zip(boxes, track_ids, class_ids):
                track_id = int(track_id)
                cls_id = int(cls_id)
                cls_name = model.names[cls_id]
                current_active_ids.add(track_id)

                if track_id not in active_tracks:
                    active_tracks[track_id] = {
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "start_frame": frame_idx,
                        "last_seen_frame": frame_idx,
                    }
                else:
                    if active_tracks[track_id]["class_id"] == cls_id:
                        active_tracks[track_id]["last_seen_frame"] = frame_idx
                    else:
                        old_event = active_tracks[track_id]
                        finished_events.append(
                            {
                                "object_id": track_id,
                                "action": old_event["class_name"],
                                "start_time": round(old_event["start_frame"] / fps, 2),
                                "end_time": round(old_event["last_seen_frame"] / fps, 2),
                            }
                        )
                        active_tracks[track_id] = {
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "start_frame": frame_idx,
                            "last_seen_frame": frame_idx,
                        }

        lost_ids = []
        for tid, data in active_tracks.items():
            if tid not in current_active_ids and (frame_idx - data["last_seen_frame"]) > patience_frames:
                finished_events.append(
                    {
                        "object_id": tid,
                        "action": data["class_name"],
                        "start_time": round(data["start_frame"] / fps, 2),
                        "end_time": round(data["last_seen_frame"] / fps, 2),
                    }
                )
                lost_ids.append(tid)

        for tid in lost_ids:
            del active_tracks[tid]

        if total_frames > 0 and frame_idx % max(1, int(fps * 30)) == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"[Phase 1] Inference progress: {frame_idx}/{total_frames} frames ({progress:.1f}%)")

    for tid, data in active_tracks.items():
        finished_events.append(
            {
                "object_id": tid,
                "action": data["class_name"],
                "start_time": round(data["start_frame"] / fps, 2),
                "end_time": round(data["last_seen_frame"] / fps, 2),
            }
        )

    # Your trained data YAML already uses clean labels:
    # ['Read', 'Write', 'Phone', 'Bow', 'Lean', 'Talk']
    # So we preserve model.names directly. No label remapping.
    target_actions = get_model_class_names(model)
    max_seconds = int(video_duration) + 1
    macro_states: List[Dict[str, Any]] = []

    print(f"[Phase 1] Aggregating micro-events into {bin_size_sec}s bins...")
    step = max(1, int(bin_size_sec))
    for start_sec in range(0, max_seconds, step):
        end_sec = start_sec + step
        bin_counts = defaultdict(int)

        for event in finished_events:
            if not (event["end_time"] <= start_sec or event["start_time"] >= end_sec):
                raw_action = str(event["action"]).strip()
                if raw_action in target_actions:
                    bin_counts[raw_action] += 1

        total_detected = sum(bin_counts.values())
        state = {"time_bin_sec": start_sec}

        for action in target_actions:
            state[action] = (
                round((bin_counts.get(action, 0) / total_detected) * 100, 2)
                if total_detected > 0
                else 0.0
            )

        macro_states.append(state)

    df = pd.DataFrame(macro_states)
    stem = _safe_stem(video_path)

    micro_path = output_dir / f"{stem}_micro_events.json"
    raw_ts_path = output_dir / f"{stem}_raw_timeseries.csv"
    metadata_path = output_dir / f"{stem}_video_metadata.json"

    with micro_path.open("w", encoding="utf-8") as f:
        json.dump([_event_to_jsonable(e) for e in finished_events], f, indent=2, ensure_ascii=False)

    df.to_csv(raw_ts_path, index=False)

    metadata = {
        "video_path": video_path,
        "model_path": model_path,
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_sec": video_duration,
        "patience_sec": patience_sec,
        "bin_size_sec": bin_size_sec,
        "debug_every_sec": debug_every_sec,
        "yolo_batch": int(yolo_batch),
        "device": device,
        "imgsz": int(imgsz),
        "half": bool(use_half),
        "vid_stride": int(vid_stride),
        "class_names": target_actions,
        "label_remapping": None,
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return df, [dict(e) for e in finished_events], target_actions, {
        "micro_events": str(micro_path),
        "raw_timeseries": str(raw_ts_path),
        "metadata": str(metadata_path),
        "debug_frames": str(debug_dir),
    }


def make_aerca_chunks(
    df: pd.DataFrame,
    variables: Sequence[str],
    *,
    chunk_size: int = 500,
    scale_percentages: bool = True,
) -> np.ndarray:
    """
    Mirrors run_synthetic_search.py: data_array = df.values / 100.0; then chunk into
    fixed windows of 500 by default.

    Difference: this accepts real videos. If the video is short and cannot make
    a full 500-row chunk, it uses one shorter chunk instead of silently producing
    zero chunks, because zero chunks is the kind of bug that waits in the bushes.
    """
    missing = [v for v in variables if v not in df.columns]
    if missing:
        raise ValueError(f"Time-series is missing behavior columns: {missing}")

    values = df[list(variables)].values.astype(np.float32)
    if scale_percentages:
        values = values / 100.0

    if len(values) < 3:
        raise ValueError(
            "Not enough time bins for causal discovery. "
            "Use a longer video or a smaller --bin_size."
        )

    full_chunks = len(values) // chunk_size
    if full_chunks > 0:
        return np.array(
            [values[i * chunk_size : (i + 1) * chunk_size] for i in range(full_chunks)],
            dtype=np.float32,
        )

    print(
        f"[WARN] Only {len(values)} time bins available, less than chunk_size={chunk_size}. "
        "Using one short chunk."
    )
    return np.array([values], dtype=np.float32)


def run_llm_guided_aerca_real_video(
    xs: np.ndarray,
    variables: Sequence[str],
    api_key: str,
    output_dir: Union[str, Path],
    *,
    baseline_threshold: float = 0.4,
    bic_penalty_lambda: float = 20.0,
    beam_width: int = 5,
    max_iterations: int = 30,
    max_age: int = 4,
    min_bic_delta: float = -0.15,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Real-video variant of run_synthetic_search.process_test_suite:
    - no ground-truth file;
    - no precision/recall/F1 because there is no ground truth for real videos;
    - keeps AERCA baseline, LLM proposals, tabu, beam search, BIC verification.
    """
    output_dir = _ensure_dir(output_dir)
    variables = list(variables)
    num_samples = int(sum(chunk.shape[0] for chunk in xs))

    log: Dict[str, Any] = {
        "metadata": {
            "timestamp": _dt.datetime.now().isoformat(),
            "variables": variables,
            "num_chunks": int(len(xs)),
            "num_samples": num_samples,
            "hyperparameters": {
                "baseline_threshold": baseline_threshold,
                "bic_penalty_lambda": bic_penalty_lambda,
                "beam_width": beam_width,
                "max_iterations": max_iterations,
                "max_age": max_age,
                "min_bic_delta": min_bic_delta,
            },
        },
        "baseline": {},
        "iterations": [],
        "final": {},
    }

    print("[Phase 2] Running AERCA baseline discovery...")
    auto_initial_graph, dense_weights, raw_discovery_matrix = get_initial_graph_from_aerca(
        xs, variables, default_threshold=baseline_threshold
    )

    agent = LLMGraphAgent(api_key, variables, DEFAULT_CONTEXT)

    base_mse, base_num_edges, base_weights, base_edges = run_masked_aerca(
        xs, auto_initial_graph, variables, init_weights=dense_weights
    )
    base_bic = calculate_bic(
        base_mse,
        base_num_edges,
        num_samples=num_samples,
        params_per_edge=bic_penalty_lambda,
    )

    log["baseline"] = {
        "bic": float(base_bic),
        "mse": float(base_mse),
        "edge_count": int(len(base_edges)),
        "graph": base_edges,
    }

    global_tabu = {hash_graph(base_edges)}
    beam = [
        {
            "graph": base_edges,
            "mse": base_mse,
            "bic": base_bic,
            "weights": base_weights,
            "age": 0,
            "local_tabu": [],
        }
    ]

    print("[Phase 2] Starting LLM-guided beam search...")
    for iteration in range(1, max_iterations + 1):
        iter_log = {"iteration": iteration, "proposals_evaluated": [], "beam_survivors": []}
        candidates = []

        for tree in beam:
            tree["has_valid_offspring"] = False
            status = "OPTIMIZING" if tree["age"] == 0 else f"STUCK (Age: {tree['age']})"
            proposed_edits = agent.propose_edits(
                tree["graph"],
                tree["local_tabu"],
                status,
                raw_discovery_matrix,
            )

            for edit in proposed_edits:
                action = edit.get("action")
                src = edit.get("source")
                tgt = edit.get("target")
                reasoning = edit.get("reasoning", "N/A")

                proposal_record = {
                    "parent_hash": hash_graph(tree["graph"]),
                    "action": action,
                    "source": src,
                    "target": tgt,
                    "reasoning": reasoning,
                    "status": "",
                    "metrics": None,
                    "improvement": 0.0,
                }

                if action not in {"add", "delete"}:
                    proposal_record["status"] = "INVALID_ACTION"
                    iter_log["proposals_evaluated"].append(proposal_record)
                    continue

                if src not in variables or tgt not in variables or src == tgt:
                    proposal_record["status"] = "INVALID_NODES"
                    iter_log["proposals_evaluated"].append(proposal_record)
                    continue

                if action == "delete":
                    i = variables.index(src)
                    j = variables.index(tgt)
                    original_signal = abs(raw_discovery_matrix[i, j])
                    if original_signal > 0.75:
                        proposal_record["status"] = "BLOCKED_BY_GUARDRAIL"
                        iter_log["proposals_evaluated"].append(proposal_record)
                        continue

                is_forward = any(e["source"] == src and e["target"] == tgt for e in tree["graph"])
                is_backward = any(e["source"] == tgt and e["target"] == src for e in tree["graph"])

                if action == "add" and (is_forward or is_backward):
                    proposal_record["status"] = "INVALID_TOPOLOGY_ALREADY_EXISTS"
                    iter_log["proposals_evaluated"].append(proposal_record)
                    continue

                if action == "delete" and not is_forward:
                    proposal_record["status"] = "INVALID_TOPOLOGY_DOES_NOT_EXIST"
                    iter_log["proposals_evaluated"].append(proposal_record)
                    continue

                new_graph = apply_edit(tree["graph"], edit)
                h_graph = hash_graph(new_graph)

                if h_graph in global_tabu:
                    proposal_record["status"] = "SKIPPED_GLOBAL_TABU"
                    iter_log["proposals_evaluated"].append(proposal_record)
                    continue

                global_tabu.add(h_graph)
                proposal_record["status"] = "PENDING_EVALUATION"
                candidates.append(
                    {
                        "graph": new_graph,
                        "parent": tree,
                        "edit": edit,
                        "log_ref": proposal_record,
                    }
                )
                iter_log["proposals_evaluated"].append(proposal_record)

        if not candidates:
            for tree in beam:
                tree["age"] += 1
        else:
            next_generation = []
            for c in candidates:
                c_mse, c_num_edges, c_weights, c_updated_graph = run_masked_aerca(
                    xs, c["graph"], variables, init_weights=dense_weights
                )
                c_bic = calculate_bic(
                    c_mse,
                    c_num_edges,
                    num_samples=num_samples,
                    params_per_edge=bic_penalty_lambda,
                )

                bic_improvement = c["parent"]["bic"] - c_bic
                threshold_delta = min_bic_delta if c["edit"]["action"] == "add" else 0.0

                c["log_ref"]["metrics"] = {
                    "bic": float(c_bic),
                    "mse": float(c_mse),
                    "edge_count": int(c_num_edges),
                }
                c["log_ref"]["improvement"] = float(bic_improvement)

                if bic_improvement >= threshold_delta:
                    c["log_ref"]["status"] = "ACCEPTED"
                    next_generation.append(
                        {
                            "graph": c_updated_graph,
                            "mse": c_mse,
                            "bic": c_bic,
                            "weights": c_weights,
                            "age": 0,
                            "local_tabu": copy.deepcopy(c["parent"]["local_tabu"]),
                        }
                    )
                    c["parent"]["has_valid_offspring"] = True
                else:
                    c["log_ref"]["status"] = "REJECTED"
                    e = c["edit"]
                    c["parent"]["local_tabu"].append(
                        f"FAILED {e['action'].upper()}: {e['source']}->{e['target']}"
                    )

            for tree in beam:
                if not tree.get("has_valid_offspring", False):
                    tree["age"] += 1

            combined = [t for t in (beam + next_generation) if t["age"] < max_age]
            if not combined:
                print("[Phase 2] Beam exhausted.")
                break

            combined.sort(key=lambda x: x["bic"])
            beam = combined[:beam_width]

        for survivor in beam:
            iter_log["beam_survivors"].append(
                {
                    "hash": hash_graph(survivor["graph"]),
                    "bic": float(survivor["bic"]),
                    "mse": float(survivor["mse"]),
                    "age": int(survivor["age"]),
                    "edge_count": int(len(survivor["graph"])),
                }
            )

        log["iterations"].append(iter_log)
        best = beam[0]
        print(
            f"[Phase 2] Iter {iteration:02d}: best BIC={best['bic']:.4f}, "
            f"MSE={best['mse']:.6f}, edges={len(best['graph'])}, beam={len(beam)}"
        )

    best_final = beam[0]
    final_graph = best_final["graph"]
    log["final"] = {
        "bic": float(best_final["bic"]),
        "mse": float(best_final["mse"]),
        "edge_count": int(len(final_graph)),
        "graph": final_graph,
    }

    return final_graph, log


def render_graph_edges_exact(graph: Sequence[Dict[str, Any]]) -> str:
    """Render graph edges from Python, not the LLM. Evidence should not be vibes-based."""
    lines = ["# 3. What graph edges were found?"]
    if not graph:
        lines.append("- No cross-behavior edges survived verification; AERCA may still rely on self-history internally.")
        return "\n".join(lines)

    def _weight(edge: Dict[str, Any]) -> float:
        try:
            return float(edge.get("weight", 0.0))
        except (TypeError, ValueError):
            return 0.0

    for e in sorted(graph, key=lambda x: abs(_weight(x)), reverse=True):
        src = str(e.get("source", "?"))
        tgt = str(e.get("target", "?"))
        w = _weight(e)
        if w > 0:
            sign = "positive"
            meaning = "increase/follow together"
        elif w < 0:
            sign = "negative"
            meaning = "decrease/trade-off"
        else:
            sign = "near-zero"
            meaning = "unclear"

        aw = abs(w)
        if aw < 0.03:
            strength = "very weak"
        elif aw < 0.10:
            strength = "weak"
        elif aw < 0.20:
            strength = "moderate"
        else:
            strength = "strong"

        lines.append(
            f"- {src} → {tgt}: {sign} {strength} temporal association "
            f"(weight {w:+.4f}; {meaning})."
        )
    return "\n".join(lines)


def call_llm_for_teacher_advice(
    api_key: str,
    graph: Sequence[Dict[str, Any]],
    variables: Sequence[str],
    time_series_summary: Dict[str, Any],
    model: str = "deepseek-v4-pro",
    report_min_abs_weight: float = 0.0,
    report_max_edges: int = 999,
    teacher_max_tokens: int = 6000,
) -> str:
    """
    Final LLM pass: produce a compact evidence-first observation report.

    Critical fix:
    - The full graph is passed to the model, not a filtered subset.
    - Section 3 is rendered deterministically by Python so every edge is shown.
    - No arbitrary "first 20 bins / first 80 seconds" summary is injected.
    """
    edges_for_prompt = []
    for e in graph:
        weight = e.get("weight")
        try:
            w = None if weight is None else float(weight)
        except (TypeError, ValueError):
            w = None

        edges_for_prompt.append(
            {
                "source": e.get("source"),
                "target": e.get("target"),
                "weight": None if w is None else round(w, 6),
                "sign": (
                    "+" if w is not None and w > 0
                    else "-" if w is not None and w < 0
                    else "unknown"
                ),
                "plain_meaning": (
                    "increase/follow together" if w is not None and w > 0
                    else "decrease/trade-off" if w is not None and w < 0
                    else "unknown"
                ),
            }
        )

    system_prompt = """
You are a classroom observation analyst. Produce a compact evidence-first report from behavior-detection data.

Hard limits:
- No lesson context is available.
- Behavior shares are detected-event shares per time bin, not student percentages.
- Graph edges are temporal associations, not proof of causality.
- Respect the graph exactly. Do not ignore, rename, reverse, merge, or invent graph edges.
- Graph direction matters: A -> B means A at an earlier step is associated with B later. Do not infer B -> A unless that edge exists.
- Use every final graph edge when forming interpretations; if an edge is weak, explicitly call it weak instead of omitting it.
- Do not mention any time window, timestamp, or phase boundary unless it is explicitly present in time_series_summary.
- Do not claim an arbitrary early slice is a meaningful lesson phase.
- Do not judge students or claim off-task behavior.
- Do not recommend punishment, phone bans, phone stacks, phone parking, shaming, or policy changes.
- Keep teacher-facing wording simple; this report is for classroom review, not a research paper.
- Section 1 must not contain graph-derived claims; save all graph reasoning for possible_interpretations.
- No markdown. Return JSON only.

Return exactly one valid JSON object with these keys:
{
  "what_happened": [2 to 4 short bullets containing ONLY descriptive detection facts: frequent/rare behaviors, active bins, mean shares; DO NOT mention graph edges, causality, associations, trade-offs, or interpretations],
  "changed_over_time": [2 to 4 short bullets based ONLY on trend_by_behavior and temporal_segments; explicitly treat early/middle/late as equal video thirds, not lesson phases],
  "possible_interpretations": [2 to 3 bullets, each with evidence + alternate explanation; use the graph here, respect direction, and mention weak edges as weak],
  "human_inspection": [3 to 5 specific debug-frame checks],
  "conditional_experiments": [2 to 4 teacher-friendly bullets, each formatted: "If human inspection confirms ..., try ..., improvement would look like ..."]
}

Section rules:
- Section 1 / what_happened must be pure observation. Do not mention Phone -> Read, Talk -> Read, negative edge, positive edge, association, trade-off, or causal language there.
- Section 2 / changed_over_time may compare equal video thirds and trend directions, but must not imply those thirds are lesson phases.
- Section 3 will be rendered exactly by the program from the verified graph. Do NOT include a graph_edges_found key.
- Section 4 / possible_interpretations may interpret graph edges, but must not reverse edge direction. A -> B does not mean B -> A.
- Section 6 / conditional_experiments must sound like a practical teacher/reviewer checklist, not a research methods note. Avoid jargon such as "condition on", "association model", "recalculate edges", "causal model", "regression", "confounder", "latent variable", or "time lag" unless absolutely necessary.
- Conditional experiments must be framed as tests after human verification, not as direct teaching advice.
- Each bullet must be one sentence. Keep the whole report readable, but do not omit required sections just to be short.
""".strip()

    user_prompt = {
        "variables": list(variables),
        "time_series_summary": time_series_summary,
        "final_graph_edge_count": len(graph),
        "final_graph_edges": edges_for_prompt,
        "instruction": (
            "Base the report only on these summaries and graph edges. "
            "The exact graph edge list will be rendered by code, but your interpretations must respect all edges."
        ),
    }

    payload = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": teacher_max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ],
    }

    res = requests.post(
        "https://api.deepseek.com/chat/completions",
        json=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=120,
    )
    res.raise_for_status()
    response_data = res.json()
    choice = response_data["choices"][0]
    finish_reason = choice.get("finish_reason", "unknown")
    content = choice["message"]["content"].strip()

    def _as_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    try:
        parsed = json.loads(content)
        sections = [
            ("1. What happened?", _as_list(parsed.get("what_happened"))),
            ("2. What changed over time?", _as_list(parsed.get("changed_over_time"))),
            (None, None),  # Exact graph section inserted here.
            ("4. Possible interpretations", _as_list(parsed.get("possible_interpretations"))),
            ("5. What should a human inspect?", _as_list(parsed.get("human_inspection"))),
            ("6. Conditional classroom experiments", _as_list(parsed.get("conditional_experiments"))),
        ]

        lines = []
        if finish_reason == "length":
            lines.append("[WARNING] The LLM response hit the token limit; report may be incomplete. Increase --teacher_max_tokens.\n")

        for title, bullets in sections:
            if title is None:
                lines.append(render_graph_edges_exact(graph))
                lines.append("")
                continue
            lines.append(f"# {title}")
            if bullets:
                for b in bullets:
                    lines.append(f"- {b}")
            else:
                lines.append("- No usable item returned by the report model.")
            lines.append("")

        return "\n".join(lines).strip()

    except Exception:
        warning = ""
        if finish_reason == "length":
            warning = "[WARNING] The LLM response hit the token limit and could not be parsed as JSON.\n\n"
        else:
            warning = "[WARNING] The LLM response could not be parsed as JSON; showing raw content.\n\n"
        return warning + content


def summarize_time_series(df: pd.DataFrame, variables: Sequence[str]) -> Dict[str, Any]:
    """Summarize detected behavior shares without pretending they are student percentages."""
    behavior_df = df[list(variables)].astype(float)

    per_behavior = {}
    for col in variables:
        values = behavior_df[col]
        per_behavior[col] = {
            "mean_detected_behavior_share_pct": round(float(values.mean()), 3),
            "max_detected_behavior_share_pct": round(float(values.max()), 3),
            "std_detected_behavior_share_pct": round(float(values.std()), 3),
            "active_bins_pct": round(float((values > 0).mean() * 100), 3),
        }

    ranked = sorted(
        per_behavior.items(),
        key=lambda item: item[1]["mean_detected_behavior_share_pct"],
        reverse=True,
    )

    # Temporal summary uses equal early/middle/late thirds rather than arbitrary fixed slices.
    temporal_segments = []
    n = len(df)
    if n > 0:
        cuts = [0, n // 3, (2 * n) // 3, n]
        labels = ["early_third", "middle_third", "late_third"]
        for label, start, end in zip(labels, cuts[:-1], cuts[1:]):
            if end <= start:
                continue
            segment = df.iloc[start:end]
            means = {
                v: round(float(segment[v].astype(float).mean()), 3)
                for v in variables
            }
            dominant = max(means, key=means.get)
            if "time_bin_sec" in df.columns:
                start_sec = float(segment["time_bin_sec"].iloc[0])
                end_sec = float(segment["time_bin_sec"].iloc[-1])
            else:
                start_sec = None
                end_sec = None
            temporal_segments.append(
                {
                    "segment": label,
                    "bin_start_index": int(start),
                    "bin_end_index_exclusive": int(end),
                    "time_start_sec": start_sec,
                    "time_end_sec": end_sec,
                    "mean_detected_behavior_share_pct": means,
                    "dominant_behavior_by_mean": dominant,
                }
            )

    trend_by_behavior = {}
    if len(df) >= 2:
        x = np.arange(len(df), dtype=float)
        for v in variables:
            y = df[v].astype(float).to_numpy()
            slope = float(np.polyfit(x, y, 1)[0])
            trend_by_behavior[v] = {
                "linear_slope_share_points_per_bin": round(slope, 6),
                "rough_direction": "increasing" if slope > 0.01 else "decreasing" if slope < -0.01 else "mostly_flat",
            }

    return {
        "important_measurement_warning": (
            "Values are percentages of detected behavior-events within each time bin, "
            "not exact percentages of all students in the classroom."
        ),
        "num_bins": int(len(df)),
        "per_behavior": per_behavior,
        "behaviors_ranked_by_mean_detected_share": [
            {"behavior": name, **stats} for name, stats in ranked
        ],
        "temporal_segments": temporal_segments,
        "trend_by_behavior": trend_by_behavior,
        "temporal_summary_note": (
            "Temporal segments are equal thirds of the observed video, not lesson phases. "
            "Do not infer a pedagogical phase boundary from these segments alone."
        ),
    }


def edge_sign(weight: float) -> str:
    if weight > 0:
        return "+"
    if weight < 0:
        return "-"
    return "0"


def format_graph_snapshot(graph: Sequence[Dict[str, Any]], max_edges: Optional[int] = None) -> str:
    if not graph:
        return "No cross-behavior edges survived. Self-history may still be used internally by AERCA."

    rows = []
    edges = []
    for e in graph:
        try:
            w = float(e.get("weight", 0.0))
        except (TypeError, ValueError):
            w = 0.0
        edges.append((abs(w), w, str(e.get("source")), str(e.get("target"))))
    edges.sort(reverse=True)
    if max_edges is not None:
        edges = edges[:max_edges]

    for rank, (_, w, src, tgt) in enumerate(edges, start=1):
        meaning = "increase/follow together" if w > 0 else "decrease/trade-off" if w < 0 else "unknown sign"
        rows.append(f"{rank:02d}. {src} -> {tgt} | sign={edge_sign(w)} | weight={w:+.4f} | {meaning}")
    return "\n".join(rows)


def format_data_snapshot(summary: Dict[str, Any], top_k: int = 6) -> str:
    ranked = summary.get("behaviors_ranked_by_mean_detected_share", [])[:top_k]
    lines = [f"Time bins: {summary.get('num_bins')}", "Behavior shares are detected-event shares, not student percentages."]
    for item in ranked:
        lines.append(
            f"- {item['behavior']}: mean={item['mean_detected_behavior_share_pct']:.1f}%, "
            f"active_bins={item['active_bins_pct']:.1f}%, max={item['max_detected_behavior_share_pct']:.1f}%"
        )
    return "\n".join(lines)


def save_graph_artifacts(
    graph: Sequence[Dict[str, Any]],
    variables: Sequence[str],
    output_dir: Union[str, Path],
    stem: str,
) -> Dict[str, str]:
    output_dir = _ensure_dir(output_dir)
    variables = list(variables)
    var2idx = {v: i for i, v in enumerate(variables)}

    adj = pd.DataFrame(0.0, index=variables, columns=variables)
    edge_rows = []
    for e in graph:
        src = str(e.get("source"))
        tgt = str(e.get("target"))
        try:
            w = float(e.get("weight", 0.0))
        except (TypeError, ValueError):
            w = 0.0
        if src in var2idx and tgt in var2idx:
            adj.loc[src, tgt] = w
        edge_rows.append({"source": src, "target": tgt, "weight": w, "sign": edge_sign(w)})

    edge_csv = output_dir / f"{stem}_final_graph_edges.csv"
    adj_csv = output_dir / f"{stem}_final_graph_adjacency.csv"
    txt_path = output_dir / f"{stem}_final_graph_readable.txt"
    png_path = output_dir / f"{stem}_final_graph.png"

    pd.DataFrame(edge_rows, columns=["source", "target", "weight", "sign"]).sort_values(by="weight", key=lambda s: s.abs(), ascending=False).to_csv(edge_csv, index=False)
    adj.to_csv(adj_csv)
    txt_path.write_text(format_graph_snapshot(graph), encoding="utf-8")

    try:
        import math
        import matplotlib.pyplot as plt

        n = max(1, len(variables))
        angles = [2 * math.pi * i / n for i in range(n)]
        pos = {v: (math.cos(a), math.sin(a)) for v, a in zip(variables, angles)}

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_title("Final temporal graph")
        ax.axis("off")

        for v, (x, y) in pos.items():
            ax.scatter([x], [y], s=900)
            ax.text(x, y, v, ha="center", va="center", fontsize=10)

        for e in graph:
            src = str(e.get("source"))
            tgt = str(e.get("target"))
            if src not in pos or tgt not in pos:
                continue
            try:
                w = float(e.get("weight", 0.0))
            except (TypeError, ValueError):
                w = 0.0
            x1, y1 = pos[src]
            x2, y2 = pos[tgt]
            ax.annotate(
                "",
                xy=(x2 * 0.83, y2 * 0.83),
                xytext=(x1 * 0.83, y1 * 0.83),
                arrowprops=dict(arrowstyle="->", lw=max(0.8, min(3.0, abs(w) * 12))),
            )
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my, f"{w:+.2f}", fontsize=8)

        fig.tight_layout()
        fig.savefig(png_path, dpi=180)
        plt.close(fig)
    except Exception as exc:
        png_path.write_text(f"Graph image generation failed: {exc}", encoding="utf-8")

    return {
        "edge_list_csv": str(edge_csv),
        "adjacency_csv": str(adj_csv),
        "readable_txt": str(txt_path),
        "graph_png": str(png_path),
    }


def save_timeseries_artifacts(
    df: pd.DataFrame,
    variables: Sequence[str],
    output_dir: Union[str, Path],
    stem: str,
) -> Dict[str, str]:
    output_dir = _ensure_dir(output_dir)
    summary_path = output_dir / f"{stem}_data_summary.json"
    plot_path = output_dir / f"{stem}_timeseries_plot.png"

    summary = summarize_time_series(df, variables)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        x = df["time_bin_sec"] if "time_bin_sec" in df.columns else np.arange(len(df))
        fig, ax = plt.subplots(figsize=(12, 5))
        for v in variables:
            ax.plot(x, df[v].astype(float), label=v, linewidth=1.5)
        ax.set_title("Behavior time series")
        ax.set_xlabel("Time bin start (sec)")
        ax.set_ylabel("Detected behavior share (%)")
        ax.legend(loc="upper right", ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
    except Exception as exc:
        plot_path.write_text(f"Time-series plot generation failed: {exc}", encoding="utf-8")

    return {
        "data_summary_json": str(summary_path),
        "timeseries_plot_png": str(plot_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end CausClass: video -> dynamics -> LLM-guided AERCA -> teacher advice"
    )
    parser.add_argument("--video", required=True, help="Path to classroom video.")
    parser.add_argument("--model", required=True, help="Path to YOLO weights.")
    parser.add_argument("--output_dir", default=None, help="Directory for all outputs.")
    parser.add_argument("--patience", type=float, default=2.0, help="Tracking patience in seconds.")
    parser.add_argument("--bin_size", type=float, default=4.0, help="Time-series bin size in seconds.")
    parser.add_argument("--debug_every", type=float, default=30.0, help="Save one debug frame every N seconds.")
    parser.add_argument("--conf", type=float, default=0.1, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.4, help="YOLO IoU threshold.")
    parser.add_argument("--yolo_batch", type=int, default=30, help="YOLO inference batch size for Phase 1. Default 30 for fat-VRAM machines.")
    parser.add_argument("--device", default=None, help="YOLO device, e.g. 0, cuda:0, or cpu. Default: CUDA 0 if available.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--no_half", action="store_true", help="Disable FP16 inference in Phase 1.")
    parser.add_argument("--vid_stride", type=int, default=1, help="Process every Nth frame. Keep 1 for exact extraction; >1 is faster but changes the time series.")
    parser.add_argument("--chunk_size", type=int, default=500, help="Verifier chunk size. Matches run_synthetic_search default.")
    parser.add_argument("--baseline_threshold", type=float, default=0.4, help="AERCA initial graph quantile threshold.")
    parser.add_argument("--bic_penalty_lambda", type=float, default=20.0, help="BIC edge penalty. Default is tuned for shorter real videos, not 1000-step synthetic suites.")
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--max_iterations", type=int, default=30)
    parser.add_argument("--max_age", type=int, default=4)
    parser.add_argument("--min_bic_delta", type=float, default=-0.15)
    parser.add_argument("--skip_teacher_advice", action="store_true", help="Skip final LLM observation report.")
    parser.add_argument("--teacher_model", default="deepseek-v4-pro", help="DeepSeek model for final observation report.")
    parser.add_argument("--teacher_max_tokens", type=int, default=6000, help="Max output tokens for the final observation report. Default is generous because truncation is stupid.")
    parser.add_argument("--report_min_abs_weight", type=float, default=0.05, help="Minimum absolute edge weight shown to the final report model.")
    parser.add_argument("--report_max_edges", type=int, default=4, help="Maximum strongest edges shown to the final report model.")
    args = parser.parse_args()

    load_env_file()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("[CRITICAL] DEEPSEEK_API_KEY is missing from environment variables.")
        return 1

    stem = _safe_stem(args.video)
    out_dir = _ensure_dir(args.output_dir or OUTPUT_DIR / "runs" / f"{stem}_{_dt.datetime.now():%Y%m%d_%H%M%S}")

    try:
        df_raw, micro_events, variables, phase1_paths = extract_dynamics_with_periodic_debug(
            args.video,
            args.model,
            out_dir,
            patience_sec=args.patience,
            bin_size_sec=args.bin_size,
            debug_every_sec=args.debug_every,
            conf=args.conf,
            iou=args.iou,
            yolo_batch=args.yolo_batch,
            device=args.device,
            imgsz=args.imgsz,
            half=not args.no_half,
            vid_stride=args.vid_stride,
        )

        xs = make_aerca_chunks(df_raw, variables, chunk_size=args.chunk_size)

        processed_path = out_dir / f"{stem}_aerca_input.npy"
        np.save(processed_path, xs)

        final_graph, search_log = run_llm_guided_aerca_real_video(
            xs,
            variables,
            api_key,
            out_dir,
            baseline_threshold=args.baseline_threshold,
            bic_penalty_lambda=args.bic_penalty_lambda,
            beam_width=args.beam_width,
            max_iterations=args.max_iterations,
            max_age=args.max_age,
            min_bic_delta=args.min_bic_delta,
        )

        graph_path = out_dir / f"{stem}_final_graph.json"
        with graph_path.open("w", encoding="utf-8") as f:
            json.dump(final_graph, f, indent=2, ensure_ascii=False)

        data_summary = summarize_time_series(df_raw, variables)
        graph_artifacts = save_graph_artifacts(final_graph, variables, out_dir, stem)
        ts_artifacts = save_timeseries_artifacts(df_raw, variables, out_dir, stem)

        log_path = out_dir / f"{stem}_search_log.json"
        search_log["phase1_outputs"] = phase1_paths
        search_log["processed_aerca_input"] = str(processed_path)
        search_log["final_graph_path"] = str(graph_path)
        search_log["graph_artifacts"] = graph_artifacts
        search_log["timeseries_artifacts"] = ts_artifacts
        with log_path.open("w", encoding="utf-8") as f:
            json.dump(search_log, f, indent=2, ensure_ascii=False)

        print("\n" + "-" * 80)
        print("DATA SNAPSHOT")
        print("-" * 80)
        print(format_data_snapshot(data_summary))
        print("\n" + "-" * 80)
        print("FINAL GRAPH SNAPSHOT")
        print("-" * 80)
        print(format_graph_snapshot(final_graph))

        advice = ""
        advice_path: Optional[Path] = None
        if not args.skip_teacher_advice:
            print("[Phase 3] Asking LLM for evidence-first observation report...")
            advice = call_llm_for_teacher_advice(
                api_key,
                final_graph,
                variables,
                data_summary,
                model=args.teacher_model,
                report_min_abs_weight=args.report_min_abs_weight,
                report_max_edges=args.report_max_edges,
                teacher_max_tokens=args.teacher_max_tokens,
            )
            advice_path = out_dir / f"{stem}_observation_report.md"
            advice_path.write_text(advice, encoding="utf-8")

        print("\n" + "=" * 80)
        print("END-TO-END PIPELINE COMPLETE")
        print("=" * 80)
        print(f"Raw time series:        {phase1_paths['raw_timeseries']}")
        print(f"Processed AERCA input:  {processed_path}")
        print(f"Micro-events:           {phase1_paths['micro_events']}")
        print(f"Debug frames:           {phase1_paths['debug_frames']}")
        print(f"Final graph JSON:       {graph_path}")
        print(f"Graph edge CSV:         {graph_artifacts['edge_list_csv']}")
        print(f"Graph adjacency CSV:    {graph_artifacts['adjacency_csv']}")
        print(f"Graph image:            {graph_artifacts['graph_png']}")
        print(f"Data summary JSON:      {ts_artifacts['data_summary_json']}")
        print(f"Time-series plot:       {ts_artifacts['timeseries_plot_png']}")
        print(f"Search log:             {log_path}")
        if advice_path:
            print(f"Observation report:     {advice_path}")
            print("\n" + "-" * 80)
            print("OBSERVATION REPORT")
            print("-" * 80)
            print(advice)
        print("=" * 80 + "\n")
        return 0

    except Exception as exc:
        crash_path = out_dir / f"{stem}_crash_log.txt"
        crash_path.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"[FATAL] {exc}")
        print(f"[FATAL] Crash log written to: {crash_path}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
