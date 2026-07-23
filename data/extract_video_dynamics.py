"""
Phase 1: High-Fidelity Perception, Time-Series Extraction & Visualization
Project: CausClass (Causal Discovery in Classroom Behaviors)
Description: Extracts micro-events using YOLOv8 + ByteTrack, aggregates them 
             into a macro-state matrix, and renders an annotated demonstration video.
"""

import argparse
import json
import cv2
import pandas as pd
from collections import defaultdict
from pathlib import Path
from ultralytics import YOLO
import torch
import torch.nn as nn
from utils.paths import OUTPUT_DIR

try:
    from ultralytics.nn import modules, tasks
except Exception:
    modules = None
    tasks = None

class MHSA(nn.Module):
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


def patch_ultralytics_mhsa():
    """Register the custom MHSA layer before loading YOLO weights."""
    if modules is None or tasks is None:
        return
    setattr(modules, "MHSA", MHSA)
    setattr(tasks, "MHSA", MHSA)
    modules.__dict__["MHSA"] = MHSA
    tasks.__dict__["MHSA"] = MHSA


def process_video(video_path, model_path="yolov8-mhsa.pt", patience_sec=2.0, bin_size_sec=4.0, save_demo=True, output_dir=None):
    print("[INFO] ========================================================")
    print("[INFO] INITIALIZING PHASE 1: PERCEPTION & TRACKING PIPELINE")
    print("[INFO] ========================================================")
    
    # Initialize YOLOv8 Model
    try:
        patch_ultralytics_mhsa()
        model = YOLO(model_path)
        print(f"[INFO] Successfully loaded model weights: {model_path}")
    except FileNotFoundError:
        print(f"[WARNING] Weights {model_path} not found. Fallback to yolov8n.pt for testing.")
        model = YOLO("yolov8n.pt")

    # Initialize Video Capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"[ERROR] Unable to read video source: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_duration = total_frames / fps
    patience_frames = int(patience_sec * fps)

    # Initialize Video Writer for Demonstration Output
    out_vid = None
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR / "phase1" / Path(video_path).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    file_prefix = str(out_dir / Path(video_path).stem)
    if save_demo:
        demo_file = f"{file_prefix}_demo.mp4"
        # Using mp4v codec for broad compatibility across different OS/environments
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_vid = cv2.VideoWriter(demo_file, fourcc, fps, (width, height))
        print(f"[INFO] Rendering Engine Enabled. Output path: {demo_file}")

    # Active tracks dictionary
    # Structure: {track_id: {'class_id': int, 'class_name': str, 'start_frame': int, 'last_seen_frame': int}}
    active_tracks = {}

    # List to store completed micro-events
    finished_events = []

    print(f"[INFO] Target Video: {video_path}")
    print(f"[INFO] Video Metadata - Res: {width}x{height} | FPS: {fps:.2f} | Duration: {video_duration:.2f}s")
    print(f"[INFO] Tracking Configuration - Patience Window: {patience_sec}s ({patience_frames} frames)")
    print("[INFO] Executing Neural Network Inference and Object Tracking...")

    # Execute YOLOv8 with ByteTrack
    # Note: save=False is intentional. We handle the rendering manually via cv2.VideoWriter.
    results = model.track(
        source=video_path,
        tracker="bytetrack.yaml",
        save=False,      
        save_conf=False,
        stream=True,
        verbose=False,
        iou=0.4,
        conf=0.1,
    )

    frame_idx = 0
    for r in results:
        frame_idx += 1
        current_active_ids = set()

        # Render annotated frame if demo generation is enabled
        if save_demo and out_vid is not None:
            annotated_frame = r.plot()
            out_vid.write(annotated_frame)

        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            track_ids = r.boxes.id.int().cpu().numpy()
            class_ids = r.boxes.cls.int().cpu().numpy()

            for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                track_id = int(track_id)
                cls_id = int(cls_id)
                cls_name = model.names[cls_id]
                current_active_ids.add(track_id)

                if track_id not in active_tracks:
                    # Initialize a new behavioral event
                    active_tracks[track_id] = {
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "start_frame": frame_idx,
                        "last_seen_frame": frame_idx
                    }
                else:
                    if active_tracks[track_id]["class_id"] == cls_id:
                        # Update the last seen timestamp for the ongoing event
                        active_tracks[track_id]["last_seen_frame"] = frame_idx
                    else:
                        # Class transition detected. Terminate previous event and initiate a new one.
                        old_event = active_tracks[track_id]
                        finished_events.append({
                            "object_id": track_id,
                            "action": old_event["class_name"],
                            "start_time": round(old_event["start_frame"] / fps, 2),
                            "end_time": round(old_event["last_seen_frame"] / fps, 2)
                        })
                        active_tracks[track_id] = {
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "start_frame": frame_idx,
                            "last_seen_frame": frame_idx
                        }

        # Patience Mechanism: Purge tracks that have been lost longer than the defined threshold
        lost_ids = []
        for tid, data in active_tracks.items():
            if tid not in current_active_ids:
                if (frame_idx - data["last_seen_frame"]) > patience_frames:
                    finished_events.append({
                        "object_id": tid,
                        "action": data["class_name"],
                        "start_time": round(data["start_frame"] / fps, 2),
                        "end_time": round(data["last_seen_frame"] / fps, 2)
                    })
                    lost_ids.append(tid)

        for tid in lost_ids:
            del active_tracks[tid]

        if frame_idx % int(fps * 30) == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"[PROCESS] Inference Progress: {frame_idx}/{total_frames} frames ({progress:.1f}%)")

    # Terminate any remaining active events upon video completion
    for tid, data in active_tracks.items():
        finished_events.append({
            "object_id": tid,
            "action": data["class_name"],
            "start_time": round(data["start_frame"] / fps, 2),
            "end_time": round(data["last_seen_frame"] / fps, 2)
        })

    cap.release()
    if out_vid is not None:
        out_vid.release()
        
    print("[INFO] Tracking and rendering phases completed. Commencing data aggregation...")

    # =========================================================================
    # MACRO-STATE AGGREGATION: GENERATING MULTIVARIATE TIME-SERIES MATRIX
    # =========================================================================
    
    # [CRITICAL FIX] Mapping raw YOLO classes to clean Variable Names for Phase 2
    CLASS_MAPPING = {
        "hand-raising": "Hand",
        "reading": "Read",
        "writing": "Write",
        "using phone": "Phone",
        "bowing the head": "Bow",
        "leaning over the table": "Lean"
    }
    
    TARGET_ACTIONS = list(CLASS_MAPPING.values())
    
    max_seconds = int(video_duration) + 1
    macro_states = []

    print(f"[INFO] Aggregating micro-events into {bin_size_sec}s temporal bins...")

    # Sliding window logic for temporal binning
    for start_sec in range(0, max_seconds, int(bin_size_sec)):
        end_sec = start_sec + int(bin_size_sec)
        bin_counts = defaultdict(int)

        # Check for event overlap within the current temporal window [start_sec, end_sec)
        for event in finished_events:
            if not (event["end_time"] <= start_sec or event["start_time"] >= end_sec):
                # Normalize string to prevent key mismatch
                raw_action = str(event["action"]).strip().lower()
                
                # Map to target variable space
                if raw_action in CLASS_MAPPING:
                    mapped_action = CLASS_MAPPING[raw_action]
                    bin_counts[mapped_action] += 1

        total_detected = sum(bin_counts.values())
        
        # Initialize the state vector for the current temporal bin
        state_dict = {"time_bin_sec": start_sec}

        # Compute percentage distribution to normalize population variance
        for action in TARGET_ACTIONS:
            if total_detected > 0:
                percentage = (bin_counts.get(action, 0) / total_detected) * 100
                state_dict[action] = round(percentage, 2)
            else:
                state_dict[action] = 0.0

        macro_states.append(state_dict)

    # =========================================================================
    # DATA EXPORT
    # =========================================================================

    # 1. Export Micro-events (JSON)
    micro_file = f"{file_prefix}_micro_events.json"
    with open(micro_file, 'w', encoding='utf-8') as f:
        json.dump(finished_events, f, indent=4)

    # 2. Export Macro-states (CSV)
    df = pd.DataFrame(macro_states)
    csv_file = f"{file_prefix}_multivariate_timeseries.csv"
    df.to_csv(csv_file, index=False)

    print("\n[SUCCESS] ========================================================")
    print(f"[SUCCESS] 1. Micro-events exported to: {micro_file}")
    print(f"[SUCCESS] 2. Multivariate Time-Series (CSV) exported to: {csv_file}")
    if save_demo:
        print(f"[SUCCESS] 3. Demonstration Video exported to: {demo_file}")
    print(f"[SUCCESS] Temporal Resolution: {bin_size_sec} seconds/bin.")
    print(f"[SUCCESS] Matrix Dimensions: {df.shape[0]} rows x {df.shape[1]} columns.")
    print("[SUCCESS] Data is fully prepared for Phase 2 (Causal Discovery).")
    print("[SUCCESS] ========================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Classroom Behavior Extraction, Tracking & Video Rendering")
    parser.add_argument("video", help="Absolute or relative path to the input video file")
    parser.add_argument("--model", default="yolov8-mhsa.pt", help="Path to the custom YOLOv8 weights")
    parser.add_argument("--patience", type=float, default=2.0, help="Tolerance window (seconds) to prevent ID switching")
    parser.add_argument("--bin_size", type=float, default=4.0, help="Temporal resolution (seconds) for the macro-state matrix")
    parser.add_argument("--no_demo", action="store_true", help="Disable demonstration video rendering to save processing time")
    parser.add_argument("--output-dir", default=None, help="Directory for generated Phase 1 artifacts")
    args = parser.parse_args()

    # save_demo is True by default, negated by --no_demo flag
    process_video(args.video, args.model, args.patience, args.bin_size, not args.no_demo, args.output_dir)
