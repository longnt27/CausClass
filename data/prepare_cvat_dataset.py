"""
Data Preparation Pipeline: Active Learning & Pseudo-Labeling
Project: CausClass (Causal Discovery in Classroom Behaviors)
Description: Extracts frames at a specific FPS rate and generates pseudo-labels 
             using the pre-trained model. Output is strictly formatted for CVAT YOLO 1.1 import.
"""

import os
import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch
import torch.nn as nn
import ultralytics.nn.modules as modules
from utils.paths import DATA_DIR

try:
    from ultralytics.nn import tasks
except Exception:
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
    setattr(modules, "MHSA", MHSA)
    modules.__dict__["MHSA"] = MHSA
    if tasks is not None:
        setattr(tasks, "MHSA", MHSA)
        tasks.__dict__["MHSA"] = MHSA


def prepare_cvat_dataset(video_path, model_path, output_dir, target_fps=0.5):
    print("[INFO] ========================================================")
    print("[INFO] INITIATING PSEUDO-LABELING & FRAME EXTRACTION PIPELINE")
    print("[INFO] ========================================================")

    try:
        patch_ultralytics_mhsa()
        model = YOLO(model_path)
    except Exception as e:
        print(f"[FATAL ERROR] Model initialization failed: {e}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"[ERROR] Cannot open video source: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the frame sampling interval
    frame_interval = int(round(source_fps / target_fps))
    
    print(f"[INFO] Video Details - FPS: {source_fps:.2f} | Total Frames: {total_frames}")
    print(f"[INFO] Sampling Configuration - Target: {target_fps} FPS | Interval: Every {frame_interval} frames")

    # Establish strictly structured directories for CVAT import
    base_dir = Path(output_dir)
    img_dir = base_dir / "images"
    lbl_dir = base_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # Define the canonical target ontology (The 6 CausClass variables)
    TARGET_CLASSES = ["Talk", "Read", "Phone", "Hand", "Lean", "Stand"]
    
    # Generate classes.txt for CVAT YOLO format
    with open(base_dir / "classes.txt", "w") as f:
        f.write("\n".join(TARGET_CLASSES))

    # Knowledge Mapping: Translate old noisy SCB classes to the new rigorous ontology
    # Format: {Old_ID: New_ID} based on manual semantic alignment.
    # Note: 'Talk' (0) and 'Stand' (5) are absent in the old model and require manual annotation.
    CLASS_MAPPING = {
        0: 3,  # old: hand-raising (0) -> new: Hand (3)
        1: 1,  # old: reading (1)      -> new: Read (1)
        2: 1,  # old: writing (2)      -> new: Read (1) (Semantic merge to reduce noise)
        3: 2,  # old: using phone (3)  -> new: Phone (2)
        4: 4,  # old: bowing head (4)  -> new: Lean (4)
        5: 4   # old: leaning (5)      -> new: Lean (4)
    }

    extracted_count = 0
    frame_idx = 0

    print("[PROCESS] Executing frame extraction and neural pseudo-labeling...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_interval == 0:
            # File naming convention optimized for sorting
            base_filename = f"frame_{frame_idx:06d}"
            img_path = img_dir / f"{base_filename}.jpg"
            lbl_path = lbl_dir / f"{base_filename}.txt"

            # 1. Save Raw Image
            cv2.imwrite(str(img_path), frame)

            # 2. Generate Pseudo-labels
            results = model.predict(source=frame, verbose=False, conf=0.1, iou=0.4)
            
            label_lines = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                # Convert to normalized xywh format required by YOLO/CVAT
                xywhn = boxes.xywhn.cpu().numpy()
                classes = boxes.cls.int().cpu().numpy()

                for box, cls_id in zip(xywhn, classes):
                    # Apply semantic mapping
                    if cls_id in CLASS_MAPPING:
                        new_cls_id = CLASS_MAPPING[cls_id]
                        # YOLO format: <class> <x_center> <y_center> <width> <height>
                        label_lines.append(f"{new_cls_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")

            # 3. Export Label File
            with open(lbl_path, "w") as f:
                f.write("\n".join(label_lines))

            extracted_count += 1
            if extracted_count % 50 == 0:
                print(f"[PROCESS] Processed {extracted_count} frames...")

        frame_idx += 1

    cap.release()
    print("\n[SUCCESS] ========================================================")
    print(f"[SUCCESS] Total valid frames extracted: {extracted_count}")
    print(f"[SUCCESS] Dataset compiled at: {base_dir}")
    print("[SUCCESS] The dataset is strictly formatted for direct CVAT ingestion.")
    print("[SUCCESS] ========================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Active Learning: Pseudo-Labeling Pipeline")
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--model", default="yolov8-mhsa.pt", help="Pre-trained weights for pseudo-labeling")
    parser.add_argument("--out", default=str(DATA_DIR / "cvat_dataset"), help="Output directory name")
    parser.add_argument("--fps", type=float, default=0.5, help="Extraction frequency (Frames Per Second)")
    args = parser.parse_args()

    prepare_cvat_dataset(args.video, args.model, args.out, args.fps)
