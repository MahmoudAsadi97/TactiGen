"""Extract frames from a football video clip."""
import cv2
import os
from pathlib import Path
from loguru import logger
from typing import List, Tuple, Dict


class FrameExtractor:
    def __init__(self, sample_rate: int = 5, output_dir: str = "outputs/frames"):
        self.sample_rate = sample_rate  # save every Nth frame
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, video_path: str) -> Dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_paths = []
        frame_idx = 0
        saved = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % self.sample_rate == 0:
                fname = self.output_dir / f"frame_{frame_idx:05d}.jpg"
                cv2.imwrite(str(fname), frame)
                frame_paths.append(str(fname))
                saved += 1
            frame_idx += 1

        cap.release()
        logger.info(f"Extracted {saved} frames from {video_path} (fps={fps:.1f}, duration={duration:.1f}s)")
        return {
            "frame_paths": frame_paths,
            "fps": fps,
            "total_frames": total_frames,
            "duration_seconds": duration,
            "resolution": (width, height),
            "sampled_count": saved
        }
