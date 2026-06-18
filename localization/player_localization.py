"""YOLOv8-based player detection and localization."""
import json
from pathlib import Path
from typing import List, Dict
from loguru import logger

from localization.localization_schema import LocalizationRecord
from localization.coordinate_mapping import PitchCoordinateMapper


class PlayerLocalizer:
    def __init__(self, model_name: str = "yolov8m.pt", confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.mapper = PitchCoordinateMapper()
        self._model = None
        self.model_name = model_name

    def _load_model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
                self._model = YOLO(self.model_name)
                logger.info(f"YOLOv8 model loaded: {self.model_name}")
            except Exception as e:
                logger.warning(f"YOLOv8 unavailable ({e}). Using synthetic localization.")

    def localize_frame(
        self,
        frame_path: str,
        frame_id: int,
        timestamp: float,
        clip_id: str
    ) -> List[LocalizationRecord]:
        self._load_model()

        if self._model is None:
            return self._synthetic_localize(frame_id, timestamp, clip_id)

        import cv2
        frame = cv2.imread(frame_path)
        if frame is None:
            logger.warning(f"Could not read frame: {frame_path}")
            return []

        h, w = frame.shape[:2]
        results = self._model(frame, classes=[0], verbose=False)  # class 0 = person
        records = []
        player_id = 0

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                bw, bh = x2 - x1, y2 - y1
                cx = x1 + bw / 2   # bottom-center x
                cy = float(y2)      # bottom of box (feet approximation)
                x_world, y_world = self.mapper.image_to_world(cx, cy, w, h)
                records.append(LocalizationRecord(
                    clip_id=clip_id,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    player_id=player_id,
                    x_image=cx,
                    y_image=cy,
                    x_world=x_world,
                    y_world=y_world,
                    bbox=[x1, y1, bw, bh],
                    confidence=conf,
                    source="yolov8"
                ))
                player_id += 1

        return records

    def localize_from_synthetic(self, synthetic_clip_path: str) -> List[LocalizationRecord]:
        """Load pre-generated synthetic player positions."""
        with open(synthetic_clip_path) as f:
            data = json.load(f)
        records = []
        clip_id = data["clip_id"]
        for frame in data["frames"]:
            for p in frame["players"]:
                records.append(LocalizationRecord(
                    clip_id=clip_id,
                    frame_id=frame["frame_id"],
                    timestamp=frame["timestamp"],
                    player_id=p["player_id"],
                    x_image=0.0,
                    y_image=0.0,
                    x_world=p["x_world"],
                    y_world=p["y_world"],
                    bbox=[0, 0, 0, 0],
                    confidence=p["confidence"],
                    source="synthetic"
                ))
        return records

    def _synthetic_localize(self, frame_id: int, timestamp: float, clip_id: str) -> List[LocalizationRecord]:
        import random
        rng = random.Random(frame_id)
        records = []
        for i in range(10):
            records.append(LocalizationRecord(
                clip_id=clip_id,
                frame_id=frame_id,
                timestamp=timestamp,
                player_id=i,
                x_image=0.0, y_image=0.0,
                x_world=round(rng.uniform(30, 90), 2),
                y_world=round(rng.uniform(5, 63), 2),
                bbox=[0, 0, 0, 0],
                confidence=round(rng.uniform(0.65, 0.92), 3),
                source="synthetic"
            ))
        return records
