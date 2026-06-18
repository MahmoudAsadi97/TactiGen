"""Segment a video clip into overlapping sub-clips."""
from typing import List, Dict


class ClipSegmenter:
    def __init__(self, clip_duration: float = 6.0, overlap: float = 2.0, fps: float = 25.0):
        self.clip_duration = clip_duration
        self.overlap = overlap
        self.fps = fps

    def segment(self, total_duration: float, clip_id_prefix: str = "clip") -> List[Dict]:
        segments = []
        step = self.clip_duration - self.overlap
        start = 0.0
        idx = 0
        while start < total_duration:
            end = min(start + self.clip_duration, total_duration)
            segments.append({
                "clip_id": f"{clip_id_prefix}_{idx:03d}",
                "start_time": round(start, 3),
                "end_time": round(end, 3),
                "frame_start": int(start * self.fps),
                "frame_end": int(end * self.fps),
                "duration": round(end - start, 3)
            })
            idx += 1
            start += step
            if end >= total_duration:
                break
        return segments
