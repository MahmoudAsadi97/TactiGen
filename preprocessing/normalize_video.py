"""Normalize video resolution and frame rate using FFmpeg."""
import ffmpeg
from pathlib import Path
from loguru import logger


class VideoNormalizer:
    def __init__(self, target_width: int = 1280, target_height: int = 720, target_fps: int = 25):
        self.target_width = target_width
        self.target_height = target_height
        self.target_fps = target_fps

    def normalize(self, input_path: str, output_path: str = None) -> str:
        input_path = Path(input_path)
        if output_path is None:
            output_path = str(input_path.parent / f"{input_path.stem}_normalized.mp4")
        logger.info(f"Normalizing {input_path} -> {output_path}")
        (
            ffmpeg
            .input(str(input_path))
            .output(
                output_path,
                vf=f"scale={self.target_width}:{self.target_height}",
                r=self.target_fps,
                vcodec="libx264",
                crf=23,
                preset="fast",
                acodec="aac"
            )
            .overwrite_output()
            .run(quiet=True)
        )
        logger.success(f"Normalized video saved to {output_path}")
        return output_path
