"""Map image pixel coordinates to pitch world coordinates."""
import numpy as np
import cv2
from typing import Tuple, Optional


class PitchCoordinateMapper:
    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.homography_matrix: Optional[np.ndarray] = None

    def calibrate_from_lines(self, src_points: np.ndarray, dst_points: np.ndarray):
        """Compute homography from at least 4 point correspondences."""
        self.homography_matrix, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC)

    def image_to_world(
        self,
        x_image: float,
        y_image: float,
        image_width: int,
        image_height: int
    ) -> Tuple[float, float]:
        if self.homography_matrix is not None:
            pt = np.array([[[x_image, y_image]]], dtype=np.float32)
            world = cv2.perspectiveTransform(pt, self.homography_matrix)
            return float(world[0][0][0]), float(world[0][0][1])
        # Linear fallback: assume image covers full visible pitch area
        x_world = (x_image / image_width) * self.pitch_length
        y_world = (y_image / image_height) * self.pitch_width
        return round(x_world, 2), round(y_world, 2)
