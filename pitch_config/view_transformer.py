import numpy as np
import cv2

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        # Forrás és célpont koordináták
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        # A homográfia mátrix számítása
        self.matrix, _ = cv2.findHomography(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        # Pontok átalakítása a cv2.perspectiveTransform által elvárt alakra
        points = points.reshape(-1, 1, 2).astype(np.float32)
        # Perspektív transzformáció alkalmazása a homográfia mátrix használatával
        points = cv2.perspectiveTransform(points, self.matrix)

        return points.reshape(-1, 2).astype(np.float32)