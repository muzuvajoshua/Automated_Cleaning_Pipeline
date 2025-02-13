from typing import List, Tuple, Optional
import logging
import cv2
import numpy as np
from PIL import Image


class ImageProcessor:
    def __init__(self, target_size: Tuple[int, int] = (800, 600),
                 quality_threshold: int = 50):
        self.target_size = target_size
        self.quality_threshold = quality_threshold
        self.logger = logging.getLogger(__name__)

    def standardize_image(self, image_path: str) -> Optional[Image.Image]:
        """
        Standardize image size, format, and quality.
        Returns None if image doesn't meet quality standards.
        """
        try:
            # Open and convert to RGB
            img = Image.open(image_path).convert('RGB')

            # Resize maintaining aspect ratio
            img.thumbnail(self.target_size, Image.Resampling.LANCZOS)

            # Convert to OpenCV format for quality check
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # Check quality using Laplacian variance
            if not self._check_quality(cv_img):
                self.logger.warning(f"Image {image_path} failed quality check")
                return None

            return img

        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            return None

    def _check_quality(self, cv_img: np.ndarray) -> bool:
        """
        Check image quality using Laplacian variance method.
        Returns True if image meets quality threshold.
        """
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance >= self.quality_threshold