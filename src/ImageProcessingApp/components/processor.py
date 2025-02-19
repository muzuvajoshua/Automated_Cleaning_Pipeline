from typing import Tuple, Optional
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
        Returns None if the image does not meet quality standards.
        """
        try:
            self.logger.debug(f"Opening image: {image_path}")
            img = Image.open(image_path).convert('RGB')

            # Resize image while maintaining aspect ratio
            img.thumbnail(self.target_size, Image.Resampling.LANCZOS)
            self.logger.debug(f"Resized image: {image_path}")

            # Convert to OpenCV format for quality check
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # Check image quality using Laplacian variance
            if not self._check_quality(cv_img):
                self.logger.debug(f"Quality check failed for image: {image_path}")
                return None

            self.logger.debug(f"Image {image_path} passed quality check.")
            return img

        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            return None

    def _check_quality(self, cv_img: np.ndarray) -> bool:
        """
        Check image quality using Laplacian variance method.
        Returns True if the image meets the quality threshold.
        """
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        self.logger.debug(f"Laplacian variance: {variance} (Threshold: {self.quality_threshold})")
        return variance >= self.quality_threshold
