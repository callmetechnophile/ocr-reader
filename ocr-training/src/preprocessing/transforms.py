import random
from typing import Any, Optional
import cv2
import numpy as np


class TextLineAugmenter:
    """
    Configurable image augmentation pipeline simulating scanned textbook distortions.
    """

    def __init__(
        self,
        rotation_degrees: float = 2.0,
        gaussian_noise: bool = True,
        blur_probability: float = 0.2,
        contrast_probability: float = 0.3,
        jpeg_probability: float = 0.2,
    ):
        self.rotation_degrees = rotation_degrees
        self.gaussian_noise = gaussian_noise
        self.blur_probability = blur_probability
        self.contrast_probability = contrast_probability
        self.jpeg_probability = jpeg_probability

    def __call__(self, image: np.ndarray) -> np.ndarray:
        img = image.copy()

        # 1. Random subtle rotation / skew
        if self.rotation_degrees > 0:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            if abs(angle) > 0.3:
                h, w = img.shape[:2]
                center = (w // 2, h // 2)
                m = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # 2. Random blur
        if random.random() < self.blur_probability:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)

        # 3. Random Gaussian noise
        if self.gaussian_noise and random.random() < 0.3:
            noise = np.random.normal(0, random.uniform(2, 10), img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # 4. Contrast & brightness jitter
        if random.random() < self.contrast_probability:
            alpha = random.uniform(0.8, 1.25)
            beta = random.uniform(-15, 15)
            img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        # 5. JPEG compression artifacts
        if random.random() < self.jpeg_probability:
            quality = random.randint(40, 85)
            _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE if len(img.shape) == 2 else cv2.IMREAD_COLOR)

        return img


class TextLinePreprocessor:
    """
    Standard preprocessing transform: converts to grayscale, resizes to target height (32)
    preserving aspect ratio, and normalizes pixel values to [-1.0, 1.0].
    """

    def __init__(self, target_height: int = 32, min_width: int = 32, max_width: int = 1024):
        self.target_height = target_height
        self.min_width = min_width
        self.max_width = max_width

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape[:2]
        if h == 0 or w == 0:
            return np.ones((self.target_height, self.min_width), dtype=np.float32)

        aspect = float(w) / float(h)
        new_w = int(round(self.target_height * aspect))
        new_w = max(self.min_width, min(self.max_width, new_w))
        if new_w % 4 != 0:
            new_w = ((new_w // 4) + 1) * 4

        resized = cv2.resize(gray, (new_w, self.target_height), interpolation=cv2.INTER_CUBIC)
        norm_img = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        return norm_img
