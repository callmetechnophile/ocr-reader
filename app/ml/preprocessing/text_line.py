from typing import Optional, Sequence
import cv2
import numpy as np
import torch


class TextLinePreprocessor:
    """
    Deterministic image preprocessing for cropped text lines.
    Preserves aspect ratio, resizes to target height (32), and normalizes pixel values.
    """

    def __init__(
        self,
        target_height: int = 32,
        min_width: int = 32,
        max_width: int = 1024,
        apply_contrast: bool = True,
        apply_denoise: bool = False,
    ):
        self.target_height = target_height
        self.min_width = min_width
        self.max_width = max_width
        self.apply_contrast = apply_contrast
        self.apply_denoise = apply_denoise

    def preprocess_single(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess a single text-line image array (BGR, RGB, or Grayscale).

        Returns:
            Preprocessed 2D grayscale image array of shape [target_height, resized_width].
        """
        # 1. Grayscale conversion
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Handle empty/flat images
        if gray.size == 0 or gray.shape[0] == 0 or gray.shape[1] == 0:
            return np.ones((self.target_height, self.min_width), dtype=np.float32)

        # 2. Optional Denoising
        if self.apply_denoise:
            gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        # 3. Optional Contrast Normalization (CLAHE)
        if self.apply_contrast:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        # 4. Aspect-ratio preserving resize to target_height
        h, w = gray.shape[:2]
        aspect_ratio = float(w) / float(h) if h > 0 else 1.0
        new_w = int(round(self.target_height * aspect_ratio))
        # Clamp width within valid bounds and ensure multiple of 4 (matching CNN pooling downsample)
        new_w = max(self.min_width, min(self.max_width, new_w))
        # Round up to multiple of 4
        if new_w % 4 != 0:
            new_w = ((new_w // 4) + 1) * 4

        resized = cv2.resize(gray, (new_w, self.target_height), interpolation=cv2.INTER_CUBIC)

        # 5. Normalize pixel values to [0.0, 1.0] (or standard [-1.0, 1.0])
        # Background is white (1.0), text is dark (0.0). Standardize to [0, 1].
        norm_img = resized.astype(np.float32) / 255.0

        return norm_img

    def to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert preprocessed single 2D image into [1, 1, H, W] tensor."""
        proc = self.preprocess_single(image)
        # Normalize to standard [-1.0, 1.0] or zero-mean
        tensor = torch.from_numpy(proc).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        # Standard image normalization (subtract 0.5, divide 0.5)
        tensor = (tensor - 0.5) / 0.5
        return tensor

    def batch_to_tensor(self, images: Sequence[np.ndarray]) -> tuple[torch.Tensor, list[int]]:
        """
        Preprocess and collate a list of arbitrary-width text-line images into a padded batch.

        Returns:
            padded_batch: Float tensor of shape [B, 1, target_height, max_batch_width]
            widths: List of individual unpadded image widths.
        """
        if not images:
            empty = torch.zeros((0, 1, self.target_height, self.min_width), dtype=torch.float32)
            return empty, []

        processed_list = [self.preprocess_single(img) for img in images]
        widths = [p.shape[1] for p in processed_list]
        max_w = max(widths)

        # Ensure max_w is divisible by 4
        if max_w % 4 != 0:
            max_w = ((max_w // 4) + 1) * 4

        batch_size = len(processed_list)
        # Pad with white background (1.0 in normalized [0, 1] space)
        padded = np.ones((batch_size, 1, self.target_height, max_w), dtype=np.float32)

        for i, proc in enumerate(processed_list):
            w = proc.shape[1]
            padded[i, 0, :, :w] = proc

        # Convert to tensor and scale to [-1.0, 1.0]
        batch_tensor = torch.from_numpy(padded)
        batch_tensor = (batch_tensor - 0.5) / 0.5

        return batch_tensor, widths
