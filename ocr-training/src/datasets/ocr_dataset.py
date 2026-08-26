from pathlib import Path
from typing import Any, Callable, Optional, Sequence
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from src.decoding.ctc import CTCDecoder
from src.preprocessing.transforms import TextLinePreprocessor


class OCRDataset(Dataset):
    """
    Text-line OCR Dataset reading from TSV manifest: image_path<TAB>transcription
    """

    def __init__(
        self,
        manifest_path: str | Path,
        base_dir: Optional[str | Path] = None,
        decoder: Optional[CTCDecoder] = None,
        preprocessor: Optional[TextLinePreprocessor] = None,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.base_dir = Path(base_dir or self.manifest_path.parent)
        self.decoder = decoder or CTCDecoder()
        self.preprocessor = preprocessor or TextLinePreprocessor(target_height=32)
        self.transform = transform

        self.samples: list[tuple[Path, str]] = []
        self._load_manifest()

    def _load_manifest(self) -> None:
        if not self.manifest_path.exists():
            return

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith("#"):
                    continue
                parts = line_str.split("\t", 1)
                if len(parts) == 2:
                    rel_path, text = parts
                elif len(parts) == 1:
                    rel_path, text = parts[0], ""
                else:
                    continue

                full_path = (self.base_dir / rel_path).resolve()
                self.samples.append((full_path, text))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_path, target_text = self.samples[idx]

        # Load image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback for missing or corrupted sample
            img = np.ones((32, 128), dtype=np.uint8) * 255

        # Apply augmentation if configured
        if self.transform is not None:
            img = self.transform(img)

        # Preprocess to normalized 2D array [32, W]
        norm_img = self.preprocessor(img)

        # Encode target transcription
        target_tokens = self.decoder.encode(target_text)

        return {
            "image": norm_img,
            "width": norm_img.shape[1],
            "target": target_tokens,
            "target_length": len(target_tokens),
            "text": target_text,
            "path": str(img_path),
        }


def ocr_collate_fn(batch: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor | list[str]]:
    """
    Pads variable-width text line images to the maximum width in the batch.
    CNN downsamples width by a factor of 4, so max_w is rounded up to a multiple of 4.
    """
    batch_size = len(batch)
    widths = [item["width"] for item in batch]
    max_w = max(widths)
    if max_w % 4 != 0:
        max_w = ((max_w // 4) + 1) * 4

    target_height = batch[0]["image"].shape[0]

    # Initialize padded images with white background (normalized to +1.0)
    padded_images = np.ones((batch_size, 1, target_height, max_w), dtype=np.float32)

    input_lengths = []
    target_lengths = []
    all_targets = []
    texts = []
    paths = []

    for i, item in enumerate(batch):
        w = item["width"]
        padded_images[i, 0, :, :w] = item["image"]
        # Output sequence length produced by CNN: W // 4
        seq_len = max(1, w // 4)
        input_lengths.append(seq_len)

        target = item["target"]
        all_targets.extend(target)
        target_lengths.append(len(target))
        texts.append(item["text"])
        paths.append(item["path"])

    images_tensor = torch.from_numpy(padded_images)
    targets_tensor = torch.tensor(all_targets, dtype=torch.long)
    target_lengths_tensor = torch.tensor(target_lengths, dtype=torch.long)
    input_lengths_tensor = torch.tensor(input_lengths, dtype=torch.long)

    return {
        "images": images_tensor,
        "targets": targets_tensor,
        "target_lengths": target_lengths_tensor,
        "input_lengths": input_lengths_tensor,
        "texts": texts,
        "paths": paths,
    }
