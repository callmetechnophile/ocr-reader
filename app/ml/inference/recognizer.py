from pathlib import Path
from typing import Any, Optional, Sequence
import numpy as np
import torch
from app.core.config import settings
from app.core.logging import logger
from app.ml.decoding.ctc import CTCDecoder
from app.ml.models.crnn import CRNN
from app.ml.preprocessing.text_line import TextLinePreprocessor


class CRNNRecognizer:
    """
    Inference wrapper for CRNN text-line recognition model.
    Loads PyTorch weights once, caches the model, and provides batch/single prediction.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        vocab_path: Optional[str | Path] = None,
        device: Optional[str] = None,
        model_version: str = "crnn_v1",
    ):
        self.model_version = model_version or getattr(settings, "MODEL_VERSION", "crnn_v1")
        self.device_str = device or getattr(settings, "MODEL_DEVICE", "cpu")
        self.device = torch.device(self.device_str if torch.cuda.is_available() and "cuda" in self.device_str else "cpu")

        # 1. Resolve and load vocabulary
        self.vocab_path = Path(vocab_path or getattr(settings, "VOCAB_PATH", "./models/ocr/vocab.json"))
        if self.vocab_path.exists():
            self.decoder = CTCDecoder.load_vocab(self.vocab_path)
            logger.info("Loaded OCR vocabulary", extra={"vocab_size": self.decoder.num_classes, "path": str(self.vocab_path)})
        else:
            self.decoder = CTCDecoder()
            # Save default vocabulary for consistency
            self.decoder.save_vocab(self.vocab_path)

        # 2. Instantiate preprocessor
        self.preprocessor = TextLinePreprocessor(target_height=32)

        # 3. Instantiate model
        self.model = CRNN(
            num_classes=self.decoder.num_classes,
            in_channels=1,
            lstm_hidden=256,
            lstm_layers=2,
        ).to(self.device)

        # 4. Load weights if checkpoint exists
        self.model_path = Path(model_path or getattr(settings, "MODEL_PATH", "./models/ocr/crnn_v1_best.pt"))
        self.is_loaded = False
        if self.model_path.exists():
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                state_dict = checkpoint.get("state_dict", checkpoint)
                self.model.load_state_dict(state_dict)
                self.is_loaded = True
                self.model_version = checkpoint.get("model", self.model_version)
                logger.info(
                    "Loaded CRNN OCR checkpoint",
                    extra={"path": str(self.model_path), "model_version": self.model_version},
                )
            except Exception as exc:
                logger.warning(
                    f"Failed to load CRNN checkpoint from {self.model_path}: {exc}. Using uninitialized/fallback model."
                )
        self.model.eval()

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> dict[str, Any]:
        """
        Recognize text in a single cropped text-line image.

        Args:
            image: 2D or 3D numpy array representing a cropped text line.

        Returns:
            dict with keys: 'text', 'confidence', 'model'
        """
        results = self.predict_batch([image])
        return results[0]

    @torch.no_grad()
    def predict_batch(self, images: Sequence[np.ndarray]) -> list[dict[str, Any]]:
        """
        Batch prediction on multiple text-line images with dynamic width padding.

        Args:
            images: List of 2D or 3D numpy arrays.

        Returns:
            List of dicts: [{'text': str, 'confidence': float, 'model': str}]
        """
        if not images:
            return []

        # 1. Preprocess and collate images to batch tensor [B, 1, 32, W_max]
        batch_tensor, widths = self.preprocessor.batch_to_tensor(images)
        batch_tensor = batch_tensor.to(self.device)

        # 2. Forward pass through CRNN -> [B, W_seq, num_classes]
        logits = self.model(batch_tensor, time_major=False)

        # 3. Calculate sequence downsampling factor (CNN downsamples width by factor of 4)
        # Sequence lengths in output space:
        seq_lengths = [max(1, w // 4) for w in widths]

        # 4. Decode greedy CTC
        decoded_batch = self.decoder.decode_greedy(logits, valid_lengths=seq_lengths)

        formatted_results = []
        for dec in decoded_batch:
            formatted_results.append({
                "text": dec["text"],
                "confidence": dec["confidence"],
                "model": self.model_version,
            })

        return formatted_results


# Global singleton recognizer instance for reuse across requests
_global_recognizer: Optional[CRNNRecognizer] = None


def get_recognizer() -> CRNNRecognizer:
    global _global_recognizer
    if _global_recognizer is None:
        _global_recognizer = CRNNRecognizer()
    return _global_recognizer
