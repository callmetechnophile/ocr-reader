import json
from pathlib import Path
from typing import Any, Optional
import torch
import torch.nn as nn


class ModelCheckpoint:
    """
    Saves best and latest model checkpoints along with metadata.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        model_name: str = "crnn_v1",
        mode: str = "min",  # min for CER / val_loss
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.mode = mode
        self.best_metric = float("inf") if mode == "min" else float("-inf")

    def step(
        self,
        model: nn.Module,
        epoch: int,
        metric: float,
        cer: float,
        wer: float,
        vocab_version: str = "v1",
        image_height: int = 32,
        extra_meta: Optional[dict[str, Any]] = None,
    ) -> bool:
        """
        Check if metric improved and save latest and best checkpoints.

        Returns:
            is_best: True if this checkpoint is the best so far.
        """
        is_best = (metric < self.best_metric) if self.mode == "min" else (metric > self.best_metric)

        metadata = {
            "model": self.model_name,
            "epoch": epoch,
            "vocab_version": vocab_version,
            "image_height": image_height,
            "cer": round(cer, 5),
            "wer": round(wer, 5),
            "metric": round(metric, 5),
            **(extra_meta or {}),
        }

        checkpoint_data = {
            "state_dict": model.state_dict(),
            "metadata": metadata,
        }

        # 1. Save latest checkpoint
        latest_path = self.checkpoint_dir / f"{self.model_name}_latest.pt"
        torch.save(checkpoint_data, latest_path)

        # 2. Save best checkpoint if improved
        if is_best:
            self.best_metric = metric
            best_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
            torch.save(checkpoint_data, best_path)

            # Save best metadata as json alongside
            meta_json = self.checkpoint_dir / f"{self.model_name}_best_meta.json"
            with open(meta_json, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

        return is_best
