import time
from typing import Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.decoding.ctc import CTCDecoder
from src.evaluation.cer import calculate_cer
from src.evaluation.wer import calculate_wer
from src.training.callbacks import ModelCheckpoint
from src.training.losses import CTCLossWrapper


class OCRTrainer:
    """
    Complete PyTorch training engine for CRNN with CTC Loss, CER/WER evaluation,
    AdamW optimizer, gradient clipping, AMP, and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        decoder: CTCDecoder,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        gradient_clip: float = 5.0,
        mixed_precision: bool = True,
        device: Optional[str] = None,
        checkpoint_dir: str = "./checkpoints",
        model_name: str = "crnn_v1",
    ):
        self.model = model
        self.decoder = decoder
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.gradient_clip = gradient_clip
        self.mixed_precision = mixed_precision and torch.cuda.is_available()

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

        # Loss & Optimizer
        self.criterion = CTCLossWrapper(blank=decoder.blank_idx, zero_infinity=True)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.mixed_precision)

        self.checkpoint = ModelCheckpoint(checkpoint_dir=checkpoint_dir, model_name=model_name)

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        batch_count = 0

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["images"].to(self.device)
            targets = batch["targets"].to(self.device)
            target_lengths = batch["target_lengths"].to(self.device)
            input_lengths = batch["input_lengths"].to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu", enabled=self.mixed_precision):
                # Forward pass: shape [T, B, C]
                logits = self.model(images, time_major=True)
                loss = self.criterion(logits, targets, input_lengths, target_lengths)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            self.scaler.scale(loss).backward()

            if self.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            batch_count += 1

        return total_loss / max(1, batch_count)

    @torch.no_grad()
    def evaluate(self, data_loader: Optional[DataLoader] = None) -> dict[str, float]:
        loader = data_loader or self.val_loader
        if loader is None:
            return {"loss": 0.0, "cer": 0.0, "wer": 0.0}

        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        all_refs: list[str] = []
        all_hyps: list[str] = []

        for batch in loader:
            images = batch["images"].to(self.device)
            targets = batch["targets"].to(self.device)
            target_lengths = batch["target_lengths"].to(self.device)
            input_lengths = batch["input_lengths"].to(self.device)
            ground_truths = batch["texts"]

            logits = self.model(images, time_major=True)
            loss = self.criterion(logits, targets, input_lengths, target_lengths)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                batch_count += 1

            # Decode greedy
            # Transpose to [B, T, C] for decoding
            b_logits = logits.permute(1, 0, 2)
            seq_lens = input_lengths.cpu().tolist()
            decoded = self.decoder.decode_greedy(b_logits, valid_lengths=seq_lens)

            predictions = [d["text"] for d in decoded]
            all_refs.extend(ground_truths)
            all_hyps.extend(predictions)

        avg_loss = total_loss / max(1, batch_count)
        cer = calculate_cer(all_refs, all_hyps)
        wer = calculate_wer(all_refs, all_hyps)

        return {
            "loss": round(avg_loss, 4),
            "cer": cer,
            "wer": wer,
        }

    def fit(
        self,
        epochs: int,
        log_interval: int = 1,
        early_stop_cer: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []

        for epoch in range(1, epochs + 1):
            start_t = time.perf_counter()
            train_loss = self.train_epoch(epoch)

            # Evaluate on validation loader or train loader if no val loader
            eval_metrics = self.evaluate(self.val_loader or self.train_loader)
            duration = time.perf_counter() - start_t

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": eval_metrics["loss"],
                "cer": eval_metrics["cer"],
                "wer": eval_metrics["wer"],
                "duration_s": round(duration, 2),
            }
            history.append(epoch_record)

            # Save checkpoint if CER improved
            is_best = self.checkpoint.step(
                model=self.model,
                epoch=epoch,
                metric=eval_metrics["cer"],
                cer=eval_metrics["cer"],
                wer=eval_metrics["wer"],
            )

            if epoch % log_interval == 0:
                print(
                    f"Epoch {epoch:03d}/{epochs:03d} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {eval_metrics['loss']:.4f} | "
                    f"CER: {eval_metrics['cer']:.4f} | "
                    f"WER: {eval_metrics['wer']:.4f} | "
                    f"Time: {duration:.2f}s"
                    f"{' [BEST]' if is_best else ''}"
                )

            # Check early stopping threshold (e.g. for sanity test when CER -> 0)
            if early_stop_cer is not None and eval_metrics["cer"] <= early_stop_cer:
                print(f"Reached target CER threshold <= {early_stop_cer}. Stopping early at epoch {epoch}.")
                break

        return history
