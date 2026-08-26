import json
from pathlib import Path
from typing import Any, Optional, Sequence
import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_VOCAB_CHARS = [
    "<blank>", " ",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    ".", ",", ";", ":", "!", "?", "'", '"', "(", ")", "[", "]", "{", "}", "-", "_",
    "+", "*", "/", "=", "%", "<", ">",
    "×", "÷", "±", "≈", "≠", "≤", "≥", "°", "∞", "√", "∑", "∫", "∂", "∇",
    "α", "β", "γ", "δ", "ε", "θ", "λ", "μ", "π", "σ", "φ", "ω", "Ω", "Δ",
]


class CTCDecoder:
    """Greedy CTC sequence decoder and vocabulary converter for training."""

    def __init__(self, vocab_chars: Optional[Sequence[str]] = None, blank_idx: int = 0):
        self.vocab_chars = list(vocab_chars) if vocab_chars is not None else list(DEFAULT_VOCAB_CHARS)
        self.blank_idx = blank_idx

        if self.blank_idx == 0 and (not self.vocab_chars or self.vocab_chars[0] not in ("<blank>", "[blank]", "")):
            self.vocab_chars.insert(0, "<blank>")

        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab_chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab_chars)}

    @property
    def num_classes(self) -> int:
        return len(self.vocab_chars)

    def encode(self, text: str) -> list[int]:
        """Convert string of text into integer token IDs."""
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]

    def decode_greedy(
        self,
        logits: torch.Tensor,
        valid_lengths: Optional[Sequence[int]] = None,
    ) -> list[dict[str, Any]]:
        """
        Greedy CTC decoding on logits.
        Args:
            logits: Shape [B, T, C] or [T, B, C].
        """
        if logits.dim() == 3 and logits.size(0) != self.num_classes and logits.size(2) == self.num_classes:
            # [T, B, C] -> permute to [B, T, C] if first dim is T
            if logits.size(0) > logits.size(1) and logits.size(1) < 1000:
                b_logits = logits.permute(1, 0, 2)
            else:
                b_logits = logits
        else:
            b_logits = logits

        probs = F.softmax(b_logits, dim=-1)
        max_probs, argmax_indices = torch.max(probs, dim=-1)  # [B, T]

        batch_size, seq_len = argmax_indices.size()
        results: list[dict[str, Any]] = []

        for b in range(batch_size):
            max_len = valid_lengths[b] if valid_lengths is not None else seq_len
            sample_indices = argmax_indices[b, :max_len].tolist()
            sample_probs = max_probs[b, :max_len].tolist()

            decoded_chars: list[str] = []
            decoded_tokens: list[int] = []
            char_confs: list[float] = []
            prev_token = self.blank_idx

            for idx, prob in zip(sample_indices, sample_probs):
                if idx != self.blank_idx:
                    if idx != prev_token:
                        char = self.idx_to_char.get(idx, "")
                        decoded_chars.append(char)
                        decoded_tokens.append(idx)
                        char_confs.append(float(prob))
                prev_token = idx

            avg_confidence = round(float(np.mean(char_confs)), 4) if char_confs else 1.0
            text = "".join(decoded_chars)
            results.append({
                "text": text,
                "confidence": avg_confidence,
                "tokens": decoded_tokens,
            })

        return results

    def save_vocab(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump({"version": "v1", "blank_idx": self.blank_idx, "vocab": self.vocab_chars}, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_vocab(cls, path: str | Path) -> "CTCDecoder":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(vocab_chars=data["vocab"], blank_idx=data.get("blank_idx", 0))
