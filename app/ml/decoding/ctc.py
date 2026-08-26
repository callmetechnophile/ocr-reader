import json
from pathlib import Path
from typing import Any, Optional, Sequence
import numpy as np
import torch
import torch.nn.functional as F


# Canonical V1 Vocabulary Characters
DEFAULT_VOCAB_CHARS = [
    # Blank token is index 0
    "<blank>",
    # Space
    " ",
    # Digits
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    # Uppercase ASCII
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    # Lowercase ASCII
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    # Punctuation & brackets
    ".", ",", ";", ":", "!", "?", "'", '"', "(", ")", "[", "]", "{", "}", "-", "_",
    # Math & Engineering Operators
    "+", "*", "/", "=", "%", "<", ">",
    "×", "÷", "±", "≈", "≠", "≤", "≥", "°", "∞", "√", "∑", "∫", "∂", "∇",
    # Greek letters
    "α", "β", "γ", "δ", "ε", "θ", "λ", "μ", "π", "σ", "φ", "ω", "Ω", "Δ",
]


class CTCDecoder:
    """
    Greedy CTC sequence decoder and vocabulary manager.
    Maps between characters and vocabulary indices, collapses repeated tokens,
    removes blanks, and calculates approximate prediction confidences.
    """

    def __init__(self, vocab_chars: Optional[Sequence[str]] = None, blank_idx: int = 0):
        self.vocab_chars = list(vocab_chars) if vocab_chars is not None else list(DEFAULT_VOCAB_CHARS)
        self.blank_idx = blank_idx

        # Ensure blank token is at blank_idx
        if self.blank_idx == 0 and (not self.vocab_chars or self.vocab_chars[0] not in ("<blank>", "[blank]", "")):
            self.vocab_chars.insert(0, "<blank>")

        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab_chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab_chars)}

    @property
    def num_classes(self) -> int:
        return len(self.vocab_chars)

    def encode(self, text: str) -> list[int]:
        """Convert string of text into list of vocabulary indices, ignoring unknown tokens."""
        indices = []
        for char in text:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
        return indices

    def decode_greedy(
        self,
        logits: torch.Tensor,
        valid_lengths: Optional[Sequence[int]] = None,
    ) -> list[dict[str, Any]]:
        """
        Greedy CTC decoding on logits tensor.

        Args:
            logits: Logits tensor of shape [B, T, C] (or [T, B, C]).
            valid_lengths: Optional valid sequence length per sample in batch.

        Returns:
            List of dicts: [{"text": str, "confidence": float, "tokens": list[int]}]
        """
        # Ensure shape is [B, T, C]
        if logits.dim() == 3 and logits.size(1) != self.num_classes and logits.size(2) == self.num_classes:
            # Already [B, T, C]
            b_logits = logits
        elif logits.dim() == 3 and logits.size(0) != self.num_classes and logits.size(2) == self.num_classes:
            # [T, B, C] -> permute to [B, T, C]
            b_logits = logits.permute(1, 0, 2)
        else:
            b_logits = logits

        # Compute softmax probabilities
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

            for t, (idx, prob) in enumerate(zip(sample_indices, sample_probs)):
                # CTC collapse rules:
                # 1. Skip if same as previous non-blank token
                # 2. Skip blank token
                if idx != self.blank_idx:
                    if idx != prev_token:
                        char = self.idx_to_char.get(idx, "")
                        decoded_chars.append(char)
                        decoded_tokens.append(idx)
                        char_confs.append(float(prob))
                prev_token = idx

            # Compute geometric mean / average of decoded token confidences
            if char_confs:
                # Average confidence across recognized characters
                avg_confidence = round(float(np.mean(char_confs)), 4)
            else:
                avg_confidence = 1.0 if not sample_indices else 0.0

            text = "".join(decoded_chars)
            results.append({
                "text": text,
                "confidence": avg_confidence,
                "tokens": decoded_tokens,
            })

        return results

    def save_vocab(self, path: str | Path) -> None:
        """Save vocabulary mapping to JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "v1",
            "blank_idx": self.blank_idx,
            "vocab": self.vocab_chars,
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_vocab(cls, path: str | Path) -> "CTCDecoder":
        """Load vocabulary mapping from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(vocab_chars=data["vocab"], blank_idx=data.get("blank_idx", 0))
