import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCLossWrapper(nn.Module):
    """
    Wrapper around torch.nn.CTCLoss that applies log_softmax on the input logits.
    """

    def __init__(self, blank: int = 0, zero_infinity: bool = True):
        super().__init__()
        self.blank = blank
        self.ctc_loss = nn.CTCLoss(blank=blank, zero_infinity=zero_infinity)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: Shape [T, B, C]
            targets: 1D Tensor of target token IDs
            input_lengths: 1D Tensor of input sequence lengths
            target_lengths: 1D Tensor of target sequence lengths
        """
        # Apply log_softmax along class dimension C
        log_probs = F.log_softmax(logits, dim=-1)
        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        return loss
