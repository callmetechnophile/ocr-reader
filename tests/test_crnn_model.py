import numpy as np
import pytest
import torch
from app.ml.decoding.ctc import CTCDecoder
from app.ml.inference.recognizer import CRNNRecognizer
from app.ml.models.crnn import CRNN
from app.ml.preprocessing.text_line import TextLinePreprocessor


def test_crnn_forward_and_shapes():
    num_classes = 87
    model = CRNN(num_classes=num_classes, in_channels=1, lstm_hidden=256, lstm_layers=2)
    model.eval()

    # Batch 1: single image width 128 (height must be 32)
    x1 = torch.randn(1, 1, 32, 128)
    out1_batch_major = model(x1, time_major=False)
    # Output width is 128 / 4 = 32
    assert out1_batch_major.shape == (1, 32, num_classes)

    out1_time_major = model(x1, time_major=True)
    assert out1_time_major.shape == (32, 1, num_classes)

    # Batch 2: batch of 4 images with width 256
    x2 = torch.randn(4, 1, 32, 256)
    out2 = model(x2, time_major=False)
    # Output width is 256 / 4 = 64
    assert out2.shape == (4, 64, num_classes)


def test_ctc_decoder_greedy():
    decoder = CTCDecoder()
    blank = decoder.blank_idx  # 0

    # Simulate token indices: [blank, 'M', 'M', blank, 'O', 'S', 'S', blank]
    m_idx = decoder.char_to_idx["M"]
    o_idx = decoder.char_to_idx["O"]
    s_idx = decoder.char_to_idx["S"]

    # Construct one-hot logits tensor of shape [1, 8, num_classes]
    seq_tokens = [blank, m_idx, m_idx, blank, o_idx, s_idx, s_idx, blank]
    seq_len = len(seq_tokens)
    logits = torch.zeros(1, seq_len, decoder.num_classes)
    for t, tok in enumerate(seq_tokens):
        logits[0, t, tok] = 10.0  # high logit

    results = decoder.decode_greedy(logits)
    assert len(results) == 1
    assert results[0]["text"] == "MOS"
    assert results[0]["confidence"] > 0.90


def test_text_line_preprocessor_aspect_ratio_and_padding():
    preprocessor = TextLinePreprocessor(target_height=32)

    # Synthetic image of shape [50, 200]
    img1 = (np.ones((50, 200), dtype=np.uint8) * 255)
    proc1 = preprocessor.preprocess_single(img1)
    assert proc1.shape[0] == 32
    # Aspect ratio ~ 4.0 -> width ~ 128 (multiple of 4)
    assert proc1.shape[1] % 4 == 0

    # Batch processing with different widths
    img2 = (np.ones((40, 100), dtype=np.uint8) * 255)
    batch_tensor, widths = preprocessor.batch_to_tensor([img1, img2])

    assert batch_tensor.shape[0] == 2
    assert batch_tensor.shape[1] == 1
    assert batch_tensor.shape[2] == 32
    assert batch_tensor.shape[3] % 4 == 0
    assert len(widths) == 2


def test_crnn_recognizer_prediction():
    recognizer = CRNNRecognizer(
        model_path="./models/ocr/crnn_v1_best.pt",
        vocab_path="./models/ocr/vocab.json",
        device="cpu",
    )

    dummy_line = np.ones((32, 200), dtype=np.uint8) * 255
    res = recognizer.predict(dummy_line)

    assert "text" in res
    assert "confidence" in res
    assert "model" in res
    assert isinstance(res["text"], str)
    assert 0.0 <= res["confidence"] <= 1.0
