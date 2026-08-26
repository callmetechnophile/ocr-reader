# Models Directory for OCR Checkpoints

This directory is designated for deep learning model weights and checkpoints:

- Future CNN + BiLSTM + CTC text recognition model weights (`checkpoint_cnn_bilstm_ctc.pt`)
- Future Vision-based Layout Detection backbone weights (`layout_detector.pt`)

The `CNNOCRExtractor` will dynamically look for registered weights in this directory when `backend="cnn_ocr"` is configured.
