import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional
import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.decoding.ctc import CTCDecoder
from src.models.crnn import CRNN
from src.preprocessing.transforms import TextLinePreprocessor


def run_inference(
    image_path: str | Path,
    checkpoint_path: str | Path,
    vocab_path: Optional[str | Path] = None,
    save_debug_dir: Optional[str | Path] = None,
    ground_truth: Optional[str] = None,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Vocab
    v_path = Path(vocab_path or "./models/ocr/vocab.json")
    decoder = CTCDecoder.load_vocab(v_path) if v_path.exists() else CTCDecoder()

    # 2. Load Model
    model = CRNN(num_classes=decoder.num_classes).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Load and preprocess image
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    preprocessor = TextLinePreprocessor(target_height=32)
    norm_img = preprocessor(img)
    tensor = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 32, W]

    with torch.no_grad():
        logits = model(tensor, time_major=False)
        decoded = decoder.decode_greedy(logits)[0]

    pred_text = decoded["text"]
    confidence = decoded["confidence"]

    result = {
        "image": str(image_path),
        "prediction": pred_text,
        "confidence": confidence,
        "ground_truth": ground_truth,
    }

    print(f"Image: {image_path}")
    print(f"Prediction: '{pred_text}' (Confidence: {confidence:.4f})")
    if ground_truth:
        print(f"Ground Truth: '{ground_truth}'")

    # Visual debugging output for mismatches
    if save_debug_dir and ground_truth and pred_text != ground_truth:
        debug_path = Path(save_debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        img_name = Path(image_path).stem
        debug_file = debug_path / f"mismatch_{img_name}.png"

        # Create annotated debug visualization
        vis_h, vis_w = img.shape
        vis_canvas = np.ones((vis_h + 60, max(vis_w, 400)), dtype=np.uint8) * 255
        vis_canvas[:vis_h, :vis_w] = img
        cv2.putText(vis_canvas, f"REF:  {ground_truth}", (5, vis_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1)
        cv2.putText(vis_canvas, f"PRED: {pred_text}", (5, vis_h + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1)
        cv2.imwrite(str(debug_file), vis_canvas)
        print(f"Saved visual debugging mismatch to {debug_file}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer single text line image with CRNN")
    parser.add_argument("--image", type=str, required=True, help="Path to text-line image")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/crnn_v1_best.pt", help="Checkpoint path")
    parser.add_argument("--vocab", type=str, default="./models/ocr/vocab.json", help="Vocab path")
    parser.add_argument("--gt", type=str, default=None, help="Optional ground truth string")
    parser.add_argument("--debug_dir", type=str, default="./evaluation/debug_failures", help="Directory for debug images")
    args = parser.parse_args()

    run_inference(args.image, args.checkpoint, args.vocab, args.debug_dir, args.gt)
