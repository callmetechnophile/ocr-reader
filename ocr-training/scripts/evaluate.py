import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.ocr_dataset import OCRDataset, ocr_collate_fn
from src.decoding.ctc import CTCDecoder
from src.evaluation.cer import calculate_cer
from src.evaluation.wer import calculate_wer
from src.models.crnn import CRNN
from src.preprocessing.transforms import TextLinePreprocessor


def evaluate(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path = "./evaluation",
    vocab_path: Optional[str | Path] = None,
    batch_size: int = 16,
    dataset_name: str = "textbook_test_v1",
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Vocab & Decoder
    v_path = Path(vocab_path or "./models/ocr/vocab.json")
    decoder = CTCDecoder.load_vocab(v_path) if v_path.exists() else CTCDecoder()

    # 2. Load Model
    model = CRNN(num_classes=decoder.num_classes).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Load Dataset
    preprocessor = TextLinePreprocessor(target_height=32)
    dataset = OCRDataset(
        manifest_path=manifest_path,
        decoder=decoder,
        preprocessor=preprocessor,
        transform=None,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=ocr_collate_fn)

    # 4. Predict
    all_refs = []
    all_hyps = []
    all_paths = []

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            input_lengths = batch["input_lengths"]
            texts = batch["texts"]
            paths = batch["paths"]

            logits = model(images, time_major=False)
            decoded = decoder.decode_greedy(logits, valid_lengths=input_lengths.tolist())

            preds = [d["text"] for d in decoded]
            all_refs.extend(texts)
            all_hyps.extend(preds)
            all_paths.extend(paths)

    # 5. Compute Metrics
    cer = calculate_cer(all_refs, all_hyps)
    wer = calculate_wer(all_refs, all_hyps)

    metrics = {
        "model": ckpt.get("metadata", {}).get("model", "crnn_v1"),
        "dataset": dataset_name,
        "cer": cer,
        "wer": wer,
        "samples": len(dataset),
    }

    # Save metrics.json
    metrics_file = out_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save predictions.tsv
    preds_file = out_dir / "predictions.tsv"
    with open(preds_file, "w", encoding="utf-8") as f:
        for p, ref, hyp in zip(all_paths, all_refs, all_hyps):
            f.write(f"{p}\t{ref}\t{hyp}\n")

    print(f"Evaluation complete. Results saved in {out_dir}")
    print(f"CER: {cer:.4f} | WER: {wer:.4f} | Samples: {len(dataset)}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CRNN OCR model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--manifest", type=str, default="./datasets/synthetic/labels.tsv", help="Path to test manifest")
    parser.add_argument("--output_dir", type=str, default="./evaluation", help="Output directory")
    args = parser.parse_args()

    evaluate(args.checkpoint, args.manifest, args.output_dir)
