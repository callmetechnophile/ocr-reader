import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_synthetic import generate_synthetic_dataset
from src.datasets.ocr_dataset import OCRDataset, ocr_collate_fn
from src.decoding.ctc import CTCDecoder
from src.models.crnn import CRNN
from src.preprocessing.transforms import TextLinePreprocessor
from src.training.trainer import OCRTrainer


def run_sanity_overfit(
    num_samples: int = 100,
    epochs: int = 60,
    target_cer: float = 0.05,
    export_to_models_dir: bool = True,
) -> bool:
    print("=" * 60)
    print("CRNN MANDATORY OVERFITTING SANITY EXPERIMENT")
    print("=" * 60)

    # 1. Prepare synthetic dataset
    dataset_dir = Path("./datasets/synthetic")
    labels_file = dataset_dir / "labels.tsv"
    if not labels_file.exists() or len(open(labels_file, "r", encoding="utf-8").readlines()) < num_samples:
        print(f"Generating {num_samples} synthetic textbook text lines for sanity check...")
        generate_synthetic_dataset(output_dir=dataset_dir, num_samples=num_samples, height=32)

    # 2. Vocabulary & Decoder
    vocab_file = Path("./models/ocr/vocab.json")
    decoder = CTCDecoder()
    decoder.save_vocab(vocab_file)
    print(f"Vocabulary classes: {decoder.num_classes} (blank index = {decoder.blank_idx})")

    # 3. Dataset & DataLoader
    preprocessor = TextLinePreprocessor(target_height=32)
    dataset = OCRDataset(
        manifest_path=labels_file,
        decoder=decoder,
        preprocessor=preprocessor,
        transform=None,  # No augmentation during overfit test
    )
    # Subset to num_samples
    dataset.samples = dataset.samples[:num_samples]
    print(f"Loaded {len(dataset)} samples for overfitting test.")

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=ocr_collate_fn,
    )

    # 4. CRNN Model
    model = CRNN(
        num_classes=decoder.num_classes,
        in_channels=1,
        lstm_hidden=256,
        lstm_layers=2,
        dropout=0.0,
    )

    # 5. Trainer
    checkpoint_dir = Path("./checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer = OCRTrainer(
        model=model,
        decoder=decoder,
        train_loader=loader,
        val_loader=loader,
        learning_rate=1e-3,
        weight_decay=0.0,
        gradient_clip=5.0,
        mixed_precision=False,  # CPU / deterministic
        checkpoint_dir=str(checkpoint_dir),
        model_name="crnn_v1",
    )

    print(f"Training CRNN to overfit {num_samples} samples (target CER <= {target_cer})...")
    history = trainer.fit(epochs=epochs, log_interval=5, early_stop_cer=target_cer)

    final_cer = history[-1]["cer"] if history else 1.0
    final_wer = history[-1]["wer"] if history else 1.0
    print(f"\nSanity experiment completed. Final CER: {final_cer:.4f}, Final WER: {final_wer:.4f}")

    if final_cer <= target_cer:
        print(">>> SANITY TEST PASSED: Model successfully overfit the text-line dataset!")
        if export_to_models_dir:
            models_ocr_dir = Path("./models/ocr")
            models_ocr_dir.mkdir(parents=True, exist_ok=True)
            # Copy checkpoint to models/ocr/crnn_v1_best.pt
            best_ckpt = checkpoint_dir / "crnn_v1_best.pt"
            if best_ckpt.exists():
                target_pt = models_ocr_dir / "crnn_v1_best.pt"
                target_pt.write_bytes(best_ckpt.read_bytes())
                print(f"Exported verified CRNN weights to {target_pt}")
            decoder.save_vocab(models_ocr_dir / "vocab.json")
        return True
    else:
        print(f"!!! SANITY TEST FAILED: Final CER {final_cer:.4f} > target {target_cer}. Check gradients / targets.")
        return False


if __name__ == "__main__":
    success = run_sanity_overfit(num_samples=50, epochs=40, target_cer=0.05)
    sys.exit(0 if success else 1)
