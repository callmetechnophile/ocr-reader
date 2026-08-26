import argparse
import sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.ocr_dataset import OCRDataset, ocr_collate_fn
from src.decoding.ctc import CTCDecoder
from src.models.crnn import CRNN
from src.preprocessing.transforms import TextLineAugmenter, TextLinePreprocessor
from src.training.trainer import OCRTrainer


def train(config_path: str | Path) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. Vocabulary & Decoder
    vocab_path = config["data"].get("vocab_path", "./models/ocr/vocab.json")
    if Path(vocab_path).exists():
        decoder = CTCDecoder.load_vocab(vocab_path)
    else:
        decoder = CTCDecoder()
        decoder.save_vocab(vocab_path)

    print(f"Loaded vocabulary with {decoder.num_classes} classes (blank_idx={decoder.blank_idx})")

    # 2. Preprocessor & Augmentation
    preprocessor = TextLinePreprocessor(
        target_height=config["data"].get("image_height", 32),
        min_width=config["data"].get("min_width", 32),
        max_width=config["data"].get("max_width", 1024),
    )

    aug_cfg = config.get("augmentation", {})
    augmenter = TextLineAugmenter(
        rotation_degrees=aug_cfg.get("rotation_degrees", 2.0),
        gaussian_noise=aug_cfg.get("gaussian_noise", True),
        blur_probability=aug_cfg.get("blur_probability", 0.2),
        contrast_probability=aug_cfg.get("contrast_probability", 0.3),
        jpeg_probability=aug_cfg.get("jpeg_probability", 0.2),
    )

    # 3. Datasets & Loaders
    train_manifest = config["data"]["train_manifest"]
    train_dataset = OCRDataset(
        manifest_path=train_manifest,
        decoder=decoder,
        preprocessor=preprocessor,
        transform=augmenter,
    )
    print(f"Training dataset: {len(train_dataset)} samples from {train_manifest}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"].get("batch_size", 16),
        shuffle=True,
        collate_fn=ocr_collate_fn,
        num_workers=config["training"].get("num_workers", 0),
    )

    val_loader = None
    val_manifest = config["data"].get("val_manifest")
    if val_manifest and Path(val_manifest).exists():
        val_dataset = OCRDataset(
            manifest_path=val_manifest,
            decoder=decoder,
            preprocessor=preprocessor,
            transform=None,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"].get("batch_size", 16),
            shuffle=False,
            collate_fn=ocr_collate_fn,
            num_workers=config["training"].get("num_workers", 0),
        )
        print(f"Validation dataset: {len(val_dataset)} samples from {val_manifest}")

    # 4. CRNN Model
    model = CRNN(
        num_classes=decoder.num_classes,
        in_channels=config["model"].get("in_channels", 1),
        lstm_hidden=config["model"].get("lstm_hidden", 256),
        lstm_layers=config["model"].get("lstm_layers", 2),
        dropout=config["model"].get("dropout", 0.1),
    )

    # 5. Trainer
    trainer = OCRTrainer(
        model=model,
        decoder=decoder,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=float(config["training"].get("learning_rate", 1e-3)),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
        gradient_clip=float(config["training"].get("gradient_clip", 5.0)),
        mixed_precision=bool(config["training"].get("mixed_precision", True)),
        checkpoint_dir=config["checkpoint"].get("directory", "./checkpoints"),
        model_name=config["model"].get("name", "crnn_v1"),
    )

    # 6. Fit
    epochs = config["training"].get("epochs", 20)
    print(f"Starting CRNN training for {epochs} epochs on device {trainer.device}...")
    trainer.fit(epochs=epochs, log_interval=1)
    print("Training finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CRNN OCR model")
    parser.add_argument("--config", type=str, default="./configs/crnn_v1.yaml", help="Path to YAML config")
    args = parser.parse_args()

    train(args.config)
