# OCR Training Framework (CRNN V1)

Standalone training and evaluation framework for the **CRNN (CNN $\rightarrow$ BiLSTM $\rightarrow$ CTC)** OCR text-line recognition model.

---

## 1. Architecture

```text
Text-Line Image [B, 1, 32, W]
           │
           ▼
7-Layer CNN Feature Extractor
           │ (Collapses height 32 → 1)
           ▼
Sequence Map [B, W/4, 512]
           │
           ▼
2-Layer Bidirectional LSTM (hidden=256)
           │
           ▼
Linear Projection [W/4, B, Num_Classes]
           │
           ▼
CTCLoss (Zero Infinity, Blank=0) & Greedy Decoder
```

---

## 2. Dataset Format

TSV Manifest (`datasets/synthetic/labels.tsv`):

```text
images/line_000001.png	The drain current increases with gate voltage.
images/line_000002.png	The threshold voltage is approximately 0.7 V.
```

---

## 3. Quickstart Commands

### Step 1: Generate Synthetic Dataset
```bash
python scripts/generate_synthetic.py --num_samples 500 --output_dir ./datasets/synthetic
```

### Step 2: Run Sanity Overfit Verification
```bash
python scripts/sanity_overfit.py
```

### Step 3: Train Model
```bash
python scripts/train.py --config configs/crnn_v1.yaml
```

### Step 4: Evaluate Model
```bash
python scripts/evaluate.py --checkpoint checkpoints/crnn_v1_best.pt --manifest datasets/synthetic/labels.tsv
```

### Step 5: Run Single Image Inference & Debugging
```bash
python scripts/infer.py --image sample.png --checkpoint checkpoints/crnn_v1_best.pt --gt "Ground Truth Text"
```
