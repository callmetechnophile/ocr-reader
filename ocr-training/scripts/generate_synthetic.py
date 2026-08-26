import argparse
import random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Curated textbook engineering corpus
ENGINEERING_SENTENCES = [
    "The drain current increases with gate voltage.",
    "The threshold voltage is approximately 0.7 V.",
    "Semiconductor devices depend on carrier concentration.",
    "VGS = VTH + VOV in the saturation region.",
    "The small-signal transconductance gm is proportional to ID.",
    "MOSFET operation in deep subthreshold exhibits exponential current.",
    "The frequency response is limited by parasitic capacitances Cgs and Cgd.",
    "Bandgap energy Eg for Silicon is 1.12 eV at room temperature.",
    "Intrinsic carrier concentration ni is temperature dependent.",
    "Doping with acceptor atoms creates p-type semiconductor material.",
    "The pn junction depletion width depends on reverse bias voltage VR.",
    "KVL states the sum of voltages in a closed loop equals zero.",
    "KCL states the sum of currents entering a node equals zero.",
    "The differential equation describes RLC transient response.",
    "Input impedance Zin looking into the gate is capacitive at low frequencies.",
    "Output resistance ro represents the channel-length modulation parameter lambda.",
    "The gain-bandwidth product of the operational amplifier is constant.",
    "Feedback improves circuit linearity and gain stability.",
    "Common-source amplifier provides moderate voltage gain with 180 degree phase shift.",
    "The Miller effect multiplies feedback capacitance Cgd by 1 + Av.",
    "Carrier mobility mu is higher for electrons than holes.",
    "Poisson's equation relates electrostatic potential to charge density rho.",
    "Diffusion current is driven by spatial concentration gradient dn/dx.",
    "Drift current velocity is given by v = mu * E for low electric fields.",
    "Thermal voltage VT = k * T / q equals approximately 25.9 mV at 300 K.",
    "CMOS inverter dynamic power dissipation is P = C * VDD^2 * f.",
    "Static power dissipation arises from subthreshold leakage currents.",
    "The noise margin NMH and NML ensure reliable digital operation.",
    "S-parameters characterize high-frequency microwave two-port networks.",
    "Fourier transform converts time-domain signals to frequency spectra.",
    "Laplace transform simplifies differential circuit equations into s-domain.",
    "Bode plots illustrate magnitude in dB and phase in degrees versus frequency.",
    "The transfer function H(s) contains poles and zeros in the complex plane.",
    "Unity-gain frequency wt corresponds to the transition frequency of the transistor.",
    "The slew rate SR limits the maximum rate of change of the output voltage.",
    "Common-mode rejection ratio CMRR measures rejection of input common signals.",
    "Phase margin PM greater than 45 degrees ensures closed-loop stability.",
    "Quasi-Fermi levels describe carrier populations under non-equilibrium.",
    "Ohm's law relates voltage, current, and resistance: V = I * R.",
    "Capacitive reactance Xc = 1 / (2 * pi * f * C) decreases with frequency.",
    "Inductive reactance XL = 2 * pi * f * L increases with frequency.",
    "Maximum power transfer theorem occurs when Zload equals Zsource conjugate.",
    "The quality factor Q quantifies resonant circuit selectivity and energy loss.",
    "Decoupling capacitors suppress high-frequency supply voltage noise spikes.",
    "Differential signaling provides high immunity to substrate and supply noise.",
    "Bandpass filter passes frequencies within lower and upper cutoff limits.",
    "Schottky diode exhibits lower forward voltage drop and faster switching.",
    "Zener diode maintains constant breakdown voltage in reverse bias regime.",
    "Avalanche breakdown occurs via impact ionization in high electric fields.",
    "Silicon dioxide SiO2 acts as an excellent gate dielectric insulator.",
]


def render_text_line(
    text: str,
    font_size: int = 24,
    height: int = 32,
    font_path: str | None = None,
) -> np.ndarray:
    """
    Renders text onto a white image strip of height 32 (or desired height).
    """
    try:
        if font_path and Path(font_path).exists():
            font = ImageFont.truetype(font_path, font_size)
        else:
            # Fallback to default system font or PIL font
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # Measure text bounding box using PIL
    dummy_img = Image.new("RGB", (1000, 100), color="white")
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_w = max(10, bbox[2] - bbox[0])
    text_h = max(10, bbox[3] - bbox[1])

    # Add margin
    pad_x = random.randint(8, 16)
    img_w = text_w + (pad_x * 2)
    img_h = max(height, text_h + 8)

    img = Image.new("L", (img_w, img_h), color=255)
    draw = ImageDraw.Draw(img)

    # Center vertically
    y_pos = (img_h - text_h) // 2
    draw.text((pad_x, y_pos), text, fill=0, font=font)

    # Resize to exact target height while keeping aspect ratio
    aspect = float(img_w) / float(img_h)
    target_w = int(round(height * aspect))
    if target_w % 4 != 0:
        target_w = ((target_w // 4) + 1) * 4

    resized = img.resize((target_w, height), Image.Resampling.LANCZOS)
    return np.array(resized, dtype=np.uint8)


def apply_scanned_artifacts(image: np.ndarray) -> np.ndarray:
    """Applies controlled perturbations simulating scanned physical paper."""
    img = image.copy()

    # 1. Background shading / paper texture
    bg_noise = np.random.normal(245, 5, img.shape).astype(np.float32)
    mask = (img > 200).astype(np.float32)
    blended = img.astype(np.float32) * (1 - mask) + bg_noise * mask
    img = np.clip(blended, 0, 255).astype(np.uint8)

    # 2. Subtle rotation (-1.5 to +1.5 degrees)
    if random.random() < 0.5:
        angle = random.uniform(-1.5, 1.5)
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(img, m, (w, h), borderValue=255)

    # 3. Slight blur
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0.5)

    # 4. Slight contrast / brightness jitter
    alpha = random.uniform(0.9, 1.1)
    beta = random.uniform(-10, 10)
    img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    return img


def generate_synthetic_dataset(
    output_dir: str | Path,
    num_samples: int = 500,
    height: int = 32,
) -> Path:
    out_path = Path(output_dir)
    images_dir = out_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path / "labels.tsv"

    records = []
    for i in range(num_samples):
        # Pick or construct sentence
        base_sentence = random.choice(ENGINEERING_SENTENCES)
        # Random variations: sometimes partial phrases or numbers
        if random.random() < 0.2:
            words = base_sentence.split()
            start = random.randint(0, max(0, len(words) - 3))
            end = random.randint(start + 2, len(words))
            text = " ".join(words[start:end])
        else:
            text = base_sentence

        # Render image
        font_size = random.randint(18, 26)
        line_img = render_text_line(text, font_size=font_size, height=height)
        augmented = apply_scanned_artifacts(line_img)

        filename = f"line_{i+1:06d}.png"
        img_file = images_dir / filename
        cv2.imwrite(str(img_file), augmented)

        rel_path = f"images/{filename}"
        records.append(f"{rel_path}\t{text}")

    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(records) + "\n")

    print(f"Generated {num_samples} synthetic samples in {out_path}")
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic text-line OCR dataset")
    parser.add_argument("--output_dir", type=str, default="./datasets/synthetic", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=300, help="Number of text lines to generate")
    parser.add_argument("--height", type=int, default=32, help="Target image height")
    args = parser.parse_args()

    generate_synthetic_dataset(args.output_dir, args.num_samples, args.height)
