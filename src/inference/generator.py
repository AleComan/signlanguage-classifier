"""Phrase-to-ASL image generation utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import torch
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.models.generative_models import (
    DEFAULT_ASL_CLASSES,
    ConditionalGenerator,
    denormalize_generated_images,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class GeneratedPhraseSequence:
    """Container returned by phrase generation."""

    labels: list[str]
    skipped_characters: list[str]
    frames: list[Image.Image]
    gif_bytes: bytes
    frame_duration: float
    source: str


@dataclass(frozen=True)
class GeneratorBundle:
    """Loaded generator checkpoint metadata."""

    generator: ConditionalGenerator
    class_names: list[str]
    class_to_idx: dict[str, int]
    latent_dim: int
    image_size: int
    device: torch.device


def _resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is None or str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_label = str(device)
    if device_label.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_label)


def resolve_default_generator_checkpoint() -> Path | None:
    """Find the most likely trained generator checkpoint."""
    load_dotenv()
    configured = os.getenv("GENERATOR_MODEL_PATH", "")
    if configured:
        configured_path = _resolve_project_path(configured)
        if configured_path.exists():
            return configured_path

    artifacts_dir = PROJECT_ROOT / "artifacts" / "generation"
    preferred = artifacts_dir / "conditional_gan.pt"
    if preferred.exists():
        return preferred

    if not artifacts_dir.exists():
        return None
    candidates = sorted(
        (path for path in artifacts_dir.rglob("*.pt") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_generator_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
) -> GeneratorBundle:
    """Load a conditional GAN generator checkpoint."""
    resolved_device = _resolve_device(device)
    path = _resolve_project_path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Generator checkpoint not found: {path}")

    payload = torch.load(path, map_location=resolved_device)
    if payload.get("model_type") != "conditional_gan":
        raise ValueError(f"Unsupported generator checkpoint type: {payload.get('model_type')}")

    class_names = list(payload.get("class_names") or DEFAULT_ASL_CLASSES)
    class_to_idx = dict(payload.get("class_to_idx") or {label: idx for idx, label in enumerate(class_names)})
    latent_dim = int(payload.get("latent_dim", 128))
    image_size = int(payload.get("image_size", 64))

    generator = ConditionalGenerator(
        num_classes=len(class_names),
        latent_dim=latent_dim,
        embedding_dim=int(payload.get("embedding_dim", 64)),
        image_size=image_size,
        image_channels=int(payload.get("image_channels", 3)),
        channels=payload.get("generator_channels"),
    )
    generator.load_state_dict(payload["generator_state_dict"])
    generator.to(resolved_device)
    generator.eval()

    return GeneratorBundle(
        generator=generator,
        class_names=class_names,
        class_to_idx=class_to_idx,
        latent_dim=latent_dim,
        image_size=image_size,
        device=resolved_device,
    )


@lru_cache(maxsize=4)
def _load_generator_checkpoint_cached(checkpoint_path: str, device_label: str) -> GeneratorBundle:
    return load_generator_checkpoint(checkpoint_path, device=device_label)


def tokenize_phrase_to_asl_labels(phrase: str) -> tuple[list[str], list[str]]:
    """Tokenize phrase characters into supported ASL class labels."""
    labels: list[str] = []
    skipped: list[str] = []
    supported_letters = set(DEFAULT_ASL_CLASSES[:26])

    for character in phrase:
        if character.isspace():
            labels.append("space")
            continue
        upper_character = character.upper()
        if upper_character in supported_letters:
            labels.append(upper_character)
            continue
        skipped.append(character)

    return labels, skipped


def _label_to_index(label: str, bundle: GeneratorBundle) -> int:
    normalized_map = {class_name.lower(): idx for class_name, idx in bundle.class_to_idx.items()}
    if label.lower() not in normalized_map:
        raise ValueError(f"Class label not available in generator checkpoint: {label}")
    return normalized_map[label.lower()]


def _tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    image_tensor = denormalize_generated_images(image_tensor.unsqueeze(0))[0].cpu()
    array = image_tensor.mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def _image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _load_reference_frame(label: str, image_size: int, data_root: str) -> Image.Image | None:
    root = _resolve_project_path(data_root)
    for split_name in ("train", "val"):
        class_dir = root / split_name / label
        if not class_dir.exists():
            continue
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            with Image.open(image_path) as image:
                return ImageOps.fit(image.convert("RGB"), (image_size, image_size), method=Image.Resampling.LANCZOS)
    return None


def _placeholder_frame(label: str, image_size: int) -> Image.Image:
    image = Image.new("RGB", (image_size, image_size), color=(242, 244, 248))
    draw = ImageDraw.Draw(image)
    margin = max(4, image_size // 16)
    draw.rounded_rectangle(
        (margin, margin, image_size - margin, image_size - margin),
        radius=max(6, image_size // 12),
        outline=(64, 91, 129),
        width=max(2, image_size // 48),
        fill=(255, 255, 255),
    )
    display_label = "_" if label == "space" else label
    try:
        font = ImageFont.truetype("arial.ttf", max(16, image_size // 2))
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), display_label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    draw.text(
        ((image_size - text_width) / 2, (image_size - text_height) / 2 - margin),
        display_label,
        fill=(27, 42, 65),
        font=font,
    )
    if label == "space":
        draw.text((margin * 2, image_size - margin * 4), "space", fill=(92, 107, 130), font=ImageFont.load_default())
    return image


@lru_cache(maxsize=512)
def _render_frame_png_cached(
    label: str,
    checkpoint_path: str,
    device_label: str,
    seed: int,
    image_size: int,
    data_root: str,
) -> tuple[bytes, str]:
    if checkpoint_path:
        bundle = _load_generator_checkpoint_cached(checkpoint_path, device_label)
        label_index = _label_to_index(label, bundle)
        noise_generator = torch.Generator(device=bundle.device)
        noise_generator.manual_seed(seed + label_index * 9973)
        noise = torch.randn(1, bundle.latent_dim, generator=noise_generator, device=bundle.device)
        labels = torch.tensor([label_index], dtype=torch.long, device=bundle.device)
        with torch.inference_mode():
            image = _tensor_to_pil(bundle.generator(noise, labels)[0])
        return _image_to_png_bytes(image), "model"

    reference = _load_reference_frame(label, image_size=image_size, data_root=data_root)
    if reference is not None:
        return _image_to_png_bytes(reference), "dataset"
    return _image_to_png_bytes(_placeholder_frame(label, image_size=image_size)), "placeholder"


def _frames_to_gif_bytes(frames: list[Image.Image], duration_ms: int) -> bytes:
    if not frames:
        raise ValueError("No frames available to build GIF.")
    buffer = BytesIO()
    frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )
    return buffer.getvalue()


def generate_phrase_sequence(
    phrase: str,
    frame_duration: float = 1.0,
    *,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
    image_size: int | None = None,
    seed: int = 42,
    output_path: str | Path | None = None,
    data_root: str | Path = "data/asl_alphabet_v1/processed",
) -> GeneratedPhraseSequence:
    """Generate an animated ASL GIF from a text phrase.

    Unsupported characters are skipped. If a trained cGAN checkpoint is not
    available, the function falls back to one reference dataset image per class
    and finally to deterministic placeholder frames so the app remains usable.
    """
    labels, skipped = tokenize_phrase_to_asl_labels(phrase)
    if not labels:
        raise ValueError("La frase no contiene caracteres ASL soportados.")

    resolved_device = _resolve_device(device)
    resolved_checkpoint = (
        _resolve_project_path(checkpoint_path) if checkpoint_path is not None else resolve_default_generator_checkpoint()
    )
    checkpoint_str = str(resolved_checkpoint.resolve()) if resolved_checkpoint and resolved_checkpoint.exists() else ""

    if checkpoint_str:
        bundle = _load_generator_checkpoint_cached(checkpoint_str, str(resolved_device))
        output_image_size = bundle.image_size
    else:
        output_image_size = int(image_size or 64)

    data_root_str = str(data_root)
    frames: list[Image.Image] = []
    sources: list[str] = []
    for label in labels:
        png_bytes, source = _render_frame_png_cached(
            label,
            checkpoint_str,
            str(resolved_device),
            int(seed),
            output_image_size,
            data_root_str,
        )
        with Image.open(BytesIO(png_bytes)) as image:
            frames.append(image.convert("RGB"))
        sources.append(source)

    duration_ms = int(max(0.05, float(frame_duration)) * 1000)
    gif_bytes = _frames_to_gif_bytes(frames, duration_ms=duration_ms)
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(gif_bytes)

    source = "model" if all(item == "model" for item in sources) else sources[0]
    return GeneratedPhraseSequence(
        labels=labels,
        skipped_characters=skipped,
        frames=frames,
        gif_bytes=gif_bytes,
        frame_duration=float(frame_duration),
        source=source,
    )
