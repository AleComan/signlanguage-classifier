"""Training utilities for conditional ASL image generation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.utils import make_grid

from src.models.generative_models import (
    ConditionalDiscriminator,
    ConditionalGenerator,
    build_conditional_gan,
    denormalize_generated_images,
    sample_conditioned_images,
)
from src.utils.config import load_yaml_config
from src.utils.reproducibility import set_global_seed
from src.utils.wandb_utils import finish_wandb_run, init_wandb_run

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train a conditional GAN for ASL image generation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/generation.yaml"),
        help="Path to generation config YAML.",
    )
    return parser.parse_args()


def _resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def build_generation_transform(image_size: int) -> transforms.Compose:
    """Create image transforms for GAN training."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def load_generation_dataset(config: dict[str, Any]) -> datasets.ImageFolder:
    """Load ASL ImageFolder dataset for generative training."""
    data_cfg = config.get("data", {})
    root_dir = _resolve_project_path(data_cfg.get("root_dir", "data/asl_alphabet_v1/processed"))
    train_subdir = data_cfg.get("train_subdir", "train")
    train_dir = root_dir / train_subdir
    if not train_dir.exists():
        raise FileNotFoundError(f"Train folder not found: {train_dir}")

    image_size = int(data_cfg.get("image_size", 64))
    return datasets.ImageFolder(root=train_dir, transform=build_generation_transform(image_size))


def build_generation_dataloader(config: dict[str, Any]) -> tuple[datasets.ImageFolder, DataLoader[Any]]:
    """Build dataset and dataloader for conditional GAN training."""
    data_cfg = config.get("data", {})
    dataset = load_generation_dataset(config)
    loader = DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 64)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    return dataset, loader


def _tensor_grid_to_pil(images: torch.Tensor, nrow: int) -> Image.Image:
    grid = make_grid(denormalize_generated_images(images).cpu(), nrow=nrow, padding=2)
    array = grid.mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def _save_sample_grid(
    *,
    generator: ConditionalGenerator,
    fixed_noise: torch.Tensor,
    fixed_labels: torch.Tensor,
    latent_dim: int,
    device: torch.device,
    samples_dir: Path,
    epoch: int,
    nrow: int,
) -> Image.Image:
    samples_dir.mkdir(parents=True, exist_ok=True)
    images = sample_conditioned_images(
        generator=generator,
        labels=fixed_labels,
        latent_dim=latent_dim,
        device=device,
        noise=fixed_noise,
    )
    grid = _tensor_grid_to_pil(images, nrow=nrow)
    grid.save(samples_dir / f"epoch_{epoch:04d}.png")
    return grid


def _checkpoint_payload(
    *,
    generator: ConditionalGenerator,
    discriminator: ConditionalDiscriminator,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    class_names: list[str],
    class_to_idx: dict[str, int],
    config: dict[str, Any],
    epoch: int,
) -> dict[str, Any]:
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    return {
        "model_type": "conditional_gan",
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_g_state_dict": optimizer_g.state_dict(),
        "optimizer_d_state_dict": optimizer_d.state_dict(),
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "image_size": int(data_cfg.get("image_size", 64)),
        "latent_dim": int(model_cfg.get("latent_dim", 128)),
        "embedding_dim": int(model_cfg.get("embedding_dim", 64)),
        "image_channels": int(model_cfg.get("image_channels", 3)),
        "generator_channels": list(model_cfg.get("generator_channels", [512, 256, 128, 64])),
        "discriminator_channels": list(model_cfg.get("discriminator_channels", [64, 128, 256, 512])),
        "config": config,
    }


def save_gan_checkpoint(
    *,
    generator: ConditionalGenerator,
    discriminator: ConditionalDiscriminator,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    class_names: list[str],
    class_to_idx: dict[str, int],
    config: dict[str, Any],
    output_path: Path,
    epoch: int,
) -> Path:
    """Save a complete conditional GAN checkpoint."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        _checkpoint_payload(
            generator=generator,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            class_names=class_names,
            class_to_idx=class_to_idx,
            config=config,
            epoch=epoch,
        ),
        output_path,
    )
    return output_path


def _prepare_inception_input(images: torch.Tensor) -> torch.Tensor:
    images = denormalize_generated_images(images)
    images = torch.nn.functional.interpolate(
        images,
        size=(299, 299),
        mode="bilinear",
        align_corners=False,
    )
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


@torch.no_grad()
def _inception_probabilities(images: torch.Tensor, device: torch.device, batch_size: int) -> torch.Tensor:
    weights = models.Inception_V3_Weights.DEFAULT
    model = models.inception_v3(weights=weights).to(device)
    model.eval()
    probabilities: list[torch.Tensor] = []
    for start in range(0, images.size(0), batch_size):
        batch = _prepare_inception_input(images[start : start + batch_size].to(device))
        outputs = model(batch)
        if hasattr(outputs, "logits"):
            outputs = outputs.logits
        probabilities.append(torch.softmax(outputs, dim=1).cpu())
    return torch.cat(probabilities, dim=0)


def calculate_inception_score(
    images: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 32,
) -> float:
    """Compute Inception Score for generated images."""
    probabilities = _inception_probabilities(images, device=device, batch_size=batch_size).clamp_min(1e-8)
    marginal = probabilities.mean(dim=0, keepdim=True).clamp_min(1e-8)
    kl_divergence = probabilities * (probabilities.log() - marginal.log())
    return float(torch.exp(kl_divergence.sum(dim=1).mean()).item())


@torch.no_grad()
def _inception_features(images: torch.Tensor, device: torch.device, batch_size: int) -> torch.Tensor:
    weights = models.Inception_V3_Weights.DEFAULT
    model = models.inception_v3(weights=weights).to(device)
    model.fc = nn.Identity()
    model.eval()
    features: list[torch.Tensor] = []
    for start in range(0, images.size(0), batch_size):
        batch = _prepare_inception_input(images[start : start + batch_size].to(device))
        outputs = model(batch)
        if hasattr(outputs, "logits"):
            outputs = outputs.logits
        features.append(outputs.cpu())
    return torch.cat(features, dim=0).double()


def _covariance(features: torch.Tensor) -> torch.Tensor:
    centered = features - features.mean(dim=0, keepdim=True)
    denominator = max(features.size(0) - 1, 1)
    return centered.t().matmul(centered) / denominator


def _matrix_sqrt_psd(matrix: torch.Tensor) -> torch.Tensor:
    symmetric = (matrix + matrix.t()) / 2.0
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    eigenvalues = eigenvalues.clamp_min(0.0).sqrt()
    return (eigenvectors * eigenvalues.unsqueeze(0)).matmul(eigenvectors.t())


def calculate_fid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 32,
) -> float:
    """Compute a lightweight FID estimate using InceptionV3 pool features."""
    real_features = _inception_features(real_images, device=device, batch_size=batch_size)
    fake_features = _inception_features(fake_images, device=device, batch_size=batch_size)

    real_mean = real_features.mean(dim=0)
    fake_mean = fake_features.mean(dim=0)
    real_cov = _covariance(real_features)
    fake_cov = _covariance(fake_features)
    diff = real_mean - fake_mean
    real_cov_sqrt = _matrix_sqrt_psd(real_cov)
    cov_mean = _matrix_sqrt_psd(real_cov_sqrt.matmul(fake_cov).matmul(real_cov_sqrt))
    fid = diff.dot(diff) + torch.trace(real_cov + fake_cov - 2.0 * cov_mean)
    return float(fid.clamp_min(0.0).item())


@torch.no_grad()
def _collect_real_images(loader: DataLoader[Any], num_samples: int, device: torch.device) -> torch.Tensor:
    batches: list[torch.Tensor] = []
    collected = 0
    for images, _ in loader:
        take = min(images.size(0), num_samples - collected)
        batches.append(images[:take].to(device))
        collected += take
        if collected >= num_samples:
            break
    return torch.cat(batches, dim=0)


@torch.no_grad()
def _sample_fake_images(
    *,
    generator: ConditionalGenerator,
    num_samples: int,
    num_classes: int,
    latent_dim: int,
    device: torch.device,
) -> torch.Tensor:
    labels = torch.arange(num_samples, device=device) % num_classes
    noise = torch.randn(num_samples, latent_dim, device=device)
    return generator(noise, labels)


def compute_generative_metrics(
    *,
    generator: ConditionalGenerator,
    loader: DataLoader[Any],
    num_classes: int,
    latent_dim: int,
    device: torch.device,
    num_samples: int,
    batch_size: int,
    compute_inception_score: bool,
    compute_fid: bool,
) -> dict[str, float]:
    """Compute optional generative metrics for W&B monitoring."""
    generator.eval()
    fake_images = _sample_fake_images(
        generator=generator,
        num_samples=num_samples,
        num_classes=num_classes,
        latent_dim=latent_dim,
        device=device,
    )

    metrics: dict[str, float] = {}
    if compute_inception_score:
        metrics["inception_score"] = calculate_inception_score(
            fake_images,
            device=device,
            batch_size=batch_size,
        )
    if compute_fid:
        real_images = _collect_real_images(loader, num_samples=num_samples, device=device)
        metrics["fid"] = calculate_fid(
            real_images=real_images,
            fake_images=fake_images[: real_images.size(0)],
            device=device,
            batch_size=batch_size,
        )
    return metrics


def train_conditional_gan(config: dict[str, Any]) -> Path:
    """Train a conditional GAN and return final checkpoint path."""
    seed = int(config.get("seed", 42))
    set_global_seed(seed)
    device = _resolve_device(str(config.get("device", "auto")))

    dataset, loader = build_generation_dataloader(config)
    class_names = list(dataset.classes)
    num_classes = len(class_names)

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    output_cfg = config.get("output", {})
    tracking_cfg = config.get("tracking", {})
    metrics_cfg = config.get("metrics", {})

    image_size = int(data_cfg.get("image_size", 64))
    latent_dim = int(model_cfg.get("latent_dim", 128))
    embedding_dim = int(model_cfg.get("embedding_dim", 64))
    image_channels = int(model_cfg.get("image_channels", 3))
    label_smoothing = float(model_cfg.get("label_smoothing", 0.9))

    generator, discriminator = build_conditional_gan(
        num_classes=num_classes,
        latent_dim=latent_dim,
        embedding_dim=embedding_dim,
        image_size=image_size,
        image_channels=image_channels,
        generator_channels=model_cfg.get("generator_channels"),
        discriminator_channels=model_cfg.get("discriminator_channels"),
    )
    generator.to(device)
    discriminator.to(device)

    betas = tuple(float(value) for value in training_cfg.get("betas", [0.5, 0.999]))
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=float(training_cfg.get("generator_lr", 0.0002)),
        betas=betas,
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=float(training_cfg.get("discriminator_lr", 0.0002)),
        betas=betas,
    )
    criterion = nn.BCEWithLogitsLoss()

    artifacts_dir = _resolve_project_path(output_cfg.get("artifacts_dir", "artifacts/generation"))
    checkpoint_path = artifacts_dir / output_cfg.get("checkpoint_name", "conditional_gan.pt")
    samples_dir = artifacts_dir / output_cfg.get("samples_dir", "samples")

    run = init_wandb_run(
        config=config,
        enabled=bool(tracking_cfg.get("use_wandb", True)),
        project=str(tracking_cfg.get("project", "image-generation")),
        run_name=tracking_cfg.get("run_name"),
        tags=tracking_cfg.get("tags"),
    )

    fixed_samples_per_class = int(training_cfg.get("fixed_samples_per_class", 1))
    fixed_labels = torch.arange(num_classes, device=device).repeat_interleave(fixed_samples_per_class)
    fixed_noise = torch.randn(fixed_labels.size(0), latent_dim, device=device)
    sample_nrow = min(8, fixed_labels.size(0))

    epochs = int(training_cfg.get("epochs", 50))
    sample_interval = int(training_cfg.get("sample_interval", 1))
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 10))
    metrics_enabled = bool(metrics_cfg.get("enabled", False))
    metrics_interval = int(metrics_cfg.get("interval", 5))

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        total_g_loss = 0.0
        total_d_loss = 0.0
        total_items = 0

        for real_images, labels in loader:
            real_images = real_images.to(device)
            labels = labels.to(device)
            batch_size = real_images.size(0)
            real_targets = torch.full((batch_size,), label_smoothing, device=device)
            fake_targets = torch.zeros(batch_size, device=device)

            optimizer_d.zero_grad(set_to_none=True)
            real_logits = discriminator(real_images, labels)
            real_loss = criterion(real_logits, real_targets)

            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_images = generator(noise, labels).detach()
            fake_logits = discriminator(fake_images, labels)
            fake_loss = criterion(fake_logits, fake_targets)
            discriminator_loss = (real_loss + fake_loss) / 2.0
            discriminator_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad(set_to_none=True)
            noise = torch.randn(batch_size, latent_dim, device=device)
            generated_images = generator(noise, labels)
            generated_logits = discriminator(generated_images, labels)
            generator_loss = criterion(generated_logits, torch.ones(batch_size, device=device))
            generator_loss.backward()
            optimizer_g.step()

            total_d_loss += float(discriminator_loss.item()) * batch_size
            total_g_loss += float(generator_loss.item()) * batch_size
            total_items += batch_size

        epoch_log: dict[str, float | int | Image.Image] = {
            "epoch": epoch,
            "generator_loss": total_g_loss / max(total_items, 1),
            "discriminator_loss": total_d_loss / max(total_items, 1),
        }

        if epoch % sample_interval == 0 or epoch == 1:
            grid = _save_sample_grid(
                generator=generator,
                fixed_noise=fixed_noise,
                fixed_labels=fixed_labels,
                latent_dim=latent_dim,
                device=device,
                samples_dir=samples_dir,
                epoch=epoch,
                nrow=sample_nrow,
            )
            if run is not None:
                import wandb

                epoch_log["fixed_noise_samples"] = wandb.Image(
                    grid,
                    caption=f"Epoch {epoch} fixed-noise conditional samples",
                )

        if metrics_enabled and (epoch % metrics_interval == 0 or epoch == epochs):
            try:
                metrics = compute_generative_metrics(
                    generator=generator,
                    loader=loader,
                    num_classes=num_classes,
                    latent_dim=latent_dim,
                    device=device,
                    num_samples=int(metrics_cfg.get("num_samples", 512)),
                    batch_size=int(metrics_cfg.get("batch_size", 32)),
                    compute_inception_score=bool(metrics_cfg.get("compute_inception_score", True)),
                    compute_fid=bool(metrics_cfg.get("compute_fid", True)),
                )
                epoch_log.update(metrics)
            except Exception as error:
                print(f"No se pudieron calcular metricas generativas en epoch {epoch}: {error}")

        if run is not None:
            run.log(epoch_log)
        else:
            print(
                f"Epoch {epoch:03d}/{epochs} "
                f"G={epoch_log['generator_loss']:.4f} D={epoch_log['discriminator_loss']:.4f}"
            )

        if epoch % checkpoint_interval == 0:
            save_gan_checkpoint(
                generator=generator,
                discriminator=discriminator,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                class_names=class_names,
                class_to_idx=dataset.class_to_idx,
                config=config,
                output_path=artifacts_dir / f"conditional_gan_epoch_{epoch:04d}.pt",
                epoch=epoch,
            )

    final_checkpoint = save_gan_checkpoint(
        generator=generator,
        discriminator=discriminator,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        class_names=class_names,
        class_to_idx=dataset.class_to_idx,
        config=config,
        output_path=checkpoint_path,
        epoch=epochs,
    )
    finish_wandb_run(run)
    return final_checkpoint


def train_conditional_gan_from_config(config_path: str | Path) -> Path:
    """Load YAML config and train the conditional GAN."""
    config = load_yaml_config(config_path)
    return train_conditional_gan(config)


def main() -> None:
    """Run training from the command line."""
    args = parse_args()
    checkpoint_path = train_conditional_gan_from_config(args.config)
    print(f"Checkpoint generativo guardado en: {checkpoint_path}")


if __name__ == "__main__":
    main()
