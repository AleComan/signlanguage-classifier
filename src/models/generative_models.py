"""Conditional generative models for ASL image synthesis."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

DEFAULT_ASL_CLASSES = [chr(code) for code in range(ord("A"), ord("Z") + 1)] + [
    "delete",
    "nothing",
    "space",
]


def _validate_power_of_two_image_size(image_size: int) -> None:
    if image_size < 16 or image_size & (image_size - 1) != 0:
        raise ValueError("image_size must be a power of two and >= 16.")


def _expected_scale_steps(image_size: int) -> int:
    _validate_power_of_two_image_size(image_size)
    return int(math.log2(image_size) - 2)


class ConditionalGenerator(nn.Module):
    """DCGAN-style conditional generator.

    The model receives a latent vector plus a learned class embedding and
    upsamples from a 4x4 feature map to the requested image resolution.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        latent_dim: int = 128,
        embedding_dim: int = 64,
        image_size: int = 64,
        image_channels: int = 3,
        channels: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        steps = _expected_scale_steps(image_size)
        if channels is None:
            channels = [512, 256, 128, 64][:steps]
        channels = list(channels)
        if len(channels) != steps:
            raise ValueError(
                f"generator channels must contain {steps} entries for image_size={image_size}; "
                f"got {len(channels)}."
            )

        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.image_size = image_size
        self.image_channels = image_channels
        self.channels = channels

        self.label_embedding = nn.Embedding(num_classes, embedding_dim)
        self.project = nn.Sequential(
            nn.Linear(latent_dim + embedding_dim, channels[0] * 4 * 4),
            nn.BatchNorm1d(channels[0] * 4 * 4),
            nn.ReLU(inplace=True),
        )

        blocks: list[nn.Module] = []
        in_channels = channels[0]
        for out_channels in channels[1:]:
            blocks.extend(
                [
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels

        blocks.extend(
            [
                nn.ConvTranspose2d(
                    in_channels,
                    image_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.Tanh(),
            ]
        )
        self.net = nn.Sequential(*blocks)

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Generate images conditioned on class labels."""
        embedded_labels = self.label_embedding(labels)
        conditioned_input = torch.cat([noise, embedded_labels], dim=1)
        projected = self.project(conditioned_input)
        projected = projected.view(noise.size(0), self.channels[0], 4, 4)
        return self.net(projected)


class ConditionalDiscriminator(nn.Module):
    """Patch-free conditional discriminator with class maps."""

    def __init__(
        self,
        *,
        num_classes: int,
        image_size: int = 64,
        image_channels: int = 3,
        channels: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        steps = _expected_scale_steps(image_size)
        if channels is None:
            channels = [64, 128, 256, 512][:steps]
        channels = list(channels)
        if len(channels) != steps:
            raise ValueError(
                f"discriminator channels must contain {steps} entries for image_size={image_size}; "
                f"got {len(channels)}."
            )

        self.num_classes = num_classes
        self.image_size = image_size
        self.image_channels = image_channels
        self.channels = channels

        self.label_projection = nn.Embedding(num_classes, image_size * image_size)

        blocks: list[nn.Module] = []
        in_channels = image_channels + 1
        for idx, out_channels in enumerate(channels):
            blocks.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            if idx > 0:
                blocks.append(nn.BatchNorm2d(out_channels))
            blocks.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels

        self.features = nn.Sequential(*blocks)
        final_spatial_size = image_size // (2 ** len(channels))
        self.classifier = nn.Linear(channels[-1] * final_spatial_size * final_spatial_size, 1)

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return real/fake logits for images conditioned on labels."""
        label_map = self.label_projection(labels)
        label_map = label_map.view(labels.size(0), 1, self.image_size, self.image_size)
        conditioned_images = torch.cat([images, label_map], dim=1)
        features = self.features(conditioned_images)
        logits = self.classifier(features.flatten(start_dim=1))
        return logits.squeeze(1)


def initialize_dcgan_weights(module: nn.Module) -> None:
    """Apply standard DCGAN weight initialization."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.BatchNorm2d | nn.BatchNorm1d):
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


def build_conditional_gan(
    *,
    num_classes: int,
    latent_dim: int,
    embedding_dim: int,
    image_size: int,
    image_channels: int = 3,
    generator_channels: Sequence[int] | None = None,
    discriminator_channels: Sequence[int] | None = None,
) -> tuple[ConditionalGenerator, ConditionalDiscriminator]:
    """Build and initialize generator/discriminator pair."""
    generator = ConditionalGenerator(
        num_classes=num_classes,
        latent_dim=latent_dim,
        embedding_dim=embedding_dim,
        image_size=image_size,
        image_channels=image_channels,
        channels=generator_channels,
    )
    discriminator = ConditionalDiscriminator(
        num_classes=num_classes,
        image_size=image_size,
        image_channels=image_channels,
        channels=discriminator_channels,
    )
    generator.apply(initialize_dcgan_weights)
    discriminator.apply(initialize_dcgan_weights)
    return generator, discriminator


def denormalize_generated_images(images: torch.Tensor) -> torch.Tensor:
    """Map generator output tensors from [-1, 1] to [0, 1]."""
    return ((images.detach() + 1.0) / 2.0).clamp(0.0, 1.0)


@torch.no_grad()
def sample_conditioned_images(
    *,
    generator: ConditionalGenerator,
    labels: torch.Tensor,
    latent_dim: int,
    device: torch.device,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate a batch for fixed labels and optional fixed noise."""
    generator.eval()
    if noise is None:
        noise = torch.randn(labels.size(0), latent_dim, device=device)
    return generator(noise.to(device), labels.to(device))
