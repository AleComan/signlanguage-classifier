"""Transfer learning model builders."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18_finetune(
    *,
    num_classes: int,
    freeze_backbone: bool = True,
    unfreeze_last_n_layers: int = 1,
) -> nn.Module:
    """Build ResNet18 for fine-tuning with partial layer unfreeze support."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    layer_sequence = [model.layer1, model.layer2, model.layer3, model.layer4]
    if freeze_backbone and unfreeze_last_n_layers > 0:
        for layer in layer_sequence[-unfreeze_last_n_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    return model
