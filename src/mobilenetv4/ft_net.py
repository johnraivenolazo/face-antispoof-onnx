from __future__ import annotations

import torch
import torch.nn as nn
import timm

from src.mobilenetv4.ft_gen import FTGen


class FTNet(nn.Module):
    def __init__(
        self,
        model_name: str = "mobilenetv4_conv_small.e2400_r224_in1k",
        num_classes: int = 3,
        pretrained: bool = True,
        ft_mid_layer: int = 2,
        ft_final_layer: int = 4,
        input_size: int = 224,
    ):
        super(FTNet, self).__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(ft_mid_layer, ft_final_layer),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.2)

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, input_size, input_size)
            dummy_features = self.backbone(dummy_input)
            ft_channels = dummy_features[0].shape[1]
            high_level_feat = dummy_features[1]
            pooled = self.avgpool(high_level_feat)
            flattened = torch.flatten(pooled, 1)
            fc_in_features = flattened.shape[1]

        self.ft_gen = FTGen(in_channels=ft_channels, out_channels=1)
        self.fc = nn.Linear(fc_in_features, num_classes)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        mid_level_feat = features[0]
        high_level_feat = features[1]

        classifier_output = self.avgpool(high_level_feat)
        classifier_output = torch.flatten(classifier_output, 1)
        classifier_output = self.dropout(classifier_output)
        classifier_output = self.fc(classifier_output)

        if self.training:
            fourier_transform = self.ft_gen(mid_level_feat)
            return classifier_output, fourier_transform
        else:
            return classifier_output
