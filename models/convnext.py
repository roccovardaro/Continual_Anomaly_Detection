import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class ConvNeXtBackbone(nn.Module):
    def __init__(self, pretrained=True, num_classes=2, embed_dim=256):
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = convnext_tiny(weights=weights)

        # feature extractor puro
        self.features = model.features

        # ConvNeXt-Tiny output channels
        self.pool = nn.AdaptiveAvgPool2d(1)

        # projection head per embedding stabile
        self.projector = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),

            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # head di classificazione necessaria per il calcolo della loss (es. DNE)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.features(x)          # (B, 768, H, W)
        x = self.pool(x)              # (B, 768, 1, 1)
        x = torch.flatten(x, 1)       # (B, 768)

        embeds = self.projector(x)        # embedding space stabile
        embeds = F.normalize(embeds, dim=-1)   # IMPORTANTISSIMO per density/anomaly
        
        logits = self.head(embeds)

        return logits, embeds

    def forward_features(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        
        embeds = self.projector(x)
        embeds = F.normalize(embeds, dim=-1)
        return embeds

    def freeze(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True