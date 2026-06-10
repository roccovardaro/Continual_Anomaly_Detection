import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetModel(nn.Module):
    def __init__(self, pretrained=True, num_classes=2):
        super(EfficientNetModel, self).__init__()
        
        if pretrained:
            weights = EfficientNet_B0_Weights.DEFAULT
        else:
            weights = None
            
        self.backbone = efficientnet_b0(weights=weights)
        
        # In efficientnet_b0, the last feature map before the classifier has 1280 channels
        last_layer = 1280
        sequential_layers = [nn.Dropout(p=0.2)]
        head_layers = [512, 512, 128]
        
        for num_neurons in head_layers:
            sequential_layers.append(nn.Linear(last_layer, num_neurons))
            sequential_layers.append(nn.BatchNorm1d(num_neurons))
            sequential_layers.append(nn.ReLU(inplace=True))
            last_layer = num_neurons

        head = nn.Sequential(
            *sequential_layers
        )
        
        # Replace the classifier of EfficientNet with Identity RIMUOVE LA TESTA ORIGINALE
        self.backbone.classifier = nn.Identity()
        
        # Create our custom head
        self.head = nn.Sequential(
            head,
            nn.Linear(last_layer, num_classes)
        )

    def forward(self, x):
        embeds = self.backbone(x)
        logits = self.head(embeds)
        return logits, embeds

    def forward_features(self, x):
        embeds = self.backbone(x)
        return embeds

    def freeze_efficientnet(self):
        # freeze full backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        # unfreeze head:
        for param in self.head.parameters():
            param.requires_grad = True

    def unfreeze(self):
        # unfreeze all:
        for param in self.parameters():
            param.requires_grad = True


    #QUESTA PARTE SUPER IMPORTANTE
    def train(self, mode=True):
        """Override del metodo train per forzare la BatchNorm del backbone in eval."""
        super(EfficientNetModel, self).train(mode)
        if mode:
            for module in self.backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()