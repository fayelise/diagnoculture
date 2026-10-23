"""Partie 2 — Transfer learning : backbones ImageNet de torchvision + 3 stratégies.

  - frozen  : extraction de features (backbone gelé, seule la nouvelle tête apprend) ;
  - partial : fine-tuning des derniers blocs + tête ;
  - full    : fine-tuning complet avec learning rate différencié
              (backbone = lr x backbone_lr_factor, tête = lr).
"""
import torch.nn as nn
from torchvision import models

BACKBONES = {
    # nom : (constructeur, poids ImageNet)
    "mobilenet_v3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
}


class TransferModel(nn.Module):
    """Backbone pré-entraîné + nouvelle tête de classification.

    `backbone` produit un vecteur de features ; `head` est la seule partie
    initialisée aléatoirement. Quand le backbone est gelé, ses BatchNorm sont
    maintenues en mode eval : sinon leurs statistiques (moyenne/variance
    glissantes) dériveraient vers PlantVillage alors que les poids, eux, ne bougent
    pas — ce ne serait plus une vraie « extraction de features ».
    """

    def __init__(self, name: str, num_classes: int, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        ctor, weights = BACKBONES[name]
        net = ctor(weights=weights if pretrained else None)
        self.name = name
        if name.startswith("mobilenet"):
            # classifier d'origine : Linear -> Hardswish -> Dropout -> Linear(1000)
            in_f = net.classifier[-1].in_features
            net.classifier[-1] = nn.Identity()
            self.backbone = net
            self.blocks = net.features            # 13 à 17 blocs inverted residual
        else:  # resnet50
            in_f = net.fc.in_features
            net.fc = nn.Identity()
            self.backbone = net
            self.blocks = nn.Sequential(net.layer1, net.layer2, net.layer3, net.layer4)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
        self.frozen_bn = False

    def forward(self, x):
        return self.head(self.backbone(x))

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.frozen_bn:
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
                    m.eval()
        return self

    def gradcam_layer(self):
        """Dernière couche convolutive (cible de Grad-CAM)."""
        return self.backbone.features[-1] if self.name.startswith("mobilenet") else self.backbone.layer4[-1]


def apply_strategy(model: TransferModel, strategy: str, lr: float, backbone_lr_factor: float = 0.1,
                   n_partial_blocks: int = 3):
    """Gèle / dégèle les couches selon la stratégie et renvoie les groupes de paramètres de l'optimiseur.

    Pour le classifieur MobileNet, la couche dense intermédiaire (Linear 960->1280
    + Hardswish) fait partie du backbone ImageNet : elle est gelée en « frozen »,
    entraînée en « partial » et « full ».
    """
    for p in model.backbone.parameters():
        p.requires_grad = False
    model.frozen_bn = strategy != "full"

    if strategy == "frozen":
        return [{"params": model.head.parameters(), "lr": lr}]

    if strategy == "partial":
        tail = list(model.blocks)[-n_partial_blocks:]
        if model.name.startswith("mobilenet"):
            tail.append(model.backbone.classifier)
        if model.name == "resnet50":
            tail = [model.backbone.layer4]   # dernier « stage » de ResNet-50 (3 blocs bottleneck)
        params = [p for m in tail for p in m.parameters()]
        for p in params:
            p.requires_grad = True
        return [{"params": params, "lr": lr * backbone_lr_factor},
                {"params": model.head.parameters(), "lr": lr}]

    if strategy == "full":
        for p in model.backbone.parameters():
            p.requires_grad = True
        return [{"params": model.backbone.parameters(), "lr": lr * backbone_lr_factor},
                {"params": model.head.parameters(), "lr": lr}]
    raise ValueError(strategy)
