"""Partie 1 — CNN entraînés de zéro : CNN-Plain et CNN-Res (128x128).

Les deux modèles ont la même profondeur (6 couches convolutives sur le chemin
principal) et un nombre de paramètres comparable (~0,35 M vs ~0,30 M) : la seule
différence de fond est la présence de connexions résiduelles + GAP.
"""
import torch.nn as nn


def conv_bn_relu(c_in: int, c_out: int) -> nn.Sequential:
    """Conv 3x3 (P=1, S=1 : conserve H et W) -> BatchNorm -> ReLU.

    bias=False car la BatchNorm qui suit possède déjà un décalage (beta) :
    le biais de la convolution serait redondant.
    """
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class CNNPlain(nn.Module):
    """6 blocs [Conv3x3-BN-ReLU -> MaxPool 2x2] puis Flatten -> Dense -> Dropout -> Dense.

    Avec une entrée 128x128, chaque MaxPool divise la résolution par 2 :
    128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2. Sortie convolutive : 128 x 2 x 2 = 512.
    """

    def __init__(self, num_classes: int = 10, channels=(32, 32, 64, 64, 128, 128),
                 hidden: int = 128, dropout: float = 0.5):
        super().__init__()
        layers, c_in = [], 3
        for c in channels:
            layers += [conv_bn_relu(c_in, c), nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_in * 2 * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),          # régularisation de la tête dense (la plus sujette au sur-apprentissage)
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ResidualBlock(nn.Module):
    """Bloc résiduel « basique » (He et al., 2015) : y = ReLU(F(x) + shortcut(x)).

    F(x) = Conv3x3-BN-ReLU-Conv3x3-BN. Si le nombre de canaux change, le
    raccourci est une convolution 1x1 (+BN) qui projette x à la bonne dimension.
    Le terme « + x » offre au gradient un chemin direct (dérivée de l'identité = 1)
    qui limite l'évanouissement du gradient dans les couches profondes.
    """

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity() if c_in == c_out else nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=1, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class CNNRes(nn.Module):
    """3 ResidualBlocks (= 6 conv 3x3 sur le chemin principal), MaxPool entre blocs, GAP.

    128 -> [RB 3->32] -> pool 64 -> [RB 32->64] -> pool 32 -> [RB 64->128] -> pool 16
    -> Global Average Pooling (128 x 16 x 16 -> 128) -> Dropout -> Linear.
    Le GAP remplace Flatten + Dense : 0 paramètre, invariance à la position, et
    beaucoup moins de sur-apprentissage qu'une grande couche dense.
    """

    def __init__(self, num_classes: int = 10, channels=(32, 64, 128), dropout: float = 0.5):
        super().__init__()
        layers, c_in = [], 3
        for c in channels:
            layers += [ResidualBlock(c_in, c), nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, num_classes))

    def forward(self, x):
        return self.classifier(self.gap(self.features(x)))


def conv_layers(model: nn.Module):
    """Liste ordonnée (nom, couche) des Conv2d 3x3 du chemin principal (pour les hooks de gradient)."""
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3)]
