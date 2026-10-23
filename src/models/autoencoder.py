"""Parties 3 et 5 — Auto-encodeur convolutif et VAE (images 64x64, pixels dans [0, 1]).

Encodeur : 4 convolutions de pas 2 (64 -> 32 -> 16 -> 8 -> 4) puis une couche
linéaire vers le code latent de dimension k (le « goulot d'étranglement »).
Décodeur : symétrique, avec des ConvTranspose2d de pas 2 (4 -> 8 -> 16 -> 32 -> 64)
et une Sigmoid finale (pixels dans [0, 1], cohérent avec la perte MSE).
"""
import torch
import torch.nn as nn

CH = (32, 64, 128, 256)


def _encoder():
    layers, c_in = [], 3
    for c in CH:  # Conv k=4, S=2, P=1 : sortie = (W - 4 + 2)/2 + 1 = W/2
        layers += [nn.Conv2d(c_in, c, 4, stride=2, padding=1), nn.BatchNorm2d(c), nn.LeakyReLU(0.2, True)]
        c_in = c
    return nn.Sequential(*layers, nn.Flatten())                         # 256 x 4 x 4 = 4096


def _decoder():
    layers = []
    rev = CH[::-1]
    for c_in, c_out in zip(rev[:-1], rev[1:]):  # ConvT k=4, S=2, P=1 : sortie = 2W
        layers += [nn.ConvTranspose2d(c_in, c_out, 4, stride=2, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(True)]
    layers += [nn.ConvTranspose2d(rev[-1], 3, 4, stride=2, padding=1), nn.Sigmoid()]
    return nn.Sequential(*layers)


class ConvAutoencoder(nn.Module):
    """Auto-encodeur convolutif à code latent de dimension k."""

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = _encoder()
        self.to_latent = nn.Linear(CH[-1] * 4 * 4, latent_dim)
        self.from_latent = nn.Linear(latent_dim, CH[-1] * 4 * 4)
        self.decoder = _decoder()

    def encode(self, x):
        return self.to_latent(self.encoder(x))

    def decode(self, z):
        return self.decoder(self.from_latent(z).view(-1, CH[-1], 4, 4))

    def forward(self, x):
        return self.decode(self.encode(x))


class VAE(ConvAutoencoder):
    """Auto-encodeur variationnel (Kingma & Welling, 2013).

    L'encodeur prédit une loi q(z|x) = N(mu, sigma²) ; on échantillonne
    z = mu + sigma * eps (astuce de reparamétrisation, qui rend l'échantillonnage
    dérivable). La perte (-ELBO) = reconstruction + beta * KL(q(z|x) || N(0, I)) ;
    le terme KL régularise l'espace latent, ce qui permet de générer en tirant z ~ N(0, I).
    """

    def __init__(self, latent_dim: int = 32):
        super().__init__(latent_dim)
        self.to_logvar = nn.Linear(CH[-1] * 4 * 4, latent_dim)

    def encode(self, x):
        h = self.encoder(x)
        return self.to_latent(h), self.to_logvar(h)          # (mu, log sigma²)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if self.training else mu
        return self.decode(z), mu, logvar


def vae_terms(x, x_hat, mu, logvar):
    """Termes de -ELBO PAR IMAGE : erreur de reconstruction (somme des carrés) et KL analytique."""
    rec = ((x_hat - x) ** 2).flatten(1).sum(1)
    kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)
    return rec, kl
