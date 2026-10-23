"""Partie 6 — Grad-CAM (Selvaraju et al., 2017), implémentation explicite.

Idée : pour la classe c, on regarde la dernière couche convolutive A (K cartes
de features). Le poids d'importance de chaque carte est la moyenne spatiale du
gradient du score y_c par rapport à cette carte : alpha_k = mean_ij dy_c/dA_k(i,j).
La carte Grad-CAM = ReLU(sum_k alpha_k A_k), remise à la taille de l'image.
ReLU : on ne garde que les zones qui augmentent le score de la classe.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.acts = self.grads = None
        self.handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        """Mémorise les activations A et accroche un hook sur LEUR gradient dy/dA."""
        self.acts = output
        output.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x: torch.Tensor, class_idx=None) -> np.ndarray:
        """x : (N, 3, H, W) normalisé. Retourne des cartes (N, H, W) dans [0, 1]."""
        x = x.requires_grad_(True)  # nécessaire si les premières couches sont gelées
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1)
        class_idx = torch.as_tensor(class_idx, device=logits.device).view(-1)
        self.model.zero_grad(set_to_none=True)
        logits.gather(1, class_idx[:, None]).sum().backward()
        alpha = self.grads.mean(dim=(2, 3), keepdim=True)              # (N, K, 1, 1)
        cam = F.relu((alpha * self.acts).sum(1, keepdim=True))          # (N, 1, h, w)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        return cam.detach().cpu().numpy()

    def remove(self):
        self.handle.remove()


def overlay(img_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Superpose la carte (colormap jet) à l'image RGB uint8 (H, W, 3)."""
    import matplotlib.cm as cm
    heat = cm.jet(cam)[..., :3]
    return ((1 - alpha) * img_uint8 / 255.0 + alpha * heat).clip(0, 1)


def leaf_mask(img_uint8: np.ndarray) -> np.ndarray:
    """Masque approximatif de la feuille (fond PlantVillage gris peu saturé).

    Saturation HSV seuillée par la méthode d'Otsu. Approximation suffisante pour
    mesurer quelle part de l'attention Grad-CAM tombe sur le fond.
    """
    from skimage.color import rgb2hsv
    from skimage.filters import threshold_otsu
    sat = rgb2hsv(img_uint8)[..., 1]
    return sat > threshold_otsu(sat)


def cam_on_leaf_ratio(cam: np.ndarray, mask: np.ndarray) -> float:
    """Part de la « masse » Grad-CAM située sur la feuille (1 = tout sur la feuille)."""
    return float((cam * mask).sum() / max(cam.sum(), 1e-8))
