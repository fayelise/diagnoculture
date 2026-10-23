"""Démo DiagnoCulture : photo de feuille -> maladie, ou « Inconnu ».

Charge le système hybride produit par le notebook 04 (results/04_hybrid/decision_rule.json) :
classifieur (9 classes connues) + auto-encodeur + règle de rejet dont les seuils ont été
fixés sur la validation. Aucun réglage n'est fait ici.

Usage :
    python app/app_gradio.py            # http://127.0.0.1:7860
    python app/app_gradio.py --share    # lien public temporaire (Colab / Kaggle)
"""
import argparse
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import build_transform, load_image  # noqa: E402
from src.models import ConvAutoencoder, TransferModel  # noqa: E402
from src.ood import msp_unknown_score, unknown_score  # noqa: E402
from src.utils import get_device, load_config, load_json, resolve_paths  # noqa: E402

LABELS_FR = {
    "Bacterial_spot": "Gale bactérienne", "Early_blight": "Alternariose (brûlure précoce)",
    "Late_blight": "Mildiou", "Leaf_Mold": "Cladosporiose (moisissure des feuilles)",
    "Septoria_leaf_spot": "Septoriose", "Spider_mites": "Acariens (tétranyques)",
    "Target_Spot": "Corynesporiose (tache cible)", "Yellow_Leaf_Curl_Virus": "Virus de l'enroulement jaune (TYLCV)",
    "Tomato_mosaic_virus": "Virus de la mosaïque (ToMV)", "healthy": "Feuille saine",
}


class DiagnoSystem:
    def __init__(self):
        cfg = load_config()
        self.results = resolve_paths(cfg)["results"]
        self.dr = load_json(self.results / "04_hybrid" / "decision_rule.json")
        self.device = get_device()
        c = self.dr["classifier"]
        self.classes = self.dr["known_classes"]
        self.clf = TransferModel(c["backbone"], len(self.classes), pretrained=False)
        self.clf.load_state_dict(torch.load(self.results / c["ckpt"], map_location=self.device, weights_only=True))
        self.clf.to(self.device).eval()
        self.tf_cls = build_transform(False, c["mean"], c["std"])
        self.ae = None
        a = self.dr["autoencoder"]
        if a:
            self.ae = ConvAutoencoder(a["latent_dim"])
            self.ae.load_state_dict(torch.load(self.results / a["ckpt"], map_location=self.device, weights_only=True))
            self.ae.to(self.device).eval()
            self.tf_ae = build_transform(False)

    @staticmethod
    def _tensor(img, size, tf):
        img = load_image(img, size)
        return tf(torch.from_numpy(np.array(img)).permute(2, 0, 1))[None]

    @torch.inference_mode()
    def __call__(self, img):
        if img is None:
            return {}, "Déposez une photo de feuille de tomate."
        c, a, rule = self.dr["classifier"], self.dr["autoencoder"], self.dr["rule"]
        logits = self.clf(self._tensor(img, c["img_size"], self.tf_cls).to(self.device)).float().cpu()
        prob = torch.softmax(logits, 1)[0].numpy()
        s_msp = msp_unknown_score(logits)
        s_ae = np.zeros(1)
        if self.ae is not None:
            x = self._tensor(img, a["img_size"], self.tf_ae).to(self.device)
            s_ae = ((self.ae(x) - x) ** 2).flatten(1).mean(1).cpu().numpy()
        score = float(unknown_score(rule, s_msp, s_ae)[0])
        top = {LABELS_FR.get(k, k): float(p) for k, p in zip(self.classes, prob)}
        best = self.classes[int(prob.argmax())]
        details = (f"Système : `{self.dr['system']}` · score d'étrangeté = {score:.3f} (seuil {rule['tau']:.3f}, "
                   f"fixé sur la validation) · confiance du classifieur = {prob.max():.1%}"
                   + (f" · erreur de reconstruction = {s_ae[0]:.5f}" if self.ae is not None else ""))
        if score > rule["tau"]:
            verdict = ("## ❓ Inconnu\nLe système ne reconnaît pas cette image comme l'une des maladies qu'il connaît "
                       "(maladie nouvelle, autre plante, ou photo inhabituelle). **Consultez un agent agricole.**")
        else:
            verdict = f"## {LABELS_FR.get(best, best)}\nDiagnostic proposé — à confirmer par un agent agricole."
        return top, f"{verdict}\n\n<small>{details}</small>"


def build_demo():
    system = DiagnoSystem()
    return gr.Interface(
        fn=system,
        inputs=gr.Image(type="pil", label="Photo d'une feuille de tomate"),
        outputs=[gr.Label(num_top_classes=3, label="Probabilités du classifieur"), gr.Markdown(label="Décision")],
        title="DiagnoCulture — aide au diagnostic des maladies de la tomate",
        description=(f"Classes connues : {', '.join(LABELS_FR.get(c, c) for c in system.classes)}. "
                     f"Classe retirée de l'entraînement (doit être déclarée « inconnu ») : "
                     f"{LABELS_FR.get(system.dr['unknown_class_held_out'])}."),
        flagging_mode="never",
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="lien public temporaire (Colab/Kaggle)")
    build_demo().launch(share=ap.parse_args().share)
