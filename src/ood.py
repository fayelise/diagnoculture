"""Partie 4 — Rejet de l'inconnu : scores MSP, auto-encodeur, règle hybride, métriques open-set.

Convention : tous les scores sont des scores d'« étrangeté » (plus c'est élevé,
plus l'image est suspecte). Un seuil τ est TOUJOURS fixé sur la validation,
à partir des seules classes CONNUES (la classe M est censée n'avoir jamais été
vue : l'utiliser pour régler τ serait tricher).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .datasets import IMG_EXT
from .utils import assert_validation


def msp_unknown_score(logits: torch.Tensor) -> np.ndarray:
    """1 - MSP (probabilité softmax maximale). MSP faible = classifieur peu sûr."""
    return (1 - torch.softmax(logits.float(), 1).max(1).values).numpy()


def fit_threshold(scores_known_val, target_false_reject: float, split_name: str) -> float:
    """τ tel qu'une fraction `target_false_reject` des connues de VALIDATION soit rejetée.

    Ex. 5 % -> τ = percentile 95 des scores des connues. On fixe ainsi le même
    taux de rejet à tort pour tous les systèmes, ce qui rend la comparaison équitable.
    """
    assert_validation(split_name, "seuil")
    return float(np.percentile(scores_known_val, 100 * (1 - target_false_reject)))


class ZScore:
    """Normalisation z = (s - moyenne) / écart-type, statistiques des connues de VALIDATION.

    Les deux signaux (1-MSP dans [0,1], erreur MSE ~1e-3) ont des échelles très
    différentes : sans normalisation, la somme serait dominée par l'un des deux.
    """

    def fit(self, scores_known_val, split_name: str):
        assert_validation(split_name, "paramètre de normalisation")
        self.mu, self.sd = float(np.mean(scores_known_val)), float(np.std(scores_known_val) + 1e-12)
        return self

    def __call__(self, s):
        return (np.asarray(s) - self.mu) / self.sd


def open_set_metrics(score_known, score_unknown, reject_known, reject_unknown,
                     pred_known, y_known) -> dict:
    """Métriques demandées (Partie 4.4).

    - AUROC connu vs inconnu (indépendante du seuil) ;
    - taux de rejet de M (on le veut élevé) ;
    - taux de rejet à tort des connues (on le veut bas) ;
    - accuracy sur les connues ACCEPTÉES (qualité des diagnostics réellement rendus).
    """
    y = np.r_[np.zeros(len(score_known)), np.ones(len(score_unknown))]
    acc_mask = ~np.asarray(reject_known)
    correct = np.asarray(pred_known) == np.asarray(y_known)
    return {
        "auroc_known_vs_unknown": float(roc_auc_score(y, np.r_[score_known, score_unknown])),
        "reject_rate_unknown": float(np.mean(reject_unknown)),
        "false_reject_rate_known": float(np.mean(reject_known)),
        "accuracy_on_accepted_known": float(correct[acc_mask].mean()) if acc_mask.any() else float("nan"),
        "coverage_known": float(acc_mask.mean()),
    }


def coverage_accuracy(score_known, correct_known):
    """Courbe couverture / précision : on accepte d'abord les images les plus « sûres ».

    Pour chaque couverture c (fraction d'images connues acceptées), accuracy des
    images acceptées. Un bon score de rejet fait monter l'accuracy quand c baisse.
    """
    order = np.argsort(score_known)
    corr = np.asarray(correct_known)[order].astype(float)
    n = np.arange(1, len(corr) + 1)
    return n / len(corr), np.cumsum(corr) / n


def plot_coverage_accuracy(curves: dict, path: Path | None = None):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, (cov, acc) in curves.items():
        ax.plot(cov, acc, label=name)
    ax.set(xlabel="couverture (fraction des connues acceptées)", ylabel="accuracy sur les acceptées",
           title="Courbe couverture / précision (test, classes connues)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def list_images(folder: Path):
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT) if folder.exists() else []


# -----------------------------------------------------------------------------
# Règles de décision (partagées par le notebook 04 et l'application Gradio)
# -----------------------------------------------------------------------------
def fit_rule(kind: str, msp_val, ae_val, target_false_reject: float, split_name: str) -> dict:
    """Fixe une règle de rejet sur les classes CONNUES de validation.

    kind :
      - "msp" : rejet si 1 - MSP > τ ;
      - "ae"  : rejet si erreur de reconstruction > τ ;
      - "or"  : rejet si (1 - MSP > τ₁) OU (erreur > τ₂). Chaque seuil est fixé à
                la moitié du budget de rejet à tort (Bonferroni) pour que le taux
                global reste ≈ target_false_reject ;
      - "z"   : score combiné = moyenne des z-scores des deux signaux, rejet si > τ.
    Tous les systèmes visent le même taux de rejet à tort sur la validation :
    on peut donc comparer leur capacité à rejeter l'inconnu à « coût » égal.
    """
    assert_validation(split_name, "seuil")
    msp_val, ae_val = np.asarray(msp_val), np.asarray(ae_val)
    rule = {"kind": kind, "target_false_reject": target_false_reject}
    if kind == "or":
        rule["tau_msp"] = fit_threshold(msp_val, target_false_reject / 2, split_name)
        rule["tau_ae"] = fit_threshold(ae_val, target_false_reject / 2, split_name)
        rule["tau"] = 1.0
        return rule
    if kind == "z":
        zm, za = ZScore().fit(msp_val, split_name), ZScore().fit(ae_val, split_name)
        rule.update(mu_msp=zm.mu, sd_msp=zm.sd, mu_ae=za.mu, sd_ae=za.sd)
    rule["tau"] = fit_threshold(unknown_score(rule, msp_val, ae_val), target_false_reject, split_name)
    return rule


def unknown_score(rule: dict, msp, ae) -> np.ndarray:
    """Score d'étrangeté continu associé à une règle (rejet si score > rule['tau']).

    Pour la règle « OU », max(s₁/τ₁, s₂/τ₂) > 1 équivaut exactement à
    (s₁ > τ₁ ou s₂ > τ₂) : on obtient un score continu, utilisable pour l'AUROC
    et la courbe couverture/précision.
    """
    msp, ae = np.asarray(msp, dtype=float), np.asarray(ae, dtype=float)
    k = rule["kind"]
    if k == "msp":
        return msp
    if k == "ae":
        return ae
    if k == "or":
        return np.maximum(msp / max(rule["tau_msp"], 1e-12), ae / max(rule["tau_ae"], 1e-12))
    if k == "z":
        return ((msp - rule["mu_msp"]) / rule["sd_msp"] + (ae - rule["mu_ae"]) / rule["sd_ae"]) / 2
    raise ValueError(k)


def reject(rule: dict, msp, ae) -> np.ndarray:
    return unknown_score(rule, msp, ae) > rule["tau"]
