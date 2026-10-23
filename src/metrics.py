"""Métriques et figures standard (classification, courbes d'apprentissage, anomalies)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, recall_score,
                             roc_auc_score, roc_curve)


def classification_metrics(y_true, y_pred, class_names) -> dict:
    """Accuracy, F1 macro, rappel par classe, matrice de confusion.

    F1 macro = moyenne NON pondérée des F1 par classe : chaque classe compte
    autant, quelle que soit sa taille. Avec des classes déséquilibrées, c'est
    plus informatif que l'accuracy (dominée par les grandes classes).
    """
    labels = list(range(len(class_names)))
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_per_class": {c: float(r) for c, r in zip(class_names, rec)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def plot_confusion(cm, class_names, path: Path | None = None, title: str = "Matrice de confusion (test)"):
    """Matrice normalisée par ligne (= rappel sur la diagonale) avec les effectifs bruts annotés."""
    cm = np.asarray(cm)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(norm, annot=cm, fmt="d", cmap="Blues", vmin=0, vmax=1, cbar_kws={"label": "proportion de la classe réelle"},
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_history(hist: pd.DataFrame, path: Path | None = None, title: str = ""):
    """Courbes perte et accuracy, apprentissage vs validation (diagnostic du sur-apprentissage)."""
    has_acc = "train_acc" in hist and hist["train_acc"].notna().any()
    fig, axes = plt.subplots(1, 2 if has_acc else 1, figsize=(11 if has_acc else 6, 4))
    axes = np.atleast_1d(axes)
    axes[0].plot(hist["epoch"], hist["train_loss"], "o-", label="apprentissage")
    axes[0].plot(hist["epoch"], hist["val_loss"], "s-", label="validation")
    axes[0].set(xlabel="epoch", ylabel="perte", title=f"Perte {title}")
    if has_acc:
        axes[1].plot(hist["epoch"], hist["train_acc"], "o-", label="apprentissage")
        axes[1].plot(hist["epoch"], hist["val_acc"], "s-", label="validation")
        axes[1].set(xlabel="epoch", ylabel="accuracy", title=f"Accuracy {title}", ylim=(0, 1.01))
    for a in axes:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def anomaly_metrics(scores_normal, scores_anomal, threshold: float) -> dict:
    """Détection d'anomalies : score élevé = anormal.

    - AUROC : probabilité qu'une anomalie ait un score plus élevé qu'un cas
      normal ; indépendante du seuil.
    - rappel (taux de détection) des anomalies au seuil choisi ;
    - taux de fausses alarmes : normaux au-dessus du seuil.
    """
    s_n, s_a = np.asarray(scores_normal), np.asarray(scores_anomal)
    y = np.r_[np.zeros(len(s_n)), np.ones(len(s_a))]
    return {
        "auroc": float(roc_auc_score(y, np.r_[s_n, s_a])),
        "recall_anomalies": float((s_a > threshold).mean()),
        "false_alarm_rate": float((s_n > threshold).mean()),
        "threshold": float(threshold),
    }


def plot_roc(curves: dict, path: Path | None = None, title: str = "Courbes ROC"):
    """curves = {nom: (scores_normal, scores_anomal)}."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (s_n, s_a) in curves.items():
        y = np.r_[np.zeros(len(s_n)), np.ones(len(s_a))]
        s = np.r_[s_n, s_a]
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=f"{name} (AUROC={roc_auc_score(y, s):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="hasard")
    ax.set(xlabel="taux de faux positifs", ylabel="taux de vrais positifs", title=title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig
