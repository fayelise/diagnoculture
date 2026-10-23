"""Boucles d'entraînement avec checkpoint à chaque epoch et reprise après coupure.

Chaque expérience vit dans son dossier results/<partie>/<expérience>/ :
  hparams.json  — hyperparamètres
  history.csv   — une ligne par epoch (réécrit à chaque epoch)
  last.pt       — état complet (modèle, optimiseur, scheduler, epoch...) pour reprendre
  best.pt       — poids du meilleur epoch selon la VALIDATION
  curves.png, test_metrics.json, confusion_test.png
Si la session Kaggle/Colab s'interrompt, relancer la même cellule reprend à
l'epoch suivant (cfg.train.resume = true).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from .metrics import classification_metrics, plot_confusion, plot_history
from .models.autoencoder import vae_terms
from .utils import TestGuard, get_device, save_json


# -----------------------------------------------------------------------------
# Checkpoints
# -----------------------------------------------------------------------------
def _save_last(path, model, opt, sched, scaler, epoch, best, train_time, history):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "train_time": train_time,
                "history": history, "rng_torch": torch.get_rng_state(), "rng_np": np.random.get_state()},
               path)


def _try_resume(exp_dir: Path, model, opt, sched, scaler, device):
    """Reprend depuis last.pt s'il existe. Retourne (epoch_suivant, best, temps, historique)."""
    f = exp_dir / "last.pt"
    if not f.exists():
        return 1, None, 0.0, []
    # Chargement sur CPU : load_state_dict recopie ensuite poids et état de l'optimiseur sur le bon
    # device, et l'état du générateur aléatoire DOIT rester un ByteTensor CPU.
    ck = torch.load(f, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    sched.load_state_dict(ck["sched"])
    if ck["scaler"]:   # vide si sauvegardé sans précision mixte (exécution CPU) : rien à restaurer
        scaler.load_state_dict(ck["scaler"])
    torch.set_rng_state(ck["rng_torch"])
    np.random.set_state(ck["rng_np"])
    print(f"Reprise depuis l'epoch {ck['epoch']} (temps déjà écoulé : {ck['train_time'] / 60:.1f} min)")
    return ck["epoch"] + 1, ck["best"], ck["train_time"], ck["history"]


def load_best(model: nn.Module, exp_dir: Path, device=None) -> nn.Module:
    device = device or get_device()
    model.load_state_dict(torch.load(Path(exp_dir) / "best.pt", map_location=device, weights_only=True))
    return model.to(device).eval()


# -----------------------------------------------------------------------------
# Norme du gradient par couche (hooks) — Partie 1.5
# -----------------------------------------------------------------------------
class GradNormRecorder:
    """Enregistre ||dL/dW|| de chaque couche conv via `register_hook` sur ses poids.

    Le hook est appelé par autograd au moment où le gradient du poids est
    calculé. En précision mixte, les gradients sont multipliés par le facteur
    d'échelle du GradScaler : on divise par ce facteur pour retrouver la vraie norme.
    """

    def __init__(self, named_layers, every: int = 20):
        self.every, self.step, self.scale = every, 0, 1.0
        self.records = []           # (step, couche, norme)
        self.handles = [m.weight.register_hook(self._hook(name)) for name, m in named_layers]

    def _hook(self, name):
        def fn(grad):
            if self.step % self.every == 0:
                self.records.append((self.step, name, grad.detach().float().norm().item() / self.scale))
        return fn

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records, columns=["step", "layer", "grad_norm"])

    def remove(self):
        for h in self.handles:
            h.remove()


# -----------------------------------------------------------------------------
# Classification
# -----------------------------------------------------------------------------
@torch.inference_mode()
def predict_logits(model, loader, device=None):
    """Logits et labels sur un loader (mode eval, sans gradient)."""
    device = device or get_device()
    model.eval()
    out, ys = [], []
    for x, y in loader:
        out.append(model(x.to(device, non_blocking=True)).float().cpu())
        ys.append(y)
    return torch.cat(out), torch.cat(ys)


def fit_classifier(model, train_loader, val_loader, exp_dir: Path, epochs: int, param_groups,
                   weight_decay: float = 1e-4, amp: bool = True, max_minutes: float = 30,
                   resume: bool = True, grad_recorder: GradNormRecorder | None = None,
                   hparams: dict | None = None) -> pd.DataFrame:
    """Entraîne un classifieur (entropie croisée, AdamW, cosine LR).

    Sélection du modèle : meilleur F1 macro sur la VALIDATION (jamais le test).
    """
    device = get_device()
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    if hparams:
        save_json(hparams, exp_dir / "hparams.json")
    model.to(device)
    amp = amp and device.type == "cuda"
    opt = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, best, train_time, history = (_try_resume(exp_dir, model, opt, sched, scaler, device)
                                        if resume else (1, None, 0.0, []))
    best = best if best is not None else -1.0
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(start, epochs + 1):
        if train_time > max_minutes * 60:
            print(f"Budget de {max_minutes} min atteint : arrêt propre à l'epoch {epoch - 1}.")
            break
        t0 = time.time()
        model.train()
        tot_loss, correct, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if grad_recorder:
                grad_recorder.scale = scaler.get_scale() if amp else 1.0
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(x)
                loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if grad_recorder:
                grad_recorder.step += 1
            tot_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        sched.step()

        # Validation : sert à suivre le sur-apprentissage et à choisir le meilleur epoch
        logits_v, y_v = predict_logits(model, val_loader, device)
        val_loss = F.cross_entropy(logits_v, y_v).item()
        pred_v = logits_v.argmax(1)
        val_f1 = f1_score(y_v, pred_v, average="macro")
        train_time += time.time() - t0
        row = {"epoch": epoch, "train_loss": tot_loss / n, "train_acc": correct / n, "val_loss": val_loss,
               "val_acc": (pred_v == y_v).float().mean().item(), "val_f1": val_f1,
               "lr": opt.param_groups[-1]["lr"], "epoch_time_s": time.time() - t0}
        history.append(row)
        pd.DataFrame(history).to_csv(exp_dir / "history.csv", index=False)
        if val_f1 > best:
            best = val_f1
            torch.save(model.state_dict(), exp_dir / "best.pt")
        _save_last(exp_dir / "last.pt", model, opt, sched, scaler, epoch, best, train_time, history)
        print(f"[{exp_dir.name}] epoch {epoch:2d}/{epochs} | loss {row['train_loss']:.3f}/{val_loss:.3f} "
              f"| acc {row['train_acc']:.3f}/{row['val_acc']:.3f} | F1 val {val_f1:.3f} | {row['epoch_time_s']:.0f}s")

    hist = pd.DataFrame(history)
    save_json({"train_time_s": train_time, "best_val_f1": best,
               "best_epoch": int(hist.loc[hist["val_f1"].idxmax(), "epoch"])}, exp_dir / "train_info.json")
    plot_history(hist, exp_dir / "curves.png", title=exp_dir.name)
    return hist


def final_test_evaluation(model, test_loader, class_names, exp_dir: Path, exp_name: str,
                          guard: TestGuard, force: bool = False) -> dict:
    """UNIQUE évaluation sur le test d'un modèle final (enregistrée par le TestGuard).

    Aucun choix (hyperparamètre, epoch, seuil) ne doit être fait après cet appel.
    """
    done = Path(exp_dir) / "test_metrics.json"
    if exp_name in guard.log and done.exists() and not force:
        # Ré-exécution du notebook (ex. après une coupure) : le test a déjà été utilisé pour ce
        # modèle -> on relit le résultat enregistré au lieu de le recalculer.
        print(f"[{exp_name}] déjà évalué sur le test : résultat relu depuis {done.name}")
        import json
        return json.loads(done.read_text())
    guard.register(exp_name, force=force)
    logits, y = predict_logits(model, test_loader)
    prob = torch.softmax(logits, 1)
    m = classification_metrics(y.numpy(), logits.argmax(1).numpy(), class_names)
    save_json(m, Path(exp_dir) / "test_metrics.json")
    # Prédictions conservées pour l'analyse d'erreurs (notebook 06) sans ré-évaluer le test
    pd.DataFrame({"y_true": y.numpy(), "y_pred": prob.argmax(1).numpy(),
                  "confidence": prob.max(1).values.numpy()}).to_csv(Path(exp_dir) / "test_predictions.csv", index=False)
    plot_confusion(m["confusion_matrix"], class_names, Path(exp_dir) / "confusion_test.png",
                   title=f"{exp_name} — test")
    return m


# -----------------------------------------------------------------------------
# Reconstruction (auto-encodeur, auto-encodeur débruiteur, VAE)
# -----------------------------------------------------------------------------
def _recon_loss(model, x, kind: str, beta: float, noise_std: float):
    """Perte d'un batch. Débruiteur : entrée bruitée, CIBLE PROPRE."""
    if kind == "vae":
        x_hat, mu, logvar = model(x)
        rec, kl = vae_terms(x, x_hat, mu, logvar)
        return (rec + beta * kl).mean()
    x_in = (x + noise_std * torch.randn_like(x)).clamp(0, 1) if noise_std > 0 and model.training else x
    return F.mse_loss(model(x_in), x)


def fit_reconstruction(model, train_loader, val_loader, exp_dir: Path, epochs: int, lr: float,
                       kind: str = "ae", noise_std: float = 0.0, beta: float = 1.0, amp: bool = True,
                       max_minutes: float = 30, resume: bool = True, hparams: dict | None = None) -> pd.DataFrame:
    """Entraîne un AE (kind='ae') ou un VAE (kind='vae'). Meilleur epoch = perte de validation minimale.

    La validation est toujours calculée sur des images PROPRES : c'est ce qui
    sera utilisé pour la détection d'anomalies.
    """
    device = get_device()
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    if hparams:
        save_json(hparams, exp_dir / "hparams.json")
    model.to(device)
    amp = amp and device.type == "cuda" and kind == "ae"   # VAE en float32 (exp(logvar) instable en fp16)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, best, train_time, history = (_try_resume(exp_dir, model, opt, sched, scaler, device)
                                        if resume else (1, None, 0.0, []))
    best = best if best is not None else float("inf")

    for epoch in range(start, epochs + 1):
        if train_time > max_minutes * 60:
            print(f"Budget de {max_minutes} min atteint : arrêt propre à l'epoch {epoch - 1}.")
            break
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = _recon_loss(model, x, kind, beta, noise_std)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(x)
            n += len(x)
        sched.step()
        model.eval()
        vtot, vn = 0.0, 0
        with torch.inference_mode():
            for x, _ in val_loader:
                x = x.to(device, non_blocking=True)
                vtot += _recon_loss(model, x, kind, beta, 0.0).item() * len(x)
                vn += len(x)
        val_loss = vtot / vn
        train_time += time.time() - t0
        history.append({"epoch": epoch, "train_loss": tot / n, "val_loss": val_loss, "epoch_time_s": time.time() - t0})
        pd.DataFrame(history).to_csv(exp_dir / "history.csv", index=False)
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), exp_dir / "best.pt")
        _save_last(exp_dir / "last.pt", model, opt, sched, scaler, epoch, best, train_time, history)
        print(f"[{exp_dir.name}] epoch {epoch:2d}/{epochs} | loss {tot / n:.5f} / val {val_loss:.5f} "
              f"| {time.time() - t0:.0f}s")

    hist = pd.DataFrame(history)
    save_json({"train_time_s": train_time, "best_val_loss": best,
               "best_epoch": int(hist.loc[hist["val_loss"].idxmin(), "epoch"])}, exp_dir / "train_info.json")
    plot_history(hist, exp_dir / "curves.png", title=exp_dir.name)
    return hist


@torch.inference_mode()
def reconstruction_scores(model, loader, kind: str = "ae", device=None) -> dict:
    """Score d'anomalie par image.

    AE  : 'mse'  = ||x - x̂||² / nb de pixels (moyenne par pixel).
    VAE : 'rec' (somme des carrés), 'kl', et 'neg_elbo' = rec + kl (borne de -log p(x)).
    """
    device = device or get_device()
    model.eval()
    out = {}
    for x, _ in loader:
        x = x.to(device)
        if kind == "vae":
            x_hat, mu, logvar = model(x)
            rec, kl = vae_terms(x, x_hat, mu, logvar)
            batch = {"rec": rec, "kl": kl, "neg_elbo": rec + kl}
        else:
            batch = {"mse": ((model(x) - x) ** 2).flatten(1).mean(1)}
        for k, v in batch.items():
            out.setdefault(k, []).append(v.float().cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


# -----------------------------------------------------------------------------
# Tableau récapitulatif
# -----------------------------------------------------------------------------
def record_experiment(results_dir: Path, exp_name: str, part: str, model, exp_dir: Path, img_size: int,
                      test_metrics: dict | None = None, hparams: dict | None = None, **extra) -> pd.DataFrame:
    """Ajoute la ligne de l'expérience à results/summary.csv (à partir des fichiers de exp_dir)."""
    from .latency import count_params, model_size_mb
    from .utils import load_json, update_summary
    exp_dir = Path(exp_dir)
    info = load_json(exp_dir / "train_info.json")
    hist = pd.read_csv(exp_dir / "history.csv")
    best = hist.loc[hist["epoch"] == info["best_epoch"]].iloc[0]
    row = {"experiment": exp_name, "part": part, "model": type(model).__name__ if not hasattr(model, "name")
           else model.name, "params": count_params(model), "size_mb": round(model_size_mb(model), 3),
           "img_size": img_size, "epochs_run": len(hist), "best_epoch": info["best_epoch"],
           "train_time_min": round(info["train_time_s"] / 60, 2),
           "val_acc": best.get("val_acc"), "val_f1_macro": best.get("val_f1"), "hparams": hparams or {}}
    if test_metrics:
        row.update(test_acc=test_metrics["accuracy"], test_f1_macro=test_metrics["f1_macro"])
    row.update(extra)
    return update_summary(results_dir, row)
