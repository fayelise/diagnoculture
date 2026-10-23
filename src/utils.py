"""Utilitaires transverses : reproductibilité, environnement, chemins, journalisation.

Tout ce qui garantit la rigueur expérimentale (graine, garde-fou « test une seule
fois », tableau récapitulatif) est centralisé ici pour être identique dans tous
les notebooks.
"""
from __future__ import annotations

import json
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]  # racine du dépôt


# -----------------------------------------------------------------------------
# Reproductibilité
# -----------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Fixe toutes les sources d'aléa (Python, NumPy, PyTorch CPU/GPU).

    cudnn.deterministic=True force des algorithmes de convolution déterministes ;
    cudnn.benchmark=False empêche cuDNN de choisir l'algorithme le plus rapide
    (choix qui peut varier d'une exécution à l'autre). On perd un peu de vitesse,
    on gagne des résultats reproductibles (exigence de l'énoncé, pénalité -1 sinon).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Graine des workers du DataLoader (augmentations reproductibles)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = 42) -> torch.Generator:
    """Générateur dédié au DataLoader (ordre de mélange reproductible)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# -----------------------------------------------------------------------------
# Environnement d'exécution
# -----------------------------------------------------------------------------
def detect_env() -> str:
    """Retourne 'colab', 'kaggle' ou 'local' (Colab testé en premier : kagglehub peut y créer /kaggle/input)."""
    if "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG"):
        return "colab"
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/input").exists():
        return "kaggle"
    return "local"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def machine_info() -> dict:
    """Description de la machine (à citer dans le rapport pour les latences)."""
    info = {
        "env": detect_env(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    return info


# -----------------------------------------------------------------------------
# Configuration et chemins
# -----------------------------------------------------------------------------
def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path | None = None) -> dict:
    """Charge configs/config.yaml.

    La variable d'environnement DIAGNO_CONFIG_OVERRIDE peut pointer vers un
    second YAML (ex. epochs=1) fusionné par-dessus : pratique pour un test
    rapide du pipeline avant de lancer les vrais entraînements.
    """
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    override = os.environ.get("DIAGNO_CONFIG_OVERRIDE")
    if override:
        with open(override, encoding="utf-8") as f:
            _deep_update(cfg, yaml.safe_load(f))
    if cfg["data"]["num_workers"] == "auto":
        cfg["data"]["num_workers"] = min(4, os.cpu_count() or 1)
    return cfg


def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def _search_tomato_dir(base: Path, cfg: dict) -> Path | None:
    """Premier dossier sous `base` contenant >= 5 sous-dossiers « Tomato* » (variante color préférée)."""
    excluded = set(cfg["data"].get("exclude_variants", []))
    keep = cfg["data"].get("keep_variant")
    best = None
    for dirpath, dirnames, _ in os.walk(base, followlinks=True):
        if set(Path(dirpath).parts) & excluded:
            dirnames[:] = []
            continue
        if sum(d.lower().startswith("tomato") for d in dirnames) >= 5:
            if best is None or (keep and keep in Path(dirpath).parts):
                best = Path(dirpath)
            dirnames[:] = []  # inutile de descendre plus bas
    return best


def find_data_root(cfg: dict) -> Path:
    """Localise le dossier PlantVillage contenant les sous-dossiers de classes tomate.

    On cherche un dossier qui contient au moins 5 sous-dossiers dont le nom
    commence par « Tomato ». Les variantes grayscale/segmented sont ignorées.
    """
    explicit = cfg["paths"]["data_root"]
    if explicit and explicit != "auto":
        return _abs(explicit)
    candidates = [ROOT / "data" / "raw", Path("/kaggle/input"), Path("/content/data"),
                  Path.home() / ".cache" / "kagglehub" / "datasets"]
    for base in candidates:
        if base.exists() and (found := _search_tomato_dir(base, cfg)):
            return found
    raise FileNotFoundError(
        "Dossier PlantVillage introuvable. Voir le README (section Données) "
        "ou renseigner paths.data_root dans configs/config.yaml.")


def ensure_data(cfg: dict) -> Path:
    """Retourne le dossier des données ; sur Colab/local, le télécharge via kagglehub si absent.

    Sur Kaggle, il suffit d'ajouter le dataset avec « Add Input » (aucun téléchargement).
    kagglehub nécessite les identifiants Kaggle (voir README).
    """
    try:
        return find_data_root(cfg)
    except FileNotFoundError:
        if detect_env() == "kaggle":
            raise
    import kagglehub
    print(f"Téléchargement de {cfg['paths']['kaggle_dataset']} via kagglehub (~2 Go, quelques minutes)...")
    dl = Path(kagglehub.dataset_download(cfg["paths"]["kaggle_dataset"]))
    print("Données téléchargées dans :", dl)
    found = _search_tomato_dir(dl, cfg)
    if found is None:
        content = sorted(p.name for p in dl.iterdir())[:20] if dl.is_dir() else dl
        raise FileNotFoundError(f"Aucun dossier « Tomato* » trouvé dans {dl}. Contenu : {content}")
    return found


def resolve_paths(cfg: dict) -> dict:
    """Construit les chemins absolus utilisés par les notebooks.

    Le cache d'images peut être déplacé sur un disque local rapide via la
    variable DIAGNO_CACHE (utile sur Colab quand le dépôt est sur Google Drive).
    """
    results = _abs(cfg["paths"]["results_dir"])
    paths = {
        "root": ROOT,
        "results": results,
        "split_dir": results / "00_split",   # CSV du découpage (versionnés dans Git)
        "cache": Path(os.environ.get("DIAGNO_CACHE", _abs(cfg["paths"]["cache_dir"]))),
        "ood_custom": _abs(cfg["paths"]["ood_custom_dir"]),
    }
    for k in ("results", "split_dir", "cache"):
        paths[k].mkdir(parents=True, exist_ok=True)
    return paths


# -----------------------------------------------------------------------------
# Garde-fou : le jeu de test n'est utilisé qu'UNE fois par modèle final
# -----------------------------------------------------------------------------
class TestGuard:
    """Registre persistant des évaluations sur le test (results/test_log.json).

    Chaque modèle final ne doit être évalué qu'une seule fois sur le test : un
    second appel lève une erreur. Cela matérialise la règle de l'énoncé (pénalité
    -2 si le test sert à choisir un hyperparamètre ou un seuil). Pour ré-exécuter
    volontairement un notebook complet (ex. après une coupure), passer
    force=True : l'évaluation est alors tracée comme « re-run » dans le journal.
    """

    def __init__(self, results_dir: Path):
        self.path = Path(results_dir) / "test_log.json"
        self.log = json.loads(self.path.read_text()) if self.path.exists() else {}

    def register(self, exp_name: str, force: bool = False) -> None:
        if exp_name in self.log and not force:
            raise RuntimeError(
                f"[TEST DÉJÀ UTILISÉ] '{exp_name}' a déjà été évalué sur le test "
                f"({self.log[exp_name]['count']} fois). Le test ne doit servir qu'une "
                "fois par modèle final. Utilisez force=True uniquement pour une "
                "ré-exécution identique (aucun choix ne doit en découler).")
        entry = self.log.get(exp_name, {"count": 0})
        entry["count"] += 1
        entry["last"] = pd.Timestamp.now().isoformat(timespec="seconds")
        self.log[exp_name] = entry
        self.path.write_text(json.dumps(self.log, indent=2))


def assert_validation(split_name: str, what: str = "seuil") -> None:
    """Garde-fou : un seuil / hyperparamètre ne se fixe QUE sur la validation."""
    if split_name != "val":
        raise ValueError(f"Un {what} doit être fixé sur la validation, pas sur '{split_name}'.")


# -----------------------------------------------------------------------------
# Journalisation des résultats
# -----------------------------------------------------------------------------
def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float))


def load_json(path: Path):
    return json.loads(Path(path).read_text())


SUMMARY_COLUMNS = ["experiment", "part", "model", "params", "size_mb", "img_size",
                   "epochs_run", "best_epoch", "train_time_min", "val_acc", "val_f1_macro",
                   "test_acc", "test_f1_macro", "latency_cpu_ms", "latency_gpu_ms",
                   "extra_metric", "extra_value", "notes", "hparams"]


def update_summary(results_dir: Path, row: dict) -> pd.DataFrame:
    """Ajoute (ou remplace) la ligne d'une expérience dans results/summary.csv.

    Une ligne = une expérience : c'est le « tableau récapitulatif global »
    demandé dans les livrables. La clé est le nom d'expérience, donc relancer
    une expérience met sa ligne à jour au lieu de créer un doublon.
    """
    path = Path(results_dir) / "summary.csv"
    df = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=SUMMARY_COLUMNS)
    row = {k: row.get(k) for k in SUMMARY_COLUMNS}
    if isinstance(row["hparams"], dict):
        row["hparams"] = json.dumps(row["hparams"], default=str)
    df = df[df["experiment"] != row["experiment"]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)
    return df
