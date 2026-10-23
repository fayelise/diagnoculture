"""Supprime les expériences invalidées par la 1re exécution (CPU + budget de 30 min).

Lors de la 1re exécution sur Colab, les notebooks 01, 04, 05 et 06 ont tourné sans GPU :
le garde-fou train.max_minutes = 30 a arrêté plusieurs entraînements après 1 à 4 epochs.
Ce script remet ces expériences à zéro pour qu'elles soient relancées sur GPU :
  - suppression de leurs dossiers (sinon last.pt fait reprendre un entraînement
    dont le budget de temps est déjà épuisé) ;
  - retrait de leurs entrées dans results/test_log.json (sinon final_test_evaluation
    relit l'ancien test_metrics.json et le TestGuard refuse la nouvelle évaluation) ;
  - retrait de leurs lignes dans results/summary.csv ;
  - trace datée dans results/rerun_log.md (à citer dans le rapport).
Les parties 00, 02 et 03 ne sont pas touchées.

Usage (depuis la racine du dépôt) :
  python scripts/clean_invalid_runs.py           # simulation : affiche ce qui serait fait
  python scripts/clean_invalid_runs.py --apply   # exécute, après confirmation
"""
import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"

DIRS = [RESULTS / "01_cnn_scratch" / d for d in ("lr_study_0.001", "lr_study_0.0001", "cnn-plain", "cnn-res")]
DIRS += sorted(p for p in (RESULTS / "04_hybrid").iterdir() if p.name != ".gitkeep")
DIRS += [RESULTS / "05_vae" / "vae_all_classes"]

TEST_LOG_KEYS = ["P1_CNN-Plain", "P1_CNN-Res", "P4_open_set_systems", "P5_vae_healthy_neg_elbo"]


def summary_row_is_invalid(experiment: str) -> bool:
    return (experiment in {"P1_lr_study_0.001", "P1_lr_study_0.0001", "P1_CNN-Plain", "P1_CNN-Res",
                           "P5_vae_all_classes"}
            or experiment.startswith("P4_"))


LOG_ENTRY = """
## {date} — Ré-exécution des notebooks 01, 04, 05 et 06

**Cause.** Lors de la 1re exécution (Colab, 21–22/10/2026), ces notebooks ont tourné **sans GPU**
(`device : cpu`, `torch +cpu`). Une epoch y est 40 à 80 fois plus lente, et le garde-fou
`train.max_minutes = 30` a arrêté plusieurs entraînements prématurément : CNN-Plain 4/15 epochs,
CNN-Res 1/15, étude du LR à 1e-4 4/8, classifieur 9 classes (P4) 1/6, AE classes connues 15/25,
VAE toutes classes 14/30. Ces résultats ne reflètent pas les modèles et ont été supprimés.

**Ce qui a été supprimé.**
{dirs}

**Entrées retirées du journal du test** (`test_log.json`) : {keys}.
Lignes retirées de `summary.csv` : {rows}.

**Protocole.** Les modèles sont ré-entraînés à l'identique (même configuration, même graine) sur GPU.
Aucun hyperparamètre, seuil ni modèle n'a été choisi d'après les résultats de test de la 1re exécution.
Le VAE-sain (P5), entraîné jusqu'au bout (30/30 epochs), est inchangé : sa ré-évaluation sur le test
est une exécution identique du même modèle, sans aucun choix qui en découle.
Les parties 00, 02 et 03 (exécutées sur GPU, ou entraînements complets) ne sont pas modifiées.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="exécuter (par défaut : simulation)")
    args = ap.parse_args()

    dirs = [d for d in DIRS if d.exists()]
    log_path = RESULTS / "test_log.json"
    test_log = json.loads(log_path.read_text())
    keys = [k for k in TEST_LOG_KEYS if k in test_log]
    summary_path = RESULTS / "summary.csv"
    with open(summary_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    removed_rows = [r[0] for r in body if summary_row_is_invalid(r[0])]

    print("Dossiers / fichiers à supprimer :")
    for d in dirs:
        print("  -", d.relative_to(RESULTS.parent))
    print("Entrées à retirer de test_log.json :", keys)
    print("Lignes à retirer de summary.csv :", removed_rows)

    if not args.apply:
        print("\nSimulation uniquement : rien n'a été modifié. Relancer avec --apply pour exécuter.")
        return
    if input("\nConfirmer la suppression ? (oui/non) ").strip().lower() != "oui":
        print("Annulé.")
        return

    for d in dirs:
        shutil.rmtree(d) if d.is_dir() else d.unlink()
    for k in keys:
        del test_log[k]
    log_path.write_text(json.dumps(test_log, indent=2))
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        # même format que pandas.to_csv (guillemets minimaux, fin de ligne \n)
        csv.writer(f, lineterminator="\n").writerows([header] + [r for r in body if not summary_row_is_invalid(r[0])])
    rerun_log = RESULTS / "rerun_log.md"
    with open(rerun_log, "a", encoding="utf-8") as f:
        if f.tell() == 0:
            f.write("# Journal des ré-exécutions\n")
        f.write(LOG_ENTRY.format(date=datetime.now().strftime("%Y-%m-%d %H:%M"),
                                 dirs="\n".join(f"- `{d.relative_to(RESULTS.parent)}`" for d in dirs),
                                 keys=", ".join(f"`{k}`" for k in keys),
                                 rows=", ".join(f"`{r}`" for r in removed_rows)))
    print("Terminé. Trace ajoutée dans results/rerun_log.md.")


if __name__ == "__main__":
    main()
