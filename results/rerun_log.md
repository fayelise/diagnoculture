# Journal des ré-exécutions

## 2026-10-23 21:48 — Ré-exécution des notebooks 01, 04, 05 et 06

**Cause.** Lors de la 1re exécution (Colab, 21–22/10/2026), ces notebooks ont tourné **sans GPU**
(`device : cpu`, `torch +cpu`). Une epoch y est 40 à 80 fois plus lente, et le garde-fou
`train.max_minutes = 30` a arrêté plusieurs entraînements prématurément : CNN-Plain 4/15 epochs,
CNN-Res 1/15, étude du LR à 1e-4 4/8, classifieur 9 classes (P4) 1/6, AE classes connues 15/25,
VAE toutes classes 14/30. Ces résultats ne reflètent pas les modèles et ont été supprimés.

**Ce qui a été supprimé.**
- `results/01_cnn_scratch/lr_study_0.001`
- `results/01_cnn_scratch/lr_study_0.0001`
- `results/01_cnn_scratch/cnn-plain`
- `results/01_cnn_scratch/cnn-res`
- `results/04_hybrid/ae_known_classes`
- `results/04_hybrid/classifier_without_Tomato_mosaic_virus`
- `results/04_hybrid/coverage_accuracy.png`
- `results/04_hybrid/decision_rule.json`
- `results/04_hybrid/ood_results.csv`
- `results/04_hybrid/open_set_results_test.csv`
- `results/04_hybrid/reject_rate_per_known_class.csv`
- `results/04_hybrid/rules_fitted_on_val.json`
- `results/04_hybrid/score_distributions.png`
- `results/05_vae/vae_all_classes`

**Entrées retirées du journal du test** (`test_log.json`) : `P1_CNN-Plain`, `P1_CNN-Res`, `P4_open_set_systems`, `P5_vae_healthy_neg_elbo`.
Lignes retirées de `summary.csv` : `P1_lr_study_0.001`, `P1_lr_study_0.0001`, `P1_CNN-Plain`, `P1_CNN-Res`, `P4_ae_known_classes`, `P4_classifier_without_Tomato_mosaic_virus`, `P4_system_msp`, `P4_system_ae_healthy`, `P4_system_ae_known`, `P4_system_or_healthy`, `P4_system_z_healthy`, `P4_system_or_known`, `P4_system_z_known`, `P5_vae_all_classes`.

**Protocole.** Les modèles sont ré-entraînés à l'identique (même configuration, même graine) sur GPU.
Aucun hyperparamètre, seuil ni modèle n'a été choisi d'après les résultats de test de la 1re exécution.
Le VAE-sain (P5), entraîné jusqu'au bout (30/30 epochs), est inchangé : sa ré-évaluation sur le test
est une exécution identique du même modèle, sans aucun choix qui en découle.
Les parties 00, 02 et 03 (exécutées sur GPU, ou entraînements complets) ne sont pas modifiées.
