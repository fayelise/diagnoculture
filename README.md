# DiagnoCulture 🍅

**Système hybride de diagnostic des maladies de la tomate, fiable et conscient de ses limites.**
Projet de fin de module *Apprentissage profond* — Master 2 SID, École Polytechnique de Diamniadio (UAM).

Question de recherche : comment combiner un **classifieur supervisé** (CNN / transfer learning) et un **auto-encodeur non supervisé** pour obtenir un système à la fois **précis**, **léger** (smartphone d'entrée de gamme) et capable de répondre **« inconnu »** face à une maladie jamais vue ou à une image qui n'est pas une feuille de tomate ?

---

## 1. Installation

### En local (Ubuntu, Python ≥ 3.12)
```bash
git clone https://github.com/fayelise/diagnoculture.git
cd diagnoculture
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Un GPU n'est pas indispensable pour le notebook 00 ni pour la démo Gradio ; il l'est pour les entraînements (01 à 06) dans le budget de temps prévu.

### Sur Kaggle Notebooks (recommandé : GPU T4, 30 h/semaine)
Prérequis : compte Kaggle **vérifié par téléphone** (obligatoire pour le GPU et Internet).

Un notebook Kaggle par notebook du projet, enchaînés en « chaîne » : chacun reçoit en entrée l'Output du précédent.

1. **Importer** : *Create → New Notebook*, puis *File → Import Notebook → onglet GitHub* : `fayelise/diagnoculture`, choisir `notebooks/00_eda_split.ipynb`. Le nommer `diagnoculture-00`.
2. **Réglages** (panneau de droite, *Session options*) : *Accelerator* **GPU T4 x2** (inutile pour le 00), *Internet* **On**.
3. **Données** : *Add Input* → rechercher `plantvillage-dataset` (auteur *abdallahalidev*) → **+**.
4. **Résultats du notebook précédent** (à partir du 01) : *Add Input → Your Work* → choisir l'Output de `diagnoculture-<n-1>`. Comme chaque notebook recopie tout ce qu'il a reçu, le dernier de la chaîne contient tout (le 04 dispose donc des résultats de 02 et 03).
5. **Exécuter** : *Save Version → Save & Run All (Commit)*. L'exécution continue même si l'onglet est fermé ; les résultats sont dans l'onglet *Output* de la version.

| Notebook Kaggle | Inputs à ajouter |
|---|---|
| diagnoculture-00 | plantvillage-dataset |
| diagnoculture-01 | plantvillage-dataset + Output de diagnoculture-00 |
| diagnoculture-02 | plantvillage-dataset + Output de diagnoculture-01 |
| diagnoculture-03 | plantvillage-dataset + Output de diagnoculture-02 |
| diagnoculture-04 | plantvillage-dataset + Output de diagnoculture-03 |
| diagnoculture-05 | plantvillage-dataset + Output de diagnoculture-04 |
| diagnoculture-06 | plantvillage-dataset + Output de diagnoculture-05 |

**Photos personnelles (≥ 30)** : créer un dataset Kaggle privé (*Create → New Dataset*, ex. `diagnoculture-ood-custom`) et l'ajouter en Input au notebook 04. Puis, dans `configs/config.yaml`, mettre `ood_custom_dir: /kaggle/input/diagnoculture-ood-custom` (commit + push).

### Sur Google Colab (GPU T4) — sans GitHub
1. Dézipper le projet et déposer le dossier `diagnoculture/` à la racine de **Mon Drive** (→ `MyDrive/diagnoculture/`).
2. Clé Kaggle : kaggle.com → *Settings → API → Create New Token* (télécharge `kaggle.json`). Dans Colab, icône **Secrets** : ajouter `KAGGLE_USERNAME` et `KAGGLE_KEY` (valeurs de `kaggle.json`), activer « Accès au notebook ».
3. Dans Drive, ouvrir `diagnoculture/notebooks/00_eda_split.ipynb` avec **Google Colaboratory**, puis *Exécution → Modifier le type d'exécution → GPU T4*.
4. La première cellule monte le Drive, télécharge PlantVillage sur le disque local de Colab (~2 Go, 1–2 min par session) et place le cache d'images dans `/content/cache`. Résultats et checkpoints sont écrits dans `MyDrive/diagnoculture/results/` : ils survivent aux déconnexions.
5. Photos personnelles : les déposer dans `MyDrive/diagnoculture/data/ood/custom/`.

## 2. Données

Jeu principal : **PlantVillage**, sous-ensemble tomate (10 classes : 9 maladies + saine).
Version recommandée : [`abdallahalidev/plantvillage-dataset`](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) — dossier `color/` (18 160 images tomate). Le code accepte aussi les autres versions Kaggle/Hugging Face (noms de dossiers normalisés automatiquement) ; le nombre exact d'images utilisé est affiché par le notebook 00 et doit être reporté dans le rapport.

**Téléchargement hors Kaggle** — créer un jeton API sur kaggle.com (*Settings → API → Create New Token*), puis :
```bash
# Option 1 : kagglehub (utilisé automatiquement par les notebooks si les données sont absentes)
export KAGGLE_USERNAME=votre_login KAGGLE_KEY=votre_cle      # sur Colab : onglet « Secrets »
python -c "import kagglehub; print(kagglehub.dataset_download('abdallahalidev/plantvillage-dataset'))"

# Option 2 : CLI Kaggle, dans data/raw/ (détecté automatiquement)
pip install kaggle && mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d abdallahalidev/plantvillage-dataset -p data/raw --unzip
```
Le dossier est trouvé automatiquement (`paths.data_root: auto`) ; sinon, renseigner son chemin dans `configs/config.yaml`.

**Jeu hors-distribution (Partie 4)** : déposer **au moins 30 photos personnelles** (mains, sol, autres plantes, feuilles de tomate sur fond non uniforme…) dans `data/ood/custom/`. Les feuilles d'autres espèces PlantVillage sont prélevées automatiquement si la version téléchargée les contient (c'est le cas de la version recommandée).

`data/` n'est pas versionné (volume, licence du jeu d'origine).

## 3. Ordre d'exécution et temps estimés (GPU T4)

| # | Notebook | Contenu (parties de l'énoncé) | Durée estimée |
|---|---|---|---|
| 00 | `00_eda_split.ipynb` | EDA, doublons (pHash), split 70/15/15 par groupe, normalisation — **à exécuter en premier** | 5–10 min (CPU suffit) |
| 01 | `01_cnn_from_scratch.ipynb` | Partie 1 : CNN-Plain, CNN-Res, étude du learning rate, normes de gradient | 25–35 min |
| 02 | `02_transfer_learning.ipynb` | Partie 2 : MobileNetV3-S/L, ResNet-50, 3 stratégies, latence, Pareto, ONNX + quantification | 35–45 min |
| 03 | `03_autoencoders.ipynb` | Partie 3 : AE k ∈ {16, 64, 256}, débruiteur, détection d'anomalies | 15–20 min |
| 04 | `04_hybrid_open_set.ipynb` | Partie 4 : leave-one-class-out, MSP / AE / hybride, OOD (requiert 02 et 03) | 20–25 min |
| 05 | `05_generative_extension.ipynb` | Partie 5 (option B) : VAE, latent, interpolations, ELBO vs AE (requiert 03) | 15–20 min |
| 06 | `06_critical_analysis.ipynb` | Partie 6 : Grad-CAM, analyse d'erreurs, questions guidées (requiert 02) | 5 min |
| — | `app/app_gradio.py` | Démo : image → maladie ou « inconnu » (requiert 04) | — |

Chaque entraînement est plafonné à `train.max_minutes` (30 min) et sauvegarde **à chaque epoch** `last.pt` (reprise) et `history.csv`. Si la session est coupée : relancer simplement la même cellule, l'entraînement reprend à l'epoch suivant.

**Test rapide du pipeline** (avant de lancer les vrais entraînements) : créer un petit YAML, par ex. `quick.yaml` contenant `part1: {epochs: 1}` etc., puis `export DIAGNO_CONFIG_OVERRIDE=quick.yaml` : il est fusionné par-dessus `configs/config.yaml`.

**Garde-fou GPU** : les notebooks 01 à 05 s'arrêtent dès la cellule de configuration s'il n'y a pas de GPU (sur CPU, le budget `train.max_minutes` couperait les entraînements après quelques epochs). Les notebooks 00 et 06 (sans entraînement) acceptent le CPU (`ALLOW_CPU = True`). Pour le seul test rapide ci-dessus, `export DIAGNO_ALLOW_CPU=1` lève le garde-fou.

## 4. Règles de rigueur appliquées dans le code

- **Reproductibilité** : `set_seed(42)` (random, NumPy, PyTorch, cuDNN déterministe), DataLoader à générateur fixé, versions figées (`requirements.txt`) et versions réellement utilisées enregistrées dans `results/env_versions.json`.
- **Anti-fuite** : variante `color` seule ; doublons exacts (MD5) ; quasi-doublons par pHash sur les 8 rotations/symétries + UUID de feuille ; découpage stratifié **par groupe** ; normalisation calculée sur le train.
- **Test intouché** : `TestGuard` (`results/test_log.json`) lève une erreur si un même modèle final est évalué deux fois sur le test ; tous les seuils (MSP, erreur de reconstruction, règle hybride, z-scores) passent par des fonctions qui refusent tout autre ensemble que la validation.
- **Traçabilité** : chaque expérience écrit ses hyperparamètres, courbes, métriques et temps dans `results/<partie>/<expérience>/` et ajoute une ligne au **tableau récapitulatif global `results/summary.csv`**.

## 5. Démonstration

```bash
python app/app_gradio.py            # local : http://127.0.0.1:7860
python app/app_gradio.py --share    # Colab/Kaggle : lien public temporaire
```
L'application charge le système hybride du notebook 04 (classifieur + auto-encodeur + seuils fixés sur la validation) et répond par une maladie ou par **« Inconnu »** avec la raison du rejet. Pour la soutenance : 2–3 images dont une de la classe retirée (*Tomato_mosaic_virus*) et une photo hors-distribution.

## 6. Structure du dépôt

```
diagnoculture/
├── README.md
├── requirements.txt            # versions figées
├── configs/config.yaml         # chemins, seed=42, résolutions, hyperparamètres
├── data/                       # non versionné (raw/, ood/custom/, cache/)
├── notebooks/                  # 00 à 06, à exécuter dans l'ordre
├── src/
│   ├── utils.py                # seed, environnement, chemins, TestGuard, summary.csv
│   ├── datasets.py             # inventaire, pHash/doublons, split par groupe, cache, Datasets
│   ├── models/                 # cnn.py (Plain, Res), transfer.py, autoencoder.py (AE, VAE)
│   ├── train.py                # boucles d'entraînement + checkpoint/reprise + hooks de gradient
│   ├── metrics.py              # accuracy, F1 macro, rappel, matrices, courbes, AUROC
│   ├── latency.py              # paramètres, taille, latence CPU/GPU, ONNX, quantification, Pareto
│   ├── ood.py                  # scores MSP/AE, seuils (validation), z-scores, métriques open-set
│   └── gradcam.py              # Grad-CAM explicite + part d'attention sur la feuille
├── app/app_gradio.py           # démo image -> maladie ou « inconnu »
├── results/                    # sous-dossiers par partie, summary.csv, test_log.json
└── report/                     # plan du rapport, fiche de contribution, déclaration IA
```

Les poids (`*.pt`, `*.onnx`) sont exclus de Git (taille) : les conserver sur Drive / dans l'Output Kaggle, ou utiliser Git LFS pour le modèle final de la démo.

## 7. Usage d'outils d'IA générative

Conformément à l'énoncé, l'usage d'IA générative est déclaré en annexe du rapport (`report/annexe_declaration_ia.md`) : outils, usages, et ce qui a été vérifié.
