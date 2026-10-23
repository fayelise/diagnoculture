"""Données : inventaire PlantVillage, détection des quasi-doublons, découpage
anti-fuite, statistiques de normalisation, Datasets PyTorch.

Principe anti-fuite : on ne découpe pas image par image mais GROUPE par groupe,
un groupe réunissant toutes les images d'une même feuille (variantes augmentées,
quasi-doublons). Une feuille ne peut donc pas se retrouver à la fois en
apprentissage et en test.
"""
from __future__ import annotations

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.fft import dct
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from .utils import make_generator, seed_worker

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Noms canoniques des 10 classes (indépendants de la version téléchargée :
# "Tomato___Early_blight", "Tomato_Early_blight", "Tomato__Tomato_mosaic_virus"...).
CLASS_KEYWORDS = [
    ("bacterial", "Bacterial_spot"),
    ("early", "Early_blight"),
    ("late", "Late_blight"),
    ("mold", "Leaf_Mold"),
    ("septoria", "Septoria_leaf_spot"),
    ("spider", "Spider_mites"),
    ("target", "Target_Spot"),
    ("curl", "Yellow_Leaf_Curl_Virus"),
    ("mosaic", "Tomato_mosaic_virus"),
    ("healthy", "healthy"),
]
CLASSES = [c for _, c in CLASS_KEYWORDS]
HEALTHY = "healthy"


def canonical_class(folder_name: str) -> str | None:
    name = folder_name.lower()
    if not name.startswith("tomato"):
        return None
    for kw, canon in CLASS_KEYWORDS:
        if kw in name:
            return canon
    return None


# -----------------------------------------------------------------------------
# Inventaire
# -----------------------------------------------------------------------------
def _iter_images(folder: Path, excluded: set[str]):
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in IMG_EXT and not (set(p.parts) & excluded):
            yield p


def scan_dataset(data_root: Path, cfg: dict) -> pd.DataFrame:
    """Liste toutes les images tomate : relpath (relatif à data_root), classe, label.

    Les chemins sont RELATIFS à data_root : les CSV de split restent valables
    que les données soient dans /kaggle/input/... ou /content/data/...
    """
    excluded = set(cfg["data"].get("exclude_variants", []))
    rows = []
    for d in sorted(p for p in Path(data_root).iterdir() if p.is_dir()):
        cls = canonical_class(d.name)
        if cls is None:
            continue
        for p in _iter_images(d, excluded):
            rows.append({"relpath": str(p.relative_to(data_root)), "class_name": cls})
    df = pd.DataFrame(rows)
    df["label"] = df["class_name"].map({c: i for i, c in enumerate(CLASSES)})
    n_sub = cfg["data"].get("subsample_per_class")
    if n_sub:
        # sous-échantillonnage aléatoire mais reproductible (graine fixée)
        df = df.sample(frac=1, random_state=cfg["seed"]).groupby("class_name").head(n_sub)
    return df.sort_values("relpath").reset_index(drop=True)


def scan_other_species(data_root: Path, cfg: dict, max_n: int) -> pd.DataFrame:
    """Images PlantVillage d'AUTRES espèces (pomme, poivron, pomme de terre...) : jeu OOD."""
    excluded = set(cfg["data"].get("exclude_variants", []))
    folders = [d for d in sorted(Path(data_root).iterdir())
               if d.is_dir() and not d.name.lower().startswith("tomato")]
    rows = [{"relpath": str(p.relative_to(data_root)), "class_name": d.name}
            for d in folders for p in _iter_images(d, excluded)]
    df = pd.DataFrame(rows, columns=["relpath", "class_name"])
    if len(df) > max_n:
        df = df.sample(max_n, random_state=cfg["seed"])
    return df.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Lecture d'image
# -----------------------------------------------------------------------------
def load_image(src, size: int | None = None) -> Image.Image:
    """Ouvre (chemin ou image PIL) en RGB, recadre au carré central (photos smartphone), redimensionne.

    Même prétraitement à l'entraînement, à l'évaluation et dans la démo Gradio.
    """
    img = (src if isinstance(src, Image.Image) else Image.open(src)).convert("RGB")
    w, h = img.size
    if w != h:
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    if size:
        img = img.resize((size, size), Image.BILINEAR)
    return img


# -----------------------------------------------------------------------------
# Détection des quasi-doublons (perceptual hashing)
# -----------------------------------------------------------------------------
def _phash_bits(gray32: np.ndarray, hash_size: int = 8) -> np.ndarray:
    """pHash (même algorithme qu'imagehash.phash) : DCT 2D de l'image 32x32 en
    niveaux de gris, on garde les 8x8 basses fréquences et on compare à la médiane."""
    coeffs = dct(dct(gray32, axis=0, norm=None), axis=1, norm=None)
    low = coeffs[:hash_size, :hash_size]
    return (low > np.median(low)).flatten()


def _pack(bits: np.ndarray) -> np.uint64:
    return np.uint64(int("".join("1" if b else "0" for b in bits), 2))


def image_hashes(path: Path, hash_size: int = 8, dihedral: bool = True):
    """Retourne (md5 du fichier, 8 pHash : identité + 3 rotations + 4 symétries).

    Les jeux PlantVillage « augmentés » contiennent des rotations/flips d'une même
    feuille ; le pHash n'y est pas invariant, d'où le calcul sur les 8 transformations
    du groupe diédral. Tourner l'image 32x32 réduite équivaut à tourner l'image
    d'origine (exact pour les multiples de 90° et les symétries).
    """
    raw = Path(path).read_bytes()
    md5 = hashlib.md5(raw).hexdigest()
    img = Image.open(path).convert("L").resize((hash_size * 4, hash_size * 4), Image.LANCZOS)
    g = np.asarray(img, dtype=np.float64)
    variants = [g]
    if dihedral:
        variants = [np.rot90(g, k) for k in range(4)] + [np.rot90(g.T, k) for k in range(4)]
    return md5, [_pack(_phash_bits(v, hash_size)) for v in variants]


def compute_hashes(df: pd.DataFrame, data_root: Path, hash_size: int = 8,
                   dihedral: bool = True, n_threads: int = 8):
    paths = [Path(data_root) / r for r in df["relpath"]]
    with ThreadPoolExecutor(n_threads) as ex:
        out = list(ex.map(lambda p: image_hashes(p, hash_size, dihedral), paths))
    md5 = np.array([o[0] for o in out])
    H = np.array([o[1] for o in out], dtype=np.uint64)  # (N, 8) ; colonne 0 = identité
    return md5, H


def popcount64(x: np.ndarray) -> np.ndarray:
    """Nombre de bits à 1 (vectorisé) : distance de Hamming = popcount(a XOR b)."""
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int16)


class UnionFind:
    """Structure classique pour fusionner des paires en groupes connexes."""

    def __init__(self, n):
        self.parent = np.arange(n)

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)


UUID_RE = re.compile(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", re.I)


def leaf_uuid(relpath: str) -> str | None:
    """Identifiant de feuille PlantVillage contenu dans le nom de fichier (s'il existe)."""
    m = UUID_RE.search(Path(relpath).name)
    return m.group(1).lower() if m else None


def group_near_duplicates(df: pd.DataFrame, H: np.ndarray, threshold: int, block: int = 256):
    """Regroupe les images d'une même feuille.

    Deux images sont liées si (a) leurs noms de fichier partagent le même UUID
    PlantVillage (variantes augmentées nommées « <uuid>___..._flipLR ») ou (b) la
    distance de Hamming minimale entre leurs pHash (sur les 8 transformations) est
    <= threshold. Les comparaisons se font à l'intérieur de chaque classe (une
    même feuille n'a qu'un diagnostic). Retourne (group_id, distance au plus
    proche voisin) ; la seconde sert à calibrer le seuil dans le notebook 00.
    """
    assert (df.index == np.arange(len(df))).all(), "df doit avoir un index 0..N-1 (reset_index)"
    n = len(df)
    uf = UnionFind(n)
    nn_dist = np.full(n, 64, dtype=np.int16)

    # (a) même UUID dans le nom de fichier
    uuids = df["relpath"].map(leaf_uuid)
    for _, idx in df.groupby(uuids).groups.items():  # les None sont ignorés par groupby
        idx = list(idx)
        for j in idx[1:]:
            uf.union(idx[0], j)

    # (b) quasi-doublons visuels, par blocs pour limiter la mémoire
    for _, idx in df.groupby("class_name").groups.items():
        idx = np.asarray(list(idx))
        Hc, Vc = H[idx, 0], H[idx]                           # (n,), (n, 8)
        for s in range(0, len(idx), block):
            xor = Hc[s:s + block, None, None] ^ Vc[None, :, :]  # (b, n, 8)
            d = popcount64(xor).min(axis=2)                    # distance min sur les 8 variantes
            rows = np.arange(s, min(s + block, len(idx)))
            d[rows - s, rows] = 64                             # ignore la distance à soi-même
            nn_dist[idx[rows]] = np.minimum(nn_dist[idx[rows]], d.min(axis=1))
            for a, b in zip(*np.nonzero(d <= threshold)):
                uf.union(idx[rows[a]], idx[b])
    groups = np.array([uf.find(i) for i in range(n)])
    return groups, nn_dist


# -----------------------------------------------------------------------------
# Découpage stratifié PAR GROUPE
# -----------------------------------------------------------------------------
def group_stratified_split(df: pd.DataFrame, ratios=(0.70, 0.15, 0.15), seed: int = 42) -> pd.DataFrame:
    """Affecte chaque GROUPE (feuille) à train/val/test, classe par classe.

    Pour chaque classe : on mélange les groupes (graine 42), puis on les place
    dans l'ordre selon la fraction cumulée d'images qu'ils représentent (point
    milieu du groupe < 70 % -> train, < 85 % -> val, sinon test). Stratification
    = mêmes proportions dans chaque classe ; groupe entier = pas de fuite.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["split"] = ""
    c1, c2 = ratios[0], ratios[0] + ratios[1]
    for _, sub in out.groupby("class_name"):
        sizes = sub.groupby("group").size()
        order = sizes.index.to_numpy()[rng.permutation(len(sizes))]
        total, cum = sizes.sum(), 0
        assign = {}
        for g in order:
            mid = (cum + sizes[g] / 2) / total
            assign[g] = "train" if mid < c1 else ("val" if mid < c2 else "test")
            cum += sizes[g]
        out.loc[sub.index, "split"] = sub["group"].map(assign)
    return out


def check_no_leakage(df: pd.DataFrame) -> None:
    """Vérifie qu'aucun groupe (feuille) ni fichier n'est partagé entre ensembles."""
    per_group = df.groupby("group")["split"].nunique()
    assert (per_group == 1).all(), f"{(per_group > 1).sum()} groupes partagés entre ensembles !"
    per_md5 = df.groupby("md5")["split"].nunique()
    assert (per_md5 == 1).all(), "Des fichiers identiques sont présents dans deux ensembles !"
    print("OK : aucun groupe ni fichier identique partagé entre train / val / test.")


def load_split(split_dir: Path, name: str) -> pd.DataFrame:
    """Lit train.csv / val.csv / test.csv produits par le notebook 00."""
    return pd.read_csv(Path(split_dir) / f"{name}.csv")


# -----------------------------------------------------------------------------
# Normalisation
# -----------------------------------------------------------------------------
def compute_norm_stats(df_train: pd.DataFrame, data_root: Path, size: int,
                       max_images: int = 3000, seed: int = 42) -> dict:
    """Moyenne / écart-type par canal calculés sur le TRAIN uniquement.

    Utiliser val ou test ici ferait « fuiter » de l'information sur ces ensembles.
    Un échantillon de 3000 images suffit : l'erreur sur la moyenne est négligeable.
    """
    sub = df_train.sample(min(max_images, len(df_train)), random_state=seed)
    s = np.zeros(3)
    s2 = np.zeros(3)
    n = 0
    for r in sub["relpath"]:
        x = np.asarray(load_image(Path(data_root) / r, size), dtype=np.float64) / 255.0
        s += x.sum(axis=(0, 1))
        s2 += (x ** 2).sum(axis=(0, 1))
        n += x.shape[0] * x.shape[1]
    mean = s / n
    std = np.sqrt(s2 / n - mean ** 2)
    return {"mean": mean.tolist(), "std": std.tolist(), "size": size, "n_images": len(sub)}


# -----------------------------------------------------------------------------
# Cache en mémoire + Dataset PyTorch
# -----------------------------------------------------------------------------
def load_or_build_cache(relpaths, data_root: Path, size: int, cache_dir: Path,
                        tag: str, n_threads: int = 8) -> torch.Tensor:
    """Charge toutes les images redimensionnées dans un tenseur uint8 (N, 3, H, W).

    Décoder 18 000 JPEG à chaque epoch est le goulot d'étranglement sur
    Kaggle/Colab (2 à 4 CPU). On décode une seule fois, on sauvegarde en .npy,
    et les notebooks suivants rechargent en quelques secondes. uint8 : 4x moins
    de mémoire que float32 (~2,7 Go pour 18k images en 224x224).
    """
    relpaths = list(relpaths)
    key = hashlib.md5("|".join(relpaths).encode()).hexdigest()[:8]
    f = Path(cache_dir) / f"{tag}_{size}_{key}.npy"
    if f.exists():
        return torch.from_numpy(np.load(f))
    f.parent.mkdir(parents=True, exist_ok=True)

    def _load(r):
        return np.asarray(load_image(Path(data_root) / r, size), dtype=np.uint8)

    with ThreadPoolExecutor(n_threads) as ex:
        arr = np.stack(list(ex.map(_load, relpaths))).transpose(0, 3, 1, 2).copy()
    np.save(f, arr)
    return torch.from_numpy(arr)


def build_transform(train: bool, mean=None, std=None, augment: str = "full"):
    """Transformations sur tenseurs uint8 (torchvision.transforms.v2).

    Augmentations (train seulement) : symétries et petites rotations (une feuille
    n'a pas d'orientation canonique) et une variation de luminosité/contraste
    modérée (robustesse à l'éclairage). Pas de variation de teinte : la couleur
    (jaunissement, taches brunes) porte une partie du diagnostic.
    mean/std = None -> pas de normalisation (auto-encodeurs : sortie Sigmoid dans [0,1]).
    """
    t = []
    if train and augment == "full":
        t += [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(), v2.RandomRotation(15),
              v2.ColorJitter(brightness=0.2, contrast=0.2)]
    elif train and augment == "flip":
        t += [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()]
    t.append(v2.ToDtype(torch.float32, scale=True))
    if mean is not None:
        t.append(v2.Normalize(mean, std))
    return v2.Compose(t)


class TensorImageDataset(Dataset):
    """Dataset sur un tenseur uint8 préchargé ; retourne (image, label)."""

    def __init__(self, images: torch.Tensor, labels, transform):
        self.images = images
        self.labels = torch.as_tensor(np.array(labels), dtype=torch.long)  # copie (tableaux pandas en lecture seule)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.transform(self.images[i]), self.labels[i]


def make_loader(dataset: Dataset, batch_size: int, train: bool, num_workers: int = 2,
                seed: int = 42) -> DataLoader:
    """DataLoader reproductible (générateur et graine des workers fixés)."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=train,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      worker_init_fn=seed_worker, generator=make_generator(seed),
                      persistent_workers=num_workers > 0)


def get_split_tensors(paths: dict, data_root: Path, size: int, splits=("train", "val", "test")):
    """Charge les CSV de split et les images associées (cache .npy).

    Retourne {split: (DataFrame, tenseur uint8)}. Le test est chargé en mémoire
    mais n'est évalué qu'une fois par modèle final (voir utils.TestGuard).
    """
    out = {}
    for s in splits:
        df = load_split(paths["split_dir"], s)
        out[s] = (df, load_or_build_cache(df["relpath"], data_root, size, paths["cache"], s))
    return out


class FolderImages(Dataset):
    """Images d'un dossier quelconque (jeu hors-distribution, photos personnelles)."""

    def __init__(self, files, size: int, transform):
        self.files = list(files)
        self.size = size
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = torch.from_numpy(np.array(load_image(self.files[i], self.size))).permute(2, 0, 1)
        return self.transform(img), -1
