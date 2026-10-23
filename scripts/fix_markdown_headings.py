"""Corrige les cellules Markdown dont chaque ligne commence par « # » (affichées en titres).

Défaut du générateur initial des notebooks : le préfixe de commentaire « # » de chaque
ligne a été conservé dans les cellules Markdown. Ce script le retire, en modifiant
UNIQUEMENT les cellules Markdown concernées : le code et les sorties des cellules de
code (preuves d'exécution) ne sont pas touchés.

Usage (depuis la racine du projet) :
    python scripts/fix_markdown_headings.py notebooks/*.ipynb            # simulation
    python scripts/fix_markdown_headings.py notebooks/*.ipynb --apply    # écriture
Sur Colab : fermer les onglets de ces notebooks avant --apply (sinon Colab réécrase le fichier).
"""
import argparse
import sys

import nbformat


def fix_source(src: str) -> str:
    lines = src.split("\n")
    non_empty = [l for l in lines if l.strip()]
    fully_prefixed = (len(non_empty) >= 2 or (non_empty and non_empty[0].startswith("# #"))) \
        and all(l.startswith("# ") or l == "#" for l in non_empty)
    if fully_prefixed:   # un vrai titre isolé « # Titre » n'est pas touché
        # Cellule entièrement préfixée : on retire « # » ou « # » en tête de chaque ligne
        return "\n".join(l[2:] if l.startswith("# ") else ("" if l == "#" else l) for l in lines)
    # Cellule partiellement préfixée : seules les lignes « # # Titre », « # ## Titre » et
    # « # **Gras** ... » (paragraphe affiché à tort en titre) sont corrigées
    return "\n".join(l[2:] if l.startswith(("# #", "# **")) else l for l in lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebooks", nargs="+")
    ap.add_argument("--apply", action="store_true", help="écrire les fichiers (sinon simulation)")
    args = ap.parse_args()
    for path in args.notebooks:
        nb = nbformat.read(path, as_version=4)
        n = 0
        for c in nb.cells:
            if c.cell_type == "markdown":
                new = fix_source(c.source)
                if new != c.source:
                    c.source = new
                    n += 1
        print(f"{path} : {n} cellule(s) Markdown à corriger")
        if args.apply and n:
            nbformat.validate(nb)
            nbformat.write(nb, path)
    if not args.apply:
        print("Simulation uniquement. Relancer avec --apply pour écrire.")


if __name__ == "__main__":
    sys.exit(main())
