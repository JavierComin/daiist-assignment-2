"""Comprobaciones de calidad sobre un dataset YOLO ya descargado.

Uso:  uv run python dataset_checks.py datasets/final-apple-data-v8

1. Fugas: la misma foto de origen aparece en mas de un split? (se compara la RAIZ del nombre,
   sin el '.rf.<hash>' de Roboflow ni prefijos de aumentacion como rotated_by_75_ o saltandpepper_)
2. Fuentes: cada prefijo del nombre es una fuente de imagenes. Una fuente que es 100 % buena o
   100 % defectuosa significa que el origen de la imagen predice la clase.
3. Fondo: se puede adivinar la clase solo con el brillo del borde? (oscuro < 60, claro > 200)
"""
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

SPLITS = {"train": ("train",), "val": ("valid", "val"), "test": ("test",)}
DEFECT_CLASSES = ["damaged_apple", "defect", "rottenapple"]
DARK, LIGHT = 60, 200
AUG_PREFIX = re.compile(r"^(?:(?:rotated_by_\d+|saltandpepper|translation|vertical_flip|horizontal_flip|flip)_)+")
MIN_SOURCE = 10  # fuentes con menos imagenes no se listan


def origin(filename):
    return AUG_PREFIX.sub("", filename.split(".rf.")[0])


def source(filename):
    """Prefijo del nombre (letras o digitos iniciales) tras quitar los prefijos de aumentacion."""
    root = origin(filename)
    m = re.match(r"^([A-Za-z]+|\d+)", root)
    return m.group(1) if m else root


def border_brightness(path):
    im = Image.open(path)
    im.thumbnail((128, 128))
    g = np.asarray(im.convert("L"), dtype=float)
    k = max(1, int(0.05 * min(g.shape)))
    border = np.concatenate([g[:k].ravel(), g[-k:].ravel(), g[:, :k].ravel(), g[:, -k:].ravel()])
    return border.mean()


def main(base):
    base = Path(base)
    names = yaml.safe_load((base / "data.yaml").read_text())["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]

    rows, background = [], defaultdict(Counter)
    for split, folders in SPLITS.items():
        folder = next((base / f for f in folders if (base / f / "images").is_dir()), None)
        if folder is None:
            continue
        images = [p for p in (folder / "images").iterdir() if p.is_file()]
        print(f"Leyendo {split}: {len(images)} imagenes ...", flush=True)
        for img in images:
            label = folder / "labels" / f"{img.stem}.txt"
            classes = (
                {names[int(ln.split()[0])] for ln in label.read_text().splitlines() if ln.strip()}
                if label.exists() else set()
            )
            if not classes:
                continue
            defective = int(any(c in DEFECT_CLASSES for c in classes))
            rows.append({"split": split, "name": img.name, "root": origin(img.name),
                         "exact": img.name.split(".rf.")[0], "source": source(img.name), "y": defective})
            b = border_brightness(img)
            kind = "oscuro" if b < DARK else "claro" if b > LIGHT else "medio"
            for c in classes:
                background[(split, c)][kind] += 1
    df = pd.DataFrame(rows)

    print("\n== 1. Fugas: fotos de origen compartidas entre splits ==")
    print("   (raiz = nombre sin '.rf.<hash>' ni prefijos de aumentacion; entre parentesis, solo nombre exacto)")
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        da, db = df[df.split == a], df[df.split == b]
        if len(da) and len(db):
            by_root = db.root.isin(set(da.root)).sum()
            by_exact = db.exact.isin(set(da.exact)).sum()
            print(f"  {b}: {by_root} de {len(db)} imagenes ({by_root / len(db):.0%}) tienen la misma raiz que alguna de {a}"
                  f"  (nombre exacto: {by_exact})")

    print(f"\n== 2. Fuentes con >= {MIN_SOURCE} imagenes (todos los splits) ==")
    g = df.groupby("source").agg(imagenes=("y", "size"), pct_defectuosa=("y", "mean"),
                                 en_train=("split", lambda s: int((s == "train").sum())),
                                 en_val=("split", lambda s: int((s == "val").sum())),
                                 en_test=("split", lambda s: int((s == "test").sum())))
    g["pct_defectuosa"] = (100 * g.pct_defectuosa).round(0).astype(int)
    big = g[g.imagenes >= MIN_SOURCE].sort_values("imagenes", ascending=False)
    print(big.head(20).to_string())
    pure = big[(big.pct_defectuosa == 0) | (big.pct_defectuosa == 100)]
    print(f"\n  {len(pure)} de {len(big)} fuentes grandes son 100 % buenas o 100 % defectuosas; "
          f"cubren el {pure.imagenes.sum() / len(df):.0%} de las imagenes.")
    aug = df[df.exact != df.root]
    print(f"  Variantes ya aumentadas (prefijo rotated_by_/saltandpepper_/...): {len(aug)} imagenes "
          f"({len(aug) / len(df):.0%}), {aug.y.mean():.0%} defectuosas." if len(aug) else "  Sin variantes aumentadas detectadas.")

    print("\n== 3. Fondo por clase (numero de imagenes que contienen la clase) ==")
    print(f"  {'split':<6}{'clase':<16}{'oscuro':>8}{'medio':>8}{'claro':>8}")
    for split in ("train", "test"):
        for c in names:
            n = background.get((split, c))
            if n:
                print(f"  {split:<6}{c:<16}{n['oscuro']:>8}{n['medio']:>8}{n['claro']:>8}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
