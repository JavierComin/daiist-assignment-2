"""Comprobaciones de calidad sobre un dataset YOLO ya descargado.

Uso (despues de haber ejecutado dataset_stats.py):
    uv run python dataset_checks.py datasets/final-apple-data-v8

1. Fugas: la misma imagen de origen aparece en mas de un split?
2. Atajos: se puede adivinar la clase solo mirando el fondo de la imagen?
   (se mide el brillo medio del borde: oscuro < 60, claro > 200, medio el resto)
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

SPLITS = {"train": ("train",), "val": ("valid", "val"), "test": ("test",)}
DARK, LIGHT = 60, 200


def origin(filename):
    """Nombre de la imagen original: Roboflow anade '.rf.<hash>' al exportar."""
    return filename.split(".rf.")[0]


def border_brightness(path):
    im = Image.open(path)
    im.thumbnail((128, 128))
    g = np.asarray(im.convert("L"), dtype=float)
    k = max(1, int(0.05 * min(g.shape)))
    border = np.concatenate(
        [g[:k].ravel(), g[-k:].ravel(), g[:, :k].ravel(), g[:, -k:].ravel()]
    )
    return border.mean()


def main(base):
    base = Path(base)
    names = yaml.safe_load((base / "data.yaml").read_text())["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]

    origins = {}
    background = defaultdict(Counter)  # (split, clase) -> {oscuro/medio/claro: n imagenes}
    for split, folders in SPLITS.items():
        folder = next((base / f for f in folders if (base / f / "images").is_dir()), None)
        if folder is None:
            continue
        origins[split] = set()
        images = [p for p in (folder / "images").iterdir() if p.is_file()]
        print(f"Leyendo {split}: {len(images)} imagenes ...", flush=True)
        for img in images:
            origins[split].add(origin(img.name))
            label = folder / "labels" / f"{img.stem}.txt"
            if not label.exists():
                continue
            classes = {
                names[int(ln.split()[0])]
                for ln in label.read_text().splitlines()
                if ln.strip()
            }
            if not classes:
                continue
            b = border_brightness(img)
            kind = "oscuro" if b < DARK else "claro" if b > LIGHT else "medio"
            for c in classes:
                background[(split, c)][kind] += 1

    print("\n== 1. Fugas: imagenes de origen compartidas entre splits ==")
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        if a in origins and b in origins:
            shared = len(origins[a] & origins[b])
            print(f"  {a} y {b}: {shared} origenes en comun (de {len(origins[b])} en {b})")

    print("\n== 2. Fondo por clase (numero de imagenes que contienen la clase) ==")
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
