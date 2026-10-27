"""Compara datasets candidatos de Roboflow Universe calculando todo desde las etiquetas.

Uso (desde la raiz del repo, con ROBOFLOW_API_KEY en el entorno o en .env):
    uv run python dataset_stats.py kajal-tdslw/final-apple-data/8 carrefour-grocery/apple-nin9v/1

Cada argumento es workspace/proyecto/version. Solo descarga la version publica; no hace fork.
Los datos van a datasets/ (anade esa carpeta a .gitignore: no se sube al repo).
"""
import os
import sys
from collections import Counter
from pathlib import Path
from statistics import median

import yaml

MIN_TEST_INSTANCES = 100  # por debajo, la metrica de esa clase en test es poco fiable
SPLITS = {"train": ("train",), "val": ("valid", "val"), "test": ("test",)}


def box_wh(values):
    """Ancho y alto normalizados de una linea de etiqueta: bbox (4 valores) o poligono."""
    if len(values) == 4:
        return values[2], values[3]
    xs, ys = values[0::2], values[1::2]
    return max(xs) - min(xs), max(ys) - min(ys)


def load_stats(base):
    """Estadisticas por split de un dataset ya descargado en formato YOLO."""
    base = Path(base)
    names = yaml.safe_load((base / "data.yaml").read_text())["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    stats = {}
    for split, folders in SPLITS.items():
        folder = next((base / f for f in folders if (base / f / "images").is_dir()), None)
        if folder is None:
            stats[split] = None
            continue
        inst, areas, empty = Counter(), [], 0
        images = [p for p in (folder / "images").iterdir() if p.is_file()]
        for img in images:
            label = folder / "labels" / f"{img.stem}.txt"
            lines = (
                [ln.split() for ln in label.read_text().splitlines() if ln.strip()]
                if label.exists()
                else []
            )
            empty += not lines
            for parts in lines:
                inst[names[int(parts[0])]] += 1
                w, h = box_wh([float(v) for v in parts[1:]])
                areas.append(w * h)
        stats[split] = {
            "images": len(images),
            "empty": empty,
            "inst": inst,
            "median_area": median(areas) if areas else 0.0,
            "tiny": sum(a < 0.01 for a in areas) / len(areas) if areas else 0.0,
        }
    return names, stats


def report(spec, names, stats):
    print(f"\n=== {spec} ===\nclases: {names}")
    for split, s in stats.items():
        if s is None:
            print(f"  {split:<5} NO EXISTE en esta version")
            continue
        counts = ", ".join(f"{c}={s['inst'][c]}" for c in names)
        print(f"  {split:<5} {s['images']:>5} imagenes ({s['empty']} sin objetos) | {counts}")
        print(
            f"        objeto mediano = {s['median_area']:.1%} del area de la imagen; "
            f"{s['tiny']:.0%} de los objetos ocupan < 1%"
        )
    test = stats["test"]
    if test:
        weak = [c for c in names if test["inst"][c] < MIN_TEST_INSTANCES]
        if weak:
            print(f"  AVISO clases con < {MIN_TEST_INSTANCES} instancias en test (cifra poco fiable): {weak}")
    else:
        print("  AVISO sin split de test: no sirve para el enunciado")


def download(spec, root="datasets"):
    from roboflow import Roboflow

    workspace, project, version = spec.split("/")
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    ds = (
        rf.workspace(workspace)
        .project(project)
        .version(int(version))
        .download("yolov8", location=f"{root}/{project}-v{version}")  # mismas etiquetas .txt que yolov11
    )
    return ds.location


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for spec in sys.argv[1:]:
        names, stats = load_stats(download(spec))
        report(spec, names, stats)
