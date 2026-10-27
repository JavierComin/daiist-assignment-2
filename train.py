"""Assignment 2: pipeline de entrenamiento (uv run python main.py train).

De principio a fin y sin pasos manuales:
  1. Descarga la version del dataset de Roboflow Universe (con tu clave, via api_key.py).
  2. Afina yolo11 desde el checkpoint preentrenado en COCO y guarda weights/best.pt.
  3. Evalua en el split de test: global y por clase, y la matriz de confusion.
  4. Compara con dos lineas base: COCO sin afinar (zero-shot) y las triviales.
  5. Puntua cada imagen (confianza maxima en clases defectuosas), barre el umbral con
     los costes de costs.json, elige los dos umbrales (pasa / revision / rechaza) y
     calcula la carga de revision.
  6. Guarda en results/ todo lo que necesita app.py (no reentrena nada).

Ajustes por variables de entorno (los valores por defecto son los de la entrega):
  EPOCHS, IMGSZ, BATCH, MODEL, PATIENCE, WORKERS
  SKIP_TRAIN=1 reutiliza weights/best.pt y solo reevalua. Util para cambiar costs.json
  sin reentrenar. Sin esa variable, siempre entrena.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import costing

ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
WEIGHTS = ROOT / "weights"

# --- Dataset (tiene que coincidir con REPORT.md) ---
WORKSPACE, PROJECT, VERSION = "kajal-tdslw", "final-apple-data", 8
FRESH_CLASSES = ["freshapples"]                                   # manzana buena
DEFECT_CLASSES = ["damaged_apple", "defect", "rottenapple"]       # positivo = defectuosa

# --- Ajustes de entrenamiento ---
MODEL = os.environ.get("MODEL", "yolo11s.pt")
IMGSZ = int(os.environ.get("IMGSZ", 416))    # las manzanas ocupan ~50 % de la imagen: no hace falta 640
EPOCHS = int(os.environ.get("EPOCHS", 30))
PATIENCE = int(os.environ.get("PATIENCE", 10))
BATCH = int(os.environ.get("BATCH", 16))
WORKERS = int(os.environ.get("WORKERS", 4))
SEED = 0
MIN_PER_CLASS = 30   # minimo de imagenes por clase para elegir umbrales sobre val limpio


# ------------------------------------------------------------------ dataset
def get_dataset():
    base = DATASETS / f"{PROJECT}-v{VERSION}"
    if (base / "data.yaml").exists():
        print(f"Dataset ya descargado en {base}")
        return base
    from roboflow import Roboflow
    from api_key import load_api_key

    key = load_api_key()
    if isinstance(key, tuple):  # por si api_key devuelve (clave, origen)
        key = key[0]
    print(f"Descargando {WORKSPACE}/{PROJECT} version {VERSION} ...")
    # yolov8 y yolov11 comparten formato de etiquetas (.txt)
    Roboflow(api_key=key).workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
        "yolov8", location=str(base)
    )
    return base


def split_dir(base, split):
    names = {"train": ("train",), "val": ("valid", "val"), "test": ("test",)}[split]
    for n in names:
        if (base / n / "images").is_dir():
            return base / n
    raise FileNotFoundError(f"No hay carpeta de {split} en {base}")


def write_local_yaml(base):
    """Un yaml con rutas absolutas: evita problemas con los '../' del yaml de Roboflow."""
    names = yaml.safe_load((base / "data.yaml").read_text())["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    data = {
        "path": str(base),
        "train": f"{split_dir(base, 'train').name}/images",
        "val": f"{split_dir(base, 'val').name}/images",
        "test": f"{split_dir(base, 'test').name}/images",
        "names": {i: n for i, n in enumerate(names)},
    }
    path = base / "data_local.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path, names


def origin(filename):
    """Imagen original: Roboflow anade '.rf.<hash>' al exportar."""
    return filename.split(".rf.")[0]


def read_labels(base, split, img_path):
    label = split_dir(base, split) / "labels" / f"{img_path.stem}.txt"
    rows = []
    if label.exists():
        for line in label.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                rows.append((int(parts[0]), *[float(v) for v in parts[1:5]]))
    return rows


# ------------------------------------------------------------------ entrenamiento
def train_model(data_yaml):
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        print("AVISO: no hay GPU. Entrenar en CPU tardara muchisimo; usa Colab (T4) para el modelo real.")
    model = YOLO(MODEL)
    started = time.time()
    model.train(
        data=str(data_yaml), imgsz=IMGSZ, epochs=EPOCHS, patience=PATIENCE, batch=BATCH,
        workers=WORKERS, seed=SEED, project=str(RUNS), name="train", exist_ok=True,
    )
    best = RUNS / "train" / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"El entrenamiento no produjo {best}")
    WEIGHTS.mkdir(exist_ok=True)
    shutil.copy(best, WEIGHTS / "best.pt")
    print(f"Pesos guardados en {WEIGHTS / 'best.pt'}")
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {"train_seconds": round(time.time() - started), "device": device}


# ------------------------------------------------------------------ evaluacion
def detection_metrics(model, data_yaml, tag):
    """AP50 y mAP50-95 globales y por clase en el split de test."""
    m = model.val(
        data=str(data_yaml), split="test", imgsz=IMGSZ, conf=0.001, iou=0.6,
        plots=False, project=str(RUNS), name=f"val_{tag}", exist_ok=True, verbose=False,
    )
    names = m.names
    per_class = []
    for i, ci in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        per_class.append({
            "class": names[int(ci)], "precision": float(p), "recall": float(r),
            "ap50": float(ap50), "map50_95": float(ap),
        })
    overall = {
        "precision": float(m.box.mp), "recall": float(m.box.mr),
        "ap50": float(m.box.map50), "map50_95": float(m.box.map),
    }
    return overall, per_class


def confusion_matrix(model, data_yaml):
    """Matriz de confusion de Ultralytics con conf=0.25 (la que se usa para dibujarla)."""
    m = model.val(
        data=str(data_yaml), split="test", imgsz=IMGSZ, conf=0.25, iou=0.6,
        plots=True, project=str(RUNS), name="val_confusion", exist_ok=True, verbose=False,
    )
    RESULTS.mkdir(exist_ok=True)
    for fname in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
        src = Path(m.save_dir) / fname
        if src.exists():
            shutil.copy(src, RESULTS / fname)
    raw = np.asarray(m.confusion_matrix.matrix)
    labels = [m.names[i] for i in range(len(m.names))] + ["background"]
    return {"labels": labels, "matrix": raw.tolist(),
            "note": "matrix[predicho][real]; el ultimo indice es 'background' (prediccion sin objeto = no detectado, "
                    "o objeto inexistente = falsa deteccion). Mira tambien confusion_matrix.png."}


def instance_counts(base, split, names):
    counts = {n: 0 for n in names}
    for img in (split_dir(base, split) / "images").iterdir():
        for cls, *_ in read_labels(base, split, img):
            counts[names[cls]] += 1
    return counts


# ------------------------------------------------------------------ puntuacion por imagen
def score_split(model, base, split, names, origins_other):
    """Una fila por imagen: verdad (hay alguna defectuosa?) y puntuacion del modelo."""
    defect_ids = [names.index(n) for n in DEFECT_CLASSES]
    images = sorted(p for p in (split_dir(base, split) / "images").iterdir() if p.is_file())
    by_name = {p.name: p for p in images}
    rows = []
    stream = model.predict(
        source=[str(p) for p in images], imgsz=IMGSZ, conf=0.001, iou=0.6,
        stream=True, verbose=False, batch=BATCH,
    )
    for r in stream:
        p = by_name[Path(r.path).name]
        gt = read_labels(base, split, p)
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        mask = np.isin(cls, defect_ids)
        rows.append({
            "image": p.name,
            "has_objects": len(gt) > 0,
            "y_true": int(any(c in defect_ids for c, *_ in gt)),
            "score": float(conf[mask].max()) if mask.any() else 0.0,
            "clean": origin(p.name) not in origins_other,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ linea base COCO
def iou_xyxy(a, b):
    """IoU entre una caja y varias. Es invariante al escalado de ejes: vale en coordenadas normalizadas."""
    x1 = np.maximum(a[0], b[:, 0]); y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2]); y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a + area_b - inter + 1e-12)


def coco_zero_shot(base, names):
    """COCO sin afinar. Sabe localizar 'apple' pero no tiene concepto de 'defectuosa'.

    Se mide como detector de manzanas sin distinguir clase (recall y precision con IoU >= 0.5).
    Como nunca puede marcar una manzana como defectuosa, su decision equivale a 'no rechazar nada'.
    """
    from ultralytics import YOLO

    coco = YOLO(MODEL)  # checkpoint preentrenado, sin tocar
    apple_ids = [i for i, n in coco.names.items() if n == "apple"]
    images = sorted(p for p in (split_dir(base, "test") / "images").iterdir() if p.is_file())
    by_name = {p.name: p for p in images}
    tp = fp = n_gt = 0
    for r in coco.predict(source=[str(p) for p in images], imgsz=IMGSZ, conf=0.25, stream=True, verbose=False):
        gt = read_labels(base, "test", by_name[Path(r.path).name])
        gt_boxes = np.array([[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2] for _, cx, cy, w, h in gt]).reshape(-1, 4)
        n_gt += len(gt_boxes)
        keep = np.isin(r.boxes.cls.cpu().numpy().astype(int), apple_ids)
        preds = r.boxes.xyxyn.cpu().numpy()[keep]
        matched = set()
        for pb in preds:
            if len(gt_boxes):
                ious = iou_xyxy(pb, gt_boxes)
                ious[list(matched)] = 0.0
                j = int(ious.argmax())
                if ious[j] >= 0.5:
                    matched.add(j); tp += 1
                    continue
            fp += 1
    return {
        "apple_recall_iou50": tp / max(n_gt, 1),
        "apple_precision_iou50": tp / max(tp + fp, 1),
        "ground_truth_boxes": n_gt,
        "defect_ap50": 0.0,
        "note": "COCO no tiene clase 'defectuosa': su decision equivale a no rechazar nada.",
    }


# ------------------------------------------------------------------ decision y costes
def main():
    RESULTS.mkdir(exist_ok=True)
    c = costing.load_costs(ROOT / "costs.json")
    base = get_dataset()
    data_yaml, names = write_local_yaml(base)
    for n in FRESH_CLASSES + DEFECT_CLASSES:
        if n not in names:
            sys.exit(f"La clase '{n}' no esta en el dataset: {names}")

    settings = {
        "model": MODEL, "imgsz": IMGSZ, "epochs": EPOCHS, "patience": PATIENCE,
        "batch": BATCH, "seed": SEED, "dataset": f"{WORKSPACE}/{PROJECT}/{VERSION}",
    }
    if os.environ.get("SKIP_TRAIN") == "1":
        if not (WEIGHTS / "best.pt").exists():
            sys.exit("SKIP_TRAIN=1 pero no existe weights/best.pt")
        print("SKIP_TRAIN=1: se reutilizan los pesos de weights/best.pt")
        prev = RESULTS / "metrics.json"
        if prev.exists():
            settings.update({k: v for k, v in json.loads(prev.read_text())["settings"].items()
                             if k in ("train_seconds", "device", "epochs", "imgsz", "model", "batch", "patience")})
    else:
        settings.update(train_model(data_yaml))

    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS / "best.pt"))

    # --- evaluacion global y por clase
    overall, per_class = detection_metrics(model, data_yaml, "test")
    counts = {s: instance_counts(base, s, names) for s in ("train", "val", "test")}
    for row in per_class:
        row["test_instances"] = counts["test"][row["class"]]
    confusion = confusion_matrix(model, data_yaml)
    (RESULTS / "confusion_matrix.json").write_text(json.dumps(confusion, indent=2))

    # --- puntuaciones por imagen (val y test)
    origins = {s: {origin(p.name) for p in (split_dir(base, s) / "images").iterdir()} for s in ("train", "val", "test")}
    val_df = score_split(model, base, "val", names, origins["train"])
    test_df = score_split(model, base, "test", names, origins["train"] | origins["val"])
    val_df.to_csv(RESULTS / "scores_val.csv", index=False)
    test_df.to_csv(RESULTS / "scores_test.csv", index=False)

    # Solo cuentan imagenes con manzanas: una cinta vacia no es una decision de calidad.
    val_df, test_df = val_df[val_df["has_objects"]], test_df[test_df["has_objects"]]

    # --- conjunto para elegir umbrales: val sin imagenes que tambien esten en train
    sel = val_df[val_df["clean"]]
    sel_name = "val (sin origenes compartidos con train)"
    if min(sel["y_true"].sum(), (1 - sel["y_true"]).sum()) < MIN_PER_CLASS:
        sel, sel_name = val_df, "val completo (el subconjunto limpio quedaba demasiado pequeno)"
    print(f"Umbrales elegidos sobre: {sel_name} ({len(sel)} imagenes)")
    s_sel, y_sel = sel["score"].to_numpy(), sel["y_true"].to_numpy()

    sweep = costing.sweep_single(s_sel, y_sel, c)
    single = costing.best_single(sweep)
    pair = costing.best_pair(s_sel, y_sel, c, max_reviewers=c.get("max_reviewers"))
    sweep.to_csv(RESULTS / "sweep_val.csv", index=False)

    def evaluate(df, label):
        s, y = df["score"].to_numpy(), df["y_true"].to_numpy()
        r = costing.outcome_rates(s, y, pair["t_low"], pair["t_high"])
        return {
            "set": label, "images": int(len(df)), "defective_images": int(y.sum()),
            "good_images": int((1 - y).sum()),
            "cost_per_1000": costing.cost_per_1000(s, y, c, pair["t_low"], pair["t_high"]),
            "interval90": costing.bootstrap_interval(s, y, c, pair["t_low"], pair["t_high"]),
            "staffing": costing.staffing(r, c),
            "alternatives": costing.alternatives(s, y, c, pair["t_low"], pair["t_high"]),
            "rates": r,
        }

    op = {
        "selection_set": sel_name, "selection_images": int(len(sel)),
        "single_threshold": single, "t_low": pair["t_low"], "t_high": pair["t_high"],
        "cost_per_1000_selection": pair["cost_per_1000"],
        "evaluation": [evaluate(test_df, "test completo")],
    }
    test_clean = test_df[test_df["clean"]]
    if min(test_clean["y_true"].sum(), (1 - test_clean["y_true"]).sum()) >= 5:
        op["evaluation"].append(evaluate(test_clean, "test sin origenes compartidos con train/val"))
    op["sensitivity_cost_fn"] = costing.sensitivity(
        s_sel, y_sel, c, "cost_fn", [c["cost_fn"] * f for f in (0.1, 0.25, 0.5, 1, 2, 4)]
    ).to_dict(orient="records")
    (RESULTS / "operating_point.json").write_text(json.dumps(op, indent=2))

    # --- lineas base
    baselines = {
        "coco_zero_shot": coco_zero_shot(base, names),
        "predict_nothing": {"ap50": 0.0, "decision": "no rechaza nada",
                            "cost_per_1000": op["evaluation"][0]["alternatives"]["no_hacer_nada"]},
        "majority_class": {
            "class": max(counts["train"], key=counts["train"].get),
            "note": "la clase mas frecuente en train; como decision sobre imagenes, rechazar todo",
            "image_level_accuracy_test": float(test_df["y_true"].mean()),
        },
    }
    metrics = {
        "settings": settings, "costs": c, "overall": overall, "per_class": per_class,
        "instances": counts, "baselines": baselines,
        "image_level": {"positive": "imagen con al menos una manzana defectuosa",
                        "defect_classes": DEFECT_CLASSES, "fresh_classes": FRESH_CLASSES},
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    shutil.copy(ROOT / "costs.json", RESULTS / "costs_used.json")

    ev = op["evaluation"][0]
    print("\n=== RESUMEN ===")
    print(f"AP50 test: {overall['ap50']:.3f} | mAP50-95: {overall['map50_95']:.3f}")
    for row in per_class:
        print(f"  {row['class']:<14} P={row['precision']:.3f} R={row['recall']:.3f} AP50={row['ap50']:.3f} (n={row['test_instances']})")
    print(f"Umbral unico (sin revision): {single['threshold']:.2f} -> {single['cost_per_1000']:.2f} EUR/1000 (en {sel_name})")
    print(f"Umbrales con revision: t_low={pair['t_low']:.2f}, t_high={pair['t_high']:.2f}")
    print(f"Test: {ev['cost_per_1000']:.2f} EUR/1000, intervalo 90 % {ev['interval90'][0]:.2f}-{ev['interval90'][1]:.2f}")
    print(f"Revision: {ev['staffing']['review_share']:.1%} de los casos, {ev['staffing']['reviews_per_day']:.0f}/dia, "
          f"{ev['staffing']['people_needed']} persona(s)")
    print("Alternativas (EUR/1000):", {k: round(v, 2) for k, v in ev["alternatives"].items()})
    print(f"Resultados en {RESULTS}")


if __name__ == "__main__":
    main()
