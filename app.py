"""Assignment 2: dashboard (uv run python main.py app).

No entrena nada. Solo carga weights/best.pt y lo que dejo train.py en results/:
metrics.json, operating_point.json, scores_val.csv, scores_test.csv, costs_used.json.

Vistas:
  1. Prediccion: sube una imagen y mira las detecciones y la decision (pasa / revision / rechaza).
  2. Coste y revision humana: mueve los dos umbrales y mira la matriz de confusion,
     el coste esperado por 1.000 manzanas, la parte que va a una persona y la curva de coste.
  3. Resultados por clase: precision, recall, AP50, matriz de confusion y lineas base.
"""
import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import costing

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
WEIGHTS = ROOT / "weights" / "best.pt"


def load_json(name):
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


METRICS = load_json("metrics.json")
OP = load_json("operating_point.json")
COSTS = load_json("costs_used.json") or (
    json.loads((ROOT / "costs.json").read_text()) if (ROOT / "costs.json").exists() else None
)
READY = all(x is not None for x in (METRICS, OP, COSTS)) and WEIGHTS.exists() \
    and (RESULTS / "scores_val.csv").exists() and (RESULTS / "scores_test.csv").exists()


def _prepare():
    """Conjuntos de evaluacion para el dashboard (solo imagenes con manzanas)."""
    val = pd.read_csv(RESULTS / "scores_val.csv")
    test = pd.read_csv(RESULTS / "scores_test.csv")
    val, test = val[val["has_objects"]], test[test["has_objects"]]
    sets = {
        "Test completo": test,
        "Test sin origenes compartidos con train/val": test[test["clean"]],
        "Validacion completa": val,
        "Validacion sin origenes compartidos con train": val[val["clean"]],
    }
    ok = {}
    for name, df in sets.items():
        if df["y_true"].sum() >= 1 and (1 - df["y_true"]).sum() >= 1:  # el calculo necesita ambas clases
            ok[f"{name} ({len(df)} imagenes)"] = df
    return ok


SETS = _prepare() if READY else {}
READY = READY and bool(SETS)

if READY:
    from ultralytics import YOLO

    C = COSTS
    MODEL = YOLO(str(WEIGHTS))
    DEFAULT_SET = next(iter(SETS))
    SWEEPS = {name: costing.sweep_single(df["score"].to_numpy(), df["y_true"].to_numpy(), C) for name, df in SETS.items()}
    DEFECT_CLASSES = METRICS["image_level"]["defect_classes"]
    T_LOW, T_HIGH = OP["t_low"], OP["t_high"]


# ------------------------------------------------------------------ vista 1: prediccion
def decide(score, lo, hi):
    if score < lo:
        return "PASA"
    return "RECHAZA" if score >= hi else "REVISION HUMANA"


def predict(img, t_low, t_high):
    if img is None:
        return None, "Sube una imagen de una manzana.", None
    lo, hi = sorted((float(t_low), float(t_high)))
    result = MODEL.predict(img[:, :, ::-1], conf=0.001, iou=0.6, verbose=False)[0]  # RGB -> BGR
    conf = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    names = result.names
    is_defect = np.array([names[c] in DEFECT_CLASSES for c in cls], dtype=bool)
    score = float(conf[is_defect].max()) if is_defect.any() else 0.0
    shown = result[conf >= max(lo, 0.001)] if len(conf) else result
    annotated = shown.plot()[:, :, ::-1]  # BGR -> RGB
    table = pd.DataFrame(
        [{"clase": names[c], "confianza": round(float(p), 3)} for c, p in zip(cls, conf) if p >= max(lo, 0.001)]
    ).sort_values("confianza", ascending=False) if (conf >= max(lo, 0.001)).any() else pd.DataFrame(columns=["clase", "confianza"])
    verdict = decide(score, lo, hi)
    text = (
        f"### Decision: **{verdict}**\n"
        f"Puntuacion de defecto (confianza maxima en {', '.join(DEFECT_CLASSES)}): **{score:.2f}**\n\n"
        f"Pasa si < {lo:.2f} · Revision si esta entre {lo:.2f} y {hi:.2f} · Rechaza si >= {hi:.2f}\n\n"
        f"Se dibujan las detecciones con confianza >= {lo:.2f}."
    )
    return annotated, text, table


# ------------------------------------------------------------------ vista 2: coste y revision
def cost_figure(set_name, lo, hi):
    sweep = SWEEPS[set_name]
    best = costing.best_single(sweep)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["cost_per_1000"], mode="lines",
                             name="Un solo umbral (sin revision)"))
    fig.add_trace(go.Scatter(x=[best["threshold"]], y=[best["cost_per_1000"]], mode="markers",
                             marker=dict(size=11, symbol="diamond"), name=f"Minimo: umbral {best['threshold']:.2f}"))
    fig.add_vrect(x0=lo, x1=hi, opacity=0.15, line_width=0,
                  annotation_text="revision humana (sliders)", annotation_position="top left")
    for x, label in ((T_LOW, "t_low elegido"), (T_HIGH, "t_high elegido")):
        fig.add_vline(x=x, line_dash="dash", annotation_text=label, annotation_position="bottom")
    fig.update_layout(
        xaxis_title="Umbral de confianza", yaxis_title="Coste esperado por 1.000 manzanas (EUR)",
        title=f"Coste frente a umbral · {set_name}", legend=dict(orientation="h", y=-0.25),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def refresh(t_low, t_high, set_name):
    lo, hi = sorted((float(t_low), float(t_high)))
    df = SETS[set_name]
    s, y = df["score"].to_numpy(), df["y_true"].to_numpy()
    r = costing.outcome_rates(s, y, lo, hi)
    cost = 1000 * costing.cost_per_apple(r, C)
    ci_lo, ci_hi = costing.bootstrap_interval(s, y, C, lo, hi, n=300)
    st = costing.staffing(r, C)
    limit = C.get("max_reviewers")
    warn = ""
    if limit is not None and st["people_needed"] > limit:
        warn = f"\n\n**Atencion:** hacen falta {st['people_needed']} personas y la plantilla maxima supuesta es {limit}."
    kpi = (
        f"### Con pasa < {lo:.2f} · revision [{lo:.2f}, {hi:.2f}) · rechaza >= {hi:.2f}\n"
        f"- **Coste esperado: {cost:.2f} EUR por cada 1.000 manzanas** (intervalo 90 %: {ci_lo:.2f} - {ci_hi:.2f})\n"
        f"- **Enviado a revision: {st['review_share']:.1%}** de los casos = {st['reviews_per_day']:,.0f} al dia "
        f"({C['apples_per_day']:,} manzanas/dia, {C['review_seconds']} s por revision) = "
        f"{st['people_exact']:.2f} personas -> **{st['people_needed']} necesaria(s)**\n"
        f"- Defectuosas que pasan (falsos negativos): {r['pos']['pass']:.1%} · "
        f"Buenas rechazadas (falsos positivos): {r['neg']['fail']:.1%}\n"
        f"- Tasa base supuesta: {C['base_rate']:.1%} de manzanas defectuosas en produccion"
        f"{warn}"
    )
    cm = costing.confusion_3way(s, y, lo, hi, C)
    counts = cm["counts"].reset_index()
    per1000 = cm["per_1000_at_base_rate"].round(1).reset_index()
    alt = costing.alternatives(s, y, C, lo, hi)
    base = alt["no_hacer_nada"]
    alt_df = pd.DataFrame([
        ("No hacer nada (se deja pasar todo)", alt["no_hacer_nada"]),
        ("Una persona revisa todo", alt["persona_revisa_todo"]),
        (f"Solo el modelo (umbral optimo {alt['modelo_solo_umbral']:.2f})", alt["modelo_solo"]),
        ("Modelo + revision humana (umbrales actuales)", alt["modelo_mas_revision"]),
    ], columns=["Alternativa", "EUR por 1.000 manzanas"])
    alt_df["EUR por 1.000 manzanas"] = alt_df["EUR por 1.000 manzanas"].round(2)
    alt_df["Ahorro frente a no hacer nada"] = (base - alt_df["EUR por 1.000 manzanas"]).round(2)
    return kpi, counts, per1000, cost_figure(set_name, lo, hi), alt_df


# ------------------------------------------------------------------ vista 3: resultados por clase
def per_class_table():
    rows = []
    for row in METRICS["per_class"]:
        n = row["test_instances"]
        rows.append({
            "Clase": row["class"], "Precision": round(row["precision"], 3), "Recall": round(row["recall"], 3),
            "AP50": round(row["ap50"], 3), "mAP50-95": round(row["map50_95"], 3), "Instancias en test": n,
            "Fiabilidad": "poca muestra (< 100)" if n < 100 else "ok",
        })
    return pd.DataFrame(rows)


def baselines_table():
    df = SETS[DEFAULT_SET]
    s, y = df["score"].to_numpy(), df["y_true"].to_numpy()
    reject_all = costing.cost_per_1000(s, y, C, 0.0, 0.0)
    nothing = costing.alternatives(s, y, C, T_LOW, T_HIGH)["no_hacer_nada"]
    ev = OP["evaluation"][0]
    coco = METRICS["baselines"]["coco_zero_shot"]
    return pd.DataFrame([
        ("Modelo afinado + revision", round(METRICS["overall"]["ap50"], 3), round(ev["cost_per_1000"], 2),
         "decision con los umbrales elegidos"),
        ("COCO sin afinar (zero-shot)", 0.0, round(nothing, 2),
         f"localiza manzanas (recall IoU 0.5 = {coco['apple_recall_iou50']:.2f}) pero no conoce 'defectuosa'"),
        ("Trivial: no predecir nada", 0.0, round(nothing, 2), "no rechaza nada"),
        ("Trivial: clase mayoritaria", "-", round(reject_all, 2),
         f"siempre '{METRICS['baselines']['majority_class']['class']}' = rechazar todo"),
    ], columns=["Sistema", "AP50 global (deteccion)", "EUR por 1.000 manzanas (test completo)", "Nota"])


# ------------------------------------------------------------------ interfaz
def build_demo():
    with gr.Blocks(title="Control de calidad de manzanas") as demo:
        if not READY:
            gr.Markdown(
                "# Falta el entrenamiento\nNo encuentro `weights/best.pt` o los ficheros de `results/`. "
                "Ejecuta `uv run python main.py train` (en una maquina con GPU) y copia `weights/` y `results/`."
            )
            return demo
        gr.Markdown(
            "# Control de calidad de manzanas en cinta\n"
            "Positivo = manzana defectuosa. Tres salidas: **pasa**, **revision humana**, **rechaza**."
        )
        with gr.Row():
            t_low = gr.Slider(0, 1, value=T_LOW, step=0.01, label="Umbral inferior: por debajo, PASA")
            t_high = gr.Slider(0, 1, value=T_HIGH, step=0.01, label="Umbral superior: desde aqui, RECHAZA")
        with gr.Tabs():
            with gr.Tab("Prediccion"):
                with gr.Row():
                    inp = gr.Image(type="numpy", label="Sube una imagen")
                    out_img = gr.Image(label="Detecciones", interactive=False)
                verdict = gr.Markdown()
                det_table = gr.Dataframe(label="Detecciones sobre el umbral inferior", interactive=False)
                gr.on([inp.change, t_low.change, t_high.change], predict, [inp, t_low, t_high],
                      [out_img, verdict, det_table])
            with gr.Tab("Coste y revision humana"):
                set_name = gr.Radio(list(SETS), value=DEFAULT_SET, label="Conjunto de evaluacion")
                kpi = gr.Markdown()
                with gr.Row():
                    cm_counts = gr.Dataframe(label="Matriz de confusion (imagenes del conjunto)", interactive=False)
                    cm_1000 = gr.Dataframe(label="Por cada 1.000 manzanas con la tasa base supuesta", interactive=False)
                plot = gr.Plot()
                alt_table = gr.Dataframe(label="Comparacion con las alternativas", interactive=False)
                outs = [kpi, cm_counts, cm_1000, plot, alt_table]
                gr.on([t_low.change, t_high.change, set_name.change], refresh, [t_low, t_high, set_name], outs)
                demo.load(refresh, [t_low, t_high, set_name], outs)
            with gr.Tab("Resultados por clase"):
                ov = METRICS["overall"]
                gr.Markdown(
                    f"### Detector afinado, split de test\n"
                    f"**AP50 = {ov['ap50']:.3f} · mAP50-95 = {ov['map50_95']:.3f}** "
                    f"(modelo `{METRICS['settings']['model']}`, {METRICS['settings']['epochs']} epocas, "
                    f"imgsz {METRICS['settings']['imgsz']}). Un promedio esconde las clases flojas: mira la tabla."
                )
                gr.Dataframe(per_class_table(), label="Por clase", interactive=False)
                png = RESULTS / "confusion_matrix.png"
                if png.exists():
                    gr.Image(str(png), label="Matriz de confusion (Ultralytics, conf 0.25)", interactive=False)
                gr.Markdown("### Lineas base")
                gr.Dataframe(baselines_table(), interactive=False)
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch()
