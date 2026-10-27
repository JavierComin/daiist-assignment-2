"""Coste esperado de una decision de tres salidas: pasar / revisar / rechazar.

Lo usan train.py (para elegir los umbrales) y app.py (para el dashboard), asi que
hay una sola definicion de "cuanto cuesta equivocarse".

Trabaja con dos arrays, uno por imagen:
    scores : confianza maxima del modelo en CUALQUIER clase defectuosa (0 si no detecta nada)
    y      : 1 si la imagen contiene al menos una manzana defectuosa, 0 si no

Idea central: las tasas (que fraccion de defectuosas se deja pasar, que fraccion de
buenas se rechaza...) se miden en el dataset, pero el coste se pondera con la TASA BASE
de produccion, no con la proporcion del dataset (aqui ~80 % defectuosas). Si se usara la
proporcion del dataset, el coste esperado saldria falseado.

Regla de decision:  score <  t_low            -> pasa
                    t_low <= score < t_high   -> revision humana
                    score >= t_high           -> rechazo
Con t_low == t_high no hay revision (decision solo del modelo).
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

GRID = np.round(np.linspace(0.0, 1.0, 101), 2)

REQUIRED_KEYS = [
    "base_rate", "cost_fp", "cost_fn", "cost_tp", "cost_tn", "cost_review",
    "human_error_rate", "apples_per_day", "review_seconds", "shift_hours", "utilization",
]


def load_costs(path="costs.json"):
    costs = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_KEYS if k not in costs]
    if missing:
        raise KeyError(f"Faltan claves en {path}: {missing}")
    if not 0 < costs["base_rate"] < 1:
        raise ValueError("base_rate debe estar entre 0 y 1 (por ejemplo 0.05 = 5 %).")
    costs.setdefault("max_reviewers", None)
    return costs


def outcome_rates(scores, y, t_low, t_high):
    """Reparto pasa/revisa/rechaza, por separado entre defectuosas (pos) y buenas (neg)."""
    if t_low > t_high:
        raise ValueError("t_low no puede ser mayor que t_high.")
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    rates = {}
    for label, mask in (("pos", y == 1), ("neg", y == 0)):
        s = scores[mask]
        if len(s) == 0:
            raise ValueError(f"No hay imagenes de la clase '{label}' para calcular tasas.")
        passed = float((s < t_low).mean())
        failed = float((s >= t_high).mean())
        rates[label] = {"pass": passed, "fail": failed, "review": 1.0 - passed - failed}
    return rates


def cost_per_apple(rates, c):
    """Coste esperado por manzana en euros, ponderado con la tasa base."""
    pi, h = c["base_rate"], c["human_error_rate"]
    pos, neg = rates["pos"], rates["neg"]
    # Una revision cuesta tiempo de persona y la persona se equivoca con probabilidad h.
    pos_review = c["cost_review"] + h * c["cost_fn"] + (1 - h) * c["cost_tp"]
    neg_review = c["cost_review"] + h * c["cost_fp"] + (1 - h) * c["cost_tn"]
    cost_pos = pos["pass"] * c["cost_fn"] + pos["review"] * pos_review + pos["fail"] * c["cost_tp"]
    cost_neg = neg["pass"] * c["cost_tn"] + neg["review"] * neg_review + neg["fail"] * c["cost_fp"]
    return pi * cost_pos + (1 - pi) * cost_neg


def cost_per_1000(scores, y, c, t_low, t_high):
    return 1000.0 * cost_per_apple(outcome_rates(scores, y, t_low, t_high), c)


def review_share(rates, c):
    """Fraccion de TODAS las manzanas de produccion que va a una persona."""
    pi = c["base_rate"]
    return pi * rates["pos"]["review"] + (1 - pi) * rates["neg"]["review"]


def staffing(rates, c):
    """Cuantos casos al dia van a una persona y cuantas personas hacen falta."""
    share = review_share(rates, c)
    reviews_per_day = c["apples_per_day"] * share
    seconds_needed = reviews_per_day * c["review_seconds"]
    seconds_per_person = c["shift_hours"] * 3600 * c["utilization"]
    people = seconds_needed / seconds_per_person
    return {
        "review_share": share,
        "reviews_per_day": reviews_per_day,
        "people_exact": people,
        "people_needed": math.ceil(people - 1e-9) if reviews_per_day > 0 else 0,
    }


def _argmin_middle(values, thresholds):
    """Si el minimo es una zona plana, devuelve el umbral central de esa zona."""
    values = np.asarray(values)
    flat = np.where(values <= values.min() + 1e-9)[0]
    return float(np.asarray(thresholds)[flat[len(flat) // 2]])


def sweep_single(scores, y, c, grid=GRID):
    """Barrido de UN umbral (sin revision): coste esperado por cada 1.000 manzanas."""
    rows = []
    for t in grid:
        r = outcome_rates(scores, y, t, t)
        rows.append({
            "threshold": float(t),
            "cost_per_1000": 1000 * cost_per_apple(r, c),
            "fn_rate": r["pos"]["pass"],      # defectuosas que se dejan pasar
            "fp_rate": r["neg"]["fail"],      # buenas que se rechazan
        })
    return pd.DataFrame(rows)


def best_single(sweep):
    t = _argmin_middle(sweep["cost_per_1000"], sweep["threshold"])
    row = sweep.loc[(sweep["threshold"] - t).abs().idxmin()]
    return {"threshold": t, "cost_per_1000": float(row["cost_per_1000"])}


def best_pair(scores, y, c, grid=GRID, max_reviewers=None):
    """Mejor par (t_low, t_high). Si hay max_reviewers, solo vale lo que esa plantilla aguanta."""
    best = None
    for i, t_low in enumerate(grid):
        for t_high in grid[i:]:
            r = outcome_rates(scores, y, t_low, t_high)
            if max_reviewers is not None and staffing(r, c)["people_needed"] > max_reviewers:
                continue
            cost = 1000 * cost_per_apple(r, c)
            if best is None or cost < best["cost_per_1000"] - 1e-9:
                best = {"t_low": float(t_low), "t_high": float(t_high), "cost_per_1000": cost}
    if best is None:
        raise ValueError("Ningun par de umbrales cabe en max_reviewers. Sube la plantilla o relaja el limite.")
    return best


def confusion_3way(scores, y, t_low, t_high, c=None):
    """Matriz de confusion de 3 salidas: filas = decision, columnas = realidad."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    decision = np.where(scores < t_low, "pasa", np.where(scores >= t_high, "rechaza", "revision"))
    counts = pd.crosstab(
        pd.Categorical(decision, ["pasa", "revision", "rechaza"]),
        pd.Categorical(np.where(y == 1, "defectuosa", "buena"), ["buena", "defectuosa"]),
        dropna=False,
    )
    counts.index.name, counts.columns.name = "decision", "realidad"
    out = {"counts": counts}
    if c is not None:
        r = outcome_rates(scores, y, t_low, t_high)
        pi = c["base_rate"]
        per1000 = pd.DataFrame(
            {
                "buena": [1000 * (1 - pi) * r["neg"][k] for k in ("pass", "review", "fail")],
                "defectuosa": [1000 * pi * r["pos"][k] for k in ("pass", "review", "fail")],
            },
            index=["pasa", "revision", "rechaza"],
        )
        per1000.index.name, per1000.columns.name = "decision", "realidad"
        out["per_1000_at_base_rate"] = per1000
    return out


def alternatives(scores, y, c, t_low, t_high, grid=GRID):
    """Tu diseno frente a: no hacer nada, que una persona mire todo, y modelo solo."""
    pi, h = c["base_rate"], c["human_error_rate"]
    do_nothing = pi * c["cost_fn"] + (1 - pi) * c["cost_tn"]
    check_all = (
        pi * (c["cost_review"] + h * c["cost_fn"] + (1 - h) * c["cost_tp"])
        + (1 - pi) * (c["cost_review"] + h * c["cost_fp"] + (1 - h) * c["cost_tn"])
    )
    model_alone = best_single(sweep_single(scores, y, c, grid))
    return {
        "no_hacer_nada": 1000 * do_nothing,
        "persona_revisa_todo": 1000 * check_all,
        "modelo_solo": model_alone["cost_per_1000"],
        "modelo_solo_umbral": model_alone["threshold"],
        "modelo_mas_revision": cost_per_1000(scores, y, c, t_low, t_high),
    }


def bootstrap_interval(scores, y, c, t_low, t_high, n=2000, seed=0, level=0.90):
    """Intervalo del coste/1000 remuestreando imagenes dentro de cada clase.

    Con pocas imagenes en test, un unico numero engana: esto enseña cuanto puede bailar.
    """
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    pos, neg = scores[y == 1], scores[y == 0]
    costs = []
    for _ in range(n):
        s = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        yy = np.concatenate([np.ones(len(pos), int), np.zeros(len(neg), int)])
        costs.append(cost_per_1000(s, yy, c, t_low, t_high))
    lo, hi = np.quantile(costs, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def sensitivity(scores, y, c, key, values, grid=GRID):
    """Como cambia el umbral optimo (y el coste) si una hipotesis fuera distinta.

    Ejemplo: sensitivity(s, y, c, "cost_fn", [0.5, 1, 2, 5, 10]).
    """
    rows = []
    for v in values:
        c2 = dict(c)
        c2[key] = v
        single = best_single(sweep_single(scores, y, c2, grid))
        pair = best_pair(scores, y, c2, grid, c2.get("max_reviewers"))
        rows.append({
            key: v,
            "umbral_solo": single["threshold"],
            "coste_solo": single["cost_per_1000"],
            "t_low": pair["t_low"],
            "t_high": pair["t_high"],
            "coste_con_revision": pair["cost_per_1000"],
        })
    return pd.DataFrame(rows)
