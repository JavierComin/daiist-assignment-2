"""Dibuja results/cost_vs_threshold.png: coste esperado por 1.000 manzanas frente al umbral.

Uso:  uv run python plot_cost.py
Lee lo que dejo train.py en results/ (no entrena nada).
  - Curva continua: validacion (el conjunto con el que SE ELIGE el umbral).
  - Curva discontinua: test (solo para comprobar; no se usa para elegir).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import costing

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"

c = json.loads((RES / "costs_used.json").read_text(encoding="utf-8"))
op = json.loads((RES / "operating_point.json").read_text(encoding="utf-8"))
val = pd.read_csv(RES / "sweep_val.csv")
test = pd.read_csv(RES / "scores_test.csv")
test = test[test["has_objects"]]
test_sweep = costing.sweep_single(test["score"].to_numpy(), test["y_true"].to_numpy(), c)

t_star = op["single_threshold"]["threshold"]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(val["threshold"], val["cost_per_1000"], lw=2, label=f"Validacion ({op['selection_images']} imagenes): se elige aqui")
ax.plot(test_sweep["threshold"], test_sweep["cost_per_1000"], lw=1.6, ls="--", label=f"Test ({len(test)} imagenes): solo comprobacion")
ax.axvline(t_star, color="black", ls=":", lw=1.5)
ax.scatter([t_star], [op["single_threshold"]["cost_per_1000"]], color="black", zorder=5)
test_cost = float(test_sweep.loc[(test_sweep["threshold"] - t_star).abs().idxmin(), "cost_per_1000"])
ax.scatter([t_star], [test_cost], color="tab:orange", zorder=5)
ax.annotate(f"umbral elegido = {t_star:.2f}\nval {op['single_threshold']['cost_per_1000']:.2f}  |  test {test_cost:.2f} EUR/1000",
            xy=(t_star, test_cost), xytext=(0.20, 0.84), textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="black"))
ax.set_yscale("log")
ax.set_xlabel("Umbral de confianza (puntuacion de defecto de la imagen)")
ax.set_ylabel("Coste esperado por 1.000 manzanas (EUR, escala log)")
ax.set_title("Coste frente a umbral (un solo umbral, sin revision humana)")
ax.grid(alpha=0.3, which="both")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(RES / "cost_vs_threshold.png", dpi=150)
print("Guardado", RES / "cost_vs_threshold.png")
