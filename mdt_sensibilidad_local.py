#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Análisis de sensibilidad local emparejado: 960 corridas (4×4×2×30).
Las salidas son cálculos del modelo, no evidencia observacional.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from backend import scenario_model as model

PARAMETROS = {
    "flujo_base": "Flujo de entrada",
    "tasa_integracion": "Tránsito directo α",
    "despeje_retenidos": "Despeje δ",
    "factor_crecimiento": "Crecimiento g",
}

def ejecutar(salida: Path, replicas: int = 30, meses: int = 12,
            semilla_base: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Un parámetro por corrida, misma semilla en referencia y perturbación."""
    salida.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for idx, (escenario, base) in enumerate(model.CONFIG_ESCENARIOS.items(), start=1):
        original = base.copy()
        try:
            reference = {}
            for rep in range(1, replicas + 1):
                seed = semilla_base + 10_000 * idx + rep
                m, _ = model.simular_montecarlo(escenario, seed, meses=meses, replica_id=rep)
                reference[rep] = (seed, int(m["stock_final_retenidos"]))
            for variable, label in PARAMETROS.items():
                for sign in (-1, +1):
                    modified = original.copy()
                    modified[variable] = original[variable] * (1 + 0.10 * sign)
                    if variable == "flujo_base":
                        modified[variable] = round(modified[variable])
                    if variable in ("tasa_integracion", "despeje_retenidos"):
                        modified[variable] = min(0.999, max(0.001, modified[variable]))
                    model.CONFIG_ESCENARIOS[escenario] = modified
                    for rep, (seed, reference_stock) in reference.items():
                        m, _ = model.simular_montecarlo(escenario, seed, meses=meses, replica_id=rep)
                        changed = int(m["stock_final_retenidos"])
                        rows.append({
                            "escenario": escenario,
                            "replica_id": rep, "semilla_aleatoria": seed,
                            "parametro": variable, "nombre_parametro": label,
                            "direccion_pct": sign * 10,
                            "valor_original": original[variable],
                            "valor_modificado": modified[variable],
                            "stock_base": reference_stock,
                            "stock_sensibilidad": changed,
                            "diferencia_stock": changed - reference_stock,
                            "diferencia_relativa_stock_pct": 100 * (changed - reference_stock) / reference_stock if reference_stock else float("nan"),
                            "primer_cruce_100": m["mes_primera_sobrecapacidad_100pct"],
                        })
        finally:
            model.CONFIG_ESCENARIOS[escenario] = original
    d = pd.DataFrame(rows)
    d.to_csv(salida / "sensibilidad_local_corridas.csv", index=False, encoding="utf-8-sig")
    group_cols = ["escenario", "parametro", "nombre_parametro", "direccion_pct"]
    summary = d.groupby(group_cols, sort=False).agg(
        n=("replica_id", "count"),
        base_media=("stock_base", "mean"),
        escenario_modificado_media=("stock_sensibilidad", "mean"),
        media_cambio_stock=("diferencia_stock", "mean"),
        de_cambio_stock=("diferencia_stock", "std"),
        media_cambio_relativo_pct=("diferencia_relativa_stock_pct", "mean"),
        pct_supera_capacidad=("primer_cruce_100", lambda values: (values != -1).mean() * 100),
    ).reset_index()
    margin = student_t.ppf(0.975, summary.n - 1) * summary.de_cambio_stock / np.sqrt(summary.n)
    summary["ic95_cambio_inf"] = summary.media_cambio_stock - margin
    summary["ic95_cambio_sup"] = summary.media_cambio_stock + margin
    summary.to_csv(salida / "sensibilidad_local_resumen.csv", index=False, encoding="utf-8-sig")
    print(f"Corridas de sensibilidad: {len(d)}; combinaciones: {len(summary)}")
    print(summary.loc[summary.direccion_pct == 10,
                      ["escenario", "parametro", "media_cambio_relativo_pct"]].to_string(index=False))
    return d, summary

if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--salida", default="verificaciones", help="Directorio de CSV")
    cli.add_argument("--replicas", type=int, default=30)
    cli.add_argument("--meses", type=int, default=12)
    cli.add_argument("--semilla-base", type=int, default=1000)
    args = cli.parse_args()
    ejecutar(Path(args.salida), args.replicas, args.meses, args.semilla_base)
