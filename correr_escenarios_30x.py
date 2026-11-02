#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""120 corridas Monte Carlo reproducibles. CSV contiene resultados simulados."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from backend.scenario_model import (
    CONFIG_ESCENARIOS, CAPACIDAD_RED_CORREDOR, STOCK_INICIAL_RETENIDOS,
    UMBRAL_ALERTA_SATURACION_PCT, simular_montecarlo,
)

N_REPLICAS = 30
HORIZONTE_MESES = 12
METRICAS_RESUMEN = {
    "total_acumulado_entrantes": "personas",
    "total_integrados_directos": "personas",
    "total_despejados_stock": "personas",
    "total_acumulado_integrados": "personas-salida",
    "stock_final_retenidos": "personas",
    "pico_maximo_retenidos": "personas",
    "saturacion_final_pct": "% sin truncar",
    "saturacion_media_pct": "% sin truncar",
    "tasa_integracion_directa_pct": "%",
    "tasa_retencion_entradas_pct": "%",
    "porcentaje_salida_total_sobre_stock_inicial_mas_entradas": "%",
    "stock_final_sobre_capacidad_pct": "%",
    "mes_primer_umbral_alerta_90pct": "mes",
    "mes_primera_sobrecapacidad_100pct": "mes",
}

def construir_resumen(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    filas = []
    for escenario in CONFIG_ESCENARIOS:
        grupo = df.loc[df["escenario"] == escenario]
        for metrica, unidad in METRICAS_RESUMEN.items():
            valores = pd.to_numeric(grupo[metrica], errors="coerce").dropna()
            n = len(valores)
            media = float(valores.mean()) if n else float("nan")
            sd = float(valores.std(ddof=1)) if n > 1 else 0.0
            margen = float(student_t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else 0.0
            filas.append(dict(escenario=escenario, metrica=metrica, unidad=unidad, n=n,
                media=media, desviacion_estandar=sd,
                ic95_inferior=media - margen, ic95_superior=media + margen))
    eventos = []
    for escenario in CONFIG_ESCENARIOS:
        grupo = df[df["escenario"] == escenario]
        for columna, etiqueta in [
            ("mes_primer_umbral_alerta_90pct", "réplicas que alcanzan alerta >=90 %"),
            ("mes_primera_sobrecapacidad_100pct", "réplicas que superan capacidad >100 %"),
        ]:
            n_total = len(grupo)
            n_evento = int((grupo[columna].astype(int) > 0).sum())
            p = n_evento / n_total if n_total else 0.0
            z = 1.959963984540054
            den = 1 + z*z/n_total if n_total else 1
            centro = (p + z*z/(2*n_total))/den if n_total else 0
            ancho = (z*np.sqrt(p*(1-p)/n_total + z*z/(4*n_total*n_total))/den) if n_total else 0
            eventos.append(dict(escenario=escenario, indicador=etiqueta,
                n_evento=n_evento, n_total=n_total, proporcion_pct=100*p,
                ic95_inferior_pct=100*max(0, centro-ancho),
                ic95_superior_pct=100*min(1, centro+ancho)))
    salida = pd.DataFrame(filas)
    salida = salida[~salida.metrica.isin(["mes_primer_umbral_alerta_90pct",
                                           "mes_primera_sobrecapacidad_100pct"])]
    return salida, pd.DataFrame(eventos)

def ejecutar(carpeta_salida: Path, replicas: int = N_REPLICAS, meses: int = HORIZONTE_MESES,
            semilla_base: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    filas, trayectorias = [], []
    for idx, escenario in enumerate(CONFIG_ESCENARIOS, start=1):
        for replica in range(1, replicas + 1):
            semilla = semilla_base + idx * 10_000 + replica
            metrica, trayectoria = simular_montecarlo(escenario, semilla, meses, replica)
            filas.append(metrica)
            trayectorias.extend(trayectoria)
    df, df_tray = pd.DataFrame(filas), pd.DataFrame(trayectorias)
    df.to_csv(carpeta_salida / "resultados_reales_mdt.csv", index=False, encoding="utf-8-sig")
    df_tray.to_csv(carpeta_salida / "trayectorias_mdt.csv", index=False, encoding="utf-8-sig")
    resumen, eventos = construir_resumen(df)
    resumen.to_csv(carpeta_salida / "resumen_estadistico_mdt.csv", index=False, encoding="utf-8-sig")
    eventos.to_csv(carpeta_salida / "resumen_eventos_mdt.csv", index=False, encoding="utf-8-sig")
    print(f"Capacidad: {CAPACIDAD_RED_CORREDOR}, stock inicial: {STOCK_INICIAL_RETENIDOS}, alerta: {UMBRAL_ALERTA_SATURACION_PCT}%")
    print(f"Corridas: {len(df)}; registros mensuales: {len(df_tray)}")
    for esc in CONFIG_ESCENARIOS:
        vals = df.loc[df.escenario == esc, "stock_final_retenidos"]
        print(f"{esc}: media stock final {vals.mean():,.2f} ± {vals.std(ddof=1):,.2f}")
    return df, df_tray, resumen

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--salida", default=".")
    parser.add_argument("--replicas", type=int, default=N_REPLICAS)
    parser.add_argument("--meses", type=int, default=HORIZONTE_MESES)
    parser.add_argument("--semilla-base", type=int, default=1000)
    args = parser.parse_args()
    ejecutar(Path(args.salida), args.replicas, args.meses, args.semilla_base)
