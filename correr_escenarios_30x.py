#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de Experimentación Monte Carlo: 30 Réplicas por Escenario - Sistema MDT
=============================================================================
Archivo: correr_escenarios_30x.py
Propósito:
    Ejecutar 30 corridas estocásticas por cada uno de los 4 escenarios de simulación
    del Sistema MDT (Línea Base, Crisis +50%, Cierre de Fronteras, Respuesta Humanitaria),
    variando la semilla pseudoaleatoria (seed=1..30) para modelar la variabilidad
    empírica de flujos y transiciones de acogida/retención.
    
    Genera el archivo CSV consolidado 'resultados_reales_mdt.csv' y reporta en consola
    los estadísticos descriptivos (media y desviación estándar) de las métricas clave.
"""

import os
import sys
import numpy as np
import pandas as pd

# Definición canónica de los 4 escenarios y sus parámetros de gobernanza (de backend/main.py)
CONFIG_ESCENARIOS = {
    "Línea Base": {
        "flujo_base": 10000,
        "tasa_integracion": 0.65,
        "tasa_retencion": 0.35,
        "factor_crecimiento": 0.015,
        "despeje_retenidos": 0.15
    },
    "Crisis +50%": {
        "flujo_base": 15000,
        "tasa_integracion": 0.45,
        "tasa_retencion": 0.55,
        "factor_crecimiento": 0.035,
        "despeje_retenidos": 0.08
    },
    "Cierre de Fronteras": {
        "flujo_base": 5500,
        "tasa_integracion": 0.22,
        "tasa_retencion": 0.78,
        "factor_crecimiento": -0.010,
        "despeje_retenidos": 0.04
    },
    "Respuesta Humanitaria": {
        "flujo_base": 11500,
        "tasa_integracion": 0.85,
        "tasa_retencion": 0.15,
        "factor_crecimiento": 0.005,
        "despeje_retenidos": 0.30
    }
}

CAPACIDAD_RED_CORREDOR = 18500  # Capacidad máxima agregada (5 nodos)
STOCK_INICIAL_RETENIDOS = 1200   # Stock inicial en albergues al inicio de la proyección
UMBRAL_COLAPSO_ALBERGUES = 250  # Capacidad crítica de contingencia en albergue de triaje
HORIZONTE_MESES = 12            # Horizonte temporal estándar (12 meses)
N_REPLICAS = 30                 # Número de réplicas independientes por escenario


def simular_corrida_estocastica(
    escenario_nombre: str,
    params: dict,
    seed: int,
    meses: int = HORIZONTE_MESES
) -> dict:
    """
    Ejecuta una corrida estocástica individual parametrizada con una semilla aleatoria.
    Modela:
    - Inflow mensual con componente determinista (estacionalidad sinusoidal + tendencia)
      más perturbación estocástica gaussiana (σ = 4% del flujo medio).
    - Asignación de transiciones de integración/retención mediante distribución Binomial.
    - Despeje mensual de albergues mediante proceso binomial estocástico.
    """
    rng = np.random.default_rng(seed)

    stock_retenidos = STOCK_INICIAL_RETENIDOS
    acumulado_integrados = 0
    acumulado_entrantes = 0
    mes_colapso_250 = None
    pico_retenidos = stock_retenidos
    saturaciones_mensuales = []

    for m in range(1, meses + 1):
        # 1. Componente determinista de flujo entrante
        variacion_estacional = np.sin((m / 12.0) * 2.0 * np.pi) * 0.08
        factor_tendencia = (1.0 + params["factor_crecimiento"]) ** (m - 1)
        flujo_esperado = params["flujo_base"] * factor_tendencia * (1.0 + variacion_estacional)

        # 2. Ruido estocástico gaussiano en la afluencia fronteriza
        flujo_mes = int(rng.normal(loc=flujo_esperado, scale=flujo_esperado * 0.04))
        flujo_mes = max(flujo_mes, 500)

        # 3. Transiciones individuales estocásticas (Proceso Binomial)
        nuevos_integrados = int(rng.binomial(n=flujo_mes, p=params["tasa_integracion"]))
        nuevos_retenidos = flujo_mes - nuevos_integrados

        # 4. Despeje paulatino estocástico de retenidos acumulados
        if stock_retenidos > 0:
            despeje = int(rng.binomial(n=stock_retenidos, p=params["despeje_retenidos"]))
        else:
            despeje = 0

        stock_retenidos = max(0, stock_retenidos + nuevos_retenidos - despeje)
        acumulado_integrados += (nuevos_integrados + despeje)
        acumulado_entrantes += flujo_mes

        if stock_retenidos > pico_retenidos:
            pico_retenidos = stock_retenidos

        if stock_retenidos > UMBRAL_COLAPSO_ALBERGUES and mes_colapso_250 is None:
            mes_colapso_250 = m

        sat_mes = min(100.0, round((stock_retenidos / CAPACIDAD_RED_CORREDOR) * 100.0, 2))
        saturaciones_mensuales.append(sat_mes)

    saturacion_final = saturaciones_mensuales[-1]
    saturacion_media = round(float(np.mean(saturaciones_mensuales)), 2)
    tasa_int_real = round((acumulado_integrados / acumulado_entrantes) * 100.0, 2)
    tasa_ret_real = round((stock_retenidos / acumulado_entrantes) * 100.0, 2)

    return {
        "escenario": escenario_nombre,
        "replica_id": seed,
        "semilla_aleatoria": seed,
        "meses_evaluados": meses,
        "total_acumulado_entrantes": acumulado_entrantes,
        "total_acumulado_integrados": acumulado_integrados,
        "stock_final_retenidos": stock_retenidos,
        "pico_maximo_retenidos": pico_retenidos,
        "saturacion_final_pct": saturacion_final,
        "saturacion_media_pct": saturacion_media,
        "tasa_integracion_real_pct": tasa_int_real,
        "tasa_retencion_final_pct": tasa_ret_real,
        "mes_primer_colapso_250": mes_colapso_250 if mes_colapso_250 is not None else -1
    }


def ejecutar_experimento_completo(ruta_salida_csv: str = "resultados_reales_mdt.csv") -> pd.DataFrame:
    """
    Ejecuta el ciclo de 30 corridas independientes por cada uno de los 4 escenarios,
    consolida los registros, exporta el CSV y genera la tabla analítica resumen.
    """
    registros = []
    print(f"\n{'='*84}")
    print(f" EJECUCIÓN EXPERIMENTAL MONTE CARLO: 30 RÉPLICAS POR ESCENARIO (SISTEMA MDT)")
    print(f"{'='*84}")
    print(f"Escenarios a evaluar: {list(CONFIG_ESCENARIOS.keys())}")
    print(f"Réplicas por escenario: {N_REPLICAS} | Total corridas: {len(CONFIG_ESCENARIOS) * N_REPLICAS}")
    print(f"Horizonte: {HORIZONTE_MESES} meses | Capacidad de red: {CAPACIDAD_RED_CORREDOR:,} personas")
    print(f"{'='*84}\n")

    for nombre_esc, params in CONFIG_ESCENARIOS.items():
        print(f"-> Simulando escenario '{nombre_esc}' (30 réplicas con seed 1..30)...")
        for replica in range(1, N_REPLICAS + 1):
            resultado = simular_corrida_estocastica(
                escenario_nombre=nombre_esc,
                params=params,
                seed=replica,
                meses=HORIZONTE_MESES
            )
            registros.append(resultado)

    df_resultados = pd.DataFrame(registros)

    # Guardar en archivo CSV
    df_resultados.to_csv(ruta_salida_csv, index=False, encoding="utf-8-sig")
    print(f"\n[OK] Datos crudos exportados exitosamente a: '{os.path.abspath(ruta_salida_csv)}'")
    print(f"     Dimensiones del archivo: {df_resultados.shape[0]} filas x {df_resultados.shape[1]} columnas.\n")

    # =========================================================================
    # REPORTE ESTADÍSTICO EN CONSOLA: PROMEDIO Y DESVIACIÓN ESTÁNDAR POR ESCENARIO
    # =========================================================================
    metricas_analizadas = [
        ("total_acumulado_entrantes", "Total Entrantes (pers.)"),
        ("total_acumulado_integrados", "Total Integrados (pers.)"),
        ("stock_final_retenidos", "Stock Final Retenidos"),
        ("pico_maximo_retenidos", "Pico Máximo Retenidos"),
        ("saturacion_final_pct", "Saturación Final (%)"),
        ("saturacion_media_pct", "Saturación Media (%)"),
        ("tasa_integracion_real_pct", "Tasa Integración Real (%)")
    ]

    print(f"{'='*100}")
    print(f"{'REPORTE TÉCNICO FORMAL: MEDIA Y DESVIACIÓN ESTÁNDAR POR ESCENARIO (N=30)':^100}")
    print(f"{'='*100}")

    for esc_name in CONFIG_ESCENARIOS.keys():
        sub_df = df_resultados[df_resultados["escenario"] == esc_name]
        print(f"\n--------------------------------------------------------------------------------")
        print(f" ESCENARIO: {esc_name.upper()} (N = 30 réplicas)")
        print(f"--------------------------------------------------------------------------------")
        print(f"{'Métrica Analizada':<35} | {'Promedio (Media)':<18} | {'Desv. Estándar (Std)':<18}")
        print(f"{'-'*35}-+-{'-'*18}-+-{'-'*18}")

        for col, etiqueta in metricas_analizadas:
            media = sub_df[col].mean()
            desv = sub_df[col].std()
            if "pct" in col or "(%)" in etiqueta:
                print(f"{etiqueta:<35} | {media:>16.2f}% | {desv:>16.2f}%")
            else:
                print(f"{etiqueta:<35} | {media:>18,.1f} | {desv:>18,.1f}")

        # Análisis específico del umbral de colapso de 250 personas
        colapsos = sub_df[sub_df["mes_primer_colapso_250"] > 0]
        tasa_colapso = (len(colapsos) / len(sub_df)) * 100.0
        mes_medio_colapso = colapsos["mes_primer_colapso_250"].mean() if len(colapsos) > 0 else 0
        print(f"{'Tasa Colapso Albergues (>250)':<35} | {tasa_colapso:>16.1f}% | Mes {mes_medio_colapso:.1f} promedio")

    print(f"\n{'='*100}")
    print(f" Fin del experimento. Resultados listos para redacción científica.")
    print(f"{'='*100}\n")

    return df_resultados


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ejecutar_experimento_completo("resultados_reales_mdt.csv")
