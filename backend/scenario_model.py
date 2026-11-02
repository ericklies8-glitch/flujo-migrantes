# -*- coding: utf-8 -*-
"""Modelo agregado compartido por API y Monte Carlo.

Los resultados son simulaciones, no mediciones de migrantes reales.
No hay agentes individuales ni decisiones calculadas de ruta.
"""
from __future__ import annotations
from math import floor, sin, pi
from typing import Any
import numpy as np

CONFIG_ESCENARIOS: dict[str, dict[str, float | int]] = {
    "Línea Base": {"flujo_base": 10000, "tasa_integracion": 0.65, "factor_crecimiento": 0.015, "despeje_retenidos": 0.15},
    "Crisis +50%": {"flujo_base": 15000, "tasa_integracion": 0.45, "factor_crecimiento": 0.035, "despeje_retenidos": 0.08},
    "Cierre de Fronteras": {"flujo_base": 5500, "tasa_integracion": 0.22, "factor_crecimiento": -0.010, "despeje_retenidos": 0.04},
    "Respuesta Humanitaria": {"flujo_base": 11500, "tasa_integracion": 0.85, "factor_crecimiento": 0.005, "despeje_retenidos": 0.30},
}
CAPACIDADES_NODOS = {"Huaquillas": 1500, "Tumbes": 2500, "Piura": 3000, "Trujillo": 3500, "Lima": 8500}
CAPACIDAD_RED_CORREDOR = sum(CAPACIDADES_NODOS.values())
STOCK_INICIAL_RETENIDOS = 1200
UMBRAL_ALERTA_SATURACION_PCT = 90.0


def flujo_esperado_mes(params: dict[str, Any], mes: int) -> int:
    variacion_estacional = sin((mes / 12.0) * 2.0 * pi) * 0.08
    factor_tendencia = (1.0 + float(params["factor_crecimiento"])) ** (mes - 1)
    estimado = floor(float(params["flujo_base"]) * factor_tendencia * (1.0 + variacion_estacional))
    return max(500, int(estimado))


def porcentaje_saturacion(stock_retenidos: int | float) -> float:
    return 100.0 * float(stock_retenidos) / CAPACIDAD_RED_CORREDOR


def simular_determinista(escenario: str, meses: int) -> dict[str, Any]:
    if escenario not in CONFIG_ESCENARIOS:
        raise ValueError(f"Escenario no reconocido: {escenario}")
    if not 1 <= int(meses) <= 24:
        raise ValueError("El horizonte debe estar entre 1 y 24 meses.")
    params = CONFIG_ESCENARIOS[escenario]
    stock = STOCK_INICIAL_RETENIDOS
    acumulado_entrantes = acumulado_integrados = acumulado_integrados_directos = acumulado_despejados = 0
    serie: list[dict[str, Any]] = []
    for mes in range(1, int(meses) + 1):
        flujo = flujo_esperado_mes(params, mes)
        nuevos_integrados = int(flujo * float(params["tasa_integracion"]))
        nuevos_retenidos = flujo - nuevos_integrados
        despeje = int(stock * float(params["despeje_retenidos"]))
        stock = max(0, stock + nuevos_retenidos - despeje)
        acumulado_entrantes += flujo
        acumulado_integrados_directos += nuevos_integrados
        acumulado_despejados += despeje
        acumulado_integrados += nuevos_integrados + despeje
        sat = porcentaje_saturacion(stock)
        serie.append({
            "mes": mes, "etiqueta_mes": f"Mes {mes:02d}",
            "flujo_entrante": int(flujo), "integrados_directos_mes": int(nuevos_integrados),
            "retenidos_nuevos_mes": int(nuevos_retenidos), "despeje_stock_previo": int(despeje),
            "retenidos_mes": int(nuevos_retenidos), "integrados_mes": int(nuevos_integrados + despeje),
            "retenidos_acumulados": int(stock), "integrados_acumulados": int(acumulado_integrados),
            "saturacion_corredor_pct": round(sat, 2),
            "alerta_90pct": bool(sat >= UMBRAL_ALERTA_SATURACION_PCT),
            "sobrecapacidad": bool(stock > CAPACIDAD_RED_CORREDOR),
        })
    return {
        "escenario": escenario, "meses_evaluados": int(meses),
        "tasa_integracion_parametrizada_pct": 100.0 * float(params["tasa_integracion"]),
        "tasa_retencion_parametrizada_pct": 100.0 * (1.0 - float(params["tasa_integracion"])),
        "total_acumulado_entrantes": acumulado_entrantes,
        "total_acumulado_retenidos": stock, "total_acumulado_integrados": acumulado_integrados,
        "total_integrados_directos": acumulado_integrados_directos,
        "total_despejados": acumulado_despejados, "serie_temporal": serie,
    }


def simular_montecarlo(escenario: str, semilla: int, meses: int = 12, replica_id: int | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if escenario not in CONFIG_ESCENARIOS:
        raise ValueError(f"Escenario no reconocido: {escenario}")
    if not 1 <= int(meses) <= 24:
        raise ValueError("El horizonte debe estar entre 1 y 24 meses.")
    params = CONFIG_ESCENARIOS[escenario]
    rng = np.random.default_rng(int(semilla))
    stock = STOCK_INICIAL_RETENIDOS
    entradas_total = integrados_directos_total = despejados_total = 0
    stock_pico = stock
    primer_alerta = primera_sobrecapacidad = None
    trayectoria: list[dict[str, Any]] = []
    for mes in range(1, int(meses) + 1):
        esperado = flujo_esperado_mes(params, mes)
        flujo = max(500, int(rng.normal(loc=esperado, scale=esperado * 0.04)))
        nuevos_integrados = int(rng.binomial(n=flujo, p=float(params["tasa_integracion"])))
        nuevos_retenidos = flujo - nuevos_integrados
        despeje = int(rng.binomial(n=stock, p=float(params["despeje_retenidos"]))) if stock > 0 else 0
        stock = max(0, stock + nuevos_retenidos - despeje)
        entradas_total += flujo
        integrados_directos_total += nuevos_integrados
        despejados_total += despeje
        stock_pico = max(stock_pico, stock)
        sat = porcentaje_saturacion(stock)
        if primer_alerta is None and sat >= UMBRAL_ALERTA_SATURACION_PCT:
            primer_alerta = mes
        if primera_sobrecapacidad is None and stock > CAPACIDAD_RED_CORREDOR:
            primera_sobrecapacidad = mes
        trayectoria.append({
            "escenario": escenario, "replica_id": int(replica_id if replica_id is not None else semilla),
            "semilla_aleatoria": int(semilla), "mes": mes, "flujo_entrante": int(flujo),
            "integrados_directos_mes": int(nuevos_integrados), "retenidos_nuevos_mes": int(nuevos_retenidos),
            "despeje_stock_previo": int(despeje), "stock_retenidos": int(stock),
            "integrados_acumulados": int(integrados_directos_total + despejados_total),
            "saturacion_red_pct": round(sat, 4),
            "alerta_saturacion_90pct": bool(sat >= UMBRAL_ALERTA_SATURACION_PCT),
            "sobrecapacidad_mayor_100pct": bool(stock > CAPACIDAD_RED_CORREDOR),
        })
    integrados_total = integrados_directos_total + despejados_total
    metricas = {
        "escenario": escenario,
        "replica_id": int(replica_id if replica_id is not None else semilla),
        "semilla_aleatoria": int(semilla),
        "meses_evaluados": int(meses),
        "stock_inicial_retenidos": STOCK_INICIAL_RETENIDOS,
        "capacidad_red_personas": CAPACIDAD_RED_CORREDOR,
        "flujo_base_personas_mes": int(params["flujo_base"]),
        "tasa_integracion_parametrizada_pct": round(100.0 * float(params["tasa_integracion"]), 4),
        "tasa_retencion_parametrizada_pct": round(100.0 * (1.0 - float(params["tasa_integracion"])), 4),
        "factor_crecimiento_mensual": float(params["factor_crecimiento"]),
        "despeje_retenidos": float(params["despeje_retenidos"]),
        "total_acumulado_entrantes": int(entradas_total),
        "total_integrados_directos": int(integrados_directos_total),
        "total_despejados_stock": int(despejados_total),
        "total_acumulado_integrados": int(integrados_total),
        "stock_final_retenidos": int(stock), "pico_maximo_retenidos": int(stock_pico),
        "saturacion_final_pct": round(porcentaje_saturacion(stock), 4),
        "saturacion_media_pct": round(float(np.mean([r["saturacion_red_pct"] for r in trayectoria])), 4),
        "tasa_integracion_directa_pct": round(100.0 * integrados_directos_total / entradas_total, 4) if entradas_total else 0.0,
        "tasa_retencion_entradas_pct": round(100.0 * (entradas_total - integrados_directos_total) / entradas_total, 4) if entradas_total else 0.0,
        "porcentaje_salida_total_sobre_stock_inicial_mas_entradas": round(100.0 * integrados_total / (STOCK_INICIAL_RETENIDOS + entradas_total), 4) if entradas_total else 0.0,
        "stock_final_sobre_capacidad_pct": round(100.0 * stock / CAPACIDAD_RED_CORREDOR, 4),
        "mes_primer_umbral_alerta_90pct": primer_alerta if primer_alerta is not None else -1,
        "mes_primera_sobrecapacidad_100pct": primera_sobrecapacidad if primera_sobrecapacidad is not None else -1,
    }
    return metricas, trayectoria
