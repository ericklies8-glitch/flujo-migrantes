# -*- coding: utf-8 -*-
"""
Servicio Backend FastAPI - Sistema MDT (Gemelo Digital de Flujo Migratorio)
==========================================================================
Módulo: backend/main.py
Propósito:
    Servidor de aplicaciones desacoplado para la exposición de inferencia en tiempo real,
    analítica CRISP-DM, telemetría geoespacial (PyDeck) y simulaciones de dinámica de sistemas.

Uso:
    uvicorn backend.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager
import datetime
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException, status
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
import joblib
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd

from backend.schemas import (
    HealthResponse,
    MigranteInput,
    NodoTelemetria,
    PrediccionOutput,
    PuntoSimulacion,
    SimulacionRequest,
    SimulacionResponse,
    TelemetriaCorredorResponse,
)

# Configuración de Logging profesional
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Backend_MDT")

# ==============================================================================
# Rutas Canónicas del Sistema
# ==============================================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUTA_MODELO = os.path.join(BASE_DIR, "motor_ml", "modelo_calibrado.joblib")
RUTA_METRICAS = os.path.join(BASE_DIR, "motor_ml", "metricas_crispdm.json")
RUTA_EDA = os.path.join(BASE_DIR, "motor_ml", "eda_results.json")
RUTA_EDA_ROOT = os.path.join(BASE_DIR, "eda_results.json")


# ==============================================================================
# Gestión del Ciclo de Vida de la Aplicación (Lifespan)
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Inicializa los artefactos analíticos de Machine Learning y los almacena
    en memoria compartida en app.state para latencia ultra-baja en inferencia.
    """
    logger.info(">>> Iniciando servicio MDT API...")

    # 1. Carga del Modelo de Machine Learning Calibrado
    app.state.modelo = None
    if os.path.exists(RUTA_MODELO):
        try:
            app.state.modelo = joblib.load(RUTA_MODELO)
            logger.info(f" Pipeline ML cargado exitosamente desde: {RUTA_MODELO}")
        except Exception as e:
            logger.error(f" Error crítico cargando modelo {RUTA_MODELO}: {e}")
    else:
        logger.warning(f" Archivo de modelo no encontrado en: {RUTA_MODELO}")

    # 2. Carga de Métricas CRISP-DM
    app.state.metricas_crispdm = {}
    if os.path.exists(RUTA_METRICAS):
        try:
            with open(RUTA_METRICAS, "r", encoding="utf-8") as f:
                app.state.metricas_crispdm = json.load(f)
            logger.info(f" Métricas CRISP-DM cargadas desde: {RUTA_METRICAS}")
        except Exception as e:
            logger.error(f" Error cargando {RUTA_METRICAS}: {e}")
    else:
        logger.warning(f" Métricas no encontradas en: {RUTA_METRICAS}")

    # 3. Carga de Resultados EDA
    app.state.eda_results = {}
    eda_path_to_use = RUTA_EDA if os.path.exists(RUTA_EDA) else RUTA_EDA_ROOT
    if os.path.exists(eda_path_to_use):
        try:
            with open(eda_path_to_use, "r", encoding="utf-8") as f:
                app.state.eda_results = json.load(f)
            logger.info(f" Resultados EDA cargados desde: {eda_path_to_use}")
        except Exception as e:
            logger.error(f" Error cargando EDA desde {eda_path_to_use}: {e}")
    else:
        logger.warning(" Archivo eda_results.json no encontrado.")

    yield

    logger.info(">>> Finalizando servicio MDT API y liberando recursos.")


# ==============================================================================
# Inicialización de la Aplicación FastAPI
# ==============================================================================

app = FastAPI(
    title="Sistema MDT - API del Gemelo Digital de Flujo Migratorio",
    description=(
        "Microservicio analítico desacoplado para inferencia predictiva, telemetría "
        "geoespacial de corredores humanitarios y simulación de dinámica de sistemas."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Configuración de CORS para clientes web (React, Next.js, Dashboards locales)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv("MDT_CORS_ORIGINS", "http://localhost:8501").split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Endpoints de Diagnóstico y Salud
# ==============================================================================

@app.get(
    "/",
    tags=["Diagnóstico"],
    summary="Ruta raíz informativa"
)
async def raiz() -> Dict[str, Any]:
    """
    Ruta raíz de bienvenida con enlaces a la documentación interactiva.
    """
    return {
        "proyecto": "Sistema MDT - Gemelo Digital de Flujo Migratorio",
        "descripcion": "Motor Analítico de Machine Learning y Gemelo Digital",
        "version": "1.0.0",
        "documentacion_swagger": "/docs",
        "documentacion_redoc": "/redoc",
        "endpoints_principales": [
            "GET  /health",
            "POST /predict",
            "GET  /metrics/crispdm",
            "GET  /eda/summary",
            "GET  /corridor/telemetry",
            "POST /corridor/simulate"
        ]
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Diagnóstico"],
    summary="Estado de salud del servicio y componentes ML"
)
async def check_health() -> HealthResponse:
    """
    Verifica el estado de funcionamiento del servidor y valida si los modelos
    y artefactos CRISP-DM se encuentran correctamente cargados en memoria.
    """
    modelo_cargado = app.state.modelo is not None
    artefactos = {
        "modelo_calibrado_joblib": os.path.exists(RUTA_MODELO) and modelo_cargado,
        "metricas_crispdm_json": bool(app.state.metricas_crispdm),
        "eda_results_json": bool(app.state.eda_results)
    }

    estado_general = "healthy" if modelo_cargado else "degraded"

    return HealthResponse(
        status=estado_general,
        version="1.0.0",
        modelo_cargado=modelo_cargado,
        artefactos_disponibles=artefactos,
        timestamp=datetime.datetime.now().isoformat()
    )


# ==============================================================================
# Endpoint de Inferencia Predictiva Individual (Machine Learning)
# ==============================================================================

def categorizar_riesgo(probabilidad: float) -> str:
    """
    Clasifica el nivel de riesgo/vulnerabilidad y celeridad de tránsito según la probabilidad.
    """
    if probabilidad >= 70.0:
        return "Bajo Riesgo (Tránsito Ágil / Alta Movilidad)"
    elif probabilidad >= 40.0:
        return "Riesgo Moderado (Tránsito Regular)"
    else:
        return "Alto Riesgo (Tránsito Lento / Alta Vulnerabilidad)"


@app.post(
    "/predict",
    response_model=PrediccionOutput,
    tags=["Machine Learning"],
    summary="Inferencia predictiva individual de tránsito rápido"
)
async def predecir_transito_migrante(migrante: MigranteInput) -> PrediccionOutput:
    """
    Recibe los atributos socio-demográficos y económicos del migrante, ejecuta
    la inferencia sobre el pipeline serializado en `motor_ml/modelo_calibrado.joblib`
    y retorna la probabilidad estimada, categoría de riesgo y latencia de respuesta en ms.
    """
    if app.state.modelo is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo de machine learning no se encuentra cargado en memoria."
        )

    t_inicio = time.perf_counter()

    # Construcción del DataFrame respetando exactamente los nombres de columna del pipeline
    registro_dict = {
        "Edad": [migrante.edad],
        "Recursos_USD": [migrante.recursos_usd],
        "Salud": [migrante.salud],
        "Nivel_Educativo": [migrante.nivel_educativo],
        "Vínculos_Familiares": [migrante.vinculos_familiares],
        "Documentacion": [migrante.documentacion],
        "Género": [migrante.genero]
    }
    df_eval = pd.DataFrame(registro_dict)

    try:
        pred_array = app.state.modelo.predict(df_eval)
        prob_estimada = float(np.clip(pred_array[0], 0.0, 100.0))
    except Exception as e:
        logger.error(f"Falla durante la predicción: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno durante la inferencia predictiva: {str(e)}"
        )

    t_final = time.perf_counter()
    tiempo_ms = round((t_final - t_inicio) * 1000.0, 3)
    categoria = categorizar_riesgo(prob_estimada)

    return PrediccionOutput(
        probabilidad_calculada=round(prob_estimada, 2),
        categoria_riesgo=categoria,
        tiempo_respuesta_ms=tiempo_ms,
        detalles_entrada={
            "edad": migrante.edad,
            "recursos_usd": migrante.recursos_usd,
            "salud": migrante.salud,
            "vinculos_familiares": migrante.vinculos_familiares,
            "documentacion": migrante.documentacion,
            "genero": migrante.genero,
            "nivel_educativo": migrante.nivel_educativo
        }
    )


# ==============================================================================
# Endpoints de Métricas CRISP-DM y Análisis Exploratorio (EDA)
# ==============================================================================

@app.get(
    "/metrics/crispdm",
    tags=["Machine Learning", "CRISP-DM"],
    summary="Métricas de Benchmarking, 5 Folds, Test de Wilcoxon e Intervalo de Confianza"
)
async def obtener_metricas_crispdm() -> Dict[str, Any]:
    """
    Retorna el informe completo de validación formal del modelo:
    - Benchmarking de modelos base (RMSE, MAE, R²).
    - Métricas detalladas de los 5 Folds de validación cruzada.
    - Mejores hiperparámetros del modelo ganador (XGBoost).
    - Pruebas estadísticas formales (Wilcoxon p-value, T-Student pareada p-value).
    - Intervalo de confianza al 95% para R² calculado mediante Bootstrapping.
    """
    if not app.state.metricas_crispdm:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Las métricas CRISP-DM no se encuentran disponibles en el servidor."
        )
    return app.state.metricas_crispdm


@app.get(
    "/eda/summary",
    tags=["Machine Learning", "EDA"],
    summary="Resumen estadístico y matrices de correlación univariada y multivariada"
)
async def obtener_resumen_eda() -> Dict[str, Any]:
    """
    Retorna el compendio analítico de datos:
    - Resumen estadístico (media, desviación, cuartiles, extremos).
    - Matriz de correlación de Pearson y Spearman frente al target.
    - Conteos demográficos de las variables cualitativas.
    """
    if not app.state.eda_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="El compendio exploratorio de datos (EDA) no se encuentra disponible."
        )
    return app.state.eda_results


# ==============================================================================
# Endpoint de Telemetría Geoespacial de Nodos (PyDeck)
# ==============================================================================

@app.get(
    "/corridor/telemetry",
    response_model=TelemetriaCorredorResponse,
    tags=["Gemelo Digital", "Telemetría"],
    summary="Telemetría en tiempo real de los 5 nodos estratégicos para visualización PyDeck"
)
async def obtener_telemetria_corredor() -> TelemetriaCorredorResponse:
    """
    Retorna el estado operativo y nivel de saturación de los 5 nodos estratégicos
    del corredor migratorio norte-centro de Perú:
    1. Huaquillas (Frontera norte Ecuador-Perú)
    2. Tumbes (Puesto fronterizo y triaje inicial)
    3. Piura (Nodo de distribución regional)
    4. Trujillo (Nodo intermedio costa norte)
    5. Lima (Nodo metropolitano / receptor principal)

    Incluye coordenadas geográficas y colores hexadecimal/RGBA listos para PyDeck.
    """
    # Definición de nodos canónicos con geolocalización y capacidades
    nodos_config = [
        {
            "id": "NODO-HUA",
            "nombre": "Huaquillas (Frontera)",
            "latitud": -3.4772,
            "longitud": -80.2294,
            "capacidad_maxima": 1500,
            "migrantes_actuales": 1280
        },
        {
            "id": "NODO-TUM",
            "nombre": "Tumbes (Triaje)",
            "latitud": -3.5669,
            "longitud": -80.4515,
            "capacidad_maxima": 2500,
            "migrantes_actuales": 1920
        },
        {
            "id": "NODO-PIU",
            "nombre": "Piura (Distribución)",
            "latitud": -5.1945,
            "longitud": -80.6328,
            "capacidad_maxima": 3000,
            "migrantes_actuales": 1650
        },
        {
            "id": "NODO-TRU",
            "nombre": "Trujillo (Intermedio)",
            "latitud": -8.1116,
            "longitud": -79.0287,
            "capacidad_maxima": 3500,
            "migrantes_actuales": 2170
        },
        {
            "id": "NODO-LIM",
            "nombre": "Lima (Metropolitano)",
            "latitud": -12.0464,
            "longitud": -77.0428,
            "capacidad_maxima": 8500,
            "migrantes_actuales": 7140
        }
    ]

    nodos_resultado: List[NodoTelemetria] = []
    total_capacidad = 0
    total_migrantes = 0

    for item in nodos_config:
        cap = item["capacidad_maxima"]
        act = item["migrantes_actuales"]
        sat_pct = round((act / cap) * 100.0, 1)

        total_capacidad += cap
        total_migrantes += act

        # Asignación de semáforo operativo y código de color PyDeck
        if sat_pct >= 80.0:
            estado = "Crítico"
            color_hex = "#EF4444"  # Rojo
            color_rgb = [239, 68, 68, 210]
        elif sat_pct >= 60.0:
            estado = "Alerta"
            color_hex = "#F59E0B"  # Ámbar / Amarillo
            color_rgb = [245, 158, 11, 210]
        else:
            estado = "Normal"
            color_hex = "#10B981"  # Verde
            color_rgb = [16, 185, 129, 210]

        nodos_resultado.append(
            NodoTelemetria(
                id=item["id"],
                nombre=item["nombre"],
                latitud=item["latitud"],
                longitud=item["longitud"],
                capacidad_maxima=cap,
                migrantes_actuales=act,
                nivel_saturacion_pct=sat_pct,
                estado_operativo=estado,
                color_hex=color_hex,
                color_rgb=color_rgb
            )
        )

    saturacion_global = round((total_migrantes / total_capacidad) * 100.0, 1)

    return TelemetriaCorredorResponse(
        timestamp=datetime.datetime.now().isoformat(),
        total_nodos=len(nodos_resultado),
        capacidad_total_red=total_capacidad,
        migrantes_totales_red=total_migrantes,
        saturacion_promedio_red_pct=saturacion_global,
        nodos=nodos_resultado
    )


# ==============================================================================
# Endpoint de Simulación de Dinámica de Sistemas (Corredor Migratorio)
# ==============================================================================

@app.post(
    "/corridor/simulate",
    response_model=SimulacionResponse,
    tags=["Gemelo Digital", "Simulación"],
    summary="Proyección temporal de stocks y flujos ('Retenidos' vs 'Integrados')"
)
async def simular_dinamica_corredor(solicitud: SimulacionRequest) -> SimulacionResponse:
    """
    Ejecuta el motor de dinámica de sistemas para proyectar la evolución mensual
    del stock de migrantes 'Retenidos' y el flujo acumulado de 'Integrados' bajo
    distintos escenarios sociopolíticos:
    - 'Línea Base'
    - 'Crisis +50%'
    - 'Cierre de Fronteras'
    - 'Respuesta Humanitaria'
    """
    escenario = solicitud.escenario
    meses = solicitud.meses

    # Parámetros de gobernanza según escenario
    config_escenarios = {
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

    params = config_escenarios.get(escenario, config_escenarios["Línea Base"])

    serie_temporal: List[PuntoSimulacion] = []
    stock_retenidos = 1200  # Stock inicial existente en el corredor
    acumulado_integrados = 0
    acumulado_entrantes = 0
    capacidad_red_corredor = 18500  # Capacidad máxima agregada

    for m in range(1, meses + 1):
        # Flujo entrante con factor estacional y tendencia
        variacion_estacional = np.sin((m / 12.0) * 2.0 * np.pi) * 0.08
        factor_tendencia = (1.0 + params["factor_crecimiento"]) ** (m - 1)
        flujo_mes = int(params["flujo_base"] * factor_tendencia * (1.0 + variacion_estacional))
        flujo_mes = max(flujo_mes, 500)

        # Flujo de nuevos retenidos e integrados del mes
        nuevos_integrados = int(flujo_mes * params["tasa_integracion"])
        nuevos_retenidos = int(flujo_mes * params["tasa_retencion"])

        # Despeje o tránsito paulatino de retenidos acumulados anteriores
        despeje = int(stock_retenidos * params["despeje_retenidos"])
        stock_retenidos = max(0, stock_retenidos + nuevos_retenidos - despeje)
        acumulado_integrados += (nuevos_integrados + despeje)
        acumulado_entrantes += flujo_mes

        # Tasa de saturación del corredor frente al stock retenido activo
        saturacion_pct = min(100.0, round((stock_retenidos / capacidad_red_corredor) * 100.0, 2))

        serie_temporal.append(
            PuntoSimulacion(
                mes=m,
                etiqueta_mes=f"Mes {m:02d}",
                flujo_entrante=flujo_mes,
                retenidos_mes=nuevos_retenidos,
                integrados_mes=nuevos_integrados + despeje,
                retenidos_acumulados=stock_retenidos,
                integrados_acumulados=acumulado_integrados,
                saturacion_corredor_pct=saturacion_pct
            )
        )

    return SimulacionResponse(
        escenario=escenario,
        meses_evaluados=meses,
        tasa_integracion_estimada_pct=round(params["tasa_integracion"] * 100.0, 1),
        tasa_retencion_estimada_pct=round(params["tasa_retencion"] * 100.0, 1),
        total_acumulado_entrantes=acumulado_entrantes,
        total_acumulado_retenidos=stock_retenidos,
        total_acumulado_integrados=acumulado_integrados,
        serie_temporal=serie_temporal
    )
