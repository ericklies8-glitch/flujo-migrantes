# -*- coding: utf-8 -*-
"""
Esquemas Pydantic - Sistema MDT (Gemelo Digital de Flujo Migratorio)
===================================================================
Módulo: backend/schemas.py
Propósito:
    Definición de modelos de validación, serialización y contratos de datos
    para los endpoints del servicio FastAPI.
"""

from typing import Any, Dict, List, Literal, Optional
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field, field_validator


# ==============================================================================
# 1. Esquema de Inferencia Individual (MigranteInput)
# ==============================================================================

class MigranteInput(BaseModel):
    """
    Modelo de entrada para la inferencia individual de probabilidad de tránsito rápido.
    Valida rangos biológicos, financieros y categorías socio-demográficas.
    """
    edad: int = Field(
        ...,
        ge=18,
        le=90,
        description="Edad cronológica del migrante en años (rango permitido: 18 a 90 años)",
        examples=[28]
    )
    salud: Literal["Buena", "Regular", "Crónica"] = Field(
        ...,
        description="Estado clínico o de salud del migrante ('Buena', 'Regular', 'Crónica')",
        examples=["Buena"]
    )
    recursos_usd: float = Field(
        ...,
        ge=0.0,
        description="Fondos económicos disponibles en dólares estadounidenses (USD >= 0)",
        examples=[180.0]
    )
    vinculos_familiares: Literal["Sí (Lima)", "Sí (Trujillo)", "No"] = Field(
        ...,
        description="Red de apoyo y lazos familiares directos en territorio peruano",
        examples=["Sí (Lima)"]
    )
    documentacion: Literal["Pasaporte", "Cédula", "Sin Papeles"] = Field(
        ...,
        description="Estatus de identificación y documentación migratoria",
        examples=["Pasaporte"]
    )
    genero: Literal["Masculino", "Femenino"] = Field(
        ...,
        description="Identidad de género biológico ('Masculino', 'Femenino')",
        examples=["Femenino"]
    )
    nivel_educativo: Literal["Básico", "Secundaria", "Técnico", "Universitario"] = Field(
        ...,
        description="Nivel formativo o grado de instrucción alcanzado",
        examples=["Universitario"]
    )

    @field_validator("salud", mode="before")
    @classmethod
    def normalizar_salud(cls, v: Any) -> Any:
        if isinstance(v, str):
            s = v.strip().capitalize()
            if s in ["Cronica", "Crónica"]:
                return "Crónica"
            return s
        return v

    @field_validator("vinculos_familiares", mode="before")
    @classmethod
    def normalizar_vinculos(cls, v: Any) -> Any:
        if isinstance(v, str):
            s = v.strip()
            if s.lower() in ["si (lima)", "sí (lima)"]:
                return "Sí (Lima)"
            if s.lower() in ["si (trujillo)", "sí (trujillo)"]:
                return "Sí (Trujillo)"
            if s.lower() == "no":
                return "No"
        return v

    @field_validator("documentacion", mode="before")
    @classmethod
    def normalizar_documentacion(cls, v: Any) -> Any:
        if isinstance(v, str):
            s = v.strip().capitalize()
            if s in ["Cedula", "Cédula"]:
                return "Cédula"
            if s.lower() == "sin papeles":
                return "Sin Papeles"
            return s
        return v

    @field_validator("nivel_educativo", mode="before")
    @classmethod
    def normalizar_educacion(cls, v: Any) -> Any:
        if isinstance(v, str):
            s = v.strip().capitalize()
            if s in ["Basico", "Básico"]:
                return "Básico"
            if s in ["Tecnico", "Técnico"]:
                return "Técnico"
            return s
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "edad": 29,
                "salud": "Buena",
                "recursos_usd": 220.0,
                "vinculos_familiares": "Sí (Lima)",
                "documentacion": "Pasaporte",
                "genero": "Femenino",
                "nivel_educativo": "Universitario"
            }
        }
    }


# ==============================================================================
# 2. Esquema de Salida de Predicción (PrediccionOutput)
# ==============================================================================

class PrediccionOutput(BaseModel):
    """
    Respuesta analítica con la probabilidad calculada, clasificación de riesgo
    y latencia del modelo en milisegundos.
    """
    probabilidad_calculada: float = Field(
        ...,
        description="Probabilidad estimada de tránsito rápido en escala porcentual (0% a 100%)",
        examples=[83.45]
    )
    categoria_riesgo: str = Field(
        ...,
        description="Categoría de riesgo o vulnerabilidad derivada de la probabilidad",
        examples=["Bajo Riesgo (Tránsito Ágil / Alta Movilidad)"]
    )
    tiempo_respuesta_ms: float = Field(
        ...,
        description="Tiempo de inferencia del pipeline en milisegundos",
        examples=[2.85]
    )
    detalles_entrada: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Resumen de los factores clave del migrante evaluado"
    )


# ==============================================================================
# 3. Esquemas de Simulación de Dinámica de Sistemas (Corredor)
# ==============================================================================

EscenarioSimulacionType = Literal[
    "Línea Base",
    "Crisis +50%",
    "Cierre de Fronteras",
    "Respuesta Humanitaria"
]

class SimulacionRequest(BaseModel):
    """
    Solicitud para proyectar la dinámica del corredor migratorio bajo escenarios de estrés.
    """
    escenario: EscenarioSimulacionType = Field(
        ...,
        description="Escenario de política o flujo migratorio a simular",
        examples=["Crisis +50%"]
    )
    meses: int = Field(
        ...,
        ge=1,
        le=24,
        description="Horizonte temporal de la simulación en meses (1 a 24 meses)",
        examples=[12]
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "escenario": "Crisis +50%",
                "meses": 12
            }
        }
    }


class PuntoSimulacion(BaseModel):
    """
    Registro mensual de dinámica de flujos (Stock y Flujo del corredor).
    """
    mes: int = Field(..., description="Mes ordinal de la serie (1..N)")
    etiqueta_mes: str = Field(..., description="Nombre del mes relativo (e.g. 'Mes 1')")
    flujo_entrante: int = Field(..., description="Cantidad de migrantes que ingresan al corredor")
    retenidos_mes: int = Field(..., description="Migrantes retenidos o en espera en el mes actual")
    integrados_mes: int = Field(..., description="Migrantes que logran tránsito efectivo o integración")
    retenidos_acumulados: int = Field(..., description="Stock acumulado de migrantes en situación de retención")
    integrados_acumulados: int = Field(..., description="Total acumulado de migrantes integrados / en tránsito exitoso")
    saturacion_corredor_pct: float = Field(..., description="Tasa de saturación global del corredor (%)")


class SimulacionResponse(BaseModel):
    """
    Respuesta integral de proyección temporal de dinámica de sistemas.
    """
    escenario: str
    meses_evaluados: int
    tasa_integracion_estimada_pct: float
    tasa_retencion_estimada_pct: float
    total_acumulado_entrantes: int
    total_acumulado_retenidos: int
    total_acumulado_integrados: int
    serie_temporal: List[PuntoSimulacion]


# ==============================================================================
# 4. Esquemas de Telemetría Geoespacial de Nodos (PyDeck)
# ==============================================================================

class NodoTelemetria(BaseModel):
    """
    Estado en tiempo real de cada uno de los 5 nodos estratégicos del corredor migratorio.
    """
    id: str = Field(..., description="Identificador único del nodo")
    nombre: str = Field(..., description="Nombre geográfico de la ciudad/nodo")
    latitud: float = Field(..., description="Coordenada de latitud decimal")
    longitud: float = Field(..., description="Coordenada de longitud decimal")
    capacidad_maxima: int = Field(..., description="Capacidad máxima de acogida y procesamiento")
    migrantes_actuales: int = Field(..., description="Población migrante activa en el nodo")
    nivel_saturacion_pct: float = Field(..., description="Porcentaje de saturación (migrantes / capacidad)")
    estado_operativo: str = Field(..., description="Semáforo operativo ('Normal', 'Alerta', 'Crítico')")
    color_hex: str = Field(..., description="Color hexadecimal para mapas y UI")
    color_rgb: List[int] = Field(..., description="Arreglo RGB [R, G, B, A] optimizado para capas PyDeck")


class TelemetriaCorredorResponse(BaseModel):
    """
    Respuesta consolidada de telemetría de toda la red del corredor.
    """
    timestamp: str
    total_nodos: int
    capacidad_total_red: int
    migrantes_totales_red: int
    saturacion_promedio_red_pct: float
    nodos: List[NodoTelemetria]


# ==============================================================================
# 5. Esquema de Monitoreo de Salud (HealthResponse)
# ==============================================================================

class HealthResponse(BaseModel):
    """
    Estado del microservicio y componentes de Machine Learning.
    """
    status: str
    version: str
    modelo_cargado: bool
    artefactos_disponibles: Dict[str, bool]
    timestamp: str
