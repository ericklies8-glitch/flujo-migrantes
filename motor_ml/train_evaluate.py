#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Motor Analítico de Machine Learning - Sistema MDT (Gemelo Digital de Flujo Migratorio)
=====================================================================================
Módulo: motor_ml/train_evaluate.py
Propósito:
    Ejecutar el ciclo completo CRISP-DM de modelado predictivo para estimar la probabilidad
    de tránsito rápido (Prob_Transito_Rapido) de migrantes.

Fases implementadas:
    1. Comprensión y Análisis Exploratorio de Datos (EDA).
    2. Preparación de Datos (Preprocesamiento con ColumnTransformer y Split 80/20).
    3. Benchmarking de Modelos Base (Linear Regression, Random Forest, XGBoost).
    4. Validación Cruzada Estratificada / K-Fold de 5 Folds.
    5. Optimización Exhaustiva de Hiperparámetros (GridSearchCV en 5 Folds).
    6. Pruebas Estadísticas Robustas (Test de Wilcoxon, T-Student Pareada, Bootstrapping 95% CI).
    7. Serialización y Persistencia de Artefactos (modelo calibrado .joblib y métricas .json).
"""

import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

# pyrefly: ignore [missing-import]
import joblib
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# pyrefly: ignore [missing-import]
from xgboost import XGBRegressor

# Configuración de Logging para trazabilidad industrial
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("MotorML_MDT")


def normalizar_nombres_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza variaciones de codificación en encabezados (por ejemplo tildes
    o caracteres especiales) para garantizar interoperabilidad robusta.
    """
    mapeo = {}
    for col in df.columns:
        col_norm = col.strip()
        # Normalizaciones de Género
        if col_norm in ["Gnero", "Género", "Genero"]:
            mapeo[col] = "Género"
        # Normalizaciones de Vínculos Familiares
        elif col_norm in ["Vnculos_Familiares", "Vínculos_Familiares", "Vinculos_Familiares"]:
            mapeo[col] = "Vínculos_Familiares"
        # Normalizaciones de Educación
        elif col_norm in ["Nivel_Educativo", "Nivel_educativo"]:
            mapeo[col] = "Nivel_Educativo"
        # Normalizaciones de Documentación
        elif col_norm in ["Documentacion", "Documentación"]:
            mapeo[col] = "Documentacion"
    if mapeo:
        df = df.rename(columns=mapeo)
    return df


def cargar_datos(ruta_csv: str) -> pd.DataFrame:
    """
    Carga el dataset histórico del sistema MDT verificando existencia, integridad
    y normalizando codificación.
    """
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el dataset en la ruta especificada: {ruta_csv}")

    logger.info(f"Cargando dataset desde: {ruta_csv}")
    try:
        df = pd.read_csv(ruta_csv, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(ruta_csv, encoding="latin1")

    df = normalizar_nombres_columnas(df)
    logger.info(f"Dataset cargado exitosamente. Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas.")
    return df


def ejecutar_eda(
    df: pd.DataFrame,
    ruta_salida_json: str,
    target_col: str = "Prob_Transito_Rapido"
) -> Dict[str, Any]:
    """
    Fase 1: Análisis Exploratorio de Datos (EDA).
    - Extrae resumen estadístico univariado (media, std, cuartiles, min, max).
    - Calcula matrices de correlación lineal (Pearson) y no lineal / monótona (Spearman).
    - Genera conteos de frecuencias para variables demográficas y categóricas.
    - Exporta resultados estructurados en un archivo JSON.
    """
    logger.info(">>> Iniciando Fase 1: Análisis Exploratorio de Datos (EDA)...")

    # Identificación de variables
    cols_continuas = ["Edad", "Recursos_USD", target_col]
    cols_categoricas = ["Salud", "Nivel_Educativo", "Vínculos_Familiares", "Documentacion", "Género"]

    # 1. Resumen Estadístico Univariado
    resumen_desc = df[cols_continuas].describe().to_dict()
    resumen_estadistico = {}
    for col, metricas in resumen_desc.items():
        resumen_estadistico[col] = {
            "media": float(metricas["mean"]),
            "desviacion_estandar": float(metricas["std"]),
            "minimo": float(metricas["min"]),
            "cuartil_25": float(metricas["25%"]),
            "mediana_cuartil_50": float(metricas["50%"]),
            "cuartil_75": float(metricas["75%"]),
            "maximo": float(metricas["max"]),
            "conteo": int(metricas["count"])
        }

    # 2. Correlaciones Pearson y Spearman frente al target
    df_cont = df[cols_continuas].dropna()
    corr_pearson = df_cont.corr(method="pearson").round(4).to_dict()
    corr_spearman = df_cont.corr(method="spearman").round(4).to_dict()

    factores_correlacion_target = {
        "pearson": {col: float(corr_pearson[col][target_col]) for col in cols_continuas if col != target_col},
        "spearman": {col: float(corr_spearman[col][target_col]) for col in cols_continuas if col != target_col}
    }

    # 3. Conteos demográficos y categóricos
    conteos_demograficos = {}
    for col in cols_categoricas:
        if col in df.columns:
            conteos_demograficos[col] = {str(k): int(v) for k, v in df[col].value_counts().to_dict().items()}

    eda_dict = {
        "total_registros": int(len(df)),
        "resumen_estadistico_variables_continuas": resumen_estadistico,
        "matriz_correlacion_pearson": corr_pearson,
        "matriz_correlacion_spearman": corr_spearman,
        "correlacion_con_target": factores_correlacion_target,
        "conteos_demograficos": conteos_demograficos
    }

    # Exportación dual para accesibilidad
    for path in [ruta_salida_json, os.path.join(os.path.dirname(ruta_salida_json), "..", "eda_results.json")]:
        try:
            target_dir = os.path.dirname(path)
            if target_dir:
                os.makedirs(target_dir, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(eda_dict, f, indent=4, ensure_ascii=False)
            logger.info(f"Resultados de EDA exportados exitosamente a: {path}")
        except Exception as e:
            logger.warning(f"No se pudo guardar copia secundaria de eda_results en {path}: {e}")

    return eda_dict


def construir_preprocesador(
    cols_numericas: List[str],
    cols_categoricas: List[str]
) -> ColumnTransformer:
    """
    Fase 2: Definición del ColumnTransformer.
    - StandardScaler para características continuas (Edad, Recursos_USD).
    - OneHotEncoder con ignorado de categorías desconocidas para variables cualitativas.
    """
    logger.info("Definiendo ColumnTransformer (StandardScaler + OneHotEncoder)...")
    preprocesador = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), cols_numericas),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cols_categoricas)
        ],
        remainder="drop"
    )
    return preprocesador


def preparar_particion_datos(
    df: pd.DataFrame,
    target_col: str = "Prob_Transito_Rapido",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], List[str]]:
    """
    Separa variables independientes del target y realiza split 80% train / 20% test con semilla fija.
    """
    cols_numericas = ["Edad", "Recursos_USD"]
    cols_categoricas = ["Salud", "Nivel_Educativo", "Vínculos_Familiares", "Documentacion", "Género"]

    columnas_requeridas = cols_numericas + cols_categoricas
    for col in columnas_requeridas:
        if col not in df.columns:
            raise KeyError(f"Columna requerida ausente en el dataset: '{col}'")

    X = df[columnas_requeridas]
    y = df[target_col].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(f"Partición de datos completada: Train={len(X_train)} filas, Test={len(X_test)} filas.")
    return X_train, X_test, y_train, y_test, cols_numericas, cols_categoricas


def evaluar_benchmarking(
    preprocesador: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Fase 3: Benchmarking de Modelos Base.
    Evalúa 3 candidatos integrando pipeline de preprocesamiento:
        a) LinearRegression
        b) RandomForestRegressor
        c) XGBRegressor
    Calcula métricas base en conjunto de test: RMSE, MAE y R².
    """
    logger.info(">>> Iniciando Fase 3: Benchmarking de Modelos Candidatos en Test Set...")

    modelos_candidatos = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(random_state=random_state, n_estimators=100),
        "XGBRegressor": XGBRegressor(
            random_state=random_state,
            objective="reg:squarederror",
            n_estimators=100,
            learning_rate=0.1
        )
    }

    resultados_benchmarking = {}

    for nombre, modelo in modelos_candidatos.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocesador),
            ("regressor", modelo)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        resultados_benchmarking[nombre] = {
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "R2": round(r2, 4),
            "pipeline": pipeline
        }
        logger.info(f"  [{nombre}] -> RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

    return resultados_benchmarking


def ejecutar_validacion_cruzada(
    preprocesador: ColumnTransformer,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, np.ndarray]]:
    """
    Fase 4: Validación Cruzada en 5 Folds con KFold(shuffle=True, random_state=42).
    Aplica cross_validate sobre los tres modelos candidatos extrayendo media y std de R² y RMSE.
    """
    logger.info(">>> Iniciando Fase 4: Validación Cruzada K-Fold (5 Folds)...")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    modelos_cv = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(random_state=random_state, n_estimators=100),
        "XGBRegressor": XGBRegressor(
            random_state=random_state,
            objective="reg:squarederror",
            n_estimators=100,
            learning_rate=0.1
        )
    }

    scoring = {
        "r2": "r2",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error"
    }

    resultados_cv = {}
    r2_scores_per_fold = {}

    for nombre, modelo in modelos_cv.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocesador),
            ("regressor", modelo)
        ])
        cv_scores = cross_validate(
            pipeline, X, y, cv=kf, scoring=scoring, return_train_score=False, n_jobs=-1
        )

        r2_folds = cv_scores["test_r2"]
        rmse_folds = -cv_scores["test_rmse"]
        mae_folds = -cv_scores["test_mae"]

        r2_scores_per_fold[nombre] = r2_folds

        resultados_cv[nombre] = {
            "r2_mean": float(np.mean(r2_folds)),
            "r2_std": float(np.std(r2_folds)),
            "rmse_mean": float(np.mean(rmse_folds)),
            "rmse_std": float(np.std(rmse_folds)),
            "mae_mean": float(np.mean(mae_folds)),
            "mae_std": float(np.std(mae_folds)),
            "folds_r2": [round(float(s), 4) for s in r2_folds],
            "folds_rmse": [round(float(s), 4) for s in rmse_folds]
        }
        logger.info(
            f"  [{nombre} CV-5] R²: {np.mean(r2_folds):.4f} (±{np.std(r2_folds):.4f}) | "
            f"RMSE: {np.mean(rmse_folds):.4f} (±{np.std(rmse_folds):.4f})"
        )

    return resultados_cv, r2_scores_per_fold


def optimizar_hiperparametros(
    preprocesador: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = 42
) -> Tuple[Pipeline, Dict[str, Any], float]:
    """
    Fase 5: Optimización de Hiperparámetros (5 Folds) para XGBRegressor.
    GridSearchCV sobre:
        n_estimators: [100, 200]
        max_depth: [3, 5, 7]
        learning_rate: [0.01, 0.05, 0.1]
        subsample: [0.8, 1.0]
    """
    logger.info(">>> Iniciando Fase 5: Optimización de Hiperparámetros (GridSearchCV - 5 Folds)...")

    base_xgb = XGBRegressor(random_state=random_state, objective="reg:squarederror")
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocesador),
        ("regressor", base_xgb)
    ])

    param_grid = {
        "regressor__n_estimators": [100, 200],
        "regressor__max_depth": [3, 5, 7],
        "regressor__learning_rate": [0.01, 0.05, 0.1],
        "regressor__subsample": [0.8, 1.0]
    }

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=kf,
        scoring="r2",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    mejores_params_raw = grid_search.best_params_
    # Limpiar prefijo del pipeline para presentación limpia
    mejores_params = {k.replace("regressor__", ""): v for k, v in mejores_params_raw.items()}
    mejor_score_cv = float(grid_search.best_score_)
    mejor_estimador = grid_search.best_estimator_

    logger.info(f"Mejores Hiperparámetros encontrados: {mejores_params}")
    logger.info(f"Mejor Score R² en CV (Train): {mejor_score_cv:.4f}")

    return mejor_estimador, mejores_params, mejor_score_cv


def ejecutar_pruebas_estadisticas(
    scores_xgb: np.ndarray,
    scores_rf: np.ndarray,
    y_test: pd.Series,
    y_pred: np.ndarray,
    n_bootstraps: int = 1000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Fase 6: Pruebas Estadísticas Robustas y Validación Formal.
    1. Test de Wilcoxon con rangos con signo y T-Student pareada sobre los scores de 5 folds.
       Comprueba significancia estadística (p < 0.05) de superioridad XGBoost vs Random Forest.
    2. Intervalo de Confianza al 95% para R² en el conjunto de test vía Bootstrapping (1000 iteraciones).
    """
    logger.info(">>> Iniciando Fase 6: Pruebas Estadísticas Robustas y Bootstrapping...")

    # 1. Comparación Estadística Pareada (XGBoost vs RandomForest)
    diff = scores_xgb - scores_rf
    logger.info(f"Diferencias pareadas XGBoost - Random Forest por fold: {np.round(diff, 5).tolist()}")

    # Prueba T de Student pareada
    t_stat, t_pval = stats.ttest_rel(scores_xgb, scores_rf, alternative="greater")

    # Prueba de rangos con signo de Wilcoxon
    try:
        w_stat, w_pval = stats.wilcoxon(scores_xgb, scores_rf, alternative="greater")
    except Exception as exc:
        logger.warning(f"Wilcoxon fallback debido a: {exc}")
        w_stat, w_pval = float("nan"), float("nan")

    significancia_alcanzada = bool(w_pval < 0.05 or t_pval < 0.05)

    # 2. Bootstrapping para Intervalo de Confianza al 95% de R² en Test
    rng = np.random.default_rng(random_state)
    n_muestras = len(y_test)
    y_test_arr = np.array(y_test)
    r2_bootstraps = []

    for _ in range(n_bootstraps):
        indices = rng.integers(0, n_muestras, size=n_muestras)
        y_true_boot = y_test_arr[indices]
        y_pred_boot = y_pred[indices]
        # Evitar división por cero si varianza en muestra bootstrap es nula
        if np.var(y_true_boot) > 0:
            score_b = r2_score(y_true_boot, y_pred_boot)
            r2_bootstraps.append(score_b)

    ic_inferior = float(np.percentile(r2_bootstraps, 2.5))
    ic_superior = float(np.percentile(r2_bootstraps, 97.5))
    r2_boot_mean = float(np.mean(r2_bootstraps))

    logger.info(f"Test Wilcoxon: W-Stat={w_stat}, p-value={w_pval:.5f}")
    logger.info(f"Test T-Student pareado: t-Stat={t_stat:.4f}, p-value={t_pval:.5f}")
    logger.info(f"Superioridad estadísticamente significativa (p < 0.05): {significancia_alcanzada}")
    logger.info(f"Intervalo de Confianza 95% de R² (Bootstrap): [{ic_inferior:.4f}, {ic_superior:.4f}]")

    pruebas_dict = {
        "comparacion_modelos_xgb_vs_rf": {
            "test_wilcoxon": {
                "estadistico_W": float(w_stat) if not np.isnan(w_stat) else None,
                "p_value": float(w_pval) if not np.isnan(w_pval) else None,
                "hipotesis_alternativa": "XGBoost > RandomForest"
            },
            "test_t_student_pareado": {
                "estadistico_t": float(t_stat),
                "p_value": float(t_pval),
                "hipotesis_alternativa": "XGBoost > RandomForest"
            },
            "significancia_alcanzada_p_menor_005": significancia_alcanzada
        },
        "intervalo_confianza_95_r2_bootstrap": {
            "n_iteraciones": n_bootstraps,
            "media_bootstrap": round(r2_boot_mean, 4),
            "limite_inferior_2_5": round(ic_inferior, 4),
            "limite_superior_97_5": round(ic_superior, 4)
        }
    }

    return pruebas_dict


def serializar_artefactos(
    modelo_final: Pipeline,
    metricas_completas: Dict[str, Any],
    ruta_modelo: str,
    ruta_metricas: str
) -> None:
    """
    Fase 7: Serialización y Persistencia de Artefactos de Producción.
    Guarda el pipeline calibrado en .joblib y las métricas estructuradas en .json.
    """
    logger.info(">>> Iniciando Fase 7: Serialización de Artefactos...")

    os.makedirs(os.path.dirname(ruta_modelo), exist_ok=True)
    os.makedirs(os.path.dirname(ruta_metricas), exist_ok=True)

    # 1. Serialización del modelo entrenado y calibrado
    joblib.dump(modelo_final, ruta_modelo)
    logger.info(f"Modelo calibrado guardado exitosamente en: {ruta_modelo}")

    # 2. Serialización de métricas completas CRISP-DM
    with open(ruta_metricas, "w", encoding="utf-8") as f:
        json.dump(metricas_completas, f, indent=4, ensure_ascii=False)
    logger.info(f"Métricas CRISP-DM guardadas exitosamente en: {ruta_metricas}")


def main() -> None:
    """
    Orquestador principal del pipeline analítico MDT.
    """
    logger.info("================================================================================")
    logger.info(" SISTEMA MDT - MOTOR ANALÍTICO ML: ENTRENAMIENTO, EVALUACIÓN Y VALIDACIÓN")
    logger.info("================================================================================")

    # Definición de rutas canónicas
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ruta_dataset = os.path.join(base_dir, "data", "dataset_historico_mdt.csv")
    ruta_eda = os.path.join(base_dir, "motor_ml", "eda_results.json")
    ruta_modelo = os.path.join(base_dir, "motor_ml", "modelo_calibrado.joblib")
    ruta_metricas = os.path.join(base_dir, "motor_ml", "metricas_crispdm.json")

    # 1. Ingesta de Datos
    df = cargar_datos(ruta_dataset)

    # 2. Análisis Exploratorio de Datos (EDA)
    eda_results = ejecutar_eda(df, ruta_salida_json=ruta_eda)

    # 3. Preprocesamiento y Partición 80/20
    X_train, X_test, y_train, y_test, cols_num, cols_cat = preparar_particion_datos(
        df, target_col="Prob_Transito_Rapido", test_size=0.2, random_state=42
    )
    preprocesador = construir_preprocesador(cols_num, cols_cat)

    # 4. Benchmarking de Modelos Candidatos (Test Set Inicial)
    benchmarking_results = evaluar_benchmarking(
        preprocesador, X_train, y_train, X_test, y_test, random_state=42
    )

    # Formatear benchmarking para reporte JSON (sin los objetos pipeline)
    benchmarking_metricas = {
        m: {"RMSE": v["RMSE"], "MAE": v["MAE"], "R2": v["R2"]}
        for m, v in benchmarking_results.items()
    }

    # 5. Validación Cruzada K-Fold (5 Folds)
    # Concatenamos X e y para validación cruzada formal global
    X_completo = pd.concat([X_train, X_test], axis=0)
    y_completo = pd.concat([y_train, y_test], axis=0)
    cv_results, r2_scores_folds = ejecutar_validacion_cruzada(
        preprocesador, X_completo, y_completo, n_splits=5, random_state=42
    )

    # 6. Optimización de Hiperparámetros para el modelo ganador (XGBoost)
    mejor_modelo_pipeline, mejores_params, mejor_score_cv = optimizar_hiperparametros(
        preprocesador, X_train, y_train, n_splits=5, random_state=42
    )

    # Evaluación final del modelo optimizado sobre el conjunto de prueba
    y_pred_opt = mejor_modelo_pipeline.predict(X_test)
    mse_opt = mean_squared_error(y_test, y_pred_opt)
    rmse_opt = float(np.sqrt(mse_opt))
    mae_opt = float(mean_absolute_error(y_test, y_pred_opt))
    r2_opt = float(r2_score(y_test, y_pred_opt))

    metricas_modelo_final_test = {
        "RMSE": round(rmse_opt, 4),
        "MAE": round(mae_opt, 4),
        "R2": round(r2_opt, 4)
    }
    logger.info(
        f"Métricas del Modelo Calibrado Final en Test: RMSE={rmse_opt:.4f} | "
        f"MAE={mae_opt:.4f} | R²={r2_opt:.4f}"
    )

    # 7. Pruebas Estadísticas Robustas (Wilcoxon / T-Student & Bootstrap 95% CI)
    pruebas_estadisticas = ejecutar_pruebas_estadisticas(
        scores_xgb=r2_scores_folds["XGBRegressor"],
        scores_rf=r2_scores_folds["RandomForestRegressor"],
        y_test=y_test,
        y_pred=y_pred_opt,
        n_bootstraps=1000,
        random_state=42
    )

    # Consolidación de informe CRISP-DM
    metricas_crispdm = {
        "proyecto": "Sistema MDT - Gemelo Digital de Flujo Migratorio",
        "fase": "Modelado y Evaluación (CRISP-DM)",
        "modelo_seleccionado": "XGBRegressor",
        "mejores_hiperparametros": mejores_params,
        "evaluacion_benchmarking_test": benchmarking_metricas,
        "validacion_cruzada_5_folds": cv_results,
        "evaluacion_modelo_calibrado_test": metricas_modelo_final_test,
        "pruebas_estadisticas_robustas": pruebas_estadisticas
    }

    # 8. Serialización de Artefactos
    serializar_artefactos(
        modelo_final=mejor_modelo_pipeline,
        metricas_completas=metricas_crispdm,
        ruta_modelo=ruta_modelo,
        ruta_metricas=ruta_metricas
    )

    logger.info("================================================================================")
    logger.info(" EJECUCIÓN DEL PIPELINE ANALÍTICO COMPLETADA CON ÉXITO")
    logger.info(f" Artefactos generados:")
    logger.info(f"   - Modelo Calibrado: {ruta_modelo}")
    logger.info(f"   - Métricas CRISP-DM: {ruta_metricas}")
    logger.info(f"   - Resultados EDA:    {ruta_eda}")
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
