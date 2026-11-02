# MDT — prototipo de stocks y flujos migratorios

Este repositorio combina un panel Streamlit, una API FastAPI y un motor agregado de simulación mensual. Los cinco nodos del mapa muestran valores configurados y **no** distribuyen dinámicamente las personas entre ciudades. El modelo no implementa agentes individuales ni decisiones de ruta.

## Reproducción del experimento

Instalar las dependencias del proyecto con `pip install -r requirements.txt`. Desde la raíz ejecutar:

```bash
python -m unittest discover -s tests -v
python correr_escenarios_30x.py --salida verificaciones --replicas 30 --meses 12 --semilla-base 1000
python mdt_sensibilidad_local.py --salida verificaciones --replicas 30 --meses 12 --semilla-base 1000
```

Se generan `resultados_reales_mdt.csv` (120 filas), `trayectorias_mdt.csv` (1440 filas), `resumen_estadistico_mdt.csv` y `resumen_eventos_mdt.csv`, `sensibilidad_local_corridas.csv` (960 filas) y `sensibilidad_local_resumen.csv` (32 combinaciones). El nombre histórico `resultados_reales_mdt.csv` se conserva por compatibilidad, pero son **datos simulados**, no observaciones. Las semillas siguen `1000 + 10000 * índice_escenario + réplica`, con escenarios enumerados desde 1 y réplicas desde 1.

Capacidad de red configurada: 19 000. Stock inicial: 1200. Alerta: saturación >=90 %. Sobrecapacidad: >100 %. No se trunca la saturación. Esos umbrales, capacidades y tasas no están calibrados con datos empíricos.

## Aplicación

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
streamlit run app.py
```

## Verificación de entrega

La rama `fix/mdt-paper-reproducibilidad-20261102` corrige el motor principal, el frontend, las pruebas y el script de sensibilidad. Comparar los resultados de cualquier nueva ejecución con los del artículo antes de actualizar gráficos. El dataset de ML auxiliar permanece independiente y no se utiliza en las 120 corridas.
