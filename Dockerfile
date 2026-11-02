FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY app.py correr_escenarios_30x.py eda_results.json resultados_reales_mdt.csv ./
COPY backend ./backend
COPY cognitive ./cognitive
COPY motor_ml/metricas_crispdm.json motor_ml/eda_results.json motor_ml/modelo_calibrado.joblib ./motor_ml/

EXPOSE 8000 8501

