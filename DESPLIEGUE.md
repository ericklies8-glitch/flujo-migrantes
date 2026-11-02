# Despliegue local del Sistema MDT

El proyecto se ejecuta como dos servicios coordinados: la API FastAPI (puerto
8000) y la interfaz Streamlit (puerto 8501). Docker Compose prepara ambos
servicios y comprueba que respondan antes de marcar el despliegue como saludable.

## Requisitos

- Docker Desktop instalado y en ejecución.
- Acceso a Internet durante la primera construcción para descargar dependencias.

## Iniciar

Desde la carpeta del proyecto:

```powershell
docker compose up --build -d
```

Abre <http://localhost:8501> para usar el sistema y <http://localhost:8001/docs>
para consultar la documentación interactiva de la API.

## Comprobar y detener

```powershell
docker compose ps
docker compose logs -f
docker compose down
```

La interfaz queda en `localhost:8501` y la API en `localhost:8001`; puedes cambiar
el puerto del host con `MDT_API_PORT`. Ambos puertos quedan vinculados a `localhost`;
la configuración no publica el sistema en Internet. Para un despliegue público hace falta definir el proveedor,
los dominios, HTTPS y los controles de acceso. Configura `MDT_CORS_ORIGINS` con
el origen exacto de la interfaz cuando esta tenga un dominio definitivo.

La carpeta `data/` se excluye de la imagen para no publicar el conjunto de datos
histórico. El modelo serializado y los archivos de métricas sí se incluyen.
