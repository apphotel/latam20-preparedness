# Imagen base Python 3.10 slim (probada por la usuaria)
FROM python:3.10-slim

# Evitar archivos .pyc y ver logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo
WORKDIR /app

# Dependencias del sistema para librerías científicas
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python primero (aprovecha caché de Docker)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar todo el proyecto
COPY . .

# PYTHONPATH incluye core/ y pages/ para que los imports funcionen
ENV PYTHONPATH="/app:/app/core:/app/pages"

# DATA_DIR apunta siempre a /app/data dentro del contenedor
ENV METANALISIS_DATA="/app/data"

# Puerto que usa Render por defecto
EXPOSE 10000

# Arrancar con gunicorn
CMD ["gunicorn", \
     "--bind", "0.0.0.0:10000", \
     "--timeout", "600", \
     "--workers", "1", \
     "--chdir", "/app", \
     "app:server"]
