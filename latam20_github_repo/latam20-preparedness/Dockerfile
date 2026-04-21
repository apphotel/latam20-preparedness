# Usamos una imagen de Python oficial (la versión que ya vimos que te funcionó)
FROM python:3.10-slim

# Evita que Python genere archivos .pyc y permite ver logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalamos dependencias del sistema necesarias para compilar librerías científicas
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiamos los requerimientos primero para aprovechar la caché de Docker
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiamos todo el contenido de tu carpeta latam20-preparedness al contenedor
COPY . .

# AJUSTE MAESTRO:
# 1. --chdir /app: asegura que gunicorn encuentre app.py y las carpetas pages/core
# 2. --bind 0.0.0.0:10000: fija el puerto para evitar el "Port scan timeout"
# 3. --timeout 120: da más tiempo para cargar los datos pesados de los dashboards
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--chdir", "/app", "--timeout", "600", "--workers", "1", "app:server"]
