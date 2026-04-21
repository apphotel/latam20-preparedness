"""
app.py — Entry point principal para Render.com
Metanálisis Índices Pandémicos — LATAM-20
"""
import os, sys
from pathlib import Path

# Rutas portables — funciona en Render, local y cualquier PC
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'core'))    # latam_common, theme, stats_utils
sys.path.insert(0, str(ROOT / 'pages'))   # dashboards

# Fijar DATA_DIR
os.environ.setdefault('METANALISIS_DATA', str(ROOT / 'data'))

# Dashboard principal
from validez_predictiva_dashboard import app

server = app.server  # Requerido por gunicorn

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0',
            port=int(os.environ.get('PORT', 8061)))
