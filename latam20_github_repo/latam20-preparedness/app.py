import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import os
import sys
from pathlib import Path

# --- 1. CONFIGURACIÓN DE RUTAS (REFORZADA) ---
# Resolvemos la ruta absoluta para que Docker no se pierda
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / 'core'))
sys.path.append(str(ROOT / 'pages'))

# --- 2. IMPORTACIÓN DE DASHBOARDS ---
# Usamos un bloque try-except detallado para diagnosticar errores en Render
try:
    import validez_predictiva_dashboard
    import ghs_dashboard
    import spar_dashboard
    import inform_dashboard
    import oxcgrt_dashboard
    import severity_dashboard
    import synthesis_dashboard
except ImportError as e:
    print(f"ERROR CRÍTICO: No se pudo importar un módulo. Detalle: {e}")
    # Esto ayuda a ver en los logs de Render exactamente qué falta
    raise e

# --- 3. INICIALIZAR LA APP ---
app = dash.Dash(
    __name__, 
    suppress_callback_exceptions=True,
    # Puedes descomentar la siguiente línea si usas Dash Bootstrap Components
    # external_stylesheets=['https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css']
)

# ESTA LÍNEA ES VITAL PARA GUNICORN/RENDER
server = app.server 

# --- 4. DISEÑO (LAYOUT) ---
app.layout = html.Div([
    html.H1("Metanálisis Índices Pandémicos — LATAM-20", 
            style={'textAlign': 'center', 'padding': '20px', 'color': '#2c3e50', 'fontFamily': 'sans-serif'}),
    
    dcc.Tabs(id="tabs-menu", value='tab-validez', children=[
        dcc.Tab(label='Validez Predictiva', value='tab-validez'),
        dcc.Tab(label='GHS Index', value='tab-ghs'),
        dcc.Tab(label='SPAR/IHR', value='tab-spar'),
        dcc.Tab(label='INFORM Risk', value='tab-inform'),
        dcc.Tab(label='OxCGRT', value='tab-oxcgrt'),
        dcc.Tab(label='Severity', value='tab-severity'),
        dcc.Tab(label='Síntesis', value='tab-synthesis'),
    ]),
    
    # Contenedor donde se cargará el contenido de cada pestaña
    html.Div(id='tabs-content', style={'padding': '20px'})
])

# --- 5. LÓGICA DE NAVEGACIÓN ---
@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs-menu', 'value')]
)
def render_content(tab):
    try:
        if tab == 'tab-validez':
            return validez_predictiva_dashboard.layout
        elif tab == 'tab-ghs':
            return ghs_dashboard.layout
        elif tab == 'tab-spar':
            return spar_dashboard.layout
        elif tab == 'tab-inform':
            return inform_dashboard.layout
        elif tab == 'tab-oxcgrt':
            return oxcgrt_dashboard.layout
        elif tab == 'tab-severity':
            return severity_dashboard.layout
        elif tab == 'tab-synthesis':
            return synthesis_dashboard.layout
    except AttributeError as e:
        return html.Div([
            html.H3(f"Error en el layout de la pestaña: {tab}"),
            html.P(str(e))
        ], style={'color': 'red'})
    
    return html.Div("Pestaña no encontrada")

# --- 6. EJECUCIÓN ---
if __name__ == '__main__':
    # Render usa el puerto 10000 por defecto para servicios web
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
