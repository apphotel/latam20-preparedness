import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import os
import sys
from pathlib import Path

# --- 1. CONFIGURACIÓN DE RUTAS ---
# Esto asegura que Python encuentre tus módulos 'core' y 'pages' en el servidor
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / 'core'))
sys.path.append(str(ROOT / 'pages'))

# --- 2. IMPORTACIÓN DE DASHBOARDS ---
# Asegúrate de que los archivos .py en la carpeta 'pages' tengan una variable llamada 'layout'
try:
    import validez_predictiva_dashboard
    import ghs_dashboard
    import spar_dashboard
    import inform_dashboard
    import oxcgrt_dashboard
    import severity_dashboard
    import synthesis_dashboard
except ImportError as e:
    print(f"Error importando módulos: {e}")

# --- 3. INICIALIZAR LA APP ---
app = dash.Dash(
    __name__, 
    suppress_callback_exceptions=True,
    # Puedes añadir un tema de bootstrap si lo deseas:
    # external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css']
)

# ESTA LÍNEA ES VITAL PARA RENDER
server = app.server 

# --- 4. DISEÑO (LAYOUT) ---
app.layout = html.Div([
    html.H1("Metanálisis Índices Pandémicos — LATAM-20", 
            style={'textAlign': 'center', 'padding': '20px', 'color': '#2c3e50'}),
    
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
    return html.Div("Pestaña no encontrada")

# --- 6. EJECUCIÓN ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
