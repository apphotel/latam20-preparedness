import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import os, sys
from pathlib import Path

# Configuración de rutas
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'core'))
sys.path.insert(0, str(ROOT / 'pages'))

# Importar tus 7 dashboards (asegúrate que los nombres coincidan con tus archivos)
import validez_predictiva_dashboard
import ghs_dashboard
import spar_dashboard
import inform_dashboard
import oxcgrt_dashboard
import severity_dashboard
import synthesis_dashboard

# Inicializar la App de Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server # Para Render

# Diseño con menú de pestañas
app.layout = html.Div([
    html.H1("Metanálisis Índices Pandémicos — LATAM-20", style={'textAlign': 'center'}),
    dcc.Tabs(id="tabs-menu", value='tab-validez', children=[
        dcc.Tab(label='Validez Predictiva', value='tab-validez'),
        dcc.Tab(label='GHS Index', value='tab-ghs'),
        dcc.Tab(label='SPAR/IHR', value='tab-spar'),
        dcc.Tab(label='INFORM Risk', value='tab-inform'),
        dcc.Tab(label='OxCGRT', value='tab-oxcgrt'),
        dcc.Tab(label='Severity', value='tab-severity'),
        dcc.Tab(label='Síntesis', value='tab-synthesis'),
    ]),
    html.Div(id='tabs-content')
])

# Lógica para cambiar de dashboard
@app.callback(Output('tabs-content', 'children'),
              [Input('tabs-menu', 'value')])
def render_content(tab):
    if tab == 'tab-validez': return validez_predictiva_dashboard.layout
    elif tab == 'tab-ghs': return ghs_dashboard.layout
    elif tab == 'tab-spar': return spar_dashboard.layout
    elif tab == 'tab-inform': return inform_dashboard.layout
    elif tab == 'tab-oxcgrt': return oxcgrt_dashboard.layout
    elif tab == 'tab-severity': return severity_dashboard.layout
    elif tab == 'tab-synthesis': return synthesis_dashboard.layout

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8061)))
