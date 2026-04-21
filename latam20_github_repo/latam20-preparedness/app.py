"""
app.py — Entry point principal · Metanálisis Índices Pandémicos LATAM-20
Combina los 7 dashboards en una sola app con navegación por pestañas.
"""
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import os, sys
from pathlib import Path

# ── Rutas portables (funciona en Docker, Render y PC local) ──
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'core'))
sys.path.insert(0, str(ROOT / 'pages'))

# METANALISIS_DATA apunta siempre a data/ junto a este archivo
os.environ['METANALISIS_DATA'] = str(ROOT / 'data')

# ── Importar dashboards ──
# Cada uno se importa por separado para aislar errores
import_errors = {}

try:
    import validez_predictiva_dashboard as vp
except Exception as e:
    import_errors['validez_predictiva'] = str(e)

try:
    import synthesis_dashboard as syn
except Exception as e:
    import_errors['synthesis'] = str(e)

try:
    import ghs_dashboard as ghs
except Exception as e:
    import_errors['ghs'] = str(e)

try:
    import spar_dashboard as spar
except Exception as e:
    import_errors['spar'] = str(e)

try:
    import inform_dashboard as inform
except Exception as e:
    import_errors['inform'] = str(e)

try:
    import oxcgrt_dashboard as oxcgrt
except Exception as e:
    import_errors['oxcgrt'] = str(e)

try:
    import severity_dashboard as severity
except Exception as e:
    import_errors['severity'] = str(e)

if import_errors:
    print(f"\n⚠️  Errores de importación: {import_errors}\n")
else:
    print("✅ Todos los dashboards importados correctamente")

# ── App principal ──
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    title='Metanálisis LATAM-20',
    update_title=None,
)

# REQUERIDO por gunicorn/Render
server = app.server

# ── Layout principal con tabs ──
app.layout = html.Div([
    # Header
    html.Div([
        html.Div("METANÁLISIS ÍNDICES PANDÉMICOS — LATAM-20",
                 style={'fontFamily':'Georgia,serif','fontWeight':'700',
                        'color':'#1E3A5F','fontSize':'18px','letterSpacing':'2px'}),
        html.Div("GHS · SPAR · INFORM · OxCGRT · Validez Predictiva COVID-19",
                 style={'color':'#6B7280','fontSize':'11px','marginTop':'3px'}),
    ], style={'padding':'14px 24px 10px','borderBottom':'1px solid #D1D5DB',
              'backgroundColor':'#FFFFFF'}),

    # Navegación
    dcc.Tabs(id='tabs-main', value='tab-validez', children=[
        dcc.Tab(label='★ Validez Predictiva', value='tab-validez',
                style={'fontSize':'11px','padding':'8px 12px'},
                selected_style={'fontSize':'11px','padding':'8px 12px',
                                'borderBottom':'2px solid #1E3A5F','fontWeight':'600'}),
        dcc.Tab(label='Síntesis', value='tab-synthesis',
                style={'fontSize':'11px','padding':'8px 12px'},
                selected_style={'fontSize':'11px','padding':'8px 12px',
                                'borderBottom':'2px solid #1E3A5F','fontWeight':'600'}),
        dcc.Tab(label='GHS Index', value='tab-ghs',
                style={'fontSize':'11px','padding':'8px 12px'},
                selected_style={'fontSize':'11px','padding':'8px 12px',
                                'borderBottom':'2px solid #1E3A5F','fontWeight':'600'}),
        dcc.Tab(label='SPAR/IHR', value='tab-spar',
                style={'fontSize':'11px','padding':'8px 12px'},
                selected_style={'fontSize':'11px','padding':'8px 12px',
                                'borderBottom':'2px solid #1E3A5F','fontWeight':'600'}),
        dcc.Tab(label='INFORM Risk', value='tab-inform',
                style={'fontSize':'11px','padding':'8px 12px'},
                selected_style={'fontSize':'11px','padding':'8px 12px',
                                'borderBottom':'2px solid #1E3A5F','fontWeight':'600'}),
        dcc.Tab(label='OxCGRT', value='tab-oxcgrt',
                style={'fontSize':'11px','padding':'8px 12px'},
                selected_style={'fontSize':'11px','padding':'8px 12px',
                                'borderBottom':'2px solid #1E3A5F','fontWeight':'600'}),
        dcc.Tab(label='Severity', value='tab-severity',
                style={'fontSize':'11px','padding':'8px 12px'},
                selected_style={'fontSize':'11px','padding':'8px 12px',
                                'borderBottom':'2px solid #1E3A5F','fontWeight':'600'}),
    ], style={'backgroundColor':'#FAFAFA','borderBottom':'1px solid #D1D5DB'}),

    # Contenido
    html.Div(id='tabs-content', style={'padding':'0'}),

], style={'backgroundColor':'#F8F9FA','minHeight':'100vh'})


def _error_div(name, err):
    return html.Div([
        html.H4(f"Error cargando {name}", style={'color':'#C53030'}),
        html.Pre(str(err), style={'background':'#FEF2F2','padding':'12px',
                                   'borderRadius':'6px','fontSize':'11px'}),
    ], style={'padding':'30px'})


@app.callback(Output('tabs-content', 'children'),
              Input('tabs-main', 'value'))
def render_tab(tab):
    if tab == 'tab-validez':
        if 'validez_predictiva' in import_errors:
            return _error_div('Validez Predictiva', import_errors['validez_predictiva'])
        return vp.app.layout

    elif tab == 'tab-synthesis':
        if 'synthesis' in import_errors:
            return _error_div('Síntesis', import_errors['synthesis'])
        return syn.app.layout

    elif tab == 'tab-ghs':
        if 'ghs' in import_errors:
            return _error_div('GHS', import_errors['ghs'])
        return ghs.app.layout

    elif tab == 'tab-spar':
        if 'spar' in import_errors:
            return _error_div('SPAR', import_errors['spar'])
        return spar.app.layout

    elif tab == 'tab-inform':
        if 'inform' in import_errors:
            return _error_div('INFORM', import_errors['inform'])
        return inform.app.layout

    elif tab == 'tab-oxcgrt':
        if 'oxcgrt' in import_errors:
            return _error_div('OxCGRT', import_errors['oxcgrt'])
        return oxcgrt.app.layout

    elif tab == 'tab-severity':
        if 'severity' in import_errors:
            return _error_div('Severity', import_errors['severity'])
        return severity.app.layout

    return html.Div("Pestaña no encontrada")


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
