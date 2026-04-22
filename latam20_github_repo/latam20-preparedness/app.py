"""
app.py — Entry point · Metanálisis Índices Pandémicos LATAM-20
Usa Dash Pages para multi-página real con callbacks funcionando.
"""
import dash
from dash import dcc, html, Input, Output
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'core'))
sys.path.insert(0, str(ROOT / 'pages'))
os.environ['METANALISIS_DATA'] = str(ROOT / 'data')

# App con pages_folder apuntando a pages/
app = dash.Dash(
    __name__,
    use_pages=True,
    pages_folder=str(ROOT / 'pages'),
    suppress_callback_exceptions=True,
    title='Metanálisis LATAM-20',
    update_title=None,
)
server = app.server

NAV_STYLE = {
    'display':'flex','flexWrap':'wrap','gap':'0',
    'backgroundColor':'#FAFAFA',
    'borderBottom':'1px solid #D1D5DB',
    'padding':'0',
}
LINK_STYLE = {
    'padding':'10px 16px','fontSize':'11px',
    'color':'#6B7280','textDecoration':'none',
    'fontFamily':'Georgia,serif','borderRight':'1px solid #D1D5DB',
    'display':'block',
}

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

    # Navegación por links
    html.Div([
        dcc.Link('★ Validez Predictiva', href='/',
                 style={**LINK_STYLE,'color':'#1E3A5F','fontWeight':'600'}),
        dcc.Link('Síntesis',     href='/synthesis',  style=LINK_STYLE),
        dcc.Link('GHS Index',    href='/ghs',         style=LINK_STYLE),
        dcc.Link('SPAR/IHR',     href='/spar',        style=LINK_STYLE),
        dcc.Link('INFORM Risk',  href='/inform',      style=LINK_STYLE),
        dcc.Link('OxCGRT',       href='/oxcgrt',      style=LINK_STYLE),
        dcc.Link('Severity',     href='/severity',    style=LINK_STYLE),
    ], style=NAV_STYLE),

    # Contenido de la página activa
    dash.page_container,

], style={'backgroundColor':'#F8F9FA','minHeight':'100vh'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
