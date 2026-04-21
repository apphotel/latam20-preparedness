"""
theme.py — Paleta y estilos unificados para los seis dashboards del metanálisis.

Paleta: CDC / Harvard Data Science (fondo claro, acentos institucionales).
Rationale: optimizado para exportación a PDF y captura para paper.

USO:
    from theme import C, LAYOUT, CARD, TS, TSS, TABLE_STYLE
    from theme import rc, sc, gc, rgba, make_header, make_kpi, make_section_title
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════════════
# PALETA CANÓNICA
# ══════════════════════════════════════════════════════════════════════
C: dict[str, str] = dict(
    # Fondos (escala de grises claros)
    bg='#FAFAFA', bg2='#F0F2F5', bg3='#E8ECF0', bg4='#DDE2E8',

    # Texto
    ink='#1A2332', mid='#2C3E50', text='#374151',
    muted='#6B7280', light='#9CA3AF',

    # Semánticos de riesgo (secuencial cálido → frío)
    very_high='#7F1D1D',  very_high2='#FECACA',
    high='#B91C1C',       high2='#FED7D7',
    medium='#B45309',     medium2='#FEF3C7',
    amber='#B7791F',      amber2='#FEFCBF',
    orange='#C05621',     orange2='#FEEBC8',
    red='#C53030',        red2='#FED7D7',

    # Semánticos positivos
    low='#15803D',        low2='#DCFCE7',
    green='#166534',      green2='#DCFCE7',
    teal='#0E7490',       teal2='#CFFAFE',
    very_low='#1D4ED8',   very_low2='#DBEAFE',
    blue='#1E40AF',       blue2='#DBEAFE',

    # Acentos institucionales
    purple='#6B21A8',     purple2='#F3E8FF',
    covid='#7C3AED',      # pandemia COVID
    latam='#1E3A5F',      # acento LATAM
    global_c='#64748B',   # referencia global

    # UI
    white='#FFFFFF',
    border='#D1D5DB',
    grid='rgba(107,114,128,0.12)',

    # Alias para retrocompatibilidad con dashboards previos
    gold='#B7791F',       # antes '#e94560' oscuro → ámbar claro
    stringency='#1E3A5F',
    gov='#166534',
    containment='#0E7490',
    economic='#B7791F',
)

# Colores por índice del metanálisis (para comparaciones cross-index)
INDEX_COLORS: dict[str, str] = {
    'GHS':             '#1E40AF',  # azul — Global Health Security
    'SPAR':            '#166534',  # verde — autorreporte estatal
    'INFORM Risk':     '#C05621',  # naranja — evaluación externa
    'INFORM Severity': '#7F1D1D',  # rojo oscuro — crisis
    'OxCGRT':          '#6B21A8',  # púrpura — respuesta gubernamental
}

# Colores por dimensión (compartidos donde aplica)
DIM_COLORS: dict[str, str] = {
    'HA':     '#C05621',  # Hazard & Exposure
    'VU':     '#B7791F',  # Vulnerability
    'CC':     '#6B21A8',  # Lack of Coping Capacity
    'INFORM': '#1A2332',
}

CLUSTER_COLORS: dict[str, str] = {
    'Alto riesgo':  C['red'],
    'Riesgo medio': C['amber'],
    'Bajo riesgo':  C['green'],
}

SUBREGION_COLORS: dict[str, str] = {
    'Sudamérica':    '#1E40AF',
    'Centroamérica': '#166534',
    'Norteamérica':  '#6B21A8',
    'Caribe':        '#C05621',
}

# ══════════════════════════════════════════════════════════════════════
# LAYOUT BASE DE PLOTLY
# ══════════════════════════════════════════════════════════════════════
LAYOUT: dict = dict(
    paper_bgcolor=C['white'],
    plot_bgcolor=C['white'],
    font=dict(family='Georgia, serif', color=C['ink'], size=11),
    margin=dict(l=55, r=30, t=45, b=45),
    xaxis=dict(gridcolor=C['grid'], zerolinecolor=C['border']),
    yaxis=dict(gridcolor=C['grid'], zerolinecolor=C['border']),
)

# Escala de color secuencial para riesgo (verde → rojo)
RISK_COLORSCALE = [
    [0.0,  '#15803D'],  # verde
    [0.25, '#B7791F'],  # ámbar
    [0.5,  '#C05621'],  # naranja
    [0.75, '#B91C1C'],  # rojo
    [1.0,  '#7F1D1D'],  # rojo oscuro
]

# Escala divergente (para diferencias vs global)
DIVERGING_COLORSCALE = [
    [0.0, '#1E40AF'],   # azul (LATAM mejor)
    [0.5, '#F0F2F5'],   # neutro
    [1.0, '#7F1D1D'],   # rojo (LATAM peor)
]

# ══════════════════════════════════════════════════════════════════════
# ESTILOS DE UI (Dash)
# ══════════════════════════════════════════════════════════════════════
CARD: dict = {
    'background':   C['white'],
    'border':       f'1px solid {C["border"]}',
    'borderRadius': '8px',
    'padding':      '14px 18px',
    'marginBottom': '12px',
    'boxShadow':    '0 1px 3px rgba(0,0,0,0.04)',
}

CARD_MUTED: dict = {**CARD, 'background': C['bg2']}

# Estilos de Tab (no seleccionada / seleccionada)
TS: dict = {
    'backgroundColor': C['bg2'],
    'color':           C['muted'],
    'border':          f'1px solid {C["border"]}',
    'padding':         '8px 14px',
    'fontSize':        '11px',
    'fontFamily':      'Georgia, serif',
    'letterSpacing':   '0.5px',
    'fontWeight':      '400',
}

TSS: dict = {
    **TS,
    'backgroundColor': C['white'],
    'color':           C['ink'],
    'fontWeight':      '600',
    'borderBottom':    f'2px solid {C["latam"]}',
}

# Estilos canónicos para dash_table.DataTable
TABLE_STYLE: dict = dict(
    style_table={'backgroundColor': C['white'], 'overflowX': 'auto'},
    style_cell={
        'backgroundColor': C['white'],
        'color':           C['ink'],
        'fontSize':        '11px',
        'padding':         '6px 10px',
        'border':          f'1px solid {C["border"]}',
        'fontFamily':      'Georgia, serif',
        'textAlign':       'left',
    },
    style_header={
        'backgroundColor': C['bg3'],
        'color':           C['ink'],
        'fontWeight':      '600',
        'border':          f'1px solid {C["border"]}',
        'fontSize':        '11px',
    },
)

# Estilo dropdown compacto
DROPDOWN_STYLE: dict = {
    'fontSize':        '12px',
    'backgroundColor': C['white'],
    'color':           C['ink'],
    'border':          f'1px solid {C["border"]}',
    'borderRadius':    '4px',
}

# ══════════════════════════════════════════════════════════════════════
# HELPERS DE COLOR
# ══════════════════════════════════════════════════════════════════════

def rc(v: float) -> str:
    """Risk color: color según nivel de riesgo (0–10, mayor = más riesgo)."""
    if v >= 6:  return C['red']
    if v >= 5:  return C['orange']
    if v >= 4:  return C['amber']
    if v >= 3:  return C['medium']
    return C['green']


def sc(slope: float) -> str:
    """Slope color: color según tendencia (pts/año)."""
    if slope >  0.08: return C['red']
    if slope >  0.03: return C['orange']
    if slope < -0.03: return C['green']
    return C['muted']


def gc(delta: float) -> str:
    """Gap color: color según brecha LATAM vs Global (rojo = LATAM peor)."""
    if delta >  0.5:  return C['red']
    if delta >  0.1:  return C['orange']
    if delta < -0.5:  return C['green']
    if delta < -0.1:  return C['teal']
    return C['muted']


def rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convierte '#RRGGBB' → 'rgba(r,g,b,a)'."""
    h = hex_color.lstrip('#')
    return f'rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{alpha})'


# ══════════════════════════════════════════════════════════════════════
# HELPERS DE LAYOUT (componentes Dash reutilizables)
# ══════════════════════════════════════════════════════════════════════

def make_header(title: str, subtitle: str = '') -> dict:
    """Estilo del header principal del dashboard."""
    from dash import html
    return html.Div([
        html.Div(title, style={
            'fontFamily':    'Georgia, serif',
            'fontWeight':    '700',
            'color':         C['ink'],
            'letterSpacing': '4px',
            'fontSize':      '22px',
            'marginBottom':  '2px',
        }),
        html.Div(subtitle, style={
            'color':         C['muted'],
            'fontSize':      '10px',
            'letterSpacing': '1.5px',
        }) if subtitle else None,
    ], style={
        'paddingTop':    '14px',
        'paddingBottom': '10px',
        'borderBottom':  f'1px solid {C["border"]}',
        'marginBottom':  '12px',
    })


def make_kpi(value, label: str, note: str = '', color: str | None = None):
    """KPI card uniforme para toda la serie de dashboards."""
    from dash import html
    import dash_bootstrap_components as dbc
    if color is None:
        color = C['ink']
    return dbc.Col(html.Div([
        html.Div(str(value), style={
            'fontSize':    '22px',
            'fontWeight':  '700',
            'color':       color,
            'fontFamily':  'Georgia, serif',
            'lineHeight':  '1',
        }),
        html.Div(label, style={
            'fontSize':    '10px',
            'color':       C['muted'],
            'marginTop':   '4px',
            'letterSpacing': '0.5px',
            'textTransform': 'uppercase',
        }),
        html.Div(note, style={
            'fontSize':    '9px',
            'color':       C['light'],
            'marginTop':   '2px',
        }) if note else None,
    ], style={**CARD, 'textAlign': 'center', 'padding': '12px 8px'}),
        md=2, sm=4, xs=6)


def make_section_title(title: str, subtitle: str = ''):
    """Título de sección dentro de una card."""
    from dash import html
    return html.Div([
        html.Div(title, style={
            'fontSize':      '10px',
            'color':         C['muted'],
            'letterSpacing': '2px',
            'textTransform': 'uppercase',
            'marginBottom':  '2px',
            'fontWeight':    '600',
        }),
        html.Div(subtitle, style={
            'fontSize':    '11px',
            'color':       C['light'],
            'marginBottom':'10px',
        }) if subtitle else None,
    ])


def make_methodology_note(text: str, accent: str = 'amber'):
    """Cajón discreto para notas metodológicas."""
    from dash import html
    accent_color = C.get(accent, C['amber'])
    return html.Div([
        html.Span('NOTA METODOLÓGICA · ', style={
            'fontWeight': '700',
            'color':      accent_color,
            'fontSize':   '10px',
            'letterSpacing': '1px',
        }),
        html.Span(text, style={
            'color':    C['text'],
            'fontSize': '10px',
        }),
    ], style={
        'background':   rgba(accent_color, 0.06),
        'border':       f'1px solid {rgba(accent_color, 0.25)}',
        'borderRadius': '6px',
        'padding':      '8px 14px',
        'marginBottom': '14px',
    })


# Config estándar para dcc.Graph — incluye exportación SVG vectorial
GRAPH_CONFIG: dict = {
    'displayModeBar': True,
    'displaylogo':    False,
    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
    'toImageButtonOptions': {
        'format':  'svg',
        'filename':'figure',
        'scale':   2,
    },
}

# Config minimal (sin modebar) para widgets secundarios
GRAPH_CONFIG_MIN: dict = {'displayModeBar': False, 'displaylogo': False}


if __name__ == '__main__':
    print('theme.py — paleta CDC/Harvard claro')
    print(f'Colores definidos: {len(C)}')
    print(f'Índices: {list(INDEX_COLORS)}')
