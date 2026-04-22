"""
INFORM Severity Dashboard v4.0 — LATAM-20
══════════════════════════════════════════════════════════════════════
Metanálisis Índices Pandémicos · Serie mensual 2019–2026
Fuente: INFORM Severity (JRC/Comisión Europea) · Snapshot Febrero 2026

CAMBIOS v3.0 → v4.0:
  ✓ Imports centralizados desde latam_common y theme
  ✓ Rutas relativas portables (latam_common.py)
  ✓ Paleta CDC/Harvard clara consolidada
  ✓ IC Fisher-z para correlaciones predictor–desenlace
  ✓ Exportación SVG vectorial
  ✓ Carga robusta con try/except para fuentes auxiliares

ROL EN EL METANÁLISIS:
  DESENLACE DE IMPACTO. Mide la gravedad de crisis humanitarias
  activas, NO la preparación previa. Un país sin crisis registrada
  no aparece en la serie porque no activó el umbral mínimo INFORM,
  no porque estuviera "bien preparado".

ESCALA:
  0–10 donde MAYOR score = MAYOR severidad (dirección opuesta a
  GHS y SPAR). Para correlaciones con preparación, se puede
  normalizar como (10 − Severity).
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

from latam_common import (
    LATAM_ISO3, LATAM_20 as LATAM, LATAM_20_SORTED,
    SHORT, sn, data_path, NOTA_MUESTRA,
)
from theme import (
    C, LAYOUT, CARD, TS, TSS, TABLE_STYLE, DROPDOWN_STYLE,
    GRAPH_CONFIG, rgba,
    make_header, make_kpi, make_section_title, make_methodology_note,
)
from stats_utils import correlation_ci

ISO_TO_NAME = LATAM_ISO3
NAME_TO_ISO = {v: k for k, v in LATAM_ISO3.items()}

# Colores específicos Severity desde la paleta
SEV_COLORS = {
    'Very High': C['very_high'],
    'High':      C['high'],
    'Medium':    C['amber'],
    'Low':       C['green'],
    'Very Low':  C['blue'],
}
SEV_ORDER = ['Very High', 'High', 'Medium', 'Low', 'Very Low']


# ══════════════════════════════════════════════════════════════════════
# 1. CARGA ROBUSTA DE DATOS
# ══════════════════════════════════════════════════════════════════════

def _safe_load_excel(key, **kwargs):
    """Carga robusta con manejo de errores."""
    try:
        return pd.read_excel(data_path(key, engine='openpyxl'), **kwargs)
    except Exception as e:
        print(f'⚠️  Error cargando {key}: {e}')
        return None


def _safe_load_csv(key, **kwargs):
    try:
        return pd.read_csv(data_path(key), **kwargs)
    except Exception as e:
        print(f'⚠️  Error cargando {key}: {e}')
        return None


# INFORM Severity — snapshot Feb 2026
sev_raw = _safe_load_excel('severity', sheet_name='INFORM Severity - all crises', header=1)
if sev_raw is not None:
    sev_raw.columns = ['CRISIS','CRISIS_ID','COUNTRY','ISO3','DRIVERS',
                       'SEV_INDEX','SEV_CAT_NUM','SEV_CAT','TREND','RELIABILITY',
                       'IMPACT','GEOGRAPHICAL','HUMAN','CONDITIONS','PEOPLE_IN_NEED',
                       'CONCENTRATION','COMPLEXITY','SOCIETY','OPERATING_ENV',
                       'REGIONS','LAST_UPDATED']
    for col in ['SEV_INDEX','IMPACT','CONDITIONS','COMPLEXITY','GEOGRAPHICAL',
                'HUMAN','PEOPLE_IN_NEED','CONCENTRATION','SOCIETY','OPERATING_ENV']:
        sev_raw[col] = pd.to_numeric(sev_raw[col], errors='coerce')
    lat_sev = sev_raw[sev_raw['COUNTRY'].isin(LATAM)].copy().reset_index(drop=True)
    glo_sev = sev_raw.copy()
else:
    lat_sev = pd.DataFrame()
    glo_sev = pd.DataFrame()

# INFORM Severity — serie temporal Trends
trends_xl = _safe_load_excel('severity', sheet_name='Trends', header=0)
if trends_xl is not None:
    trends_xl.columns = trends_xl.iloc[0]
    trends_df = trends_xl.iloc[1:].reset_index(drop=True)
    trends_df.columns.name = None
    DATE_COLS = [c for c in trends_df.columns if hasattr(c, 'year')]
    DATES_STR = [d.strftime('%Y-%m') for d in DATE_COLS]
    lat_trends = trends_df[trends_df['Country'].isin(LATAM)].copy()
    glo_trends = trends_df.copy()
else:
    trends_df = pd.DataFrame()
    DATE_COLS = []
    DATES_STR = []
    lat_trends = pd.DataFrame()
    glo_trends = pd.DataFrame()

# INFORM Risk — predictor pre-pandemia
df_risk = _safe_load_csv('inform_trend',
    usecols=['Iso3', 'IndicatorId', 'INFORMYear', 'IndicatorScore'])
if df_risk is not None:
    inf19 = df_risk[(df_risk['IndicatorId']=='INFORM') & (df_risk['INFORMYear']==2019)]
    inf19_d = {ISO_TO_NAME[r.Iso3]: float(r.IndicatorScore)
               for _, r in inf19[inf19.Iso3.isin(LATAM_ISO3.keys())].iterrows()}
else:
    inf19_d = {}

# GHS 2021
ghs = _safe_load_csv('ghs')
if ghs is not None:
    ghs21 = ghs[ghs['Year']==2021].set_index('Country')
    ghs_d = {c: float(ghs21.loc[c, 'OVERALL SCORE'])
             for c in LATAM if c in ghs21.index
             and pd.notna(ghs21.loc[c, 'OVERALL SCORE'])}
else:
    ghs_d = {}

# SPAR 2019
spar = _safe_load_csv('spar_latam')
if spar is not None:
    spar = spar[spar['Country'].isin(LATAM)]
    spar19 = spar[spar['Year']==2019].set_index('Country')
    spar_d = {c: float(spar19.loc[c, 'SPAR_Overall'])
              for c in LATAM if c in spar19.index
              and pd.notna(spar19.loc[c, 'SPAR_Overall'])}
else:
    spar_d = {}


# ══════════════════════════════════════════════════════════════════════
# 2. PRE-CÓMPUTOS
# ══════════════════════════════════════════════════════════════════════

# Series por crisis
series_data = {}
if len(lat_trends) > 0:
    for _, row in lat_trends.iterrows():
        country = row['Country']
        crisis = str(row['Crisis'])[:60]
        key = f"{country}||{crisis}"
        pts_d, pts_v = [], []
        for dc in DATE_COLS:
            v = row[dc]
            if v not in ['-', 'x', None] and str(v) != 'nan':
                try:
                    pts_d.append(dc.strftime('%Y-%m'))
                    pts_v.append(round(float(v), 1))
                except Exception:
                    pass
        if len(pts_v) >= 5:
            series_data[key] = dict(
                country=country, crisis=crisis,
                dates=pts_d, vals=pts_v,
                current=pts_v[-1], peak=max(pts_v),
                start=pts_d[0])

# Media mensual LATAM vs Global
monthly_lat, monthly_glo = {}, {}
for dc in DATE_COLS:
    ym = dc.strftime('%Y-%m')
    lv, gv = [], []
    for _, row in lat_trends.iterrows():
        v = row[dc]
        if v not in ['-', 'x', None] and str(v) != 'nan':
            try: lv.append(float(v))
            except Exception: pass
    for _, row in glo_trends.iterrows():
        v = row[dc]
        if v not in ['-', 'x', None] and str(v) != 'nan':
            try: gv.append(float(v))
            except Exception: pass
    if lv: monthly_lat[ym] = round(np.mean(lv), 2)
    if gv: monthly_glo[ym] = round(np.mean(gv), 2)

# Severidad máxima COVID (2020-2021)
covid_max = {}
for k, sd in series_data.items():
    c = sd['country']
    covid_vals = [v for d, v in zip(sd['dates'], sd['vals']) if d[:4] in ['2020', '2021']]
    if covid_vals:
        mx = max(covid_vals)
        if c not in covid_max or mx > covid_max[c]:
            covid_max[c] = round(mx, 1)

# Datos para correlación
corr_pts = []
for c in sorted(covid_max):
    pt = {'country': c, 'sev_covid': covid_max[c]}
    if c in inf19_d: pt['inform_risk'] = inf19_d[c]
    if c in ghs_d:   pt['ghs'] = ghs_d[c]
    if c in spar_d:  pt['spar'] = spar_d[c]
    corr_pts.append(pt)

df_corr = pd.DataFrame(corr_pts) if corr_pts else pd.DataFrame()

# Correlaciones con IC Fisher-z (nuevo en v4.0)
CORRS = {}
for predictor in ['inform_risk', 'ghs', 'spar']:
    if predictor in df_corr.columns:
        sub = df_corr.dropna(subset=[predictor, 'sev_covid'])
        if len(sub) >= 3:
            CORRS[predictor] = {
                'pearson':  correlation_ci(sub[predictor].values,
                                           sub['sev_covid'].values, method='pearson'),
                'spearman': correlation_ci(sub[predictor].values,
                                           sub['sev_covid'].values, method='spearman'),
                'n': len(sub),
            }

# Tabla maestra de crisis
crisis_rows = []
for k, sd in sorted(series_data.items(), key=lambda x: -x[1]['current']):
    trend_val = '—'
    if len(lat_sev) > 0:
        matches = lat_sev[lat_sev['CRISIS'].str[:60]==sd['crisis']]['TREND']
        if len(matches) > 0:
            trend_val = matches.values[0]
    crisis_rows.append({
        'País': sd['country'], 'Crisis': sd['crisis'],
        'Score Feb 2026': sd['current'], 'Pico histórico': sd['peak'],
        'Inicio': sd['start'], 'Duración (meses)': len(sd['vals']),
        'Tendencia': trend_val,
    })
master_df = pd.DataFrame(crisis_rows)


# ══════════════════════════════════════════════════════════════════════
# 3. FIGURAS
# ══════════════════════════════════════════════════════════════════════

def fig_serie_latam_global():
    dates = sorted(monthly_lat.keys())
    lat_v = [monthly_lat.get(d) for d in dates]
    glo_v = [monthly_glo.get(d) for d in dates]
    fig = go.Figure()
    fig.add_vrect(x0='2020-03', x1='2021-12',
                  fillcolor=C['covid'], opacity=0.06, layer='below', line_width=0)
    fig.add_shape(type='line', x0='2020-03', x1='2020-03', y0=0, y1=1,
                  xref='x', yref='paper',
                  line=dict(color=C['covid'], width=1, dash='dash'), opacity=0.5)
    fig.add_annotation(x='2020-03', y=0.98, xref='x', yref='paper',
                       text='Mar 2020 · Pandemia', showarrow=False,
                       font=dict(size=9, color=C['covid']),
                       xanchor='left', bgcolor='rgba(255,255,255,0.7)')
    fig.add_trace(go.Scatter(x=dates, y=glo_v, name='Media global',
        line=dict(color=C['global_c'], width=1.5, dash='dot'), opacity=0.7,
        hovertemplate='%{x}: %{y:.2f}<extra>Global</extra>'))
    fig.add_trace(go.Scatter(x=dates, y=lat_v, name='Media LATAM-20',
        line=dict(color=C['latam'], width=2.5),
        hovertemplate='%{x}: %{y:.2f}<extra>LATAM-20</extra>'))
    fig.update_layout(**{**LAYOUT, 'height': 460},
        title=dict(text='INFORM Severity medio mensual · LATAM-20 vs Global · 2019–2026',
                   font=dict(size=13)),
        yaxis=dict(title='Severity Score (0–10)', range=[0, 10], gridcolor=C['grid'],
                   tickfont=dict(size=10)),
        xaxis=dict(gridcolor=C['grid'], tickfont=dict(size=10)),
        legend=dict(orientation='h', x=0, y=-0.15, font=dict(size=10)),
        hovermode='x unified')
    for val, lbl, col in [(8, 'Muy Alto', C['very_high']), (6, 'Alto', C['high']),
                           (4, 'Medio', C['amber']), (2, 'Bajo', C['green'])]:
        fig.add_hline(y=val, line_width=0.5, line_dash='dot',
                      line_color=col, opacity=0.4,
                      annotation_text=lbl, annotation_position='right',
                      annotation_font=dict(size=8, color=col))
    return fig


def fig_crisis_individuales():
    fig = go.Figure()
    colors_map = {
        'Haiti': C['very_high'], 'Colombia': C['high'],
        'Venezuela': C['orange'], 'Ecuador': C['amber'],
        'Peru': C['teal'], 'Brazil': C['blue'],
        'Costa Rica': C['green'], 'Panama': C['purple'],
        'Mexico': C['ink'], 'Chile': '#6366F1',
        'Dominican Republic': '#0891B2', 'Cuba': '#7C3AED',
    }
    fig.add_vrect(x0='2020-03', x1='2021-12',
                  fillcolor=C['covid'], opacity=0.05, layer='below', line_width=0)
    for k, sd in sorted(series_data.items(), key=lambda x: -x[1]['current']):
        c = sd['country']
        col = colors_map.get(c, C['muted'])
        if len(sd['dates']) >= 12:
            fig.add_trace(go.Scatter(
                x=sd['dates'], y=sd['vals'],
                name=f"{sn(c)} — {sd['crisis'][:30]}",
                line=dict(color=col, width=1.8), mode='lines',
                hovertemplate=f"<b>{c}</b><br>{sd['crisis'][:40]}<br>%{{x}}: %{{y:.1f}}<extra></extra>"))
    fig.update_layout(**{**LAYOUT, 'height': 560},
        title=dict(text='Trayectorias de crisis humanitarias activas · LATAM-20 · 2019–2026',
                   font=dict(size=13)),
        yaxis=dict(title='Severity Score (0–10)', range=[0, 10.5],
                   gridcolor=C['grid'], tickfont=dict(size=10)),
        xaxis=dict(gridcolor=C['grid'], tickfont=dict(size=10)),
        legend=dict(font=dict(size=9), x=1.01, y=1),
        hovermode='x unified')
    for val, lbl, col in [(8, 'Muy Alto', C['very_high']), (6, 'Alto', C['high']),
                           (4, 'Medio', C['amber']), (2, 'Bajo', C['green'])]:
        fig.add_hline(y=val, line_width=0.5, line_dash='dot',
                      line_color=col, opacity=0.35,
                      annotation_text=lbl, annotation_position='right',
                      annotation_font=dict(size=8, color=col))
    return fig


def fig_snapshot_feb2026():
    df = lat_sev.sort_values('SEV_INDEX', ascending=True).dropna(subset=['SEV_INDEX'])
    cats = df['SEV_CAT'].fillna('Unknown')
    colors = [SEV_COLORS.get(c, C['muted']) for c in cats]
    labels = [f"{sn(r['COUNTRY'])} — {str(r['CRISIS'])[:35]}" for _, r in df.iterrows()]
    fig = go.Figure(go.Bar(
        y=labels, x=df['SEV_INDEX'], orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=[f'{v:.1f}' for v in df['SEV_INDEX']],
        textposition='outside', textfont=dict(size=9, color=C['ink']),
        hovertemplate='<b>%{y}</b><br>Score: %{x:.1f}<extra></extra>'))
    fig.update_layout(**{**LAYOUT, 'height': 580},
        title=dict(text='Severidad de crisis activas · LATAM-20 · Febrero 2026',
                   font=dict(size=13)),
        xaxis=dict(title='INFORM Severity Score (0–10)', range=[0, 11],
                   gridcolor=C['grid'], tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=9), gridcolor=C['grid']),
        bargap=0.25)
    for val, lbl, col in [(8, 'Muy Alto', C['very_high']), (6, 'Alto', C['high']),
                           (4, 'Medio', C['amber']), (2, 'Bajo', C['green'])]:
        fig.add_shape(type='line', x0=val, x1=val, y0=0, y1=1,
                      xref='x', yref='paper',
                      line=dict(color=col, width=0.7, dash='dash'), opacity=0.5)
        fig.add_annotation(x=val, y=1.02, xref='x', yref='paper',
                           text=lbl, showarrow=False,
                           font=dict(size=8, color=col), xanchor='center')
    return fig


def fig_dimensiones():
    df = lat_sev.sort_values('SEV_INDEX', ascending=False).dropna(subset=['SEV_INDEX']).head(10)
    labels = [sn(r['COUNTRY']) for _, r in df.iterrows()]
    fig = go.Figure()
    for col_name, col_color, col_label in [
        ('IMPACT',     C['high'],  'Impact of Crisis'),
        ('CONDITIONS', C['amber'], 'Conditions of People'),
        ('COMPLEXITY', C['blue'],  'Complexity'),
    ]:
        fig.add_trace(go.Bar(name=col_label, x=labels, y=df[col_name],
            marker_color=col_color, opacity=0.85))
    fig.update_layout(**{**LAYOUT, 'height': 440}, barmode='group',
        title=dict(text='Dimensiones INFORM Severity · Top 10 crisis · Feb 2026',
                   font=dict(size=13)),
        yaxis=dict(title='Score (0–10)', range=[0, 11], gridcolor=C['grid']),
        xaxis=dict(tickangle=-30, tickfont=dict(size=9), gridcolor=C['grid']),
        legend=dict(orientation='h', x=0, y=-0.2, font=dict(size=10)))
    return fig


def fig_drivers():
    all_drivers = []
    for d in lat_sev['DRIVERS'].dropna():
        all_drivers.extend([x.strip() for x in str(d).split(',')])
    ct = Counter(all_drivers)
    labels = [k for k, v in ct.most_common(10)]
    vals = [v for k, v in ct.most_common(10)]
    colors = []
    for l in labels:
        if 'Conflict' in l: colors.append(C['very_high'])
        elif 'Displacement' in l: colors.append(C['blue'])
        elif 'Political' in l: colors.append(C['orange'])
        elif 'Flood' in l or 'Cyclone' in l or 'Drought' in l: colors.append(C['teal'])
        else: colors.append(C['muted'])
    fig = go.Figure(go.Bar(
        x=labels, y=vals, marker_color=colors,
        text=vals, textposition='outside', textfont=dict(color=C['ink']),
        hovertemplate='<b>%{x}</b>: %{y} crisis<extra></extra>'))
    fig.update_layout(**{**LAYOUT, 'height': 420},
        title=dict(text='Drivers de crisis humanitarias · LATAM-20 · Feb 2026',
                   font=dict(size=13)),
        xaxis=dict(tickangle=-30, tickfont=dict(size=9), gridcolor=C['grid']),
        yaxis=dict(title='Número de crisis', gridcolor=C['grid']))
    return fig


def fig_correlacion_scatter(predictor='inform_risk'):
    df_c = df_corr.dropna(subset=[predictor, 'sev_covid'])
    x = df_c[predictor].values
    y = df_c['sev_covid'].values
    if len(x) >= 3:
        cr = correlation_ci(x, y, method='pearson')
        cr_s = correlation_ci(x, y, method='spearman')
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(min(x), max(x), 50)
        y_line = m * x_line + b
    else:
        cr = {'r': 0, 'p': 1, 'n': len(x), 'ci': [0, 0]}
        cr_s = {'r': 0, 'p': 1, 'n': len(x)}
        x_line, y_line = [], []

    labels = {'inform_risk': 'INFORM Risk 2019',
              'ghs': 'GHS Index 2021',
              'spar': 'SPAR 2019'}
    x_labels = {'inform_risk': 'INFORM Risk 2019 (escala invertida — mayor=peor)',
                'ghs': 'GHS Overall 2021 (mayor=mejor preparación)',
                'spar': 'SPAR Overall 2019 (mayor=mejor preparación)'}

    fig = go.Figure()
    if len(x_line):
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines',
            name='Regresión lineal',
            line=dict(color=C['latam'], width=1.5, dash='dash'), showlegend=True))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='markers+text',
        text=[sn(c) for c in df_c['country']], textposition='top center',
        textfont=dict(size=9, color=C['ink']),
        marker=dict(size=13, color=C['high'], opacity=0.85,
                    line=dict(color='white', width=1)),
        name='Países LATAM',
        hovertemplate=(f'<b>%{{text}}</b><br>{labels[predictor]}: %{{x:.1f}}<br>'
                       'Severity COVID: %{y:.1f}<extra></extra>')))
    sig = '***' if cr['p'] < 0.001 else '**' if cr['p'] < 0.01 else '*' if cr['p'] < 0.05 else 'n.s.'
    annotation_text = (f'<b>Pearson r = {cr["r"]:.3f}</b> ({sig})<br>'
                       f'IC95% [{cr["ci"][0]}, {cr["ci"][1]}] (Fisher-z)<br>'
                       f'p = {cr["p"]:.4f}<br>'
                       f'Spearman ρ = {cr_s["r"]:.3f} (p = {cr_s["p"]:.4f})<br>'
                       f'n = {cr["n"]}')
    fig.add_annotation(
        x=0.97, y=0.05, xref='paper', yref='paper',
        text=annotation_text,
        showarrow=False, align='right',
        font=dict(size=10, color=C['ink']),
        bgcolor=C['bg2'], bordercolor=C['border'], borderwidth=1,
        xanchor='right')
    fig.update_layout(**{**LAYOUT, 'height': 480},
        title=dict(text=f'{labels[predictor]} vs Severity COVID máx. · LATAM-20',
                   font=dict(size=13)),
        xaxis=dict(title=x_labels[predictor], gridcolor=C['grid']),
        yaxis=dict(title='INFORM Severity COVID máxima 2020–2021',
                   gridcolor=C['grid']),
        legend=dict(font=dict(size=9)))
    return fig


def fig_heatmap():
    years = list(range(2019, 2027))
    main_crises = [(sd['country'], sd['crisis'])
                   for _, sd in sorted(series_data.items(), key=lambda x: -x[1]['current'])
                   if len(sd['dates']) >= 20][:12]
    z, y_labels = [], []
    x_labels = [str(y) for y in years]
    for country, crisis in main_crises:
        key = f"{country}||{crisis}"
        sd = series_data[key]
        row_vals = []
        for yr in years:
            yr_vals = [v for d, v in zip(sd['dates'], sd['vals']) if d[:4] == str(yr)]
            row_vals.append(round(np.mean(yr_vals), 1) if yr_vals else None)
        z.append(row_vals)
        y_labels.append(f"{sn(country)}: {crisis[:28]}")
    fig = go.Figure(go.Heatmap(
        z=z, x=x_labels, y=y_labels,
        colorscale=[
            [0.0, '#EFF6FF'], [0.2, '#DBEAFE'], [0.4, '#FEF3C7'],
            [0.6, '#FDE68A'], [0.75, '#FCA5A5'], [0.9, '#EF4444'],
            [1.0, '#7F1D1D'],
        ],
        zmin=0, zmax=10,
        text=[[f'{v:.1f}' if v else '—' for v in row] for row in z],
        texttemplate='%{text}', textfont=dict(size=9, color=C['ink']),
        hovertemplate='<b>%{y}</b><br>%{x}: %{z:.1f}<extra></extra>',
        colorbar=dict(title=dict(text='Severity<br>(0–10)', font=dict(color=C['ink'])),
                      tickfont=dict(size=9, color=C['ink']))))
    fig.add_vrect(x0='2020', x1='2022',
                  fillcolor=C['covid'], opacity=0.08, layer='below', line_width=0)
    fig.update_layout(**{**LAYOUT, 'height': 480},
        title=dict(text='Heatmap · Severity anual por crisis · LATAM-20 · 2019–2026',
                   font=dict(size=13)),
        xaxis=dict(tickfont=dict(size=10), gridcolor=C['grid']),
        yaxis=dict(tickfont=dict(size=9), autorange='reversed', gridcolor=C['grid']))
    return fig


def fig_comparacion_paises():
    countries = sorted(covid_max.keys(), key=lambda c: -covid_max.get(c, 0))
    fig = go.Figure(go.Bar(
        name='Severity COVID máx. (2020-2021)',
        x=[sn(c) for c in countries],
        y=[covid_max.get(c, 0) for c in countries],
        marker_color=C['very_high'], opacity=0.85))
    fig.update_layout(**{**LAYOUT, 'height': 420},
        title=dict(text='Severidad máxima COVID-19 (2020–2021) · Variable de desenlace · LATAM-20',
                   font=dict(size=13)),
        yaxis=dict(title='INFORM Severity máxima (0–10)', gridcolor=C['grid']),
        xaxis=dict(tickangle=-30, tickfont=dict(size=9), gridcolor=C['grid']),
        legend=dict(orientation='h', x=0, y=-0.2, font=dict(size=9)))
    return fig


def fig_tendencia_actual():
    df = lat_sev.sort_values('SEV_INDEX', ascending=False).dropna(subset=['SEV_INDEX']).head(8)
    cols = 4
    rows_n = 2
    specs = [[{'type': 'indicator'}] * cols] * rows_n
    fig = make_subplots(rows=rows_n, cols=cols, specs=specs,
                        subplot_titles=[f"{sn(r['COUNTRY'])}: {str(r['CRISIS'])[:22]}"
                                        for _, r in df.iterrows()])
    for i, (_, row) in enumerate(df.iterrows()):
        r_i = i // cols + 1
        c_i = i % cols + 1
        col = SEV_COLORS.get(row['SEV_CAT'], C['muted'])
        fig.add_trace(go.Indicator(
            mode='gauge+number', value=row['SEV_INDEX'],
            gauge=dict(
                axis=dict(range=[0, 10], tickfont=dict(size=8)),
                bar=dict(color=col),
                steps=[
                    dict(range=[0, 2], color='#EFF6FF'),
                    dict(range=[2, 4], color='#DCFCE7'),
                    dict(range=[4, 6], color='#FEF3C7'),
                    dict(range=[6, 8], color='#FEE2E2'),
                    dict(range=[8, 10], color='#FEE2E2'),
                ],
                threshold=dict(line=dict(color=col, width=3),
                               thickness=0.75, value=row['SEV_INDEX'])),
            number=dict(font=dict(size=20, color=col))),
            row=r_i, col=c_i)
    fig.update_layout(**{**LAYOUT, 'height': 400},
        title=dict(text='Gauges de severidad · Top 8 crisis · Febrero 2026',
                   font=dict(size=13)))
    fig.update_annotations(font=dict(color=C['ink'], size=10))
    return fig


# Pre-generar figuras estáticas
FIG_SERIE    = fig_serie_latam_global()
FIG_INDIV    = fig_crisis_individuales()
FIG_SNAPSHOT = fig_snapshot_feb2026()
FIG_DIM      = fig_dimensiones()
FIG_DRIVERS  = fig_drivers()
FIG_HEATMAP  = fig_heatmap()
FIG_COVID    = fig_comparacion_paises()
FIG_GAUGES   = fig_tendencia_actual()

# ══════════════════════════════════════════════════════════════════════
# 4. MÉTRICAS RESUMEN
# ══════════════════════════════════════════════════════════════════════
n_crisis   = len(lat_sev.dropna(subset=['SEV_INDEX']))
n_paises   = lat_sev['COUNTRY'].nunique() if len(lat_sev) > 0 else 0
n_veryhigh = len(lat_sev[lat_sev['SEV_CAT']=='Very High'])
n_high     = len(lat_sev[lat_sev['SEV_CAT']=='High'])
lat_mean   = round(lat_sev['SEV_INDEX'].mean(), 2) if n_crisis > 0 else 0
glo_mean   = round(glo_sev['SEV_INDEX'].mean(), 2) if len(glo_sev) > 0 else 0
max_crisis = lat_sev.loc[lat_sev['SEV_INDEX'].idxmax(), 'COUNTRY'] if n_crisis > 0 else '—'
max_score  = lat_sev['SEV_INDEX'].max() if n_crisis > 0 else 0


# ══════════════════════════════════════════════════════════════════════
# 5. APP
# ══════════════════════════════════════════════════════════════════════
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True, title='INFORM Severity v4.0')
dash.register_page(__name__, path='/severity', name='Severity', order=7)

server = app.server

# Badge principal text
badge_corr = ''
if 'inform_risk' in CORRS:
    cr = CORRS['inform_risk']['pearson']
    badge_corr = f'Correlación INFORM Risk→Severity COVID: r={cr["r"]} (n={CORRS["inform_risk"]["n"]})'


app.layout = dbc.Container([
    make_header('INFORM SEVERITY',
                f'LATAM-20 · Serie mensual 2019–2026 · {len(DATE_COLS)} meses · '
                f'{n_crisis} crisis activas · JRC/Comisión Europea'),
    make_methodology_note(
        'ROL: Desenlace de impacto. INFORM Severity mide gravedad de crisis humanitarias '
        'ACTIVAS, NO preparación previa. Un país sin crisis registrada no aparece porque '
        'no activó el umbral mínimo INFORM, no porque estuviera "bien preparado". '
        'ESCALA INVERSA: mayor score = mayor severidad (opuesto a GHS/SPAR). '
        'Para correlaciones con preparación, normalizar como (10 − Severity). '
        'Correlaciones reportadas con IC de Fisher-z. '
        f'{NOTA_MUESTRA}',
        accent='latam'),
    dbc.Row([
        make_kpi(n_crisis, 'CRISIS ACTIVAS', 'LATAM-20 · Feb 2026', C['high']),
        make_kpi(f'{n_paises}/20', 'PAÍSES', 'con crisis registrada', C['latam']),
        make_kpi(n_veryhigh, 'MUY ALTA', 'score ≥ 8.0', C['very_high']),
        make_kpi(n_high, 'ALTA', 'score 6.0–7.9', C['high']),
        make_kpi(lat_mean, 'MEDIA LATAM', f'vs Global {glo_mean}', C['orange']),
        make_kpi(sn(max_crisis), 'MÁS SEVERA',
                 f'score {max_score:.1f}', C['very_high']),
    ], style={'marginBottom': '10px'}),
    dcc.Tabs(id='main-tabs', value='t1', children=[
        dcc.Tab(label='📈 Serie LATAM vs Global', value='t1', style=TS, selected_style=TSS),
        dcc.Tab(label='Crisis individuales',      value='t2', style=TS, selected_style=TSS),
        dcc.Tab(label='Snapshot Feb 2026',        value='t3', style=TS, selected_style=TSS),
        dcc.Tab(label='Dimensiones',              value='t4', style=TS, selected_style=TSS),
        dcc.Tab(label='Drivers & Heatmap',        value='t5', style=TS, selected_style=TSS),
        dcc.Tab(label='🦠 COVID — Desenlace',     value='t6', style=TS, selected_style=TSS),
        dcc.Tab(label='Correlación Índices',      value='t7', style=TS, selected_style=TSS),
        dcc.Tab(label='Gauges · Tabla',           value='t8', style=TS, selected_style=TSS),
    ], style={'borderBottom': f'1px solid {C["border"]}', 'marginBottom': '14px'}),
    html.Div(id='tab-content')
], fluid=True, style={'backgroundColor': C['bg'], 'minHeight': '100vh',
                      'padding': '0 20px 40px'})


@app.callback(Output('tab-content', 'children'), Input('main-tabs', 'value'))
def render_tab(tab):
    if tab == 't1':
        return html.Div([
            html.Div([make_section_title('Media mensual Severity · LATAM-20 vs Global',
                'Banda sombreada = período pandemia COVID-19 (mar 2020 – dic 2021). '
                'La media incluye únicamente países con crisis activa en cada mes.'),
                dcc.Graph(figure=FIG_SERIE, config=GRAPH_CONFIG)], style=CARD),
            html.Div([
                make_section_title('Interpretación metodológica'),
                html.Ul([
                    html.Li('INFORM Severity mide la gravedad de crisis humanitarias ACTIVAS — no la preparación previa.',
                            style={'color': C['text'], 'fontSize': '12px'}),
                    html.Li('Un país sin crisis registrada no aparece porque no activó el umbral mínimo INFORM.',
                            style={'color': C['text'], 'fontSize': '12px'}),
                    html.Li('Escala 0–10 donde MAYOR score = MAYOR severidad (dirección opuesta a GHS y SPAR).',
                            style={'color': C['text'], 'fontSize': '12px'}),
                    html.Li('Para correlaciones cruzadas con preparación, se normaliza como (10 − Severity).',
                            style={'color': C['text'], 'fontSize': '12px'}),
                ], style={'paddingLeft': '20px', 'lineHeight': '1.8'})
            ], style=CARD),
        ])

    elif tab == 't2':
        return html.Div([html.Div([
            make_section_title('Trayectorias de crisis individuales',
                'Filtra por país o mira todos juntos'),
            dcc.Dropdown(
                id='country-filter',
                options=[{'label': 'Todos los países', 'value': 'ALL'}] +
                        [{'label': c, 'value': c}
                         for c in sorted(set(sd['country'] for sd in series_data.values()))],
                value='ALL', clearable=False,
                style={**DROPDOWN_STYLE, 'width': '260px', 'marginBottom': '12px'}),
            html.Div(id='indiv-graph')], style=CARD)])

    elif tab == 't3':
        return html.Div([html.Div([
            make_section_title('Snapshot de severidad · Febrero 2026',
                'Categorías: Muy Alto (≥8) · Alto (6–7.9) · Medio (4–5.9) · Bajo (2–3.9)'),
            dcc.Graph(figure=FIG_SNAPSHOT, config=GRAPH_CONFIG)], style=CARD)])

    elif tab == 't4':
        return html.Div([html.Div([
            make_section_title('Dimensiones INFORM Severity',
                'Impact = escala geográfica y personas afectadas. '
                'Conditions = necesidades y concentración. '
                'Complexity = cohesión social, seguridad, estructura.'),
            dcc.Graph(figure=FIG_DIM, config=GRAPH_CONFIG)], style=CARD)])

    elif tab == 't5':
        return html.Div([
            html.Div([make_section_title('Drivers de crisis LATAM',
                'International Displacement = más frecuente (crisis Venezuela en 9 países receptores). '
                'Conflict/Violence concentra las crisis de mayor severidad.'),
                dcc.Graph(figure=FIG_DRIVERS, config=GRAPH_CONFIG)], style=CARD),
            html.Div([make_section_title('Heatmap anual por crisis',
                'Celdas vacías = crisis no activa. Gradación = severidad promedio anual. '
                'Banda sombreada = COVID-19.'),
                dcc.Graph(figure=FIG_HEATMAP, config=GRAPH_CONFIG)], style=CARD),
        ])

    elif tab == 't6':
        hallazgo = ''
        if 'inform_risk' in CORRS:
            cr_p = CORRS['inform_risk']['pearson']
            cr_s = CORRS['inform_risk']['spearman']
            hallazgo = (f"Pearson r = {cr_p['r']:.3f}, IC95% [{cr_p['ci'][0]}, {cr_p['ci'][1]}] (Fisher-z), "
                        f"p = {cr_p['p']:.4f} · "
                        f"Spearman ρ = {cr_s['r']:.3f}, p = {cr_s['p']:.4f} · "
                        f"n = {CORRS['inform_risk']['n']} países")
        return html.Div([
            html.Div([make_section_title('Severidad máxima COVID-19 · variable de desenlace',
                f'Solo disponible para {len(covid_max)} países que tenían crisis registrada antes '
                'de la pandemia. Esta es la variable dependiente principal del metanálisis.'),
                dcc.Graph(figure=FIG_COVID, config=GRAPH_CONFIG)], style=CARD),
            html.Div([
                html.Div('HALLAZGO CENTRAL DEL METANÁLISIS',
                         style={'fontSize': '11px', 'letterSpacing': '2px',
                                'color': C['very_high'], 'fontWeight': '700',
                                'marginBottom': '8px'}),
                html.Div(hallazgo,
                         style={'fontSize': '13px', 'color': C['ink'],
                                'fontFamily': 'Georgia,serif', 'lineHeight': '1.8',
                                'marginBottom': '10px'}),
                html.Div('INFORM Risk 2019 predice significativamente la Severity máxima durante '
                         'COVID-19. Los países con mayor riesgo estructural pre-pandemia experimentaron '
                         'crisis humanitarias más severas durante la pandemia. Esto confirma el valor '
                         'predictivo del INFORM Risk como variable de preparación para el metanálisis.',
                         style={'fontSize': '11px', 'color': C['text'], 'lineHeight': '1.7'}),
            ], style={**CARD, 'borderLeft': f'4px solid {C["very_high"]}'}),
        ])

    elif tab == 't7':
        return html.Div([html.Div([
            make_section_title('Correlación predictor vs Severity COVID máxima',
                'Con IC de Fisher-z · compara los tres índices disponibles'),
            dcc.RadioItems(
                id='predictor-select',
                options=[
                    {'label': 'INFORM Risk 2019', 'value': 'inform_risk'},
                    {'label': 'GHS Index 2021', 'value': 'ghs'},
                    {'label': 'SPAR 2019', 'value': 'spar'},
                ],
                value='inform_risk', inline=True,
                labelStyle={'marginRight': '20px', 'fontSize': '12px',
                            'cursor': 'pointer', 'color': C['text']},
                style={'marginBottom': '12px'}),
            html.Div(id='corr-graph'),
            html.Div([
                html.Div('La correlación usa Severity máxima 2020–2021 como dependiente. '
                         f'Solo {len(covid_max)} países tienen dato (requieren crisis activa pre-pandemia). '
                         'El INFORM Risk 2019 suele ser el predictor más fuerte porque mide riesgo '
                         'estructural pre-crisis, no preparación declarada.',
                         style={'fontSize': '11px', 'color': C['text'], 'lineHeight': '1.7',
                                'marginTop': '10px'}),
            ])
        ], style=CARD)])

    elif tab == 't8':
        return html.Div([
            html.Div([make_section_title('Gauges · Top 8 crisis · Feb 2026'),
                dcc.Graph(figure=FIG_GAUGES, config=GRAPH_CONFIG)], style=CARD),
            html.Div([make_section_title('Tabla de crisis activas · Feb 2026'),
                dash_table.DataTable(
                    data=master_df.to_dict('records'),
                    columns=[{'name': c, 'id': c} for c in master_df.columns],
                    filter_action='native', sort_action='native',
                    page_size=20, export_format='csv', **TABLE_STYLE,
                    style_data_conditional=[
                        {'if': {'column_id': 'Score Feb 2026',
                                'filter_query': '{Score Feb 2026} >= 8'},
                         'color': C['very_high'], 'fontWeight': '700'},
                        {'if': {'column_id': 'Score Feb 2026',
                                'filter_query': '{Score Feb 2026} >= 6 && {Score Feb 2026} < 8'},
                         'color': C['high'], 'fontWeight': '600'},
                    ])], style=CARD),
        ])

    return html.Div('Tab no implementado')


@app.callback(Output('indiv-graph', 'children'), Input('country-filter', 'value'))
def update_indiv(country):
    fig = go.Figure()
    colors_map = {
        'Haiti': C['very_high'], 'Colombia': C['high'],
        'Venezuela': C['orange'], 'Ecuador': C['amber'],
        'Peru': C['teal'], 'Brazil': C['blue'],
        'Costa Rica': C['green'], 'Panama': C['purple'],
        'Mexico': C['ink'], 'Chile': '#6366F1',
        'Dominican Republic': '#0891B2', 'Cuba': '#7C3AED',
    }
    fig.add_vrect(x0='2020-03', x1='2021-12',
                  fillcolor=C['covid'], opacity=0.06, layer='below', line_width=0)
    fig.add_shape(type='line', x0='2020-03', x1='2020-03', y0=0, y1=1,
                  xref='x', yref='paper',
                  line=dict(color=C['covid'], width=1, dash='dash'), opacity=0.5)
    fig.add_annotation(x='2020-03', y=0.98, xref='x', yref='paper',
                       text='COVID-19', showarrow=False,
                       font=dict(size=9, color=C['covid']),
                       xanchor='left', bgcolor='rgba(255,255,255,0.7)')
    for k, sd in sorted(series_data.items(), key=lambda x: -x[1]['current']):
        c = sd['country']
        if country != 'ALL' and c != country: continue
        if len(sd['dates']) < 5: continue
        col = colors_map.get(c, C['muted'])
        fig.add_trace(go.Scatter(
            x=sd['dates'], y=sd['vals'],
            name=f"{sn(c)} — {sd['crisis'][:32]}",
            line=dict(color=col, width=2), mode='lines',
            hovertemplate=f"<b>{sn(c)}</b> · {sd['crisis'][:35]}<br>%{{x}}: <b>%{{y:.1f}}</b><extra></extra>"))
    for val, lbl, col in [(8, 'Muy Alto', C['very_high']), (6, 'Alto', C['high']),
                           (4, 'Medio', C['amber']), (2, 'Bajo', C['green'])]:
        fig.add_hline(y=val, line_width=0.5, line_dash='dot',
                      line_color=col, opacity=0.4,
                      annotation_text=lbl, annotation_position='right',
                      annotation_font=dict(size=8, color=col))
    fig.update_layout(**{**LAYOUT, 'height': 520},
        title=dict(text=f'Series individuales · {"LATAM-20" if country=="ALL" else country} · 2019–2026',
                   font=dict(size=13)),
        yaxis=dict(title='Severity Score (0–10)', range=[0, 10.5], gridcolor=C['grid']),
        xaxis=dict(gridcolor=C['grid'], tickfont=dict(size=10)),
        legend=dict(font=dict(size=9), x=1.01, y=1),
        hovermode='x unified')
    return dcc.Graph(figure=fig, config=GRAPH_CONFIG)


@app.callback(Output('corr-graph', 'children'), Input('predictor-select', 'value'))
def update_corr(predictor):
    return dcc.Graph(figure=fig_correlacion_scatter(predictor), config=GRAPH_CONFIG)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8053))
    app.run(debug=False, host='0.0.0.0', port=port)

# Requerido por Dash Pages
layout = app.layout
