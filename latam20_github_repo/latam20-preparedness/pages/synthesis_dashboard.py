"""
SYNTHESIS Dashboard v1.0 — Metanálisis Cross-Index · LATAM-20
══════════════════════════════════════════════════════════════════════
El corazón del metanálisis. Integra los 5 índices pandémicos en un
solo análisis comparativo sobre la misma muestra LATAM-20.

ÍNDICES INTEGRADOS:
  • GHS Index 2021            — Preparación (externo)
  • SPAR Overall 2019 + 2024  — Capacidad RSI (autorreporte)
  • INFORM Risk 2019 + 2024   — Riesgo estructural (externo, invertido)
  • OxCGRT Stringency 2020    — Respuesta pandémica (desenlace proceso)
  • INFORM Severity máx 20-21 — Impacto humanitario (desenlace)

ANÁLISIS PRINCIPALES:
  1. Matriz 5×5 de correlaciones entre índices (Pearson + Spearman + IC Fisher-z)
  2. Scatter GHS×INFORM 2021 coloreado por OxCGRT stringency
  3. Radar multi-índice por país (5 ejes normalizados 0–100)
  4. Tabla maestra 20 países × 5 índices · exportable
  5. Ranking consolidado "Score compuesto de preparación"

DIRECCIONALIDAD (crítico):
  GHS, SPAR:            mayor = mejor preparación
  INFORM Risk:          mayor = más riesgo (invertido)
  OxCGRT Stringency:    mayor = respuesta más intensa (≠ más correcta)
  INFORM Severity:      mayor = impacto humanitario más severo

Para la matriz de correlaciones todos los índices se normalizan 0–100
con dirección "mayor = mejor preparación". INFORM se invierte como
(10 − score) × 10. Severity no se normaliza porque es desenlace.

Puerto: 8060
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

from latam_common import (
    LATAM_ISO3, LATAM_20 as LATAM, LATAM_20_SORTED,
    SHORT, sn, data_path, NOTA_MUESTRA, SUBREGION,
)
from theme import (
    C, LAYOUT, CARD, TS, TSS, TABLE_STYLE, DROPDOWN_STYLE,
    INDEX_COLORS, SUBREGION_COLORS,
    DIVERGING_COLORSCALE, RISK_COLORSCALE, GRAPH_CONFIG,
    rgba,
    make_header, make_kpi, make_section_title, make_methodology_note,
)
from stats_utils import correlation_ci, mean_ci

ISO_TO_NAME = LATAM_ISO3
NAME_TO_ISO = {v: k for k, v in LATAM_ISO3.items()}


# ══════════════════════════════════════════════════════════════════════
# 1. CARGA CROSS-INDEX — tolerante a faltantes
# ══════════════════════════════════════════════════════════════════════

def _safe_load_excel(key, **kwargs):
    try:
        return pd.read_excel(data_path(key, engine='openpyxl'), **kwargs)
    except Exception as e:
        print(f'⚠️  No se pudo cargar {key}: {e}')
        return None


def _safe_load_csv(key, **kwargs):
    try:
        return pd.read_csv(data_path(key), **kwargs)
    except Exception as e:
        print(f'⚠️  No se pudo cargar {key}: {e}')
        return None


# --- INFORM Risk 2017-2026 ---
inform_raw = _safe_load_csv('inform_trend',
    usecols=['Iso3', 'IndicatorId', 'INFORMYear', 'IndicatorScore'])
INFORM_RISK_2019 = {}
INFORM_RISK_2024 = {}
if inform_raw is not None:
    for yr, d in [(2019, INFORM_RISK_2019), (2024, INFORM_RISK_2024)]:
        sub = inform_raw[(inform_raw['IndicatorId']=='INFORM') &
                         (inform_raw['INFORMYear']==yr) &
                         (inform_raw['Iso3'].isin(LATAM_ISO3.keys()))]
        for _, r in sub.iterrows():
            d[ISO_TO_NAME[r['Iso3']]] = round(float(r['IndicatorScore']), 2)

# --- GHS 2019 y 2021 ---
ghs_raw = _safe_load_csv('ghs')
GHS_2019, GHS_2021 = {}, {}
if ghs_raw is not None:
    for yr, d in [(2019, GHS_2019), (2021, GHS_2021)]:
        sub = ghs_raw[(ghs_raw['Year']==yr) &
                      (ghs_raw['Country'].isin(LATAM))]
        for _, r in sub.iterrows():
            v = r['OVERALL SCORE']
            if pd.notna(v):
                d[r['Country']] = round(float(v), 2)

# --- SPAR Overall 2019 y 2024 ---
spar_raw = _safe_load_csv('spar_latam')
SPAR_2019, SPAR_2024 = {}, {}
if spar_raw is not None:
    for yr, d in [(2019, SPAR_2019), (2024, SPAR_2024)]:
        sub = spar_raw[(spar_raw['Year']==yr) &
                       (spar_raw['Country'].isin(LATAM))]
        for _, r in sub.iterrows():
            v = r['SPAR_Overall']
            if pd.notna(v):
                d[r['Country']] = round(float(v), 2)

# --- OxCGRT Stringency 2020 (media y pico) ---
oxcgrt_raw = _safe_load_csv('oxcgrt_compact', low_memory=False)
OXCGRT_MEAN_2020, OXCGRT_PEAK_2020 = {}, {}
if oxcgrt_raw is not None:
    oxcgrt_raw['Date'] = pd.to_datetime(oxcgrt_raw['Date'], format='%Y%m%d')
    for iso, name in LATAM_ISO3.items():
        sub = oxcgrt_raw[(oxcgrt_raw['CountryCode']==iso) &
                         (oxcgrt_raw['Jurisdiction']=='NAT_TOTAL') &
                         (oxcgrt_raw['Date'].dt.year==2020)]
        s = pd.to_numeric(sub['StringencyIndex_Average'], errors='coerce').dropna()
        if len(s) > 0:
            OXCGRT_MEAN_2020[name] = round(float(s.mean()), 2)
            OXCGRT_PEAK_2020[name] = round(float(s.max()), 2)

# --- INFORM Severity máxima COVID ---
sev_raw = _safe_load_excel('severity', sheet_name='Trends', header=0)
SEV_COVID_MAX = {}
if sev_raw is not None:
    sev_raw.columns = sev_raw.iloc[0]
    sev_df = sev_raw.iloc[1:].reset_index(drop=True)
    sev_df.columns.name = None
    DATE_COLS = [c for c in sev_df.columns if hasattr(c, 'year')]
    lat_sev_trends = sev_df[sev_df['Country'].isin(LATAM)]
    for _, row in lat_sev_trends.iterrows():
        c = row['Country']
        vals_covid = []
        for dc in DATE_COLS:
            if dc.year in [2020, 2021]:
                v = row[dc]
                if v not in ['-', 'x', None] and str(v) != 'nan':
                    try: vals_covid.append(float(v))
                    except Exception: pass
        if vals_covid:
            mx = max(vals_covid)
            if c not in SEV_COVID_MAX or mx > SEV_COVID_MAX[c]:
                SEV_COVID_MAX[c] = round(mx, 2)


# ══════════════════════════════════════════════════════════════════════
# 2. TABLA MAESTRA CROSS-INDEX
# ══════════════════════════════════════════════════════════════════════

def _inform_inverted(x):
    """INFORM 0-10 escala invertida → 0-100 escala 'preparación'."""
    if x is None or pd.isna(x): return None
    return round((10 - float(x)) * 10, 2)


master_rows = []
for c in LATAM:
    master_rows.append({
        'País':                  c,
        'Subregión':             SUBREGION.get(c, '—'),
        'GHS_2019':              GHS_2019.get(c),
        'GHS_2021':              GHS_2021.get(c),
        'SPAR_2019':             SPAR_2019.get(c),
        'SPAR_2024':             SPAR_2024.get(c),
        'INFORM_Risk_2019':      INFORM_RISK_2019.get(c),
        'INFORM_Risk_2024':      INFORM_RISK_2024.get(c),
        'INFORM_Prep_2019':      _inform_inverted(INFORM_RISK_2019.get(c)),
        'INFORM_Prep_2024':      _inform_inverted(INFORM_RISK_2024.get(c)),
        'OxCGRT_Mean_2020':      OXCGRT_MEAN_2020.get(c),
        'OxCGRT_Peak_2020':      OXCGRT_PEAK_2020.get(c),
        'Severity_COVID_Max':    SEV_COVID_MAX.get(c),
    })
MASTER = pd.DataFrame(master_rows)

# Score compuesto de preparación (media de GHS 2021 + SPAR 2024 + INFORM_Prep 2024)
def _compuesto(row):
    vals = [row['GHS_2021'], row['SPAR_2024'], row['INFORM_Prep_2024']]
    vals = [v for v in vals if v is not None and pd.notna(v)]
    return round(np.mean(vals), 2) if vals else None


MASTER['Score_Compuesto'] = MASTER.apply(_compuesto, axis=1)


# ══════════════════════════════════════════════════════════════════════
# 3. MATRIZ DE CORRELACIONES 5x5 — núcleo del metanálisis
# ══════════════════════════════════════════════════════════════════════
# Todos los índices normalizados "mayor = mejor preparación" o el
# desenlace correspondiente. Se reportan n por par (distintos por NAs).

CORR_COLS = {
    'GHS 2021':       'GHS_2021',
    'SPAR 2024':      'SPAR_2024',
    'INFORM Prep 24': 'INFORM_Prep_2024',
    'OxCGRT max 20':  'OxCGRT_Peak_2020',
    'Severity COVID': 'Severity_COVID_Max',
}
CORR_LABELS = list(CORR_COLS.keys())
CORR_FIELDS = list(CORR_COLS.values())


def build_corr_matrix(method='pearson'):
    """Retorna DataFrames con r, p, n, ic_lo, ic_hi para cada par."""
    r_mat = pd.DataFrame(index=CORR_LABELS, columns=CORR_LABELS, dtype=float)
    p_mat = pd.DataFrame(index=CORR_LABELS, columns=CORR_LABELS, dtype=float)
    n_mat = pd.DataFrame(index=CORR_LABELS, columns=CORR_LABELS, dtype=int)
    ci_lo = pd.DataFrame(index=CORR_LABELS, columns=CORR_LABELS, dtype=float)
    ci_hi = pd.DataFrame(index=CORR_LABELS, columns=CORR_LABELS, dtype=float)
    for la, fa in CORR_COLS.items():
        for lb, fb in CORR_COLS.items():
            sub = MASTER[[fa, fb]].dropna()
            n = len(sub)
            if la == lb:
                r_mat.loc[la, lb] = 1.0
                p_mat.loc[la, lb] = 0.0
                n_mat.loc[la, lb] = n
                ci_lo.loc[la, lb] = 1.0
                ci_hi.loc[la, lb] = 1.0
            elif n >= 3:
                cr = correlation_ci(sub[fa].values, sub[fb].values, method=method)
                r_mat.loc[la, lb] = cr['r']
                p_mat.loc[la, lb] = cr['p']
                n_mat.loc[la, lb] = cr['n']
                ci_lo.loc[la, lb] = cr['ci'][0]
                ci_hi.loc[la, lb] = cr['ci'][1]
            else:
                r_mat.loc[la, lb] = np.nan
                p_mat.loc[la, lb] = np.nan
                n_mat.loc[la, lb] = n
                ci_lo.loc[la, lb] = np.nan
                ci_hi.loc[la, lb] = np.nan
    return r_mat, p_mat, n_mat, ci_lo, ci_hi


R_PEAR, P_PEAR, N_PEAR, CI_LO_P, CI_HI_P = build_corr_matrix('pearson')
R_SPEAR, P_SPEAR, N_SPEAR, CI_LO_S, CI_HI_S = build_corr_matrix('spearman')

# Tabla pareja-a-pareja (para presentación plana)
pair_rows = []
for i, la in enumerate(CORR_LABELS):
    for j, lb in enumerate(CORR_LABELS):
        if i >= j: continue  # solo triángulo superior
        pair_rows.append({
            'Par':        f'{la} × {lb}',
            'n':          int(N_PEAR.loc[la, lb]),
            'Pearson r':  round(R_PEAR.loc[la, lb], 3) if pd.notna(R_PEAR.loc[la, lb]) else np.nan,
            'Pearson p':  round(P_PEAR.loc[la, lb], 4) if pd.notna(P_PEAR.loc[la, lb]) else np.nan,
            'IC95 Pearson': (f"[{CI_LO_P.loc[la, lb]:.2f}, {CI_HI_P.loc[la, lb]:.2f}]"
                             if pd.notna(CI_LO_P.loc[la, lb]) else '—'),
            'Spearman ρ': round(R_SPEAR.loc[la, lb], 3) if pd.notna(R_SPEAR.loc[la, lb]) else np.nan,
            'Spearman p': round(P_SPEAR.loc[la, lb], 4) if pd.notna(P_SPEAR.loc[la, lb]) else np.nan,
            'IC95 Spearman': (f"[{CI_LO_S.loc[la, lb]:.2f}, {CI_HI_S.loc[la, lb]:.2f}]"
                              if pd.notna(CI_LO_S.loc[la, lb]) else '—'),
            'Sig. Bonferroni': ('sig.*' if pd.notna(P_PEAR.loc[la, lb])
                                and P_PEAR.loc[la, lb] < (0.05/10)  # 10 pares = C(5,2)
                                else 'n.s.'),
        })
PAIR_DF = pd.DataFrame(pair_rows)


# ══════════════════════════════════════════════════════════════════════
# 4. NORMALIZACIÓN 0–100 PARA RADAR
# ══════════════════════════════════════════════════════════════════════
# Cada índice se normaliza al rango observado en LATAM-20 para hacer
# el radar comparable. Desenlaces (OxCGRT, Severity) también se llevan
# a 0-100 manteniendo su direccionalidad original.

def _normalize_0_100(values_dict):
    """Min-max sobre los valores observados en LATAM-20."""
    vals = [v for v in values_dict.values() if v is not None and pd.notna(v)]
    if not vals: return {}
    lo, hi = min(vals), max(vals)
    if hi == lo: return {c: 50 for c in values_dict}
    return {c: round((v - lo) / (hi - lo) * 100, 1) if v is not None else None
            for c, v in values_dict.items()}


RADAR_NORM = {
    'GHS':           _normalize_0_100(GHS_2021),
    'SPAR':          _normalize_0_100(SPAR_2024),
    'INFORM Prep':   _normalize_0_100({c: _inform_inverted(v)
                                        for c, v in INFORM_RISK_2024.items()}),
    'OxCGRT':        _normalize_0_100(OXCGRT_PEAK_2020),
    'Severity inv':  _normalize_0_100({c: (10 - v) if v is not None else None
                                        for c, v in SEV_COVID_MAX.items()}),
}


# KPIs resumen
N_COUNTRIES = len(LATAM)
N_PAIRS_SIG = int((PAIR_DF['Sig. Bonferroni']=='sig.*').sum())
N_COMP = int(MASTER['Score_Compuesto'].notna().sum())
PROM_COMP = round(MASTER['Score_Compuesto'].mean(), 2) if N_COMP > 0 else 0


# ══════════════════════════════════════════════════════════════════════
# 5. FIGURAS
# ══════════════════════════════════════════════════════════════════════

def fig_matriz_correlaciones():
    """Matriz 5×5 de correlaciones Pearson + celdas anotadas con r y n."""
    z = R_PEAR.values.astype(float)
    text = []
    for i, la in enumerate(CORR_LABELS):
        row_txt = []
        for j, lb in enumerate(CORR_LABELS):
            r = R_PEAR.loc[la, lb]
            n = int(N_PEAR.loc[la, lb])
            p = P_PEAR.loc[la, lb]
            if pd.isna(r):
                row_txt.append('—')
            elif i == j:
                row_txt.append('1.00')
            else:
                sig = ''
                if pd.notna(p):
                    if p < 0.001: sig = '***'
                    elif p < 0.01: sig = '**'
                    elif p < 0.05: sig = '*'
                row_txt.append(f'{r:.2f}{sig}<br><span style="font-size:9px">n={n}</span>')
        text.append(row_txt)

    fig = go.Figure(go.Heatmap(
        z=z, x=CORR_LABELS, y=CORR_LABELS,
        text=text, texttemplate='%{text}',
        textfont=dict(size=11, color=C['ink']),
        colorscale='RdBu_r', zmid=0, zmin=-1, zmax=1,
        hovertemplate='<b>%{y} × %{x}</b><br>r=%{z:.3f}<extra></extra>',
        colorbar=dict(title=dict(text='Pearson r', font=dict(color=C['ink'])),
                      tickfont=dict(size=9, color=C['ink']))))
    fig.update_layout(**{**LAYOUT, 'height': 520},
        title=dict(text='Matriz 5×5 · Correlaciones Pearson entre los 5 índices · LATAM-20 · '
                        f'* p<.05 · ** p<.01 · *** p<.001 · Bonferroni: {N_PAIRS_SIG}/10 pares sig.',
                   font=dict(size=10, color=C['muted'])),
        xaxis=dict(side='top', gridcolor=C['grid']),
        yaxis=dict(autorange='reversed', gridcolor=C['grid']),
        margin=dict(l=130, r=60, t=80, b=30))
    return fig


def fig_scatter_ghs_inform(size_by='OxCGRT_Peak_2020'):
    """Scatter GHS 2021 × INFORM Prep 2021, burbuja por OxCGRT."""
    df = MASTER.dropna(subset=['GHS_2021', 'INFORM_Prep_2019']).copy()
    if size_by in df.columns:
        df = df[df[size_by].notna()]
    cr = correlation_ci(df['GHS_2021'].values, df['INFORM_Prep_2019'].values, method='pearson')
    fig = go.Figure()
    # Línea de regresión
    x = df['GHS_2021'].values
    y = df['INFORM_Prep_2019'].values
    if len(x) >= 3:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=slope*xs+intercept, mode='lines',
            line=dict(color=C['amber'], width=1.5, dash='dash'),
            showlegend=False, hoverinfo='skip'))
    # Puntos
    sizes = df[size_by].fillna(df[size_by].mean()) if size_by in df.columns else [12]*len(df)
    size_min, size_max = 10, 30
    if size_by in df.columns and len(df[size_by].dropna()) > 0:
        s_lo, s_hi = df[size_by].min(), df[size_by].max()
        if s_hi > s_lo:
            sizes = size_min + (df[size_by] - s_lo) / (s_hi - s_lo) * (size_max - size_min)
    fig.add_trace(go.Scatter(
        x=df['GHS_2021'], y=df['INFORM_Prep_2019'],
        mode='markers+text',
        text=[sn(c) for c in df['País']], textposition='top center',
        textfont=dict(size=9, color=C['ink']),
        marker=dict(size=sizes,
                    color=[SUBREGION_COLORS.get(s, C['muted']) for s in df['Subregión']],
                    line=dict(color='white', width=1.5), opacity=0.85),
        customdata=df[['País', 'Subregión', size_by]].values if size_by in df.columns
                   else df[['País', 'Subregión']].values,
        hovertemplate=('<b>%{customdata[0]}</b> (%{customdata[1]})<br>'
                       'GHS 2021: %{x:.1f}<br>INFORM Prep 2019: %{y:.1f}<br>'
                       + (f'{size_by}: %{{customdata[2]:.1f}}' if size_by in df.columns else '')
                       + '<extra></extra>'),
        showlegend=False))
    # Leyenda de subregiones manual
    for sr, col in SUBREGION_COLORS.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            marker=dict(size=12, color=col), name=sr))
    title = (f'GHS 2021 × INFORM Prep 2019 · Tamaño: {size_by} · '
             f'n={cr["n"]} · Pearson r={cr["r"]:.3f} '
             f'IC95% [{cr["ci"][0]}, {cr["ci"][1]}], p={cr["p"]:.4f}')
    fig.update_layout(**{**LAYOUT, 'height': 520},
        title=dict(text=title, font=dict(size=10, color=C['muted'])),
        xaxis=dict(title='GHS Overall 2021 (mayor = mejor preparación)',
                   gridcolor=C['grid']),
        yaxis=dict(title='INFORM Prep 2019 · (10−Risk)×10 (mayor = mejor)',
                   gridcolor=C['grid']),
        legend=dict(orientation='h', y=-0.14, bgcolor='rgba(0,0,0,0)',
                    font=dict(size=9)),
        margin=dict(l=60, r=20, t=75, b=70))
    return fig


def fig_radar_pais(country='Mexico'):
    """Radar multi-índice para un país: 5 ejes normalizados 0-100."""
    axes = ['GHS', 'SPAR', 'INFORM Prep', 'OxCGRT', 'Severity inv']
    vals = [RADAR_NORM[a].get(country) for a in axes]
    # Media LATAM para cada eje
    lat_means = []
    for a in axes:
        v_list = [v for v in RADAR_NORM[a].values() if v is not None]
        lat_means.append(round(np.mean(v_list), 1) if v_list else 0)
    theta = axes + [axes[0]]
    vals_plot = [v if v is not None else 0 for v in vals] + [vals[0] if vals[0] is not None else 0]
    lat_plot = lat_means + [lat_means[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=lat_plot, theta=theta, name='Media LATAM-20',
        line=dict(color=C['muted'], width=1.5, dash='dot'),
        fill='toself', fillcolor=rgba(C['muted'], 0.08)))
    fig.add_trace(go.Scatterpolar(
        r=vals_plot, theta=theta, name=country,
        line=dict(color=C['latam'], width=2.8),
        fill='toself', fillcolor=rgba(C['latam'], 0.18)))
    fig.update_layout(**{**LAYOUT, 'height': 500},
        title=dict(text=f'{country} · Perfil multi-índice (0-100 min-max LATAM-20)',
                   font=dict(size=12, color=C['ink'])),
        polar=dict(bgcolor=C['white'],
                   radialaxis=dict(visible=True, range=[0, 100],
                                   tickfont=dict(size=8, color=C['muted']),
                                   gridcolor=C['grid'], linecolor=C['border']),
                   angularaxis=dict(tickfont=dict(size=10, color=C['ink']),
                                    gridcolor=C['grid'], linecolor=C['border'])),
        legend=dict(orientation='h', y=-0.08, bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=60, r=60, t=60, b=50))
    return fig


def fig_radar_comparativo(countries):
    """Radar comparativo multi-país (hasta 5)."""
    axes = ['GHS', 'SPAR', 'INFORM Prep', 'OxCGRT', 'Severity inv']
    pal = [C['latam'], C['green'], C['amber'], C['purple'], C['teal']]
    theta = axes + [axes[0]]
    fig = go.Figure()
    for i, country in enumerate(countries[:5]):
        col = pal[i % len(pal)]
        vals = [RADAR_NORM[a].get(country) for a in axes]
        vals_plot = ([v if v is not None else 0 for v in vals]
                     + [vals[0] if vals[0] is not None else 0])
        fig.add_trace(go.Scatterpolar(
            r=vals_plot, theta=theta, name=sn(country),
            line=dict(color=col, width=2.2),
            fill='toself', fillcolor=rgba(col, 0.08)))
    fig.update_layout(**{**LAYOUT, 'height': 520},
        polar=dict(bgcolor=C['white'],
                   radialaxis=dict(visible=True, range=[0, 100],
                                   tickfont=dict(size=8, color=C['muted']),
                                   gridcolor=C['grid'], linecolor=C['border']),
                   angularaxis=dict(tickfont=dict(size=10, color=C['ink']),
                                    gridcolor=C['grid'], linecolor=C['border'])),
        legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=60, r=60, t=40, b=50))
    return fig


def fig_ranking_compuesto():
    """Ranking por score compuesto de preparación."""
    df = MASTER.dropna(subset=['Score_Compuesto']).sort_values('Score_Compuesto',
                                                                ascending=True)
    colors = [SUBREGION_COLORS.get(s, C['muted']) for s in df['Subregión']]
    fig = go.Figure(go.Bar(
        y=[sn(c) for c in df['País']], x=df['Score_Compuesto'],
        orientation='h', marker=dict(color=colors, line_width=0),
        customdata=df[['GHS_2021', 'SPAR_2024', 'INFORM_Prep_2024',
                       'Subregión']].values,
        hovertemplate=('<b>%{y}</b><br>Compuesto: %{x:.1f}<br>'
                       'GHS 2021: %{customdata[0]}<br>'
                       'SPAR 2024: %{customdata[1]}<br>'
                       'INFORM Prep 2024: %{customdata[2]}<br>'
                       'Subregión: %{customdata[3]}<extra></extra>')))
    mean_v = df['Score_Compuesto'].mean()
    fig.add_vline(x=mean_v, line_dash='dash', line_color=C['amber'],
                  annotation_text=f'Media: {mean_v:.1f}',
                  annotation_font=dict(size=9, color=C['amber']))
    fig.update_layout(**{**LAYOUT, 'height': 560},
        title=dict(text='Score compuesto de preparación · media(GHS 2021, SPAR 2024, INFORM Prep 2024)',
                   font=dict(size=10, color=C['muted'])),
        xaxis=dict(title='Score compuesto (0-100, mayor = mejor preparación)',
                   gridcolor=C['grid']),
        yaxis=dict(gridcolor=C['grid']),
        margin=dict(l=100, r=40, t=55, b=40))
    return fig


def fig_stripplot_por_indice():
    """Stripplot (puntos por país) para cada uno de los 5 índices normalizados."""
    axes = ['GHS', 'SPAR', 'INFORM Prep', 'OxCGRT', 'Severity inv']
    fig = go.Figure()
    for i, a in enumerate(axes):
        countries = [c for c in LATAM if RADAR_NORM[a].get(c) is not None]
        vals = [RADAR_NORM[a][c] for c in countries]
        colors = [SUBREGION_COLORS.get(SUBREGION.get(c), C['muted'])
                  for c in countries]
        fig.add_trace(go.Scatter(
            x=[i] * len(countries), y=vals,
            mode='markers+text',
            text=[sn(c) for c in countries],
            textposition='middle right',
            textfont=dict(size=8, color=C['ink']),
            marker=dict(size=11, color=colors,
                        line=dict(color='white', width=1), opacity=0.85),
            name=a, showlegend=False,
            hovertemplate=(f'<b>{a}</b><br>%{{text}}: %{{y:.1f}}<extra></extra>')))
        # Media
        fig.add_shape(type='line', x0=i-0.15, x1=i+0.15,
                      y0=np.mean(vals), y1=np.mean(vals),
                      line=dict(color=C['amber'], width=2))
    fig.update_layout(**{**LAYOUT, 'height': 540},
        title=dict(text='Distribución por índice · LATAM-20 · Línea ámbar = media · valores normalizados min-max 0-100',
                   font=dict(size=10, color=C['muted'])),
        xaxis=dict(tickvals=list(range(len(axes))), ticktext=axes,
                   gridcolor=C['grid'], tickfont=dict(size=11)),
        yaxis=dict(title='Score normalizado (0-100)',
                   gridcolor=C['grid'], range=[-5, 105]),
        margin=dict(l=60, r=20, t=60, b=50))
    return fig


def fig_heatmap_pais_indice():
    """Heatmap países (ordenados por compuesto) × 5 índices normalizados."""
    df = MASTER.dropna(subset=['Score_Compuesto']).sort_values('Score_Compuesto',
                                                                ascending=False)
    axes = ['GHS', 'SPAR', 'INFORM Prep', 'OxCGRT', 'Severity inv']
    z = []
    for _, row in df.iterrows():
        c = row['País']
        z.append([RADAR_NORM[a].get(c) for a in axes])
    z = np.array([[v if v is not None else np.nan for v in row] for row in z])
    text = np.where(np.isnan(z), 'N/D', np.round(z, 0).astype('object'))
    fig = go.Figure(go.Heatmap(
        z=z, x=axes, y=[sn(c) for c in df['País']],
        text=text, texttemplate='%{text}',
        textfont=dict(size=9, color=C['ink']),
        colorscale=[[0, C['red']], [0.5, C['amber']], [1, C['green']]],
        zmin=0, zmax=100,
        hovertemplate='<b>%{y}</b> · %{x}: %{z:.1f}<extra></extra>',
        colorbar=dict(title=dict(text='0-100 norm.', font=dict(color=C['ink'])),
                      tickfont=dict(size=9, color=C['ink']))))
    fig.update_layout(**{**LAYOUT, 'height': 560},
        xaxis=dict(side='top', gridcolor=C['grid'], tickfont=dict(size=10)),
        yaxis=dict(gridcolor=C['grid']),
        margin=dict(l=100, r=20, t=70, b=20))
    return fig


# ══════════════════════════════════════════════════════════════════════
# 6. APP
# ══════════════════════════════════════════════════════════════════════
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True, title='Synthesis v1.0')
dash.register_page(__name__, path='/synthesis', name='Síntesis', order=2)

server = app.server

app.layout = dbc.Container([
    make_header('SYNTHESIS · METANÁLISIS CROSS-INDEX',
                'LATAM-20 · GHS × SPAR × INFORM Risk × OxCGRT × INFORM Severity · Integración 5 índices'),
    make_methodology_note(
        'Este dashboard integra los 5 índices pandémicos sobre la misma muestra LATAM-20. '
        'Direccionalidad unificada: GHS, SPAR, INFORM Prep → mayor = mejor. '
        'OxCGRT Stringency → mayor = respuesta más intensa (no necesariamente correcta). '
        'Severity → mayor = impacto humanitario más severo. '
        'Para correlaciones se normaliza INFORM como (10−Risk)×10. '
        'IC de Fisher-z reportados. Sig. Bonferroni para 10 pares (α=0.005). '
        'Severity solo disponible en ~7-8 países con crisis activa pre-pandemia → '
        f'n reducido para correlaciones con Severity. {NOTA_MUESTRA}',
        accent='latam'),
    dbc.Row([
        make_kpi(str(N_COUNTRIES), 'PAÍSES', 'LATAM-20', C['latam']),
        make_kpi('5', 'ÍNDICES INTEGRADOS', 'GHS·SPAR·INFORM·OxCGRT·Severity', C['blue']),
        make_kpi(f'{N_PAIRS_SIG}/10', 'PARES SIG.', f'Bonferroni α={0.05/10:.3f}', C['amber']),
        make_kpi(f'{PROM_COMP:.1f}', 'COMPUESTO MEDIO',
                 f'n={N_COMP} países', C['green']),
        make_kpi(str(len(SEV_COVID_MAX)), 'CON SEVERITY',
                 'Crisis activa pre-pandemia', C['red']),
        make_kpi(str(len(OXCGRT_PEAK_2020)), 'CON OXCGRT',
                 'Stringency 2020 completo', C['purple']),
    ], style={'marginBottom': '10px'}),
    dcc.Tabs(id='tabs', value='t1', children=[
        dcc.Tab(label='🔗 Matriz 5×5',      value='t1', style=TS, selected_style=TSS),
        dcc.Tab(label='Scatter GHS×INFORM',  value='t2', style=TS, selected_style=TSS),
        dcc.Tab(label='🎯 Radar por país',   value='t3', style=TS, selected_style=TSS),
        dcc.Tab(label='Radar comparativo',   value='t4', style=TS, selected_style=TSS),
        dcc.Tab(label='🏆 Ranking compuesto', value='t5', style=TS, selected_style=TSS),
        dcc.Tab(label='Distribución índices', value='t6', style=TS, selected_style=TSS),
        dcc.Tab(label='Heatmap cross-index', value='t7', style=TS, selected_style=TSS),
        dcc.Tab(label='📊 Tabla maestra',    value='t8', style=TS, selected_style=TSS),
    ], style={'borderBottom': f'1px solid {C["border"]}', 'marginBottom': '14px'}),
    html.Div(id='content')
], fluid=True, style={'backgroundColor': C['bg'], 'minHeight': '100vh',
                      'padding': '0 20px 40px'})


@app.callback(Output('content', 'children'), Input('tabs', 'value'))
def render(tab):
    if tab == 't1':
        return html.Div([
            html.Div([make_section_title(
                'Matriz 5×5 · Correlaciones cross-index',
                'El núcleo cuantitativo del metanálisis. Cada celda muestra r, significancia y n de '
                'la correlación de Pearson entre dos índices sobre los países LATAM-20 '
                'con dato disponible en ambos.'),
                dcc.Graph(figure=fig_matriz_correlaciones(), config=GRAPH_CONFIG)], style=CARD),
            html.Div([make_section_title('Tabla completa · 10 pares · Pearson y Spearman con IC Fisher-z'),
                dash_table.DataTable(
                    data=PAIR_DF.to_dict('records'),
                    columns=[{'name': c, 'id': c} for c in PAIR_DF.columns],
                    sort_action='native', export_format='csv', **TABLE_STYLE,
                    style_data_conditional=[
                        {'if': {'filter_query': '{Sig. Bonferroni} = "sig.*"',
                                'column_id': 'Sig. Bonferroni'},
                         'color': C['green'], 'fontWeight': '700'},
                        {'if': {'filter_query': '{Pearson r} > 0.5', 'column_id': 'Pearson r'},
                         'color': C['green']},
                        {'if': {'filter_query': '{Pearson r} < -0.5', 'column_id': 'Pearson r'},
                         'color': C['red']},
                    ])], style=CARD),
        ])

    elif tab == 't2':
        return html.Div([
            make_methodology_note(
                'Cada punto es un país LATAM-20 ubicado por su GHS 2021 (eje X) vs su INFORM Prep '
                '2019 (eje Y, invertido de INFORM Risk). El tamaño de la burbuja codifica la tercera '
                'variable seleccionada, permitiendo ver tres índices a la vez. Colores por subregión.',
                accent='blue'),
            html.Div([make_section_title('GHS 2021 × INFORM Prep 2019 con tamaño por tercer índice',
                'Explora la relación bivariada y añade una tercera dimensión'),
                dcc.Dropdown(
                    [{'label': 'Tamaño: OxCGRT pico 2020', 'value': 'OxCGRT_Peak_2020'},
                     {'label': 'Tamaño: OxCGRT media 2020', 'value': 'OxCGRT_Mean_2020'},
                     {'label': 'Tamaño: Severity COVID máx', 'value': 'Severity_COVID_Max'},
                     {'label': 'Tamaño: SPAR 2024', 'value': 'SPAR_2024'}],
                    value='OxCGRT_Peak_2020', id='scatter-size',
                    clearable=False,
                    style={**DROPDOWN_STYLE, 'width': '320px', 'marginBottom': '12px'}),
                dcc.Graph(id='scatter-graph', config=GRAPH_CONFIG)], style=CARD),
        ])

    elif tab == 't3':
        return html.Div([html.Div([
            make_section_title('Perfil multi-índice por país',
                'Los 5 índices normalizados 0-100 por min-max sobre LATAM-20. La línea punteada '
                'muestra la media regional como referencia. País marcado como área sólida.'),
            dcc.Dropdown([{'label': c, 'value': c} for c in LATAM_20_SORTED],
                value='Mexico', id='radar-country', clearable=False,
                style={**DROPDOWN_STYLE, 'width': '260px', 'marginBottom': '12px'}),
            dcc.Graph(id='radar-graph', config=GRAPH_CONFIG)], style=CARD)])

    elif tab == 't4':
        return html.Div([html.Div([
            make_section_title('Radar comparativo (hasta 5 países)',
                'Compara perfiles multi-índice. Útil para identificar trade-offs entre preparación '
                'formal (GHS, SPAR) y respuesta real (OxCGRT) o impacto (Severity inv).'),
            dcc.Dropdown(
                [{'label': c, 'value': c} for c in LATAM_20_SORTED],
                value=['Mexico', 'Chile', 'Haiti', 'Uruguay'], multi=True,
                id='radar-multi',
                style={**DROPDOWN_STYLE, 'marginBottom': '12px'}),
            dcc.Graph(id='radar-multi-graph', config=GRAPH_CONFIG)], style=CARD)])

    elif tab == 't5':
        return html.Div([html.Div([
            make_section_title('Ranking por score compuesto de preparación',
                'media(GHS 2021, SPAR 2024, INFORM Prep 2024) · una sola métrica resumen que '
                'combina los tres índices de preparación (excluye desenlaces). Colores por subregión.'),
            dcc.Graph(figure=fig_ranking_compuesto(), config=GRAPH_CONFIG)], style=CARD)])

    elif tab == 't6':
        return html.Div([html.Div([
            make_section_title('Distribución de cada índice · stripplot',
                'Un punto por país, coloreado por subregión. Permite ver dispersión y outliers en '
                'cada índice normalizado. Línea ámbar = media.'),
            dcc.Graph(figure=fig_stripplot_por_indice(), config=GRAPH_CONFIG)], style=CARD)])

    elif tab == 't7':
        return html.Div([html.Div([
            make_section_title('Heatmap países × 5 índices',
                'Países ordenados por score compuesto descendente. Escala unificada 0-100 '
                'normalizada min-max. Verde = mejor relativo; rojo = peor relativo.'),
            dcc.Graph(figure=fig_heatmap_pais_indice(), config=GRAPH_CONFIG)], style=CARD)])

    elif tab == 't8':
        cols = ['País', 'Subregión', 'GHS_2019', 'GHS_2021',
                'SPAR_2019', 'SPAR_2024',
                'INFORM_Risk_2019', 'INFORM_Risk_2024',
                'INFORM_Prep_2024', 'OxCGRT_Mean_2020', 'OxCGRT_Peak_2020',
                'Severity_COVID_Max', 'Score_Compuesto']
        return html.Div([html.Div([
            make_section_title('Tabla maestra cross-index · 20 países × 5 índices',
                'Exportable a CSV · filtrable · ordenable. '
                'INFORM_Prep = (10−Risk)×10 (invertido para dirección consistente).'),
            dash_table.DataTable(
                data=MASTER[cols].to_dict('records'),
                columns=[{'name': c, 'id': c} for c in cols],
                filter_action='native', sort_action='native',
                page_size=24, export_format='csv', **TABLE_STYLE,
                style_data_conditional=[
                    {'if': {'filter_query': '{Score_Compuesto} > 65',
                            'column_id': 'Score_Compuesto'},
                     'color': C['green'], 'fontWeight': '700'},
                    {'if': {'filter_query': '{Score_Compuesto} < 40',
                            'column_id': 'Score_Compuesto'},
                     'color': C['red'], 'fontWeight': '700'},
                ])], style=CARD)])


# ══════════════════════════════════════════════════════════════════════
# 7. CALLBACKS
# ══════════════════════════════════════════════════════════════════════
@app.callback(Output('scatter-graph', 'figure'), Input('scatter-size', 'value'))
def cb_scatter(size_by): return fig_scatter_ghs_inform(size_by)

@app.callback(Output('radar-graph', 'figure'), Input('radar-country', 'value'))
def cb_radar(c): return fig_radar_pais(c)

@app.callback(Output('radar-multi-graph', 'figure'), Input('radar-multi', 'value'))
def cb_radar_multi(countries):
    if not countries: return go.Figure()
    return fig_radar_comparativo(countries)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8060))
    app.run(debug=False, host='0.0.0.0', port=port)

# Requerido por Dash Pages
layout = app.layout
