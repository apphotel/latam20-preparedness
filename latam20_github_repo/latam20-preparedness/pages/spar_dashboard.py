"""
SPAR Dashboard v4.0 — LATAM-20 · 2018–2024
══════════════════════════════════════════════════════════════════════
Metanálisis Índices Pandémicos · Fuente: WHO IHR State Party Annual Report (e-SPAR)

CAMBIOS v3.0 → v4.0:
  ✓ Tema claro CDC/Harvard unificado (theme.py)
  ✓ Muestra LATAM-20 consistente (antes header decía "24 PAÍSES")
  ✓ Rutas relativas portables (latam_common.py)
  ✓ Wilcoxon apareado robusto con tamaño de efecto (r de Rosenthal + Cliff's δ)
  ✓ Tendencias con corrección FDR Benjamini-Hochberg
  ✓ IC 95% bootstrap para medias · Fisher-z para correlaciones
  ✓ Exportación SVG vectorial

PERÍODOS METODOLÓGICOS:
  P1 (2018–2020): 13 capacidades — SPAR 1ª edición. Comparable entre sí.
  P2 (2021–2024): 15 capacidades — SPAR 2ª edición. Comparable entre sí.
  Overall comparable en toda la serie 2018–2024.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

from latam_common import (
    LATAM_20 as LATAM, LATAM_20_SORTED,
    YEARS_SPAR as YEARS, YEARS_SPAR_P1, YEARS_SPAR_P2,
    SHORT, sn, data_path, NOTA_MUESTRA,
)
from theme import (
    C, LAYOUT, CARD, TS, TSS, TABLE_STYLE, DROPDOWN_STYLE,
    RISK_COLORSCALE, GRAPH_CONFIG,
    rc, sc, gc, rgba,
    make_header, make_kpi, make_section_title, make_methodology_note,
)
from stats_utils import wilcoxon_paired, trend_analysis, mean_ci, correlation_ci

# ══════════════════════════════════════════════════════════════════════
# 1. CARGA
# ══════════════════════════════════════════════════════════════════════
spar     = pd.read_csv(data_path('spar_latam'))
spar_g   = pd.read_csv(data_path('spar_global'))
spar_ind = pd.read_csv(data_path('spar_indicators'))

# Filtro LATAM-20 (los archivos pueden venir con LAC-24)
spar     = spar[spar['Country'].isin(LATAM)].copy()
spar_ind = spar_ind[spar_ind['Country'].isin(LATAM)].copy()

CAP_P1 = [c for c in [f'C.{i}' for i in range(1, 14)] if c in spar.columns]
CAP_P2 = [c for c in [f'C.{i}' for i in range(1, 16)] if c in spar_ind.columns]
IND_P2 = [c for c in spar_ind.columns if c.startswith('C.') and c.count('.') == 2]

CAP_P1_NAMES = {
    'C.1': 'Legislación y financiación', 'C.2': 'Coordinación RSI',
    'C.3': 'Eventos zoonóticos', 'C.4': 'Inocuidad de alimentos',
    'C.5': 'Laboratorio', 'C.6': 'Vigilancia', 'C.7': 'Recursos humanos',
    'C.8': 'Marco de emergencias', 'C.9': 'Servicios de salud',
    'C.10': 'Comunicación de riesgos', 'C.11': 'Puntos de entrada',
    'C.12': 'Eventos químicos', 'C.13': 'Emergencias por radiación',
}
CAP_P2_NAMES = {
    'C.1': 'Instrumentos jurídicos RSI', 'C.2': 'Coordinación RSI',
    'C.3': 'Financiación', 'C.4': 'Laboratorio', 'C.5': 'Vigilancia',
    'C.6': 'Recursos humanos', 'C.7': 'Gestión emergencias',
    'C.8': 'Servicios de salud', 'C.9': 'Prevención infecciones (PCI)',
    'C.10': 'Comunicación de riesgos', 'C.11': 'Puntos de entrada',
    'C.12': 'Enfermedades zoonóticas', 'C.13': 'Inocuidad de alimentos',
    'C.14': 'Eventos químicos', 'C.15': 'Emergencias por radiación',
}
IND_NAMES = {
    'C.1.1':'Instrumentos políticos y normativos','C.1.2':'Igualdad de género en emergencias',
    'C.2.1':'Funciones del CNE para el RSI','C.2.2':'Mecanismos de coordinación multisectorial',
    'C.2.3':'Promoción del RSI','C.3.1':'Financiación para emergencias',
    'C.3.2':'Sistema de derivación de muestras','C.4.1':'Bioseguridad en laboratorios',
    'C.4.2':'Calidad de laboratorio','C.4.3':'Capacidad de pruebas de laboratorio',
    'C.4.4':'Red de diagnóstico eficaz','C.4.5':'Vigilancia de alerta temprana',
    'C.5.1':'Gestión de eventos (verificación)','C.5.2':'RRHH para el RSI',
    'C.6.1':'Aumento de fuerza laboral','C.6.2':'Planificación para emergencias',
    'C.7.1':'Gestión de respuesta a emergencias','C.7.2':'Cadena de suministro y logística',
    'C.7.3':'Gestión de casos','C.8.1':'Utilización de servicios de salud',
    'C.8.2':'Continuidad de servicios esenciales','C.8.3':'Programas de PCI',
    'C.9.1':'Vigilancia de IRAS','C.9.2':'Entorno seguro en establecimientos',
    'C.9.3':'Sistema CRPC para emergencias','C.10.1':'Comunicación de riesgos — planificación',
    'C.10.2':'Comunicación de riesgos — ejecución','C.10.3':'Participación de la comunidad',
    'C.11.1':'Capacidad básica en puntos de entrada','C.11.2':'Respuesta en puntos de entrada',
    'C.11.3':'Enfoque basado en riesgos para viajes','C.12.1':'Colaboración Una sola salud',
    'C.13.1':'Colaboración inocuidad de alimentos','C.14.1':'Recursos detección y alerta',
    'C.15.1':'Capacidad y recursos — radiación',
}

# ══════════════════════════════════════════════════════════════════════
# 2. ESTADÍSTICOS
# ══════════════════════════════════════════════════════════════════════
def desc(s):
    s = pd.to_numeric(s, errors='coerce').dropna()
    return dict(n=len(s), mean=round(s.mean(), 1), median=round(s.median(), 1),
                sd=round(s.std(), 1), q25=round(s.quantile(.25), 1),
                q75=round(s.quantile(.75), 1))

STATS = {}
for yr in YEARS:
    lat_vals = pd.to_numeric(spar[spar['Year']==yr]['SPAR_Overall'], errors='coerce').dropna()
    glb_vals = pd.to_numeric(spar_g[spar_g['Year']==yr]['SPAR_Overall'], errors='coerce').dropna()
    lat_ci = mean_ci(lat_vals.values)
    STATS[yr] = {
        'lat': {**desc(lat_vals),
                'ci_lo': lat_ci['ci95_boot'][0],
                'ci_hi': lat_ci['ci95_boot'][1]},
        'glob': desc(glb_vals),
    }

# Wilcoxon apareados robustos
def _pair_series(ya, yb):
    sa = spar[spar['Year']==ya].set_index('Country')['SPAR_Overall']
    sb = spar[spar['Year']==yb].set_index('Country')['SPAR_Overall']
    common = [c for c in LATAM if c in sa.index and c in sb.index]
    return ([float(sa[c]) if pd.notna(sa[c]) else np.nan for c in common],
            [float(sb[c]) if pd.notna(sb[c]) else np.nan for c in common],
            common)

_x19, _y19, _l19 = _pair_series(2020, 2019)
_x21, _y21, _l21 = _pair_series(2021, 2020)
_x24, _y24, _l24 = _pair_series(2024, 2021)

WX = {
    '19_20': wilcoxon_paired(_x19, _y19, labels=_l19),  # x=2020, y=2019
    '20_21': wilcoxon_paired(_x21, _y21, labels=_l21),
    '21_24': wilcoxon_paired(_x24, _y24, labels=_l24),
}

# Tendencias 2021-2024 con FDR
pivot_p2 = spar[spar['Year'].isin(YEARS_SPAR_P2)].pivot_table(
    index='Country', columns='Year', values='SPAR_Overall')
TRENDS_DF = trend_analysis(pivot_p2, min_points=3, adjust_method='fdr_bh')
N_SIG     = int((TRENDS_DF['p'] < 0.05).sum())
N_SIG_ADJ = int((TRENDS_DF['p_adj'] < 0.05).sum())

# Clustering 2024
si24 = spar_ind[spar_ind['Year']==2024][['Country'] + CAP_P2].dropna()
if len(si24) >= 3:
    X_sc = StandardScaler().fit_transform(si24[CAP_P2].values)
    km_labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_sc)
    cluster_means = {k: spar[(spar['Year']==2024) &
                             (spar['Country'].isin(si24['Country'].values[km_labels==k]))
                             ]['SPAR_Overall'].mean() for k in range(3)}
    sorted_clusters = sorted(cluster_means, key=cluster_means.get)
    CLUSTER_NAMES = {sorted_clusters[0]: 'Baja preparación',
                     sorted_clusters[1]: 'Preparación media',
                     sorted_clusters[2]: 'Alta preparación'}
    SPAR_CLUSTER_COLORS = {'Baja preparación': C['red'],
                           'Preparación media': C['amber'],
                           'Alta preparación': C['green']}
    COUNTRY_CLUSTER = {c: CLUSTER_NAMES[km_labels[i]]
                       for i, c in enumerate(si24['Country'].values)}
    pca2 = PCA(n_components=2)
    X_pca = pca2.fit_transform(X_sc)
    PCA_DF = pd.DataFrame({
        'País':    si24['Country'].values,
        'PC1':     X_pca[:, 0],
        'PC2':     X_pca[:, 1],
        'Cluster': [COUNTRY_CLUSTER.get(c, '') for c in si24['Country'].values],
        'Overall': [float(spar[(spar['Country']==c)&(spar['Year']==2024)]['SPAR_Overall'].values[0])
                    if len(spar[(spar['Country']==c)&(spar['Year']==2024)]) > 0 else 0
                    for c in si24['Country'].values],
    })
    PCA_VAR = pca2.explained_variance_ratio_
else:
    PCA_DF = pd.DataFrame()
    COUNTRY_CLUSTER = {}
    SPAR_CLUSTER_COLORS = {}
    PCA_VAR = [0, 0]

# Índice de riesgo sistémico (ADVERTENCIA: métrica exploratoria no validada)
RISK = []
for c in LATAM:
    d24  = spar[(spar['Country']==c) & (spar['Year']==2024)]['SPAR_Overall']
    dall = spar[(spar['Country']==c) & (spar['Year'].isin(YEARS_SPAR_P2))]['SPAR_Overall'].dropna()
    rc_  = spar_ind[(spar_ind['Country']==c) & (spar_ind['Year']==2024)]
    if len(d24) == 0 or len(dall) < 3 or len(rc_) == 0:
        continue
    score = float(d24.values[0]); vol = float(dall.std())
    v = rc_[CAP_P2].values.flatten().astype(float); v = v[~np.isnan(v)]
    cv = float(np.std(v) / np.mean(v) * 100) if len(v) > 5 else 0
    d_y = spar[(spar['Country']==c) & (spar['Year'].isin(YEARS_SPAR_P2))
               ].dropna(subset=['SPAR_Overall'])
    sl2 = stats.linregress(d_y['Year'], d_y['SPAR_Overall'])[0] if len(d_y) >= 3 else 0
    ri = round((100-score)*0.4 + vol*0.3 + cv*0.2 + max(0, -float(sl2))*0.1, 1)
    RISK.append({'País': c, 'Score_2024': score,
                 'Volatilidad': round(vol, 1), 'CV%': round(cv, 1),
                 'Tendencia': round(float(sl2), 2), 'RiskIdx': ri,
                 'Nivel': 'Alto' if ri > 30 else 'Medio' if ri > 18 else 'Bajo'})
RISK_DF = pd.DataFrame(RISK).sort_values('RiskIdx', ascending=False)

# ══════════════════════════════════════════════════════════════════════
# 3. FIGURAS
# ══════════════════════════════════════════════════════════════════════

def fig_serie_completa():
    lat_means  = [STATS[yr]['lat']['mean']  for yr in YEARS]
    lat_q25    = [STATS[yr]['lat']['q25']   for yr in YEARS]
    lat_q75    = [STATS[yr]['lat']['q75']   for yr in YEARS]
    lat_ci_lo  = [STATS[yr]['lat']['ci_lo'] for yr in YEARS]
    lat_ci_hi  = [STATS[yr]['lat']['ci_hi'] for yr in YEARS]
    glob_means = [STATS[yr]['glob']['mean'] for yr in YEARS]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=YEARS+YEARS[::-1], y=lat_q75+lat_q25[::-1],
        fill='toself', fillcolor=rgba(C['latam'], 0.07), line=dict(width=0),
        name='IQR LATAM', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=YEARS+YEARS[::-1], y=lat_ci_hi+lat_ci_lo[::-1],
        fill='toself', fillcolor=rgba(C['latam'], 0.18), line=dict(width=0),
        name='IC 95% media', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=YEARS, y=glob_means, mode='lines+markers',
        name='Media global', line=dict(color=C['global_c'], width=2, dash='dash'),
        marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=YEARS, y=lat_means, mode='lines+markers+text',
        name='Media LATAM', line=dict(color=C['latam'], width=3),
        marker=dict(size=8), text=[f'{v:.1f}' for v in lat_means],
        textposition='top center', textfont=dict(size=9, color=C['latam'])))
    fig.add_vline(x=2019.5, line_dash='dash', line_color=C['covid'], line_width=1.5,
                  annotation_text='COVID-19',
                  annotation_font=dict(size=9, color=C['covid']))
    fig.add_vline(x=2020.5, line_dash='dash', line_color=C['amber'], line_width=1.5,
                  annotation_text='Cambio instrumento (P1→P2)',
                  annotation_font=dict(size=9, color=C['amber']),
                  annotation_position='top right')
    fig.update_layout(**{**LAYOUT, 'height': 440},
        xaxis=dict(tickvals=YEARS, gridcolor=C['grid'], title='Año'),
        yaxis=dict(title='SPAR Overall (0–100)', range=[50, 90], gridcolor=C['grid']),
        legend=dict(orientation='h', y=-0.18, bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=55, r=20, t=40, b=70))
    return fig


def tabla_stats():
    rows = []
    for yr in YEARS:
        l = STATS[yr]['lat']; g = STATS[yr]['glob']
        rows.append({'Año': yr, 'n LATAM': l['n'],
                     'Media LATAM': l['mean'],
                     'IC 95%': f"[{l['ci_lo']}, {l['ci_hi']}]",
                     'Mediana': l['median'], 'SD': l['sd'],
                     'Media Global': g['mean'],
                     'Δ vs Global': round(l['mean'] - g['mean'], 1),
                     'Período': '1ª ed.' if yr <= 2020 else '2ª ed.'})
    return rows


def fig_pandemia_barras():
    rows = []
    for c in LATAM:
        v19 = spar[(spar['Country']==c) & (spar['Year']==2019)]['SPAR_Overall']
        v20 = spar[(spar['Country']==c) & (spar['Year']==2020)]['SPAR_Overall']
        if len(v19) == 0 or len(v20) == 0: continue
        v19f, v20f = float(v19.values[0]), float(v20.values[0])
        if pd.notna(v19f) and pd.notna(v20f):
            rows.append({'País': c, 'v19': v19f, 'v20': v20f,
                         'delta': round(v20f - v19f, 1)})
    df = pd.DataFrame(rows).sort_values('delta', ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=[sn(c) for c in df['País']], x=df['v19'],
        name='2019 (basal)', orientation='h',
        marker=dict(color=rgba(C['global_c'], 0.55), line_width=0)))
    fig.add_trace(go.Bar(y=[sn(c) for c in df['País']], x=df['v20'],
        name='2020 (pandemia)', orientation='h',
        marker=dict(color=C['covid'], line_width=0),
        customdata=df['delta'].values,
        hovertemplate='<b>%{y}</b><br>2020: %{x:.1f}<br>Δ: %{customdata:+.1f}<extra></extra>'))
    w = WX['19_20']
    title = (f'Wilcoxon apareado 2019→2020 (n={w["n"]}): '
             f'W={w["W"]:.0f}, p={w["p"]:.4f}, r={w["r_rosenthal"]}, '
             f'Cliff\'s δ={w["cliffs_delta"]} · {w["interpretation"]}')
    fig.update_layout(**{**LAYOUT, 'height': 580}, barmode='overlay',
        title=dict(text=title, font=dict(size=10, color=C['muted'])),
        xaxis=dict(range=[30, 105], gridcolor=C['grid'], title='SPAR Overall'),
        yaxis=dict(gridcolor=C['grid']),
        legend=dict(orientation='h', y=1.05, bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=100, r=50, t=75, b=40))
    return fig


def fig_pandemia_caps():
    COVID_CAPS = ['C.6','C.5','C.8','C.7','C.10','C.9','C.3','C.2','C.1']
    caps_avail = [c for c in COVID_CAPS if c in spar.columns]
    names = [CAP_P1_NAMES.get(c, c) for c in caps_avail]
    m19   = [spar[spar['Year']==2019][c].mean() for c in caps_avail]
    m20   = [spar[spar['Year']==2020][c].mean() for c in caps_avail]
    mg19  = [spar_g[spar_g['Year']==2019][c].mean() for c in caps_avail]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=m19, name='LATAM 2019 (basal)',
        marker=dict(color=rgba(C['global_c'], 0.65), line_width=0)))
    fig.add_trace(go.Bar(x=names, y=m20, name='LATAM 2020 (pandemia)',
        marker=dict(color=C['covid'], line_width=0)))
    fig.add_trace(go.Scatter(x=names, y=mg19, mode='markers',
        name='Media global 2019',
        marker=dict(symbol='diamond', color=C['amber'], size=11, line_width=0)))
    for nm, v19, v20 in zip(names, m19, m20):
        d = round(v20 - v19, 1)
        col = C['green'] if d > 0 else C['red']
        fig.add_annotation(x=nm, y=max(v19, v20)+3, text=f'{d:+.1f}',
                           showarrow=False, font=dict(size=9, color=col))
    fig.update_layout(**{**LAYOUT, 'height': 400}, barmode='group',
        yaxis=dict(title='Score promedio LATAM', range=[40, 100], gridcolor=C['grid']),
        xaxis=dict(tickangle=-30, gridcolor=C['grid']),
        legend=dict(orientation='h', y=-0.25, bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=55, r=20, t=40, b=90))
    return fig


def fig_paradoja_2019():
    rows = []
    for c in LATAM:
        v19 = spar[(spar['Country']==c) & (spar['Year']==2019)]['SPAR_Overall']
        v20 = spar[(spar['Country']==c) & (spar['Year']==2020)]['SPAR_Overall']
        if len(v19) == 0 or len(v20) == 0: continue
        v19f, v20f = float(v19.values[0]), float(v20.values[0])
        if pd.notna(v19f) and pd.notna(v20f):
            rows.append({'País': c, 'v19': v19f, 'delta': round(v20f - v19f, 1)})
    df = pd.DataFrame(rows)
    cr_p = correlation_ci(df['v19'], df['delta'], method='pearson')
    cr_s = correlation_ci(df['v19'], df['delta'], method='spearman')
    fig = go.Figure(go.Scatter(
        x=df['v19'], y=df['delta'], mode='markers+text',
        text=[sn(c) for c in df['País']], textposition='top center',
        textfont=dict(size=9, color=C['ink']),
        marker=dict(size=12, color=df['delta'],
                    colorscale=[[0, C['red']], [0.5, C['amber']], [1, C['green']]],
                    showscale=True, line=dict(color='white', width=1),
                    colorbar=dict(title=dict(text='Δ', font=dict(color=C['ink'])),
                                  tickfont=dict(size=9, color=C['ink']))),
        hovertemplate='<b>%{text}</b><br>Basal 2019: %{x}<br>Δ pandemia: %{y:+.1f}<extra></extra>'))
    fig.add_hline(y=0, line_color=C['muted'], line_width=1, line_dash='dot')
    title = (f'Score basal 2019 vs respuesta pandémica 2020 (n={cr_p["n"]}) · '
             f'Pearson r={cr_p["r"]} [IC95% {cr_p["ci"][0]}, {cr_p["ci"][1]}], p={cr_p["p"]:.3f} · '
             f'Spearman ρ={cr_s["r"]}, p={cr_s["p"]:.3f} · '
             f'Sin correlación → el autorreporte basal no predice la respuesta')
    fig.update_layout(**{**LAYOUT, 'height': 460},
        title=dict(text=title, font=dict(size=10, color=C['muted'])),
        xaxis=dict(title='SPAR Overall 2019 (preparación basal)', gridcolor=C['grid']),
        yaxis=dict(title='Δ 2019→2020 (respuesta pandémica)', gridcolor=C['grid']),
        margin=dict(l=60, r=80, t=75, b=50))
    return fig


def fig_heatmap_completo():
    pivot = spar.pivot_table(index='Country', columns='Year', values='SPAR_Overall')
    pivot = pivot.reindex(sorted(pivot.index, key=lambda c:
        -(pivot.loc[c, 2024] if 2024 in pivot.columns and pd.notna(pivot.loc[c, 2024]) else 0)))
    z = pivot.values
    text = np.where(pd.isna(z), 'N/D', np.round(z, 0).astype('object'))
    # Escala invertida: MAYOR score = MEJOR → verde
    fig = go.Figure(go.Heatmap(
        z=z, x=[str(yr) for yr in pivot.columns], y=[sn(c) for c in pivot.index],
        text=text, texttemplate='%{text}', textfont=dict(size=9, color=C['ink']),
        colorscale=[[0, C['red']], [0.4, C['orange']], [0.7, C['amber']], [1, C['green']]],
        zmin=30, zmax=100,
        hovertemplate='<b>%{y}</b> %{x}: %{z:.0f}<extra></extra>',
        colorbar=dict(title=dict(text='Score', font=dict(color=C['ink'])),
                      tickfont=dict(size=9, color=C['ink']))))
    fig.add_vline(x=2.5, line_dash='dash', line_color=C['amber'], line_width=1.5)
    fig.update_layout(**{**LAYOUT, 'height': 620},
        xaxis=dict(side='top', gridcolor=C['grid']),
        yaxis=dict(gridcolor=C['grid']),
        margin=dict(l=110, r=20, t=75, b=20),
        annotations=[
            dict(x=1, y=1.03, xref='paper', yref='paper',
                 text='← 1ª edición (13 cap.) | 2ª edición (15 cap.) →',
                 showarrow=False, font=dict(size=9, color=C['amber'])),
        ])
    return fig


def fig_tendencias_slope():
    df = TRENDS_DF.sort_values('slope', ascending=True)
    # Color por significancia FDR
    colors = [C['red'] if (row['p_adj'] < 0.05 and row['slope'] < 0)
              else C['green'] if (row['p_adj'] < 0.05 and row['slope'] > 0)
              else C['muted']
              for _, row in df.iterrows()]
    fig = go.Figure(go.Bar(
        y=[sn(c) for c in df['País']], x=df['slope'], orientation='h',
        marker=dict(color=colors, line_width=0),
        error_x=dict(type='data', array=df['ci95_slope'].values, visible=True,
                     color=rgba(C['ink'], 0.35), thickness=1.5),
        customdata=df[['r2', 'p', 'p_adj', 'sig_adj']].values,
        hovertemplate=('<b>%{y}</b><br>Slope: %{x:+.2f} pts/año<br>'
                       'R²=%{customdata[0]}<br>p=%{customdata[1]}<br>'
                       'p_adj=%{customdata[2]} %{customdata[3]}<extra></extra>')))
    fig.add_vline(x=0, line_color=C['muted'], line_width=1)
    w = WX['21_24']
    title = (f'Tendencias 2021–2024 · Wilcoxon emparejado: W={w["W"]:.0f}, p={w["p"]:.4f}, '
             f'r={w["r_rosenthal"]}, n={w["n"]} · '
             f'{N_SIG_ADJ}/{len(TRENDS_DF)} países sig. tras FDR ({N_SIG} sin ajustar)')
    fig.update_layout(**{**LAYOUT, 'height': 580},
        title=dict(text=title, font=dict(size=10, color=C['muted'])),
        xaxis=dict(title='Cambio anual (pts/año) · barras = IC 95% del slope',
                   gridcolor=C['grid']),
        yaxis=dict(gridcolor=C['grid']),
        margin=dict(l=100, r=40, t=75, b=50))
    return fig


def fig_tendencias_lineas():
    fig = go.Figure()
    for c in LATAM:
        d = spar[(spar['Country']==c) & (spar['Year'].isin(YEARS_SPAR_P2))
                 ].sort_values('Year').dropna(subset=['SPAR_Overall'])
        if len(d) < 2: continue
        tr = TRENDS_DF[TRENDS_DF['País']==c]
        if len(tr) > 0:
            slope_v = tr['slope'].values[0]
            p_adj_v = tr['p_adj'].values[0]
            col = (C['green'] if (p_adj_v < 0.05 and slope_v > 0)
                   else C['red'] if (p_adj_v < 0.05 and slope_v < 0)
                   else C['muted'])
        else:
            col = C['muted']
        fig.add_trace(go.Scatter(
            x=d['Year'], y=d['SPAR_Overall'], mode='lines+markers', name=sn(c),
            line=dict(color=col, width=1.5), marker=dict(size=4), opacity=0.75,
            hovertemplate=f'<b>{sn(c)}</b> %{{x}}: %{{y:.1f}}<extra></extra>'))
    means = [STATS[yr]['lat']['mean'] for yr in YEARS_SPAR_P2]
    fig.add_trace(go.Scatter(x=YEARS_SPAR_P2, y=means, mode='lines+markers',
        name='Media LATAM', line=dict(color=C['latam'], width=3),
        marker=dict(size=9)))
    glob = [STATS[yr]['glob']['mean'] for yr in YEARS_SPAR_P2]
    fig.add_trace(go.Scatter(x=YEARS_SPAR_P2, y=glob, mode='lines',
        name='Media global', line=dict(color=C['global_c'], width=2, dash='dash')))
    fig.update_layout(**{**LAYOUT, 'height': 440},
        xaxis=dict(tickvals=YEARS_SPAR_P2, gridcolor=C['grid'], title='Año'),
        yaxis=dict(title='SPAR Overall (2ª edición)', range=[20, 105], gridcolor=C['grid']),
        legend=dict(orientation='h', y=-0.2, bgcolor='rgba(0,0,0,0)', font=dict(size=9)),
        margin=dict(l=55, r=20, t=40, b=70))
    return fig


def fig_pca_scatter():
    if PCA_DF.empty:
        return go.Figure()
    fig = go.Figure()
    for cluster, col in SPAR_CLUSTER_COLORS.items():
        df = PCA_DF[PCA_DF['Cluster']==cluster]
        if len(df) == 0: continue
        fig.add_trace(go.Scatter(
            x=df['PC1'], y=df['PC2'], mode='markers+text', name=cluster,
            text=[sn(c) for c in df['País']], textposition='top center',
            textfont=dict(size=9, color=C['ink']),
            marker=dict(size=df['Overall']/5, color=col,
                        line=dict(color='white', width=1), opacity=0.85),
            hovertemplate=('<b>%{text}</b><br>PC1=%{x:.2f} PC2=%{y:.2f}<br>'
                           'Overall=%{marker.size:.0f}<extra></extra>')))
    fig.update_layout(**{**LAYOUT, 'height': 460},
        title=dict(text=f'PCA · PC1={PCA_VAR[0]:.1%} · PC2={PCA_VAR[1]:.1%} · Tamaño = Overall 2024',
                   font=dict(size=10, color=C['muted'])),
        xaxis=dict(title='PC1 — Capacidad general', gridcolor=C['grid']),
        yaxis=dict(title='PC2 — Contraste entre dimensiones', gridcolor=C['grid']),
        legend=dict(orientation='h', y=-0.14, bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=60, r=20, t=60, b=60))
    return fig


def fig_radar_clusters():
    if not SPAR_CLUSTER_COLORS:
        return go.Figure()
    fig = go.Figure()
    theta = list(CAP_P2_NAMES.values()) + [list(CAP_P2_NAMES.values())[0]]
    for cluster, col in SPAR_CLUSTER_COLORS.items():
        countries = [c for c, cl in COUNTRY_CLUSTER.items() if cl == cluster]
        vals = []
        for cap in CAP_P2:
            d = spar_ind[(spar_ind['Year']==2024) &
                         (spar_ind['Country'].isin(countries))][cap].mean()
            vals.append(round(float(d), 1) if pd.notna(d) else 0)
        vals = vals + [vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=theta, name=cluster,
            line=dict(color=col, width=2.5),
            fill='toself', fillcolor=rgba(col, 0.12)))
    fig.update_layout(**{**LAYOUT, 'height': 480},
        polar=dict(bgcolor=C['white'],
                   radialaxis=dict(visible=True, range=[0, 100],
                                   tickfont=dict(size=8, color=C['muted']),
                                   gridcolor=C['grid'], linecolor=C['border']),
                   angularaxis=dict(tickfont=dict(size=9, color=C['ink']),
                                    gridcolor=C['grid'], linecolor=C['border'])),
        legend=dict(orientation='h', y=-0.12, bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=60, r=60, t=40, b=70))
    return fig


def fig_bubble_riesgo():
    df = RISK_DF.copy()
    color_map = {'Alto': C['red'], 'Medio': C['amber'], 'Bajo': C['green']}
    fig = go.Figure()
    for nivel, col in color_map.items():
        sub = df[df['Nivel']==nivel]
        if len(sub) == 0: continue
        fig.add_trace(go.Scatter(
            x=sub['Score_2024'], y=sub['Volatilidad'],
            mode='markers+text', name=f'Riesgo {nivel}',
            text=[sn(c) for c in sub['País']], textposition='top center',
            textfont=dict(size=9, color=C['ink']),
            marker=dict(size=sub['CV%']/1.8 + 10, color=col,
                        line=dict(color='white', width=1), opacity=0.85),
            customdata=sub[['RiskIdx', 'CV%', 'Tendencia']].values,
            hovertemplate=('<b>%{text}</b><br>Score 2024: %{x}<br>'
                           'Volatilidad SD: %{y}<br>CV%: %{customdata[1]:.1f}%<br>'
                           'Tendencia: %{customdata[2]:+.2f}/año<br>'
                           '<b>RiskIdx: %{customdata[0]}</b><extra></extra>')))
    mean_sc = df['Score_2024'].mean()
    mean_vol = df['Volatilidad'].mean()
    fig.add_hline(y=mean_vol, line_dash='dot', line_color=C['muted'], line_width=0.8)
    fig.add_vline(x=mean_sc, line_dash='dot', line_color=C['muted'], line_width=0.8)
    for txt, x, y in [('Alto riesgo\nestructural', 35, 22),
                      ('Riesgo por\nvolatilidad', 80, 22),
                      ('Deterioro\nactivo', 35, 5),
                      ('Estable o\nmejorando', 80, 5)]:
        fig.add_annotation(x=x, y=y, text=txt, showarrow=False,
                           font=dict(size=9, color=C['light']), align='center')
    fig.update_layout(**{**LAYOUT, 'height': 480},
        title=dict(text='Índice de Riesgo Sistémico = f(score, volatilidad, consistencia, tendencia) · '
                        'Tamaño = CV% capacidades',
                   font=dict(size=10, color=C['muted'])),
        xaxis=dict(title='SPAR Overall 2024', range=[25, 100], gridcolor=C['grid']),
        yaxis=dict(title='Volatilidad (SD 2021–2024)', range=[0, 32], gridcolor=C['grid']),
        legend=dict(orientation='h', y=-0.14, bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=60, r=20, t=60, b=60))
    return fig


def fig_riesgo_barras():
    df = RISK_DF.sort_values('RiskIdx', ascending=True)
    colors = [{'Alto': C['red'], 'Medio': C['amber'], 'Bajo': C['green']}[n]
              for n in df['Nivel']]
    fig = go.Figure(go.Bar(
        y=[sn(c) for c in df['País']], x=df['RiskIdx'], orientation='h',
        marker=dict(color=colors, line_width=0),
        customdata=df[['Score_2024','Volatilidad','CV%','Tendencia']].values,
        hovertemplate=('<b>%{y}</b><br>RiskIdx: %{x}<br>'
                       'Score 2024: %{customdata[0]}<br>'
                       'Volatilidad: %{customdata[1]}<br>'
                       'CV%: %{customdata[2]:.1f}%<br>'
                       'Tendencia: %{customdata[3]:+.2f}/año<extra></extra>')))
    fig.update_layout(**{**LAYOUT, 'height': 560},
        xaxis=dict(title='Índice de Riesgo Sistémico (mayor = más vulnerable)',
                   gridcolor=C['grid']),
        yaxis=dict(gridcolor=C['grid']),
        margin=dict(l=110, r=40, t=30, b=40))
    return fig


def master_order():
    ov = spar[spar['Year']==2024].set_index('Country')['SPAR_Overall']
    return sorted(LATAM, key=lambda c: -float(ov.loc[c])
                  if c in ov.index and pd.notna(ov.loc[c]) else 0)


def fig_caps_heatmap_paises(yr=2024):
    countries = master_order()
    z = np.array([[float(spar_ind[(spar_ind['Country']==c) &
                                  (spar_ind['Year']==yr)][cap].values[0])
                   if len(spar_ind[(spar_ind['Country']==c) & (spar_ind['Year']==yr)]) > 0
                   and cap in spar_ind.columns
                   and pd.notna(spar_ind[(spar_ind['Country']==c) & (spar_ind['Year']==yr)][cap].values[0])
                   else np.nan for cap in CAP_P2] for c in countries])
    fig = go.Figure(go.Heatmap(
        z=z, x=[CAP_P2_NAMES[c] for c in CAP_P2], y=[sn(c) for c in countries],
        text=np.where(np.isnan(z), 'N/D', np.round(z, 0).astype('object')),
        texttemplate='%{text}', textfont=dict(size=9, color=C['ink']),
        colorscale=[[0, C['red']], [0.5, C['amber']], [1, C['green']]],
        zmin=0, zmax=100,
        hovertemplate='<b>%{y}</b><br>%{x}: %{z:.0f}<extra></extra>',
        colorbar=dict(title=dict(text='Score', font=dict(color=C['ink'])),
                      tickfont=dict(size=9, color=C['ink']))))
    fig.update_layout(**{**LAYOUT, 'height': 620},
        xaxis=dict(side='top', tickangle=-35, tickfont=dict(size=9), gridcolor=C['grid']),
        yaxis=dict(gridcolor=C['grid']),
        margin=dict(l=110, r=20, t=150, b=20))
    return fig


def fig_caps_delta_latam():
    m21 = [spar_ind[spar_ind['Year']==2021][c].mean() for c in CAP_P2]
    m24 = [spar_ind[spar_ind['Year']==2024][c].mean() for c in CAP_P2]
    mg24 = [spar_g[spar_g['Year']==2024][c].mean() if c in spar_g.columns else np.nan
            for c in CAP_P2]
    deltas = [round(b - a, 1) for a, b in zip(m21, m24)]
    vs_g = [round(float(l) - float(g), 1) if not np.isnan(g) else 0
            for l, g in zip(m24, mg24)]
    names = [CAP_P2_NAMES[c] for c in CAP_P2]
    df_c = pd.DataFrame({'name': names, 'delta21_24': deltas, 'vs_global': vs_g,
                         'm21': m21, 'm24': m24}).sort_values('delta21_24')
    fig = make_subplots(1, 2, subplot_titles=['Δ 2021→2024 (LATAM)',
                                               'LATAM 2024 vs Global'])
    # Colores SPAR: mayor Δ = mejor → verde
    def _spar_delta_col(v):
        if v > 5: return C['green']
        if v > 0: return '#22C55E'
        if v > -5: return C['amber']
        return C['red']
    fig.add_trace(go.Bar(y=df_c['name'], x=df_c['delta21_24'], orientation='h',
        marker=dict(color=[_spar_delta_col(v) for v in df_c['delta21_24']], line_width=0),
        customdata=df_c[['m21', 'm24']].values,
        hovertemplate='%{y}<br>2021→2024: %{x:+.1f}<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Bar(y=df_c['name'], x=df_c['vs_global'], orientation='h',
        marker=dict(color=[_spar_delta_col(v) for v in df_c['vs_global']], line_width=0),
        hovertemplate='%{y}<br>vs Global: %{x:+.1f}<extra></extra>'), row=1, col=2)
    fig.add_vline(x=0, line_color=C['muted'], line_width=1, row=1, col=1)
    fig.add_vline(x=0, line_color=C['muted'], line_width=1, row=1, col=2)
    fig.update_annotations(font=dict(color=C['ink'], size=11))
    fig.update_layout(**{**LAYOUT, 'height': 520}, showlegend=False,
        margin=dict(l=220, r=20, t=60, b=30))
    fig.update_xaxes(gridcolor=C['grid'])
    fig.update_yaxes(gridcolor=C['grid'])
    return fig


def fig_perfil(country='Mexico'):
    d_all = spar[spar['Country']==country].sort_values('Year').dropna(subset=['SPAR_Overall'])
    d_g_all = spar_g.groupby('Year')['SPAR_Overall'].mean().reset_index()
    fig = make_subplots(1, 2, specs=[[{'type': 'polar'}, {'type': 'xy'}]],
        subplot_titles=['Perfil capacidades 2021 vs 2024', 'Serie 2018–2024'])
    cap_cols = [c for c in CAP_P2 if c in spar_ind.columns]
    theta = [CAP_P2_NAMES[c] for c in cap_cols] + [CAP_P2_NAMES[cap_cols[0]]]
    for yr, col, dash_, alpha in [(2024, C['latam'], 'solid', 0.9),
                                   (2021, C['global_c'], 'dot', 0.5)]:
        row = spar_ind[(spar_ind['Country']==country) & (spar_ind['Year']==yr)]
        if len(row) == 0: continue
        vals = [float(row[c].values[0]) if pd.notna(row[c].values[0]) else 0 for c in cap_cols]
        vals = vals + [vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=theta, name=str(yr),
            line=dict(color=col, width=2.5 if yr==2024 else 1.5, dash=dash_),
            fill='toself', fillcolor=rgba(col, 0.15 if yr==2024 else 0.06)),
            row=1, col=1)
    fig.add_trace(go.Scatter(
        x=d_all['Year'], y=d_all['SPAR_Overall'],
        mode='lines+markers+text', name=country,
        text=[str(int(v)) for v in d_all['SPAR_Overall']],
        textposition='top center', textfont=dict(size=9, color=C['latam']),
        line=dict(color=C['latam'], width=2.5), marker=dict(size=7),
        showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=d_g_all['Year'], y=d_g_all['SPAR_Overall'],
        mode='lines', name='Media global',
        line=dict(color=C['global_c'], width=1.5, dash='dash')), row=1, col=2)
    lat_means = spar.groupby('Year')['SPAR_Overall'].mean().reset_index()
    fig.add_trace(go.Scatter(
        x=lat_means['Year'], y=lat_means['SPAR_Overall'],
        mode='lines', name='Media LATAM',
        line=dict(color=C['amber'], width=1.5, dash='dot')), row=1, col=2)
    fig.add_shape(type='line', x0=2020.5, x1=2020.5, y0=0, y1=1, yref='paper',
                  line=dict(dash='dash', color=C['amber'], width=1), row=1, col=2)
    cl = COUNTRY_CLUSTER.get(country, '—')
    fig.add_annotation(x=0.99, y=0.02, xref='paper', yref='paper',
                       text=f'Cluster 2024: {cl}', showarrow=False,
                       font=dict(size=9,
                                 color=SPAR_CLUSTER_COLORS.get(cl, C['muted'])),
                       align='right')
    ri_row = RISK_DF[RISK_DF['País']==country]
    if len(ri_row) > 0:
        ri = ri_row['RiskIdx'].values[0]; nv = ri_row['Nivel'].values[0]
        fig.add_annotation(x=0.99, y=0.08, xref='paper', yref='paper',
                           text=f'RiskIdx: {ri} ({nv})', showarrow=False,
                           font=dict(size=9,
                                     color={'Alto': C['red'], 'Medio': C['amber'],
                                            'Bajo': C['green']}.get(nv, C['muted'])),
                           align='right')
    fig.update_annotations(font=dict(color=C['ink'], size=11))
    fig.update_layout(**{**LAYOUT, 'height': 500},
        polar=dict(bgcolor=C['white'],
                   radialaxis=dict(visible=True, range=[0, 100],
                                   tickfont=dict(size=8, color=C['muted']),
                                   gridcolor=C['grid'], linecolor=C['border']),
                   angularaxis=dict(tickfont=dict(size=8, color=C['ink']),
                                    gridcolor=C['grid'], linecolor=C['border'])),
        legend=dict(orientation='h', y=-0.14, bgcolor='rgba(0,0,0,0)', font=dict(size=9)),
        margin=dict(l=20, r=20, t=55, b=60))
    fig.update_xaxes(tickvals=YEARS, gridcolor=C['grid'], row=1, col=2)
    fig.update_yaxes(range=[20, 105], gridcolor=C['grid'], title='SPAR Overall', row=1, col=2)
    return fig


def fig_indicadores_pais(country='Mexico', yr=2024):
    row = spar_ind[(spar_ind['Country']==country) & (spar_ind['Year']==yr)]
    row21 = spar_ind[(spar_ind['Country']==country) & (spar_ind['Year']==2021)]
    if len(row) == 0:
        return go.Figure()
    inds = [i for i in IND_P2 if i in row.columns]
    rows = []
    for ind in inds:
        v = float(row[ind].values[0]) if pd.notna(row[ind].values[0]) else 0
        v21 = float(row21[ind].values[0]) if len(row21) > 0 and pd.notna(row21[ind].values[0]) else 0
        gm = round(float(spar_ind[spar_ind['Year']==yr][ind].mean()), 1)
        rows.append({'name': IND_NAMES.get(ind, ind), 'val': v, 'val21': v21,
                     'delta': round(v - v21, 1), 'global': gm,
                     'cap': CAP_P2_NAMES.get('.'.join(ind.split('.')[:2]), '')})
    df_i = pd.DataFrame(rows).sort_values('val', ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df_i['name'], x=df_i['val21'], name='2021',
        orientation='h', marker=dict(color=rgba(C['global_c'], 0.5), line_width=0)))
    fig.add_trace(go.Bar(y=df_i['name'], x=df_i['val'], name=str(yr),
        orientation='h', marker=dict(color=C['latam'], line_width=0),
        customdata=df_i[['delta', 'global', 'cap']].values,
        hovertemplate=(f'<b>%{{y}}</b><br>{yr}: %{{x:.0f}}<br>'
                       f'Δ vs 2021: %{{customdata[0]:+.0f}}<br>'
                       f'Media global: %{{customdata[1]}}<br>'
                       f'%{{customdata[2]}}<extra></extra>')))
    fig.add_trace(go.Scatter(x=df_i['global'], y=df_i['name'], mode='markers',
        name='Media global',
        marker=dict(symbol='line-ns', color=C['amber'], size=10, line_width=2)))
    fig.update_layout(**{**LAYOUT, 'height': 900}, barmode='overlay',
        title=dict(text=f'{country} — 35 indicadores SPAR 2ª edición',
                   font=dict(size=11, color=C['muted'])),
        xaxis=dict(range=[0, 105], gridcolor=C['grid']),
        yaxis=dict(tickfont=dict(size=9), gridcolor=C['grid']),
        legend=dict(orientation='h', y=1.02, bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=310, r=20, t=60, b=30))
    return fig


# ══════════════════════════════════════════════════════════════════════
# 4. APP
# ══════════════════════════════════════════════════════════════════════
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True, title='SPAR v4.0')
dash.register_page(__name__, path='/spar', name='SPAR/IHR', order=4)

server = app.server

app.layout = dbc.Container([
    make_header(
        'SPAR INDEX',
        f'IHR STATES PARTIES SELF-ASSESSMENT · LATAM-20 · 2018–2024 · WHO e-SPAR'),
    make_methodology_note(
        'Período 1 (2018–2020): 13 capacidades (SPAR 1ª ed.) — comparable entre sí. '
        'Período 2 (2021–2024): 15 capacidades (SPAR 2ª ed.) — comparable entre sí. '
        'Overall = único indicador cross-período. '
        'Tests apareados con tamaño de efecto (r de Rosenthal, Cliff\'s δ). '
        'Tendencias ajustadas por FDR (Benjamini-Hochberg). '
        f'{NOTA_MUESTRA}',
        accent='amber'),
    dbc.Row([
        make_kpi(f'{STATS[2024]["lat"]["mean"]:.1f}', 'MEDIA LATAM 2024',
                 f'IC95% [{STATS[2024]["lat"]["ci_lo"]}, {STATS[2024]["lat"]["ci_hi"]}]',
                 C['latam']),
        make_kpi(f'{STATS[2019]["lat"]["mean"]:.1f}', 'MEDIA LATAM 2019',
                 'Basal pre-pandemia', C['blue']),
        make_kpi(f'{STATS[2024]["glob"]["mean"]:.1f}', 'MEDIA GLOBAL 2024',
                 f'Δ LATAM: {round(STATS[2024]["lat"]["mean"]-STATS[2024]["glob"]["mean"], 1):+.1f}',
                 C['global_c']),
        make_kpi(f'p={WX["19_20"]["p"]:.3f}', 'WILCOXON 19→20',
                 f'r={WX["19_20"]["r_rosenthal"]} · {WX["19_20"]["interpretation"][:22]}',
                 C['covid']),
        make_kpi(f'p={WX["21_24"]["p"]:.3f}', 'WILCOXON 21→24',
                 f'r={WX["21_24"]["r_rosenthal"]} · caída post-pandémica', C['red']),
        make_kpi(f'{N_SIG_ADJ}/{len(TRENDS_DF)}', 'TEND. SIG. (FDR)',
                 f'vs {N_SIG} sin ajustar', C['amber']),
    ], style={'marginBottom': '10px'}),
    dcc.Tabs(id='tabs', value='t1', children=[
        dcc.Tab(label='📈 Serie completa',   value='t1', style=TS, selected_style=TSS),
        dcc.Tab(label='🦠 Análisis pandémico',value='t2', style=TS, selected_style=TSS),
        dcc.Tab(label='Heatmap histórico',   value='t3', style=TS, selected_style=TSS),
        dcc.Tab(label='Tendencias (FDR)',    value='t4', style=TS, selected_style=TSS),
        dcc.Tab(label='🧩 Tipología',         value='t5', style=TS, selected_style=TSS),
        dcc.Tab(label='Riesgo sistémico',    value='t6', style=TS, selected_style=TSS),
        dcc.Tab(label='Capacidades P2',      value='t7', style=TS, selected_style=TSS),
        dcc.Tab(label='Perfil de país',      value='t8', style=TS, selected_style=TSS),
        dcc.Tab(label='Indicadores N2',      value='t9', style=TS, selected_style=TSS),
        dcc.Tab(label='Tabla maestra',       value='t10',style=TS, selected_style=TSS),
    ], style={'borderBottom': f'1px solid {C["border"]}', 'marginBottom': '14px'}),
    html.Div(id='content')
], fluid=True, style={'backgroundColor': C['bg'], 'minHeight': '100vh',
                      'padding': '0 20px 40px'})


@app.callback(Output('content', 'children'), Input('tabs', 'value'))
def render(tab):
    if tab == 't1':
        return html.Div([
            html.Div([make_section_title(
                'Serie completa 2018–2024: LATAM vs Global',
                'IC 95% bootstrap (banda más fuerte) · IQR (banda más tenue) · línea ámbar = cambio de instrumento 2020→2021'),
                dcc.Graph(figure=fig_serie_completa(), config=GRAPH_CONFIG)], style=CARD),
            html.Div([make_section_title('Estadísticas descriptivas por año'),
                dash_table.DataTable(
                    data=tabla_stats(),
                    columns=[{'name': c, 'id': c}
                             for c in ['Año','n LATAM','Media LATAM','IC 95%','Mediana',
                                       'SD','Media Global','Δ vs Global','Período']],
                    **TABLE_STYLE,
                    style_data_conditional=[
                        {'if':{'filter_query':'{Período} = "2ª ed."'},
                         'borderLeft': f'3px solid {C["latam"]}'},
                        {'if':{'filter_query':'{Δ vs Global} < 0','column_id':'Δ vs Global'},
                         'color': C['red'], 'fontWeight':'700'},
                        {'if':{'filter_query':'{Δ vs Global} > 0','column_id':'Δ vs Global'},
                         'color': C['green'], 'fontWeight':'700'},
                    ])], style=CARD),
        ])

    elif tab == 't2':
        return html.Div([
            make_methodology_note(
                '2019 = basal pre-pandemia · 2020 = respuesta pandémica. Ambos usan '
                'la misma 1ª edición SPAR (13 capacidades) → comparación directa válida. '
                f'Wilcoxon apareado n={WX["19_20"]["n"]}: {WX["19_20"]["interpretation"]}. '
                'Reportamos Cliff\'s δ (tamaño de efecto) e IC bootstrap de la mediana.',
                accent='covid'),
            html.Div([make_section_title('Cambio Overall 2019→2020: respuesta pandémica por país',
                'Azul = 2019 basal · morado = 2020 pandemia · Δ anotado al final de cada par'),
                dcc.Graph(figure=fig_pandemia_barras(), config=GRAPH_CONFIG)], style=CARD),
            html.Div([make_section_title('Capacidades críticas COVID: LATAM 2019 vs 2020 vs Global',
                'Las 9 capacidades más relevantes · diamante ámbar = referencia global'),
                dcc.Graph(figure=fig_pandemia_caps(), config=GRAPH_CONFIG)], style=CARD),
            html.Div([make_section_title('La paradoja del autorreporte: score basal no predice la respuesta',
                'Si SPAR midiera preparación real, un score alto en 2019 debería correlacionar con mantenimiento o mejora en 2020'),
                dcc.Graph(figure=fig_paradoja_2019(), config=GRAPH_CONFIG)], style=CARD),
        ])

    elif tab == 't3':
        return html.Div([html.Div([
            make_section_title('Heatmap países × años: serie completa 2018–2024',
                'Ordenado por score 2024 · línea ámbar = cambio de edición'),
            dcc.Graph(figure=fig_heatmap_completo(), config=GRAPH_CONFIG)], style=CARD)])

    elif tab == 't4':
        return html.Div([
            make_methodology_note(
                'Clasificación de tendencias basada en significancia AJUSTADA por FDR '
                '(Benjamini-Hochberg). Esto controla falsos descubrimientos esperados al 5%.',
                accent='blue'),
            html.Div([make_section_title('Tendencia lineal por país 2021–2024',
                f'{N_SIG_ADJ} países con tendencia sig. tras FDR · barras = IC 95% del slope'),
                dcc.Graph(figure=fig_tendencias_slope(), config=GRAPH_CONFIG)], style=CARD),
            html.Div([make_section_title('Trayectorias individuales 2021–2024',
                'Verde = mejora significativa · Rojo = deterioro significativo · Gris = no significativo'),
                dcc.Graph(figure=fig_tendencias_lineas(), config=GRAPH_CONFIG)], style=CARD),
            html.Div([make_section_title('Tabla de tendencias'),
                dash_table.DataTable(
                    data=TRENDS_DF.to_dict('records'),
                    columns=[{'name': c, 'id': c}
                             for c in ['País','n_años','slope','ci95_slope','r2',
                                       'p','p_adj','sig_adj','Dirección']],
                    sort_action='native', **TABLE_STYLE,
                    style_data_conditional=[
                        {'if':{'filter_query':'{Dirección} contains "Aumento"','column_id':'Dirección'},
                         'color': C['green']},
                        {'if':{'filter_query':'{Dirección} contains "Descenso"','column_id':'Dirección'},
                         'color': C['red']},
                        {'if':{'filter_query':'{sig_adj} = "*"'},'fontWeight':'700'}
                    ])], style=CARD),
        ])

    elif tab == 't5':
        return html.Div([
            html.Div([make_section_title('k-means + PCA sobre capacidades 2024',
                f'3 clusters · PC1={PCA_VAR[0]:.1%} · PC2={PCA_VAR[1]:.1%}'),
                dcc.Graph(figure=fig_pca_scatter(), config=GRAPH_CONFIG)], style=CARD),
            html.Div([make_section_title('Radar de clusters: perfil promedio por cluster'),
                dcc.Graph(figure=fig_radar_clusters(), config=GRAPH_CONFIG)], style=CARD),
            html.Div([make_section_title('Composición de clusters'),
                html.Div([html.Div([
                    html.Div(cl, style={'fontSize':'12px','fontWeight':'600',
                                         'color': SPAR_CLUSTER_COLORS.get(cl, C['muted']),
                                         'marginBottom':'4px'}),
                    html.Div(', '.join([c for c, cluster in COUNTRY_CLUSTER.items() if cluster == cl]),
                             style={'fontSize':'11px','color': C['text']})
                ], style={**CARD,
                          'borderLeft': f'3px solid {SPAR_CLUSTER_COLORS.get(cl, C["muted"])}',
                          'borderRadius':'0 8px 8px 0','marginBottom':'8px'})
                    for cl in ['Alta preparación','Preparación media','Baja preparación']])],
                style=CARD),
        ])

    elif tab == 't6':
        return html.Div([
            make_methodology_note(
                'Índice de Riesgo Sistémico = 0.4·(100−Score) + 0.3·Volatilidad + '
                '0.2·CV% + 0.1·max(0,−tendencia). Métrica exploratoria no validada '
                'contra desenlaces externos. Interpretar como sistema de alerta temprana.',
                accent='amber'),
            html.Div([make_section_title('Bubble chart: Score vs Volatilidad',
                'Tamaño = CV% de las 15 capacidades · cuadrantes = tipos de riesgo'),
                dcc.Graph(figure=fig_bubble_riesgo(), config=GRAPH_CONFIG)], style=CARD),
            html.Div([make_section_title('Ranking por Índice de Riesgo Sistémico',
                'Mayor = más vulnerable'),
                dcc.Graph(figure=fig_riesgo_barras(), config=GRAPH_CONFIG)], style=CARD),
            html.Div([make_section_title('Tabla de riesgo sistémico'),
                dash_table.DataTable(
                    data=RISK_DF.to_dict('records'),
                    columns=[{'name': c, 'id': c}
                             for c in ['País','Score_2024','Volatilidad','CV%','Tendencia','RiskIdx','Nivel']],
                    sort_action='native', export_format='csv', **TABLE_STYLE,
                    style_data_conditional=[
                        {'if':{'filter_query':'{Nivel} = "Alto"','column_id':'Nivel'},
                         'color': C['red'],'fontWeight':'700'},
                        {'if':{'filter_query':'{Nivel} = "Medio"','column_id':'Nivel'},
                         'color': C['amber'],'fontWeight':'700'},
                        {'if':{'filter_query':'{Nivel} = "Bajo"','column_id':'Nivel'},
                         'color': C['green'],'fontWeight':'700'},
                    ])], style=CARD),
        ])

    elif tab == 't7':
        return html.Div([
            html.Div([make_section_title('Scores por capacidad y país — 2ª edición',
                'Ordenado por Overall 2024 descendente'),
                dbc.Row([dbc.Col([dcc.Dropdown(
                    [{'label':str(y),'value':y} for y in YEARS_SPAR_P2],
                    value=2024, id='cap-yr', clearable=False, style=DROPDOWN_STYLE)
                ], md=3)], style={'marginBottom':'10px'}),
                dcc.Graph(id='cap-heatmap', config=GRAPH_CONFIG)], style=CARD),
            html.Div([make_section_title('Cambio LATAM 2021→2024 vs Brecha con Global'),
                dcc.Graph(figure=fig_caps_delta_latam(), config=GRAPH_CONFIG)], style=CARD),
        ])

    elif tab == 't8':
        return html.Div([html.Div([
            make_section_title('Perfil de país: radar 2024 vs 2021 + serie histórica',
                'Cluster y Riesgo Sistémico anotados'),
            dcc.Dropdown([{'label': c, 'value': c} for c in LATAM_20_SORTED],
                value='Mexico', id='pais-sel', clearable=False,
                style={**DROPDOWN_STYLE,'marginBottom':'10px','width':'260px'}),
            dcc.Graph(id='perfil-graph', config=GRAPH_CONFIG)], style=CARD)])

    elif tab == 't9':
        return html.Div([html.Div([
            make_section_title('35 indicadores por país — 2ª edición (2021–2024)',
                '2021 (gris) vs año seleccionado (azul) · línea ámbar = media global'),
            dbc.Row([
                dbc.Col([dcc.Dropdown([{'label':c,'value':c} for c in LATAM_20_SORTED],
                    value='Mexico', id='ind-pais', clearable=False, style=DROPDOWN_STYLE)], md=4),
                dbc.Col([dcc.Dropdown([{'label':str(y),'value':y} for y in YEARS_SPAR_P2],
                    value=2024, id='ind-yr', clearable=False, style=DROPDOWN_STYLE)], md=3),
            ], style={'marginBottom': '12px'}),
            dcc.Graph(id='ind-graph', config=GRAPH_CONFIG)], style=CARD)])

    elif tab == 't10':
        pivot = spar.pivot_table(index='Country', columns='Year', values='SPAR_Overall').reset_index()
        pivot.columns = [str(c) for c in pivot.columns]
        ri_map = {r['País']: r['RiskIdx'] for _, r in RISK_DF.iterrows()}
        pivot['RiskIdx'] = [ri_map.get(c, '—') for c in pivot['Country']]
        pivot['Cluster'] = [COUNTRY_CLUSTER.get(c, '—') for c in pivot['Country']]
        yr_cols = [str(yr) for yr in YEARS]
        cols_show = ['Country'] + yr_cols + ['RiskIdx', 'Cluster']
        return html.Div([html.Div([
            make_section_title('Tabla maestra — serie completa + RiskIdx + Cluster',
                'Exportable · filtrable · ordenable'),
            dash_table.DataTable(
                data=pivot[cols_show].to_dict('records'),
                columns=[{'name': c, 'id': c} for c in cols_show],
                filter_action='native', sort_action='native',
                page_size=24, export_format='csv', **TABLE_STYLE,
                style_data_conditional=[
                    {'if':{'filter_query':'{Cluster} = "Alta preparación"','column_id':'Cluster'},
                     'color': C['green']},
                    {'if':{'filter_query':'{Cluster} = "Preparación media"','column_id':'Cluster'},
                     'color': C['amber']},
                    {'if':{'filter_query':'{Cluster} = "Baja preparación"','column_id':'Cluster'},
                     'color': C['red']},
                ])], style=CARD)])


# ══════════════════════════════════════════════════════════════════════
# 5. CALLBACKS
# ══════════════════════════════════════════════════════════════════════
@app.callback(Output('perfil-graph', 'figure'), Input('pais-sel', 'value'))
def cb_perfil(c): return fig_perfil(c)

@app.callback(Output('ind-graph', 'figure'),
              [Input('ind-pais', 'value'), Input('ind-yr', 'value')])
def cb_ind(c, yr): return fig_indicadores_pais(c, yr)

@app.callback(Output('cap-heatmap', 'figure'), Input('cap-yr', 'value'))
def cb_cap_heat(yr): return fig_caps_heatmap_paises(yr)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8051))
    app.run(debug=False, host='0.0.0.0', port=port)

# Requerido por Dash Pages
layout = app.layout
