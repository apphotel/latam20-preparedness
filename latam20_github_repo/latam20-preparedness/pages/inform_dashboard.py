"""
INFORM Risk Dashboard v4.0 — LATAM-20 · 2017–2025
══════════════════════════════════════════════════════════════════════
Metanálisis Índices Pandémicos · Fuente: INFORM2026_TREND v7.2 (JRC/UE)

CAMBIOS v3.0 → v4.0:
  ✓ Tema claro CDC/Harvard unificado (theme.py)
  ✓ Muestra LATAM-20 consistente (antes mezclaba /20 y /24)
  ✓ Rutas relativas portables (latam_common.py)
  ✓ Stats corregidos: Wilcoxon apareado con dropna conjunto,
    tamaño de efecto (r de Rosenthal + Cliff's delta), IC bootstrap,
    corrección FDR (Benjamini-Hochberg) en tendencias (stats_utils.py)
  ✓ Pestañas reorganizadas en 4 secciones temáticas
  ✓ Exportación SVG vectorial activa

MUESTRA — LATAM-20 (ver latam_common.py):
  Sudamérica (10): Argentina, Bolivia, Brasil, Chile, Colombia, Ecuador,
                   Paraguay, Perú, Uruguay, Venezuela
  Centroamérica (6): Costa Rica, El Salvador, Guatemala, Honduras, Nicaragua, Panamá
  Norteamérica (1): México
  Caribe (3): Cuba, Haití, República Dominicana
  Excluidos de LAC-24: Jamaica, Trinidad & Tobago, Guyana, Surinam.
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

# Módulos unificados del metanálisis
from latam_common import (
    LATAM_ISO3, LATAM_20 as LATAM, LATAM_20_SORTED, YEARS_INFORM as YEARS,
    SHORT, sn, data_path, NOTA_MUESTRA,
)
from theme import (
    C, LAYOUT, CARD, TS, TSS, TABLE_STYLE, DROPDOWN_STYLE,
    RISK_COLORSCALE, DIVERGING_COLORSCALE, DIM_COLORS, CLUSTER_COLORS,
    GRAPH_CONFIG, GRAPH_CONFIG_MIN,
    rc, sc, gc, rgba,
    make_header, make_kpi, make_section_title, make_methodology_note,
)
from stats_utils import (
    wilcoxon_paired, trend_analysis, mean_ci, correlation_ci,
)

# ══════════════════════════════════════════════════════════════════════
# 1. JERARQUÍA PIRAMIDAL
# ══════════════════════════════════════════════════════════════════════
PYRAMID = {
    'N0': {'INFORM': 'INFORM Risk Index'},
    'N1': {'HA': 'Hazard & Exposure', 'VU': 'Vulnerabilidad',
           'CC': 'Falta Capacidad Respuesta'},
    'N2_HA': {
        'HA.NAT':  'Amenaza Natural',
        'HA.HUM':  'Amenaza Humana',
        'HA.VECT': 'Enfermedades Vectoriales',
        'HA.ZOON': 'Zoonosis',
        'HA.FWB':  'Alimentos y Agua',
    },
    'N2_VU': {
        'VU.SEV': 'Vulnerabilidad Socioeconómica',
        'VU.VGR': 'Grupos Vulnerables',
    },
    'N2_CC': {
        'CC.INF': 'Infraestructura',
        'CC.INS': 'Institucional',
    },
    'N3': {
        'HA.NAT.EPI':      'Riesgo Epidémico',
        'HA.NAT.EQ':       'Terremoto',
        'HA.NAT.FL':       'Inundación fluvial',
        'HA.NAT.TC':       'Ciclón tropical',
        'HA.NAT.DR':       'Sequía',
        'HA.NAT.CFL':      'Inundación costera',
        'HA.NAT.TS':       'Tsunami',
        'HA.HUM.CON.GCRI': 'Probabilidad conflicto',
        'HA.HUM.CON.GPI':  'Intensidad conflicto',
        'HA.VECT.MAL':     'Malaria',
        'HA.VECT.DENG':    'Dengue',
        'HA.VECT.AEDES':   'Exposición Aedes',
        'HA.VECT.ZIKV':    'Zika',
        'VU.SEV.PD':       'Pobreza y desarrollo',
        'VU.SEV.INQ':      'Desigualdad',
        'VU.SEV.AD':       'Dependencia económica',
        'VU.VGR.UP':       'Personas desplazadas',
        'VU.VGR.OG.HE':    'Condiciones de salud',
        'VU.VGR.OG.FS':    'Seguridad alimentaria',
        'CC.INF.AHC':      'Acceso a salud',
        'CC.INF.COM':      'Comunicación',
        'CC.INF.PHY':      'Infraestructura física',
        'CC.INS.GOV':      'Gobernanza',
        'CC.INS.DRR':      'Reducción riesgo desastres',
    },
    'N4_COVID': {
        'CC.INF.AHC.PHYS':       'Densidad de médicos',
        'CC.INF.AHC.HEALTH-EXP': 'Gasto en salud per cápita',
        'CC.INF.AHC.IMM':        'Cobertura vacunación',
        'CC.INF.AHC.MMR':        'Mortalidad materna',
        'CC.INF.COM.NETUS':      'Usuarios de internet',
        'CC.INS.GOV.CPI':        'Índice de corrupción',
        'CC.INS.GOV.GE':         'Efectividad gubernamental',
        'VU.SEV.PD.HDI':         'Índice desarrollo humano',
        'VU.SEV.INQ.GINI':       'Coeficiente GINI',
        'VU.VGR.OG.U5':          'Salud infantil (menores 5)',
    }
}

ALL_IDS = []
for level in PYRAMID.values():
    ALL_IDS.extend(level.keys())

def ind_name(id_):
    for level in PYRAMID.values():
        if id_ in level:
            return level[id_]
    return id_

# ══════════════════════════════════════════════════════════════════════
# 2. CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════
df_raw = pd.read_excel(data_path('inform_trend'))

records = []
for _, row in df_raw.iterrows():
    iso = row['Iso3']
    ind = row['IndicatorId']
    if ind not in ALL_IDS:
        continue
    is_latam = iso in LATAM_ISO3
    records.append({
        'Country':     LATAM_ISO3.get(iso, iso),
        'ISO3':        iso,
        'IsLATAM':     is_latam,
        'Year':        int(row['INFORMYear']),
        'IndicatorId': ind,
        'Score':       float(row['IndicatorScore']) if pd.notna(row['IndicatorScore']) else np.nan,
    })
df = pd.DataFrame(records)
df_latam = df[df['IsLATAM']].copy()
df_glob  = df.groupby(['Year', 'IndicatorId'])['Score'].mean().reset_index()

def get_pivot(ind_id, latam_only=True):
    src = df_latam if latam_only else df
    sub = src[src['IndicatorId'] == ind_id]
    return sub.pivot_table(index='Country', columns='Year', values='Score')

# ══════════════════════════════════════════════════════════════════════
# 3. ESTADÍSTICOS CANÓNICOS (con IC bootstrap y stats_utils)
# ══════════════════════════════════════════════════════════════════════
PIVOT_INFORM = get_pivot('INFORM')

LATAM_GLOB_STATS = {}
for yr in YEARS:
    lat_vals = df_latam[(df_latam['IndicatorId']=='INFORM') &
                        (df_latam['Year']==yr)]['Score'].dropna()
    glob_row = df_glob[(df_glob['IndicatorId']=='INFORM') & (df_glob['Year']==yr)]['Score']
    ci_info = mean_ci(lat_vals.values)
    LATAM_GLOB_STATS[yr] = {
        'lat_mean':  ci_info['mean'],
        'lat_sd':    ci_info['sd'],
        'lat_med':   round(float(lat_vals.median()), 3) if len(lat_vals) else np.nan,
        'lat_q25':   round(float(lat_vals.quantile(0.25)), 3) if len(lat_vals) else np.nan,
        'lat_q75':   round(float(lat_vals.quantile(0.75)), 3) if len(lat_vals) else np.nan,
        'lat_ci_lo': ci_info['ci95_boot'][0],
        'lat_ci_hi': ci_info['ci95_boot'][1],
        'glob_mean': round(float(glob_row.values[0]), 3) if len(glob_row) else np.nan,
    }

# Wilcoxon pandémico robusto — emparejamiento correcto por país
wilcoxon_pairs = []
for c in LATAM:
    if c not in PIVOT_INFORM.index:
        continue
    v19 = PIVOT_INFORM.loc[c, 2019] if 2019 in PIVOT_INFORM.columns else np.nan
    v21 = PIVOT_INFORM.loc[c, 2021] if 2021 in PIVOT_INFORM.columns else np.nan
    wilcoxon_pairs.append((c, v19, v21))

_labels = [p[0] for p in wilcoxon_pairs]
_v19    = [p[1] for p in wilcoxon_pairs]
_v21    = [p[2] for p in wilcoxon_pairs]
WX_19_21 = wilcoxon_paired(_v21, _v19, labels=_labels)  # x=post, y=pre

# Tendencias lineales con corrección FDR
TRENDS_DF = trend_analysis(PIVOT_INFORM, min_points=5, adjust_method='fdr_bh')
N_SIG     = int((TRENDS_DF['p'] < 0.05).sum())      # sin ajustar
N_SIG_ADJ = int((TRENDS_DF['p_adj'] < 0.05).sum())  # con FDR

# Clustering 2025
sub_cl = df_latam[(df_latam['Year']==2025) &
                  (df_latam['IndicatorId'].isin(['HA','VU','CC']))
                  ].pivot_table(index='Country', columns='IndicatorId', values='Score').dropna()
if len(sub_cl) >= 3:
    Xsc = StandardScaler().fit_transform(sub_cl.values)
    km  = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = km.fit_predict(Xsc)
    ov25 = {c: float(PIVOT_INFORM.loc[c, 2025]) if c in PIVOT_INFORM.index else 5
            for c in sub_cl.index}
    sk = sorted(range(3), key=lambda k: np.mean([ov25.get(c, 5) for c in sub_cl.index[labels==k]]))
    CNAMES = {sk[0]: 'Bajo riesgo', sk[1]: 'Riesgo medio', sk[2]: 'Alto riesgo'}
    COUNTRY_CLUSTER = {c: CNAMES[labels[i]] for i, c in enumerate(sub_cl.index)}
    pca2 = PCA(n_components=2); Xpca = pca2.fit_transform(Xsc)
    PCA_DF = pd.DataFrame({
        'País':    sub_cl.index,
        'PC1':     Xpca[:, 0],
        'PC2':     Xpca[:, 1],
        'Cluster': [COUNTRY_CLUSTER.get(c, '') for c in sub_cl.index],
        'Risk':    [ov25.get(c, 5) for c in sub_cl.index],
    })
    PCA_VAR = pca2.explained_variance_ratio_
else:
    COUNTRY_CLUSTER = {}
    PCA_DF = pd.DataFrame()
    PCA_VAR = [0, 0]

# ══════════════════════════════════════════════════════════════════════
# 4. FIGURAS
# ══════════════════════════════════════════════════════════════════════

def fig_serie():
    lat = [LATAM_GLOB_STATS[yr]['lat_mean'] for yr in YEARS]
    q25 = [LATAM_GLOB_STATS[yr]['lat_q25']  for yr in YEARS]
    q75 = [LATAM_GLOB_STATS[yr]['lat_q75']  for yr in YEARS]
    glo = [LATAM_GLOB_STATS[yr]['glob_mean'] for yr in YEARS]
    ci_lo = [LATAM_GLOB_STATS[yr]['lat_ci_lo'] for yr in YEARS]
    ci_hi = [LATAM_GLOB_STATS[yr]['lat_ci_hi'] for yr in YEARS]

    fig = go.Figure()
    # IQR (banda más tenue)
    fig.add_trace(go.Scatter(
        x=YEARS + YEARS[::-1], y=q75 + q25[::-1],
        fill='toself', fillcolor=rgba(C['latam'], 0.08), line=dict(width=0),
        name='IQR LATAM', hoverinfo='skip'))
    # IC 95% bootstrap de la media
    fig.add_trace(go.Scatter(
        x=YEARS + YEARS[::-1], y=ci_hi + ci_lo[::-1],
        fill='toself', fillcolor=rgba(C['latam'], 0.18), line=dict(width=0),
        name='IC 95% media', hoverinfo='skip'))
    # Líneas por país (tenues, coloreadas por tendencia)
    for c in LATAM:
        if c not in PIVOT_INFORM.index:
            continue
        d = PIVOT_INFORM.loc[c].dropna()
        tr = TRENDS_DF[TRENDS_DF['País'] == c]
        sl = tr['slope'].values[0] if len(tr) > 0 else 0
        fig.add_trace(go.Scatter(
            x=list(d.index), y=list(d.values), mode='lines',
            name=sn(c), line=dict(color=sc(sl), width=0.8),
            opacity=0.3, showlegend=False,
            hovertemplate=f'<b>{sn(c)}</b> %{{x}}: %{{y:.2f}}<extra></extra>'))
    # Media global
    fig.add_trace(go.Scatter(
        x=YEARS, y=glo, mode='lines+markers', name='Media global',
        line=dict(color=C['global_c'], width=2, dash='dash'),
        marker=dict(size=5)))
    # Media LATAM
    fig.add_trace(go.Scatter(
        x=YEARS, y=lat, mode='lines+markers+text', name='Media LATAM',
        line=dict(color=C['latam'], width=3), marker=dict(size=8),
        text=[f'{v:.2f}' for v in lat], textposition='top center',
        textfont=dict(size=9, color=C['latam'])))
    fig.add_vline(x=2019.5, line_dash='dash', line_color=C['covid'], line_width=1.5,
                  annotation_text='COVID-19',
                  annotation_font=dict(size=9, color=C['covid']),
                  annotation_position='top right')
    fig.update_layout(**{**LAYOUT, 'height': 440},
        xaxis=dict(tickvals=YEARS, gridcolor=C['grid'], title='Año'),
        yaxis=dict(title='INFORM Risk (0–10 · mayor = más riesgo)',
                   range=[1.5, 8.5], gridcolor=C['grid']),
        legend=dict(orientation='h', y=-0.18, bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=55, r=20, t=30, b=70))
    return fig


def fig_heatmap():
    sc_list = sorted(LATAM, key=lambda c: -(float(PIVOT_INFORM.loc[c, 2025])
                     if c in PIVOT_INFORM.index and pd.notna(PIVOT_INFORM.loc[c, 2025])
                     else 0))
    z = np.array([[float(PIVOT_INFORM.loc[c, yr])
                   if c in PIVOT_INFORM.index and yr in PIVOT_INFORM.columns
                      and pd.notna(PIVOT_INFORM.loc[c, yr])
                   else np.nan for yr in YEARS] for c in sc_list])
    text = np.where(np.isnan(z), 'N/D', np.round(z, 1).astype('object'))
    fig = go.Figure(go.Heatmap(
        z=z, x=[str(y) for y in YEARS], y=[sn(c) for c in sc_list],
        text=text, texttemplate='%{text}', textfont=dict(size=9, color=C['ink']),
        colorscale=RISK_COLORSCALE, zmin=2, zmax=8,
        hovertemplate='<b>%{y}</b> %{x}: %{z:.2f}<extra></extra>',
        colorbar=dict(title=dict(text='Riesgo', font=dict(color=C['ink'])),
                      tickfont=dict(size=9, color=C['ink']),
                      tickvals=[2, 4, 6, 8], ticktext=['Bajo', 'Medio', 'Alto', 'Muy alto'])))
    fig.add_vline(x=2.5, line_dash='dot', line_color=C['covid'], line_width=1)
    fig.update_layout(**{**LAYOUT, 'height': 620},
        xaxis=dict(side='top', gridcolor=C['grid']),
        yaxis=dict(gridcolor=C['grid']),
        margin=dict(l=110, r=20, t=50, b=20))
    return fig


def fig_pandemia():
    rows = []
    for c in LATAM:
        if c not in PIVOT_INFORM.index:
            continue
        v19 = PIVOT_INFORM.loc[c, 2019] if 2019 in PIVOT_INFORM.columns else np.nan
        v21 = PIVOT_INFORM.loc[c, 2021] if 2021 in PIVOT_INFORM.columns else np.nan
        if pd.notna(v19) and pd.notna(v21):
            rows.append({'País': c, 'v19': float(v19), 'v21': float(v21),
                         'delta': round(float(v21) - float(v19), 2)})
    df_p = pd.DataFrame(rows).sort_values('delta', ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[sn(c) for c in df_p['País']], x=df_p['v19'],
        name='2019 (basal pre-COVID)', orientation='h',
        marker=dict(color=rgba(C['global_c'], 0.55), line_width=0)))
    fig.add_trace(go.Bar(
        y=[sn(c) for c in df_p['País']], x=df_p['v21'],
        name='2021 (post 1er año COVID)', orientation='h',
        marker=dict(color=C['covid'], line_width=0),
        customdata=df_p['delta'].values,
        hovertemplate='<b>%{y}</b><br>2021: %{x:.2f}<br>Δ: %{customdata:+.2f}<extra></extra>'))
    for _, row in df_p.iterrows():
        col = C['red'] if row['delta'] > 0 else C['green']
        fig.add_annotation(x=max(row['v19'], row['v21']) + 0.1, y=sn(row['País']),
                           text=f'{row["delta"]:+.2f}', showarrow=False,
                           font=dict(size=9, color=col))
    title = (f'Wilcoxon 2019 vs 2021 (emparejado, n={WX_19_21["n"]}): '
             f'W={WX_19_21["W"]:.0f}, p={WX_19_21["p"]:.4f} · '
             f'r Rosenthal={WX_19_21["r_rosenthal"]} · '
             f'Cliff\'s δ={WX_19_21["cliffs_delta"]} · {WX_19_21["interpretation"]}')
    fig.update_layout(**{**LAYOUT, 'height': 600}, barmode='overlay',
        title=dict(text=title, font=dict(size=10, color=C['muted'])),
        xaxis=dict(range=[1, 8.5], gridcolor=C['grid']),
        yaxis=dict(gridcolor=C['grid']),
        legend=dict(orientation='h', y=1.05, bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=100, r=80, t=80, b=40))
    return fig


def fig_tendencias_slope():
    df_t = TRENDS_DF.sort_values('slope', ascending=True)
    # Color por significancia AJUSTADA (no por magnitud arbitraria)
    colors = [C['red'] if (row['p_adj'] < 0.05 and row['slope'] > 0)
              else C['green'] if (row['p_adj'] < 0.05 and row['slope'] < 0)
              else C['muted']
              for _, row in df_t.iterrows()]
    fig = go.Figure(go.Bar(
        y=[sn(c) for c in df_t['País']], x=df_t['slope'], orientation='h',
        marker=dict(color=colors, line_width=0),
        error_x=dict(type='data', array=df_t['ci95_slope'].values, visible=True,
                     color=rgba(C['ink'], 0.35), thickness=1.5),
        customdata=df_t[['r2', 'p', 'p_adj', 'sig_adj']].values,
        hovertemplate=('<b>%{y}</b><br>Slope: %{x:+.4f} pts/año<br>'
                       'R²=%{customdata[0]}<br>p=%{customdata[1]}<br>'
                       'p_adj (FDR)=%{customdata[2]} %{customdata[3]}<extra></extra>')))
    fig.add_vline(x=0, line_color=C['muted'], line_width=1)
    title = (f'{N_SIG_ADJ}/{len(TRENDS_DF)} países con tendencia significativa tras '
             f'corrección FDR (Benjamini-Hochberg) · vs {N_SIG} sin corregir · '
             f'Barras = IC 95% del slope')
    fig.update_layout(**{**LAYOUT, 'height': 600},
        title=dict(text=title, font=dict(size=10, color=C['muted'])),
        xaxis=dict(title='Cambio anual (pts/año)', gridcolor=C['grid']),
        yaxis=dict(gridcolor=C['grid']),
        margin=dict(l=100, r=40, t=75, b=50))
    return fig


def fig_piramide_latam(yr=2021):
    niveles = [
        ('N1', list(PYRAMID['N1'].items())),
        ('N2', list({**PYRAMID['N2_HA'], **PYRAMID['N2_VU'], **PYRAMID['N2_CC']}.items())),
        ('N3', [(k, v) for k, v in PYRAMID['N3'].items()
                if k in ['HA.NAT.EPI','HA.VECT.DENG','HA.VECT.ZIKV','VU.SEV.INQ',
                         'VU.VGR.UP','CC.INF.AHC','CC.INS.GOV','HA.NAT.EQ','HA.VECT.MAL']]),
    ]
    fig = make_subplots(1, 3, subplot_titles=[
        'Nivel 1 — 3 Dimensiones',
        'Nivel 2 — Categorías',
        'Nivel 3 — Indicadores clave'])
    for col_i, (lbl, items) in enumerate(niveles, 1):
        ids = [k for k, _ in items]
        names = [v for _, v in items]
        lat_v, glo_v = [], []
        for id_ in ids:
            lat = df_latam[(df_latam['IndicatorId']==id_) &
                           (df_latam['Year']==yr)]['Score'].mean()
            glo = df_glob[(df_glob['IndicatorId']==id_) & (df_glob['Year']==yr)]['Score']
            lat_v.append(round(float(lat), 2) if pd.notna(lat) else 0)
            glo_v.append(round(float(glo.values[0]), 2) if len(glo) > 0 else 0)
        colors = [DIM_COLORS.get(id_, C['latam']) for id_ in ids]
        fig.add_trace(go.Bar(
            y=names, x=lat_v, name='LATAM', orientation='h',
            marker=dict(color=colors, opacity=0.85, line_width=0),
            customdata=[[round(l-g, 2)] for l, g in zip(lat_v, glo_v)],
            hovertemplate='%{y}<br>LATAM: %{x:.2f}<br>Δ vs Global: %{customdata[0]:+.2f}<extra></extra>',
            showlegend=(col_i == 1)), row=1, col=col_i)
        fig.add_trace(go.Scatter(
            x=glo_v, y=names, mode='markers',
            name='Global' if col_i == 1 else None,
            marker=dict(symbol='line-ns', color=C['amber'], size=10, line_width=2),
            hovertemplate='Global: %{x:.2f}<extra></extra>',
            showlegend=(col_i == 1)), row=1, col=col_i)
    fig.update_annotations(font=dict(color=C['ink'], size=11))
    fig.update_layout(**{**LAYOUT, 'height': 500}, barmode='overlay',
        legend=dict(orientation='h', y=-0.1, bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=160, r=20, t=55, b=60))
    fig.update_xaxes(range=[0, 9], gridcolor=C['grid'])
    fig.update_yaxes(tickfont=dict(size=9), gridcolor=C['grid'])
    return fig


def fig_n3_heatmap(yr=2025):
    ids = list(PYRAMID['N3'].keys())
    ids_avail = [i for i in ids if i in df_latam['IndicatorId'].values]
    sorted_c = sorted(LATAM, key=lambda c: -(float(PIVOT_INFORM.loc[c, yr])
                       if c in PIVOT_INFORM.index and yr in PIVOT_INFORM.columns
                       and pd.notna(PIVOT_INFORM.loc[c, yr]) else 0))
    z = []
    for c in sorted_c:
        row = []
        for id_ in ids_avail:
            v = df_latam[(df_latam['Country']==c) &
                         (df_latam['IndicatorId']==id_) &
                         (df_latam['Year']==yr)]['Score']
            row.append(float(v.values[0]) if len(v) > 0 and pd.notna(v.values[0]) else np.nan)
        z.append(row)
    z = np.array(z)
    names = [PYRAMID['N3'].get(i, i)[:22] for i in ids_avail]
    text = np.where(np.isnan(z), '—', np.round(z, 1).astype('object'))
    fig = go.Figure(go.Heatmap(
        z=z, x=names, y=[sn(c) for c in sorted_c],
        text=text, texttemplate='%{text}', textfont=dict(size=9, color=C['ink']),
        colorscale=RISK_COLORSCALE, zmin=0, zmax=10,
        hovertemplate='<b>%{y}</b><br>%{x}: %{z:.2f}<extra></extra>',
        colorbar=dict(title=dict(text='Riesgo', font=dict(color=C['ink'])),
                      tickfont=dict(size=9, color=C['ink']))))
    fig.update_layout(**{**LAYOUT, 'height': 620},
        xaxis=dict(side='top', tickangle=-40, tickfont=dict(size=9), gridcolor=C['grid']),
        yaxis=dict(gridcolor=C['grid']),
        margin=dict(l=110, r=20, t=160, b=20))
    return fig


def fig_n3_brechas(yr=2021):
    ids = [i for i in PYRAMID['N3'] if i in df_latam['IndicatorId'].values]
    rows = []
    for id_ in ids:
        lat = df_latam[(df_latam['IndicatorId']==id_) &
                       (df_latam['Year']==yr)]['Score'].mean()
        glo = df_glob[(df_glob['IndicatorId']==id_) & (df_glob['Year']==yr)]['Score']
        if pd.notna(lat) and len(glo) > 0:
            diff = round(float(lat) - float(glo.values[0]), 2)
            rows.append({'id': id_, 'name': PYRAMID['N3'][id_], 'diff': diff,
                         'lat': round(float(lat), 2),
                         'glo': round(float(glo.values[0]), 2)})
    df_b = pd.DataFrame(rows).sort_values('diff', ascending=True)
    fig = go.Figure(go.Bar(
        y=df_b['name'], x=df_b['diff'], orientation='h',
        marker=dict(color=[gc(v) for v in df_b['diff']], line_width=0),
        customdata=df_b[['lat', 'glo']].values,
        hovertemplate=('<b>%{y}</b><br>LATAM: %{customdata[0]:.2f}<br>'
                       'Global: %{customdata[1]:.2f}<br>Δ: %{x:+.2f}<extra></extra>')))
    fig.add_vline(x=0, line_color=C['muted'], line_width=1)
    fig.update_layout(**{**LAYOUT, 'height': 600},
        title=dict(text=f'Δ LATAM vs Global {yr} · Rojo = LATAM peor · Verde = LATAM mejor',
                   font=dict(size=10, color=C['muted'])),
        xaxis=dict(title='Δ LATAM − Global', gridcolor=C['grid']),
        yaxis=dict(tickfont=dict(size=10), gridcolor=C['grid']),
        margin=dict(l=220, r=40, t=55, b=40))
    return fig


def fig_n4_covid(yr=2021):
    ids = [i for i in PYRAMID['N4_COVID'] if i in df_latam['IndicatorId'].values]
    rows = []
    for id_ in ids:
        lat = df_latam[(df_latam['IndicatorId']==id_) &
                       (df_latam['Year']==yr)]['Score'].mean()
        glo = df_glob[(df_glob['IndicatorId']==id_) & (df_glob['Year']==yr)]['Score']
        if pd.notna(lat) and len(glo) > 0:
            diff = round(float(lat) - float(glo.values[0]), 2)
            rows.append({'id': id_, 'name': PYRAMID['N4_COVID'][id_], 'diff': diff,
                         'lat': round(float(lat), 2),
                         'glo': round(float(glo.values[0]), 2)})
    df_c = pd.DataFrame(rows).sort_values('diff', ascending=True)
    fig = go.Figure(go.Bar(
        y=df_c['name'], x=df_c['diff'], orientation='h',
        marker=dict(color=[gc(v) for v in df_c['diff']], line_width=0),
        customdata=df_c[['lat', 'glo']].values,
        hovertemplate=('<b>%{y}</b><br>LATAM: %{customdata[0]:.2f}<br>'
                       'Global: %{customdata[1]:.2f}<br>Δ: %{x:+.2f}<extra></extra>')))
    fig.add_vline(x=0, line_color=C['muted'], line_width=1)
    fig.update_layout(**{**LAYOUT, 'height': 420},
        title=dict(text=f'Indicadores sanitarios COVID-relevantes N4 {yr} — Δ LATAM vs Global',
                   font=dict(size=10, color=C['muted'])),
        xaxis=dict(title='Δ LATAM − Global', gridcolor=C['grid']),
        yaxis=dict(tickfont=dict(size=10), gridcolor=C['grid']),
        margin=dict(l=220, r=40, t=55, b=40))
    return fig


def fig_indicador_paises(ind_id='HA.NAT.EPI', yr=2021):
    d = df_latam[(df_latam['IndicatorId']==ind_id) &
                 (df_latam['Year']==yr)].sort_values('Score', ascending=True)
    glob_m = df_glob[(df_glob['IndicatorId']==ind_id) & (df_glob['Year']==yr)]['Score']
    gm = float(glob_m.values[0]) if len(glob_m) > 0 else 0
    fig = go.Figure(go.Bar(
        y=[sn(c) for c in d['Country']], x=d['Score'], orientation='h',
        marker=dict(color=[rc(v) for v in d['Score']], line_width=0),
        hovertemplate='<b>%{y}</b>: %{x:.2f}<extra></extra>'))
    fig.add_vline(x=gm, line_dash='dot', line_color=C['amber'],
        annotation_text=f'Global {yr}: {gm:.2f}',
        annotation_font=dict(size=9, color=C['amber']))
    fig.update_layout(**{**LAYOUT, 'height': 580},
        title=dict(text=f'{ind_name(ind_id)} — {yr}',
                   font=dict(size=11, color=C['muted'])),
        xaxis=dict(range=[0, 10.5], gridcolor=C['grid'], title='Score (0–10)'),
        yaxis=dict(gridcolor=C['grid']),
        margin=dict(l=110, r=60, t=50, b=30))
    return fig


def fig_perfil(country='Haiti'):
    fig = make_subplots(1, 2, specs=[[{'type': 'polar'}, {'type': 'xy'}]],
        subplot_titles=['Dimensiones N2 — 2025 vs 2019',
                        'Tendencia INFORM Risk 2017–2025'])
    ids_r = ['HA.NAT', 'HA.HUM', 'HA.VECT', 'VU.SEV', 'VU.VGR', 'CC.INF', 'CC.INS']
    names_r = ['Nat.', 'Hum.', 'Vect.', 'Soc-Ec', 'Vul.Gr', 'Infra.', 'Instit.']
    theta = names_r + [names_r[0]]
    for yr, col, dash_ in [(2025, C['latam'], 'solid'), (2019, C['global_c'], 'dot')]:
        vals = []
        for id_ in ids_r:
            v = df_latam[(df_latam['Country']==country) &
                         (df_latam['IndicatorId']==id_) &
                         (df_latam['Year']==yr)]['Score']
            vals.append(float(v.values[0]) if len(v) > 0 and pd.notna(v.values[0]) else 0)
        vals = vals + [vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=theta, name=str(yr),
            line=dict(color=col, width=2.5 if yr == 2025 else 1.5, dash=dash_),
            fill='toself', fillcolor=rgba(col, 0.15 if yr == 2025 else 0.06)),
            row=1, col=1)
    if country in PIVOT_INFORM.index:
        d_c = PIVOT_INFORM.loc[country].dropna()
        fig.add_trace(go.Scatter(
            x=list(d_c.index), y=list(d_c.values),
            mode='lines+markers+text', name=country,
            text=[f'{v:.1f}' for v in d_c.values], textposition='top center',
            textfont=dict(size=9, color=C['latam']),
            line=dict(color=C['latam'], width=2.5),
            marker=dict(size=7), showlegend=False), row=1, col=2)
    lat_m = [LATAM_GLOB_STATS[yr]['lat_mean'] for yr in YEARS]
    glo_m = [LATAM_GLOB_STATS[yr]['glob_mean'] for yr in YEARS]
    fig.add_trace(go.Scatter(x=YEARS, y=lat_m, mode='lines', name='Media LATAM',
        line=dict(color=C['amber'], width=1.5, dash='dot')), row=1, col=2)
    fig.add_trace(go.Scatter(x=YEARS, y=glo_m, mode='lines', name='Media global',
        line=dict(color=C['global_c'], width=1.5, dash='dash')), row=1, col=2)
    fig.add_shape(type='line', x0=2019.5, x1=2019.5, y0=0, y1=1, yref='paper',
                  line=dict(dash='dash', color=C['covid'], width=1), row=1, col=2)
    cl = COUNTRY_CLUSTER.get(country, '—')
    tr = TRENDS_DF[TRENDS_DF['País'] == country]
    sl = tr['slope'].values[0] if len(tr) > 0 else 0
    p_adj = tr['p_adj'].values[0] if len(tr) > 0 else 1
    for ann_text, ann_y, ann_col in [
        (f'Cluster 2025: {cl}', 0.02, CLUSTER_COLORS.get(cl, C['muted'])),
        (f'Tendencia: {sl:+.4f} pts/año (p_adj={p_adj:.3f})', 0.09, sc(sl))]:
        fig.add_annotation(x=0.99, y=ann_y, xref='paper', yref='paper',
            text=ann_text, showarrow=False,
            font=dict(size=9, color=ann_col), align='right')
    fig.update_annotations(font=dict(color=C['ink'], size=11))
    fig.update_layout(**{**LAYOUT, 'height': 480},
        polar=dict(bgcolor=C['white'],
                   radialaxis=dict(visible=True, range=[0, 10],
                                   tickfont=dict(size=8, color=C['muted']),
                                   gridcolor=C['grid'], linecolor=C['border']),
                   angularaxis=dict(tickfont=dict(size=9, color=C['ink']),
                                    gridcolor=C['grid'], linecolor=C['border'])),
        legend=dict(orientation='h', y=-0.14, bgcolor='rgba(0,0,0,0)', font=dict(size=9)),
        margin=dict(l=20, r=20, t=55, b=60))
    fig.update_xaxes(tickvals=YEARS, gridcolor=C['grid'], row=1, col=2)
    fig.update_yaxes(range=[1, 9], gridcolor=C['grid'], title='INFORM Risk', row=1, col=2)
    return fig


def fig_paradoja():
    try:
        ghs = pd.read_csv(data_path('ghs'))
        ghs21 = ghs[(ghs['Year']==2021) & (ghs['Country'].isin(LATAM))].set_index('Country')
        inf21 = df_latam[(df_latam['IndicatorId']=='INFORM') &
                         (df_latam['Year']==2021)].set_index('Country')['Score']
        common = [c for c in LATAM if c in ghs21.index and c in inf21.index]
        x_i = [float(inf21[c]) for c in common]
        x_g = [float(ghs21.loc[c, 'OVERALL SCORE']) for c in common]
        cr = correlation_ci(x_i, x_g, method='pearson')
        fig = go.Figure(go.Scatter(
            x=x_i, y=x_g, mode='markers+text',
            text=[sn(c) for c in common], textposition='top center',
            textfont=dict(size=9, color=C['ink']),
            marker=dict(size=12, color=x_i, colorscale=RISK_COLORSCALE,
                        showscale=True, line=dict(color='white', width=1),
                        colorbar=dict(title=dict(text='INFORM', font=dict(color=C['ink'])),
                                      tickfont=dict(size=9, color=C['ink']))),
            hovertemplate='<b>%{text}</b><br>INFORM: %{x:.2f}<br>GHS: %{y:.1f}<extra></extra>'))
        title = (f'INFORM Risk 2021 vs GHS Index 2021 (n={cr["n"]}) · '
                 f'Pearson r={cr["r"]} [IC95% {cr["ci"][0]}, {cr["ci"][1]}], p={cr["p"]:.3f} · '
                 f'Correlación moderada no concluyente con n=20 — '
                 f'los índices miden constructos distintos')
        fig.update_layout(**{**LAYOUT, 'height': 480},
            title=dict(text=title, font=dict(size=10, color=C['muted'])),
            xaxis=dict(title='INFORM Risk 2021 (mayor = peor)', gridcolor=C['grid']),
            yaxis=dict(title='GHS Index 2021 (mayor = mejor)', gridcolor=C['grid']),
            margin=dict(l=60, r=80, t=85, b=50))
        return fig
    except Exception as e:
        print(f'[paradoja] aviso: {e}')
        return go.Figure()


def fig_clustering():
    if PCA_DF.empty:
        return go.Figure()
    fig = go.Figure()
    for cl, col in CLUSTER_COLORS.items():
        sub = PCA_DF[PCA_DF['Cluster'] == cl]
        if len(sub) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=sub['PC1'], y=sub['PC2'], mode='markers+text', name=cl,
            text=[sn(c) for c in sub['País']], textposition='top center',
            textfont=dict(size=9, color=C['ink']),
            marker=dict(size=sub['Risk']*3, color=col,
                        line=dict(color='white', width=1), opacity=0.85),
            hovertemplate='<b>%{text}</b><br>Risk=%{marker.size:.1f}<extra></extra>'))
    fig.update_layout(**{**LAYOUT, 'height': 460},
        title=dict(text=f'k-means sobre HA+VU+CC 2025 · PCA: PC1={PCA_VAR[0]:.1%} PC2={PCA_VAR[1]:.1%}',
                   font=dict(size=10, color=C['muted'])),
        xaxis=dict(title='PC1 — Riesgo general', gridcolor=C['grid']),
        yaxis=dict(title='PC2 — Contraste dimensiones', gridcolor=C['grid']),
        legend=dict(orientation='h', y=-0.14, bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=60, r=20, t=60, b=60))
    return fig


# ══════════════════════════════════════════════════════════════════════
# 5. APP DASH
# ══════════════════════════════════════════════════════════════════════
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True, title='INFORM Risk v4.0')

IND_N3_OPTS = [{'label': f'{k} — {v}', 'value': k}
               for k, v in PYRAMID['N3'].items()
               if k in df_latam['IndicatorId'].values]

# KPIs con IC 95%
lat25 = LATAM_GLOB_STATS[2025]['lat_mean']
lat25_lo = LATAM_GLOB_STATS[2025]['lat_ci_lo']
lat25_hi = LATAM_GLOB_STATS[2025]['lat_ci_hi']
glo25 = LATAM_GLOB_STATS[2025]['glob_mean']
lat19 = LATAM_GLOB_STATS[2019]['lat_mean']
epi19_vals = df_latam[(df_latam['IndicatorId']=='HA.NAT.EPI') &
                      (df_latam['Year']==2019)]['Score'].dropna()
epi19 = round(float(epi19_vals.mean()), 2) if len(epi19_vals) else 0
epi19g_row = df_glob[(df_glob['IndicatorId']=='HA.NAT.EPI') & (df_glob['Year']==2019)]['Score']
epi19g = round(float(epi19g_row.values[0]), 2) if len(epi19g_row) else 0

server = app.server  # Requerido por gunicorn/Render
app.layout = dbc.Container([
    # Header
    make_header(
        'INFORM RISK INDEX',
        f'PIRÁMIDE DE RIESGO · 4 NIVELES · 286 INDICADORES · LATAM-20 · 2017–2025 · JRC/UE'),

    # Nota metodológica destacada
    make_methodology_note(
        'ESCALA INVERTIDA: 0–10 donde MAYOR = MÁS RIESGO. '
        'Pirámide: N0 → 3 dimensiones → 9 categorías → 25 componentes → indicadores base. '
        'Tendencias ajustadas por FDR (Benjamini-Hochberg). '
        f'{NOTA_MUESTRA}',
        accent='amber'),

    # KPIs
    dbc.Row([
        make_kpi(f'{lat25}', 'RISK LATAM 2025',
                 f'IC95% [{lat25_lo}, {lat25_hi}] · Global: {glo25}', C['red']),
        make_kpi(f'{lat19}', 'RISK LATAM 2019',
                 'Basal pre-pandemia', C['blue']),
        make_kpi(f'+{round(lat25-lat19,2)}', 'AUMENTO 2019→2025',
                 'Δ acumulado', C['orange']),
        make_kpi(f'{epi19}', 'EPIDÉMICO 2019',
                 f'Global: {epi19g} · Δ=+{round(epi19-epi19g, 2)}', C['red']),
        make_kpi(f'{N_SIG_ADJ}/{len(TRENDS_DF)}', 'TEND. SIG. (FDR)',
                 f'p_adj<0.05 · {N_SIG} sin ajustar', C['amber']),
        make_kpi(f'p={WX_19_21["p"]:.3f}', 'WILCOXON 19→21',
                 WX_19_21['interpretation'][:28], C['purple']),
    ], style={'marginBottom': '10px'}),

    # Tabs organizados en 4 secciones temáticas (antes 13 planos)
    dcc.Tabs(id='tabs', value='t1', children=[
        # Sección 1: SERIE & TENDENCIAS
        dcc.Tab(label='📈 Serie completa',     value='t1', style=TS, selected_style=TSS),
        dcc.Tab(label='Heatmap histórico',     value='t3', style=TS, selected_style=TSS),
        dcc.Tab(label='Tendencias (FDR)',      value='t4', style=TS, selected_style=TSS),
        # Sección 2: PANDEMIA
        dcc.Tab(label='🦠 Análisis pandémico',  value='t2', style=TS, selected_style=TSS),
        dcc.Tab(label='Indicadores COVID N4',  value='t8', style=TS, selected_style=TSS),
        # Sección 3: ESTRUCTURA PIRAMIDAL
        dcc.Tab(label='🔻 Pirámide N1→N2→N3', value='t5', style=TS, selected_style=TSS),
        dcc.Tab(label='Heatmap N3',            value='t6', style=TS, selected_style=TSS),
        dcc.Tab(label='Brechas LATAM–Global',  value='t7', style=TS, selected_style=TSS),
        dcc.Tab(label='Indicador × países',    value='t9', style=TS, selected_style=TSS),
        # Sección 4: TIPOLOGÍA & PAÍS
        dcc.Tab(label='🧩 Clustering',          value='t10', style=TS, selected_style=TSS),
        dcc.Tab(label='Perfil de país',        value='t11', style=TS, selected_style=TSS),
        # Sección 5: CRUCE
        dcc.Tab(label='⚖️ Paradoja vs GHS',     value='t12', style=TS, selected_style=TSS),
        dcc.Tab(label='Tabla maestra',         value='t13', style=TS, selected_style=TSS),
    ], style={'borderBottom': f'1px solid {C["border"]}', 'marginBottom': '14px'}),
    html.Div(id='content'),
], fluid=True, style={'backgroundColor': C['bg'], 'minHeight': '100vh',
                      'padding': '0 20px 40px'})


# ══════════════════════════════════════════════════════════════════════
# 6. ROUTING
# ══════════════════════════════════════════════════════════════════════
@app.callback(Output('content', 'children'), Input('tabs', 'value'))
def render(tab):
    if tab == 't1':
        rows = [{'Año': yr,
                 'Media LATAM': LATAM_GLOB_STATS[yr]['lat_mean'],
                 'IC 95%': f'[{LATAM_GLOB_STATS[yr]["lat_ci_lo"]}, {LATAM_GLOB_STATS[yr]["lat_ci_hi"]}]',
                 'Mediana': LATAM_GLOB_STATS[yr]['lat_med'],
                 'SD': LATAM_GLOB_STATS[yr]['lat_sd'],
                 'Media Global': LATAM_GLOB_STATS[yr]['glob_mean'],
                 'Δ LATAM-Global': round(LATAM_GLOB_STATS[yr]['lat_mean'] -
                                         LATAM_GLOB_STATS[yr]['glob_mean'], 3)}
                for yr in YEARS]
        return html.Div([
            html.Div([
                make_section_title(
                    'Serie INFORM Risk 2017–2025 · LATAM vs Global',
                    'Banda clara = IC 95% bootstrap · Banda tenue = IQR · discontinua = global · colores por tendencia'),
                dcc.Graph(figure=fig_serie(), config=GRAPH_CONFIG)], style=CARD),
            html.Div([
                make_section_title('Estadísticas por año (con IC 95% bootstrap)'),
                dash_table.DataTable(
                    data=rows,
                    columns=[{'name': c, 'id': c} for c in rows[0].keys()],
                    **TABLE_STYLE,
                    style_data_conditional=[
                        {'if': {'filter_query': '{Δ LATAM-Global} > 0',
                                'column_id': 'Δ LATAM-Global'},
                         'color': C['red'], 'fontWeight': '700'}])], style=CARD),
        ])

    elif tab == 't2':
        return html.Div([
            make_methodology_note(
                '2019 = basal pre-pandemia · 2021 = post primer año COVID. '
                'INFORM es evaluación externa, no autorreporte — un aumento refleja '
                'deterioro estructural real. Reportamos tamaño de efecto '
                '(r de Rosenthal y Cliff\'s δ) además del p-valor.',
                accent='covid'),
            html.Div([
                make_section_title('INFORM Risk 2019 vs 2021 por país',
                    f'Wilcoxon W={WX_19_21["W"]:.0f}, p={WX_19_21["p"]:.4f}, '
                    f'r={WX_19_21["r_rosenthal"]}, δ={WX_19_21["cliffs_delta"]}, n={WX_19_21["n"]}'),
                dcc.Graph(figure=fig_pandemia(), config=GRAPH_CONFIG)], style=CARD),
            html.Div([
                make_section_title('Riesgo epidémico por año',
                    f'LATAM entró a COVID con +{round(epi19-epi19g,2)} pts sobre el mundo'),
                dbc.Row([dbc.Col([dcc.Dropdown(
                    [{'label': str(y), 'value': y} for y in YEARS],
                    value=2019, id='epi-yr', clearable=False, style=DROPDOWN_STYLE)
                ], md=3)], style={'marginBottom': '10px'}),
                dcc.Graph(id='epi-bar', config=GRAPH_CONFIG)], style=CARD),
        ])

    elif tab == 't3':
        return html.Div([html.Div([
            make_section_title('Heatmap INFORM Risk 2017–2025',
                               'Verde = bajo riesgo · Rojo = alto · línea = COVID-19'),
            dcc.Graph(figure=fig_heatmap(), config=GRAPH_CONFIG)], style=CARD)])

    elif tab == 't4':
        return html.Div([
            make_methodology_note(
                'Ajuste por comparaciones múltiples vía Benjamini-Hochberg (FDR). '
                'El p_adj controla la tasa de falsos descubrimientos esperada al 5%. '
                'Clasificación de dirección basada en significancia AJUSTADA, no en magnitud arbitraria.',
                accent='blue'),
            html.Div([
                make_section_title('Tendencias lineales 2017–2025 (pts/año)',
                    f'{N_SIG_ADJ}/{len(TRENDS_DF)} significativas tras FDR · Barras = IC 95% del slope'),
                dcc.Graph(figure=fig_tendencias_slope(), config=GRAPH_CONFIG)], style=CARD),
            html.Div([
                make_section_title('Tabla de tendencias (ordenable)'),
                dash_table.DataTable(
                    data=TRENDS_DF.to_dict('records'),
                    columns=[{'name': c, 'id': c}
                             for c in ['País', 'n_años', 'slope', 'ci95_slope',
                                       'r2', 'p', 'p_adj', 'sig_adj', 'Dirección']],
                    sort_action='native', **TABLE_STYLE,
                    style_data_conditional=[
                        {'if': {'filter_query': '{Dirección} contains "Aumento"',
                                'column_id': 'Dirección'}, 'color': C['red']},
                        {'if': {'filter_query': '{Dirección} contains "Descenso"',
                                'column_id': 'Dirección'}, 'color': C['green']},
                        {'if': {'filter_query': '{sig_adj} = "*"'},
                         'fontWeight': '700'}])], style=CARD),
        ])

    elif tab == 't5':
        return html.Div([html.Div([
            make_section_title('Pirámide INFORM: N1 → N2 → N3 · LATAM vs Global',
                'Barras = LATAM · marca ámbar = media global'),
            dbc.Row([dbc.Col([dcc.Dropdown(
                [{'label': str(y), 'value': y} for y in YEARS],
                value=2021, id='pir-yr', clearable=False, style=DROPDOWN_STYLE)
            ], md=3)], style={'marginBottom': '10px'}),
            dcc.Graph(id='pir-graph', config=GRAPH_CONFIG)], style=CARD)])

    elif tab == 't6':
        return html.Div([html.Div([
            make_section_title('Heatmap N3 — 25 componentes por país',
                'Ordenado por INFORM Risk descendente · verde = bajo · rojo = alto'),
            dbc.Row([dbc.Col([dcc.Dropdown(
                [{'label': str(y), 'value': y} for y in YEARS],
                value=2025, id='n3h-yr', clearable=False, style=DROPDOWN_STYLE)
            ], md=3)], style={'marginBottom': '10px'}),
            dcc.Graph(id='n3h-graph', config=GRAPH_CONFIG)], style=CARD)])

    elif tab == 't7':
        return html.Div([html.Div([
            make_section_title('Brechas LATAM vs Global por indicador N3',
                'Rojo = LATAM más vulnerable · Verde = LATAM mejor'),
            dbc.Row([dbc.Col([dcc.Dropdown(
                [{'label': str(y), 'value': y} for y in YEARS],
                value=2021, id='br-yr', clearable=False, style=DROPDOWN_STYLE)
            ], md=3)], style={'marginBottom': '10px'}),
            dcc.Graph(id='br-graph', config=GRAPH_CONFIG)], style=CARD)])

    elif tab == 't8':
        return html.Div([html.Div([
            make_section_title('Indicadores sanitarios N4 — relevantes para COVID',
                'Densidad médicos · gasto salud · vacunación · gobernanza · HDI'),
            dbc.Row([dbc.Col([dcc.Dropdown(
                [{'label': str(y), 'value': y} for y in YEARS],
                value=2019, id='n4-yr', clearable=False, style=DROPDOWN_STYLE)
            ], md=3)], style={'marginBottom': '10px'}),
            dcc.Graph(id='n4-graph', config=GRAPH_CONFIG)], style=CARD)])

    elif tab == 't9':
        return html.Div([html.Div([
            make_section_title('Un indicador N3 × todos los países',
                'Verde bajo riesgo · Rojo alto · línea ámbar = media global'),
            dbc.Row([
                dbc.Col([dcc.Dropdown(IND_N3_OPTS, value='HA.NAT.EPI',
                    id='ind-sel', clearable=False, style=DROPDOWN_STYLE)], md=7),
                dbc.Col([dcc.Dropdown(
                    [{'label': str(y), 'value': y} for y in YEARS],
                    value=2021, id='ind-yr', clearable=False,
                    style=DROPDOWN_STYLE)], md=3),
            ], style={'marginBottom': '12px'}),
            dcc.Graph(id='ind-graph', config=GRAPH_CONFIG)], style=CARD)])

    elif tab == 't10':
        return html.Div([
            html.Div([
                make_section_title('Tipología k-means sobre HA + VU + CC 2025',
                    f'PC1={PCA_VAR[0]:.1%} · PC2={PCA_VAR[1]:.1%} · tamaño = INFORM Risk'),
                dcc.Graph(figure=fig_clustering(), config=GRAPH_CONFIG)], style=CARD),
            html.Div([
                make_section_title('Composición de clusters'),
                html.Div([html.Div([
                    html.Div(cl, style={'fontSize': '12px', 'fontWeight': '600',
                                         'color': CLUSTER_COLORS.get(cl, C['muted']),
                                         'marginBottom': '4px'}),
                    html.Div(', '.join([c for c, cluster in COUNTRY_CLUSTER.items()
                                        if cluster == cl]),
                             style={'fontSize': '11px', 'color': C['text']})
                ], style={**CARD, 'borderLeft': f'3px solid {CLUSTER_COLORS.get(cl, C["muted"])}',
                          'borderRadius': '0 8px 8px 0', 'marginBottom': '8px'})
                    for cl in ['Alto riesgo', 'Riesgo medio', 'Bajo riesgo']])], style=CARD),
        ])

    elif tab == 't11':
        return html.Div([html.Div([
            make_section_title('Perfil N2 2025 vs 2019 + tendencia histórica',
                'Radar: 7 categorías N2 · Cluster y tendencia anotados'),
            dcc.Dropdown([{'label': c, 'value': c} for c in LATAM_20_SORTED],
                value='Haiti', id='pais-sel', clearable=False,
                style={**DROPDOWN_STYLE, 'marginBottom': '10px', 'width': '260px'}),
            dcc.Graph(id='perfil-graph', config=GRAPH_CONFIG)], style=CARD)])

    elif tab == 't12':
        return html.Div([
            make_methodology_note(
                'La correlación moderada no concluyente (n=20) entre INFORM y GHS '
                'confirma que los tres índices miden constructos distintos — '
                'justificación del metanálisis. INFORM: evaluación externa de riesgo · '
                'GHS: evaluación externa de preparación · SPAR: autorreporte estatal.',
                accent='red'),
            html.Div([
                make_section_title('INFORM Risk 2021 vs GHS Index 2021'),
                dcc.Graph(figure=fig_paradoja(), config=GRAPH_CONFIG)], style=CARD),
        ])

    elif tab == 't13':
        piv = PIVOT_INFORM.reset_index()
        piv.columns = [str(c) for c in piv.columns]
        tr_map = {r['País']: r['slope'] for _, r in TRENDS_DF.iterrows()}
        padj_map = {r['País']: r['p_adj'] for _, r in TRENDS_DF.iterrows()}
        piv['Tendencia'] = [round(tr_map.get(c, 0), 4) for c in piv['Country']]
        piv['p_adj']     = [round(padj_map.get(c, 1), 4) for c in piv['Country']]
        piv['Cluster']   = [COUNTRY_CLUSTER.get(c, '—') for c in piv['Country']]
        yr_cols = [str(y) for y in YEARS]
        cols = ['Country'] + yr_cols + ['Tendencia', 'p_adj', 'Cluster']
        return html.Div([html.Div([
            make_section_title('Tabla maestra INFORM Risk 2017–2025',
                'Exportable · filtrable · ordenable'),
            dash_table.DataTable(
                data=piv[cols].to_dict('records'),
                columns=[{'name': c, 'id': c} for c in cols],
                filter_action='native', sort_action='native',
                page_size=24, export_format='csv',
                **TABLE_STYLE,
                style_data_conditional=[
                    {'if': {'filter_query': '{Cluster} = "Alto riesgo"',
                            'column_id': 'Cluster'}, 'color': C['red']},
                    {'if': {'filter_query': '{Cluster} = "Riesgo medio"',
                            'column_id': 'Cluster'}, 'color': C['amber']},
                    {'if': {'filter_query': '{Cluster} = "Bajo riesgo"',
                            'column_id': 'Cluster'}, 'color': C['green']},
                ])], style=CARD)])


# ══════════════════════════════════════════════════════════════════════
# 7. CALLBACKS
# ══════════════════════════════════════════════════════════════════════
@app.callback(Output('perfil-graph', 'figure'), Input('pais-sel', 'value'))
def cb_perfil(c): return fig_perfil(c)

@app.callback(Output('ind-graph', 'figure'),
              [Input('ind-sel', 'value'), Input('ind-yr', 'value')])
def cb_ind(ind, yr): return fig_indicador_paises(ind, yr)

@app.callback(Output('pir-graph', 'figure'), Input('pir-yr', 'value'))
def cb_pir(yr): return fig_piramide_latam(yr)

@app.callback(Output('n3h-graph', 'figure'), Input('n3h-yr', 'value'))
def cb_n3h(yr): return fig_n3_heatmap(yr)

@app.callback(Output('br-graph', 'figure'), Input('br-yr', 'value'))
def cb_br(yr): return fig_n3_brechas(yr)

@app.callback(Output('n4-graph', 'figure'), Input('n4-yr', 'value'))
def cb_n4(yr): return fig_n4_covid(yr)

@app.callback(Output('epi-bar', 'figure'), Input('epi-yr', 'value'))
def cb_epi(yr): return fig_indicador_paises('HA.NAT.EPI', yr)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8052))
    app.run(debug=False, host='0.0.0.0', port=port)
