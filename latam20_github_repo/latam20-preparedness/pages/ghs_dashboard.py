"""
GHS Index Dashboard v4.0 — LATAM-20 · 2019 vs 2021
══════════════════════════════════════════════════════════════════════
Metanálisis Índices Pandémicos · Fuente: GHS Index CSV oficial (ghsindex.org, abril 2022)
Autor: Gisselle Rey

CAMBIOS v3.0 → v4.0:
  ✓ Tema claro CDC/Harvard unificado (theme.py)
  ✓ Muestra LATAM-20 consistente (antes header decía "24 PAÍSES")
  ✓ Rutas relativas portables (latam_common.py)
  ✓ Wilcoxon apareado robusto con tamaño de efecto (stats_utils)
  ✓ Correlaciones con IC de Fisher-z
  ✓ VulnIdx documentado como métrica EXPLORATORIA (no validada vs. outcomes)
  ✓ Exportación SVG vectorial

NIVELES JERÁRQUICOS GHS:
  N1: Overall Score (1)
  N2: 6 dominios
  N3: 37 indicadores
  N4: 96 sub-indicadores + 171 ítems binarios

MÓDULOS ANALÍTICOS:
  M1: Tipología de países (k-means + PCA)
  M2: Índice de Vulnerabilidad Pandémica (VulnIdx) — EXPLORATORIO
  M3: Convergencia sistémica 2019→2021
  M4: Matriz de oportunidades de política pública
  M5: Predictores del Overall
  M6: Trayectorias notables
"""

import os
import re
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
    LATAM_20 as LATAM, LATAM_20_SORTED, YEARS_GHS,
    SHORT, sn, data_path, NOTA_MUESTRA,
)
from theme import (
    C, LAYOUT, CARD, TS, TSS, TABLE_STYLE, DROPDOWN_STYLE,
    RISK_COLORSCALE, DIVERGING_COLORSCALE, GRAPH_CONFIG,
    rc, sc, gc, rgba,
    make_header, make_kpi, make_section_title, make_methodology_note,
)
from stats_utils import (
    wilcoxon_paired, cohens_d, correlation_ci, mean_ci,
)

# ══════════════════════════════════════════════════════════════════════
# 1. CARGA Y CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════
df_raw = pd.read_csv(data_path('ghs'))

CAT_MAP = {
    'Prevención': '1) PREVENTION OF THE EMERGENCE OR RELEASE OF PATHOGENS',
    'Detección':  "2) EARLY DETECTION & REPORTING FOR EPIDEMICS OF POTENTIAL INT'L CONCERN",
    'Respuesta':  '3) RAPID RESPONSE TO AND MITIGATION OF THE SPREAD OF AN EPIDEMIC',
    'Salud':      '4) SUFFICIENT & ROBUST HEALTH SECTOR TO TREAT THE SICK & PROTECT HEALTH WORKERS',
    'Normas':     '5) COMMITMENTS TO IMPROVING NATIONAL CAPACITY, FINANCING AND ADHERENCE TO NORMS',
    'Riesgo':     '6) OVERALL RISK ENVIRONMENT AND COUNTRY VULNERABILITY TO BIOLOGICAL THREATS',
}
CAT_ES   = list(CAT_MAP.keys())
CAT_COLS = list(CAT_MAP.values())
OVERALL  = 'OVERALL SCORE'

IND_COLS  = [c for c in df_raw.columns
             if re.match(r'^\d\.\d\) ', c) and not re.match(r'^\d\.\d\.\d', c)]
SUB_COLS  = [c for c in df_raw.columns
             if re.match(r'^\d\.\d\.\d\) ', c) and not re.match(r'^\d\.\d\.\d[a-z]', c)]
ITEM_COLS = [c for c in df_raw.columns if re.match(r'^\d\.\d\.\d[a-z]\)', c)]

# ══════════════════════════════════════════════════════════════════════
# 2. SUBSETS Y RANKINGS
# ══════════════════════════════════════════════════════════════════════
df19 = df_raw[df_raw['Year']==2019].copy()
df21 = df_raw[df_raw['Year']==2021].copy()
df19['rank_global'] = df19[OVERALL].rank(ascending=False, method='min').astype(int)
df21['rank_global'] = df21[OVERALL].rank(ascending=False, method='min').astype(int)

dl19 = df19[df19['Country'].isin(LATAM)].set_index('Country')
dl21 = df21[df21['Country'].isin(LATAM)].set_index('Country')
dg19 = df19.set_index('Country')
dg21 = df21.set_index('Country')
COMMON = [c for c in LATAM if c in dl19.index and c in dl21.index]

# ══════════════════════════════════════════════════════════════════════
# 3. ESTADÍSTICAS BASE
# ══════════════════════════════════════════════════════════════════════
def desc(s):
    s = pd.to_numeric(s, errors='coerce').dropna()
    return dict(n=len(s), mean=round(s.mean(), 2), median=round(s.median(), 2),
                sd=round(s.std(), 2), q25=round(s.quantile(.25), 1),
                q75=round(s.quantile(.75), 1), min=round(s.min(), 1),
                max=round(s.max(), 1))

# IC bootstrap para media LATAM 2021
lat21_ci = mean_ci(dl21[OVERALL].values)

ST = {'latam_21': {**desc(dl21[OVERALL]),
                   'ci_lo': lat21_ci['ci95_boot'][0],
                   'ci_hi': lat21_ci['ci95_boot'][1]},
      'latam_19': desc(dl19[OVERALL]),
      'global_21': desc(dg21[OVERALL]),
      'global_19': desc(dg19[OVERALL])}
CAT_ST = {cat: {'latam_21': desc(dl21[col]), 'global_21': desc(dg21[col]),
                'latam_19': desc(dl19[col]), 'global_19': desc(dg19[col])}
          for cat, col in CAT_MAP.items()}

# ══════════════════════════════════════════════════════════════════════
# 4. TESTS ESTADÍSTICOS ROBUSTOS
# ══════════════════════════════════════════════════════════════════════
_, sw_p19 = stats.shapiro(dl19.loc[COMMON, OVERALL])
_, sw_p21 = stats.shapiro(dl21.loc[COMMON, OVERALL])

# Wilcoxon apareado 2019 vs 2021 (stats_utils)
WX_19_21 = wilcoxon_paired(
    dl21.loc[COMMON, OVERALL].values,
    dl19.loc[COMMON, OVERALL].values,
    labels=COMMON)

# Cohen's d para LATAM vs Global 2021 (no apareado)
CD = cohens_d(dl21[OVERALL].values, dg21[OVERALL].dropna().values, paired=False)

# Correlación INFORM vs GHS con IC Fisher-z (referencia para metanálisis)
q1, q3 = dl21[OVERALL].quantile(.25), dl21[OVERALL].quantile(.75)
outliers = dl21[(dl21[OVERALL] < q1-1.5*(q3-q1)) |
                (dl21[OVERALL] > q3+1.5*(q3-q1))].index.tolist()

def pct_global(score, yr_df):
    return round(stats.percentileofscore(yr_df[OVERALL].dropna(), score, kind='rank'), 1)

cat_df = dl21[CAT_COLS].rename(columns={v: k for k, v in CAT_MAP.items()})
corr_p = cat_df.corr('pearson').round(3)
corr_s = cat_df.corr('spearman').round(3)

# ══════════════════════════════════════════════════════════════════════
# 5. TABLA MAESTRA N1
# ══════════════════════════════════════════════════════════════════════
rows = []
for c in COMMON:
    o19, o21 = dl19.loc[c, OVERALL], dl21.loc[c, OVERALL]
    cats21 = {k: round(float(dl21.loc[c, v]), 1) for k, v in CAT_MAP.items()}
    cats19 = {k: round(float(dl19.loc[c, v]), 1) for k, v in CAT_MAP.items()}
    worst = min(cats21, key=cats21.get)
    rows.append({'País': c, 'Score_2019': round(o19, 1), 'Score_2021': round(o21, 1),
                 'Δ': round(o21 - o19, 1),
                 'Rank_2019': int(dg19.loc[c, 'rank_global']),
                 'Rank_2021': int(dg21.loc[c, 'rank_global']),
                 'Rank_Δ': int(dg19.loc[c, 'rank_global']) - int(dg21.loc[c, 'rank_global']),
                 'Percentil_2021': pct_global(o21, dg21), 'Cat_Débil': worst,
                 **{f'{k}_2019': cats19[k] for k in CAT_ES},
                 **{f'{k}_2021': cats21[k] for k in CAT_ES},
                 **{f'Δ_{k}': round(cats21[k] - cats19[k], 1) for k in CAT_ES}})
master = pd.DataFrame(rows).sort_values('Score_2021', ascending=False).reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════
# 6. TABLAS N3 Y N4
# ══════════════════════════════════════════════════════════════════════
sub_rows = []
for col in SUB_COLS:
    dom = col[0]
    l19, l21 = dl19[col].mean(), dl21[col].mean()
    g21_v, g21_sd = dg21[col].mean(), dg21[col].std()
    n_lat = dl21[col].notna().sum()
    se = dl21[col].std() / np.sqrt(n_lat) if n_lat > 1 else 0
    ci95 = stats.t.ppf(0.975, df=n_lat-1) * se if n_lat > 1 else 0
    try:
        _, wp = stats.wilcoxon(dl19.loc[COMMON, col].values, dl21.loc[COMMON, col].values)
    except Exception:
        wp = np.nan
    sub_rows.append({'col': col, 'dominio': dom, 'dominio_es': CAT_ES[int(dom)-1],
                     'latam_19': round(l19, 1), 'latam_21': round(l21, 1),
                     'global_21': round(g21_v, 1), 'global_sd': round(g21_sd, 1),
                     'delta': round(l21 - l19, 1),
                     'vs_global': round(l21 - g21_v, 1), 'ci95': round(ci95, 1),
                     'wilcoxon_p': round(wp, 3) if not np.isnan(wp) else np.nan,
                     'sig': '*' if (not np.isnan(wp) and wp < 0.05) else ''})
sub_master = pd.DataFrame(sub_rows)

item_rows = []
for col in ITEM_COLS:
    dom = col[0]
    pct21 = round((dl21[col] > 0).mean() * 100, 1)
    pct19 = round((dl19[col] > 0).mean() * 100, 1)
    gp21 = round((dg21[col] > 0).mean() * 100, 1)
    item_rows.append({'col': col, 'dominio': dom, 'dominio_es': CAT_ES[int(dom)-1],
                      'pct_2019': pct19, 'pct_2021': pct21,
                      'delta': round(pct21 - pct19, 1), 'global_pct': gp21,
                      'vs_global': round(pct21 - gp21, 1)})
item_master = pd.DataFrame(item_rows)

# ══════════════════════════════════════════════════════════════════════
# 7. MÓDULO 1 — CLUSTERING K-MEANS + PCA
# ══════════════════════════════════════════════════════════════════════
X = dl21[CAT_COLS].values
X_sc = StandardScaler().fit_transform(X)

inertias = []
for k in range(2, 7):
    km_k = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_k.fit(X_sc)
    inertias.append(round(km_k.inertia_, 2))

km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
raw_labels = km3.fit_predict(X_sc)
cl_means = {k: dl21[OVERALL].values[raw_labels==k].mean() for k in range(3)}
srt = sorted(cl_means, key=cl_means.get)
name_map = {srt[0]: 'Capacidad Baja', srt[1]: 'Capacidad Media', srt[2]: 'Capacidad Alta'}
cluster_labels = {c: name_map[l] for c, l in zip(dl21.index, raw_labels)}
GHS_CLUSTER_COLORS = {'Capacidad Alta': C['green'],
                      'Capacidad Media': C['amber'],
                      'Capacidad Baja': C['red']}

pca2 = PCA(n_components=2)
X_pca = pca2.fit_transform(X_sc)
pca_df = pd.DataFrame({
    'País': list(dl21.index), 'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1],
    'Cluster': [cluster_labels[c] for c in dl21.index],
    'Overall': dl21[OVERALL].values,
    'Label': [sn(c) for c in dl21.index]
})
pc1_var = round(pca2.explained_variance_ratio_[0] * 100, 1)
pc2_var = round(pca2.explained_variance_ratio_[1] * 100, 1)
loadings = pd.DataFrame(pca2.components_.T, index=CAT_ES, columns=['PC1', 'PC2'])

cluster_profiles = {}
for cl in ['Capacidad Alta', 'Capacidad Media', 'Capacidad Baja']:
    countries_cl = [c for c, l in cluster_labels.items() if l == cl]
    cluster_profiles[cl] = {
        'países': countries_cl, 'n': len(countries_cl),
        'overall_mean': round(dl21.loc[countries_cl, OVERALL].mean(), 1),
        'overall_sd': round(dl21.loc[countries_cl, OVERALL].std(), 1),
        **{cat: round(dl21.loc[countries_cl, col].mean(), 1) for cat, col in CAT_MAP.items()}
    }

master['Cluster'] = master['País'].map(cluster_labels)

# ══════════════════════════════════════════════════════════════════════
# 8. MÓDULO 2 — VulnIdx (EXPLORATORIO)
# ══════════════════════════════════════════════════════════════════════
vuln_rows = []
for c in dl21.index:
    vals = np.array([float(dl21.loc[c, v]) for v in CAT_MAP.values()])
    overall = float(dl21.loc[c, OVERALL])
    cv = vals.std() / vals.mean()
    vuln_idx = round((100 - overall) * (1 + cv), 1)
    gap = round(vals.max() - vals.min(), 1)
    best_d = CAT_ES[np.argmax(vals)]
    worst_d = CAT_ES[np.argmin(vals)]
    vuln_rows.append({'País': c, 'Overall': overall, 'CV': round(cv, 3),
                      'VulnIdx': vuln_idx, 'Gap': gap,
                      'Cluster': cluster_labels[c],
                      'Mejor': best_d, 'Peor': worst_d,
                      'Score_Mejor': round(vals.max(), 1),
                      'Score_Peor': round(vals.min(), 1)})
vuln_df = pd.DataFrame(vuln_rows).sort_values('VulnIdx', ascending=False).reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════
# 9. MÓDULO 3 — CONVERGENCIA SISTÉMICA
# ══════════════════════════════════════════════════════════════════════
conv_rows = []
for c in COMMON:
    o19, o21 = float(dl19.loc[c, OVERALL]), float(dl21.loc[c, OVERALL])
    v19 = np.array([float(dl19.loc[c, v]) for v in CAT_MAP.values()])
    v21 = np.array([float(dl21.loc[c, v]) for v in CAT_MAP.values()])
    cv19, cv21 = v19.std() / v19.mean(), v21.std() / v21.mean()
    converge = (o21 > o19 and cv21 < cv19)
    diverge = (o21 < o19 and cv21 > cv19)
    tipo = ('Convergencia' if converge else
            'Divergencia' if diverge else
            'Mejora asimétrica' if o21 > o19 else
            'Deterioro asimétrico' if o21 < o19 else 'Estable')
    conv_rows.append({'País': c, 'Overall_2019': round(o19, 1),
                      'Overall_2021': round(o21, 1),
                      'Δ_Overall': round(o21 - o19, 1),
                      'CV_2019': round(cv19, 3), 'CV_2021': round(cv21, 3),
                      'Δ_CV': round(cv21 - cv19, 3), 'Tipo': tipo,
                      'Cluster': cluster_labels.get(c, '')})
conv_df = pd.DataFrame(conv_rows).sort_values('Δ_Overall', ascending=False)

# ══════════════════════════════════════════════════════════════════════
# 10. MÓDULO 4 — MATRIZ DE OPORTUNIDADES
# ══════════════════════════════════════════════════════════════════════
opps_rows = []
for col in ITEM_COLS:
    lat21 = round((dl21[col] > 0).mean() * 100, 1)
    lat19 = round((dl19[col] > 0).mean() * 100, 1)
    glob = round((dg21[col] > 0).mean() * 100, 1)
    brecha = round(glob - lat21, 1)
    dom = col[0]
    prioridad = ('Alta' if brecha > 20 and glob > 40 else
                 'Media' if brecha > 10 and glob > 25 else 'Baja')
    opps_rows.append({'Ítem': col, 'Dominio': CAT_ES[int(dom)-1],
                      'LATAM_2021': lat21, 'LATAM_2019': lat19, 'Global': glob,
                      'Brecha': brecha, 'Δ_LATAM': round(lat21 - lat19, 1),
                      'Prioridad': prioridad})
opps_df = pd.DataFrame(opps_rows).sort_values('Brecha', ascending=False)

# ══════════════════════════════════════════════════════════════════════
# 11. MÓDULO 5 — PREDICTORES DEL OVERALL (con IC Fisher-z)
# ══════════════════════════════════════════════════════════════════════
pred_rows = []
for cat, col in CAT_MAP.items():
    cr_s = correlation_ci(dl21[OVERALL].values, dl21[col].values, method='spearman')
    cr_p = correlation_ci(dl21[OVERALL].values, dl21[col].values, method='pearson')
    slope, intercept, _, _, se_slope = stats.linregress(dl21[col], dl21[OVERALL])
    pred_rows.append({
        'Dominio': cat,
        'Spearman_r': cr_s['r'], 'Spearman_p': cr_s['p'],
        'Spearman_CI': f'[{cr_s["ci"][0]}, {cr_s["ci"][1]}]',
        'Pearson_r': cr_p['r'], 'Pearson_p': cr_p['p'],
        'OLS_slope': round(slope, 3),
        'OLS_intercept': round(intercept, 3),
        'R2': round(cr_p['r']**2, 3)})
pred_df = pd.DataFrame(pred_rows).sort_values('Spearman_r', ascending=False)

# ══════════════════════════════════════════════════════════════════════
# 12. MÓDULO 6 — TRAYECTORIAS
# ══════════════════════════════════════════════════════════════════════
traj_rows = []
for c in COMMON:
    for cat, col in CAT_MAP.items():
        v19, v21 = float(dl19.loc[c, col]), float(dl21.loc[c, col])
        traj_rows.append({'País': c, 'Dominio': cat, 'V_2019': round(v19, 1),
                          'V_2021': round(v21, 1), 'Delta': round(v21 - v19, 1),
                          'Cluster': cluster_labels.get(c, '')})
traj_df = pd.DataFrame(traj_rows)

traj_top = {}
for cat in CAT_ES:
    df_c = traj_df[traj_df['Dominio']==cat].sort_values('Delta', ascending=False)
    traj_top[cat] = {
        'mejor': df_c.iloc[0][['País','Delta','V_2019','V_2021']].to_dict(),
        'peor':  df_c.iloc[-1][['País','Delta','V_2019','V_2021']].to_dict()
    }

# ══════════════════════════════════════════════════════════════════════
# 13. HELPERS LOCALES
# ══════════════════════════════════════════════════════════════════════
TIPO_COLORS = {'Convergencia':       C['green'],
               'Divergencia':        C['red'],
               'Mejora asimétrica':  C['amber'],
               'Deterioro asimétrico': C['orange'],
               'Estable':            C['muted']}

def score_color(v):
    if v >= 60: return C['green']
    if v >= 45: return '#22C55E'
    if v >= 35: return C['amber']
    if v >= 25: return C['orange']
    return C['red']

def delta_color(v):
    if v > 2: return C['green']
    if v > 0: return '#22C55E'
    if v > -2: return C['amber']
    return C['red']

# ══════════════════════════════════════════════════════════════════════
# 14. FIGURAS N1–N4
# ══════════════════════════════════════════════════════════════════════

def fig_bars_overview():
    m = master.sort_values('Score_2021', ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[sn(c) for c in m['País']], x=m['Score_2019'], name='2019',
        orientation='h', marker=dict(color=rgba(C['global_c'], 0.5), line_width=0)))
    fig.add_trace(go.Bar(
        y=[sn(c) for c in m['País']], x=m['Score_2021'], name='2021',
        orientation='h', marker=dict(color=C['latam'], line_width=0),
        customdata=m[['Δ','Rank_2021','Percentil_2021','Cat_Débil','Cluster']].values,
        hovertemplate=('<b>%{y}</b> 2021: %{x:.1f}<br>Δ: %{customdata[0]:+.1f}<br>'
                       'Rank: #%{customdata[1]}/195<br>Pct: %{customdata[2]}%<br>'
                       'Débil: %{customdata[3]}<br>Cluster: %{customdata[4]}<extra></extra>')))
    fig.add_vline(x=ST['global_21']['mean'], line_dash='dot', line_color=C['amber'],
                  annotation_text=f'Media global: {ST["global_21"]["mean"]}',
                  annotation_font=dict(size=9, color=C['amber']))
    fig.add_vline(x=ST['global_21']['median'], line_dash='dash', line_color=C['muted'],
                  annotation_text=f'Mediana: {ST["global_21"]["median"]}',
                  annotation_font=dict(size=9, color=C['muted']),
                  annotation_position='bottom right')
    fig.update_layout(**{**LAYOUT, 'height': 620}, barmode='overlay',
        xaxis=dict(title='Puntaje GHS (0–100)', range=[0, 78], gridcolor=C['grid']),
        yaxis=dict(gridcolor=C['grid']),
        legend=dict(orientation='h', y=1.03, bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=100, r=40, t=30, b=40))
    return fig


def fig_boxplot():
    no_lat = dg21[~dg21.index.isin(LATAM)][OVERALL].dropna()
    fig = go.Figure()
    fig.add_trace(go.Box(y=no_lat, name=f'Resto del mundo (n={len(no_lat)})',
        marker=dict(color=C['muted'], size=3), line=dict(color=C['muted']),
        boxmean='sd', fillcolor=rgba(C['global_c'], 0.15)))
    fig.add_trace(go.Box(y=dl21[OVERALL], name=f'LATAM-20 (n={len(dl21)})',
        marker=dict(color=C['latam'], size=5), line=dict(color=C['latam']),
        boxmean='sd', fillcolor=rgba(C['latam'], 0.15),
        boxpoints='all', pointpos=-1.6, text=list(dl21.index),
        hovertemplate='<b>%{text}</b>: %{y:.1f}<extra></extra>'))
    fig.add_annotation(x=0.5, y=1.09, xref='paper', yref='paper',
        text=(f'Shapiro-Wilk: p₂₀₁₉={sw_p19:.3f}, p₂₀₂₁={sw_p21:.3f} · '
              f'Wilcoxon apareado: W={WX_19_21["W"]:.0f}, p={WX_19_21["p"]:.3f}, '
              f'r={WX_19_21["r_rosenthal"]} · Cohen\'s d={CD:+.3f}'),
        showarrow=False, font=dict(size=9, color=C['muted']))
    fig.update_layout(**{**LAYOUT, 'height': 380},
        yaxis=dict(title='Overall 2021', gridcolor=C['grid']),
        legend=dict(orientation='h', y=1.03, bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=60, r=20, t=75, b=30))
    return fig


def fig_heatmap_scores():
    m = master.sort_values('Score_2021', ascending=False)
    z = np.array([[dl21.loc[c, CAT_MAP[k]] for k in CAT_ES] for c in m['País']])
    fig = go.Figure(go.Heatmap(
        z=z, x=CAT_ES, y=[sn(c) for c in m['País']],
        text=np.round(z, 1).astype(str), texttemplate='%{text}',
        textfont=dict(size=9, color=C['ink']),
        colorscale=[[0, C['red']], [0.4, C['amber']], [1, C['green']]],
        zmin=0, zmax=80,
        hovertemplate='<b>%{y}</b><br>%{x}: %{z:.1f}<extra></extra>',
        colorbar=dict(title=dict(text='Score', font=dict(color=C['ink'])),
                      tickfont=dict(size=9, color=C['ink']))))
    fig.update_layout(**{**LAYOUT, 'height': 580},
        margin=dict(l=100, r=20, t=60, b=20),
        xaxis=dict(side='top', tickangle=-20, gridcolor=C['grid']),
        yaxis=dict(gridcolor=C['grid']))
    return fig


def fig_heatmap_delta():
    m = master.sort_values('Score_2021', ascending=False)
    z = np.array([[master.loc[master['País']==c, f'Δ_{k}'].values[0] for k in CAT_ES]
                  for c in m['País']])
    text = np.where(z>0, '+'+np.round(z, 1).astype(str), np.round(z, 1).astype(str))
    fig = go.Figure(go.Heatmap(
        z=z, x=CAT_ES, y=[sn(c) for c in m['País']],
        text=text, texttemplate='%{text}', textfont=dict(size=9, color=C['ink']),
        colorscale='RdYlGn', zmid=0, zmin=-15, zmax=15,
        hovertemplate='<b>%{y}</b><br>%{x}: %{z:+.1f}<extra></extra>',
        colorbar=dict(title=dict(text='Δ pts', font=dict(color=C['ink'])),
                      tickfont=dict(size=9, color=C['ink']))))
    fig.update_layout(**{**LAYOUT, 'height': 580},
        margin=dict(l=100, r=20, t=60, b=20),
        xaxis=dict(side='top', tickangle=-20, gridcolor=C['grid']),
        yaxis=dict(gridcolor=C['grid']))
    return fig


def fig_cat_bars():
    l19 = [CAT_ST[c]['latam_19']['mean'] for c in CAT_ES]
    l21 = [CAT_ST[c]['latam_21']['mean'] for c in CAT_ES]
    g21 = [CAT_ST[c]['global_21']['mean'] for c in CAT_ES]
    sd_l = [CAT_ST[c]['latam_21']['sd'] for c in CAT_ES]
    fig = go.Figure()
    fig.add_trace(go.Bar(name='LATAM 2019', x=CAT_ES, y=l19,
        marker=dict(color=rgba(C['global_c'], 0.45), line_width=0)))
    fig.add_trace(go.Bar(name='LATAM 2021', x=CAT_ES, y=l21,
        marker=dict(color=C['latam'], line_width=0),
        error_y=dict(type='data', array=sd_l, visible=True,
                     color=rgba(C['latam'], 0.6), thickness=1.5)))
    fig.add_trace(go.Bar(name='Global 2021', x=CAT_ES, y=g21,
        marker=dict(color=rgba(C['muted'], 0.55), line_width=0)))
    for i, (cat, d) in enumerate(zip(CAT_ES, [round(a-b, 1) for a, b in zip(l21, l19)])):
        fig.add_annotation(x=cat, y=max(l21[i], g21[i])+sd_l[i]+4,
            text=f'{d:+.1f}', showarrow=False,
            font=dict(size=10, color=C['green'] if d>0 else C['red']))
    fig.update_layout(**{**LAYOUT, 'height': 420}, barmode='group',
        yaxis=dict(title='Promedio (0–100)', range=[0, 82], gridcolor=C['grid']),
        legend=dict(orientation='h', y=-0.15, bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=55, r=20, t=50, b=20))
    return fig


def fig_corr_matrix():
    fig = make_subplots(1, 2, subplot_titles=['Pearson r', 'Spearman ρ'])
    for i, (corr, _) in enumerate([(corr_p, 'P'), (corr_s, 'S')], 1):
        z = corr.values
        fig.add_trace(go.Heatmap(z=z, x=CAT_ES, y=CAT_ES,
            colorscale='RdBu_r', zmid=0, zmin=-1, zmax=1,
            text=np.round(z, 2), texttemplate='%{text}',
            textfont=dict(size=10, color=C['ink']),
            showscale=(i==2),
            colorbar=dict(x=1.02, tickfont=dict(size=9, color=C['ink']))),
            row=1, col=i)
    fig.update_annotations(font=dict(color=C['ink'], size=11))
    fig.update_layout(**{**LAYOUT, 'height': 380},
        margin=dict(l=70, r=60, t=55, b=70))
    fig.update_xaxes(tickangle=-30, gridcolor=C['grid'])
    fig.update_yaxes(gridcolor=C['grid'])
    return fig


def fig_scatter(cat='Detección'):
    x = master[f'{cat}_2021'].values
    y = master['Score_2021'].values
    cr_p = correlation_ci(x, y, method='pearson')
    cr_s = correlation_ci(x, y, method='spearman')
    slope, intercept, *_ = stats.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 80)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=slope*xs+intercept, mode='lines',
        line=dict(color=C['amber'], width=1.5, dash='dash'), showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers+text',
        text=[sn(c) for c in master['País']], textposition='top center',
        textfont=dict(size=9, color=C['ink']),
        marker=dict(size=12, color=master['Score_2021'].values,
            colorscale=[[0, C['red']], [0.5, C['amber']], [1, C['green']]],
            showscale=True, line=dict(color='white', width=1),
            colorbar=dict(title=dict(text='Overall', font=dict(color=C['ink'])),
                          tickfont=dict(size=9, color=C['ink']))),
        customdata=master[['País', f'Δ_{cat}', 'Δ']].values,
        hovertemplate=('<b>%{customdata[0]}</b><br>%{x:.1f} → Overall: %{y:.1f}<br>'
                       'Δ cat: %{customdata[1]:+.1f}<extra></extra>'), showlegend=False))
    title = (f'Overall vs {cat} (n={cr_p["n"]}) · '
             f'Pearson r={cr_p["r"]} [IC95% {cr_p["ci"][0]}, {cr_p["ci"][1]}], p={cr_p["p"]} · '
             f'Spearman ρ={cr_s["r"]}, p={cr_s["p"]}')
    fig.update_layout(**{**LAYOUT, 'height': 460},
        title=dict(text=title, font=dict(size=10, color=C['muted'])),
        xaxis=dict(title=f'{cat} 2021', gridcolor=C['grid']),
        yaxis=dict(title='Overall 2021', gridcolor=C['grid']),
        margin=dict(l=60, r=80, t=70, b=50))
    return fig


def fig_radar(countries):
    pal = [C['latam'], C['green'], C['amber'], C['purple'], C['blue']]
    theta = CAT_ES + [CAT_ES[0]]
    fig = go.Figure()
    for i, country in enumerate(countries[:5]):
        if country not in dl21.index: continue
        col = pal[i % len(pal)]
        v21 = ([float(dl21.loc[country, CAT_MAP[k]]) for k in CAT_ES]
               + [float(dl21.loc[country, CAT_MAP[CAT_ES[0]]])])
        v19 = ([float(dl19.loc[country, CAT_MAP[k]]) for k in CAT_ES]
               + [float(dl19.loc[country, CAT_MAP[CAT_ES[0]]])])
        fig.add_trace(go.Scatterpolar(r=v21, theta=theta, name=f'{sn(country)} 2021',
            line=dict(color=col, width=2.5), fill='toself', fillcolor=rgba(col, 0.12)))
        fig.add_trace(go.Scatterpolar(r=v19, theta=theta, name=f'{sn(country)} 2019',
            line=dict(color=col, width=1.2, dash='dot'), fill=None, opacity=0.55))
    fig.update_layout(**{**LAYOUT, 'height': 520},
        polar=dict(bgcolor=C['white'],
                   radialaxis=dict(visible=True, range=[0, 100],
                                   tickfont=dict(size=8, color=C['muted']),
                                   gridcolor=C['grid'], linecolor=C['border']),
                   angularaxis=dict(tickfont=dict(size=10, color=C['ink']),
                                    gridcolor=C['grid'], linecolor=C['border'])),
        legend=dict(font=dict(size=9), bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=60, r=60, t=50, b=50))
    return fig


def fig_rank_change():
    m = master.sort_values('Rank_Δ', ascending=True)
    fig = go.Figure(go.Bar(x=m['Rank_Δ'], y=[sn(c) for c in m['País']], orientation='h',
        marker=dict(color=[delta_color(v) for v in m['Rank_Δ']], line_width=0),
        customdata=m[['Rank_2019', 'Rank_2021']].values,
        hovertemplate=('<b>%{y}</b><br>2019: #%{customdata[0]} → 2021: #%{customdata[1]}<br>'
                       'Cambio: %{x} puestos<extra></extra>')))
    fig.add_vline(x=0, line_color=C['muted'], line_width=1)
    fig.update_layout(**{**LAYOUT, 'height': 560},
        margin=dict(l=100, r=40, t=40, b=50),
        xaxis=dict(title='Cambio ranking global (+ = mejora)', gridcolor=C['grid']),
        yaxis=dict(gridcolor=C['grid']))
    return fig


def fig_percentil():
    m = master.sort_values('Percentil_2021', ascending=False)
    fig = go.Figure(go.Bar(x=[sn(c) for c in m['País']], y=m['Percentil_2021'],
        marker=dict(color=[score_color(v) for v in m['Score_2021']], line_width=0),
        customdata=m[['Score_2021', 'Rank_2021']].values,
        hovertemplate=('<b>%{x}</b><br>Percentil: %{y}%<br>Score: %{customdata[0]}<br>'
                       'Rank: #%{customdata[1]}/195<extra></extra>')))
    fig.add_hline(y=50, line_dash='dot', line_color=C['amber'],
                  annotation_text='Percentil 50',
                  annotation_font=dict(size=9, color=C['amber']))
    fig.update_layout(**{**LAYOUT, 'height': 360},
        margin=dict(l=60, r=20, t=40, b=100),
        xaxis=dict(tickangle=-40, gridcolor=C['grid']),
        yaxis=dict(title='Percentil global 2021', gridcolor=C['grid']))
    return fig


def fig_indicadores(country='Mexico'):
    if country not in dl21.index:
        return go.Figure()
    rows_i = []
    for col in IND_COLS:
        v21 = round(float(dl21.loc[country, col]), 1)
        v19 = round(float(dl19.loc[country, col]), 1)
        g21 = round(float(dg21[col].mean()), 1)
        rows_i.append({'Indicador': col[:58]+'…' if len(col) > 58 else col,
                       'v2019': v19, 'v2021': v21, 'global': g21,
                       'delta': round(v21-v19, 1), 'vs_g': round(v21-g21, 1)})
    df_i = pd.DataFrame(rows_i).sort_values('v2021', ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df_i['Indicador'], x=df_i['v2019'], name='2019',
        orientation='h', marker=dict(color=rgba(C['global_c'], 0.45), line_width=0)))
    fig.add_trace(go.Bar(y=df_i['Indicador'], x=df_i['v2021'], name='2021',
        orientation='h', marker=dict(color=C['latam'], line_width=0),
        customdata=df_i[['delta', 'global', 'vs_g']].values,
        hovertemplate=('%{y}<br>2021: %{x}<br>Δ: %{customdata[0]:+.1f}<br>'
                       'Media global: %{customdata[1]}<br>'
                       'vs Global: %{customdata[2]:+.1f}<extra></extra>')))
    fig.add_trace(go.Scatter(x=df_i['global'], y=df_i['Indicador'], mode='markers',
        name='Media global',
        marker=dict(symbol='line-ns', color=C['amber'], size=10, line_width=2)))
    fig.update_layout(**{**LAYOUT, 'height': 900}, barmode='overlay',
        title=dict(text=f'{country} — 37 indicadores vs. media global',
                   font=dict(size=12, color=C['ink'])),
        xaxis=dict(range=[0, 105], gridcolor=C['grid']),
        yaxis=dict(tickfont=dict(size=9), gridcolor=C['grid']),
        legend=dict(orientation='h', y=1.02, bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=310, r=20, t=60, b=30))
    return fig


def fig_sub_graph(dom, view):
    df_d = sub_master[sub_master['dominio']==dom].copy()
    if view == 'heat':
        countries = master.sort_values('Score_2021', ascending=False)['País'].tolist()
        z = np.array([[dl21.loc[c, row['col']] for _, row in df_d.iterrows()]
                      for c in countries])
        labels = [r['col'][:50]+'…' if len(r['col']) > 50 else r['col']
                  for _, r in df_d.iterrows()]
        fig = go.Figure(go.Heatmap(
            z=z, x=labels, y=[sn(c) for c in countries],
            colorscale=[[0, C['red']], [0.4, C['amber']], [1, C['green']]],
            zmin=0, zmax=100,
            text=np.round(z, 0).astype(int).astype(str), texttemplate='%{text}',
            textfont=dict(size=9, color=C['ink']),
            hovertemplate='<b>%{y}</b><br>%{x}: %{z:.1f}<extra></extra>',
            colorbar=dict(title=dict(text='Score', font=dict(color=C['ink'])),
                          tickfont=dict(size=9, color=C['ink']))))
        fig.update_layout(**{**LAYOUT, 'height': 580},
            margin=dict(l=100, r=20, t=140, b=20),
            xaxis=dict(side='top', tickangle=-35, tickfont=dict(size=8), gridcolor=C['grid']),
            yaxis=dict(gridcolor=C['grid']))
    elif view == 'delta':
        df_d = df_d.sort_values('delta')
        labels = [r['col'][:55]+'…' if len(r['col']) > 55 else r['col']
                  for _, r in df_d.iterrows()]
        fig = go.Figure(go.Bar(y=labels, x=df_d['delta'], orientation='h',
            marker=dict(color=[delta_color(v) for v in df_d['delta']], line_width=0),
            customdata=df_d[['latam_19', 'latam_21', 'wilcoxon_p']].values,
            hovertemplate=('%{y}<br>2019: %{customdata[0]} → 2021: %{customdata[1]}<br>'
                           'Δ: %{x:+.1f}<br>Wilcoxon p: %{customdata[2]:.3f}<extra></extra>')))
        fig.add_vline(x=0, line_color=C['muted'], line_width=1)
        fig.update_layout(**{**LAYOUT, 'height': max(400, len(df_d)*28)},
            margin=dict(l=340, r=40, t=30, b=40),
            xaxis=dict(title='Cambio 2019→2021', gridcolor=C['grid']),
            yaxis=dict(tickfont=dict(size=9), gridcolor=C['grid']))
    else:  # vsg
        df_d = df_d.sort_values('vs_global')
        labels = [r['col'][:55]+'…' if len(r['col']) > 55 else r['col']
                  for _, r in df_d.iterrows()]
        fig = go.Figure(go.Bar(y=labels, x=df_d['vs_global'], orientation='h',
            marker=dict(color=[C['green'] if v>=0 else C['red'] for v in df_d['vs_global']], line_width=0),
            customdata=df_d[['latam_21', 'global_21', 'delta', 'sig']].values,
            hovertemplate=('%{y}<br>LATAM: %{customdata[0]}<br>Global: %{customdata[1]}<br>'
                           'Δ: %{customdata[2]:+.1f}%{customdata[3]}<br>'
                           'vs Global: %{x:+.1f}<extra></extra>')))
        fig.add_vline(x=0, line_color=C['muted'], line_width=1)
        fig.update_layout(**{**LAYOUT, 'height': max(400, len(df_d)*28)},
            margin=dict(l=340, r=40, t=30, b=40),
            xaxis=dict(title='LATAM vs. Media global', gridcolor=C['grid']),
            yaxis=dict(tickfont=dict(size=9), gridcolor=C['grid']))
    return fig


def fig_items_coverage(dom, ordenar):
    df_d = item_master[item_master['dominio']==dom].copy().sort_values(ordenar, ascending=True)
    labels = [r['col'][:65]+'…' if len(r['col']) > 65 else r['col']
              for _, r in df_d.iterrows()]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=labels, x=df_d['pct_2019'], name='LATAM 2019',
        orientation='h', marker=dict(color=rgba(C['global_c'], 0.5), line_width=0)))
    fig.add_trace(go.Bar(y=labels, x=df_d['pct_2021'], name='LATAM 2021',
        orientation='h', marker=dict(color=C['latam'], line_width=0),
        customdata=df_d[['delta', 'global_pct', 'vs_global']].values,
        hovertemplate=('%{y}<br>LATAM 2021: %{x}%<br>Δ: %{customdata[0]:+.1f}%<br>'
                       'Global: %{customdata[1]}%<br>'
                       'vs Global: %{customdata[2]:+.1f}%<extra></extra>')))
    fig.add_trace(go.Scatter(x=df_d['global_pct'], y=labels, mode='markers',
        name='Media global',
        marker=dict(symbol='line-ns', color=C['amber'], size=10, line_width=2)))
    fig.update_layout(**{**LAYOUT, 'height': max(420, len(df_d)*22)}, barmode='overlay',
        xaxis=dict(title='% países con cumplimiento (score>0)',
                   range=[0, 108], gridcolor=C['grid']),
        yaxis=dict(tickfont=dict(size=8), gridcolor=C['grid']),
        legend=dict(orientation='h', y=1.02, bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=400, r=20, t=50, b=30))
    return fig


# ══════════════════════════════════════════════════════════════════════
# 15. FIGURAS MÓDULOS M1–M6
# ══════════════════════════════════════════════════════════════════════

def fig_m1_pca():
    fig = go.Figure()
    for cl, col in GHS_CLUSTER_COLORS.items():
        df_cl = pca_df[pca_df['Cluster']==cl]
        if len(df_cl) == 0: continue
        fig.add_trace(go.Scatter(
            x=df_cl['PC1'], y=df_cl['PC2'], mode='markers+text', name=cl,
            text=[sn(c) for c in df_cl['País']], textposition='top center',
            textfont=dict(size=9, color=C['ink']),
            marker=dict(size=df_cl['Overall'].values/3+6, color=col,
                        line=dict(color='white', width=1), opacity=0.9),
            customdata=df_cl[['País', 'Overall', 'Cluster']].values,
            hovertemplate=('<b>%{customdata[0]}</b><br>Overall: %{customdata[1]}<br>'
                           'PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>')))
    scale = 2.5
    for cat, row in loadings.iterrows():
        fig.add_annotation(x=row['PC1']*scale, y=row['PC2']*scale,
            text=cat, showarrow=True, arrowhead=2, arrowsize=1,
            arrowcolor=C['amber'], font=dict(size=9, color=C['amber']), ax=0, ay=0)
    fig.update_layout(**{**LAYOUT, 'height': 520},
        title=dict(text=f'PCA de 6 dominios · PC1={pc1_var}% · PC2={pc2_var}% de varianza explicada',
                   font=dict(size=11, color=C['muted'])),
        xaxis=dict(title=f'PC1 ({pc1_var}% var) — capacidad sanitaria general',
                   gridcolor=C['grid'], zeroline=True, zerolinecolor=C['border']),
        yaxis=dict(title=f'PC2 ({pc2_var}% var) — Riesgo/entorno vs Normas',
                   gridcolor=C['grid'], zeroline=True, zerolinecolor=C['border']),
        legend=dict(orientation='h', y=1.03, bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=60, r=60, t=60, b=60))
    return fig


def fig_m1_radar_clusters():
    theta = CAT_ES + [CAT_ES[0]]
    fig = go.Figure()
    for cl, col in GHS_CLUSTER_COLORS.items():
        prof = cluster_profiles[cl]
        vals = [prof[cat] for cat in CAT_ES] + [prof[CAT_ES[0]]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=theta, name=f'{cl} (n={prof["n"]})',
            line=dict(color=col, width=2.5),
            fill='toself', fillcolor=rgba(col, 0.12),
            hovertemplate='%{theta}: %{r:.1f}<extra></extra>'))
    fig.update_layout(**{**LAYOUT, 'height': 480},
        polar=dict(bgcolor=C['white'],
                   radialaxis=dict(visible=True, range=[0, 100],
                                   tickfont=dict(size=8, color=C['muted']),
                                   gridcolor=C['grid'], linecolor=C['border']),
                   angularaxis=dict(tickfont=dict(size=10, color=C['ink']),
                                    gridcolor=C['grid'], linecolor=C['border'])),
        legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=60, r=60, t=50, b=50))
    return fig


def fig_m1_elbow():
    fig = go.Figure(go.Scatter(x=list(range(2, 7)), y=inertias,
        mode='lines+markers',
        line=dict(color=C['latam'], width=2),
        marker=dict(size=9, color=C['latam']),
        hovertemplate='k=%{x}<br>Inercia=%{y:.1f}<extra></extra>'))
    fig.add_vline(x=3, line_dash='dot', line_color=C['green'],
                  annotation_text='k=3 óptimo',
                  annotation_font=dict(size=9, color=C['green']))
    fig.update_layout(**{**LAYOUT, 'height': 280},
        xaxis=dict(title='Número de clusters (k)',
                   tickvals=list(range(2, 7)), gridcolor=C['grid']),
        yaxis=dict(title='Inercia (WCSS)', gridcolor=C['grid']),
        margin=dict(l=60, r=20, t=30, b=50))
    return fig


def fig_m2_vuln():
    fig = go.Figure(go.Scatter(
        x=vuln_df['Overall'], y=vuln_df['CV'], mode='markers+text',
        text=[sn(c) for c in vuln_df['País']], textposition='top center',
        textfont=dict(size=9, color=C['ink']),
        marker=dict(size=vuln_df['VulnIdx'].values/6+8,
                    color=vuln_df['VulnIdx'].values,
                    colorscale=[[0, C['green']], [0.4, C['amber']], [1, C['red']]],
                    showscale=True, line=dict(color='white', width=1),
                    colorbar=dict(title=dict(text='VulnIdx', font=dict(color=C['ink'])),
                                  tickfont=dict(size=9, color=C['ink']))),
        customdata=vuln_df[['VulnIdx', 'Gap', 'Mejor', 'Peor']].values,
        hovertemplate=('<b>%{text}</b><br>Overall: %{x:.1f}<br>CV: %{y:.3f}<br>'
                       'VulnIdx: %{customdata[0]}<br>Gap dominios: %{customdata[1]}<br>'
                       'Mejor: %{customdata[2]}<br>Peor: %{customdata[3]}<extra></extra>')))
    mx = dl21[OVERALL].mean(); mCV = vuln_df['CV'].mean()
    fig.add_vline(x=mx, line_dash='dot', line_color=C['muted'], line_width=1)
    fig.add_hline(y=mCV, line_dash='dot', line_color=C['muted'], line_width=1)
    fig.add_annotation(x=25, y=0.47, text='Alta vulnerabilidad\n(bajo Overall + alta dispersión)',
                       showarrow=False, font=dict(size=9, color=rgba(C['red'], 0.7)))
    fig.add_annotation(x=55, y=0.08, text='Baja vulnerabilidad\n(alto Overall + baja dispersión)',
                       showarrow=False, font=dict(size=9, color=rgba(C['green'], 0.8)))
    fig.update_layout(**{**LAYOUT, 'height': 520},
        title=dict(text='VulnIdx EXPLORATORIO · tamaño = magnitud · '
                        'no validado contra outcomes externos',
                   font=dict(size=10, color=C['muted'])),
        xaxis=dict(title='Overall Score 2021', gridcolor=C['grid']),
        yaxis=dict(title='Coeficiente de variación entre dominios (CV)',
                   gridcolor=C['grid']),
        margin=dict(l=60, r=80, t=70, b=60))
    return fig


def fig_m2_bars():
    m = vuln_df.sort_values('VulnIdx', ascending=True)
    colors = [GHS_CLUSTER_COLORS[c] for c in m['Cluster']]
    fig = go.Figure(go.Bar(y=[sn(c) for c in m['País']], x=m['VulnIdx'],
        orientation='h', marker=dict(color=colors, line_width=0),
        customdata=m[['Overall', 'CV', 'Gap', 'Mejor', 'Peor']].values,
        hovertemplate=('<b>%{y}</b><br>VulnIdx: %{x}<br>'
                       'Overall: %{customdata[0]}<br>CV: %{customdata[1]}<br>'
                       'Gap: %{customdata[2]}<br>Mejor: %{customdata[3]}<br>'
                       'Peor: %{customdata[4]}<extra></extra>')))
    fig.update_layout(**{**LAYOUT, 'height': 560},
        margin=dict(l=100, r=40, t=30, b=40),
        xaxis=dict(title='VulnIdx — mayor = más vulnerable', gridcolor=C['grid']),
        yaxis=dict(gridcolor=C['grid']))
    return fig


def fig_m3_convergence():
    fig = go.Figure()
    for tipo, col in TIPO_COLORS.items():
        df_t = conv_df[conv_df['Tipo']==tipo]
        if len(df_t) == 0: continue
        fig.add_trace(go.Scatter(
            x=df_t['Δ_CV'], y=df_t['Δ_Overall'], mode='markers+text',
            name=f'{tipo} (n={len(df_t)})',
            text=[sn(c) for c in df_t['País']], textposition='top center',
            textfont=dict(size=9, color=C['ink']),
            marker=dict(size=12, color=col, line=dict(color='white', width=1), opacity=0.9),
            customdata=df_t[['País', 'Overall_2021', 'CV_2019', 'CV_2021']].values,
            hovertemplate=('<b>%{customdata[0]}</b><br>Overall 2021: %{customdata[1]}<br>'
                           'CV 2019→2021: %{customdata[2]:.3f}→%{customdata[3]:.3f}<br>'
                           'Δ Overall: %{y:+.1f}<br>Δ CV: %{x:+.3f}<extra></extra>')))
    fig.add_vline(x=0, line_color=C['muted'], line_width=1)
    fig.add_hline(y=0, line_color=C['muted'], line_width=1)
    fig.add_annotation(x=-0.09, y=4.5, text='Convergencia\n(↑ Overall, ↓ CV)',
        showarrow=False, font=dict(size=9, color=rgba(C['green'], 0.8)))
    fig.add_annotation(x=0.04, y=-6, text='Divergencia\n(↓ Overall, ↑ CV)',
        showarrow=False, font=dict(size=9, color=rgba(C['red'], 0.8)))
    fig.update_layout(**{**LAYOUT, 'height': 520},
        title=dict(text='Convergencia sistémica 2019→2021 · cuadrante superior izq. = convergencia real',
                   font=dict(size=10, color=C['muted'])),
        xaxis=dict(title='Δ Coeficiente de variación (negativo = mayor equilibrio)',
                   gridcolor=C['grid'], zeroline=True, zerolinecolor=C['border']),
        yaxis=dict(title='Δ Overall Score (positivo = mejora)',
                   gridcolor=C['grid'], zeroline=True, zerolinecolor=C['border']),
        legend=dict(orientation='h', y=1.03, bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=70, r=20, t=60, b=70))
    return fig


def fig_m4_matrix():
    pri_colors = {'Alta': C['red'], 'Media': C['amber'], 'Baja': rgba(C['muted'], 0.4)}
    fig = go.Figure()
    for pri, col in pri_colors.items():
        df_p = opps_df[opps_df['Prioridad']==pri].head(30 if pri != 'Alta' else 100)
        if len(df_p) == 0: continue
        fig.add_trace(go.Scatter(
            x=df_p['Global'], y=df_p['Brecha'], mode='markers',
            name=f'Prioridad {pri} (n={len(opps_df[opps_df["Prioridad"]==pri])})',
            marker=dict(size=7 if pri=='Baja' else 10, color=col,
                        line_width=0, opacity=0.85),
            customdata=df_p[['Ítem', 'LATAM_2021', 'Global', 'Δ_LATAM', 'Dominio']].values,
            hovertemplate=('<b>%{customdata[0]}</b><br>LATAM: %{customdata[1]}%<br>'
                           'Global: %{customdata[2]}%<br>Brecha: %{y:.1f}pp<br>'
                           'Δ LATAM: %{customdata[3]:+.1f}%<br>'
                           'Dominio: %{customdata[4]}<extra></extra>')))
    fig.add_vline(x=40, line_dash='dot', line_color=C['muted'], line_width=1,
        annotation_text='Factibilidad umbral',
        annotation_font=dict(size=8, color=C['muted']))
    fig.add_hline(y=20, line_dash='dot', line_color=C['muted'], line_width=1,
        annotation_text='Impacto umbral',
        annotation_font=dict(size=8, color=C['muted']))
    fig.add_annotation(x=72, y=52, text='Cuadrante de\nprioridad máxima',
        showarrow=False, font=dict(size=9, color=rgba(C['red'], 0.85)))
    fig.update_layout(**{**LAYOUT, 'height': 520},
        title=dict(text='Matriz de oportunidades · Factibilidad (% global) vs Impacto (brecha LATAM)',
                   font=dict(size=10, color=C['muted'])),
        xaxis=dict(title='Factibilidad — % países globales con cumplimiento',
                   gridcolor=C['grid']),
        yaxis=dict(title='Impacto — Brecha LATAM vs Global (pp)', gridcolor=C['grid']),
        legend=dict(orientation='h', y=1.03, bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=70, r=20, t=60, b=70))
    return fig


def fig_m5_predictors():
    m = pred_df.sort_values('Spearman_r')
    colors = [score_color(v*100) for v in m['Spearman_r']]
    fig = go.Figure(go.Bar(y=m['Dominio'], x=m['Spearman_r'], orientation='h',
        marker=dict(color=colors, line_width=0),
        customdata=m[['Pearson_r', 'R2', 'OLS_slope', 'Spearman_p', 'Spearman_CI']].values,
        hovertemplate=('<b>%{y}</b><br>Spearman ρ: %{x:.3f} %{customdata[4]}<br>'
                       'Pearson r: %{customdata[0]:.3f}<br>'
                       'R²: %{customdata[1]:.3f}<br>'
                       'OLS slope: %{customdata[2]:.3f}<br>'
                       'p: %{customdata[3]:.4f}<extra></extra>')))
    fig.add_vline(x=0.7, line_dash='dot', line_color=C['amber'], line_width=1,
        annotation_text='r=0.7', annotation_font=dict(size=9, color=C['amber']))
    fig.update_layout(**{**LAYOUT, 'height': 340},
        margin=dict(l=100, r=60, t=50, b=50),
        title=dict(text='Predictores del Overall — Spearman ρ con IC Fisher-z (n=20)',
                   font=dict(size=10, color=C['muted'])),
        xaxis=dict(title='Spearman ρ', range=[0.5, 1.0], gridcolor=C['grid']),
        yaxis=dict(gridcolor=C['grid']))
    return fig


def fig_m5_scatter_all():
    fig = make_subplots(2, 3, subplot_titles=CAT_ES)
    pal = [C['latam'], C['green'], C['amber'], C['purple'], C['blue'], C['teal']]
    for idx, (cat, col) in enumerate(CAT_MAP.items()):
        r, c = divmod(idx, 3); r += 1; c += 1
        x = dl21[col].values; y = dl21[OVERALL].values
        slope, intercept, *_ = stats.linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 50)
        rho, _ = stats.spearmanr(x, y)
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers',
            marker=dict(size=7, color=pal[idx], opacity=0.85, line_width=0),
            showlegend=False, text=list(dl21.index),
            hovertemplate='<b>%{text}</b><br>%{x:.1f} → %{y:.1f}<extra></extra>'),
            row=r, col=c)
        fig.add_trace(go.Scatter(x=xs, y=slope*xs+intercept, mode='lines',
            line=dict(color=pal[idx], width=1.5, dash='dash'), showlegend=False),
            row=r, col=c)
        fig.add_annotation(x=x.max()*0.7, y=y.min()*1.05, text=f'ρ={rho:.2f}',
            showarrow=False, font=dict(size=9, color=pal[idx]), row=r, col=c)
    fig.update_layout(**{**LAYOUT, 'height': 560}, showlegend=False,
        margin=dict(l=50, r=20, t=60, b=40))
    fig.update_xaxes(gridcolor=C['grid'], tickfont=dict(size=8))
    fig.update_yaxes(gridcolor=C['grid'], tickfont=dict(size=8))
    fig.update_annotations(font=dict(color=C['ink'], size=10))
    return fig


def fig_m6_heatmap():
    paises = master.sort_values('Score_2021', ascending=False)['País'].tolist()
    z = np.array([[traj_df[(traj_df['País']==p) &
                           (traj_df['Dominio']==cat)]['Delta'].values[0]
                   for cat in CAT_ES] for p in paises])
    text = np.where(z>0, '+'+np.round(z, 1).astype(str), np.round(z, 1).astype(str))
    fig = go.Figure(go.Heatmap(z=z, x=CAT_ES, y=[sn(p) for p in paises],
        text=text, texttemplate='%{text}',
        textfont=dict(size=9, color=C['ink']),
        colorscale='RdYlGn', zmid=0, zmin=-20, zmax=20,
        hovertemplate='<b>%{y}</b><br>%{x}: %{z:+.1f}<extra></extra>',
        colorbar=dict(title=dict(text='Δ', font=dict(color=C['ink'])),
                      tickfont=dict(size=9, color=C['ink']))))
    fig.update_layout(**{**LAYOUT, 'height': 600},
        margin=dict(l=100, r=20, t=60, b=20),
        xaxis=dict(side='top', tickangle=-20, gridcolor=C['grid']),
        yaxis=dict(gridcolor=C['grid']))
    return fig


def fig_m6_notable():
    cats_list = []; paises_m = []; vals_m = []; paises_c = []; vals_c = []
    for cat in CAT_ES:
        t = traj_top[cat]
        cats_list.append(cat)
        paises_m.append(f'{sn(t["mejor"]["País"])} ({t["mejor"]["V_2019"]:.0f}→{t["mejor"]["V_2021"]:.0f})')
        vals_m.append(t['mejor']['Delta'])
        paises_c.append(f'{sn(t["peor"]["País"])} ({t["peor"]["V_2019"]:.0f}→{t["peor"]["V_2021"]:.0f})')
        vals_c.append(t['peor']['Delta'])
    fig = make_subplots(1, 2, subplot_titles=['Mayor mejora por dominio',
                                              'Mayor caída por dominio'])
    fig.add_trace(go.Bar(y=cats_list, x=vals_m, orientation='h',
        text=paises_m, textposition='auto', textfont=dict(size=9, color=C['white']),
        marker=dict(color=C['green'], line_width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(y=cats_list, x=vals_c, orientation='h',
        text=paises_c, textposition='auto', textfont=dict(size=9, color=C['white']),
        marker=dict(color=C['red'], line_width=0), showlegend=False), row=1, col=2)
    fig.update_layout(**{**LAYOUT, 'height': 340}, margin=dict(l=100, r=20, t=50, b=30))
    fig.update_xaxes(gridcolor=C['grid'], tickfont=dict(size=9))
    fig.update_yaxes(gridcolor=C['grid'], tickfont=dict(size=9))
    fig.update_annotations(font=dict(color=C['ink'], size=11))
    return fig


# ══════════════════════════════════════════════════════════════════════
# 16. APP DASH
# ══════════════════════════════════════════════════════════════════════
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True, title='GHS Index v4.0')
server = app.server

DOM_OPTS = [{'label': f'Dominio {i+1} — {cat}', 'value': str(i+1)}
            for i, cat in enumerate(CAT_ES)]

app.layout = dbc.Container([
    make_header('GHS INDEX',
                f'LATAM-20 · 195 PAÍSES REFERENCIA · 4 NIVELES + 6 MÓDULOS · 2019–2021 · ghsindex.org'),
    make_methodology_note(
        f'Scores 2019 retroproyectados metodología 2021. '
        f'Shapiro-Wilk p₂₀₁₉={sw_p19:.3f}, p₂₀₂₁={sw_p21:.3f} → Wilcoxon justificado. '
        f'Clustering k=3 (método del codo). VulnIdx EXPLORATORIO (no validado vs. outcomes). '
        f'Correlaciones con IC de Fisher-z. {NOTA_MUESTRA}',
        accent='amber'),
    dbc.Row([
        make_kpi(f'{ST["latam_21"]["mean"]:.1f}', 'MEDIA LATAM 2021',
                 f'IC95% [{ST["latam_21"]["ci_lo"]}, {ST["latam_21"]["ci_hi"]}]',
                 C['latam']),
        make_kpi(f'{ST["global_21"]["mean"]:.1f}', 'MEDIA GLOBAL 2021',
                 f'Mediana: {ST["global_21"]["median"]}', C['global_c']),
        make_kpi(master.iloc[0]['Score_2021'],
                 f'MEJOR · {sn(master.iloc[0]["País"])}',
                 f'Rank #{master.iloc[0]["Rank_2021"]}/195', C['green']),
        make_kpi(master.iloc[-1]['Score_2021'],
                 f'PEOR · {sn(master.iloc[-1]["País"])}',
                 f'Rank #{master.iloc[-1]["Rank_2021"]}/195', C['red']),
        make_kpi(f'p={WX_19_21["p"]:.3f}', 'WILCOXON 19→21',
                 f'r={WX_19_21["r_rosenthal"]} · '
                 f'{"sig.*" if WX_19_21["p"]<0.05 else "n.s."}',
                 C['amber'] if WX_19_21['p']<0.05 else C['muted']),
        make_kpi(f'd={CD:+.2f}', "COHEN'S d",
                 f'{"Efecto pequeño" if abs(CD)<0.2 else "Efecto moderado"}',
                 C['purple']),
    ], style={'marginBottom': '10px'}),
    dcc.Tabs(id='tabs', value='t1', children=[
        dcc.Tab(label='📈 Visión general',  value='t1', style=TS, selected_style=TSS),
        dcc.Tab(label='Mapas de calor',    value='t2', style=TS, selected_style=TSS),
        dcc.Tab(label='Categorías',        value='t3', style=TS, selected_style=TSS),
        dcc.Tab(label='Correlaciones',     value='t4', style=TS, selected_style=TSS),
        dcc.Tab(label='Perfiles',          value='t5', style=TS, selected_style=TSS),
        dcc.Tab(label='Ranking',           value='t6', style=TS, selected_style=TSS),
        dcc.Tab(label='Indicadores N2',    value='t7', style=TS, selected_style=TSS),
        dcc.Tab(label='Sub-ind N3',        value='t8', style=TS, selected_style=TSS),
        dcc.Tab(label='Ítems N4',          value='t9', style=TS, selected_style=TSS),
        dcc.Tab(label='M1 Tipología',      value='m1', style=TS, selected_style=TSS),
        dcc.Tab(label='M2 Vulnerabilidad', value='m2', style=TS, selected_style=TSS),
        dcc.Tab(label='M3 Convergencia',   value='m3', style=TS, selected_style=TSS),
        dcc.Tab(label='M4 Política',       value='m4', style=TS, selected_style=TSS),
        dcc.Tab(label='M5 Predictores',    value='m5', style=TS, selected_style=TSS),
        dcc.Tab(label='M6 Trayectorias',   value='m6', style=TS, selected_style=TSS),
        dcc.Tab(label='Tabla maestra',     value='t10',style=TS, selected_style=TSS),
    ], style={'borderBottom': f'1px solid {C["border"]}', 'marginBottom': '14px'}),
    html.Div(id='content')
], fluid=True, style={'backgroundColor': C['bg'], 'minHeight': '100vh',
                      'padding': '0 20px 40px'})


@app.callback(Output('content', 'children'), Input('tabs', 'value'))
def render(tab):
    if tab == 't1':
        return html.Div([
            html.Div([make_section_title('Score Overall 2019 vs 2021',
                'Hover para cluster y categoría más débil'),
                dcc.Graph(figure=fig_bars_overview(), config=GRAPH_CONFIG)], style=CARD),
            html.Div([make_section_title('Distribución LATAM vs Resto del mundo 2021',
                f'Shapiro-Wilk + Wilcoxon apareado + Cohen\'s d'),
                dcc.Graph(figure=fig_boxplot(), config=GRAPH_CONFIG)], style=CARD),
        ])
    elif tab == 't2':
        return html.Div([
            html.Div([make_section_title('Puntajes 2021 por dominio'),
                dcc.Graph(figure=fig_heatmap_scores(), config=GRAPH_CONFIG)], style=CARD),
            html.Div([make_section_title('Cambio 2019→2021 por dominio'),
                dcc.Graph(figure=fig_heatmap_delta(), config=GRAPH_CONFIG)], style=CARD),
        ])
    elif tab == 't3':
        return html.Div([
            html.Div([make_section_title('Promedio por dominio: LATAM vs Global'),
                dcc.Graph(figure=fig_cat_bars(), config=GRAPH_CONFIG)], style=CARD),
            html.Div([make_section_title('Dispersión Overall vs dominio',
                'OLS + Pearson r + Spearman ρ con IC Fisher-z'),
                dcc.Dropdown(CAT_ES, value='Detección', id='scat-cat', clearable=False,
                    style={**DROPDOWN_STYLE, 'marginBottom': '10px', 'width': '300px'}),
                dcc.Graph(id='scat-graph', config=GRAPH_CONFIG)], style=CARD),
            html.Div([make_section_title('Estadísticas descriptivas LATAM 2021'),
                dash_table.DataTable(
                    data=[{'Dominio': cat,
                           'Media': CAT_ST[cat]['latam_21']['mean'],
                           'Mediana': CAT_ST[cat]['latam_21']['median'],
                           'SD': CAT_ST[cat]['latam_21']['sd'],
                           'Q25': CAT_ST[cat]['latam_21']['q25'],
                           'Q75': CAT_ST[cat]['latam_21']['q75'],
                           'Media Global': CAT_ST[cat]['global_21']['mean'],
                           'Δ vs Global': round(CAT_ST[cat]['latam_21']['mean']-CAT_ST[cat]['global_21']['mean'], 2)}
                          for cat in CAT_ES],
                    columns=[{'name':c,'id':c} for c in
                             ['Dominio','Media','Mediana','SD','Q25','Q75','Media Global','Δ vs Global']],
                    **TABLE_STYLE,
                    style_data_conditional=[
                        {'if':{'filter_query':'{Δ vs Global} > 0','column_id':'Δ vs Global'},
                         'color': C['green'], 'fontWeight':'700'},
                        {'if':{'filter_query':'{Δ vs Global} < 0','column_id':'Δ vs Global'},
                         'color': C['red'], 'fontWeight':'700'},
                    ])], style=CARD),
        ])
    elif tab == 't4':
        return html.Div([
            html.Div([make_section_title('Matrices de correlación entre dominios — LATAM 2021',
                'Pearson r y Spearman ρ · n=20 · Spearman preferido con distribución no normal'),
                dcc.Graph(figure=fig_corr_matrix(), config=GRAPH_CONFIG)], style=CARD),
            html.Div([html.Ul([
                html.Li(f'Shapiro-Wilk p₂₀₂₁={sw_p21:.3f} → Wilcoxon justificado sobre t-test.',
                        style={'color':C['amber'],'fontSize':'11px','marginBottom':'6px','fontWeight':'600'}),
                html.Li('Correlación Detección–Respuesta alta: ambas miden capacidad operativa activa.',
                        style={'color':C['text'],'fontSize':'11px','marginBottom':'6px'}),
                html.Li('Riesgo tiene menor correlación con el resto: mide entorno político/social.',
                        style={'color':C['text'],'fontSize':'11px','marginBottom':'6px'}),
                html.Li(f'Wilcoxon 2019 vs 2021 (n={WX_19_21["n"]}): W={WX_19_21["W"]:.0f}, '
                        f'p={WX_19_21["p"]:.3f}, r de Rosenthal={WX_19_21["r_rosenthal"]} → '
                        f'{WX_19_21["interpretation"]}.',
                        style={'color':C['amber'],'fontSize':'11px','fontWeight':'600'}),
            ])], style={**CARD, 'background': rgba(C['amber'], 0.05)}),
        ])
    elif tab == 't5':
        return html.Div([html.Div([
            make_section_title('Perfil por dominio: 2021 (sólido) vs 2019 (punteado)',
                'Hasta 5 países'),
            dcc.Dropdown([{'label':c,'value':c} for c in LATAM_20_SORTED],
                value=['Mexico','Chile','Peru','Venezuela'], multi=True, id='radar-sel',
                style={**DROPDOWN_STYLE, 'marginBottom':'10px'}),
            dcc.Graph(id='radar-graph', config=GRAPH_CONFIG)], style=CARD)])
    elif tab == 't6':
        return html.Div([
            html.Div([make_section_title('Cambio ranking global 2019→2021',
                'Sobre 195 países'),
                dcc.Graph(figure=fig_rank_change(), config=GRAPH_CONFIG)], style=CARD),
            html.Div([make_section_title('Percentil en distribución global 2021'),
                dcc.Graph(figure=fig_percentil(), config=GRAPH_CONFIG)], style=CARD),
        ])
    elif tab == 't7':
        return html.Div([html.Div([
            make_section_title('37 indicadores por país vs. media global (línea ámbar)'),
            dcc.Dropdown([{'label':c,'value':c} for c in LATAM_20_SORTED],
                value='Mexico', id='ind-country', clearable=False,
                style={**DROPDOWN_STYLE, 'marginBottom':'10px','width':'300px'}),
            dcc.Graph(id='ind-graph', config=GRAPH_CONFIG)], style=CARD)])
    elif tab == 't8':
        return html.Div([
            html.Div([make_section_title('96 sub-indicadores N3',
                '* = Wilcoxon p<0.05'),
                dbc.Row([
                    dbc.Col([dcc.Dropdown(DOM_OPTS, value='1', id='sub-dom',
                        clearable=False, style=DROPDOWN_STYLE)], md=5),
                    dbc.Col([dcc.RadioItems(
                        options=[{'label':'Scores 2021','value':'heat'},
                                 {'label':'Δ 2019→2021','value':'delta'},
                                 {'label':'vs Media Global','value':'vsg'}],
                        value='vsg', id='sub-view',
                        labelStyle={'display':'inline-block','marginRight':'14px',
                                    'fontSize':'11px','color': C['text']})], md=7),
                ], style={'marginBottom':'10px'}),
                dcc.Graph(id='sub-graph', config=GRAPH_CONFIG)], style=CARD),
            html.Div([make_section_title('Tabla N3 — filtrable y exportable'),
                dash_table.DataTable(
                    data=sub_master[['col','dominio_es','latam_19','latam_21','global_21',
                                     'delta','vs_global','sig']].rename(
                        columns={'col':'Sub-indicador','dominio_es':'Dominio',
                                 'latam_19':'LATAM_2019','latam_21':'LATAM_2021',
                                 'global_21':'Global_2021','delta':'Δ',
                                 'vs_global':'vs_Global','sig':'Sig'}).to_dict('records'),
                    columns=[{'name':c,'id':c} for c in
                             ['Sub-indicador','Dominio','LATAM_2019','LATAM_2021',
                              'Global_2021','Δ','vs_Global','Sig']],
                    filter_action='native', sort_action='native',
                    page_size=20, export_format='csv', **TABLE_STYLE,
                    style_data_conditional=[
                        {'if':{'filter_query':'{vs_Global} > 0','column_id':'vs_Global'},
                         'color': C['green']},
                        {'if':{'filter_query':'{vs_Global} < 0','column_id':'vs_Global'},
                         'color': C['red']},
                        {'if':{'filter_query':'{Sig} = "*"'},
                         'fontWeight':'700','color': C['amber']},
                    ])], style=CARD),
        ])
    elif tab == 't9':
        return html.Div([html.Div([
            make_section_title('171 ítems N4 — cobertura de cumplimiento LATAM vs Global'),
            dbc.Row([
                dbc.Col([dcc.Dropdown(DOM_OPTS, value='1', id='item-dom',
                    clearable=False, style=DROPDOWN_STYLE)], md=5),
                dbc.Col([dcc.RadioItems(
                    options=[{'label':'Cobertura 2021','value':'pct_2021'},
                             {'label':'Brecha vs Global','value':'vs_global'},
                             {'label':'Cambio 2019→21','value':'delta'}],
                    value='pct_2021', id='item-orden',
                    labelStyle={'display':'inline-block','marginRight':'12px',
                                'fontSize':'11px','color': C['text']})], md=7),
            ], style={'marginBottom':'10px'}),
            dcc.Graph(id='item-graph', config=GRAPH_CONFIG)], style=CARD)])
    elif tab == 'm1':
        return html.Div([
            html.Div([make_section_title('Tipología de países — clustering k-means en espacio de 6 dominios',
                f'k=3 óptimo · PC1={pc1_var}% + PC2={pc2_var}% = {round(pc1_var+pc2_var,1)}% varianza'),
                dbc.Row([
                    dbc.Col([dcc.Graph(figure=fig_m1_elbow(), config=GRAPH_CONFIG)], md=4),
                    dbc.Col([dcc.Graph(figure=fig_m1_pca(), config=GRAPH_CONFIG)], md=8),
                ])], style=CARD),
            html.Div([make_section_title('Perfil medio por cluster',
                'Cada cluster tiene un patrón de fortalezas/debilidades distinto'),
                dcc.Graph(figure=fig_m1_radar_clusters(), config=GRAPH_CONFIG)], style=CARD),
            html.Div([make_section_title('Tabla de pertenencia a clusters'),
                dash_table.DataTable(
                    data=master[['País','Cluster','Score_2021','Δ']+[f'{c}_2021' for c in CAT_ES]].to_dict('records'),
                    columns=[{'name':c,'id':c} for c in
                             ['País','Cluster','Score_2021','Δ']+[f'{c}_2021' for c in CAT_ES]],
                    filter_action='native', sort_action='native', export_format='csv',
                    **TABLE_STYLE,
                    style_data_conditional=[
                        {'if':{'filter_query':'{Cluster} = "Capacidad Alta"','column_id':'Cluster'},
                         'color': C['green'],'fontWeight':'700'},
                        {'if':{'filter_query':'{Cluster} = "Capacidad Media"','column_id':'Cluster'},
                         'color': C['amber'],'fontWeight':'700'},
                        {'if':{'filter_query':'{Cluster} = "Capacidad Baja"','column_id':'Cluster'},
                         'color': C['red'],'fontWeight':'700'},
                    ])], style=CARD),
        ])
    elif tab == 'm2':
        return html.Div([
            make_methodology_note(
                'VulnIdx = (100 − Overall) × (1 + CV_dominios). Métrica EXPLORATORIA '
                'no validada contra desenlaces externos (exceso de mortalidad COVID, etc.). '
                'Interpretar como ranking relativo, no como predictor validado.',
                accent='amber'),
            html.Div([make_section_title('Índice de Vulnerabilidad Pandémica (VulnIdx)'),
                dcc.Graph(figure=fig_m2_vuln(), config=GRAPH_CONFIG)], style=CARD),
            html.Div([make_section_title('Ranking de vulnerabilidad',
                'Color por cluster'),
                dcc.Graph(figure=fig_m2_bars(), config=GRAPH_CONFIG)], style=CARD),
            html.Div([make_section_title('Tabla VulnIdx completa'),
                dash_table.DataTable(
                    data=vuln_df.to_dict('records'),
                    columns=[{'name':c,'id':c} for c in
                             ['País','Cluster','Overall','CV','VulnIdx','Gap',
                              'Mejor','Peor','Score_Mejor','Score_Peor']],
                    filter_action='native', sort_action='native', export_format='csv',
                    **TABLE_STYLE,
                    style_data_conditional=[
                        {'if':{'filter_query':'{VulnIdx} > 100'},
                         'color': C['red'],'fontWeight':'700'},
                        {'if':{'filter_query':'{VulnIdx} < 55'},
                         'color': C['green']},
                    ])], style=CARD),
        ])
    elif tab == 'm3':
        return html.Div([
            html.Div([make_section_title('Convergencia sistémica 2019→2021',
                'Cuadrante superior izquierdo = convergencia real (↑ Overall + ↓ disparidad)'),
                dcc.Graph(figure=fig_m3_convergence(), config=GRAPH_CONFIG)], style=CARD),
            html.Div([make_section_title('Tabla de trayectorias sistémicas'),
                dash_table.DataTable(
                    data=conv_df.to_dict('records'),
                    columns=[{'name':c,'id':c} for c in
                             ['País','Cluster','Tipo','Overall_2019','Overall_2021',
                              'Δ_Overall','CV_2019','CV_2021','Δ_CV']],
                    filter_action='native', sort_action='native', export_format='csv',
                    **TABLE_STYLE,
                    style_data_conditional=[
                        {'if':{'filter_query':'{Tipo} = "Convergencia"','column_id':'Tipo'},
                         'color': C['green'],'fontWeight':'700'},
                        {'if':{'filter_query':'{Tipo} = "Divergencia"','column_id':'Tipo'},
                         'color': C['red'],'fontWeight':'700'},
                        {'if':{'filter_query':'{Tipo} = "Mejora asimétrica"','column_id':'Tipo'},
                         'color': C['amber']},
                    ])], style=CARD),
        ])
    elif tab == 'm4':
        return html.Div([
            html.Div([make_section_title('Matriz de oportunidades de política pública',
                'Cuadrante superior derecho = alta factibilidad + alto impacto'),
                dcc.Graph(figure=fig_m4_matrix(), config=GRAPH_CONFIG)], style=CARD),
            html.Div([make_section_title('Tabla de ítems por prioridad'),
                dash_table.DataTable(
                    data=opps_df[['Ítem','Dominio','Prioridad','LATAM_2021','LATAM_2019',
                                  'Global','Brecha','Δ_LATAM']].to_dict('records'),
                    columns=[{'name':c,'id':c} for c in
                             ['Ítem','Dominio','Prioridad','LATAM_2021','LATAM_2019',
                              'Global','Brecha','Δ_LATAM']],
                    filter_action='native', sort_action='native',
                    page_size=20, export_format='csv', **TABLE_STYLE,
                    style_data_conditional=[
                        {'if':{'filter_query':'{Prioridad} = "Alta"','column_id':'Prioridad'},
                         'color': C['red'],'fontWeight':'700'},
                        {'if':{'filter_query':'{Prioridad} = "Media"','column_id':'Prioridad'},
                         'color': C['amber']},
                        {'if':{'filter_query':'{Brecha} > 20','column_id':'Brecha'},
                         'color': C['red'],'fontWeight':'700'},
                    ])], style=CARD),
        ])
    elif tab == 'm5':
        return html.Div([
            html.Div([make_section_title('Predictores del Overall — Spearman ρ por dominio',
                'Con IC Fisher-z'),
                dcc.Graph(figure=fig_m5_predictors(), config=GRAPH_CONFIG)], style=CARD),
            html.Div([make_section_title('Regresión OLS por dominio — 6 scatter'),
                dcc.Graph(figure=fig_m5_scatter_all(), config=GRAPH_CONFIG)], style=CARD),
            html.Div([make_section_title('Tabla de coeficientes (con IC Fisher-z)'),
                dash_table.DataTable(
                    data=pred_df.to_dict('records'),
                    columns=[{'name':c,'id':c} for c in
                             ['Dominio','Spearman_r','Spearman_CI','Spearman_p',
                              'Pearson_r','Pearson_p','R2','OLS_slope','OLS_intercept']],
                    **TABLE_STYLE,
                    style_data_conditional=[
                        {'if':{'filter_query':'{Spearman_r} > 0.8','column_id':'Spearman_r'},
                         'color': C['green'],'fontWeight':'700'},
                    ])], style=CARD),
        ])
    elif tab == 'm6':
        return html.Div([
            html.Div([make_section_title('Heatmap de cambios 2019→2021'),
                dcc.Graph(figure=fig_m6_heatmap(), config=GRAPH_CONFIG)], style=CARD),
            html.Div([make_section_title('Trayectorias más extremas por dominio'),
                dcc.Graph(figure=fig_m6_notable(), config=GRAPH_CONFIG)], style=CARD),
            html.Div([make_section_title('Tabla completa de trayectorias',
                'Filtrable por dominio'),
                dcc.Dropdown([{'label':'Todos','value':'Todos'}]+
                             [{'label':cat,'value':cat} for cat in CAT_ES],
                             value='Todos', id='traj-dom', clearable=False,
                             style={**DROPDOWN_STYLE,'marginBottom':'10px','width':'300px'}),
                dash_table.DataTable(id='traj-table',
                    columns=[{'name':c,'id':c} for c in
                             ['País','Dominio','Cluster','V_2019','V_2021','Delta']],
                    filter_action='native', sort_action='native',
                    page_size=24, export_format='csv', **TABLE_STYLE,
                    style_data_conditional=[
                        {'if':{'filter_query':'{Delta} > 10','column_id':'Delta'},
                         'color': C['green'],'fontWeight':'700'},
                        {'if':{'filter_query':'{Delta} < -10','column_id':'Delta'},
                         'color': C['red'],'fontWeight':'700'},
                    ])], style=CARD),
        ])
    elif tab == 't10':
        cols = (['País','Cluster','Score_2019','Score_2021','Δ','Rank_2019','Rank_2021',
                 'Rank_Δ','Percentil_2021','Cat_Débil']
                + [f'{c}_2021' for c in CAT_ES] + [f'Δ_{c}' for c in CAT_ES])
        return html.Div([html.Div([
            make_section_title('Tabla maestra completa'),
            dash_table.DataTable(
                data=master[cols].to_dict('records'),
                columns=[{'name':c,'id':c} for c in cols],
                filter_action='native', sort_action='native',
                page_size=24, export_format='csv', **TABLE_STYLE,
                style_data_conditional=[
                    {'if':{'filter_query':'{Δ} > 0','column_id':'Δ'},
                     'color': C['green'],'fontWeight':'700'},
                    {'if':{'filter_query':'{Δ} < 0','column_id':'Δ'},
                     'color': C['red'],'fontWeight':'700'},
                    {'if':{'filter_query':'{Cluster} = "Capacidad Alta"','column_id':'Cluster'},
                     'color': C['green']},
                    {'if':{'filter_query':'{Cluster} = "Capacidad Baja"','column_id':'Cluster'},
                     'color': C['red']},
                ])], style=CARD)])


# ══════════════════════════════════════════════════════════════════════
# 17. CALLBACKS
# ══════════════════════════════════════════════════════════════════════
@app.callback(Output('scat-graph', 'figure'), Input('scat-cat', 'value'))
def cb_scat(cat): return fig_scatter(cat)

@app.callback(Output('radar-graph', 'figure'), Input('radar-sel', 'value'))
def cb_radar(c):
    if not c: return go.Figure()
    return fig_radar(c)

@app.callback(Output('ind-graph', 'figure'), Input('ind-country', 'value'))
def cb_ind(c): return fig_indicadores(c)

@app.callback(Output('sub-graph', 'figure'),
              [Input('sub-dom', 'value'), Input('sub-view', 'value')])
def cb_sub(dom, view): return fig_sub_graph(dom, view)

@app.callback(Output('item-graph', 'figure'),
              [Input('item-dom', 'value'), Input('item-orden', 'value')])
def cb_item(dom, orden): return fig_items_coverage(dom, orden)

@app.callback(Output('traj-table', 'data'), Input('traj-dom', 'value'))
def cb_traj(dom):
    df = traj_df if dom == 'Todos' else traj_df[traj_df['Dominio']==dom]
    return df.sort_values('Delta', ascending=False).to_dict('records')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(debug=False, host='0.0.0.0', port=port)
