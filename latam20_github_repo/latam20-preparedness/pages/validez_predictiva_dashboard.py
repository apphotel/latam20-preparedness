"""
validez_predictiva_dashboard.py — Dashboard de Validez Predictiva
Análisis A1-A7 del paper · P-score Karlinsky · GINI · WGI
Puerto 8061
"""
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

from latam_common import LATAM_ISO3, LATAM_20, sn, data_path, SUBREGION
from theme import C, LAYOUT, CARD, TS, TSS, TABLE_STYLE, GRAPH_CONFIG, make_header, make_section_title, rgba
from stats_utils import correlation_ci, adjust_pvalues, bootstrap_ci, wilcoxon_paired

# ═══════════════════════════════════════════════════════
# CARGA DE DATOS
# ═══════════════════════════════════════════════════════

def load_master():
    try:
        df = pd.read_csv(data_path('tabla_maestra'))
        return df
    except:
        pass
    # Fallback: construir desde fuentes
    return build_master()

def build_master():
    from pathlib import Path
    data_dir = Path(data_path('ghs')).parent

    LATAM = LATAM_ISO3

    # GHS
    ghs = pd.read_csv(data_path('ghs'), low_memory=False)
    ghs19 = ghs[(ghs['Country'].isin(LATAM.values())) & (ghs['Year']==2019)].copy()
    ghs19 = ghs19.rename(columns={
        'OVERALL SCORE':'GHS_total',
        'Country':'country',
        '1) PREVENTION OF THE EMERGENCE OR RELEASE OF PATHOGENS':'GHS_Prevention',
        "2) EARLY DETECTION & REPORTING FOR EPIDEMICS OF POTENTIAL INT'L CONCERN":'GHS_Detection',
        '3) RAPID RESPONSE TO AND MITIGATION OF THE SPREAD OF AN EPIDEMIC':'GHS_Response',
        '4) SUFFICIENT & ROBUST HEALTH SECTOR TO TREAT THE SICK & PROTECT HEALTH WORKERS':'GHS_HealthSystem',
        '5) COMMITMENTS TO IMPROVING NATIONAL CAPACITY, FINANCING AND ADHERENCE TO NORMS':'GHS_RSICompliance',
        '6) OVERALL RISK ENVIRONMENT AND COUNTRY VULNERABILITY TO BIOLOGICAL THREATS':'GHS_RiskEnv',
    })

    # SPAR (latam csv generado)
    spar = pd.read_csv(data_path('spar_latam'))
    s18 = spar[spar['year']==2018].set_index('iso3')
    s20 = spar[spar['year']==2020].set_index('iso3')
    spar_rows = []
    spar_cols = ['SPAR_total'] + [f'C{i}' for i in range(1,14)]
    for iso, country in LATAM.items():
        rec = {'iso3':iso, 'country':country}
        for col in spar_cols:
            v18 = s18.loc[iso, col] if iso in s18.index and col in s18.columns else np.nan
            v20 = s20.loc[iso, col] if iso in s20.index and col in s20.columns else np.nan
            vals = [v for v in [v18,v20] if pd.notna(v)]
            target = f'SPAR_{col}' if col!='SPAR_total' else 'SPAR_total'
            rec[target] = round(np.mean(vals),1) if vals else np.nan
        spar_rows.append(rec)
    spar19 = pd.DataFrame(spar_rows)

    # P-scores
    karl = pd.read_csv(data_dir/'world_mortality.csv')
    lk = karl[karl['iso3c'].isin(LATAM.keys())]
    pscore = {}
    for iso in LATAM:
        cdf = lk[lk['iso3c']==iso]
        bl = cdf[cdf['year'].between(2015,2019)].groupby('year')['deaths'].sum()
        if len(bl)<3: continue
        bl_m = bl.mean()
        for yr in [2020,2021]:
            yd = cdf[cdf['year']==yr]
            if len(yd)<10: continue
            pscore[(iso,yr)] = round(((yd['deaths'].sum()-bl_m)/bl_m)*100,1)

    # OMS
    oms_raw = pd.read_excel(data_dir/'WHO_COVID_Excess_Deaths_EstimatesByCountry.xlsx',
                            sheet_name='Deaths by year', header=None)
    oms = oms_raw.iloc[8:,:].copy()
    oms.columns = ['country_name','iso3','year','excess_mean','excess_low','excess_high']
    oms = oms[oms['iso3'].isin(LATAM.keys())].copy()
    oms['year'] = pd.to_numeric(oms['year'], errors='coerce')
    for c in ['excess_mean','excess_low','excess_high']:
        oms[c] = pd.to_numeric(oms[c], errors='coerce')
    oms20 = oms[oms['year']==2020].set_index('iso3')

    # GINI
    gini_df = pd.read_csv(data_dir/'economic-inequality-gini-index.csv')
    gini_dict = {}
    for iso in LATAM:
        sub = gini_df[gini_df['Code']==iso]
        pre = sub[sub['Year']<=2019]
        if len(pre):
            row = pre.loc[pre['Year'].idxmax()]
            gini_dict[iso] = round(float(row['Gini coefficient']),3)
    gini_dict.update({'ARG':0.420,'CUB':0.380,'VEN':0.440,'HTI':0.605})

    # WGI
    wgi = pd.read_csv(data_dir/'wgi_governance_data.csv')
    for col in [c for c in wgi.columns if 'YR' in c]:
        wgi[col] = pd.to_numeric(wgi[col], errors='coerce')
    wgi_l = wgi[wgi['Country Code'].isin(LATAM.keys())].copy()
    wgi_l['pre_avg'] = wgi_l[['2017 [YR2017]','2018 [YR2018]','2019 [YR2019]']].mean(axis=1)
    sm = {'GOV_WGI_CC.EST':'WGI_Corruption','GOV_WGI_GE.EST':'WGI_GovEff',
          'GOV_WGI_RL.EST':'WGI_RuleOfLaw','GOV_WGI_PV.EST':'WGI_PolitStab'}
    wgi_l['ind'] = wgi_l['Series Code'].map(sm)
    wgi_piv = wgi_l.pivot_table(index='Country Code',columns='ind',values='pre_avg').round(3)

    # OxCGRT
    ox = pd.read_csv(data_path('oxcgrt_compact'), low_memory=False)
    ox['Date'] = pd.to_datetime(ox['Date'], format='%Y%m%d')
    ox20 = ox[(ox['CountryCode'].isin(LATAM.keys()))&(ox['Jurisdiction']=='NAT_TOTAL')&(ox['Date'].dt.year==2020)]
    gov_resp = ox20.groupby('CountryCode')['GovernmentResponseIndex_Average'].mean().round(2)

    # INFORM
    inf = pd.read_excel(data_dir/'INFORM2026_TREND_2017_2026_v72_ALL.xlsx', sheet_name='INFORM2026Trend')
    KEY = {'INFORM':'INFORM_Risk_total','HA':'INFORM_Hazard','VU':'INFORM_Vulnerability','CC':'INFORM_LackCoping'}
    mask = (inf['Iso3'].isin(LATAM.keys()))&(inf['INFORMYear']==2019)&(inf['IndicatorId'].isin(KEY.keys()))
    inf19 = inf[mask].copy()
    inf19['var'] = inf19['IndicatorId'].map(KEY)
    inf_piv = inf19.pivot_table(index='Iso3',columns='var',values='IndicatorScore',aggfunc='first').reset_index()

    # Ensamblar
    rows = []
    for iso, country in LATAM.items():
        rec = {'iso3':iso,'country':country,'subregion':SUBREGION.get(country,'')}
        g = ghs19[ghs19['country']==country]
        if len(g):
            for c in ['GHS_total','GHS_Prevention','GHS_Detection','GHS_Response',
                      'GHS_HealthSystem','GHS_RSICompliance','GHS_RiskEnv']:
                rec[c] = round(float(g.iloc[0][c]),1) if c in g.columns else None
        s = spar19[spar19['iso3']==iso]
        if len(s):
            s0 = s.iloc[0]
            for c in ['SPAR_total']+[f'SPAR_C{i}' for i in range(1,14)]:
                rec[c] = float(s0[c]) if c in s0.index and pd.notna(s0[c]) else None
        ig = inf_piv[inf_piv['Iso3']==iso]
        if len(ig):
            for c in ['INFORM_Risk_total','INFORM_Hazard','INFORM_Vulnerability','INFORM_LackCoping']:
                if c in ig.columns: rec[c] = float(ig.iloc[0][c]) if pd.notna(ig.iloc[0][c]) else None
        rec['Pscore_2020'] = pscore.get((iso,2020))
        rec['Pscore_2021'] = pscore.get((iso,2021))
        rec['GovResponse_2020'] = float(gov_resp[iso]) if iso in gov_resp else None
        if iso in oms20.index:
            rec['OMS_excess_2020'] = float(oms20.loc[iso,'excess_mean']) if pd.notna(oms20.loc[iso,'excess_mean']) else None
        rec['GINI'] = gini_dict.get(iso)
        if iso in wgi_piv.index:
            for c in wgi_piv.columns: rec[c] = float(wgi_piv.loc[iso,c])
        rec['DataTier'] = 1 if pscore.get((iso,2020)) else (2 if iso in ['SLV','HND'] else 3)
        rows.append(rec)
    return pd.DataFrame(rows)

print("Cargando datos...", flush=True)
df = load_master()
print(f"Tabla maestra: {df.shape}", flush=True)

PREDICTORES = {
    'GHS_total':'GHS Total',
    'GHS_Prevention':'GHS Prevención','GHS_Detection':'GHS Detección',
    'GHS_Response':'GHS Respuesta','GHS_HealthSystem':'GHS Sist.Salud',
    'GHS_RSICompliance':'GHS Cumpl.RSI','GHS_RiskEnv':'GHS Entorno',
    'SPAR_total':'SPAR Total',
    'SPAR_C5':'SPAR Lab.','SPAR_C6':'SPAR Vigilancia',
    'SPAR_C8':'SPAR Emergencias','SPAR_C9':'SPAR Serv.Salud',
    'INFORM_Risk_total':'INFORM Risk','INFORM_Hazard':'INFORM Amenaza',
    'INFORM_Vulnerability':'INFORM Vulnerab.','INFORM_LackCoping':'INFORM Capacidad',
}
DESENLACES = {
    'Pscore_2020':'P-score 2020 (Karlinsky)',
    'Pscore_2021':'P-score 2021 (Karlinsky)',
    'OMS_excess_2020':'Exceso OMS 2020',
    'GovResponse_2020':'Respuesta Gov. 2020',
}
TIER_COLORS = {1:C['green'],2:C['amber'],3:C['red']}

# ═══════════════════════════════════════════════════════
# FIGURAS
# ═══════════════════════════════════════════════════════

def fig_scatter(predictor, desenlace):
    sub = df[[predictor, desenlace, 'country', 'DataTier','GINI']].dropna(subset=[predictor, desenlace])
    if len(sub) < 5:
        return go.Figure().update_layout(**LAYOUT, title="Datos insuficientes")
    r = correlation_ci(sub[predictor], sub[desenlace], method='spearman')
    colors = [TIER_COLORS.get(t, C['muted']) for t in sub['DataTier']]
    # Línea de regresión
    m, b, *_ = stats.linregress(sub[predictor], sub[desenlace])
    x_line = np.linspace(sub[predictor].min(), sub[predictor].max(), 50)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub[predictor], y=sub[desenlace],
        mode='markers+text',
        text=[sn(c) for c in sub['country']],
        textposition='top center',
        textfont=dict(size=9, color=C['muted']),
        marker=dict(size=10, color=colors, line=dict(width=1, color=C['border'])),
        hovertemplate='<b>%{text}</b><br>'
                      f'{PREDICTORES.get(predictor,predictor)}: %{{x:.1f}}<br>'
                      f'{DESENLACES.get(desenlace,desenlace)}: %{{y:.1f}}<br>'
                      '<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=x_line, y=m*x_line+b, mode='lines',
        line=dict(color=C['latam'], width=1.5, dash='dash'),
        showlegend=False, hoverinfo='skip',
    ))
    p_fmt = f"{r['p']:.3f}" if r['p'] >= 0.001 else "<0.001"
    sig = "✱" if r['p'] < 0.05 else "n.s."
    ci_lo, ci_hi = r['ci']
    title = (f"ρ={r['r']:+.3f}  IC95%=[{ci_lo:.3f},{ci_hi:.3f}]  "
             f"p={p_fmt} {sig}  n={r['n']}")
    fig.update_layout(**LAYOUT, height=420,
        title=dict(text=title, font=dict(size=10, color=C['muted'])),
        xaxis=dict(title=PREDICTORES.get(predictor,predictor), gridcolor=C['grid']),
        yaxis=dict(title=DESENLACES.get(desenlace,desenlace), gridcolor=C['grid']),
    )
    return fig

def fig_heatmap_correlaciones():
    pred_list = [p for p in PREDICTORES if p in df.columns]
    des_list  = [d for d in DESENLACES if d in df.columns]
    rho_mat, p_mat, n_mat = [], [], []
    for des in des_list:
        row_r, row_p, row_n = [], [], []
        for pred in pred_list:
            sub = df[[pred,des]].dropna()
            if len(sub) >= 5:
                r,p = stats.spearmanr(sub[pred],sub[des])
                row_r.append(round(r,3)); row_p.append(round(p,4)); row_n.append(len(sub))
            else:
                row_r.append(np.nan); row_p.append(np.nan); row_n.append(0)
        rho_mat.append(row_r); p_mat.append(row_p); n_mat.append(row_n)

    # FDR sobre todos los p-valores
    all_p = [p for row in p_mat for p in row if not np.isnan(p)]
    all_p_adj = adjust_pvalues(all_p)
    idx = 0
    p_adj_mat = []
    for row in p_mat:
        adj_row = []
        for p in row:
            if np.isnan(p): adj_row.append(np.nan)
            else: adj_row.append(all_p_adj[idx]); idx+=1
        p_adj_mat.append(adj_row)

    # Texto de celda
    text_mat = []
    for i,row_r in enumerate(rho_mat):
        text_row = []
        for j,r in enumerate(row_r):
            if np.isnan(r): text_row.append("")
            else:
                p = p_adj_mat[i][j]
                sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
                text_row.append(f"{r:+.2f}{sig}")
        text_mat.append(text_row)

    pred_labels = [PREDICTORES.get(p,p) for p in pred_list]
    des_labels  = [DESENLACES.get(d,d) for d in des_list]

    fig = go.Figure(go.Heatmap(
        z=rho_mat, x=pred_labels, y=des_labels,
        text=text_mat, texttemplate='%{text}',
        colorscale=[[0,'#C53030'],[0.5,'#F0F2F5'],[1,'#1E40AF']],
        zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title='ρ Spearman', len=0.8),
        hoverongaps=False,
    ))
    fig.update_layout(**LAYOUT, height=320,
        title=dict(text='Correlaciones Spearman · corrección FDR (*p<0.05  **p<0.01  ***p<0.001)',
                   font=dict(size=10, color=C['muted'])),
        xaxis=dict(tickangle=-40, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
        margin=dict(l=200, r=40, t=60, b=130),
    )
    return fig

def fig_moderacion_gini(predictor='GHS_total', desenlace='Pscore_2020'):
    sub = df[[predictor, desenlace, 'GINI', 'country']].dropna()
    if len(sub) < 6:
        return go.Figure().update_layout(**LAYOUT, title="Datos insuficientes")
    terciles = pd.qcut(sub['GINI'], q=3, labels=['Bajo GINI\n(+igualdad)',
                                                   'GINI medio',
                                                   'Alto GINI\n(-igualdad)'])
    colores = [C['green'], C['amber'], C['red']]
    fig = go.Figure()
    for i, (tier, label) in enumerate(zip(['Bajo GINI\n(+igualdad)','GINI medio','Alto GINI\n(-igualdad)'],
                                           ['Bajo GINI (mayor igualdad)','GINI medio','Alto GINI (mayor desigualdad)'])):
        mask = terciles == tier
        s = sub[mask]
        if len(s) < 3: continue
        r, p = stats.spearmanr(s[predictor], s[desenlace])
        # Línea de tendencia
        if len(s) >= 3:
            m, b, *_ = stats.linregress(s[predictor], s[desenlace])
            xr = np.linspace(sub[predictor].min(), sub[predictor].max(), 30)
            fig.add_trace(go.Scatter(x=xr, y=m*xr+b, mode='lines',
                line=dict(color=colores[i], width=2, dash='solid'),
                name=f'{label} (ρ={r:+.2f}, p={p:.3f})', showlegend=True))
        fig.add_trace(go.Scatter(
            x=s[predictor], y=s[desenlace], mode='markers',
            marker=dict(size=9, color=colores[i], opacity=0.8),
            text=[sn(c) for c in s['country']],
            hovertemplate='<b>%{text}</b><br>X=%{x:.1f}, Y=%{y:.1f}<extra></extra>',
            showlegend=False,
        ))
    fig.update_layout(**LAYOUT, height=400,
        title=dict(text=f'Moderación GINI — {PREDICTORES.get(predictor,predictor)} vs {DESENLACES.get(desenlace,desenlace)}',
                   font=dict(size=10, color=C['muted'])),
        xaxis=dict(title=PREDICTORES.get(predictor,predictor), gridcolor=C['grid']),
        yaxis=dict(title=DESENLACES.get(desenlace,desenlace), gridcolor=C['grid']),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.9)',
                    font=dict(size=9)),
    )
    return fig

def fig_mahajan_h2():
    sub = df[['GHS_RSICompliance','GHS_HealthSystem','Pscore_2020']].dropna()
    if len(sub)<5: return go.Figure().update_layout(**LAYOUT)
    r1,p1 = stats.spearmanr(sub['GHS_RSICompliance'], sub['Pscore_2020'])
    r2,p2 = stats.spearmanr(sub['GHS_HealthSystem'], sub['Pscore_2020'])
    # Fisher z test
    n = len(sub)
    z1 = np.arctanh(r1); z2 = np.arctanh(r2)
    z_diff = (z1-z2) / np.sqrt(2/(n-3))
    from scipy.stats import norm
    p_fisher = 2*(1-norm.cdf(abs(z_diff)))
    bars = go.Bar(
        x=['GHS Cumplimiento RSI\n(H2: menor predictor)','GHS Sistema de Salud\n(H2: mayor predictor)'],
        y=[r1, r2],
        marker_color=[C['amber'] if r1>r2 else C['green'],
                      C['green'] if r2>r1 else C['amber']],
        text=[f'ρ={r1:+.3f}<br>p={p1:.3f}', f'ρ={r2:+.3f}<br>p={p2:.3f}'],
        textposition='outside',
        width=0.4,
    )
    fig = go.Figure(bars)
    fig.add_hline(y=0, line_color=C['border'])
    result = "H2 CONFIRMADA ✓" if r1 < r2 and p_fisher < 0.05 else \
             "H2 tendencia (n.s.)" if r1 < r2 else "H2 NO confirmada"
    fig.update_layout(**LAYOUT, height=350,
        title=dict(text=f'Test hipótesis H2 Mahajan — Fisher z: z={z_diff:.2f}, p={p_fisher:.3f} — {result}',
                   font=dict(size=10, color=C['muted'])),
        yaxis=dict(title='ρ Spearman vs P-score 2020', range=[-0.8,0.8], gridcolor=C['grid']),
        xaxis=dict(tickfont=dict(size=10)),
        showlegend=False,
    )
    return fig

def fig_h3_sesgo():
    sub = df[['SPAR_total','WGI_Corruption','country']].dropna()
    if len(sub)<5: return go.Figure().update_layout(**LAYOUT)
    r,p = stats.spearmanr(sub['WGI_Corruption'], sub['SPAR_total'])
    m,b,*_ = stats.linregress(sub['WGI_Corruption'], sub['SPAR_total'])
    xr = np.linspace(sub['WGI_Corruption'].min(), sub['WGI_Corruption'].max(), 30)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xr, y=m*xr+b, mode='lines',
        line=dict(color=C['latam'],width=1.5,dash='dash'), showlegend=False))
    fig.add_trace(go.Scatter(
        x=sub['WGI_Corruption'], y=sub['SPAR_total'],
        mode='markers+text',
        text=[sn(c) for c in sub['country']],
        textposition='top center',
        textfont=dict(size=8, color=C['muted']),
        marker=dict(size=9, color=C['amber'], opacity=0.85,
                    line=dict(width=1,color=C['border'])),
        hovertemplate='<b>%{text}</b><br>WGI Corrupción=%{x:.2f}<br>SPAR=%{y:.0f}<extra></extra>',
    ))
    h3_result = "H3 CONFIRMADA ✓" if r < -0.2 and p < 0.1 else "H3 tendencia" if r < 0 else "H3 no confirmada"
    p_fmt = f"{p:.3f}" if p >= 0.001 else "<0.001"
    fig.update_layout(**LAYOUT, height=400,
        title=dict(text=f'H3 Sesgo autorreporte SPAR — ρ={r:+.3f}, p={p_fmt} — {h3_result}',
                   font=dict(size=10, color=C['muted'])),
        xaxis=dict(title='WGI Control de Corrupción (mayor = mejor gobernanza)', gridcolor=C['grid']),
        yaxis=dict(title='SPAR Total 2019', gridcolor=C['grid']),
    )
    return fig

def tabla_resumen():
    pred_list = [p for p in PREDICTORES if p in df.columns]
    rows = []
    for pred in pred_list:
        for des_key, des_label in DESENLACES.items():
            if des_key not in df.columns: continue
            sub = df[[pred, des_key]].dropna()
            if len(sub) < 5: continue
            r, p = stats.spearmanr(sub[pred], sub[des_key])
            ci = correlation_ci(sub[pred], sub[des_key], method='spearman')
            rows.append({
                'Predictor': PREDICTORES.get(pred,pred),
                'Desenlace': des_label,
                'n': len(sub),
                'ρ Spearman': round(r,3),
                'IC95% [lo–hi]': f"[{ci['ci'][0]:.3f}, {ci['ci'][1]:.3f}]",
                'p': round(p,4),
                'Significancia': '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.',
            })
    df_res = pd.DataFrame(rows)
    # FDR
    if len(df_res):
        df_res['p_FDR'] = adjust_pvalues(df_res['p'].values).round(4)
        df_res['Sig.FDR'] = df_res['p_FDR'].apply(lambda p: '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.')
    return df_res

# ═══════════════════════════════════════════════════════
# APP
# ═══════════════════════════════════════════════════════
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)
app.title = "Validez Predictiva — LATAM-20"

pred_options  = [{'label':v,'value':k} for k,v in PREDICTORES.items() if k in df.columns]
des_options   = [{'label':v,'value':k} for k,v in DESENLACES.items() if k in df.columns]

server = app.server  # Requerido por gunicorn/Render
app.layout = dbc.Container([
    make_header("VALIDEZ PREDICTIVA DE ÍNDICES PANDÉMICOS — LATAM-20",
                "Análisis A1-A7 del protocolo OSF · GHS · SPAR · INFORM · P-score Karlinsky · OMS"),
    dcc.Tabs(id='tabs', value='t1', children=[
        dcc.Tab(label='📊 Heatmap correlaciones',  value='t1', style=TS, selected_style=TSS),
        dcc.Tab(label='🔍 Scatter predictor',       value='t2', style=TS, selected_style=TSS),
        dcc.Tab(label='🧪 H2 Mahajan',              value='t3', style=TS, selected_style=TSS),
        dcc.Tab(label='⚖️ H3 Sesgo SPAR',           value='t4', style=TS, selected_style=TSS),
        dcc.Tab(label='📐 H4 Moderación GINI',      value='t5', style=TS, selected_style=TSS),
        dcc.Tab(label='📋 Tabla A2 completa',       value='t6', style=TS, selected_style=TSS),
        dcc.Tab(label='🗂️ Tabla maestra',           value='t7', style=TS, selected_style=TSS),
    ]),
    html.Div(id='tab-content', style={'paddingTop':'12px'}),
], fluid=True, style={'backgroundColor':C['bg'],'padding':'0 20px 30px'})

@app.callback(Output('tab-content','children'), Input('tabs','value'))
def render(tab):
    if tab == 't1':
        return html.Div([
            html.Div(style=CARD, children=[
                make_section_title("Matriz de correlaciones Spearman",
                    "Todos los predictores × todos los desenlaces · corrección FDR Benjamini-Hochberg"),
                dcc.Graph(figure=fig_heatmap_correlaciones(), config=GRAPH_CONFIG),
            ])
        ])
    elif tab == 't2':
        return html.Div([
            html.Div(style={**CARD,'marginBottom':'8px'}, children=[
                dbc.Row([
                    dbc.Col([html.Label("Predictor", style={'fontSize':'11px','color':C['muted']}),
                             dcc.Dropdown(id='sel-pred', options=pred_options,
                                          value='GHS_total', clearable=False)], md=4),
                    dbc.Col([html.Label("Desenlace", style={'fontSize':'11px','color':C['muted']}),
                             dcc.Dropdown(id='sel-des', options=des_options,
                                          value='Pscore_2020', clearable=False)], md=4),
                ])
            ]),
            html.Div(style=CARD, children=[
                dcc.Graph(id='scatter-fig', config=GRAPH_CONFIG),
                html.Div([
                    html.Span("■ Tier 1 — Alta calidad CRVS", style={'color':C['green'],'fontSize':'10px','marginRight':'16px'}),
                    html.Span("■ Tier 2 — Calidad aceptable", style={'color':C['amber'],'fontSize':'10px','marginRight':'16px'}),
                    html.Span("■ Tier 3 — Calidad limitada",  style={'color':C['red'],'fontSize':'10px'}),
                ], style={'marginTop':'6px','textAlign':'center'}),
            ])
        ])
    elif tab == 't3':
        return html.Div(style=CARD, children=[
            make_section_title("Test H2 — Hipótesis Mahajan",
                "¿La categoría de Cumplimiento RSI predice peor que el Sistema de Salud? (Fisher z)"),
            dcc.Graph(figure=fig_mahajan_h2(), config=GRAPH_CONFIG),
        ])
    elif tab == 't4':
        return html.Div(style=CARD, children=[
            make_section_title("Test H3 — Sesgo de autorreporte SPAR",
                "Los países con menor control de corrupción ¿inflan sus scores SPAR?"),
            dcc.Graph(figure=fig_h3_sesgo(), config=GRAPH_CONFIG),
        ])
    elif tab == 't5':
        return html.Div([
            html.Div(style={**CARD,'marginBottom':'8px'}, children=[
                dbc.Row([
                    dbc.Col([html.Label("Predictor", style={'fontSize':'11px','color':C['muted']}),
                             dcc.Dropdown(id='mod-pred', options=pred_options,
                                          value='GHS_total', clearable=False)], md=4),
                    dbc.Col([html.Label("Desenlace", style={'fontSize':'11px','color':C['muted']}),
                             dcc.Dropdown(id='mod-des', options=des_options,
                                          value='Pscore_2020', clearable=False)], md=4),
                ])
            ]),
            html.Div(style=CARD, children=[
                dcc.Graph(id='mod-fig', config=GRAPH_CONFIG),
            ])
        ])
    elif tab == 't6':
        tbl = tabla_resumen()
        if len(tbl) == 0:
            return html.Div("Sin datos suficientes", style={'color':C['muted'],'padding':'20px'})
        return html.Div(style=CARD, children=[
            make_section_title("Tabla A2 — Todas las correlaciones con FDR",
                "Exportable · * p<0.05 · ** p<0.01 · *** p<0.001 (corrección Benjamini-Hochberg)"),
            dash_table.DataTable(
                data=tbl.to_dict('records'),
                columns=[{'name':c,'id':c} for c in tbl.columns],
                **TABLE_STYLE,
                sort_action='native',
                export_format='csv',
                style_data_conditional=[
                    {'if':{'filter_query':'{Sig.FDR} = "*" || {Sig.FDR} = "**" || {Sig.FDR} = "***"'},
                     'backgroundColor':'rgba(30,64,175,0.06)'},
                ],
            )
        ])
    elif tab == 't7':
        cols_show = ['country','GHS_total','SPAR_total','INFORM_Risk_total',
                     'Pscore_2020','Pscore_2021','OMS_excess_2020',
                     'GINI','WGI_Corruption','GovResponse_2020','DataTier']
        cols_show = [c for c in cols_show if c in df.columns]
        return html.Div(style=CARD, children=[
            make_section_title("Tabla maestra LATAM-20",
                f"20 países × {len(df.columns)} variables · Exportable como CSV"),
            dash_table.DataTable(
                data=df[cols_show].round(2).to_dict('records'),
                columns=[{'name':c,'id':c} for c in cols_show],
                **TABLE_STYLE,
                sort_action='native',
                export_format='csv',
            )
        ])
    return html.Div()

@app.callback(Output('scatter-fig','figure'),
              Input('sel-pred','value'), Input('sel-des','value'))
def update_scatter(pred, des):
    return fig_scatter(pred, des)

@app.callback(Output('mod-fig','figure'),
              Input('mod-pred','value'), Input('mod-des','value'))
def update_mod(pred, des):
    return fig_moderacion_gini(pred, des)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8061))
    app.run(debug=False, host='0.0.0.0', port=port)
