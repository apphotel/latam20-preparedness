"""
OxCGRT Response Dashboard v4.0 — LATAM-20
══════════════════════════════════════════════════════════════════════
Metanálisis Índices Pandémicos · Desenlace de Respuesta Gubernamental
Fuente: Oxford COVID-19 Government Response Tracker · 2020–2022

CAMBIOS v3.0 → v4.0:
  ✓ Imports centralizados desde latam_common y theme
  ✓ Rutas relativas portables (latam_common.py)
  ✓ Paleta CDC/Harvard clara consolidada en theme.py
  ✓ Exportación SVG vectorial (GRAPH_CONFIG)
  ✓ Muestra LATAM-20 explícita en header

ROL EN EL METANÁLISIS:
  DESENLACE DE PROCESO. Mide cuántas políticas se implementaron y con
  qué grado, NO si fueron efectivas. Un puntaje alto indica respuesta
  robusta, no necesariamente correcta.

OUTLIER METODOLÓGICO — Nicaragua (NIC):
  Único país LATAM-20 sin medidas formales de confinamiento
  (Stringency 2020 ≈ 10.8 vs media LATAM ≈ 61.3). Requiere análisis
  de sensibilidad con y sin Nicaragua en la regresión principal.
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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

# ══════════════════════════════════════════════════════════════════════
# 1. CARGA
# ══════════════════════════════════════════════════════════════════════
df_raw = pd.read_csv(data_path('oxcgrt_compact'), low_memory=False)
df_raw['Date'] = pd.to_datetime(df_raw['Date'], format='%Y%m%d')
latam = df_raw[
    (df_raw['CountryCode'].isin(LATAM_ISO3.keys())) &
    (df_raw['Jurisdiction']=='NAT_TOTAL')
].copy()
latam['country'] = latam['CountryCode'].map(LATAM_ISO3)
latam['year']    = latam['Date'].dt.year
latam['month']   = latam['Date'].dt.to_period('M').astype(str)

# Colores específicos OxCGRT desde theme
STRINGENCY_C  = C['stringency']
GOV_C         = C['gov']
CONTAINMENT_C = C['containment']
ECONOMIC_C    = C['economic']
NIC_C         = C['red']  # highlight Nicaragua

INDICES = {
    'StringencyIndex_Average':        ('Stringency Index',        STRINGENCY_C),
    'GovernmentResponseIndex_Average':('Gov. Response Index',     GOV_C),
    'ContainmentHealthIndex_Average': ('Containment & Health',    CONTAINMENT_C),
    'EconomicSupportIndex':           ('Economic Support Index',  ECONOMIC_C),
}
INDIV_INDICATORS = {
    'C1M_School closing':              'Cierre de escuelas',
    'C2M_Workplace closing':           'Cierre de negocios',
    'C6M_Stay at home requirements':   'Confinamiento en hogares',
    'C8EV_International travel controls':'Control viajes internacionales',
    'H2_Testing policy':               'Política de testeo',
    'H3_Contact tracing':              'Rastreo de contactos',
}

pandemic   = latam[latam['year'].isin([2020, 2021])].copy()
pandemic22 = latam[latam['year'].isin([2020, 2021, 2022])].copy()

# ══════════════════════════════════════════════════════════════════════
# 2. MÉTRICAS POR PAÍS
# ══════════════════════════════════════════════════════════════════════
records = []
for iso, country in LATAM_ISO3.items():
    for yr in [2020, 2021]:
        sub = pandemic[(pandemic['CountryCode']==iso) & (pandemic['year']==yr)]
        if len(sub) == 0: continue
        days_hi = int((sub['StringencyIndex_Average'] > 60).sum())
        days_md = int((sub['StringencyIndex_Average'] > 40).sum())
        above50 = sub[sub['StringencyIndex_Average'] > 50].sort_values('Date')
        speed = int(above50['Date'].iloc[0].dayofyear) if len(above50) else 999
        records.append({
            'iso3': iso, 'country': country, 'year': yr,
            'stringency_mean':  round(sub['StringencyIndex_Average'].mean(), 1),
            'stringency_max':   round(sub['StringencyIndex_Average'].max(), 1),
            'gov_mean':         round(sub['GovernmentResponseIndex_Average'].mean(), 1),
            'containment_mean': round(sub['ContainmentHealthIndex_Average'].mean(), 1),
            'economic_mean':    round(sub['EconomicSupportIndex'].mean(), 1),
            'days_high60':      days_hi,
            'days_mod40':       days_md,
            'speed_day':        speed,
            'is_nicaragua':     (iso == 'NIC'),
        })
df_metrics = pd.DataFrame(records)

monthly_all = pandemic22.groupby('month')['StringencyIndex_Average'].mean().reset_index()
monthly_no_nic = pandemic22[pandemic22['CountryCode'] != 'NIC'].groupby(
    'month')['StringencyIndex_Average'].mean().reset_index()


# ══════════════════════════════════════════════════════════════════════
# 3. FIGURAS
# ══════════════════════════════════════════════════════════════════════

def fig_serie_temporal():
    fig = go.Figure()
    fig.add_vrect(x0='2020-03', x1='2021-12',
                  fillcolor=C['blue'], opacity=0.04, layer='below', line_width=0)
    fig.add_annotation(x='2020-07', y=92, text='Pico pandemia',
                       showarrow=False, font=dict(size=9, color=C['blue']),
                       bgcolor='rgba(255,255,255,0.8)')
    fig.add_trace(go.Scatter(
        x=monthly_all['month'], y=monthly_all['StringencyIndex_Average'],
        mode='lines', name='LATAM-20 (con Nicaragua)',
        line=dict(color=STRINGENCY_C, width=2.5),
        hovertemplate='%{x}: <b>%{y:.1f}</b><extra>Con Nicaragua</extra>'))
    fig.add_trace(go.Scatter(
        x=monthly_no_nic['month'], y=monthly_no_nic['StringencyIndex_Average'],
        mode='lines', name='LATAM-19 (sin Nicaragua)',
        line=dict(color=STRINGENCY_C, width=1.5, dash='dot'),
        hovertemplate='%{x}: <b>%{y:.1f}</b><extra>Sin Nicaragua</extra>'))
    peak_idx = monthly_all['StringencyIndex_Average'].idxmax()
    peak_m = monthly_all.loc[peak_idx, 'month']
    peak_v = monthly_all.loc[peak_idx, 'StringencyIndex_Average']
    fig.add_trace(go.Scatter(
        x=[peak_m], y=[peak_v], mode='markers+text',
        marker=dict(color=C['red'], size=10, symbol='star'),
        text=[f'Pico: {peak_v:.1f}'], textposition='top center',
        textfont=dict(size=9, color=C['red']), showlegend=False,
        hovertemplate=f'Pico LATAM: <b>{peak_v:.1f}</b> ({peak_m})<extra></extra>'))
    for val, lbl in [(75, 'Alto (75)'), (50, 'Moderado (50)'), (25, 'Bajo (25)')]:
        fig.add_hline(y=val, line=dict(color=C['muted'], width=0.7, dash='dot'),
                      annotation_text=lbl, annotation_position='right',
                      annotation_font=dict(size=8, color=C['muted']))
    fig.update_layout(**{**LAYOUT, 'height': 440},
        title=dict(text='Stringency Index promedio LATAM-20 · Serie mensual 2020–2022',
                   font=dict(size=13)),
        xaxis=dict(title='', gridcolor=C['grid'], tickangle=-45,
                   tickfont=dict(size=9)),
        yaxis=dict(title='Stringency Index (0–100)', gridcolor=C['grid'],
                   range=[0, 105]),
        legend=dict(x=0.7, y=0.95, font=dict(size=10)),
        hovermode='x unified')
    return fig


def fig_ranking_stringency(year):
    sub = df_metrics[df_metrics['year']==year].sort_values('stringency_mean', ascending=True)
    colors = [NIC_C if r['is_nicaragua'] else STRINGENCY_C for _, r in sub.iterrows()]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[sn(c) for c in sub['country']],
        x=sub['stringency_mean'], orientation='h',
        marker=dict(color=colors, line=dict(width=0.5, color=C['border'])),
        customdata=sub[['stringency_max', 'days_high60', 'gov_mean']].values,
        hovertemplate=('<b>%{y}</b><br>Stringency medio: <b>%{x:.1f}</b><br>'
                       'Máximo: %{customdata[0]:.1f}<br>'
                       'Días >60: %{customdata[1]}<br>'
                       'Gov. Response: %{customdata[2]:.1f}<extra></extra>')))
    mean_val = sub['stringency_mean'].mean()
    fig.add_vline(x=mean_val, line=dict(color=C['amber'], width=1.3, dash='dash'),
                  annotation_text=f'Media: {mean_val:.1f}',
                  annotation_font=dict(size=9, color=C['amber']))
    nic_val = sub[sub['country']=='Nicaragua']['stringency_mean'].values
    if len(nic_val):
        fig.add_annotation(
            x=nic_val[0]+2, y=sn('Nicaragua'),
            text='⚠️ Sin confinamiento formal',
            showarrow=True, arrowhead=2, arrowcolor=NIC_C,
            font=dict(size=9, color=NIC_C), ax=80, ay=0)
    fig.update_layout(**{**LAYOUT, 'height': 560},
        title=dict(text=f'Stringency Index medio · LATAM-20 · {year}',
                   font=dict(size=13)),
        xaxis=dict(title='Stringency Index promedio (0–100)', gridcolor=C['grid']),
        yaxis=dict(tickfont=dict(size=10), gridcolor=C['grid']),
        showlegend=False)
    return fig


def fig_cuatro_indices(year):
    sub = df_metrics[df_metrics['year']==year].sort_values('gov_mean', ascending=False)
    fig = go.Figure()
    for _, row in sub.iterrows():
        col = NIC_C if row['is_nicaragua'] else GOV_C
        opacity = 1.0 if not row['is_nicaragua'] else 0.9
        fig.add_trace(go.Bar(
            name=sn(row['country']),
            x=['Stringency', 'Gov. Response', 'Contención & Salud', 'Apoyo Económico'],
            y=[row['stringency_mean'], row['gov_mean'],
               row['containment_mean'], row['economic_mean']],
            marker=dict(color=col, opacity=opacity,
                        line=dict(width=0.3, color=C['border'])),
            hovertemplate=f'<b>{sn(row["country"])}</b><br>%{{x}}: <b>%{{y:.1f}}</b><extra></extra>'))
    means = df_metrics[df_metrics['year']==year][
        ['stringency_mean', 'gov_mean', 'containment_mean', 'economic_mean']].mean()
    fig.add_trace(go.Scatter(
        x=['Stringency', 'Gov. Response', 'Contención & Salud', 'Apoyo Económico'],
        y=[means['stringency_mean'], means['gov_mean'],
           means['containment_mean'], means['economic_mean']],
        mode='lines+markers', name='Media LATAM-20',
        line=dict(color=C['amber'], width=2.5),
        marker=dict(size=9, color=C['amber'],
                    line=dict(width=1.5, color=C['ink'])),
        hovertemplate='Media LATAM: <b>%{y:.1f}</b><extra></extra>'))
    fig.update_layout(**{**LAYOUT, 'height': 480}, barmode='overlay',
        title=dict(text=f'Los 4 índices OxCGRT · LATAM-20 · {year}',
                   font=dict(size=13)),
        xaxis=dict(tickfont=dict(size=11), gridcolor=C['grid']),
        yaxis=dict(title='Score (0–100)', gridcolor=C['grid'], range=[0, 105]),
        legend=dict(font=dict(size=8), x=1.01, y=1))
    return fig


def fig_heatmap_indiv(year):
    sub = latam[latam['year']==year]
    rows = []
    countries_sorted = df_metrics[df_metrics['year']==year].sort_values(
        'gov_mean', ascending=False)['country'].tolist()
    for country in countries_sorted:
        row_data = {'country': country}
        csub = sub[sub['country']==country]
        for col, name in INDIV_INDICATORS.items():
            row_data[name] = round(csub[col].mean(), 2) if col in csub.columns else np.nan
        rows.append(row_data)
    df_heat = pd.DataFrame(rows).set_index('country')
    ind_names = list(INDIV_INDICATORS.values())
    fig = go.Figure(go.Heatmap(
        z=df_heat.values, x=ind_names, y=[sn(c) for c in df_heat.index],
        colorscale='Blues',
        text=np.round(df_heat.values, 1), texttemplate='%{text}',
        textfont=dict(size=9, color=C['ink']),
        hovertemplate='<b>%{y}</b> · %{x}<br>Score: <b>%{z:.2f}</b><extra></extra>',
        colorbar=dict(title=dict(text='Score', font=dict(color=C['ink'])),
                      tickfont=dict(size=9, color=C['ink']), thickness=14)))
    fig.update_layout(**{**LAYOUT, 'margin': dict(l=130, r=30, t=45, b=120),
                         'height': 580},
        title=dict(text=f'Indicadores individuales de respuesta · LATAM-20 · {year}',
                   font=dict(size=13)),
        xaxis=dict(tickangle=-35, tickfont=dict(size=10), gridcolor=C['grid']),
        yaxis=dict(tickfont=dict(size=10), gridcolor=C['grid']))
    return fig


def fig_velocidad():
    sub2020 = df_metrics[df_metrics['year']==2020].copy()
    sub2020 = sub2020[sub2020['speed_day'] < 999].sort_values('speed_day')
    colors = [NIC_C if r['is_nicaragua'] else C['teal']
              for _, r in sub2020.iterrows()]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[sn(c) for c in sub2020['country']],
        x=sub2020['speed_day'], orientation='h',
        marker=dict(color=colors, line=dict(width=0.5, color=C['border'])),
        hovertemplate='<b>%{y}</b><br>Día del año: <b>%{x}</b><extra></extra>'))
    fig.add_vline(x=71, line=dict(color=C['covid'], width=1.5, dash='dash'),
                  annotation_text='11-Mar (OMS declara pandemia)',
                  annotation_font=dict(size=9, color=C['covid']),
                  annotation_position='top right')
    fig.update_layout(**{**LAYOUT, 'height': 500},
        title=dict(text='Velocidad de respuesta · Día del año en que Stringency superó 50 · 2020',
                   font=dict(size=13)),
        xaxis=dict(title='Día del año 2020 (1 = 1 enero)',
                   gridcolor=C['grid'], range=[0, 110]),
        yaxis=dict(tickfont=dict(size=10), gridcolor=C['grid']),
        showlegend=False)
    return fig


def fig_serie_pais(country):
    sub = latam[(latam['country']==country) &
                (latam['year'].isin([2020, 2021, 2022]))].sort_values('Date')
    fig = go.Figure()
    fig.add_vrect(x0='2020-03-01', x1='2021-12-31',
                  fillcolor=C['blue'], opacity=0.05, layer='below', line_width=0)
    for idx_col, (idx_name, idx_col_c) in INDICES.items():
        fig.add_trace(go.Scatter(
            x=sub['Date'], y=sub[idx_col],
            mode='lines', name=idx_name,
            line=dict(color=idx_col_c, width=1.8),
            hovertemplate=f'%{{x|%b %Y}}: <b>%{{y:.1f}}</b><extra>{idx_name}</extra>'))
    for val, lbl in [(75, 'Alto'), (50, 'Moderado'), (25, 'Bajo')]:
        fig.add_hline(y=val, line=dict(color=C['muted'], width=0.5, dash='dot'),
                      annotation_text=lbl,
                      annotation_font=dict(size=8, color=C['muted']),
                      annotation_position='right')
    fig.update_layout(**{**LAYOUT, 'height': 440},
        title=dict(text=f'Serie temporal OxCGRT · {country} · 2020–2022',
                   font=dict(size=13)),
        xaxis=dict(gridcolor=C['grid'], tickfont=dict(size=9)),
        yaxis=dict(title='Score (0–100)', gridcolor=C['grid'], range=[0, 105]),
        legend=dict(font=dict(size=10), x=0, y=-0.2, orientation='h'),
        hovermode='x unified')
    return fig


def fig_perfil_respuesta():
    sub = df_metrics[df_metrics['year']==2020].copy()
    fig = go.Figure()
    for _, row in sub.iterrows():
        col = NIC_C if row['is_nicaragua'] else STRINGENCY_C
        size = max(row['economic_mean']/3, 8)
        fig.add_trace(go.Scatter(
            x=[row['stringency_mean']], y=[row['days_high60']],
            mode='markers+text',
            marker=dict(color=col, size=size,
                        line=dict(width=1, color='white'), opacity=0.85),
            text=[sn(row['country'])], textposition='top center',
            textfont=dict(size=9, color=C['ink']),
            name=sn(row['country']), showlegend=False,
            hovertemplate=(f'<b>{sn(row["country"])}</b><br>'
                           f'Stringency medio: {row["stringency_mean"]:.1f}<br>'
                           f'Días >60: {row["days_high60"]}<br>'
                           f'Apoyo económico: {row["economic_mean"]:.1f}<extra></extra>')))
    mean_s = sub['stringency_mean'].mean()
    mean_d = sub['days_high60'].mean()
    fig.add_vline(x=mean_s, line=dict(color=C['muted'], width=0.8, dash='dot'))
    fig.add_hline(y=mean_d, line=dict(color=C['muted'], width=0.8, dash='dot'))
    fig.add_annotation(x=80, y=350, text='Alta intensidad<br>Alta duración',
                       font=dict(size=9, color=C['red']), showarrow=False,
                       bgcolor='rgba(255,255,255,0.7)')
    fig.add_annotation(x=25, y=50, text='Baja intensidad<br>Baja duración',
                       font=dict(size=9, color=C['green']), showarrow=False,
                       bgcolor='rgba(255,255,255,0.7)')
    fig.update_layout(**{**LAYOUT, 'height': 520},
        title=dict(text='Perfil 2020: Intensidad × Duración (tamaño = apoyo económico)',
                   font=dict(size=13)),
        xaxis=dict(title='Stringency medio 2020', gridcolor=C['grid']),
        yaxis=dict(title='Días con Stringency > 60', gridcolor=C['grid']))
    return fig


def build_table():
    sub = df_metrics.copy()
    sub['Nicaragua'] = sub['country'].apply(lambda x: '⚠️' if x == 'Nicaragua' else '')
    display = sub[['country','year','stringency_mean','stringency_max',
                   'gov_mean','containment_mean','economic_mean',
                   'days_high60','speed_day','Nicaragua']].copy()
    display.columns = ['País','Año','Stringency medio','Stringency máx.',
                       'Gov. Response','Contención & Salud','Apoyo Económico',
                       'Días >60','Día respuesta','Nota']
    display['Día respuesta'] = display['Día respuesta'].apply(
        lambda x: '—' if x == 999 else str(x))
    return display


TABLE_DF = build_table()

# Pre-compute figuras estáticas
FIG_SERIE     = fig_serie_temporal()
FIG_RANK_2020 = fig_ranking_stringency(2020)
FIG_RANK_2021 = fig_ranking_stringency(2021)
FIG_VELOCIDAD = fig_velocidad()
FIG_PERFIL    = fig_perfil_respuesta()

# Stats para KPIs
m2020 = df_metrics[df_metrics['year']==2020]
max_row = m2020.loc[m2020['gov_mean'].idxmax()]
min_row = m2020.loc[m2020['gov_mean'].idxmin()]
peak_stringency = float(monthly_all['StringencyIndex_Average'].max())


# ══════════════════════════════════════════════════════════════════════
# 4. APP LAYOUT
# ══════════════════════════════════════════════════════════════════════
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True, title='OxCGRT v4.0')
server = app.server

app.layout = dbc.Container([
    make_header('OxCGRT · RESPUESTA GUBERNAMENTAL COVID-19',
                f'LATAM-20 · Oxford COVID-19 Government Response Tracker · 2020–2022 · DESENLACE DE PROCESO'),
    make_methodology_note(
        'ROL: Desenlace de proceso — ¿los países con mayor preparación (GHS/SPAR/INFORM) '
        'respondieron más rápido y con mayor intensidad? '
        'LIMITACIÓN: los índices miden cuántas políticas se implementaron, NO si fueron '
        'efectivas. Un puntaje alto indica respuesta robusta, no necesariamente correcta. '
        'OUTLIER Nicaragua: único país LATAM-20 sin medidas formales de confinamiento '
        '(Stringency 2020 ≈ 10.8 vs media ≈ 61). Requiere análisis de sensibilidad con y '
        f'sin Nicaragua en la regresión principal. {NOTA_MUESTRA}',
        accent='teal'),
    dbc.Row([
        make_kpi('20', 'PAÍSES', 'LATAM-20 completo', C['ink']),
        make_kpi('20/20', 'COBERTURA', 'Todos los países', C['blue']),
        make_kpi(f'{max_row["gov_mean"]:.1f}', 'MAYOR GOV. RESPONSE',
                 f'{sn(max_row["country"])} 2020', C['blue']),
        make_kpi(f'{min_row["gov_mean"]:.1f}', 'MENOR GOV. RESPONSE',
                 f'{sn(min_row["country"])} 2020', C['red']),
        make_kpi(f'{peak_stringency:.1f}', 'PICO LATAM',
                 'Stringency abr–may 2020', C['covid']),
        make_kpi('NIC 10.8', 'OUTLIER',
                 'Sin confinamiento formal', C['orange']),
    ], style={'marginBottom': '10px'}),
    dcc.Tabs(id='tabs', value='t1', children=[
        dcc.Tab(label='📈 Serie temporal',      value='t1', style=TS, selected_style=TSS),
        dcc.Tab(label='Ranking 2020',           value='t2', style=TS, selected_style=TSS),
        dcc.Tab(label='Ranking 2021',           value='t3', style=TS, selected_style=TSS),
        dcc.Tab(label='4 Índices comparados',   value='t4', style=TS, selected_style=TSS),
        dcc.Tab(label='Indicadores individuales', value='t5', style=TS, selected_style=TSS),
        dcc.Tab(label='⚡ Velocidad respuesta',  value='t6', style=TS, selected_style=TSS),
        dcc.Tab(label='Perfil intensidad×duración', value='t7', style=TS, selected_style=TSS),
        dcc.Tab(label='Serie por país',         value='t8', style=TS, selected_style=TSS),
        dcc.Tab(label='Tabla maestra',          value='t9', style=TS, selected_style=TSS),
    ], style={'borderBottom': f'1px solid {C["border"]}', 'marginBottom': '14px'}),
    html.Div(id='tab-content')
], fluid=True, style={'backgroundColor': C['bg'], 'minHeight': '100vh',
                      'padding': '0 20px 40px'})


# ══════════════════════════════════════════════════════════════════════
# 5. CALLBACKS
# ══════════════════════════════════════════════════════════════════════
@app.callback(Output('tab-content', 'children'), Input('tabs', 'value'))
def render(tab):
    if tab == 't1':
        return html.Div([html.Div([
            make_section_title('Stringency Index mensual LATAM-20 · 2020–2022',
                'El pico regional fue abr–may 2020. Línea punteada = LATAM-19 sin Nicaragua. '
                'La brecha entre ambas líneas es la "distancia Nicaragua" — hallazgo metodológico '
                'relevante para el análisis de moderación.'),
            dcc.Graph(figure=FIG_SERIE, config=GRAPH_CONFIG)], style=CARD)])

    elif tab == 't2':
        return html.Div([html.Div([
            make_section_title('Ranking por Stringency medio 2020',
                'Honduras y Argentina lideraron la intensidad de respuesta. Nicaragua (rojo) es '
                'el outlier estadístico que requiere tratamiento especial en la regresión.'),
            dcc.Graph(figure=FIG_RANK_2020, config=GRAPH_CONFIG)], style=CARD)])

    elif tab == 't3':
        return html.Div([html.Div([
            make_section_title('Ranking por Stringency medio 2021',
                'En 2021 la varianza entre países aumenta: Honduras mantiene el mayor Stringency, '
                'mientras Bolivia y El Salvador caen drásticamente tras levantar restricciones. '
                'Patrón reflejando fatiga pandémica y presión económica.'),
            dcc.Graph(figure=FIG_RANK_2021, config=GRAPH_CONFIG)], style=CARD)])

    elif tab == 't4':
        return html.Div([html.Div([
            make_section_title('Los 4 índices OxCGRT comparados',
                'Stringency y Gov. Response se mueven juntos, pero Economic Support varía '
                'independientemente — refleja capacidad fiscal, no solo voluntad política.'),
            dcc.RadioItems(
                id='year-4idx',
                options=[{'label': '2020', 'value': 2020}, {'label': '2021', 'value': 2021}],
                value=2020, inline=True,
                labelStyle={'marginRight': '20px', 'fontSize': '12px',
                            'color': C['text']},
                style={'marginBottom': '12px'}),
            html.Div(id='graph-4idx')], style=CARD)])

    elif tab == 't5':
        return html.Div([html.Div([
            make_section_title('Indicadores individuales de respuesta',
                'El cierre de escuelas fue casi universal, pero el rastreo de contactos '
                '— indicador de vigilancia — muestra alta variabilidad. '
                'Esta dimensión conecta directamente con los predictores SPAR y GHS.'),
            dcc.RadioItems(
                id='year-heat',
                options=[{'label': '2020', 'value': 2020}, {'label': '2021', 'value': 2021}],
                value=2020, inline=True,
                labelStyle={'marginRight': '20px', 'fontSize': '12px',
                            'color': C['text']},
                style={'marginBottom': '12px'}),
            html.Div(id='graph-heat')], style=CARD)])

    elif tab == 't6':
        return html.Div([html.Div([
            make_section_title('Velocidad de respuesta 2020',
                'Día del año en que Stringency superó 50. La mayoría de países respondió en '
                'la semana del 15 al 19 de marzo — cuando la OMS declaró pandemia (11 mar, '
                'día 71). Sincronización regional notable: 18 de 20 países en ~10 días. '
                'Nicaragua nunca superó el umbral de 50.'),
            dcc.Graph(figure=FIG_VELOCIDAD, config=GRAPH_CONFIG)], style=CARD)])

    elif tab == 't7':
        return html.Div([html.Div([
            make_section_title('Perfil intensidad × duración 2020',
                'Cuadrantes: alta/baja intensidad × alta/baja duración. '
                'Argentina, Honduras y Venezuela: respuestas largas e intensas. '
                'Uruguay: respuesta corta y suave inicialmente, aumentada en 2021. '
                'Tamaño de burbuja = apoyo económico.'),
            dcc.Graph(figure=FIG_PERFIL, config=GRAPH_CONFIG)], style=CARD)])

    elif tab == 't8':
        return html.Div([html.Div([
            make_section_title('Serie temporal por país — 4 índices',
                'Evolución de los 4 índices OxCGRT durante la pandemia'),
            dcc.Dropdown(
                id='country-select',
                options=[{'label': c, 'value': c} for c in LATAM_20_SORTED],
                value='Peru', clearable=False,
                style={**DROPDOWN_STYLE, 'width': '260px', 'marginBottom': '12px'}),
            html.Div(id='graph-pais')], style=CARD)])

    elif tab == 't9':
        return html.Div([html.Div([
            make_section_title('Tabla maestra OxCGRT · LATAM-20 · 2020–2021',
                'Gov. Response = índice compuesto de todas las políticas. '
                'Días >60 = días del año con Stringency > 60. '
                'Día respuesta = día del año en que Stringency superó 50.'),
            dash_table.DataTable(
                data=TABLE_DF.to_dict('records'),
                columns=[{'name': c, 'id': c} for c in TABLE_DF.columns],
                filter_action='native', sort_action='native',
                page_size=20, export_format='csv', **TABLE_STYLE,
                style_data_conditional=[
                    {'if': {'filter_query': '{País} = Nicaragua'},
                     'color': C['red'], 'fontWeight': '700'},
                ])], style=CARD)])

    return html.Div()


@app.callback(Output('graph-4idx', 'children'), Input('year-4idx', 'value'))
def update_4idx(year):
    return dcc.Graph(figure=fig_cuatro_indices(year), config=GRAPH_CONFIG)


@app.callback(Output('graph-heat', 'children'), Input('year-heat', 'value'))
def update_heat(year):
    return dcc.Graph(figure=fig_heatmap_indiv(year), config=GRAPH_CONFIG)


@app.callback(Output('graph-pais', 'children'), Input('country-select', 'value'))
def update_pais(country):
    return dcc.Graph(figure=fig_serie_pais(country), config=GRAPH_CONFIG)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8055))
    app.run(debug=False, host='0.0.0.0', port=port)
