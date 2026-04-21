# Metanálisis Índices Pandémicos — LATAM-20

**Validez predictiva de GHS · SPAR · INFORM Risk para el exceso de mortalidad COVID-19**  
Investigadora: ----- · Protocolo pre-registrado OSF · Pendiente 

---

## Acceso en línea

> Los dashboards están disponibles en:  
> **https://latam20-preparedness.onrender.com**  
> *(la primera carga puede tardar ~30 segundos si el servicio estuvo inactivo)*

---

## Estructura del repositorio

```
latam20-preparedness/
│
├── app.py                          # Entry point → Render/gunicorn
├── render.yaml                     # Configuración del deploy
├── requirements.txt                # Dependencias Python
│
├── pages/                          # Dashboards individuales
│   ├── validez_predictiva_dashboard.py   ★ Paper: H2, H3, H4, correlaciones
│   ├── synthesis_dashboard.py            Cross-index metanálisis
│   ├── ghs_dashboard.py                  GHS Index 2019-2021
│   ├── spar_dashboard.py                 SPAR/IHR 2015-2024
│   ├── inform_dashboard.py               INFORM Risk 2017-2025
│   ├── oxcgrt_dashboard.py               Respuesta gubernamental OxCGRT
│   └── severity_dashboard.py             INFORM Severity
│
├── core/                           # Módulos compartidos
│   ├── latam_common.py             Muestra LATAM-20, rutas, subregiones
│   ├── theme.py                    Paleta CDC/Harvard, estilos
│   └── stats_utils.py              FDR, Fisher-z, bootstrap, Wilcoxon
│
├── data/                           # Datos filtrados LATAM-20 (< 8 MB)
│   ├── tabla_maestra_latam20.csv   ★ 20 países × 48 variables
│   ├── 2021-GHS-Index-April-2022.csv
│   ├── INFORM_latam20_trend.csv    (filtrado de 195→20 países)
│   ├── OxCGRT_latam20.csv          (filtrado de 195→20 países)
│   ├── spar_latam.csv
│   ├── spar_global.csv
│   ├── spar_indicators.csv
│   ├── world_mortality.csv
│   ├── WHO_COVID_Excess_Deaths_EstimatesByCountry.xlsx
│   ├── economic-inequality-gini-index.csv
│   └── wgi_governance_data.csv
│
├── static/                         # Figuras HTML exportables
│   ├── INDEX.html                  Página de entrada para evaluadores
│   └── figuras/
│       ├── figura1_heatmap_correlaciones.html
│       ├── figura2_scatter_predictores.html
│       ├── figura3_h2_mahajan.html
│       ├── figura4_h3_sesgo_spar.html
│       ├── figura5_h4_moderacion_gini.html
│       ├── figura6_ranking_comparativo.html
│       └── tabla_maestra_interactiva.html
│
└── docs/                           # Documentación del estudio
    ├── Protocolo_OSF_Rey_LATAM20_Final.docx
    └── INSTRUCCIONES.txt
```

---

## Dashboards disponibles

| Dashboard | Contenido | Relevancia |
|---|---|---|
| **Validez Predictiva** ★ | Correlaciones, H2 Mahajan, H3 sesgo SPAR, H4 GINI, tabla A2 | Paper principal |
| Síntesis Cross-index | Matriz 5×5, radar, ranking compuesto | Exploración |
| GHS Index | 24 figuras, pirámide N1→N4 | Predictor 1 |
| SPAR/IHR | Serie 2015-2024, tendencias FDR | Predictor 2 |
| INFORM Risk | Clustering, brechas LATAM-Global | Predictor 3 |
| OxCGRT | Velocidad respuesta, stringency | Desenlace proceso |
| INFORM Severity | Crisis activas 2019-2026 | Desenlace impacto |

---

## Correr localmente

```bash
# 1. Clonar repositorio
git clone https://github.com/TU_USUARIO/latam20-preparedness
cd latam20-preparedness

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Correr dashboard principal
python app.py
# → http://localhost:8061
```

---

## Datos

Los archivos pesados originales (OxCGRT completo 42 MB, INFORM global 20 MB)  
fueron filtrados a los 20 países de LATAM antes del deploy, manteniendo  
**100% de los datos relevantes** para el análisis.

| Fuente | Archivo en repo | Reducción |
|---|---|---|
| OxCGRT compact (42 MB) | OxCGRT_latam20.csv (4.6 MB) | 89% |
| INFORM Trend (20 MB) | INFORM_latam20_trend.csv (0.6 MB) | 97% |

---

