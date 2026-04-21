"""
latam_common.py — Definiciones canónicas compartidas por los seis dashboards
del metanálisis de índices pandémicos (GHS · SPAR · INFORM Risk · INFORM Severity · OxCGRT).

USO:
    from latam_common import LATAM_20, LATAM_ISO3, YEARS, SHORT, sn, data_path

RATIONALE:
    Antes había cinco copias divergentes de la lista de países y el mapeo ISO3.
    Este módulo garantiza que la muestra LATAM-20 sea idéntica en todo el metanálisis.
"""

from __future__ import annotations
import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# MUESTRA CANÓNICA: LATAM-20
# ──────────────────────────────────────────────────────────────
# Criterio: países de habla latina (español/portugués/francés) de América.
# Excluidos de LAC-24: Jamaica (inglés), Trinidad & Tobago (inglés),
#                      Guyana (inglés), Surinam (neerlandés).

LATAM_ISO3: dict[str, str] = {
    # Sudamérica (10)
    'ARG': 'Argentina',  'BOL': 'Bolivia',  'BRA': 'Brazil',    'CHL': 'Chile',
    'COL': 'Colombia',   'ECU': 'Ecuador',  'PRY': 'Paraguay',  'PER': 'Peru',
    'URY': 'Uruguay',    'VEN': 'Venezuela',
    # Centroamérica (6)
    'CRI': 'Costa Rica', 'SLV': 'El Salvador', 'GTM': 'Guatemala',
    'HND': 'Honduras',   'NIC': 'Nicaragua',   'PAN': 'Panama',
    # Norteamérica (1)
    'MEX': 'Mexico',
    # Caribe (3)
    'CUB': 'Cuba', 'HTI': 'Haiti', 'DOM': 'Dominican Republic',
}

NAME_TO_ISO: dict[str, str] = {v: k for k, v in LATAM_ISO3.items()}
LATAM_20: list[str] = list(LATAM_ISO3.values())  # 20 nombres ordenados por subregión
LATAM_20_SORTED: list[str] = sorted(LATAM_20)     # alfabético (para dropdowns)
N_LATAM: int = len(LATAM_20)                      # 20

assert N_LATAM == 20, "LATAM_20 debe contener exactamente 20 países"

# ──────────────────────────────────────────────────────────────
# SUBREGIONES
# ──────────────────────────────────────────────────────────────
SUBREGION: dict[str, str] = {
    **{c: 'Sudamérica' for c in ['Argentina','Bolivia','Brazil','Chile','Colombia',
                                  'Ecuador','Paraguay','Peru','Uruguay','Venezuela']},
    **{c: 'Centroamérica' for c in ['Costa Rica','El Salvador','Guatemala',
                                     'Honduras','Nicaragua','Panama']},
    'Mexico': 'Norteamérica',
    **{c: 'Caribe' for c in ['Cuba','Haiti','Dominican Republic']},
}

# ──────────────────────────────────────────────────────────────
# RANGO TEMPORAL POR FUENTE
# ──────────────────────────────────────────────────────────────
YEARS_INFORM  = list(range(2017, 2026))   # 2017–2025
YEARS_SPAR    = list(range(2018, 2025))   # 2018–2024
YEARS_SPAR_P1 = [2018, 2019, 2020]        # 13 capacidades
YEARS_SPAR_P2 = [2021, 2022, 2023, 2024]  # 15 capacidades
YEARS_GHS     = [2019, 2021]              # solo dos ediciones
YEARS_OXCGRT  = [2020, 2021, 2022]        # respuesta COVID
YEARS_SEVERITY = list(range(2019, 2027))  # 2019–2026

# Años canónicos de inflexión
YR_BASAL   = 2019  # pre-pandemia
YR_PANDEM  = 2020  # primera respuesta
YR_POST1   = 2021  # post primer año COVID
YR_ACTUAL  = 2025  # snapshot reciente

# ──────────────────────────────────────────────────────────────
# ABREVIATURAS PARA GRÁFICOS (evitar desbordes en ejes)
# ──────────────────────────────────────────────────────────────
SHORT: dict[str, str] = {
    'Dominican Republic': 'Dom. Rep.',
    'El Salvador':        'El Salv.',
    'Costa Rica':         'C. Rica',
}

def sn(country: str) -> str:
    """Nombre corto para uso en gráficos/tablas compactas."""
    return SHORT.get(country, country)

# ──────────────────────────────────────────────────────────────
# RUTAS DE DATOS (relativas y portables)
# ──────────────────────────────────────────────────────────────
# Por orden de prioridad:
#   1. Variable de entorno METANALISIS_DATA
#   2. ./data/ junto al script
#   3. ~/metanalisis_data/
# Esto permite correr el dashboard en local, servidor o contenedor
# sin tocar ninguna línea de código.

def _resolve_data_dir() -> Path:
    """
    Resuelve la ruta a los datos de forma portable.
    Prioridad:
      1. Variable de entorno METANALISIS_DATA
      2. Carpeta 'data/' junto al SCRIPT (usando __file__) ← Fix principal
      3. Carpeta 'data/' en el directorio de trabajo actual
      4. ~/metanalisis_data/
    """
    # 1. Variable de entorno explícita
    env = os.environ.get('METANALISIS_DATA')
    if env and Path(env).is_dir():
        return Path(env)

    # 2. *** FIX PRINCIPAL ***
    # __file__ = ruta de latam_common.py en disco
    # Siempre funciona sin importar desde dónde se ejecute el script
    script_dir = Path(__file__).resolve().parent
    # En estructura con core/, subir un nivel para llegar a la raíz
    script_data = script_dir / 'data'
    if not script_data.is_dir():
        script_data = script_dir.parent / 'data'
    if script_data.is_dir():
        return script_data

    # 3. Directorio de trabajo actual
    cwd_data = Path.cwd() / 'data'
    if cwd_data.is_dir():
        return cwd_data

    # 4. Carpeta home
    home_data = Path.home() / 'metanalisis_data'
    if home_data.is_dir():
        return home_data

    # Fallback con mensaje claro
    print(f"\n⚠️  ADVERTENCIA: no se encontró la carpeta 'data/'")
    print(f"   Buscado en: {script_data}")
    print(f"   Solución: asegúrate de que la carpeta 'data/' esté junto a los scripts .py")
    return script_data

DATA_DIR: Path = _resolve_data_dir()

# Archivos canónicos por fuente
FILES = {
    # Archivos LATAM-20 filtrados (GitHub-friendly, <5 MB total)
    'inform_trend':    'INFORM_latam20_trend.csv',       # filtrado: 20 países × indicadores clave
    'inform_mid':      'INFORM_latam20_trend.csv',       # alias — misma fuente
    'inform_severity': '202602_INFORM_Severity_-_February_2026.xlsx',  # opcional
    'ghs':             '2021-GHS-Index-April-2022.csv',
    'spar_latam':      'spar_latam.csv',
    'spar_global':     'spar_global.csv',
    'spar_indicators': 'spar_indicators.csv',
    'oxcgrt_compact':  'OxCGRT_latam20.csv',            # filtrado: 20 países (1.1 MB)
    'oxcgrt_simple':   'OxCGRT_latam20.csv',            # alias
    'world_mortality':  'world_mortality.csv',
    'oms_excess':       'WHO_COVID_Excess_Deaths_EstimatesByCountry.xlsx',
    'gini':             'economic-inequality-gini-index.csv',
    'wgi':              'wgi_governance_data.csv',
    'tabla_maestra':    'tabla_maestra_latam20.csv',
}

def data_path(key: str) -> Path:
    """
    Devuelve la ruta al archivo de datos identificado por key.
    Lanza FileNotFoundError con mensaje claro si no está presente.
    """
    if key not in FILES:
        raise KeyError(
            f"Clave de archivo desconocida: '{key}'. Disponibles: {list(FILES)}"
        )
    p = DATA_DIR / FILES[key]
    if not p.exists():
        raise FileNotFoundError(
            f"\n❌ No se encuentra el archivo '{FILES[key]}'.\n"
            f"   Directorio buscado: {DATA_DIR}\n"
            f"   Solución: coloca el archivo ahí, o define METANALISIS_DATA=<ruta>."
        )
    return p


# ──────────────────────────────────────────────────────────────
# METADATOS DEL METANÁLISIS
# ──────────────────────────────────────────────────────────────
METANALISIS = dict(
    muestra='LATAM-20',
    n=20,
    excluidos_lac24=['Jamaica', 'Trinidad and Tobago', 'Guyana', 'Suriname'],
    criterio_exclusion='habla no-latina (inglés/neerlandés)',
    indices=['GHS', 'SPAR', 'INFORM Risk', 'INFORM Severity', 'OxCGRT'],
    autor='Gisselle Rey',
    version='v3.0',
)

NOTA_MUESTRA = (
    'LATAM-20 (n=20): países de habla latina de América. '
    'Excluidos de LAC-24: Jamaica, Trinidad & Tobago, Guyana, Surinam.'
)


if __name__ == '__main__':
    # Auto-test rápido
    print(f'LATAM-20: {N_LATAM} países')
    print(f'DATA_DIR: {DATA_DIR} (existe: {DATA_DIR.is_dir()})')
    print(f'Primer país: {LATAM_20[0]} → {NAME_TO_ISO[LATAM_20[0]]}')
    print(f'Subregiones: {set(SUBREGION.values())}')
