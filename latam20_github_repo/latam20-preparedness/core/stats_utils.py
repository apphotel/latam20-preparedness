"""
stats_utils.py — Utilidades estadísticas rigurosas para el metanálisis.

Corrige problemas metodológicos identificados en los dashboards originales:
    1. Wilcoxon con emparejamiento robusto (dropna conjunto).
    2. Tamaños de efecto (Cohen's d, Cliff's delta, r de Rosenthal).
    3. Corrección por comparaciones múltiples (Benjamini-Hochberg FDR).
    4. Intervalos de confianza bootstrap.
    5. Regresión lineal de tendencia con IC y clasificación por significancia.

Todas las funciones devuelven diccionarios estructurados para facilitar
su integración en tablas Dash.
"""

from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd
from scipy import stats
try:
    from statsmodels.stats.multitest import multipletests
    _HAS_SM = True
except ImportError:
    _HAS_SM = False


# ══════════════════════════════════════════════════════════════════════
# TESTS APAREADOS ROBUSTOS
# ══════════════════════════════════════════════════════════════════════

def wilcoxon_paired(
    x: Sequence[float],
    y: Sequence[float],
    labels: Sequence[str] | None = None,
) -> dict:
    """
    Test de Wilcoxon apareado con emparejamiento robusto.

    - Construye pares (x_i, y_i) y elimina filas con NaN en cualquier columna.
    - Calcula r de Rosenthal (tamaño de efecto) = |Z| / √N.
    - Calcula Cliff's delta como tamaño de efecto no paramétrico.

    Parámetros
    ----------
    x, y : secuencias numéricas de igual longitud
    labels : opcional, etiquetas para los pares (útil para diagnóstico)

    Devuelve
    --------
    dict con: n, W, p, r_rosenthal, cliffs_delta, median_diff,
              ci95_diff (bootstrap), interpretation
    """
    df = pd.DataFrame({'x': list(x), 'y': list(y)})
    if labels is not None:
        df['label'] = list(labels)
    df = df.dropna(subset=['x', 'y'])
    n = len(df)

    if n < 3:
        return dict(n=n, W=np.nan, p=np.nan, r_rosenthal=np.nan,
                    cliffs_delta=np.nan, median_diff=np.nan,
                    ci95_diff=(np.nan, np.nan),
                    interpretation='n insuficiente (<3 pares)')

    # Wilcoxon (mode='exact' para n<=25, sino 'approx')
    try:
        res = stats.wilcoxon(df['x'].values, df['y'].values,
                             zero_method='wilcox',
                             alternative='two-sided')
        W, p = float(res.statistic), float(res.pvalue)
    except ValueError:
        W, p = np.nan, np.nan

    # r de Rosenthal (aprox. para n>=6). Reconstruimos Z desde el test normal-approx.
    try:
        diffs = df['x'].values - df['y'].values
        # Z-score: diferencia estandarizada via ranks
        non_zero = diffs[diffs != 0]
        if len(non_zero) >= 3:
            ranks = stats.rankdata(np.abs(non_zero))
            pos_rank_sum = ranks[non_zero > 0].sum()
            mean_W = len(non_zero) * (len(non_zero) + 1) / 4
            sd_W = np.sqrt(len(non_zero) * (len(non_zero) + 1) *
                           (2 * len(non_zero) + 1) / 24)
            z = (pos_rank_sum - mean_W) / sd_W if sd_W > 0 else 0
            r_rosenthal = abs(z) / np.sqrt(len(non_zero))
        else:
            r_rosenthal = np.nan
    except Exception:
        r_rosenthal = np.nan

    # Cliff's delta (no paramétrico, rango [-1, +1])
    cliffs = cliffs_delta(df['x'].values, df['y'].values)

    # IC 95% bootstrap para la mediana de diferencias
    diffs = (df['x'] - df['y']).values
    ci_lo, ci_hi = bootstrap_ci(diffs, func=np.median, n_boot=5000)

    # Interpretación automatizada
    interp = _interpret_wilcoxon(p, r_rosenthal, cliffs, n)

    return dict(
        n=n,
        W=W,
        p=p,
        r_rosenthal=round(r_rosenthal, 3) if not np.isnan(r_rosenthal) else np.nan,
        cliffs_delta=round(cliffs, 3),
        median_diff=round(float(np.median(diffs)), 3),
        ci95_diff=(round(ci_lo, 3), round(ci_hi, 3)),
        interpretation=interp,
    )


def _interpret_wilcoxon(p, r, cliffs, n):
    """Interpretación verbal siguiendo convenciones APA."""
    if np.isnan(p):
        return 'No calculable'
    sig = '*' if p < 0.05 else 'n.s.'

    # Magnitud por Cliff's delta (Romano et al. 2006)
    abs_c = abs(cliffs)
    if   abs_c < 0.147: mag = 'efecto trivial'
    elif abs_c < 0.33:  mag = 'efecto pequeño'
    elif abs_c < 0.474: mag = 'efecto mediano'
    else:               mag = 'efecto grande'

    direction = 'x>y' if cliffs > 0 else ('x<y' if cliffs < 0 else 'x≈y')
    return f'{sig} · {mag} ({direction}) · n={n}'


# ══════════════════════════════════════════════════════════════════════
# TAMAÑOS DE EFECTO
# ══════════════════════════════════════════════════════════════════════

def cohens_d(x: Sequence[float], y: Sequence[float], paired: bool = False) -> float:
    """Cohen's d (paramétrico). Umbrales: 0.2 / 0.5 / 0.8 = small/medium/large."""
    x = np.array([v for v in x if not np.isnan(v)])
    y = np.array([v for v in y if not np.isnan(v)])
    if len(x) < 2 or len(y) < 2:
        return np.nan
    if paired and len(x) == len(y):
        diffs = x - y
        return float(np.mean(diffs) / np.std(diffs, ddof=1))
    # Cohen's d de pooled SD
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_sd = np.sqrt(((nx-1)*vx + (ny-1)*vy) / (nx+ny-2))
    if pooled_sd == 0:
        return np.nan
    return float((np.mean(x) - np.mean(y)) / pooled_sd)


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Cliff's delta — tamaño de efecto no paramétrico en [-1, +1].
    δ = (#{x>y} - #{x<y}) / (nx·ny).
    Umbrales (Romano 2006): 0.147 / 0.33 / 0.474.
    """
    x = np.array([v for v in x if not np.isnan(v)])
    y = np.array([v for v in y if not np.isnan(v)])
    if len(x) == 0 or len(y) == 0:
        return np.nan
    greater = sum(xi > yj for xi in x for yj in y)
    less    = sum(xi < yj for xi in x for yj in y)
    return float((greater - less) / (len(x) * len(y)))


# ══════════════════════════════════════════════════════════════════════
# INTERVALOS DE CONFIANZA BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════

def bootstrap_ci(
    data: Sequence[float],
    func=np.mean,
    n_boot: int = 5000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """IC bootstrap percentilado (bias-corrected no implementado por simplicidad)."""
    arr = np.array([v for v in data if not np.isnan(v)])
    if len(arr) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boot = func(arr[idx], axis=1)
    alpha = (1 - ci) / 2
    return (float(np.percentile(boot, 100*alpha)),
            float(np.percentile(boot, 100*(1-alpha))))


def mean_ci(data: Sequence[float], ci: float = 0.95) -> dict:
    """Media con IC bootstrap y paramétrico (t-student)."""
    arr = np.array([v for v in data if not np.isnan(v)])
    n = len(arr)
    if n < 3:
        return dict(n=n, mean=np.nan, sd=np.nan,
                    ci95_param=(np.nan, np.nan),
                    ci95_boot=(np.nan, np.nan))
    m, sd = float(arr.mean()), float(arr.std(ddof=1))
    # IC paramétrico t-student
    se = sd / np.sqrt(n)
    tval = stats.t.ppf(1 - (1-ci)/2, df=n-1)
    ci_param = (m - tval*se, m + tval*se)
    # IC bootstrap
    ci_boot = bootstrap_ci(arr, func=np.mean, ci=ci)
    return dict(
        n=n, mean=round(m, 3), sd=round(sd, 3),
        ci95_param=(round(ci_param[0], 3), round(ci_param[1], 3)),
        ci95_boot=(round(ci_boot[0], 3), round(ci_boot[1], 3)),
    )


# ══════════════════════════════════════════════════════════════════════
# CORRECCIÓN POR COMPARACIONES MÚLTIPLES
# ══════════════════════════════════════════════════════════════════════

def adjust_pvalues(pvalues: Sequence[float], method: str = 'fdr_bh') -> np.ndarray:
    """
    Corrección de p-valores. Por defecto Benjamini-Hochberg (FDR).
    Si statsmodels no está disponible, aplica BH manualmente.
    """
    p = np.array(pvalues, dtype=float)
    mask = ~np.isnan(p)
    p_valid = p[mask]
    if len(p_valid) == 0:
        return p

    if _HAS_SM:
        _, p_adj_valid, _, _ = multipletests(p_valid, method=method)
    else:
        # BH manual
        n = len(p_valid)
        order = np.argsort(p_valid)
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(1, n+1)
        p_adj_valid = np.minimum(1.0, p_valid * n / ranks)
        # Monotonicidad (BH step-up)
        p_adj_sorted = p_adj_valid[order]
        for i in range(n-2, -1, -1):
            p_adj_sorted[i] = min(p_adj_sorted[i], p_adj_sorted[i+1])
        p_adj_valid[order] = p_adj_sorted

    p_adj = p.copy()
    p_adj[mask] = p_adj_valid
    return p_adj


# ══════════════════════════════════════════════════════════════════════
# ANÁLISIS DE TENDENCIAS
# ══════════════════════════════════════════════════════════════════════

def trend_analysis(
    pivot: pd.DataFrame,
    min_points: int = 5,
    adjust_method: str = 'fdr_bh',
) -> pd.DataFrame:
    """
    Regresión lineal de tendencia por entidad (fila) con p ajustado por FDR.

    Parámetros
    ----------
    pivot : DataFrame índice=entidad (país), columnas=año, valores=score
    min_points : puntos mínimos para ajustar (default 5)
    adjust_method : método de ajuste ('fdr_bh', 'bonferroni', etc.)

    Devuelve
    --------
    DataFrame con columnas: país, slope, intercept, r2, p, p_adj, ci95_slope,
                            sig, sig_adj, direction
    """
    rows = []
    for country in pivot.index:
        s = pivot.loc[country].dropna()
        if len(s) < min_points:
            continue
        x = s.index.values.astype(float)
        y = s.values.astype(float)
        res = stats.linregress(x, y)
        rows.append({
            'País':       country,
            'n_años':     len(s),
            'slope':      round(float(res.slope), 4),
            'intercept':  round(float(res.intercept), 3),
            'r2':         round(float(res.rvalue)**2, 3),
            'p':          float(res.pvalue),
            'ci95_slope': round(1.96 * float(res.stderr), 4),
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    # Ajuste por comparaciones múltiples
    df['p_adj'] = adjust_pvalues(df['p'].values, method=adjust_method)
    df['p']     = df['p'].round(4)
    df['p_adj'] = df['p_adj'].round(4)
    df['sig']     = df['p'].apply(lambda p: '*' if p < 0.05 else '')
    df['sig_adj'] = df['p_adj'].apply(lambda p: '*' if p < 0.05 else '')

    # Clasificación por significancia (no por magnitud arbitraria)
    def _direction(r):
        if r['p_adj'] >= 0.05: return '→ Sin tendencia significativa'
        return '↑ Aumento sig.' if r['slope'] > 0 else '↓ Descenso sig.'
    df['Dirección'] = df.apply(_direction, axis=1)

    return df.sort_values('slope', ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
# CORRELACIONES CON IC (Fisher z)
# ══════════════════════════════════════════════════════════════════════

def correlation_ci(
    x: Sequence[float],
    y: Sequence[float],
    method: str = 'pearson',
    ci: float = 0.95,
) -> dict:
    """
    Correlación con IC vía transformación de Fisher z.
    method: 'pearson' o 'spearman'.
    """
    df = pd.DataFrame({'x': list(x), 'y': list(y)}).dropna()
    n = len(df)
    if n < 4:
        return dict(n=n, r=np.nan, p=np.nan,
                    ci=(np.nan, np.nan), method=method)

    if method == 'pearson':
        r, p = stats.pearsonr(df['x'], df['y'])
    elif method == 'spearman':
        r, p = stats.spearmanr(df['x'], df['y'])
    else:
        raise ValueError("method debe ser 'pearson' o 'spearman'")

    # Fisher z
    if abs(r) >= 1:
        return dict(n=n, r=round(r, 3), p=round(p, 4),
                    ci=(round(r, 3), round(r, 3)), method=method)
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    zval = stats.norm.ppf(1 - (1-ci)/2)
    z_lo, z_hi = z - zval*se, z + zval*se
    ci_lo, ci_hi = np.tanh(z_lo), np.tanh(z_hi)

    return dict(
        n=n,
        r=round(float(r), 3),
        p=round(float(p), 4),
        ci=(round(float(ci_lo), 3), round(float(ci_hi), 3)),
        method=method,
    )


# ══════════════════════════════════════════════════════════════════════
# TEST MANUAL
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    rng = np.random.default_rng(0)

    # Test Wilcoxon apareado
    x = rng.normal(5, 1, 20)
    y = x + rng.normal(0.3, 0.5, 20)  # tendencia ligeramente superior
    res = wilcoxon_paired(x, y)
    print('Wilcoxon apareado:', res)

    # Test tendencia
    years = list(range(2017, 2026))
    countries = ['A', 'B', 'C', 'D']
    data = {}
    for c in countries:
        slope = rng.normal(0.05, 0.04)
        data[c] = {y: 4 + slope*(y-2017) + rng.normal(0, 0.2) for y in years}
    piv = pd.DataFrame(data).T
    piv.columns = years
    trends = trend_analysis(piv)
    print('\nTendencias:')
    print(trends.to_string(index=False))

    # Test correlación
    a = rng.normal(50, 10, 20)
    b = a * 0.7 + rng.normal(0, 8, 20)
    cr = correlation_ci(a, b, method='spearman')
    print('\nCorrelación Spearman:', cr)
