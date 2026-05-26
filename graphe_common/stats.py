from __future__ import annotations

import math
from typing import Any

import networkx as nx
import numpy as np
from scipy.stats import pearsonr, t as student_t


def log10_two_tail_from_t(tstat: float) -> float:
    '''
    En entrée : tstat (statistique t de Student, deux queues).

    En sortie : log10 de la p-value bilatérale approximée via erfc.

    Variables : x, log_erfc.
    '''
    x = abs(tstat) / math.sqrt(2.0)
    if x > 8:
        log_erfc = -x * x - math.log(x) - math.log(math.sqrt(math.pi))
    else:
        from math import erfc

        log_erfc = math.log(erfc(x))
    return (math.log(2.0) + log_erfc) / math.log(10.0)


def fmt_p_value(p: float, *, test: str = "", stat: float = float("nan"), n: int = 0) -> str:
    '''
    En entrée : p (p-value), test (nom du test), stat (statistique), n (effectif).

    En sortie : chaîne lisible pour affichage CSV/console (notation scientifique si p ≈ 0).

    Variables : df, tstat, log_p.
    '''
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "NaN"
    if p != 0.0:
        if p < 0.0001:
            return f"{p:.6e}"
        return f"{p:.10f}".rstrip("0").rstrip(".")
    if test == "pearson" and n >= 3 and abs(stat) < 1.0:
        df = n - 2
        tstat = stat * math.sqrt(df / (1.0 - stat * stat))
        log_p = math.log(2.0) + float(student_t.logsf(abs(tstat), df))
        if math.isfinite(log_p) and log_p > -50:
            return f"{math.exp(log_p):.6e}"
        return f"10^({log10_two_tail_from_t(tstat):.2f})"
    if test in ("ttest_ind_welch", "ttest_ind_welch_onesided") and math.isfinite(stat):
        return f"10^({log10_two_tail_from_t(stat):.2f})"
    return "≈ 0 (sous-flux double)"


def aligned_pearson(values_a: dict[str, float], values_b: dict[str, float], *, min_n: int = 3) -> tuple[float, float, int] | None:
    '''
    En entrée : deux dicts clé → float, min_n (effectif minimal).

    En sortie : tuple (r, p, n) via scipy.stats.pearsonr ou None si variance nulle / n insuffisant.

    Variables : common, xa, xb, result.
    '''
    common = sorted(set(values_a) & set(values_b))
    if len(common) < min_n:
        return None
    xa = np.array([float(values_a[k]) for k in common])
    xb = np.array([float(values_b[k]) for k in common])
    if np.std(xa) == 0 or np.std(xb) == 0:
        return None
    result = pearsonr(xa, xb)
    return float(result.statistic), float(result.pvalue), len(common)


def safe_eigenvector(G: nx.Graph | nx.DiGraph, *, max_iter: int = 1000, tol: float = 1e-6) -> dict[Any, float]:
    '''
    En entrée : G (graphe NetworkX), max_iter, tol (paramètres eigenvector_centrality).

    En sortie : centralité eigenvector ; repli sur la plus grande composante connexe si échec de convergence.

    Variables : largest, sub.
    '''
    if G.number_of_edges() == 0:
        return {n: 0.0 for n in G.nodes}
    try:
        return nx.eigenvector_centrality(G, max_iter=max_iter, tol=tol)
    except (nx.NetworkXException, nx.PowerIterationFailedConvergence):
        largest = max(nx.weakly_connected_components(G) if G.is_directed() else nx.connected_components(G), key=len)
        sub = nx.eigenvector_centrality(G.subgraph(largest), max_iter=max_iter, tol=tol)
        return {n: float(sub.get(n, 0.0)) for n in G.nodes}
