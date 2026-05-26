"""Tests inférentiels TP4 : scipy, papiers, auteurs/inventeurs, ANOVA domaines."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import chi2_contingency, f_oneway, fisher_exact, pearsonr, shapiro, ttest_1samp, ttest_ind

from graphe_common.builders import build_coauthor_graph, build_paper_citation_graph, load_paper_authors, normalize_author_key
from graphe_common.graphs import EXCLUDED_GROUND_TRUTH_KEYWORD, compute_louvain, extract_paper_fields_of_study
from graphe_common.io import get_default_papers_path, has_elided_references, load_papers, safe_int

ALPHA = 0.05
MIN_GROUP_SIZE = 10
Row = dict[str, str | float | int | bool]
DEFAULT_COAUTHOR_GRAPH = Path("coauthor_graph.graphml")
DEFAULT_HINDEX_CACHE = Path("author_hindex_cache.json")
DEFAULT_AUTHOR_NAME_CACHE = Path("author_data_maps_cache.json")
DEFAULT_MATCH_THRESHOLD = 0.65
DEFAULT_PATENTS_PATH = "lens-export.csv"


def inferential_row(test: str, variable: str, statistic: float, p_value: float, n: int, *, alpha: float = ALPHA, detail: str = "", **extra: Any) -> Row:
    '''
    En entrée : test, variable, statistic, p_value, n ; alpha, detail, extra optionnels.

    En sortie : dictionnaire Row (test, variable, statistic, p_value, n, alpha, significant, detail, …).

    Variables : alpha (seuil de significativité), significant (p_value < alpha si p_value valide).
    '''
    return {"test": test, "variable": variable, "statistic": float(statistic), "p_value": float(p_value), "n": n, "alpha": alpha, "significant": bool(p_value < alpha) if not np.isnan(p_value) else False, "detail": detail, **extra}


def contingency_table(keys_a: list[str], keys_b: list[str]) -> tuple[list[list[int]], list[str], list[str]]:
    '''
    En entrée : deux listes de clés catégorielles alignées (keys_a, keys_b).

    En sortie : tableau de contingence, libellés des lignes, libellés des colonnes.

    Variables : table (comptages croisés), rows/cols (catégories uniques triées).
    '''
    rows, cols = sorted(set(keys_a)), sorted(set(keys_b))
    ia, ib = {k: i for i, k in enumerate(rows)}, {k: j for j, k in enumerate(cols)}
    table = [[0] * len(cols) for _ in rows]
    for a, b in zip(keys_a, keys_b, strict=True):
        table[ia[a]][ib[b]] += 1
    return table, rows, cols


def independence_tests(keys_a: list[str], keys_b: list[str], *, label: str, alpha: float = ALPHA, n: int | None = None, detail: str = "") -> list[Row]:
    '''
    En entrée : clés catégorielles appariées ; label, alpha, n, detail.

    En sortie : liste de Row (Fisher exact + éventuellement chi² d'indépendance).

    Variables : table, meta (détail du tableau), fs/fp et c2/cp (statistiques et p-values).
    '''
    table, row_labels, col_labels = contingency_table(keys_a, keys_b)
    n = n if n is not None else len(keys_a)
    meta = f"{detail}; rows={row_labels}; cols={col_labels}; table={table}" if detail else f"rows={row_labels}; cols={col_labels}; table={table}"
    fs, fp = fisher_exact(table)
    out = [inferential_row("fisher_exact", label, fs, fp, n, alpha=alpha, detail=meta)]
    if len(table) >= 2 and len(table[0]) >= 2:
        c2, cp, _, _ = chi2_contingency(table)
        out.append(inferential_row("chi2_independence", label, c2, cp, n, alpha=alpha, detail=meta))
    return out


def pearson_test(xs: list[float] | np.ndarray, ys: list[float] | np.ndarray, *, label: str, alpha: float = ALPHA, detail: str = "") -> Row:
    '''
    En entrée : séries numériques xs, ys ; label, alpha, detail.

    En sortie : Row pearson (r, p-value) ou NaN si effectif < 3.

    Variables : n (taille des échantillons), r (résultat scipy pearsonr).
    '''
    n = len(xs)
    if n < 3:
        return inferential_row("pearson", label, np.nan, np.nan, n, alpha=alpha, detail=detail or "effectif insuffisant")
    r = pearsonr(xs, ys)
    return inferential_row("pearson", label, r.statistic, r.pvalue, n, alpha=alpha, detail=detail)


def welch_test(a: list[float], b: list[float], *, label: str, alpha: float = ALPHA, detail: str = "") -> Row | None:
    '''
    En entrée : deux échantillons a, b ; label, alpha, detail.

    En sortie : Row t-test de Welch ou None si un groupe a moins de 2 observations.

    Variables : t, p (statistique et p-value ttest_ind, equal_var=False).
    '''
    if len(a) < 2 or len(b) < 2:
        return None
    t, p = ttest_ind(a, b, equal_var=False)
    return inferential_row("ttest_ind_welch", label, t, p, len(a) + len(b), alpha=alpha, detail=detail)


def shapiro_test(values: list[float], *, label: str, alpha: float = ALPHA, max_n: int = 5000, seed: int = 42) -> Row:
    '''
    En entrée : valeurs numériques ; label, alpha, max_n (sous-échantillonnage), seed.

    En sortie : Row test de Shapiro-Wilk (W, p-value).

    Variables : arr (tableau numpy), w/p (statistique et p-value).
    '''
    arr = np.asarray(values, dtype=float)
    if arr.size < 3:
        return inferential_row("shapiro", label, np.nan, np.nan, int(arr.size), alpha=alpha)
    if arr.size > max_n:
        arr = np.asarray(random.Random(seed).sample(arr.tolist(), max_n), dtype=float)
    w, p = shapiro(arr)
    return inferential_row("shapiro", label, w, p, int(arr.size), alpha=alpha, detail="normal if p >= alpha")


def run_anova(groups: dict[str, list[float]], *, min_n: int = 10) -> tuple[float, float, int, list[str]]:
    '''
    En entrée : dictionnaire groupe -> valeurs ; min_n (taille minimale par groupe).

    En sortie : (F, p-value, nombre de groupes éligibles, noms des groupes).

    Variables : eligible (groupes avec len >= min_n), stat/p_value (f_oneway).
    '''
    eligible = [(d, v) for d, v in groups.items() if len(v) >= min_n]
    if len(eligible) < 2:
        return float("nan"), float("nan"), 0, []
    stat, p_value = f_oneway(*(v for _, v in eligible))
    return float(stat), float(p_value), len(eligible), [d for d, _ in eligible]


def _paper_conclusion(row: Row) -> str:
    '''
    En entrée : une ligne de résultat inférentiel (Row).

    En sortie : libellé court interprétant le test (normalité, indépendance, corrélation, etc.).

    Variables : test, significant, p_value (lus depuis row).
    '''
    test, significant = row["test"], bool(row["significant"])
    p_value = float(row["p_value"])
    if test == "shapiro":
        return "non normale" if significant else "compatible avec la normale"
    if test in ("chi2_independence", "fisher_exact"):
        return "dependance" if significant else "independance"
    if test == "bootstrap_fisher":
        return f"part significative (p<{row['alpha']}) = {p_value:.2%}"
    if test == "summary":
        return "effectifs"
    if test == "pearson":
        return "correlation significative" if significant else "pas de correlation lineaire"
    return "difference de moyenne" if significant else "pas de difference"


def print_paper_inferential_stats(rows: list[Row], *, title: str = "Tests statistiques papiers") -> None:
    '''
    En entrée : liste de Row ; title (titre affiché).

    En sortie : aucune (affichage console).

    Variables : row, detail (lignes parcourues pour impression).
    '''
    print(f"\n[{title}] (alpha = 0.05)")
    for row in rows:
        detail = f" | {row['detail']}" if row.get("detail") else ""
        print(f"  {row['test']:18} {row['variable']:40} stat={row['statistic']:.6f}  p={float(row['p_value']):.6g}  n={row['n']}{detail}  -> {_paper_conclusion(row)}")


def count_authors(paper: dict[str, Any]) -> int:
    '''
    En entrée : dictionnaire papier (clé authors).

    En sortie : nombre d'auteurs.

    Variables : aucune locale significative.
    '''
    return len(paper.get("authors") or [])


def collect_paper_metrics(papers: list[dict[str, Any]]) -> tuple[list[int], list[int], list[int]]:
    '''
    En entrée : liste de papiers (dict Semantic Scholar).

    En sortie : (citations, n_fields, n_authors) — trois listes parallèles.

    Variables : citations, n_fields, n_authors (accumulateurs par papier).
    '''
    citations, n_fields, n_authors = [], [], []
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        citations.append(safe_int(paper.get("citationCount"), default=0) or 0)
        n_fields.append(sum(1 for item in paper.get("s2FieldsOfStudy") or [] if isinstance(item, dict) and str(item.get("category") or "").strip()))
        n_authors.append(count_authors(paper))
    return citations, n_fields, n_authors


def run_inferential_stats(papers: list[dict[str, Any]], *, alpha: float = ALPHA, seed: int = 42) -> list[Row]:
    '''
    En entrée : corpus de papiers ; alpha, seed.

    En sortie : liste de Row (Pearson, Shapiro, indépendance, Welch, t-test un échantillon).

    Variables : citations, n_fields, n_authors, median_c, author_bins, field_bins, citation_bins.
    '''
    citations, n_fields, n_authors = collect_paper_metrics(papers)
    if (n_papers := len(citations)) < 3:
        raise ValueError("Pas assez de papiers pour les tests statistiques.")
    median_c = float(np.median(citations))
    rows: list[Row] = [pearson_test(citations, n_fields, label="citationCount vs n_s2FieldsOfStudy", alpha=alpha), pearson_test(citations, n_authors, label="citationCount vs n_authors", alpha=alpha)]
    rows += [shapiro_test([float(v) for v in values], label=name, alpha=alpha, seed=seed) for name, values in [("citationCount", citations), ("n_s2FieldsOfStudy", n_fields), ("n_authors", n_authors)]]
    author_bins = ["1" if n <= 1 else "2-3" if n <= 3 else "4-6" if n <= 6 else "7+" for n in n_authors]
    field_bins = ["0" if n == 0 else "1" if n == 1 else "2+" for n in n_fields]
    citation_bins = ["high" if c > median_c else "low" for c in citations]
    for label, keys_a, keys_b in [("n_authors_bin vs citation_bin", author_bins, citation_bins), ("n_fields_bin vs citation_bin", field_bins, citation_bins), ("n_authors_bin vs n_fields_bin", author_bins, field_bins)]:
        rows.extend(independence_tests(keys_a, keys_b, label=label, alpha=alpha, n=n_papers))
    for low, high, lbl, det in [([float(c) for c, n in zip(citations, n_authors, strict=True) if n <= 2], [float(c) for c, n in zip(citations, n_authors, strict=True) if n >= 5], "citationCount: n_authors<=2 vs >=5", f"n_low={sum(1 for n in n_authors if n <= 2)}; n_high={sum(1 for n in n_authors if n >= 5)}"), ([float(c) for c, nf in zip(citations, n_fields, strict=True) if nf == 0], [float(c) for c, nf in zip(citations, n_fields, strict=True) if nf >= 1], "citationCount: 0 fields vs >=1", f"n_0={sum(1 for nf in n_fields if nf == 0)}; n_1+={sum(1 for nf in n_fields if nf >= 1)}")]:
        if w := welch_test(low, high, label=lbl, alpha=alpha, detail=det):
            rows.append(w)
    t, p = ttest_1samp(citations, popmean=median_c)
    rows.append(inferential_row("ttest_1samp", f"citationCount vs median={median_c:.1f}", t, p, n_papers, alpha=alpha))
    return rows


def run_team_disruption_test(papers: list[dict[str, Any]], disruption_by_paper_id: dict[str, float], *, alpha: float = ALPHA, small_team_max: int = 2, large_team_min: int = 5) -> list[Row]:
    '''
    En entrée : papiers, disruption par paperId ; alpha, seuils petite/grande équipe.

    En sortie : Row Pearson et Welch (disruption vs taille d'équipe).

    Variables : n_authors, disruption, small, large (sous-échantillons).
    '''
    n_authors, disruption = [], []
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        paper_id = str(paper.get("paperId") or "").strip()
        if paper_id and paper_id in disruption_by_paper_id:
            n_authors.append(count_authors(paper))
            disruption.append(float(disruption_by_paper_id[paper_id]))
    label = "Petites équipes <=> Haute disruption"
    rows: list[Row] = [pearson_test(n_authors, disruption, label=f"{label}: n_authors vs disruption", alpha=alpha, detail="r<0 attendu si petites équipes => haute disruption")]
    small = [d for d, na in zip(disruption, n_authors, strict=True) if na <= small_team_max]
    large = [d for d, na in zip(disruption, n_authors, strict=True) if na >= large_team_min]
    if w := welch_test(small, large, label=f"{label}: disruption n_authors<={small_team_max} vs >={large_team_min}", alpha=alpha, detail=f"n_small={len(small)}; n_large={len(large)}; mean_small={np.mean(small):.4f}; mean_large={np.mean(large):.4f}"):
        rows.append(w)
    return rows


def collect_mean_team_size_by_author(papers: list[dict[str, Any]], *, author_keys: set[str] | None = None) -> dict[str, float]:
    '''
    En entrée : papiers ; author_keys optionnel (filtre sur clés auteur).

    En sortie : dictionnaire clé auteur -> taille moyenne d'équipe.

    Variables : team_sizes (liste des tailles par auteur).
    '''
    team_sizes: dict[str, list[int]] = defaultdict(list)
    for paper in papers:
        if isinstance(paper, dict):
            n = count_authors(paper)
            for author in paper.get("authors") or []:
                if (key := normalize_author_key(author)) and (author_keys is None or key in author_keys):
                    team_sizes[key].append(n)
    return {k: float(np.mean(v)) for k, v in team_sizes.items() if v}


def matched_inventor_author_keys(patents: list[dict[str, Any]], papers: list[dict[str, Any]], *, author_name_cache_path: Path = DEFAULT_AUTHOR_NAME_CACHE, threshold: float = DEFAULT_MATCH_THRESHOLD) -> set[str]:
    '''
    En entrée : brevets, papiers ; chemin cache noms, seuil de matching.

    En sortie : ensemble de clés auteurs (id:…) matchées à un inventeur.

    Variables : inventors, paper_authors, inv_feats, paper_feats, blocking.
    '''
    from graphe_brevets.graph import _entity_feats, _match_entities, parse_inventor_entries

    paper_ids = {str(p["paperId"]) for p in papers if p.get("paperId") and not has_elided_references(p)}
    inventors: dict[str, str] = {}
    for row in patents:
        for key, display in parse_inventor_entries(str(row.get("Inventors") or "")):
            inventors.setdefault(key, display)
    _, raw_authors = load_paper_authors(papers, paper_ids)
    name_by_id = json.loads(author_name_cache_path.read_text(encoding="utf-8")).get("author_name_map", {}) if author_name_cache_path.exists() else {}
    paper_authors = {k: name_by_id[k[3:]] for k in raw_authors if k.startswith("id:") and name_by_id.get(k[3:])}
    inv_feats, paper_feats = _entity_feats(inventors, last_first=True), _entity_feats(paper_authors, last_first=False)
    blocking: dict[str, list[str]] = defaultdict(list)
    for entity_id, feat in paper_feats.items():
        for key in feat["keys"]:
            blocking[key].append(entity_id)
    return {paper for _, _, paper in _match_entities(inv_feats, paper_feats, blocking, threshold=threshold)}


def load_coauthor_centralities(graph_path: Path) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    '''
    En entrée : chemin vers un graphe co-auteurs GraphML.

    En sortie : (degree_centrality, eigenvector_centrality, pagerank) par nœud.

    Variables : G (graphe NetworkX lu depuis graph_path).
    '''
    import networkx as nx

    G = nx.read_graphml(graph_path)
    return nx.degree_centrality(G), nx.eigenvector_centrality(G), nx.pagerank(G.to_directed(), weight="weight", tol=1e-06)


def load_hindex_by_author_key(cache_path: Path) -> dict[str, float]:
    '''
    En entrée : chemin JSON cache h-index (clés author id).

    En sortie : dictionnaire id:authorId -> h-index (float).

    Variables : raw (JSON chargé).
    '''
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    return {} if not isinstance(raw, dict) else {f"id:{aid}": float(value) for aid, value in raw.items() if str(aid).strip() and value is not None}


def run_author_inventor_inferential(papers: list[dict[str, Any]], *, patents_path: Path = Path(DEFAULT_PATENTS_PATH), coauthor_graph_path: Path = DEFAULT_COAUTHOR_GRAPH, hindex_cache_path: Path = DEFAULT_HINDEX_CACHE, author_name_cache_path: Path = DEFAULT_AUTHOR_NAME_CACHE, match_threshold: float = DEFAULT_MATCH_THRESHOLD, alpha: float = ALPHA, bootstrap_samples: int = 500, bootstrap_subsample_n: int = 2000, seed: int = 42) -> list[Row]:
    '''
    En entrée : papiers ; chemins brevets, graphe, caches ; alpha, bootstrap, seed.

    En sortie : liste de Row (inventeur vs h-index / centralités, Fisher, bootstrap).

    Variables : records, inventor_keys, hindex_by_key, degree/eigen/pagerank, fisher_ps.
    '''
    from graphe_brevets.graph import load_patents

    if not coauthor_graph_path.exists():
        raise FileNotFoundError(f"Graphe co-auteurs introuvable: {coauthor_graph_path}")
    if not hindex_cache_path.exists():
        raise FileNotFoundError(f"Cache h-index introuvable: {hindex_cache_path}")
    patents = load_patents(str(patents_path))
    inventor_keys = matched_inventor_author_keys(patents, papers, author_name_cache_path=author_name_cache_path, threshold=match_threshold)
    hindex_by_key = load_hindex_by_author_key(hindex_cache_path)
    degree, eigen, pagerank = load_coauthor_centralities(coauthor_graph_path)
    paper_ids = {str(p["paperId"]) for p in papers if p.get("paperId") and not has_elided_references(p)}
    _, raw_authors = load_paper_authors(papers, paper_ids)
    records = [{"key": key, "hindex": hindex_by_key[key], "inventeur": key in inventor_keys, **({"degree": float(degree[key]), "eigenvector": float(eigen[key]), "pagerank": float(pagerank[key])} if key in degree else {})} for key in raw_authors if key.startswith("id:") and key in hindex_by_key]
    if len(records) < 3:
        raise ValueError("Pas assez d'auteurs pour les tests inventeur / influence.")
    n_authors, n_inventors = len(records), sum(1 for r in records if r["inventeur"])
    rows: list[Row] = [inferential_row("summary", "auteurs id+h-index", n_inventors, n_inventors / n_authors if n_authors else 0.0, n_authors, alpha=alpha, detail=f"match_threshold={match_threshold}; inventeurs_matchés={n_inventors}")]
    rows[-1]["significant"] = False
    for metric_label, field in [("hindex", "hindex"), ("degree_centrality", "degree"), ("eigenvector_centrality", "eigenvector"), ("pagerank", "pagerank")]:
        subset = records if field == "hindex" else [r for r in records if field in r]
        if len(subset) < 3:
            continue
        median_value = float(np.median([float(r[field]) for r in subset]))
        high_bins = ["high" if float(r[field]) > median_value else "low" for r in subset]
        inventor_bins = ["inventeur" if r["inventeur"] else "non_inventeur" for r in subset]
        split_detail = f"median={median_value:.6g}; n={len(subset)}; n_inventeur={sum(1 for b in inventor_bins if b == 'inventeur')}"
        rows.extend(independence_tests(high_bins, inventor_bins, label=f"Haut {metric_label} <=> Inventeur", alpha=alpha, n=len(subset), detail=f"split={split_detail}"))
        if metric_label == "hindex" and bootstrap_samples > 0:
            rng, sub_n = random.Random(seed), min(bootstrap_subsample_n, len(subset))
            fisher_ps = [float(fisher_exact([[sum(1 for r in s if float(r["hindex"]) > med and r["inventeur"]), sum(1 for r in s if float(r["hindex"]) > med and not r["inventeur"])], [sum(1 for r in s if float(r["hindex"]) <= med and r["inventeur"]), sum(1 for r in s if float(r["hindex"]) <= med and not r["inventeur"])]])[1]) for s in (rng.sample(subset, sub_n) for _ in range(bootstrap_samples)) for med in [float(np.median([float(r["hindex"]) for r in s]))]]
            rows.append(inferential_row("bootstrap_fisher", "Haut hindex <=> Inventeur (sous-échantillons)", float(np.median(fisher_ps)), float(np.mean([p < alpha for p in fisher_ps])), bootstrap_samples, alpha=alpha, detail=f"subsample_n={sub_n}; p_median={np.median(fisher_ps):.6g}; p5_p95={[round(x, 6) for x in np.percentile(fisher_ps, [5, 95])]}"))
            rows[-1]["significant"] = False
    if w := welch_test([float(r["hindex"]) for r in records if r["inventeur"]], [float(r["hindex"]) for r in records if not r["inventeur"]], label="hindex: inventeur vs non_inventeur", alpha=alpha, detail=f"n_inventeur={n_inventors}; n_non={n_authors - n_inventors}"):
        rows.append(w)
    return rows


def run_npl_citation_inferential(papers: list[dict[str, Any]], *, patents: list[dict[str, Any]] | None = None, patents_path: Path = Path(DEFAULT_PATENTS_PATH), coauthor_graph_path: Path = DEFAULT_COAUTHOR_GRAPH, alpha: float = ALPHA, seed: int = 42, edge_source: str = "citations") -> list[Row]:
    import networkx as nx
    from graphe_brevets.graph import load_patents, npl_cited_paper_ids

    patents = patents if patents is not None else load_patents(str(patents_path))
    paper_by_id = {str(p["paperId"]): p for p in papers if p.get("paperId")}
    cited_ids = sorted(npl_cited_paper_ids(patents, papers) & paper_by_id.keys())
    n_cited = len(cited_ids)
    pool = [pid for pid in paper_by_id if pid not in cited_ids]
    control_ids = random.Random(seed).sample(pool, n_cited) if n_cited and len(pool) >= n_cited else []

    rows: list[Row] = [inferential_row("summary", "papiers corpus / cités NPL", n_cited, n_cited / len(paper_by_id) if paper_by_id else 0.0, len(paper_by_id), alpha=alpha, detail=f"n_cité_NPL={n_cited}; n_témoin={len(control_ids)}; n_brevets={len(patents)}; seed={seed}")]
    rows[-1]["significant"] = False
    if n_cited < 2 or len(control_ids) < 2:
        return rows

    paper_ids = set(paper_by_id)
    in_deg = nx.in_degree_centrality(build_paper_citation_graph(papers, paper_ids, edge_source=edge_source))  # type: ignore[arg-type]
    if coauthor_graph_path.exists():
        G_co = nx.read_graphml(coauthor_graph_path)
    else:
        p2a, names = load_paper_authors(papers, paper_ids)
        G_co = build_coauthor_graph(p2a, names)
    comm_sizes = {str(n): float(len(c)) for c in compute_louvain(G_co, seed=seed)[0] for n in c}

    def vals(ids: list[str], metric: str) -> list[float]:
        out: list[float] = []
        for pid in ids:
            p = paper_by_id[pid]
            if metric == "citationCount":
                out.append(float(safe_int(p.get("citationCount"), default=0) or 0))
            elif metric == "in_degree_centrality":
                out.append(float(in_deg.get(pid, 0.0)))
            else:
                aids = [f"id:{a['authorId']}" for a in p.get("authors") or [] if a.get("authorId")]
                cs = [comm_sizes[a] for a in aids if a in comm_sizes]
                if cs:
                    out.append(float(np.mean(cs)))
        return out

    base = f"n_cité={n_cited}; n_témoin={len(control_ids)}; seed={seed}"
    for metric in ("citationCount", "in_degree_centrality", "mean_author_community_size"):
        a, b = vals(cited_ids, metric), vals(control_ids, metric)
        if len(a) >= 2 and len(b) >= 2 and (w := welch_test(a, b, label=f"{metric}: cité_NPL vs témoin", alpha=alpha, detail=f"{base}; mean_cité={np.mean(a):.4g}; mean_témoin={np.mean(b):.4g}")):
            rows.append(w)
    return rows


def run_large_team_centrality_test(G: Any, mean_team_size: dict[str, float], hindex_map: dict[str, int | float | None], *, alpha: float = ALPHA, small_team_max: float = 2.0, large_team_min: float = 5.0) -> list[Row]:
    '''
    En entrée : graphe co-auteurs G, taille moyenne d'équipe par nœud, h-index ; alpha, seuils équipe.

    En sortie : Row Pearson et Welch (taille d'équipe vs centralités / h-index).

    Variables : authors, degree_c, pr, eigen_c, hindex_by_node, pairs, small_vals, large_vals.
    '''
    import networkx as nx

    from graphe_common.stats import safe_eigenvector

    label, authors = "Grandes équipes <=> Haute centralité / hindex / pagerank", [n for n in G.nodes if n in mean_team_size]
    if not authors:
        return []
    degree_c, pr = nx.degree_centrality(G), nx.pagerank(G, weight="weight", tol=1e-06)
    eigen_c = safe_eigenvector(G)
    hindex_by_node = {n: float(raw) for n in G.nodes if str(n).startswith("id:") and (raw := hindex_map.get(str(n).split("id:", 1)[1])) is not None}
    rows: list[Row] = []
    for metric_name, metric_dict in [("degree_centrality", degree_c), ("eigenvector_centrality", eigen_c), ("pagerank", pr), ("hIndex", hindex_by_node)]:
        pairs = [(mean_team_size[n], float(metric_dict[n])) for n in authors if n in metric_dict]
        xs, ys = zip(*pairs) if pairs else ([], [])
        rows.append(pearson_test(xs, ys, label=f"{label}: mean_team_size vs {metric_name}", alpha=alpha, detail="r>0 attendu si grandes équipes => haute métrique"))
        small_vals = [float(metric_dict[n]) for n in authors if mean_team_size[n] <= small_team_max and n in metric_dict]
        large_vals = [float(metric_dict[n]) for n in authors if mean_team_size[n] >= large_team_min and n in metric_dict]
        if w := welch_test(large_vals, small_vals, label=f"{label}: {metric_name} mean_team<={small_team_max} vs >={large_team_min}", alpha=alpha, detail=f"n_small={len(small_vals)}; n_large={len(large_vals)}"):
            rows.append(w)
    return rows


def non_cs_fields(fields: set[str]) -> list[str]:
    '''
    En entrée : ensemble de champs / domaines d'un nœud ou papier.

    En sortie : champs triés hors Computer Science (EXCLUDED_GROUND_TRUTH_KEYWORD).

    Variables : aucune locale significative.
    '''
    return sorted((f for f in fields if f != EXCLUDED_GROUND_TRUTH_KEYWORD), key=str.lower)


def _collect_multilabel(field_sets: dict[str, set[str]], values: dict[str, float]) -> dict[str, list[float]]:
    '''
    En entrée : domaines par nœud, métrique par nœud (ex. citations).

    En sortie : domaine -> liste de valeurs (un nœud peut contribuer à plusieurs domaines).

    Variables : by_field (accumulateur par domaine non-CS).
    '''
    by_field: dict[str, list[float]] = defaultdict(list)
    for node, fields in field_sets.items():
        if node in values:
            for field in non_cs_fields(fields):
                by_field[field].append(float(values[node]))
    return dict(by_field)


def _collect_exclusive(field_sets: dict[str, set[str]], values: dict[str, float]) -> tuple[dict[str, list[float]], int]:
    '''
    En entrée : domaines par nœud, métrique par nœud.

    En sortie : (groupes par domaine unique, nombre de nœuds multi-domaines exclus).

    Variables : by_field, multi_label (compteur exclus).
    '''
    by_field: dict[str, list[float]] = defaultdict(list)
    multi_label = 0
    for node, fields in field_sets.items():
        if node not in values:
            continue
        non_cs = non_cs_fields(fields)
        if EXCLUDED_GROUND_TRUTH_KEYWORD in fields and not non_cs:
            continue
        if len(non_cs) != 1:
            if len(non_cs) > 1:
                multi_label += 1
            continue
        by_field[non_cs[0]].append(float(values[node]))
    return dict(by_field), multi_label


def _paper_field_sets_and_values(papers: list[dict[str, Any]]) -> tuple[dict[str, set[str]], dict[str, float]]:
    '''
    En entrée : liste de papiers.

    En sortie : (paperId -> champs d'étude, paperId -> citationCount).

    Variables : field_sets, values.
    '''
    field_sets, values = {}, {}
    for i, paper in enumerate(papers):
        if not isinstance(paper, dict):
            continue
        key = str(paper.get("paperId") or i)
        field_sets[key] = extract_paper_fields_of_study(paper)
        values[key] = float(safe_int(paper.get("citationCount"), default=0) or 0)
    return field_sets, values


def domain_summary(by_field: dict[str, list[float]]) -> list[dict[str, float | int | str]]:
    '''
    En entrée : domaine -> liste de citationCount (ou autre métrique).

    En sortie : liste de dicts (domain, n, mean, median, std) triée par mean décroissant.

    Variables : arr (tableau numpy par domaine).
    '''
    return [{"domain": domain, "n": len(arr := np.asarray(values, dtype=float)), "mean_citation_count": float(np.mean(arr)), "median_citation_count": float(np.median(arr)), "std_citation_count": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0} for domain, values in sorted(by_field.items(), key=lambda kv: (-np.mean(kv[1]), kv[0]))]


def _append_domain_tests(rows: list[Row], assignment: str, groups: dict[str, list[float]], *, min_group_size: int, alpha: float, entity: str) -> None:
    '''
    En entrée : liste rows (modifiée in-place), type d'assignation, groupes ; min_group_size, alpha, entity.

    En sortie : aucune (ajoute ANOVA et Welch top-2 domaines à rows).

    Variables : stat, p_value, k, group_names, top2.
    '''
    stat, p_value, k, group_names = run_anova(groups, min_n=min_group_size)
    rows.append({"test": "f_oneway", "assignment": assignment, "k_groups": k, "statistic": stat, "p_value": p_value, "alpha": alpha, "significant": p_value < alpha if k >= 2 and not np.isnan(p_value) else False, "groups": ";".join(group_names), "detail": f"min_group_size={min_group_size}; entity={entity}"})
    top2 = sorted([(d, v) for d, v in groups.items() if len(v) >= min_group_size], key=lambda kv: -np.mean(kv[1]))[:2]
    if len(top2) >= 2 and (w := welch_test([float(x) for x in top2[0][1]], [float(x) for x in top2[1][1]], label=f"CitationCount {assignment}: {top2[0][0]} vs {top2[1][0]}", alpha=alpha, detail=f"n1={len(top2[0][1])}; n2={len(top2[1][1])}; mean1={np.mean(top2[0][1]):.2f}; mean2={np.mean(top2[1][1]):.2f}")):
        rows.append({**w, "assignment": assignment, "k_groups": w["n"], "groups": w["variable"]})


def _run_domain_inferential_core(field_sets: dict[str, set[str]], values: dict[str, float], *, entity: str, extra_pearson: Callable[[], Row], min_group_size: int = MIN_GROUP_SIZE, alpha: float = ALPHA) -> dict[str, Any]:
    '''
    En entrée : field_sets, values, entity (papers|authors), extra_pearson (test Pearson additionnel).

    En sortie : dict multilabel/exclusive, rows, résumés par domaine.

    Variables : multilabel, exclusive, rows, row.
    '''
    multilabel, (exclusive, multi_label_excluded) = _collect_multilabel(field_sets, values), _collect_exclusive(field_sets, values)
    rows: list[Row] = []
    for assignment, groups in [("multilabel", multilabel), ("exclusive", exclusive)]:
        _append_domain_tests(rows, assignment, groups, min_group_size=min_group_size, alpha=alpha, entity=entity)
    row = extra_pearson()
    rows.append({**row, "assignment": "all_fields", "k_groups": row["n"], "groups": row.get("variable", "")})
    return {"multilabel": multilabel, "exclusive": exclusive, "multi_label_excluded": multi_label_excluded, "rows": rows, "multilabel_summary": domain_summary(multilabel), "exclusive_summary": domain_summary(exclusive)}


def run_paper_domain_inferential(papers: list[dict[str, Any]], *, min_group_size: int = MIN_GROUP_SIZE, alpha: float = ALPHA) -> dict[str, Any]:
    '''
    En entrée : papiers ; min_group_size, alpha.

    En sortie : dict multilabel/exclusive, rows de tests, résumés par domaine.

    Variables : field_sets, values, multilabel, exclusive, influence, n_fields.
    '''
    field_sets, values = _paper_field_sets_and_values(papers)
    influence, n_fields = [], []
    for paper in papers:
        influence.append(float(safe_int(paper.get("influentialCitationCount"), default=0) or 0))
        n_fields.append(len(extract_paper_fields_of_study(paper)))
    return _run_domain_inferential_core(field_sets, values, entity="papers", min_group_size=min_group_size, alpha=alpha, extra_pearson=lambda: pearson_test(influence, [float(x) for x in n_fields], label="influentialCitationCount vs n_fieldsOfStudy", alpha=alpha, detail="tous champs; entity=papers"))


def run_author_domain_inferential(author_fields: dict[str, set[str]], author_citation_count: dict[str, int], *, min_group_size: int = MIN_GROUP_SIZE, alpha: float = ALPHA) -> dict[str, Any]:
    '''
    En entrée : domaines par auteur, citationCount par auteur ; min_group_size, alpha.

    En sortie : même structure que run_paper_domain_inferential (entity=authors).

    Variables : values, multilabel, exclusive, metric, n_fields.
    '''
    values = {k: float(v) for k, v in author_citation_count.items()}
    metric, n_fields = [], []
    for node, fields in author_fields.items():
        if node in values:
            metric.append(values[node])
            n_fields.append(float(len(fields)))
    return _run_domain_inferential_core(author_fields, values, entity="authors", min_group_size=min_group_size, alpha=alpha, extra_pearson=lambda: pearson_test(metric, n_fields, label="citationCount vs n_fieldsOfStudy", alpha=alpha, detail="tous champs; entity=authors"))


def print_domain_inferential_report(result: dict[str, Any], *, title: str, focus_domains: list[str] | None = None, alpha: float = ALPHA) -> None:
    '''
    En entrée : résultat run_*_domain_inferential ; title, focus_domains, alpha.

    En sortie : aucune (rapport console moyennes + tests).

    Variables : row, p_value, significant, test (lignes parcourues).
    '''
    print(f"\n[{title}] Moyennes CitationCount par domaine — hors Computer Science")
    for label, key in [("Assignment multilabel (un nœud peut compter dans plusieurs domaines):", "multilabel_summary"), (f"\nAssignment exclusive (discipline unique, {result['multi_label_excluded']} multi-domaines exclus):", "exclusive_summary")]:
        print(label)
        for row in result[key][:12]:
            print(f"  {str(row['domain']).title():32s} n={row['n']:5d}  mean={row['mean_citation_count']:.2f}  median={row['median_citation_count']:.0f}")
    print(f"\n[{title}] Tests inférentiels (alpha={alpha})")
    for row in result["rows"]:
        p_value, significant, test = float(row["p_value"]), bool(row["significant"]), str(row["test"])
        if test in ("pearson", "ttest_ind_welch"):
            label, sym = row.get("assignment", row.get("variable", "")), "r" if test == "pearson" else "t"
            conclusion = "corrélation significative" if test == "pearson" and significant else "différence significative" if significant else ("pas de corrélation linéaire" if test == "pearson" else "pas de différence")
            print(f"  {test:22} {label:16} n={row.get('k_groups', row.get('n', 0)):5}  {sym}={float(row['statistic']):.4f}  p={p_value:.6g}  -> {conclusion}")
        else:
            print(f"  {test:22} {row['assignment']:16} k={row['k_groups']:2}  F={float(row['statistic']):.4f}  p={p_value:.6g}  -> {'rejeter H0 (différence significative)' if significant else 'ne pas rejeter H0'}")
        if row.get("detail"):
            print(f"    {row['detail']}")
    if focus_domains:
        print(f"\n[{title}] Domaines suivis: {', '.join(focus_domains)}")


def cli_domain_anova(argv: list[str] | None = None) -> None:
    '''
    En entrée : argv optionnel (arguments CLI, défaut sys.argv).

    En sortie : aucune (charge papiers, lance ANOVA domaines, affiche rapport).

    Variables : parser, args, papers, result.
    '''
    parser = argparse.ArgumentParser(description="ANOVA CitationCount par domaine (hors Computer Science)")
    parser.add_argument("--input", type=Path, default=get_default_papers_path())
    parser.add_argument("--min-group-size", type=int, default=MIN_GROUP_SIZE)
    parser.add_argument("--top-influential", type=int, default=10)
    parser.add_argument("--focus-domains", nargs="*", default=["medicine", "chemistry"])
    args = parser.parse_args(argv)
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    papers = load_papers(args.input)
    result = run_paper_domain_inferential(papers, min_group_size=args.min_group_size)
    print_domain_inferential_report(result, title="Corpus complet", focus_domains=args.focus_domains)


if __name__ == "__main__":
    cli_domain_anova()
