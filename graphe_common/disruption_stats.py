"""Stats disruption : corrélations pagerank + tests taille d'équipe (strict / allégé)."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import numpy as np
from scipy.stats import ttest_ind

from graphe_common.builders import build_paper_citation_graph
from graphe_common.disruption import ALL_RULES, STRICT_RULE, DisruptionRule, annotate_disruptions
from graphe_common.inferential import ALPHA, Row, count_authors, inferential_row, pearson_test, welch_test
from graphe_common.io import get_default_papers_path, has_elided_citations, has_elided_references, load_papers, safe_int

STATS_DIR = Path(__file__).resolve().parents[1] / "graphe_coauteurs" / "data" / "stats"
RULE_SETS = ALL_RULES
_TOL_LABELS = {r.name: r.tolerance_label for r in ALL_RULES}
_CSV_FIELDS = (
    "record_type", "rules", "tolerance", "test", "variable", "statistic", "p_value", "p_value_fmt",
    "n", "alpha", "significant", "detail", "n_pertinent_in_corpus", "n_total_computable", "n_with_numeric_value", "corpus_n",
)


@dataclass(frozen=True)
class DisruptionTeamFilter:
    min_citation_count: int = 10
    small_team_max: int = 4
    large_team_min: int = 10
    citation_strict_gt: bool = False


def _relevant(paper: dict[str, Any], rule: DisruptionRule, *, total: bool = False) -> bool:
    if paper.get(rule.relevant_flag) is not False:
        return False
    if total:
        return paper.get(rule.total_bad_flag) is False and paper.get(rule.total_key) is not None
    return paper.get(rule.in_key) is not None


def filter_team_sample(papers: list[dict[str, Any]], rule: DisruptionRule, flt: DisruptionTeamFilter) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for paper in papers:
        if not _relevant(paper, rule, total=True):
            continue
        cc = safe_int(paper.get("citationCount"), default=0) or 0
        if flt.citation_strict_gt:
            if cc <= flt.min_citation_count:
                continue
        elif cc < flt.min_citation_count:
            continue
        if count_authors(paper) <= 0:
            continue
        out.append(paper)
    return out


def _pairs(papers: list[dict[str, Any]], ids: set[str], pr: dict[str, float], key: str, pred: Callable[[dict[str, Any]], bool]) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for paper in papers:
        if not pred(paper):
            continue
        pid = str(paper.get("paperId") or "")
        if pid not in ids or pid not in pr or (val := paper.get(key)) is None:
            continue
        xs.append(float(pr[pid]))
        ys.append(float(val))
    return xs, ys


def _tag(row: Row | None, rule: DisruptionRule) -> Row | None:
    if row:
        row["rules"] = rule.name
    return row


def _onesided_row(a: list[float], b: list[float], *, label: str, alpha: float, detail: str, test: str, equal_var: bool) -> Row | None:
    if len(a) < 2 or len(b) < 2:
        return inferential_row(test, label, np.nan, np.nan, len(a) + len(b), alpha=alpha, detail=f"{detail}; effectif insuffisant")
    t_stat, p_two = ttest_ind(a, b, equal_var=equal_var)
    p_one = float(p_two / 2 if t_stat > 0 else 1 - p_two / 2)
    return inferential_row(test, label, float(t_stat), p_one, len(a) + len(b), alpha=alpha, detail=detail)


def run_disruption_correlations(papers: list[dict[str, Any]], rule: DisruptionRule, pr_ref: dict[str, float], pr_cit: dict[str, float], ids_ref: set[str], ids_cit: set[str], *, alpha: float = ALPHA) -> list[Row]:
    rel = lambda p: _relevant(p, rule, total=True)
    p = f"[{rule.name}]"
    key = rule.total_key
    return [
        _tag(pearson_test(*_pairs(papers, ids_ref, pr_ref, key, rel), label=f"{p} pagerank references vs {key}", alpha=alpha), rule),
        _tag(pearson_test(*_pairs(papers, ids_cit, pr_cit, key, rel), label=f"{p} pagerank citations vs {key}", alpha=alpha), rule),
    ]


def run_disruption_team_tests(papers: list[dict[str, Any]], rule: DisruptionRule, *, flt: DisruptionTeamFilter | None = None, alpha: float = ALPHA) -> list[Row]:
    flt = flt or DisruptionTeamFilter(min_citation_count=rule.min_citation_count)
    sample = filter_team_sample(papers, rule, flt)
    metric = rule.total_key
    small = [float(p[metric]) for p in sample if count_authors(p) <= flt.small_team_max]
    large = [float(p[metric]) for p in sample if count_authors(p) >= flt.large_team_min]
    label = f"[{rule.name}] disruption total: n_authors<={flt.small_team_max} vs >={flt.large_team_min}"
    cc = f">{flt.min_citation_count}" if flt.citation_strict_gt else f">={flt.min_citation_count}"
    detail = f"metric={metric}; citationCount{cc}; {rule.relevant_flag}=false; {rule.total_bad_flag}=false; n_small={len(small)}; n_large={len(large)}; mean_small={np.mean(small) if small else np.nan:.6f}; mean_large={np.mean(large) if large else np.nan:.6f}"
    h0 = f"H0: mean(<={flt.small_team_max}) = mean(>={flt.large_team_min})"
    h1 = f"{h0}; H1: mean(<={flt.small_team_max}) > mean(>={flt.large_team_min})"
    rows: list[Row] = []
    if len(small) >= 2 and len(large) >= 2:
        t, p = ttest_ind(small, large, equal_var=True)
        rows.append(_tag(inferential_row("ttest_ind", label, t, p, len(small) + len(large), alpha=alpha, detail=f"{h0}; {detail}"), rule))
        if w := welch_test(small, large, label=label, alpha=alpha, detail=f"{h0}; {detail}"):
            rows.append(_tag(w, rule))
        rows.append(_tag(_onesided_row(small, large, label=label, alpha=alpha, detail=f"{h1}; {detail}", test="ttest_ind_onesided", equal_var=True), rule))
        rows.append(_tag(_onesided_row(small, large, label=label, alpha=alpha, detail=f"{h1}; {detail}", test="ttest_ind_welch_onesided", equal_var=False), rule))
    return [r for r in rows if r]


def run_disruption_team_inferential(papers: list[dict[str, Any]], *, flt: DisruptionTeamFilter | None = None, alpha: float = ALPHA) -> dict[str, Any]:
    flt = flt or DisruptionTeamFilter(min_citation_count=STRICT_RULE.min_citation_count)
    rule = next((r for r in ALL_RULES if r.min_citation_count == flt.min_citation_count), ALL_RULES[1])
    sample = filter_team_sample(papers, rule, flt)
    rows = run_disruption_team_tests(papers, rule, flt=flt, alpha=alpha)
    return {"rows": rows, "filter": flt, "sample_n": len(sample), "n_small": sum(1 for p in sample if count_authors(p) <= flt.small_team_max), "n_large": sum(1 for p in sample if count_authors(p) >= flt.large_team_min)}


def run_all_disruption_stats(papers: list[dict[str, Any]], *, alpha: float = ALPHA) -> list[Row]:
    ids_ref = {str(p["paperId"]) for p in papers if p.get("paperId") and not has_elided_references(p)}
    ids_cit = {str(p["paperId"]) for p in papers if p.get("paperId") and not has_elided_citations(p)}
    pr_ref = nx.pagerank(build_paper_citation_graph(papers, ids_ref), weight="weight", tol=1e-6)
    pr_cit = nx.pagerank(build_paper_citation_graph(papers, ids_cit, edge_source="citations"), weight="weight", tol=1e-6)
    rows: list[Row] = []
    for rule in RULE_SETS:
        rows.extend(r for r in run_disruption_correlations(papers, rule, pr_ref, pr_cit, ids_ref, ids_cit, alpha=alpha) if r)
        rows.extend(run_disruption_team_tests(papers, rule, alpha=alpha))
    return rows


def _p_value_fmt(p: float) -> str:
    if p != p:
        return ""
    return f"{p:.6g}" if p < 1e-4 else f"{p:.10g}"


def _csv_row(record_type: str, rule: DisruptionRule | None, row: Row) -> dict[str, str]:
    empty = {k: "" for k in _CSV_FIELDS}
    if record_type == "coverage":
        assert rule is not None
        n_tot = int(row["n"])
        return empty | {"record_type": "coverage", "rules": rule.name, "tolerance": rule.tolerance_label, "test": "coverage_summary", "variable": "pertinence disruption (total)", "n": str(n_tot), "detail": str(row["detail"]), "n_pertinent_in_corpus": str(row["n_pertinent_in_corpus"]), "n_total_computable": str(n_tot), "n_with_numeric_value": str(n_tot), "corpus_n": str(row["corpus_n"])}
    p = float(row["p_value"])
    return empty | {"record_type": "inferential", "rules": str(row.get("rules", "")), "test": str(row["test"]), "variable": str(row["variable"]), "statistic": f"{float(row['statistic']):.16g}" if row.get("statistic") == row.get("statistic") else "", "p_value": f"{p:.16g}", "p_value_fmt": _p_value_fmt(p), "n": str(row["n"]), "alpha": str(row.get("alpha", ALPHA)), "significant": str(row.get("significant", False)), "detail": str(row.get("detail", ""))}


def write_disruption_csvs(papers: list[dict[str, Any]], inferential: list[Row], *, out_dir: Path | None = None) -> tuple[Path, Path]:
    out_dir = out_dir or STATS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    cov = [{**{"n": sum(1 for p in papers if _relevant(p, r, total=True)), "n_pertinent_in_corpus": sum(1 for p in papers if p.get(r.relevant_flag) is False), "corpus_n": len(papers), "detail": ""}, "rules": r.name} for r in RULE_SETS]
    for c, r in zip(cov, RULE_SETS):
        c["detail"] = f"Papiers pertinents in-corpus={c['n_pertinent_in_corpus']}; disruption total calculable={c['n']}"
    cov_rows = [_csv_row("coverage", r, c) for c, r in zip(cov, RULE_SETS)]
    inf_rows = [_csv_row("inferential", None, row) for row in inferential]
    paths = out_dir / "disruption_stats.csv", out_dir / "disruption_consolidated.csv"
    for path, rows in ((paths[0], inf_rows), (paths[1], cov_rows + inf_rows)):
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    return paths


def run_and_print(papers_path: Path | None = None, *, fetch_total: bool = True, alpha: float = ALPHA, write_corpus: bool = True, write_csv: bool = True) -> list[Row]:
    path = Path(papers_path) if papers_path is not None else get_default_papers_path()
    papers = load_papers(path)
    annotate_disruptions(papers, fetch_total=fetch_total)
    if write_corpus:
        path.write_text(json.dumps(papers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = run_all_disruption_stats(papers, alpha=alpha)
    if write_csv:
        write_disruption_csvs(papers, rows)
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stats disruption (corrélations + tests équipe)")
    parser.add_argument("--input", type=Path, default=get_default_papers_path())
    parser.add_argument("--no-api-total", action="store_true")
    parser.add_argument("--no-write-corpus", action="store_true")
    parser.add_argument("--alpha", type=float, default=ALPHA)
    args = parser.parse_args(argv)
    for row in run_and_print(args.input, fetch_total=not args.no_api_total, alpha=args.alpha, write_corpus=not args.no_write_corpus):
        print(f"  {row.get('rules', ''):12} {row['test']:22} {row['variable'][:50]:50} stat={row['statistic']:.6f}  p={float(row['p_value']):.6g}  n={row['n']}")


if __name__ == "__main__":
    main()
