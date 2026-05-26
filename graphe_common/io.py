from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

PAPERS_PATH = "papers_array.json"


def get_default_papers_path() -> Path:
    '''
    En entrée : aucun paramètre (lit éventuellement la variable d'environnement ML_GRAPH_INPUT).

    En sortie : chemin Path vers le fichier JSON des papiers par défaut.

    Variables : override (surcharge du chemin via ML_GRAPH_INPUT).
    '''
    override = os.environ.get("ML_GRAPH_INPUT", "").strip()
    root = Path(__file__).resolve().parents[1]
    return root / override if override else root / PAPERS_PATH


def safe_int(x: Any, default: int | None = None) -> int | None:
    '''
    En entrée : valeur x à convertir ; default optionnel si conversion impossible.

    En sortie : entier converti ou default (None par défaut).

    Variables : s (chaîne nettoyée pour les conversions depuis str).
    '''
    if x is None: return default
    if isinstance(x, int) and not isinstance(x, bool): return x
    if isinstance(x, float) and x == int(x): return int(x)
    if isinstance(x, str) and (s := x.strip()).lstrip("-+").isdigit(): return int(s)
    return default


def iter_papers(papers_path: str | Path | None = None, *, limit: int | None = None) -> Iterator[dict[str, Any]]:
    path = Path(papers_path) if papers_path is not None else get_default_papers_path()
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if limit is not None and i >= limit:
                    break
                if line.strip():
                    yield json.loads(line)
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Format attendu: tableau JSON dans {path}")
    for i, paper in enumerate(data):
        if limit is not None and i >= limit:
            break
        yield paper


def load_papers(papers_path: str | Path | None = None, *, limit: int | None = None) -> list[dict[str, Any]]:
    return list(iter_papers(papers_path, limit=limit))


def has_elided_references(paper: dict[str, Any]) -> bool:
    '''
    En entrée : dictionnaire papier (champ openAccessPdf).

    En sortie : True si le disclaimer indique des références élidées.

    Variables : oa (sous-dictionnaire openAccessPdf).
    '''
    oa = paper.get("openAccessPdf")
    return isinstance(oa, dict) and isinstance(oa.get("disclaimer"), str) and "elided" in oa["disclaimer"] and "'references'" in oa["disclaimer"]


def has_elided_citations(paper: dict[str, Any]) -> bool:

    oa = paper.get("openAccessPdf")
    return isinstance(oa, dict) and isinstance(oa.get("disclaimer"), str) and "elided" in oa["disclaimer"] and "'citations'" in oa["disclaimer"]


def iter_link_paper_ids(papers: list[dict[str, Any]], field: str, usable_ids: set[str]):

    for paper in papers:
        if str(paper.get("paperId") or "") not in usable_ids or not (items := paper.get(field) or []):
            continue
        for item in items:
            if isinstance(item, dict) and (pid := item.get("paperId")):
                yield str(pid)


def count_within_tol(actual: int, expected: int | None, tol: float = 0.03) -> bool:

    if expected is None:
        return False
    if expected == 0:
        return actual == 0
    return abs(actual - expected) / expected <= tol


def reference_count_unusable(paper: dict[str, Any]) -> bool:
    """Wu requiert une bibliographie : referenceCount absent ou nul."""
    rc = safe_int(paper.get("referenceCount"))
    return rc is None or rc <= 0


def not_relevant_disruption(paper: dict[str, Any]) -> bool:

    """Papier non fiable pour la disruption (listes incomplètes / references élidées)."""
    if reference_count_unusable(paper):
        return True
    refs = paper.get("references")
    ref_count = safe_int(paper.get("referenceCount"))
    if not refs:
        return True
    if len(refs) != ref_count:
        return True
    cit_count = safe_int(paper.get("citationCount"))
    if cit_count is None or len(paper.get("citations") or []) != cit_count:
        return True
    return has_elided_references(paper)


def not_relevant_disruption_relaxed(paper: dict[str, Any], tol: float) -> bool:

    """Disruption allégée : referenceCount requis, tolérance relative sur les listes."""
    if reference_count_unusable(paper):
        return True
    ref_count = safe_int(paper.get("referenceCount"))
    refs = paper.get("references") or []
    if not refs:
        return True
    cit_count = safe_int(paper.get("citationCount"))
    if cit_count is None or not count_within_tol(len(paper.get("citations") or []), cit_count, tol):
        return True
    if not count_within_tol(len(refs), ref_count, tol):
        return True
    return False


def ref_in_corpus_not_relevant(focal: dict[str, Any], by_id: dict[str, dict[str, Any]], *, flag: str = "notrelevantdisruption") -> bool:

    for item in focal.get("references") or []:
        if isinstance(item, dict) and (raw := item.get("paperId")):
            if (ref := by_id.get(str(raw))) and (ref.get(flag) or reference_count_unusable(ref)):
                return True
    return False


def corpus_link_coverage(papers: list[dict[str, Any]]) -> dict[str, Any]:

    """paperIds distincts dans references / citations, part hors corpus JSON."""
    corpus_ids = {str(p["paperId"]) for p in papers if p.get("paperId")}
    out: dict[str, Any] = {"n_corpus_papers": len(corpus_ids)}
    for key, field, usable in (("references", "references", {str(p["paperId"]) for p in papers if p.get("paperId") and not has_elided_references(p)}), ("citations", "citations", {str(p["paperId"]) for p in papers if p.get("paperId") and not has_elided_citations(p)})):
        ids = set(iter_link_paper_ids(papers, field, usable))
        in_c = ids & corpus_ids
        n = len(ids)
        out[key] = {"distinct_paper_ids": n, "distinct_in_corpus": len(in_c), "distinct_out_of_corpus": len(ids - corpus_ids), "rate_in_corpus": len(in_c) / n if n else 0.0, "rate_out_of_corpus": (n - len(in_c)) / n if n else 0.0, "papers_with_list": sum(1 for p in papers if str(p.get("paperId") or "") in usable and p.get(field))}
    return out


def corpus_link_coverage_rows(coverage: dict[str, Any]) -> list[dict[str, Any]]:

    fields = ("distinct_paper_ids", "distinct_in_corpus", "distinct_out_of_corpus", "rate_in_corpus", "rate_out_of_corpus", "papers_with_list")
    n = coverage["n_corpus_papers"]
    return [{"kind": k, "n_corpus_papers": n, **{f: coverage[k][f] for f in fields}} for k in ("references", "citations")]
