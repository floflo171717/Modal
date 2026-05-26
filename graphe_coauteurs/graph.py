"""Graphe de co-auteurs : analyse Louvain, ground truth et corrélations communautaires."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import pearsonr

from graphe_common.graphs import (
    compute_louvain,
    normalize_keyword,
    print_centralities,
    print_degrees,
    print_graph_header,
    print_louvain_summary,
)
from graphe_common.inferential import count_authors
from graphe_common.io import get_default_papers_path, load_papers, safe_int

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
DEFAULT_GRAPH_PATH = REPO_ROOT / "coauthor_graph.graphml"
DEFAULT_CACHE_PATH = REPO_ROOT / "author_data_maps_cache.json"

_CORRELATION_PAIRS = [
    ("size", "union_paper_count"),
    ("size", "field_of_study_purity"),
    ("field_of_study_purity", "union_paper_count"),
    ("field_of_study_purity", "publication_venue_purity"),
    ("size", "publication_venue_purity"),
    ("size", "citation_count_mean"),
    ("union_paper_count", "citation_count_mean"),
    ("field_of_study_purity", "citation_count_mean"),
    ("publication_venue_purity", "citation_count_mean"),
    ("field_of_study_purity", "mean_authors_per_paper"),
]


def load_author_data_maps(papers_path: Path, cache_path: Path) -> tuple[dict, dict, dict]:
    '''
    En entrée : papers_path (corpus JSON), cache_path (fichier cache auteurs)

    En sortie : tuple (hindex_map, author_name_map, author_details_map)

    Variables : cached
    '''
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding='utf-8'))
        print(f'Cache chargé: {cache_path}')
        return cached.get('hindex_map') or {}, cached.get('author_name_map') or {}, cached.get('author_details_map') or {}
    from graphe_common.builders import load_or_build_author_data_maps
    return load_or_build_author_data_maps(papers_path, cache_path)


def author_id_from_node(node: str) -> str | None:
    '''
    En entrée : node, identifiant de nœud du graphe (ex. "id:…")

    En sortie : author_id extrait ou None si le préfixe id: est absent

    Variables : aucune
    '''
    node = str(node)
    return node.split("id:", 1)[1] if node.startswith("id:") else None


def build_author_fields_from_details(author_details_map: dict[str, dict], graph_nodes: list[str]) -> dict[str, set[str]]:
    '''
    En entrée : author_details_map, graph_nodes (nœuds du graphe co-auteurs)

    En sortie : dict nœud → ensemble de champs d'étude normalisés (ground truth)

    Variables : fields_by_node, node, aid, raw, fields
    '''
    fields_by_node: dict[str, set[str]] = {}
    for node in graph_nodes:
        if not (aid := author_id_from_node(node)): continue
        raw = (author_details_map.get(aid) or {}).get("fieldsOfStudy") or {}
        fields = {normalize_keyword(f) for f, c in raw.items() if c and str(f).strip()}
        if fields: fields_by_node[node] = fields
    return fields_by_node


def build_author_venues_from_details(author_details_map: dict[str, dict], graph_nodes: list[str]) -> dict[str, set[str]]:
    '''
    En entrée : author_details_map, graph_nodes

    En sortie : dict nœud → ensemble de noms de revues normalisés

    Variables : venues_by_node, node, aid, raw, names
    '''
    venues_by_node: dict[str, set[str]] = {}
    for node in graph_nodes:
        if not (aid := author_id_from_node(node)): continue
        raw = (author_details_map.get(aid) or {}).get("publicationVenues") or {}
        names = {normalize_keyword(str(v.get("name") or "").strip()) for v in raw.values() if isinstance(v, dict) and str(v.get("name") or "").strip()}
        if names: venues_by_node[node] = names
    return venues_by_node


def _community_correlation_vectors(
    communities: list[set[str]],
    *,
    author_details_map: dict,
    author_fields: dict[str, set[str]],
    author_venues: dict[str, set[str]],
    citation_counts: dict[str, int],
    paper_author_counts: dict[str, int],
) -> dict[str, list[float]]:
    '''
    En entrée : communities, author_details_map, author_fields, author_venues, citation_counts, paper_author_counts

    En sortie : dict métrique → vecteur (une valeur par communauté) pour corrélations Pearson

    Variables : vecs, comm, size, papers, fc, vc, n, cit_vals, auth_vals
    '''
    keys = ["size", "union_paper_count", "field_of_study_purity", "publication_venue_purity", "citation_count_mean", "mean_authors_per_paper"]
    vecs: dict[str, list[float]] = {k: [] for k in keys}
    for comm in communities:
        size = len(comm)
        papers: set[str] = set()
        for n in comm:
            if aid := author_id_from_node(n):
                papers.update(str(pid) for pid in ((author_details_map.get(aid) or {}).get("paperIds") or []) if pid)
        fc, vc = Counter(), Counter()
        for n in comm:
            for f in author_fields.get(n, set()): fc[f] += 1
            for v in author_venues.get(n, set()): vc[v] += 1
        cit_vals = [citation_counts.get(pid, 0) for pid in papers]
        auth_vals = [paper_author_counts.get(pid, 0) for pid in papers]
        vecs["size"].append(float(size))
        vecs["union_paper_count"].append(float(len(papers)))
        vecs["field_of_study_purity"].append(max(fc.values(), default=0) / size if size else 0.0)
        vecs["publication_venue_purity"].append(max(vc.values(), default=0) / size if size else 0.0)
        vecs["citation_count_mean"].append(float(np.mean(cit_vals)) if cit_vals else 0.0)
        vecs["mean_authors_per_paper"].append(float(np.mean(auth_vals)) if auth_vals else 0.0)
    return vecs


def print_graph_study(
    G: Any,
    *,
    graph_title: str | None = None,
    top_k: int = 10,
    seed: int = 42,
    author_fields: dict[str, set[str]] | None = None,
    author_venues: dict[str, set[str]] | None = None,
    author_details_map: dict | None = None,
    citation_counts: dict[str, int] | None = None,
    author_name_map: dict[str, str] | None = None,
    paper_author_counts: dict[str, int] | None = None,
) -> None:
    '''
    En entrée : G (graphe co-auteurs), graph_title, top_k, seed, ground truth fields/venues, métriques

    En sortie : aucune (header, degrés, Louvain, centralités, corrélations communautaires)

    Variables : DG, node_label, in_wdeg, communities, vecs, n, xa, ya, r, p
    '''
    DG = G.to_directed() if not G.is_directed() else G
    if DG.number_of_nodes() == 0: raise ValueError("Le graphe est vide.")
    print_graph_header(DG, title=graph_title)

    def node_label(n: str) -> str:
        if author_name_map and (aid := author_id_from_node(n)):
            name = author_name_map.get(aid, "")
            if name: return name[:40]
        return (G.nodes[n].get("label") or n)[:40]

    _, _, in_wdeg, _ = print_degrees(DG, top_k=top_k, node_label=node_label)
    communities = print_louvain_summary(
        DG, in_wdeg=in_wdeg, node_label=node_label, top_k=min(top_k, 5), seed=seed,
        node_fields=author_fields, node_venues=author_venues, member_label="auteurs",
    )
    print_centralities(DG, top_k=top_k, node_label=node_label)

    if author_details_map is not None:
        vecs = _community_correlation_vectors(
            communities,
            author_details_map=author_details_map,
            author_fields=author_fields or {},
            author_venues=author_venues or {},
            citation_counts=citation_counts or {},
            paper_author_counts=paper_author_counts or {},
        )
        n = len(communities)
        print("\n[Corrélations de Pearson]")
        for x, y in _CORRELATION_PAIRS:
            xa, ya = np.array(vecs[x]), np.array(vecs[y])
            if np.std(xa) > 0 and np.std(ya) > 0:
                r, p = pearsonr(xa, ya)
                print(f" - {x} vs {y}: r={r:.6f}, p={p:.6f}, n={n}")

    print("\n" + "=" * 60)


def main() -> None:
    '''
    En entrée : arguments CLI (--graph, --cache, --papers, --top-k, --seed)

    En sortie : aucune (charge le graphe GraphML et lance print_graph_study)

    Variables : p, args, G, names, details, papers, nodes, cc, pac
    '''
    import networkx as nx
    p = argparse.ArgumentParser(description="Louvain co-auteurs")
    p.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    p.add_argument("--papers", type=Path, default=get_default_papers_path())
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    G = nx.read_graphml(args.graph)
    print(f"Graphe {args.graph.name}: |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")
    _, names, details = load_author_data_maps(args.papers, args.cache)
    papers = load_papers(args.papers)
    cc = {str(obj.get("paperId") or "").strip(): safe_int(obj.get("citationCount")) for obj in papers if str(obj.get("paperId") or "").strip()}
    pac = {str(obj.get("paperId") or "").strip(): count_authors(obj) for obj in papers if str(obj.get("paperId") or "").strip()}
    nodes = list(G.nodes)
    print_graph_study(
        G, graph_title="Co-auteurs", top_k=args.top_k, seed=args.seed,
        author_fields=build_author_fields_from_details(details, nodes),
        author_venues=build_author_venues_from_details(details, nodes),
        author_details_map=details, citation_counts=cc, author_name_map=names,
        paper_author_counts=pac,
    )


def run_corpus_stats() -> None:
    '''
    En entrée : argument CLI --input (chemin corpus JSON papiers)

    En sortie : aucune (stats descriptives corpus + tests inférentiels sur la console)

    Variables : papers, venues, years, kws, authors, pr, ar
    '''
    from collections import Counter as _Counter
    from graphe_common.builders import normalize_author_key
    from graphe_common.inferential import print_paper_inferential_stats, run_author_inventor_inferential, run_inferential_stats
    p = argparse.ArgumentParser(description="Stats corpus")
    p.add_argument("--input", type=Path, default=get_default_papers_path())
    args = p.parse_args()
    papers = json.loads(args.input.read_text(encoding="utf-8"))
    venues, years, kws = _Counter(), _Counter(), _Counter()
    authors: set[str] = set()
    for paper in papers:
        if not isinstance(paper, dict): continue
        y = safe_int(paper.get("year"))
        if y: years[y] += 1
        v = (paper.get("publicationVenue") or {}).get("name") if isinstance(paper.get("publicationVenue"), dict) else paper.get("venue")
        if v: venues[str(v).strip()] += 1
        for kw in (paper.get("fieldsOfStudy") or []) + [x.get("category") for x in (paper.get("s2FieldsOfStudy") or []) if isinstance(x, dict)]:
            if kw: kws[str(kw).strip()] += 1
        for a in paper.get("authors") or []:
            if k := normalize_author_key(a): authors.add(k)
    print(f"\n[Corpus] {len(papers)} papiers, {len(authors)} auteurs")
    print("Top venues:", ", ".join(f"{v}({c})" for v, c in venues.most_common(10)))
    print("Top keywords:", ", ".join(f"{k}({c})" for k, c in kws.most_common(10)))
    print("Publications/an:", ", ".join(f"{y}:{c}" for y, c in sorted(years.items())[-8:]))
    pr = run_inferential_stats(papers)
    print_paper_inferential_stats(pr, title="Tests papiers (corpus)")
    ar = run_author_inventor_inferential(papers)
    print_paper_inferential_stats(ar, title="Tests inventeur")


if __name__ == "__main__":
    main()
