"""Points d'entrée unifiés pour les graphes de citations papiers/auteurs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import networkx as nx

from graphe_common.builders import (
    EdgeSource,
    build_author_citation_graph,
    build_paper_citation_graph,
    load_author_citation_count,
    load_author_h_index,
    load_influential_citation_counts,
    load_paper_authors,
)
from graphe_common.graphs import (
    extract_paper_fields_of_study,
    load_author_fields_of_study,
    load_author_publication_venues,
    load_paper_publication_venues,
)
from graphe_common.inferential import (
    print_domain_inferential_report,
    print_paper_inferential_stats,
    run_author_domain_inferential,
    run_inferential_stats,
    run_paper_domain_inferential,
    run_team_disruption_test,
)
from graphe_common.io import has_elided_citations, has_elided_references, load_papers

HINDEX_CACHE = Path("author_hindex_cache.json")
CITATION_CACHE = Path("author_citation_count_cache.json")


@dataclass(frozen=True)
class CitationGraphSpec:
    entity: Literal["papers", "authors"]
    edge_source: EdgeSource
    graph_title: str
    build_msg: str
    inferential_title: str | None = None


def _paper_ids(papers: list[dict], edge_source: EdgeSource) -> set[str]:
    if edge_source == "citations":
        return {str(p["paperId"]) for p in papers if p.get("paperId") and not has_elided_citations(p)}
    return {str(p["paperId"]) for p in papers if p.get("paperId") and not has_elided_references(p)}


def run_paper_citation_main(spec: CitationGraphSpec) -> None:
    from graphe_papiers_citations.graph import compute_disruption_scores, print_graph_study

    papers = load_papers()
    paper_ids = _paper_ids(papers, spec.edge_source)
    print(f"-> {len(paper_ids)} papiers (edge_source={spec.edge_source})")
    influential = load_influential_citation_counts(papers, paper_ids)
    paper_fields = {str(obj["paperId"]): extract_paper_fields_of_study(obj) for obj in papers if obj.get("paperId") in paper_ids}
    paper_venues = load_paper_publication_venues(papers, paper_ids)
    print(spec.build_msg)
    G = build_paper_citation_graph(papers, paper_ids, edge_source=spec.edge_source)
    assert isinstance(G, nx.DiGraph)
    print(f"-> nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
    disruption = compute_disruption_scores(G)
    subset = [obj for obj in papers if obj.get("paperId") in paper_ids]
    rows = run_inferential_stats(subset)
    if disruption:
        rows.extend(run_team_disruption_test(subset, disruption))
    title = spec.inferential_title.format(n=len(subset)) if spec.inferential_title and "{n}" in spec.inferential_title else (spec.inferential_title or f"Tests statistiques papiers (graphe {spec.edge_source}, n={len(subset)})")
    print_paper_inferential_stats(rows, title=title)
    print_graph_study(G, graph_title=spec.graph_title, influential_counts=influential, paper_fields=paper_fields, paper_venues=paper_venues)


def run_author_citation_main(spec: CitationGraphSpec) -> None:
    from graphe_auteurs_citations.graph import print_graph_study

    papers = load_papers()
    paper_ids = _paper_ids(papers, spec.edge_source)
    print(f"-> {len(paper_ids)} papiers (edge_source={spec.edge_source})")
    paper_to_authors, author_names = load_paper_authors(papers, paper_ids)
    author_fields = load_author_fields_of_study(papers, paper_ids, paper_to_authors)
    author_venues = load_author_publication_venues(papers, paper_ids, paper_to_authors)
    print(f"-> {len(author_names)} auteurs")
    author_keys = set(author_names)
    h_index = load_author_h_index(author_keys, cache_path=HINDEX_CACHE)
    citation_count = load_author_citation_count(author_keys, cache_path=CITATION_CACHE)
    print(spec.build_msg)
    G = build_author_citation_graph(papers, paper_ids, paper_to_authors, author_names, edge_source=spec.edge_source)
    assert isinstance(G, nx.DiGraph)
    print(f"-> nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
    print_graph_study(G, graph_title=spec.graph_title, author_h_index=h_index, author_citation_count=citation_count, author_fields=author_fields, author_venues=author_venues)
    graph_papers = [p for p in papers if p.get("paperId") in paper_ids]
    print_domain_inferential_report(run_paper_domain_inferential(graph_papers), title="Papiers du graphe")
    print_domain_inferential_report(
        run_author_domain_inferential({n: author_fields[n] for n in G.nodes if n in author_fields}, citation_count),
        title="Auteurs du graphe",
    )
