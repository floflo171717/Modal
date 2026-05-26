"""Point d'entrée : python -m graphe_papiers_citations2 (graphe 6, edge_source=citations)"""
from graphe_common.cli import CitationGraphSpec, run_paper_citation_main

if __name__ == "__main__":
    run_paper_citation_main(CitationGraphSpec(
        entity="papers", edge_source="citations",
        graph_title="Graphe citations papiers 2 (citations)",
        build_msg="Construction du graphe (A cite B ssi A in citations(B))...",
        inferential_title="Tests statistiques papiers (graphe citations, n={n})",
    ))
