"""Point d'entrée : python -m graphe_auteurs_citations2 (graphe 5, edge_source=citations)"""
from graphe_common.cli import CitationGraphSpec, run_author_citation_main

if __name__ == "__main__":
    run_author_citation_main(CitationGraphSpec(
        entity="authors", edge_source="citations",
        graph_title="Graphe citations auteurs 2 (citations)",
        build_msg="Construction du graphe (A cite B ssi A in citations(B))...",
    ))
