"""Point d'entrée : python -m graphe_brevets"""
from __future__ import annotations

import networkx as nx

from graphe_common.inferential import print_paper_inferential_stats, run_author_inventor_inferential, run_npl_citation_inferential
from graphe_common.io import load_papers

from .graph import build_coinventor_graph, load_patents, print_author_matching, print_graph_study, print_npl_matching

if __name__ == "__main__":
    patents, papers = load_patents(), load_papers()
    print(f"-> {len(patents)} brevets, {len(papers)} papiers")
    print("Construction du graphe co-inventeurs...")
    G = build_coinventor_graph(patents)
    assert isinstance(G, nx.Graph)
    print(f"-> nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
    print_graph_study(G, graph_title="Graphe co-inventeurs (brevets)")
    print_npl_matching(patents, papers)
    print_paper_inferential_stats(run_npl_citation_inferential(papers), title="Tests centralité / citations ↔ cité NPL")
    print_author_matching(patents, papers)
    print_paper_inferential_stats(run_author_inventor_inferential(papers), title="Tests h-index / centralités / PageRank ↔ inventeur")
