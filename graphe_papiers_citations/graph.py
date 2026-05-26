"""Graphe papier → papier (via citations) : construction et analyse."""
from __future__ import annotations

from typing import Any, Callable

import networkx as nx

from graphe_common.graphs import print_centralities, print_degrees, print_graph_header, print_louvain_summary
import numpy as np

from graphe_common.stats import aligned_pearson


def compute_disruption_scores(G: nx.DiGraph) -> dict[str, float]:
    '''
    En entrée : G, graphe orienté papier → références citées

    En sortie : dict paperId → score de disruption (Wu et al.)

    Variables : refs_by_node, scores, a, refs_a, citers, a_prime
    '''
    refs_by_node = {n: set(G.successors(n)) for n in G.nodes}
    scores: dict[str, float] = {}
    for a, refs_a in refs_by_node.items():
        if not refs_a or not (citers := list(G.predecessors(a))): continue
        scores[a] = sum(1 for a_prime in citers if not refs_a.intersection(refs_by_node[a_prime])) / len(citers)
    return scores


def print_graph_study(G: Any, *, graph_title: str | None = None, top_k: int = 10, seed: int = 42, influential_counts: dict[str, int] | None = None, paper_fields: dict[str, set[str]] | None = None, paper_venues: dict[str, set[str]] | None = None) -> None:
    '''
    En entrée : G (graphe papier→papier), graph_title, top_k, seed, influential_counts, paper_fields

    En sortie : aucune (degrés, Louvain, centralités, disruption, corrélations sur la console)

    Variables : node_label, in_deg, in_wdeg, pr, disruption, inf, vals
    '''
    if G.number_of_nodes() == 0: raise ValueError("Le graphe est vide.")
    print_graph_header(G, title=graph_title)
    node_label: Callable[[str], str] = lambda pid: (G.nodes[pid].get("title") or pid)[:40]
    in_deg, _, in_wdeg, _ = print_degrees(G, top_k=top_k, node_label=node_label)
    if influential_counts:
        inf = {pid: influential_counts.get(pid, 0) for pid in G.nodes}
        print(f"\nTop influentialCitationCount ({top_k}):")
        for pid, score in sorted(inf.items(), key=lambda kv: kv[1], reverse=True)[:top_k]:
            print(f" - {node_label(pid)}: {score}")
    print_louvain_summary(G, in_wdeg=in_wdeg, node_label=node_label, top_k=min(top_k, 5), seed=seed, node_fields=paper_fields, node_venues=paper_venues, member_label="papiers")
    pr = print_centralities(G, top_k=top_k, node_label=node_label)
    disruption = compute_disruption_scores(G)
    if disruption:
        vals = list(disruption.values())
        print("\n[Disruption]")
        print(f"Articles évalués: {len(disruption)}, moyenne={float(np.mean(vals)) if vals else 0.0:.4f}, max={float(max(vals)):.4f}")
        print(f"Top disruption ({top_k}):")
        for pid, score in sorted(disruption.items(), key=lambda kv: kv[1], reverse=True)[:top_k]:
            print(f" - {node_label(pid)}: {float(score):.4f}")
    print("\n[Corrélations]")
    print(f"Pearson(in-degree, pagerank): {float(r[0] if (r := aligned_pearson(in_deg, pr)) else 0):.4f}")
    if influential_counts:
        inf = {pid: influential_counts.get(pid, 0) for pid in G.nodes}
        print(f"Pearson(in-degree, influential): {float(r[0] if (r := aligned_pearson(in_deg, inf)) else 0):.4f}")
    if disruption:
        print(f"Pearson(in-degree, disruption): {float(r[0] if (r := aligned_pearson(in_deg, disruption)) else 0):.4f}")
        print(f"Pearson(pagerank, disruption): {float(r[0] if (r := aligned_pearson(pr, disruption)) else 0):.4f}")
    print("\n" + "=" * 60)
