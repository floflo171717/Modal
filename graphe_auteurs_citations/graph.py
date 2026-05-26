"""Graphe auteur → auteur (via citations) : construction et analyse."""
from __future__ import annotations

from typing import Any, Callable

from graphe_common.graphs import print_centralities, print_degrees, print_graph_header, print_louvain_summary
from graphe_common.stats import aligned_pearson


def print_graph_study(G: Any, *, graph_title: str | None = None, top_k: int = 10, seed: int = 42, author_h_index: dict[str, int] | None = None, author_citation_count: dict[str, int] | None = None, author_fields: dict[str, set[str]] | None = None, author_venues: dict[str, set[str]] | None = None) -> None:
    '''
    En entrée : G (graphe auteur→auteur), graph_title, top_k, seed, métriques auteurs optionnelles.

    En sortie : aucune (rapport degrés, Louvain, centralités, corrélations Pearson).

    Variables : node_label, in_deg, in_wdeg, pr, metrics, name, values, m, r.
    '''
    if G.number_of_nodes() == 0: raise ValueError("Le graphe est vide.")
    print_graph_header(G, title=graph_title)
    node_label: Callable[[str], str] = lambda aid: (G.nodes[aid].get("title") or aid)[:40]
    in_deg, _, in_wdeg, _ = print_degrees(G, top_k=top_k, node_label=node_label)
    print_louvain_summary(G, in_wdeg=in_wdeg, node_label=node_label, top_k=min(top_k, 5), seed=seed, node_fields=author_fields, node_venues=author_venues, member_label="auteurs")
    pr = print_centralities(G, top_k=top_k, node_label=node_label)

    print("\n[Corrélations]")
    if r := aligned_pearson(in_deg, pr): print(f"Pearson(in-degree, pagerank): {float(r[0]):.4f}")
    metrics: dict[str, dict[str, float]] = {}
    for name, values in [("h-index", author_h_index), ("citationCount", author_citation_count)]:
        if not values: continue
        m = {n: float(values[n]) for n in G.nodes if n in values}
        metrics[name] = m
        if r := aligned_pearson(m, in_deg): print(f"Pearson({name}, in-degree): {float(r[0]):.4f}")
        if r := aligned_pearson(m, pr): print(f"Pearson({name}, pagerank): {float(r[0]):.4f}")
    if "citationCount" in metrics and "h-index" in metrics:
        if r := aligned_pearson(metrics["citationCount"], metrics["h-index"]): print(f"Pearson(citationCount, h-index): {float(r[0]):.4f}")

    for name, m in metrics.items():
        print(f"\nTop {name} ({top_k}):")
        for aid, val in sorted(m.items(), key=lambda kv: kv[1], reverse=True)[:top_k]:
            print(f" - {node_label(aid)}: {int(val)}")

    print("\n" + "=" * 60)
