"""Graphe de similarité brevets (domaines d'abstract) + Louvain."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx

from graphe_common.graphs import compute_louvain, louvain_purity_evaluation, normalize_keyword, print_graph_header
from graphe_common.io import load_papers

from .patents_io import load_patents, patent_abstract, patent_id

from .config import DOMAINS_PATH, GRAPH_PATH, MIN_SHARED, N_SAMPLE, SEED, SUMMARY_PATH, SYNONYM_PATH
from .extraction import build_dictionary, match_domains


def label_patents(patents: list[dict[str, Any]], synonym_map: dict[str, str]) -> dict[str, set[str]]:
    return {pid: match_domains(patent_abstract(row), synonym_map) for row in patents if (pid := patent_id(row))}


def build_graph(patents: list[dict[str, Any]], labels: dict[str, set[str]], *, min_shared: int = MIN_SHARED) -> nx.Graph:
    G, by_domain = nx.Graph(), defaultdict(set)
    for row in patents:
        pid = patent_id(row)
        if not pid:
            continue
        domains = labels.get(pid, set())
        G.add_node(pid, title=str(row.get("Title") or "")[:120], canonicals="|".join(sorted(domains)), n_canonicals=len(domains))
        for domain in domains:
            by_domain[domain].add(pid)
    seen: set[tuple[str, str]] = set()
    for pid, domains in labels.items():
        if len(domains) < min_shared:
            continue
        for other in set().union(*(by_domain[d] for d in domains)) - {pid}:
            pair = tuple(sorted((pid, other)))
            if pair in seen:
                continue
            shared = domains & labels.get(other, set())
            if len(shared) >= min_shared:
                seen.add(pair)
                G.add_edge(pair[0], pair[1], weight=len(shared), shared_canonicals="|".join(sorted(shared)))
    return G


def _node_fields(labels: dict[str, set[str]]) -> dict[str, set[str]]:
    return {pid: {normalize_keyword(d) for d in domains} for pid, domains in labels.items()}


def _louvain_stats(G: nx.Graph, node_fields: dict[str, set[str]], *, seed: int) -> dict[str, Any]:
    communities, modularity = compute_louvain(G, seed=seed)
    sizes = sorted((len(c) for c in communities), reverse=True)
    labels = {str(n): node_fields.get(str(n), set()) for n in G.nodes}
    return {
        "louvain_seed": seed,
        "community_count": len(communities),
        "modularity": modularity,
        "singleton_count": sum(s == 1 for s in sizes),
        "communities": communities,
        "fields_of_study": louvain_purity_evaluation(G, labels, louvain_communities=communities, seed=seed),
        "fields_of_study_excl_singletons": louvain_purity_evaluation(G, labels, louvain_communities=communities, seed=seed, min_community_size=2),
    }


def _print_louvain(G: nx.Graph, stats: dict[str, Any]) -> None:
    sizes = sorted((len(c) for c in stats["communities"]), reverse=True)
    print(f"\n[Louvain] nb={stats['community_count']}, mod={float(stats['modularity']):.4f}, singletons={stats['singleton_count']}")
    print(f"Top tailles: {sizes[:10]}")
    for i, nodes in enumerate(sorted(stats["communities"], key=len, reverse=True)[:5], 1):
        if len(nodes) < 2:
            continue
        leaders = sorted(((n, G.degree(n, weight="weight")) for n in nodes), key=lambda kv: kv[1], reverse=True)[:3]
        print(f" - C{i} (taille={len(nodes)}): " + ", ".join(f"{str(G.nodes[n].get('title', n))[:40]} ({int(w)})" for n, w in leaders))
    for key, title in (("fields_of_study", "toutes comm."), ("fields_of_study_excl_singletons", "comm. >= 2")):
        block = stats.get(key) or {}
        if block.get("node_count"):
            print(f"[Purete {title}] {float(block['weighted_mean_purity']):.4f} (n={block['node_count']})")
            for row in block.get("top_keywords", [])[:10]:
                print(f" - {row['label']}: purete={float(row['purity']):.4f} (support={row['support']})")


def save(G: nx.Graph, summary: dict[str, Any], domains: dict[str, Any], synonym_map: dict[str, str]) -> None:
    for path in (GRAPH_PATH, SUMMARY_PATH, DOMAINS_PATH, SYNONYM_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, GRAPH_PATH)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    DOMAINS_PATH.write_text(json.dumps(domains, ensure_ascii=False, indent=2), encoding="utf-8")
    SYNONYM_PATH.write_text(json.dumps(synonym_map, ensure_ascii=False, indent=2), encoding="utf-8")


def run(*, patents_path: str | Path | None = None, n: int = N_SAMPLE, seed: int = SEED, min_shared: int = MIN_SHARED, seed_only: bool = True) -> nx.Graph:
    patents, papers = load_patents(patents_path), load_papers()
    synonym_map, domains = build_dictionary(patents, papers, n=n, seed=seed, seed_only=seed_only)
    labels = label_patents(patents, synonym_map)
    G = build_graph(patents, labels, min_shared=min_shared)
    node_fields = _node_fields(labels)

    n_canon = len(set(synonym_map.values()))
    labeled = sum(1 for d in labels.values() if d)
    edge_ready = sum(1 for d in labels.values() if len(d) >= min_shared)
    print(f"[Domaines] canoniques={n_canon}, mappings={len(synonym_map)}, brevets etiquetes={labeled}/{len(labels)} ({100*labeled/len(labels):.1f}%), >={min_shared} domaines={edge_ready}")
    per_domain = sorted(((dom, sum(1 for ds in labels.values() if dom in ds)) for dom in set(synonym_map.values())), key=lambda kv: -kv[1])
    print("Top domaines:", ", ".join(f"{d} ({c})" for d, c in per_domain[:12]))

    print_graph_header(G, title="Graphe brevets par domaines d'abstract (seed 41)")
    louvain = _louvain_stats(G, node_fields, seed=seed)
    _print_louvain(G, louvain)

    components = list(nx.connected_components(G))
    louvain_json = {k: v for k, v in louvain.items() if k != "communities"}
    summary = {
        "seed_only": seed_only,
        "canonical_domains": n_canon,
        "synonym_mappings": len(synonym_map),
        "patents_labeled": labeled,
        "patents_labeled_rate": round(labeled / len(labels), 4) if labels else 0.0,
        "patents_min_shared_domains": edge_ready,
        "domain_counts": {d: c for d, c in per_domain},
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": round(nx.density(G), 6),
        "components": len(components),
        "largest_component": max((len(c) for c in components), default=0),
        "top_degree": [{"id": pid, "degree": d, "title": G.nodes[pid].get("title", "")} for pid, d in sorted(G.degree(), key=lambda kv: kv[1], reverse=True)[:10]],
        "louvain": louvain_json,
    }
    save(G, summary, domains, synonym_map)
    print(f"\n-> {GRAPH_PATH}\n-> {SUMMARY_PATH}")
    return G


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Graphe domaines brevets + Louvain")
    p.add_argument("--patents", type=Path, default=None)
    p.add_argument("--n", type=int, default=N_SAMPLE)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--min-shared", type=int, default=MIN_SHARED)
    p.add_argument("--spaCy-sample", action="store_true", help="Enrichir le dictionnaire via spaCy (defaut: seed 41 domaines seulement)")
    args = p.parse_args()
    run(patents_path=args.patents, n=args.n, seed=args.seed, min_shared=args.min_shared, seed_only=not args.spaCy_sample)


if __name__ == "__main__":
    main()
