"""Graphes NetworkX : Louvain, ground truth, métriques, export GraphML."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import networkx as nx
import numpy as np

from graphe_common.stats import safe_eigenvector

EXCLUDED_GROUND_TRUTH_KEYWORD = "computer science"
LOUVAIN_COMMUNITY_ATTR = "community"


def to_undirected_weighted_graph(G: Any) -> nx.Graph:
    '''
    En entrée : G, graphe orienté ou non avec nœuds et arêtes pondérées

    En sortie : graphe NetworkX non orienté où les poids des arêtes parallèles sont sommés

    Variables : UG (graphe non orienté construit), u, v, data, w (poids d'arête)
    '''
    UG = nx.Graph()
    UG.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))
        if UG.has_edge(u, v):
            UG[u][v]["weight"] = float(UG[u][v].get("weight", 1.0)) + w
        else:
            UG.add_edge(u, v, weight=w)
    return UG


def compute_louvain(G: Any, *, seed: int = 42) -> tuple[list[set[str]], float]:
    '''
    En entrée : G (graphe), seed (graine pour l'algorithme Louvain, défaut 42)

    En sortie : tuple (liste de communautés comme ensembles de nœuds, score de modularité)

    Variables : ug (graphe non orienté pondéré), raw (partition Louvain brute), communities
    '''
    ug = to_undirected_weighted_graph(G)
    raw = nx.community.louvain_communities(ug, weight="weight", seed=seed)
    communities = [set(c) for c in raw]
    return communities, nx.community.modularity(ug, raw, weight="weight")


def graph_connectivity_diagnostics(G: Any) -> dict[str, int]:

    isolated_directed = sum(1 for n in G.nodes if G.in_degree(n) == 0 and G.out_degree(n) == 0)
    ug = to_undirected_weighted_graph(G)
    isolated_undirected = sum(1 for n in ug.nodes if ug.degree(n) == 0)
    components = list(nx.weakly_connected_components(G) if G.is_directed() else nx.connected_components(G))
    sizes = sorted((len(c) for c in components), reverse=True)
    return {"isolated_directed": isolated_directed, "isolated_undirected": isolated_undirected, "weak_component_count": len(components), "largest_weak_component_size": sizes[0] if sizes else 0, "second_largest_weak_component_size": sizes[1] if len(sizes) > 1 else 0}


def print_graph_header(G: Any, *, title: str | None = None) -> None:
    '''
    En entrée : G (graphe), title (titre optionnel affiché dans l'en-tête)

    En sortie : aucune (affichage console du nombre de nœuds, arêtes et densité)

    Variables : aucune variable locale significative
    '''
    print("\n" + "=" * 60)
    if title: print(title)
    print(f"|V|={G.number_of_nodes()}, |E|={G.number_of_edges()}, densité={nx.density(G):.6f}")
    print("=" * 60)


def print_degrees(G: Any, *, top_k: int, node_label: Callable[[str], str]) -> tuple[dict, dict, dict, dict]:
    '''
    En entrée : G (graphe orienté), top_k (nombre de nœuds à afficher), node_label (fonction de libellé)

    En sortie : tuple (in_deg, out_deg, in_wdeg, out_wdeg) dictionnaires de degrés simples et pondérés

    Variables : in_deg, out_deg, in_wdeg, out_wdeg, name, vals, v, wvals, node, score
    '''
    in_deg, out_deg = dict(G.in_degree()), dict(G.out_degree())
    in_wdeg = {n: deg for n, deg in G.in_degree(weight="weight")}
    out_wdeg = {n: deg for n, deg in G.out_degree(weight="weight")}
    print("\n[Degrés]")
    for name, vals in [("in", in_deg.values()), ("out", out_deg.values())]:
        v = list(vals)
        print(f"{name}-degree: min={min(v)}, mean={float(np.mean(v)) if v else 0.0:.4f}, max={max(v)}")
    wvals = list(in_wdeg.values())
    print(f"in-degree pondéré: mean={float(np.mean(wvals)) if wvals else 0.0:.4f}, max={max(in_wdeg.values())}")
    print(f"\nTop in-degree ({top_k}):")
    for node, score in sorted(in_deg.items(), key=lambda kv: kv[1], reverse=True)[:top_k]:
        print(f" - {node_label(node)}: {score}")
    return in_deg, out_deg, in_wdeg, out_wdeg


def print_centralities(G: Any, *, top_k: int, node_label: Callable[[str], str]) -> dict[str, float]:
    '''
    En entrée : G (graphe orienté pondéré), top_k (nombre de nœuds à lister), node_label (libellé nœud)

    En sortie : dictionnaire PageRank {nœud: score}

    Variables : in_deg_c, out_deg_c, in_cvals, out_cvals, pr, pr_vals, authorities, node, score
    '''
    print("\n[Centralités]")
    in_deg_c, out_deg_c = nx.in_degree_centrality(G), nx.out_degree_centrality(G)
    in_cvals, out_cvals = list(in_deg_c.values()), list(out_deg_c.values())
    print(f"In-degree centrality: max={float(max(in_deg_c.values())):.4f}, mean={float(np.mean(in_cvals)) if in_cvals else 0.0:.4f}")
    print(f"Out-degree centrality: max={float(max(out_deg_c.values())):.4f}, mean={float(np.mean(out_cvals)) if out_cvals else 0.0:.4f}")
    pr = nx.pagerank(G, weight="weight", tol=1e-06)
    pr_vals = list(pr.values())
    print(f"PageRank: max={float(max(pr.values())):.4f}, mean={float(np.mean(pr_vals)) if pr_vals else 0.0:.4f}")
    print("Top PageRank:")
    for node, score in sorted(pr.items(), key=lambda kv: kv[1], reverse=True)[:top_k]:
        print(f" - {node_label(node)}: {float(score):.6f}")
    _, authorities = nx.hits(G, max_iter=1000, normalized=True)
    print("Top authorities (HITS):")
    for node, score in sorted(authorities.items(), key=lambda kv: kv[1], reverse=True)[:top_k]:
        print(f" - {node_label(node)}: {float(score):.6f}")
    return pr


def normalize_keyword(keyword: str) -> str:
    '''
    En entrée : keyword (chaîne de mot-clé ou discipline)

    En sortie : mot-clé normalisé (minuscules, espaces multiples fusionnés)

    Variables : aucune variable locale significative
    '''
    return " ".join((keyword or "").strip().lower().split())


def extract_paper_fields_of_study(paper: dict[str, Any]) -> set[str]:
    '''
    En entrée : paper (dictionnaire article Semantic Scholar)

    En sortie : ensemble de disciplines normalisées extraites de l'article

    Variables : fields (ensemble accumulé), item, value, cat
    '''
    fields: set[str] = set()
    for item in paper.get("fieldsOfStudy") or []:
        if value := str(item).strip(): fields.add(normalize_keyword(value))
    for item in paper.get("s2FieldsOfStudy") or []:
        if isinstance(item, dict) and (cat := item.get("category")) and str(cat).strip():
            fields.add(normalize_keyword(str(cat)))
    return fields


def load_paper_fields_of_study(papers: list[dict[str, Any]], paper_ids: set[str]) -> dict[str, set[str]]:
    '''
    En entrée : papers (liste d'articles), paper_ids (identifiants à conserver)

    En sortie : dictionnaire {paperId: ensemble de disciplines}

    Variables : obj (article courant)
    '''
    return {str(obj["paperId"]): extract_paper_fields_of_study(obj) for obj in papers if obj.get("paperId") in paper_ids}


def load_author_fields_of_study(papers: list[dict[str, Any]], paper_ids: set[str], paper_to_authors: dict[str, list[str]]) -> dict[str, set[str]]:
    '''
    En entrée : papers, paper_ids, paper_to_authors (mapping article → clés auteurs)

    En sortie : dictionnaire {clé auteur: ensemble de disciplines agrégées sur ses articles}

    Variables : fields_by_author, obj, pid, paper_fields, author_key
    '''
    fields_by_author: dict[str, set[str]] = {}
    for obj in papers:
        pid = obj.get("paperId")
        if not pid or pid not in paper_ids: continue
        paper_fields = extract_paper_fields_of_study(obj)
        if not paper_fields: continue
        for author_key in paper_to_authors.get(str(pid), []):
            fields_by_author.setdefault(author_key, set()).update(paper_fields)
    return fields_by_author


def extract_paper_publication_venues(paper: dict[str, Any]) -> set[str]:

    venues: set[str] = set()
    if isinstance(publication_venue := paper.get("publicationVenue"), dict):
        if name := str(publication_venue.get("name") or "").strip():
            venues.add(normalize_keyword(name))
    if name := str(paper.get("venue") or "").strip():
        venues.add(normalize_keyword(name))
    return venues


def load_paper_publication_venues(papers: list[dict[str, Any]], paper_ids: set[str]) -> dict[str, set[str]]:

    return {str(obj["paperId"]): extract_paper_publication_venues(obj) for obj in papers if obj.get("paperId") in paper_ids}


def load_author_publication_venues(papers: list[dict[str, Any]], paper_ids: set[str], paper_to_authors: dict[str, list[str]]) -> dict[str, set[str]]:

    venues_by_author: dict[str, set[str]] = {}
    for obj in papers:
        pid = obj.get("paperId")
        if not pid or pid not in paper_ids: continue
        paper_venues = extract_paper_publication_venues(obj)
        if not paper_venues: continue
        for author_key in paper_to_authors.get(str(pid), []):
            venues_by_author.setdefault(author_key, set()).update(paper_venues)
    return venues_by_author


def purity(detected: list[set[str]], ground_truth: list[set[str]]) -> float:
    '''
    En entrée : detected (communautés détectées), ground_truth (partition de référence)

    En sortie : score de pureté entre 0 et 1 (fraction de nœuds bien classés)

    Variables : nodes (union des nœuds), d, t
    '''
    nodes = set().union(*detected, *ground_truth)
    if not nodes: return 0.0
    return sum(max((len(d & t) for t in ground_truth), default=0) for d in detected if d) / len(nodes)


def binary_ground_truth_partition(nodes: list[str], node_fields: dict[str, set[str]], keyword: str) -> list[set[str]]:
    '''
    En entrée : nodes (liste de nœuds), node_fields (disciplines par nœud), keyword (mot-clé cible)

    En sortie : partition binaire [nœuds avec le mot-clé, nœuds sans le mot-clé]

    Variables : kw (mot-clé normalisé), with_kw
    '''
    kw = normalize_keyword(keyword)
    with_kw = {n for n in nodes if kw in node_fields.get(n, set())}
    return [with_kw, set(nodes) - with_kw]


def collect_evaluation_keywords(node_fields: dict[str, set[str]], *, exclude: str | None = EXCLUDED_GROUND_TRUTH_KEYWORD) -> list[str]:
    '''
    En entrée : node_fields (disciplines par nœud), exclude (mot-clé à exclure, optionnel)

    En sortie : liste triée de mots-clés candidats pour l'évaluation ground truth

    Variables : keywords (ensemble puis liste triée)
    '''
    keywords: set[str] = set().union(*node_fields.values())
    if exclude: keywords.discard(normalize_keyword(exclude))
    return sorted(keywords, key=str.lower)


def evaluate_keyword_ground_truths(nodes: list[str], node_fields: dict[str, set[str]], louvain_communities: list[set[str]], keywords: list[str]) -> list[tuple[str, float, int, int]]:
    '''
    En entrée : nodes, node_fields, louvain_communities, keywords (liste de disciplines à tester)

    En sortie : liste de tuples (mot-clé, pureté, support positif, support négatif) triée par pureté

    Variables : results, keyword, gt, support
    '''
    results = []
    for keyword in keywords:
        gt = binary_ground_truth_partition(nodes, node_fields, keyword)
        support = len(gt[0])
        results.append((keyword, purity(louvain_communities, gt) if support else 0.0, support, len(nodes) - support))
    results.sort(key=lambda row: (-row[1], -row[2], row[0].lower()))
    return results


def weighted_mean_purity(results: list[tuple[str, float, int, int]]) -> float:

    weighted = sum(score * support for _, score, support, _ in results if support)
    total = sum(support for _, _, support, _ in results if support)
    return weighted / total if total else 0.0


def louvain_purity_evaluation(G: Any, node_labels: dict[str, set[str]], *, louvain_communities: list[set[str]] | None = None, seed: int = 42, exclude: str | None = EXCLUDED_GROUND_TRUTH_KEYWORD, min_community_size: int = 1) -> dict[str, Any]:

    louvain_communities = louvain_communities or compute_louvain(G, seed=seed)[0]
    eval_communities = [c for c in louvain_communities if len(c) >= min_community_size]
    nodes = sorted({str(n) for c in eval_communities for n in c})
    if not nodes:
        return {"weighted_mean_purity": 0.0, "best_label": "", "best_purity": 0.0, "best_support": 0, "best_rest": 0, "keyword_count": 0, "community_count": 0, "node_count": 0, "top_keywords": []}
    labels_in_graph = {n: node_labels.get(n, set()) for n in nodes}
    keywords = collect_evaluation_keywords(labels_in_graph, exclude=exclude)
    results = evaluate_keyword_ground_truths(nodes, labels_in_graph, eval_communities, keywords)
    best_label, best_purity, best_support, best_rest = results[0] if results else ("", 0.0, 0, len(nodes))
    return {"weighted_mean_purity": weighted_mean_purity(results), "best_label": best_label, "best_purity": best_purity, "best_support": best_support, "best_rest": best_rest, "keyword_count": len(keywords), "community_count": len(eval_communities), "node_count": len(nodes), "top_keywords": [{"label": label, "purity": score, "support": support, "rest": rest} for label, score, support, rest in results[:15]]}


def analyze_louvain_communities(G: Any, *, node_fields: dict[str, set[str]] | None = None, node_venues: dict[str, set[str]] | None = None, seed: int = 42, field_exclude: str | None = EXCLUDED_GROUND_TRUTH_KEYWORD) -> dict[str, Any]:
    '''
    En entrée : G, node_fields, node_venues, seed, field_exclude.

    En sortie : métriques ou structure de résultats (voir type de retour).

    Variables : exclude, int, labels, louvain_communities, min_community_size, modularity, s, sizes, venues.
    '''
    communities, modularity = compute_louvain(G, seed=seed)
    sizes = [len(c) for c in communities]
    out: dict[str, Any] = {"louvain_seed": seed, "community_count": len(communities), "modularity": modularity, "singleton_count": sum(s == 1 for s in sizes), "average_community_size": float(np.mean([float(s) for s in sizes])) if sizes else 0.0, "max_community_size": max(sizes) if sizes else 0, "communities": communities, **graph_connectivity_diagnostics(G)}
    if node_fields is not None:
        labels = {str(n): node_fields.get(str(n), set()) for n in G.nodes}
        out["fields_of_study"] = louvain_purity_evaluation(G, labels, louvain_communities=communities, seed=seed, exclude=field_exclude)
        out["fields_of_study_excl_singletons"] = louvain_purity_evaluation(G, labels, louvain_communities=communities, seed=seed, exclude=field_exclude, min_community_size=2)
    if node_venues is not None:
        venues = {str(n): node_venues.get(str(n), set()) for n in G.nodes}
        out["publication_venues"] = louvain_purity_evaluation(G, venues, louvain_communities=communities, seed=seed, exclude=None)
        out["publication_venues_excl_singletons"] = louvain_purity_evaluation(G, venues, louvain_communities=communities, seed=seed, exclude=None, min_community_size=2)
    return out


def count_communities_with_labels(louvain: list[set[str]], node_labels: dict[str, set[str]], labels: list[str]) -> dict[str, int]:
    '''
    En entrée : louvain (communautés), node_labels (étiquettes par nœud), labels (étiquettes à compter)

    En sortie : dictionnaire {étiquette: nombre de communautés contenant au moins un nœud avec cette étiquette}

    Variables : counts, community, comm_labels, label
    '''
    counts = {label: 0 for label in labels}
    for community in louvain:
        comm_labels = set().union(*(node_labels.get(n, set()) for n in community))
        for label in labels:
            if label in comm_labels: counts[label] += 1
    return counts


def print_binary_ground_truth_evaluation(G: Any, node_labels: dict[str, set[str]], *, louvain_communities: list[set[str]] | None = None, seed: int = 42, category: str = "discipline", exclude: str | None = EXCLUDED_GROUND_TRUTH_KEYWORD) -> list[tuple[str, float, int, int]]:
    '''
    En entrée : G, node_labels, louvain_communities (optionnel), seed, category, exclude

    En sortie : liste de résultats d'évaluation (mot-clé, pureté, supports) comme evaluate_keyword_ground_truths

    Variables : nodes, labels_in_graph, labels, results, best_label, best_purity, best_support, best_rest, label, score, support, rest
    '''
    nodes = list(G.nodes)
    if not nodes: print(f"Ground truth {category}: graphe vide."); return []
    labels_in_graph = {n: node_labels.get(n, set()) for n in nodes}
    labels = collect_evaluation_keywords(labels_in_graph, exclude=exclude)
    if not labels: print(f"Ground truth {category}: aucune étiquette."); return []
    louvain_communities = louvain_communities or compute_louvain(G, seed=seed)[0]
    results = evaluate_keyword_ground_truths(nodes, labels_in_graph, louvain_communities, labels)
    print(f"\n[Ground truth — pureté par {category}]")
    if results:
        best_label, best_purity, best_support, best_rest = results[0]
        print(f"Meilleure: « {best_label} » (pureté={float(best_purity):.4f}, support={best_support}/{best_support + best_rest})")
        print(f"Pureté moyenne pondérée ({category}): {float(weighted_mean_purity(results)):.4f}")
        for label, score, support, rest in results[:15]:
            print(f" - {label}: pureté={float(score):.4f} (support={support}, hors={rest})")
    return results


def print_louvain_summary(G: Any, *, in_wdeg: dict[str, float], node_label: Callable[[str], str], top_k: int = 5, max_communities: int = 5, seed: int = 42, node_fields: dict[str, set[str]] | None = None, node_venues: dict[str, set[str]] | None = None, member_label: str = "nœuds") -> list[set[str]]:
    '''
    En entrée : G, in_wdeg, node_label, top_k, max_communities, seed, node_fields (optionnel), member_label

    En sortie : liste des communautés Louvain détectées

    Variables : communities, modularity, sizes, idx, nodes, leaders
    '''
    print("\n[Communautés]")
    stats = analyze_louvain_communities(G, node_fields=node_fields, node_venues=node_venues, seed=seed)
    communities = stats["communities"]
    modularity = float(stats["modularity"])
    sizes = sorted((len(c) for c in communities), reverse=True)
    print(f"Louvain: nb={stats['community_count']}, modularité={float(modularity):.4f}, singletons={stats['singleton_count']}")
    print(f"Top tailles: {sizes[:10]}")
    for idx, nodes in enumerate(sorted(communities, key=len, reverse=True)[:max_communities], start=1):
        leaders = sorted({n: float(in_wdeg.get(n, 0.0)) for n in nodes}.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        print(f" - C{idx} (taille={len(nodes)}): {', '.join(f'{node_label(n)} ({int(s)})' for n, s in leaders)}")
    if node_fields is not None:
        field_stats = stats.get("fields_of_study") or {}
        print(f"\n[Ground truth — pureté par discipline]")
        print(f"Meilleure: « {field_stats.get('best_label', '')} » (pureté={float(field_stats.get('best_purity', 0.0)):.4f}, support={field_stats.get('best_support', 0)}/{field_stats.get('best_support', 0) + field_stats.get('best_rest', 0)})")
        print(f"Pureté moyenne pondérée (fields): {float(field_stats.get('weighted_mean_purity', 0.0)):.4f}")
        field_ns = stats.get("fields_of_study_excl_singletons") or {}
        if field_ns.get("node_count"):
            print(f"Pureté moyenne pondérée (fields, sans singletons, n={field_ns['node_count']}): {float(field_ns.get('weighted_mean_purity', 0.0)):.4f}")
        for row in field_stats.get("top_keywords") or []:
            print(f" - {row['label']}: pureté={float(row['purity']):.4f} (support={row['support']}, hors={row['rest']})")
    if node_venues is not None:
        venue_stats = stats.get("publication_venues") or {}
        print(f"\n[Ground truth — pureté par venue]")
        print(f"Meilleure: « {venue_stats.get('best_label', '')} » (pureté={float(venue_stats.get('best_purity', 0.0)):.4f}, support={venue_stats.get('best_support', 0)}/{venue_stats.get('best_support', 0) + venue_stats.get('best_rest', 0)})")
        print(f"Pureté moyenne pondérée (venue): {float(venue_stats.get('weighted_mean_purity', 0.0)):.4f}")
        venue_ns = stats.get("publication_venues_excl_singletons") or {}
        if venue_ns.get("node_count"):
            print(f"Pureté moyenne pondérée (venue, sans singletons, n={venue_ns['node_count']}): {float(venue_ns.get('weighted_mean_purity', 0.0)):.4f}")
        for row in venue_stats.get("top_keywords") or []:
            print(f" - {row['label']}: pureté={float(row['purity']):.4f} (support={row['support']}, hors={row['rest']})")
    return communities


def write_graphml_with_louvain(G: Any, path: str | Path, *, seed: int = 42, source_file: str | None = None) -> tuple[list[set[str]], float, Path]:
    '''
    En entrée : G, path (fichier de sortie), seed, source_file (métadonnée optionnelle)

    En sortie : tuple (communautés Louvain, modularité, chemin Path du fichier GraphML écrit)

    Variables : communities, modularity, node_map, out
    '''
    communities, modularity = compute_louvain(G, seed=seed)
    node_map = {str(n): str(i) for i, comm in enumerate(communities) for n in comm}
    nx.set_node_attributes(G, {str(n): node_map.get(str(n), "") for n in G.nodes}, LOUVAIN_COMMUNITY_ATTR)
    G.graph["louvain_modularity"], G.graph["louvain_seed"] = float(modularity), int(seed)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if source_file: G.graph["source_file"] = source_file
    nx.write_graphml(G, out)
    print(f"GraphML: {out} (|V|={G.number_of_nodes()}, communities={len(communities)}, mod={float(modularity):.4f})")
    return communities, modularity, out
