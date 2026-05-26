"""Graphe co-inventeurs brevets + matching inventeurs / auteurs papiers (TP2)."""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import jellyfish
import networkx as nx
import pandas as pd

from graphe_common.builders import load_paper_authors
from graphe_common.graphs import print_centralities, print_degrees, print_graph_header, print_louvain_summary
from graphe_common.io import has_elided_references
from graphe_common.stats import aligned_pearson

PATENTS_PATH = "lens-export.csv"
AUTHOR_NAME_CACHE = Path("author_data_maps_cache.json")
MATCH_THRESHOLD = 0.65


def normalize_str(text: str) -> str:
    '''
    En entrée : text, chaîne à normaliser (nom, titre, etc.)

    En sortie : chaîne ASCII minuscule sans ponctuation, espaces unifiés

    Variables : aucune variable locale notable
    '''
    text = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def parse_inventor_entries(raw: str) -> list[tuple[str, str]]:
    '''
    En entrée : raw, champ Inventors brut (séparateur ;;)

    En sortie : liste de paires (clé entité, nom affiché) sans doublons

    Variables : entries, seen, chunk, tokens, last, first_parts, display, key
    '''
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for chunk in (raw or "").split(";;"):
        tokens = normalize_str(chunk).split()
        if not tokens:
            continue
        last, first_parts = tokens[0], tokens[1:]
        display = " ".join(p.capitalize() for p in [*first_parts, last]) if first_parts else last.capitalize()
        key = f"name:{' '.join(display.lower().split())}"
        if key not in seen:
            seen.add(key)
            entries.append((key, display))
    return entries


def load_patents(patents_path: str = PATENTS_PATH) -> list[dict[str, Any]]:
    '''
    En entrée : patents_path, chemin du CSV Lens brevets (défaut PATENTS_PATH)

    En sortie : liste de dictionnaires, une ligne brevet par enregistrement

    Variables : aucune (lecture pandas directe)
    '''
    return pd.read_csv(patents_path, dtype=str, keep_default_na=False).to_dict("records")


def build_coinventor_graph(patents: list[dict[str, Any]]) -> nx.Graph:
    '''
    En entrée : patents, liste des brevets chargés

    En sortie : graphe non orienté pondéré des co-inventeurs (poids = co-brevets)

    Variables : G, edge_weights, row, inventors, a, b
    '''
    G, edge_weights = nx.Graph(), Counter()
    for row in patents:
        inventors = parse_inventor_entries(str(row.get("Inventors") or ""))
        if not inventors:
            continue
        for key, display in inventors:
            G.add_node(key, title=display)
        for (a, _), (b, _) in combinations(inventors, 2):
            edge_weights[tuple(sorted((a, b)))] += 1
    G.add_weighted_edges_from((u, v, float(w)) for (u, v), w in edge_weights.items())
    return G


def _entity_feats(entities: dict[str, str], *, last_first: bool) -> dict[str, dict[str, Any]]:
    '''
    En entrée : entities (id → nom affiché), last_first (nom de famille en premier ou dernier)

    En sortie : dictionnaire id → features (tokens, trigrammes, noms, clés de blocking)

    Variables : feats, entity_id, display, norm, toks, parts, last_names, ln0, compact, trigrams
    '''
    feats: dict[str, dict[str, Any]] = {}
    for entity_id, display in entities.items():
        norm = normalize_str(display)
        toks = norm.split()
        parts = re.split(r"\s*;;\s*|\s*;\s*|\s*&\s*|\s+and\s+", display)
        last_names = {normalize_str(p).split()[0 if last_first else -1] for p in parts if normalize_str(p).split()}
        ln0 = next(iter(last_names), "")
        compact = re.sub(r"\s+", "", norm)
        trigrams = {compact[i : i + 3] for i in range(len(compact) - 2)} if len(compact) >= 3 else ({compact} if compact else set())
        feats[entity_id] = {"display": display, "tokens": set(toks), "trigrams": trigrams, "last_names": last_names, "keys": ({f"L|{ln0}", f"S|{jellyfish.soundex(ln0)}"} if ln0 else set())}
    return feats


def _jaccard(a: set[str], b: set[str]) -> float:
    '''
    En entrée : a et b, deux ensembles de chaînes

    En sortie : coefficient de Jaccard |a∩b|/|a∪b|, ou 0.0 si union vide

    Variables : aucune
    '''
    return len(a & b) / len(a | b) if a | b else 0.0


def _combined_score(fa: dict[str, Any], fb: dict[str, Any]) -> float:
    '''
    En entrée : fa et fb, dictionnaires de features d'entités (_entity_feats)

    En sortie : score de similarité combiné ∈ [0, 1] (trigrammes, tokens, Jaro-Winkler, noms)

    Variables : t_j
    '''
    t_j = 0.6 * _jaccard(fa["trigrams"], fb["trigrams"]) + 0.4 * _jaccard(fa["tokens"], fb["tokens"])
    return 0.45 * t_j + 0.35 * jellyfish.jaro_winkler_similarity(fa["display"], fb["display"]) + 0.2 * _jaccard(fa["last_names"], fb["last_names"])


def _match_entities(source_feats: dict[str, dict[str, Any]], target_feats: dict[str, dict[str, Any]], blocking: dict[str, list[str]], threshold: float = MATCH_THRESHOLD) -> list[tuple[float, str, str]]:
    '''
    En entrée : source_feats, target_feats, blocking (clé → ids cibles), threshold (seuil minimal)

    En sortie : liste triée de (score, id_source, id_cible) au-dessus du seuil

    Variables : matches, source_id, fa, candidates, best_score, best_target
    '''
    matches: list[tuple[float, str, str]] = []
    for source_id, fa in source_feats.items():
        candidates = {tid for key in fa["keys"] for tid in blocking.get(key, [])}
        if not candidates:
            continue
        best_score, best_target = max(((_combined_score(fa, target_feats[tid]), tid) for tid in candidates), key=lambda x: x[0], default=(0.0, None))
        if best_target is not None and best_score >= threshold:
            matches.append((best_score, source_id, best_target))
    return sorted(matches, reverse=True, key=lambda x: x[0])


def _extract_npl_titles(cite: str) -> list[str]:
    '''
    En entrée : cite, texte brut du champ NPL Citations

    En sortie : liste de titres normalisés extraits entre guillemets (20–180 car.)

    Variables : aucune (compréhension regex)
    '''
    return [normalize_str(m.group(1)) for m in re.finditer(r'["\']([^"\']{20,180})["\']', cite or "")]


def _paper_title_index(papers: list[dict[str, Any]]) -> dict[str, list[tuple[str, str, str]]]:
    '''
    En entrée : papers, liste des papiers du corpus

    En sortie : index des 3 premiers mots du titre → [(titre norm., paperId, titre tronqué)]

    Variables : index, p, pid, title, nt
    '''
    index: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for p in papers:
        pid, title = str(p.get("paperId") or ""), str(p.get("title") or "")
        if not pid or not title:
            continue
        nt = normalize_str(title)
        if len(nt.split()) < 3:
            continue
        index[" ".join(nt.split()[:3])].append((nt, pid, title[:80]))
    return index


def match_npl_titles_to_papers(patents: list[dict[str, Any]], papers: list[dict[str, Any]]) -> list[tuple[float, str, str, str, str, str]]:
    '''
    En entrée : patents et papers, listes brevets et papiers

    En sortie : appariements triés (score, patent_id, titre brevet, paper_id, titre papier, kind)

    Variables : index, matches, seen, row, patent_id, ct, pt, paper_id, score, kind
    '''
    index, matches, seen = _paper_title_index(papers), [], set()
    for row in patents:
        if float(str(row.get("NPL Resolved Citation Count") or "0").strip() or "0") <= 0:
            continue
        patent_id, patent_title, cite = str(row.get("Lens ID") or ""), str(row.get("Title") or "")[:60], str(row.get("NPL Citations") or "")
        for ct in _extract_npl_titles(cite):
            if len(ct) < 20:
                continue
            for pt, paper_id, paper_title in index.get(" ".join(ct.split()[:3]), []):
                score, kind = (1.0, "exact") if ct == pt else ((0.92, "substr") if ct in pt or pt in ct else (0.0, ""))
                if score > 0 and (patent_id, paper_id) not in seen:
                    seen.add((patent_id, paper_id))
                    matches.append((score, patent_id, patent_title, paper_id, paper_title, kind))
                    break
    return sorted(matches, reverse=True, key=lambda x: x[0])


def npl_cited_paper_ids(patents: list[dict[str, Any]], papers: list[dict[str, Any]]) -> set[str]:
    return {paper_id for _, _, _, paper_id, _, _ in match_npl_titles_to_papers(patents, papers)}


def print_npl_matching(patents: list[dict[str, Any]], papers: list[dict[str, Any]]) -> None:
    '''
    En entrée : patents et papers, listes pour le matching NPL titres ↔ papiers

    En sortie : aucune (affichage console des métriques P/R/F1 et top 15)

    Variables : matches, exact_truth, predicted, tp, precision, recall, f1, resolved_n
    '''
    matches = match_npl_titles_to_papers(patents, papers)
    exact_truth = {(pat, pap) for score, pat, _, pap, _, kind in matches if kind == "exact"}
    predicted = {(pat, pap) for _, pat, _, pap, _, _ in matches}
    tp = len(predicted & exact_truth)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(exact_truth) if exact_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    resolved_n = sum(1 for r in patents if float(str(r.get("NPL Resolved Citation Count") or "0").strip() or "0") > 0)
    print(f"\n[Matching NPL titres ↔ papiers]")
    print(f"Brevets NPL résolus: {resolved_n}, liens: {len(matches)} (exact={sum(1 for *_, k in matches if k == 'exact')}, substr={sum(1 for *_, k in matches if k == 'substr')})")
    print(f"Vérité argent (exact): {len(exact_truth)}, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
    print("Top 15:")
    for score, patent_id, patent_title, paper_id, paper_title, kind in matches[:15]:
        print(f" - {score:.2f} ({kind}) | brevet {patent_id[:18]}… {patent_title} → {paper_title}")


def print_author_matching(patents: list[dict[str, Any]], papers: list[dict[str, Any]], *, threshold: float = MATCH_THRESHOLD) -> None:
    '''
    En entrée : patents, papers, threshold (seuil d'appariement inventeur ↔ auteur)

    En sortie : aucune (affichage console matching inventeurs / auteurs et F1)

    Variables : inventors, paper_authors, inv_feats, paper_feats, blocking, matches, truth, predicted
    '''
    paper_ids = {str(p["paperId"]) for p in papers if p.get("paperId") and not has_elided_references(p)}
    inventors: dict[str, str] = {}
    for row in patents:
        for key, display in parse_inventor_entries(str(row.get("Inventors") or "")):
            inventors.setdefault(key, display)
    _, raw_authors = load_paper_authors(papers, paper_ids)
    name_by_id = json.loads(AUTHOR_NAME_CACHE.read_text(encoding="utf-8")).get("author_name_map", {}) if AUTHOR_NAME_CACHE.exists() else {}
    paper_authors = {k: name_by_id[k[3:]] for k in raw_authors if k.startswith("id:") and name_by_id.get(k[3:])}
    inv_feats, paper_feats = _entity_feats(inventors, last_first=True), _entity_feats(paper_authors, last_first=False)
    blocking: dict[str, list[str]] = defaultdict(list)
    for entity_id, feat in paper_feats.items():
        for key in feat["keys"]:
            blocking[key].append(entity_id)
    matches = _match_entities(inv_feats, paper_feats, blocking, threshold=threshold)
    predicted = {(inv, paper) for _, inv, paper in matches}
    by_name: dict[str, list[str]] = defaultdict(list)
    for key, display in paper_authors.items():
        if norm := normalize_str(display):
            by_name[norm].append(key)
    truth = {(inv_k, pap_k) for inv_k, inv_d in inventors.items() for pap_k in by_name.get(normalize_str(inv_d), [])}
    tp = len(predicted & truth)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"\n[Matching inventeurs ↔ auteurs]")
    print(f"Inventeurs: {len(inventors)}, auteurs résolus: {len(paper_authors)}, blocking: {len(blocking)}, seuil={threshold}")
    print(f"Appariements: {len(matches)}, vérité argent: {len(truth)}, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
    print("Top 15:")
    for score, inv_key, paper_key in matches[:15]:
        print(f" - {score:.3f} | {inventors[inv_key]} ↔ {paper_authors[paper_key]} | Jaccard nom={_jaccard(set(inv_feats[inv_key]['last_names']), set(paper_feats[paper_key]['last_names'])):.2f}")


def print_graph_study(G: nx.Graph, *, graph_title: str | None = None, top_k: int = 10, seed: int = 42) -> None:
    '''
    En entrée : G (graphe co-inventeurs), graph_title, top_k, seed (Louvain)

    En sortie : aucune (rapport degrés, Louvain, centralités sur la console)

    Variables : DG, node_label, in_wdeg
    '''
    if G.number_of_nodes() == 0:
        raise ValueError("Le graphe est vide.")
    DG = G.to_directed()
    print_graph_header(DG, title=graph_title)
    node_label: Callable[[str], str] = lambda n: (G.nodes[n].get("title") or n)[:40]
    in_deg, _, in_wdeg, _ = print_degrees(DG, top_k=top_k, node_label=node_label)
    print_louvain_summary(DG, in_wdeg=in_wdeg, node_label=node_label, top_k=min(top_k, 5), seed=seed, node_fields=None, member_label="inventeurs")
    pr = print_centralities(DG, top_k=top_k, node_label=node_label)
    print("\n[Corrélations]")
    if r := aligned_pearson(in_deg, pr): print(f"Pearson(in-degree, pagerank): {float(r[0]):.4f}")
    print("\n" + "=" * 60)
