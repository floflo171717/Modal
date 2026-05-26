"""Construction de graphes auteurs/papiers et enrichissement S2."""
from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterator, Literal

import networkx as nx
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from graphe_common.graphs import EXCLUDED_GROUND_TRUTH_KEYWORD, extract_paper_fields_of_study, normalize_keyword
from graphe_common.io import get_default_papers_path, iter_papers, load_papers, safe_int
from graphe_common.stats import aligned_pearson

GRAPHS_DIR = Path(__file__).resolve().parents[1]
GRAPH_BASE = "https://api.semanticscholar.org/graph/v1"
API_KEY = os.environ.get("S2_API_KEY", "2y5uBuUCdd49YFT5Jsuhv5LGG2Wfh5CJaLIZfFz5")

EdgeSource = Literal["references", "citations"]


def normalize_author_name(name: str) -> str:
    '''
    En entrée : name (nom d'auteur brut)

    En sortie : nom normalisé (espaces multiples réduits, sans espaces en bordure)

    Variables : aucune variable locale significative
    '''
    return " ".join((name or "").split()).strip()


def extract_author_key(author_obj: Any) -> tuple[str, str] | None:
    '''
    En entrée : author_obj (objet auteur Semantic Scholar, dict ou autre)

    En sortie : tuple (clé auteur « id:… » ou « name:… », nom affichable) ou None si invalide

    Variables : author_id, author_name
    '''
    if not isinstance(author_obj, dict): return None
    author_id, author_name = str(author_obj.get("authorId") or "").strip(), normalize_author_name(str(author_obj.get("name") or ""))
    if author_id: return (f"id:{author_id}", author_name or author_id)
    if author_name: return (f"name:{author_name.lower()}", author_name)
    return None


def normalize_author_key(author: dict[str, Any]) -> str | None:
    '''
    En entrée : author (dictionnaire auteur)

    En sortie : clé auteur normalisée (chaîne « id:… » ou « name:… ») ou None

    Variables : p (résultat intermédiaire de extract_author_key)
    '''
    return (p := extract_author_key(author)) and p[0]


def load_paper_authors(papers: list[dict[str, Any]], paper_ids: set[str]) -> tuple[dict[str, list[str]], dict[str, str]]:
    '''
    En entrée : papers (liste d'articles), paper_ids (identifiants à traiter)

    En sortie : tuple (paper_to_authors, author_display_names)

    Variables : paper_to_authors, author_display_names, obj, pid, keys, seen, author_obj, parsed
    '''
    paper_to_authors, author_display_names = {}, {}
    for obj in papers:
        if not (pid := obj.get("paperId")) or pid not in paper_ids or not isinstance(obj.get("authors") or [], list): continue
        keys, seen = [], set()
        for author_obj in obj["authors"]:
            if (parsed := extract_author_key(author_obj)) and parsed[0] not in seen:
                seen.add(parsed[0]); keys.append(parsed[0])
                if parsed[1]: author_display_names[parsed[0]] = parsed[1]
        if keys: paper_to_authors[str(pid)] = keys
    return paper_to_authors, author_display_names


def _add_weighted_edges(G: nx.Graph | nx.DiGraph, weights: Counter) -> None:
    '''
    En entrée : G (graphe NetworkX), weights (compteur de paires (u,v) → poids entier)

    En sortie : aucune (modifie G en place en ajoutant les arêtes pondérées)

    Variables : aucune variable locale significative (générateur dans add_edges_from)
    '''
    G.add_edges_from((u, v, {"weight": int(w)}) for (u, v), w in weights.items())


def _paper_citation_edge_weights(papers: list[dict[str, Any]], paper_ids: set[str], *, edge_source: EdgeSource) -> Counter[tuple[str, str]]:

    weights: Counter[tuple[str, str]] = Counter()
    for src, tgt in iter_paper_citation_edges(papers, paper_ids, edge_source=edge_source):
        weights[(src, tgt)] += 1
    return weights


def _init_paper_citation_nodes(G: nx.DiGraph, papers: list[dict[str, Any]], ids: set[str]) -> None:

    for obj in papers:
        if (pid := obj.get("paperId")) and str(pid) in ids:
            G.add_node(str(pid), year=safe_int(obj.get("year")), title=obj.get("title") or "")


def iter_paper_citation_edges(papers: list[dict[str, Any]], paper_ids: set[str], *, edge_source: EdgeSource = "references") -> Iterator[tuple[str, str]]:

    """Arêtes orientées (A, B) = « A cite B ».

    - ``references`` : B apparaît dans ``references`` de A (bibliographie du citeur).
    - ``citations``  : A apparaît dans ``citations`` de B (citeurs du cité).
    """
    ids = {str(pid) for pid in paper_ids}
    if edge_source == "references":
        for obj in papers:
            if not (src := obj.get("paperId")) or str(src) not in ids:
                continue
            src = str(src)
            for ref in obj.get("references") or []:
                if isinstance(ref, dict) and (raw := ref.get("paperId")) and (tgt := str(raw)) in ids:
                    yield src, tgt
        return
    if edge_source == "citations":
        by_id = {str(p["paperId"]): p for p in papers if p.get("paperId") and str(p["paperId"]) in ids}
        for tgt in ids:
            for citer in by_id.get(tgt, {}).get("citations") or []:
                if isinstance(citer, dict) and (raw := citer.get("paperId")) and (src := str(raw)) in ids:
                    yield src, tgt
        return
    raise ValueError(f"edge_source inconnu: {edge_source!r} (attendu 'references' ou 'citations')")


def build_paper_citation_graph(papers: list[dict[str, Any]], paper_ids: set[str], *, edge_source: EdgeSource = "references") -> nx.DiGraph:
    '''
    En entrée : papers, paper_ids, edge_source (« references » ou « citations »)

    En sortie : graphe orienté article→article cité, pondéré par nombre de citations

    Variables : ids, G, obj, pid, weights, src, tgt
    '''
    ids = {str(pid) for pid in paper_ids}
    G = nx.DiGraph()
    _init_paper_citation_nodes(G, papers, ids)
    _add_weighted_edges(G, _paper_citation_edge_weights(papers, ids, edge_source=edge_source))
    G.add_nodes_from((pid, {"year": None, "title": ""}) for pid in ids - set(G.nodes))
    return G


def build_author_citation_graph(papers: list[dict[str, Any]], paper_ids: set[str], paper_to_authors: dict[str, list[str]], author_display_names: dict[str, str], *, edge_source: EdgeSource = "references") -> nx.DiGraph:
    '''
    En entrée : papers, paper_ids, paper_to_authors, author_display_names, edge_source

    En sortie : graphe orienté auteur→auteur pondéré par citations entre articles

    Variables : G, weights, src_pid, tgt_pid, src_authors, tgt_authors, sa, ta
    '''
    G = nx.DiGraph()
    G.add_nodes_from((k, {"title": d}) for k, d in author_display_names.items())
    weights: Counter[tuple[str, str]] = Counter()
    for src_pid, tgt_pid in iter_paper_citation_edges(papers, paper_ids, edge_source=edge_source):
        src_authors = paper_to_authors.get(src_pid, [])
        tgt_authors = paper_to_authors.get(tgt_pid, [])
        if src_authors and tgt_authors:
            weights.update((sa, ta) for sa in src_authors for ta in tgt_authors)
    _add_weighted_edges(G, weights)
    nx.set_node_attributes(G, {n: author_display_names.get(n, n) for n in G.nodes}, "title")
    return G


def coauthor_node_label(author_key: str, display_name: str) -> str:
    '''
    En entrée : author_key (clé « id:… » ou « name:… »), display_name (nom affichable)

    En sortie : libellé court pour l'affichage du nœud (identifiant ou nom)

    Variables : aucune variable locale significative
    '''
    return author_key.split(":", 1)[1] if author_key.startswith("id:") else (display_name or author_key)


def build_coauthor_graph(paper_to_authors: dict[str, list[str]], author_display_names: dict[str, str]) -> nx.Graph:
    '''
    En entrée : paper_to_authors, author_display_names

    En sortie : graphe non orienté de co-auteurship pondéré par nombre de articles communs

    Variables : G, weights, paper_count, authors, a, b
    '''
    G = nx.Graph(); G.add_nodes_from((k, {"label": coauthor_node_label(k, d)}) for k, d in author_display_names.items())
    weights, paper_count = Counter(), Counter()
    for authors in paper_to_authors.values():
        if not authors: continue
        paper_count.update(authors); weights.update(tuple(sorted((a, b))) for a, b in combinations(authors, 2))
    G.add_weighted_edges_from((u, v, float(w)) for (u, v), w in weights.items())
    nx.set_node_attributes(G, {n: int(paper_count.get(n, 0)) for n in G.nodes}, "paper_count")
    return G


def _load_author_cache(path: Path) -> dict[str, int]:
    '''
    En entrée : path (fichier JSON de cache métriques auteur par identifiant S2)

    En sortie : dictionnaire {authorId: valeur entière ≥ 0} ou {} si fichier absent/invalide

    Variables : raw (contenu JSON), k, x, v
    '''
    if not path.exists(): return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(k).strip(): v for k, x in (raw if isinstance(raw, dict) else {}).items() if str(k).strip() and (v := safe_int(x)) is not None and v >= 0}


def _metrics_for_author_keys(author_keys: set[str], cache_by_id: dict[str, int], *, log_prefix: str) -> dict[str, int]:
    '''
    En entrée : author_keys (clés « id:… »), cache_by_id (métriques par authorId), log_prefix (préfixe log)

    En sortie : dictionnaire {clé auteur « id:… »: métrique} pour les auteurs trouvés dans le cache

    Variables : author_ids, result, aid
    '''
    author_ids = {k[3:] for k in author_keys if k.startswith("id:") and len(k) > 3}
    result = {f"id:{aid}": cache_by_id[aid] for aid in author_ids if aid in cache_by_id}
    print(f"{log_prefix}: {len(result)} auteurs disponibles (cache local)."); return result


def _load_author_metric(author_keys: set[str], *, cache_path: Path, name: str) -> dict[str, int]:
    cache_by_id = _load_author_cache(cache_path)
    print(f"{name}: {len(cache_by_id)} entrées chargées depuis {cache_path.name}." if cache_by_id else f"{name}: cache vide ou introuvable ({cache_path.name}).")
    return _metrics_for_author_keys(author_keys, cache_by_id, log_prefix=name)


def load_author_h_index(author_keys: set[str], *, cache_path: Path) -> dict[str, int]:
    '''
    En entrée : author_keys, cache_path (fichier cache h-index)

    En sortie : dictionnaire {clé auteur: h-index}

    Variables : cache_by_id
    '''
    return _load_author_metric(author_keys, cache_path=cache_path, name="h-index")


def load_author_citation_count(author_keys: set[str], *, cache_path: Path) -> dict[str, int]:
    '''
    En entrée : author_keys, cache_path (fichier cache citationCount)

    En sortie : dictionnaire {clé auteur: nombre de citations}

    Variables : cache_by_id
    '''
    return _load_author_metric(author_keys, cache_path=cache_path, name="citationCount")


def load_influential_citation_counts(papers: list[dict[str, Any]], paper_ids: set[str]) -> dict[str, int]:
    '''
    En entrée : papers, paper_ids

    En sortie : dictionnaire {paperId: influentialCitationCount}

    Variables : obj
    '''
    return {str(obj["paperId"]): safe_int(obj.get("influentialCitationCount"), default=0) or 0 for obj in papers if obj.get("paperId") in paper_ids}


def build_author_position_weighted_counters(papers_path: Path, author_ids: set[str]) -> dict[str, Counter[str]]:
    '''
    En entrée : papers_path (fichier JSON articles), author_ids (identifiants S2 ciblés)

    En sortie : dictionnaire {authorId: Counter(discipline → poids positionnel 1/équipe)}

    Variables : counters, paper, fields, authors, team, w, author, aid, field_name
    '''
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    if not author_ids: return counters
    for paper in iter_papers(papers_path):
        fields, authors = list(dict.fromkeys(extract_paper_fields_of_study(paper))), paper.get("authors") or []
        if not fields or not isinstance(authors, list) or not (team := sum(1 for a in authors if isinstance(a, dict) and a.get("authorId"))): continue
        w = 1.0 / team
        for author in authors:
            if isinstance(author, dict) and (aid := str(author.get("authorId") or "").strip()) in author_ids:
                for field_name in fields: counters[aid][field_name] += w
    return counters


def build_author_position_weighted_fields(papers_path: Path, author_ids: set[str], *, top_k: int) -> dict[str, set[str]]:
    '''
    En entrée : papers_path, author_ids, top_k (nombre max de disciplines par auteur)

    En sortie : dictionnaire {clé « id:authorId »: ensemble des top_k disciplines pondérées}

    Variables : excluded, out, aid, counter, selected, field_name, norm
    '''
    excluded, out = normalize_keyword(EXCLUDED_GROUND_TRUTH_KEYWORD), {}
    for aid, counter in build_author_position_weighted_counters(papers_path, author_ids).items():
        selected: set[str] = set()
        for field_name, _ in counter.most_common():
            if (norm := normalize_keyword(field_name)) and norm != excluded: selected.add(norm)
            if len(selected) >= top_k: break
        if selected: out[f"id:{aid}"] = selected
    return out


def empty_author_details(author_id: str) -> dict:
    '''
    En entrée : author_id (identifiant Semantic Scholar)

    En sortie : dictionnaire de détails auteur vide avec structure par défaut

    Variables : aucune variable locale significative
    '''
    return {"authorId": author_id, "name": "", "affiliations": [], "homepage": "", "paperCount": None, "citationCount": None, "hIndex": None, "externalIds": {}, "paperIds": [], "publicationVenues": {}, "fieldsOfStudy": {}}


def ensure_author_details_shape(details: dict) -> dict:
    '''
    En entrée : details (dictionnaire détails auteur partiel ou complet)

    En sortie : dictionnaire fusionné et typé selon le schéma empty_author_details

    Variables : author_id, shaped, key, default
    '''
    author_id = str(details.get("authorId") or "").strip(); shaped = {**empty_author_details(author_id), **details}
    for key, default in empty_author_details(author_id).items():
        if key not in shaped or not isinstance(shaped[key], type(default)): shaped[key] = default
    return shaped


def merge_unique_strings(target: list[str], items) -> None:
    '''
    En entrée : target (liste à enrichir), items (itérable de chaînes candidates)

    En sortie : aucune (ajoute les chaînes non vides uniques à target)

    Variables : item, x
    '''
    target.extend(x for item in items or [] if (x := str(item).strip()) and x not in target)


def extract_paper_venue(paper: dict) -> tuple[str, dict]:
    '''
    En entrée : paper (dictionnaire article)

    En sortie : tuple (clé venue « id:… » ou « name:… », métadonnées venue) ou ("", {})

    Variables : publication_venue, venue_name, venue_id, venue_key
    '''
    if isinstance(publication_venue := paper.get("publicationVenue"), dict) and (venue_name := str(publication_venue.get("name") or "").strip()):
        venue_id = str(publication_venue.get("id") or "").strip(); venue_key = venue_id or f"name:{venue_name}"
        return venue_key, {"id": venue_id, "name": venue_name, "type": str(publication_venue.get("type") or "").strip()}
    if (venue_name := str(paper.get("venue") or "").strip()): return f"name:{venue_name}", {"id": "", "name": venue_name, "type": ""}
    return "", {}


def cache_needs_paper_enrichment(author_details_map: dict[str, dict]) -> bool:
    '''
    En entrée : author_details_map (cache détails auteurs)

    En sortie : True si paperIds, publicationVenues ou fieldsOfStudy manquent pour au moins un auteur

    Variables : flags, details, key
    '''
    if not author_details_map: return True
    flags = {"paperIds": False, "publicationVenues": False, "fieldsOfStudy": False}
    for details in author_details_map.values():
        if not isinstance(details, dict): continue
        for key in flags:
            if details.get(key): flags[key] = True
    return not all(flags.values())


def clear_paper_derived_fields(author_details_map: dict[str, dict]) -> None:
    '''
    En entrée : author_details_map (modifié en place)

    En sortie : aucune (réinitialise paperIds, publicationVenues, fieldsOfStudy pour chaque auteur)

    Variables : author_id, details, shaped
    '''
    for author_id, details in author_details_map.items():
        if isinstance(details, dict):
            shaped = ensure_author_details_shape(details); shaped["paperIds"], shaped["publicationVenues"], shaped["fieldsOfStudy"] = [], {}, {}; author_details_map[author_id] = shaped


def enrich_author_details_from_papers(papers_json_path: Path, author_details_map: dict[str, dict], author_name_map: dict[str, str], hindex_map: dict[str, int | float | None]) -> None:
    '''
    En entrée : papers_json_path, author_details_map, author_name_map, hindex_map (tous modifiés en place)

    En sortie : aucune (enrichit les maps depuis le fichier papers JSON)

    Variables : papers, paper, paper_id, venue_key, venue_info, paper_fields, authors, author, author_id, author_name, details, venues, field, key, raw_hindex, prev
    '''
    if not papers_json_path.exists(): print(f"Attention: {papers_json_path} introuvable, enrichissement papers ignore."); return
    clear_paper_derived_fields(author_details_map)
    for paper in iter_papers(papers_json_path):
        paper_id, venue_key, venue_info, paper_fields = str(paper.get("paperId") or "").strip(), *extract_paper_venue(paper), list(extract_paper_fields_of_study(paper))
        authors = paper.get("authors") or []
        if not isinstance(authors, list): continue
        for author in authors:
            if not isinstance(author, dict) or not (author_id := str(author.get("authorId") or "").strip()): continue
            if (author_name := str(author.get("name") or "").strip()): author_name_map[author_id] = author_name
            details = ensure_author_details_shape(author_details_map.setdefault(author_id, empty_author_details(author_id)))
            if paper_id and paper_id not in details["paperIds"]: details["paperIds"].append(paper_id)
            if author_name: details["name"] = author_name
            if venue_key:
                venues = details["publicationVenues"]; venues.setdefault(venue_key, {**venue_info, "count": 0}); venues[venue_key]["count"] += 1
            for field in paper_fields: details["fieldsOfStudy"][field] = details["fieldsOfStudy"].get(field, 0) + 1
            if isinstance(author.get("affiliations"), list): merge_unique_strings(details["affiliations"], author.get("affiliations"))
            for key in ("homepage", "paperCount", "citationCount", "hIndex"):
                if author.get(key) is not None: details[key] = author.get(key)
            if isinstance(author.get("externalIds"), dict): details["externalIds"] = author.get("externalIds")
            if (raw_hindex := author.get("hIndex")) is not None and ((prev := hindex_map.get(author_id)) is None or raw_hindex > prev):
                hindex_map[author_id], details["hIndex"] = raw_hindex, raw_hindex


def build_author_data_maps(papers_json_path: Path) -> tuple[dict[str, int | float | None], dict[str, str], dict[str, dict]]:
    '''
    En entrée : papers_json_path (fichier papers_array.json)

    En sortie : tuple (hindex_map, author_name_map, author_details_map)

    Variables : hindex_map, author_name_map, author_details_map
    '''
    hindex_map, author_name_map, author_details_map = {}, {}, {}
    enrich_author_details_from_papers(papers_json_path, author_details_map, author_name_map, hindex_map)
    return hindex_map, author_name_map, author_details_map


def get_session() -> requests.Session:
    '''
    En entrée : aucune (utilise API_KEY et constantes module)

    En sortie : Session requests avec en-tête API et retries HTTP

    Variables : session
    '''
    session = requests.Session()
    if API_KEY: session.headers.update({"x-api-key": API_KEY})
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=5, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET", "POST"])))
    return session


def chunked(values: list[str], chunk_size: int):
    '''
    En entrée : values (liste de chaînes), chunk_size (taille des lots)

    En sortie : générateur de sous-listes consécutives de taille chunk_size

    Variables : i
    '''
    for i in range(0, len(values), chunk_size): yield values[i : i + chunk_size]


def is_author_details_incomplete(details: dict) -> bool:
    '''
    En entrée : details (dictionnaire détails auteur)

    En sortie : True si un champ obligatoire (nom, affiliations, homepage, métriques, externalIds) est manquant

    Variables : aucune variable locale significative
    '''
    return not details or any([not details.get("name"), details.get("affiliations") in (None, []), not details.get("homepage"), details.get("paperCount") is None, details.get("citationCount") is None, details.get("hIndex") is None, details.get("externalIds") in (None, {})])


def fill_missing_author_details_from_api(author_ids: list[str], author_name_map: dict[str, str], author_details_map: dict[str, dict], hindex_map: dict[str, int | float | None]) -> None:
    '''
    En entrée : author_ids, author_name_map, author_details_map, hindex_map (modifiés en place)

    En sortie : aucune (complète les détails manquants via l'API Semantic Scholar batch)

    Variables : missing_ids, session, endpoint, author_chunk, response, payload, requested_id, author_obj, name, details, key
    '''
    missing_ids = [author_id for author_id in author_ids if is_author_details_incomplete(author_details_map.get(author_id, {}))]
    if not missing_ids: return
    session, endpoint = get_session(), f"{GRAPH_BASE}/author/batch"; print(f"Enrichissement details API pour {len(missing_ids)} auteurs...")
    for author_chunk in chunked(missing_ids, 1000):
        response = session.post(endpoint, params={"fields": "authorId,name,affiliations,homepage,paperCount,citationCount,hIndex,externalIds"}, json={"ids": author_chunk}, timeout=120)
        response.raise_for_status(); payload = response.json()
        if not isinstance(payload, list): continue
        for requested_id, author_obj in zip(author_chunk, payload):
            if not author_obj: continue
            name = str(author_obj.get("name") or "").strip()
            if name: author_name_map[requested_id] = name
            details = ensure_author_details_shape(author_details_map.setdefault(requested_id, empty_author_details(requested_id)))
            details.update({"authorId": requested_id, "name": name or details.get("name", ""), "homepage": author_obj.get("homepage") or details.get("homepage", "")})
            merge_unique_strings(details["affiliations"], author_obj.get("affiliations"))
            for key in ("paperCount", "citationCount", "hIndex"):
                if author_obj.get(key) is not None:
                    details[key] = author_obj.get(key)
                    if key == "hIndex": hindex_map[requested_id] = author_obj.get(key)
            if isinstance(author_obj.get("externalIds"), dict): details["externalIds"] = author_obj.get("externalIds")
        time.sleep(1.1)


def _write_author_cache(cache_path: Path, hindex_map: dict, author_name_map: dict, author_details_map: dict) -> None:
    '''
    En entrée : cache_path, hindex_map, author_name_map, author_details_map

    En sortie : aucune (écrit le cache JSON sur disque)

    Variables : aucune variable locale significative
    '''
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"hindex_map": hindex_map, "author_name_map": author_name_map, "author_details_map": author_details_map}, ensure_ascii=False), encoding="utf-8")


def load_or_build_author_data_maps(papers_json_path: Path, cache_path: Path) -> tuple[dict[str, int | float | None], dict[str, str], dict[str, dict]]:
    '''
    En entrée : papers_json_path, cache_path (fichier cache author_data_maps)

    En sortie : tuple (hindex_map, author_name_map, author_details_map) depuis cache ou construction

    Variables : cached, hindex_map, author_name_map, author_details_map, aid, d
    '''
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        hindex_map, author_name_map, author_details_map = cached.get("hindex_map") or {}, cached.get("author_name_map") or {}, cached.get("author_details_map") or {}
        author_details_map = {aid: ensure_author_details_shape(d) for aid, d in author_details_map.items() if isinstance(d, dict)}
        if cache_needs_paper_enrichment(author_details_map):
            print("Cache incomplet (paperIds/publicationVenues/fieldsOfStudy), enrichissement depuis papers_array.json...")
            enrich_author_details_from_papers(papers_json_path, author_details_map, author_name_map, hindex_map); _write_author_cache(cache_path, hindex_map, author_name_map, author_details_map)
        print(f"Cache charge: {cache_path}"); return hindex_map, author_name_map, author_details_map
    hindex_map, author_name_map, author_details_map = build_author_data_maps(papers_json_path); _write_author_cache(cache_path, hindex_map, author_name_map, author_details_map)
    print(f"Cache ecrit: {cache_path}"); return hindex_map, author_name_map, author_details_map


def visualize_coauthor_graph(graph_path: Path):
    '''
    En entrée : graph_path (fichier GraphML du graphe co-auteurs)

    En sortie : instance Sigma ipysigma ou None si fichier absent

    Variables : G (graphe chargé)
    '''
    from ipysigma import Sigma
    if not graph_path.exists(): print(f"Erreur : Le fichier {graph_path} n'existe pas."); return
    G = nx.read_graphml(graph_path); print(f"Visualisation de {G.number_of_nodes()} auteurs...")
    return Sigma(G, node_label="label", node_size="paper_count", node_color="label", edge_weight="weight", edge_size_range=(1, 10), label_font="sans-serif", node_border_color_from="node", default_edge_type="curve", clickable_edges=True)


def _author_id_from_node(node: str) -> str | None:
    '''
    En entrée : node (identifiant de nœud graphe, ex. « id:12345 »)

    En sortie : authorId extrait ou None si le nœud n'est pas de type « id:… »

    Variables : aucune variable locale significative
    '''
    return node.split("id:", 1)[1] if str(node).startswith("id:") else None


def _run_author_graph_analysis() -> None:
    '''
    En entrée : aucune (lit graphe et caches depuis GRAPHS_DIR et chemins par défaut)

    En sortie : aucune (affiche tops centralités, h-index et corrélations Pearson)

    Variables : G, papers_path, cache_path, hindex_map, author_name_map, author_details_map, degree_centrality, eigenvector_centrality, top_degree_nodes, top_eigen_nodes, top_hindex_items, top_author_ids, all_known_author_ids, resolve_author_name, hindex_by_node, aid, n, m, lbl
    '''
    G = nx.read_graphml(GRAPHS_DIR / "coauthor_graph.graphml")
    papers_path, cache_path = get_default_papers_path(), GRAPHS_DIR / "author_data_maps_cache.json"
    hindex_map, author_name_map, author_details_map = load_or_build_author_data_maps(papers_path, cache_path)
    degree_centrality, eigenvector_centrality = nx.degree_centrality(G), nx.eigenvector_centrality(G)
    top_degree_nodes = [n for n, _ in sorted(degree_centrality.items(), key=lambda kv: kv[1], reverse=True)[:10]]; top_eigen_nodes = [n for n, _ in sorted(eigenvector_centrality.items(), key=lambda kv: kv[1], reverse=True)[:10]]
    top_hindex_items = sorted({k: float(v) for k, v in hindex_map.items()}.items(), key=lambda kv: kv[1], reverse=True)[:10]
    top_author_ids = {_author_id_from_node(n) for n in top_degree_nodes + top_eigen_nodes if _author_id_from_node(n)} | {aid for aid, _ in top_hindex_items}
    all_known_author_ids = set(author_details_map) | set(hindex_map) | {_author_id_from_node(n) for n in G.nodes if _author_id_from_node(n)} | top_author_ids
    fill_missing_author_details_from_api(sorted(all_known_author_ids), author_name_map, author_details_map, hindex_map)
    if cache_needs_paper_enrichment(author_details_map): enrich_author_details_from_papers(papers_path, author_details_map, author_name_map, hindex_map)
    author_details_map = {aid: ensure_author_details_shape(d) for aid, d in author_details_map.items() if isinstance(d, dict)}; _write_author_cache(cache_path, hindex_map, author_name_map, author_details_map)
    resolve_author_name = lambda author_id, node_id: (m := str(author_name_map.get(author_id) or "").strip()) and m != author_id and m or (lbl := str(G.nodes.get(node_id, {}).get("label", "")).strip()) and lbl not in {node_id, author_id} and lbl or author_id
    hindex_by_node = {str(n): float(hindex_map[aid]) for n in G.nodes if (aid := _author_id_from_node(n)) and hindex_map.get(aid) is not None}
    print("Top 10 auteurs avec le plus haut degree centrality :", [G.nodes[n].get("label", n) for n in top_degree_nodes])
    print("Top 10 auteurs avec le plus haut eigenvector centrality :", [G.nodes[n].get("label", n) for n in top_eigen_nodes])
    print("Top 10 auteurs avec le plus haut hIndex :", [(resolve_author_name(aid, f"id:{aid}"), h) for aid, h in top_hindex_items])
    print(f"Nombre d'auteurs utilises pour la correlation : {len(hindex_by_node)}")
    print(f"Pearson(hIndex, degree_centrality) = {float(r[0] if (r := aligned_pearson(hindex_by_node, degree_centrality)) else 0):.4f}")
    print(f"Pearson(hIndex, eigenvector_centrality) = {float(r[0] if (r := aligned_pearson(hindex_by_node, eigenvector_centrality)) else 0):.4f}")
    print("Top degree:", [(G.nodes[n].get("label", n), degree_centrality[n]) for n in top_degree_nodes])
    print("Top eigen:", [(G.nodes[n].get("label", n), eigenvector_centrality[n]) for n in top_eigen_nodes])
    print("Top h-index:", top_hindex_items)


if __name__ == "__main__":
    _run_author_graph_analysis()


_DISRUPTION_CACHE_DIR = Path(__file__).resolve().parents[1] / "graphe_coauteurs" / "data" / "graphs"
S2_CIT_CACHE = _DISRUPTION_CACHE_DIR / "disruption_s2_cache.json"
OUTSIDE_REFS_CACHE = _DISRUPTION_CACHE_DIR / "disruption_outside_refs.json"


def load_s2_citation_cache() -> dict[str, Any]:

    if S2_CIT_CACHE.exists():
        raw = json.loads(S2_CIT_CACHE.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    return {}


def load_outside_refs_cache() -> dict[str, Any]:

    cache = load_s2_citation_cache()
    if OUTSIDE_REFS_CACHE.exists():
        raw = json.loads(OUTSIDE_REFS_CACHE.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            for pid, entry in raw.items():
                if pid not in cache and isinstance(entry, dict):
                    cache[pid] = {"valid": True, "ids": entry.get("citations") or []}
    return cache


def save_outside_refs_cache(cache: dict[str, Any]) -> None:

    OUTSIDE_REFS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out = {pid: {"citations": entry["ids"]} for pid, entry in cache.items() if entry.get("valid")}
    OUTSIDE_REFS_CACHE.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


def fetch_s2_citation_ids(session: requests.Session, paper_id: str, cache: dict[str, Any]) -> tuple[bool, set[str]]:

    """(valid, citeurs) ; valid=False si citations élidées ou len ≠ citationCount."""
    from graphe_common.io import has_elided_citations, safe_int

    paper_id = str(paper_id)
    if paper_id in cache:
        entry = cache[paper_id]
        return bool(entry.get("valid")), set(entry.get("ids") or [])

    response = session.get(f"{GRAPH_BASE}/paper/{paper_id}", params={"fields": "citationCount,openAccessPdf"}, timeout=60)
    if response.status_code == 404:
        cache[paper_id] = {"valid": False, "ids": []}
        return False, set()
    response.raise_for_status()
    meta = response.json()
    stub = {"citationCount": meta.get("citationCount"), "openAccessPdf": meta.get("openAccessPdf")}
    if has_elided_citations(stub):
        cache[paper_id] = {"valid": False, "ids": []}
        return False, set()

    ids: list[str] = []
    for offset in range(0, 10_000, 1000):
        batch = session.get(f"{GRAPH_BASE}/paper/{paper_id}/citations", params={"fields": "paperId", "limit": 1000, "offset": offset}, timeout=60).json().get("data") or []
        for item in batch:
            if isinstance(item, dict) and isinstance(item.get("citingPaper"), dict):
                if pid := item["citingPaper"].get("paperId"):
                    ids.append(str(pid))
        if len(batch) < 1000:
            break
        time.sleep(0.05)

    cc = safe_int(meta.get("citationCount"))
    valid = cc is not None and len(ids) == cc
    cache[paper_id] = {"valid": valid, "ids": ids}
    time.sleep(0.05)
    return valid, set(ids)
