"""Dictionnaire de domaines depuis un échantillon d'abstracts (spaCy)."""
from __future__ import annotations

import random
import re
from typing import Any

import spacy
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Doc

from graphe_common.graphs import normalize_keyword
from .patents_io import patent_abstract, patent_id

from .config import (
    DOMAIN_PATTERNS,
    DOMAIN_SIGNALS,
    MANUAL_SYNONYMS,
    PATENT_NOISE,
    PATENT_NOISE_PREFIXES,
    PATENT_NOISE_TOKENS,
    N_SAMPLE,
    SEED,
    SPACY_MODEL,
)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip(" \t\n\r.,;:()[]{}\"'"))


def _looks_like_domain(norm: str) -> bool:
    return any(signal in norm for signal in DOMAIN_SIGNALS)


def _is_patent_noise(norm: str) -> bool:
    if not norm or norm in PATENT_NOISE:
        return True
    if any(norm.startswith(prefix) for prefix in PATENT_NOISE_PREFIXES):
        return True
    if any(phrase in norm for phrase in PATENT_NOISE if " " in phrase):
        return True
    if re.search(r"\b(one or more|at least one|a plurality of|plurality of)\b", norm):
        return True
    tokens = norm.split()
    if tokens and all(token in PATENT_NOISE_TOKENS for token in tokens):
        return True
    generic_tails = {"resources", "instances", "items", "objects", "processors", "users", "documents", "patents", "queries", "events", "constraints", "attributes", "definitions", "modules", "devices", "components", "elements", "steps", "units"}
    if len(tokens) >= 2 and tokens[-1] in generic_tails and tokens[0] in PATENT_NOISE_TOKENS | {"one", "more", "least", "available", "alternative", "computing", "common", "continuous", "automatically", "generated", "virtual", "reduced", "definable"}:
        return True
    if len(tokens) >= 2 and tokens[-1] in {"nodes", "capabilities", "capability", "instances", "modules", "processors", "resources"}:
        return True
    return False


def _relevant(text: str, *, from_noun_chunk: bool = False, seed_norms: set[str] | None = None) -> bool:
    cleaned, norm = _clean(text), normalize_keyword(_clean(text))
    if seed_norms and norm in seed_norms:
        return True
    if not cleaned or len(cleaned) < 3 or _is_patent_noise(norm):
        return False
    tokens = norm.split()
    if len(tokens) == 1 and (len(norm) < 4 or norm in PATENT_NOISE_TOKENS):
        return False
    if from_noun_chunk and not _looks_like_domain(norm):
        return False
    if not from_noun_chunk and len(tokens) == 1 and not _looks_like_domain(norm):
        return False
    return True


def seed_index(papers: list[dict[str, Any]]) -> dict[str, str]:
    index: dict[str, str] = {}
    for paper in papers:
        for item in paper.get("fieldsOfStudy") or []:
            if value := str(item).strip():
                index.setdefault(normalize_keyword(value), value)
        for item in paper.get("s2FieldsOfStudy") or []:
            if isinstance(item, dict) and (cat := item.get("category")) and str(cat).strip():
                value = str(cat).strip()
                index.setdefault(normalize_keyword(value), value)
    for canonical, variants in MANUAL_SYNONYMS.items():
        display = index.get(normalize_keyword(canonical), canonical.title())
        index[normalize_keyword(canonical)] = display
        for variant in variants:
            index[normalize_keyword(variant)] = display
    return index


def _nlp(index: dict[str, str]) -> Language:
    nlp = spacy.load(SPACY_MODEL)
    ruler = nlp.add_pipe("entity_ruler", before="ner") if "entity_ruler" not in nlp.pipe_names else nlp.get_pipe("entity_ruler")
    seen: set[str] = set()
    patterns: list[dict[str, Any]] = []
    for norm, display in index.items():
        if norm in seen:
            continue
        seen.add(norm)
        patterns.append({"label": "DOMAIN", "pattern": [{"LOWER": t} for t in display.lower().split()]})
    for canonical, variants in MANUAL_SYNONYMS.items():
        for term in [canonical, *variants]:
            norm = normalize_keyword(term)
            if norm in seen:
                continue
            seen.add(norm)
            tokens = [t for t in re.split(r"[\s\-]+", term.lower()) if t]
            if tokens:
                patterns.append({"label": "DOMAIN", "pattern": [{"LOWER": t} for t in tokens]})
    ruler.add_patterns(patterns)
    return nlp


def _matcher(nlp: Language) -> Matcher:
    matcher = Matcher(nlp.vocab)
    for name, pattern, _ in DOMAIN_PATTERNS:
        matcher.add(name, [pattern], greedy="LONGEST")
    return matcher


def _extract(doc: Doc, matcher: Matcher, *, seed_norms: set[str]) -> list[str]:
    hits, covered = [], []
    for ent in doc.ents:
        if ent.label_ == "DOMAIN" and _relevant(ent.text, seed_norms=seed_norms):
            hits.append(_clean(ent.text))
            covered.append((ent.start, ent.end))
    prefix = {name: off for name, _, off in DOMAIN_PATTERNS}
    for match_id, start, end in matcher(doc):
        name = doc.vocab.strings[match_id]
        term = _clean(doc[start + prefix[name] : end].text)
        if _relevant(term, seed_norms=seed_norms):
            hits.append(term)
            covered.append((start + prefix[name], end))
    overlap = lambda s, e: any(not (e <= a or s >= b) for a, b in covered)
    for chunk in doc.noun_chunks:
        if overlap(chunk.start, chunk.end):
            continue
        term, root = _clean(chunk.text), normalize_keyword(chunk.root.lemma_)
        if chunk.root.pos_ not in {"NOUN", "PROPN"}:
            continue
        if root in PATENT_NOISE_TOKENS or root in PATENT_NOISE:
            continue
        if not _relevant(term, from_noun_chunk=True, seed_norms=seed_norms):
            continue
        if len(term.split()) >= 2 or (len(term) >= 8 and chunk.root.pos_ == "PROPN"):
            hits.append(term)
    return hits


def build_seed_dictionary(papers: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, Any]]:
    """Dictionnaire fige : 41 canoniques du seed (papiers + MANUAL_SYNONYMS), sans spaCy."""
    index = seed_index(papers)
    by_canon: dict[str, set[str]] = {}
    for norm, display in index.items():
        by_canon.setdefault(display, set()).add(norm)
        by_canon[display].add(normalize_keyword(display))
    synonym_map = dict(sorted(index.items(), key=lambda kv: (kv[1].lower(), kv[0])))
    domains = {
        canon: {"canonical": canon, "synonyms": sorted(syns, key=str.lower), "abstract_count": 0}
        for canon, syns in sorted(by_canon.items(), key=lambda kv: kv[0].lower())
    }
    return synonym_map, domains


def build_dictionary(patents: list[dict[str, Any]], papers: list[dict[str, Any]], *, n: int = N_SAMPLE, seed: int = SEED, seed_only: bool = False) -> tuple[dict[str, str], dict[str, Any]]:
    if seed_only:
        return build_seed_dictionary(papers)
    index = seed_index(papers)
    seed_norms = set(index)
    nlp = _nlp(index)
    matcher = _matcher(nlp)
    entries: dict[str, dict[str, Any]] = {}
    synonyms = dict(index)

    def canonical(surface: str) -> str:
        norm = normalize_keyword(_clean(surface))
        if norm in synonyms:
            return synonyms[norm]
        for seed_norm, display in synonyms.items():
            if " " in seed_norm and (norm in seed_norm.split() or seed_norm.startswith(norm + " ")):
                return display
        return _clean(surface)

    for row in random.Random(seed).sample(patents, min(n, len(patents))):
        pid = patent_id(row)
        abstract = patent_abstract(row)
        if not pid or not abstract:
            continue
        for term in _extract(nlp(abstract), matcher, seed_norms=seed_norms):
            if not _relevant(term, seed_norms=seed_norms):
                continue
            cleaned, canon = _clean(term), canonical(term)
            synonyms[normalize_keyword(cleaned)] = canon
            synonyms[normalize_keyword(canon)] = canon
            entry = entries.setdefault(canon, {"canonical": canon, "synonyms": set(), "abstract_ids": set()})
            entry["synonyms"].add(cleaned)
            entry["abstract_ids"].add(pid)

    synonym_map = dict(synonyms)
    for canon, entry in entries.items():
        synonym_map[normalize_keyword(canon)] = canon
        for syn in entry["synonyms"]:
            synonym_map[normalize_keyword(syn)] = canon
    domains = {
        c: {"canonical": e["canonical"], "synonyms": sorted(e["synonyms"], key=str.lower), "abstract_count": len(e["abstract_ids"])}
        for c, e in sorted(entries.items(), key=lambda kv: (-len(kv[1]["abstract_ids"]), kv[0].lower()))
    }
    return dict(sorted(synonym_map.items(), key=lambda kv: (kv[1].lower(), kv[0]))), domains


def match_domains(text: str, synonym_map: dict[str, str]) -> set[str]:
    text = text.lower()
    found, used = set(), []
    for syn, canon in sorted(synonym_map.items(), key=lambda kv: (-len(kv[0]), kv[0])):
        if not syn or (len(syn) < 3 and " " not in syn):
            continue
        for m in re.finditer(rf"\b{re.escape(syn)}\b", text):
            span = m.span()
            if any(not (span[1] <= s or span[0] >= e) for s, e in used):
                continue
            used.append(span)
            found.add(canon)
            break
    return found
