"""Indice de disruption : strict et allégé (in-corpus + total)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from graphe_common.builders import fetch_s2_citation_ids, get_session, load_outside_refs_cache, save_outside_refs_cache
from graphe_common.io import get_default_papers_path, load_papers, not_relevant_disruption, not_relevant_disruption_relaxed, reference_count_unusable, ref_in_corpus_not_relevant

ALLEGEE_VARIANTS = (("allegee", 0.03), ("allegee_v2", 0.005))
_LEGACY_NAMES = ("allegee_v3", "allegee_v4", "allegee_v5")


@dataclass(frozen=True)
class DisruptionRule:
    name: str
    tol: float | None = None
    min_citation_count: int = 50

    @property
    def is_strict(self) -> bool:
        return self.tol is None

    @property
    def relevant_flag(self) -> str:
        return "notrelevantdisruption" if self.is_strict else f"notrelevantdisruption_{self.name}"

    @property
    def total_bad_flag(self) -> str:
        return f"{self.relevant_flag}_total"

    @property
    def in_key(self) -> str:
        return "disruption_in_corpus" if self.is_strict else f"disruption_{self.name}_in_corpus"

    @property
    def total_key(self) -> str:
        return "disruption_total" if self.is_strict else f"disruption_{self.name}_total"

    @property
    def in_alias(self) -> str | None:
        return "disruption" if self.is_strict else None

    @property
    def fields(self) -> tuple[str, ...]:
        extra = (self.in_alias,) if self.in_alias else ()
        return (self.in_key, self.total_key, self.relevant_flag, self.total_bad_flag) + extra

    @property
    def tolerance_label(self) -> str:
        return "exact" if self.is_strict else f"{self.tol * 100:g}%"

    def not_relevant(self, paper: dict[str, Any]) -> bool:
        return not_relevant_disruption(paper) if self.is_strict else not_relevant_disruption_relaxed(paper, self.tol)

    def ref_is_bad(self, ref: dict[str, Any]) -> bool:
        return bool(ref.get(self.relevant_flag))


STRICT_RULE = DisruptionRule("strict", min_citation_count=10)
ALLEGEE_RULES = tuple(DisruptionRule(n, t) for n, t in ALLEGEE_VARIANTS)
ALL_RULES = (STRICT_RULE, *ALLEGEE_RULES)


def _all_fields() -> tuple[str, ...]:
    legacy = tuple(f for n in _LEGACY_NAMES for r in (DisruptionRule(n, 0.001),) for f in r.fields)
    return tuple(f for r in ALL_RULES for f in r.fields) + legacy


def _link_ids(paper: dict[str, Any], field: str) -> set[str]:
    return {str(raw) for item in (paper.get(field) or []) if isinstance(item, dict) and (raw := item.get("paperId"))}


def _disruption(citers: set[str], ref_citers: set[str]) -> float | None:
    nj = sum(c in ref_citers for c in citers)
    total = len(citers) + len(ref_citers - citers) - nj
    return None if not total else (len(citers) - 2 * nj) / total


def _ref_citers_in_corpus(refs: set[str], by_id: dict[str, dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for rid in refs:
        if (ref := by_id.get(rid)) and not reference_count_unusable(ref):
            out |= _link_ids(ref, "citations")
    return out


def disruption_in_corpus(focal: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> float | None:
    return _disruption(_link_ids(focal, "citations"), _ref_citers_in_corpus(_link_ids(focal, "references"), by_id))


def disruption_total(focal: dict[str, Any], by_id: dict[str, dict[str, Any]], outside: dict[str, Any], *, ref_bad: Callable[[dict[str, Any]], bool] | None = None) -> tuple[float | None, bool]:
    refs = _link_ids(focal, "references")
    ref_citers = _ref_citers_in_corpus(refs, by_id)
    for rid in refs:
        if ref := by_id.get(rid):
            if reference_count_unusable(ref) or (ref_bad and ref_bad(ref)):
                return None, True
            continue
        entry = outside.get(rid)
        if not entry or not entry.get("valid"):
            return None, True
        ref_citers |= set(entry.get("ids") or [])
    return _disruption(_link_ids(focal, "citations"), ref_citers), False


def _mark_relevant(papers: list[dict[str, Any]], by_id: dict[str, dict[str, Any]], rule: DisruptionRule) -> tuple[list[dict[str, Any]], dict[str, int]]:
    flag = rule.relevant_flag
    stats = {"own": 0, "refs": 0}
    for paper in papers:
        if not paper.get("paperId"):
            continue
        if rule.not_relevant(paper):
            paper[flag] = True
            stats["own"] += 1
    for paper in papers:
        if paper.get("paperId") and not paper.get(flag) and ref_in_corpus_not_relevant(paper, by_id, flag=flag):
            paper[flag] = True
            stats["refs"] += 1
    relevant = [p for p in papers if p.get("paperId") and not p.get(flag)]
    for paper in relevant:
        paper[flag] = False
    return relevant, stats


def _score_relevant(relevant: list[dict[str, Any]], by_id: dict[str, dict[str, Any]], outside: dict[str, Any], rule: DisruptionRule) -> int:
    n_total = 0
    ref_bad = rule.ref_is_bad
    for paper in relevant:
        paper[rule.in_key] = disruption_in_corpus(paper, by_id)
        if rule.in_alias:
            paper[rule.in_alias] = paper[rule.in_key]
        value, bad = disruption_total(paper, by_id, outside, ref_bad=ref_bad)
        paper[rule.total_bad_flag] = bad
        if bad or value is None:
            continue
        paper[rule.total_key] = value
        n_total += 1
    return n_total


def _outside_ref_ids(papers: list[dict[str, Any]], by_id: dict[str, dict[str, Any]], outside: dict[str, Any]) -> list[str]:
    return sorted({rid for p in papers for rid in _link_ids(p, "references") if rid not in by_id and rid not in outside})


def annotate_disruptions(papers: list[dict[str, Any]], *, fetch_total: bool = True) -> dict[str, int]:
    by_id = {str(p["paperId"]): p for p in papers if p.get("paperId")}
    for paper in papers:
        if paper.get("paperId"):
            for key in _all_fields():
                paper.pop(key, None)

    relevant, strict_st = _mark_relevant(papers, by_id, STRICT_RULE)
    allegee_runs = [(r, *_mark_relevant(papers, by_id, r)) for r in ALLEGEE_RULES]
    outside = load_outside_refs_cache()
    pool = relevant + [p for _, rel, _ in allegee_runs for p in rel]

    if fetch_total:
        for i, rid in enumerate(_outside_ref_ids(pool, by_id, outside), 1):
            try:
                fetch_s2_citation_ids(get_session(), rid, outside)
            except Exception:
                outside[rid] = {"valid": False, "ids": []}
            if i % 100 == 0:
                save_outside_refs_cache(outside)
        save_outside_refs_cache(outside)

    stats: dict[str, int | float] = {
        "relevant_in_corpus": len(relevant),
        "relevant_total": _score_relevant(relevant, by_id, outside, STRICT_RULE),
        "not_relevant_own": strict_st["own"],
        "not_relevant_refs": strict_st["refs"],
        "relevant": len(relevant),
        "not_relevant": strict_st["own"] + strict_st["refs"],
    }
    stats["not_relevant_total"] = len(relevant) - stats["relevant_total"]

    for rule, rel, st in allegee_runs:
        stats |= {
            f"relevant_{rule.name}_in_corpus": len(rel),
            f"relevant_{rule.name}_total": _score_relevant(rel, by_id, outside, rule),
            f"not_relevant_{rule.name}_own": st["own"],
            f"not_relevant_{rule.name}_refs": st["refs"],
        }

    for paper in papers:
        for rule in ALL_RULES:
            if paper.get(rule.relevant_flag) and paper.get(rule.total_bad_flag) is None:
                paper[rule.total_bad_flag] = True
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annoter disruption strict et allégé")
    parser.add_argument("--input", type=Path, default=get_default_papers_path())
    parser.add_argument("--no-api-total", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    corpus = load_papers(args.input, limit=args.limit)
    stats = annotate_disruptions(corpus, fetch_total=not args.no_api_total)
    parts = [f"strict in-corpus={stats['relevant_in_corpus']} total={stats['relevant_total']}"]
    for rule in ALLEGEE_RULES:
        n = rule.name
        parts.append(f"{n} in-corpus={stats[f'relevant_{n}_in_corpus']} total={stats[f'relevant_{n}_total']} refs={stats[f'not_relevant_{n}_refs']}")
    print(f"Papiers: {len(corpus)} | " + " | ".join(parts))
    args.input.write_text(json.dumps(corpus, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
