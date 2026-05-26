"""Tests unitaires : pertinence disruption (strict / allégé) et propagation."""

from __future__ import annotations

import unittest

from graphe_common.disruption import annotate_disruptions
from graphe_common.io import not_relevant_disruption_relaxed, reference_count_unusable, ref_in_corpus_not_relevant


def _paper(pid: str, *, refs: list | None = None, ref_count: int = 1, citations: list | None = None, cit_count: int = 1) -> dict:
    return {
        "paperId": pid,
        "referenceCount": ref_count,
        "references": refs or [{"paperId": "r1"}],
        "citationCount": cit_count,
        "citations": citations or [{"paperId": "c1"}],
    }


class TestDisruptionRelevance(unittest.TestCase):
    def test_ref_in_corpus_uses_flag(self):
        focal = _paper("f1")
        ref = {"paperId": "r1", "referenceCount": 1, "references": [{"paperId": "x"}], "notrelevantdisruption_allegee": True}
        by_id = {"r1": ref}
        self.assertTrue(ref_in_corpus_not_relevant(focal, by_id, flag="notrelevantdisruption_allegee"))
        self.assertFalse(ref_in_corpus_not_relevant(focal, by_id, flag="notrelevantdisruption"))

    def test_allegee_propagation_excludes_focal_with_bad_ref(self):
        bad_ref = _paper("r1", citations=[{"paperId": "x"}], cit_count=100)
        focal = _paper("f1", refs=[{"paperId": "r1"}])
        papers = [bad_ref, focal]
        stats = annotate_disruptions(papers, fetch_total=False)
        self.assertEqual(stats["relevant_allegee_in_corpus"], 0)
        self.assertTrue(focal["notrelevantdisruption_allegee"])
        self.assertGreater(stats["not_relevant_allegee_refs"], 0)

    def test_reference_count_zero_excluded(self):
        self.assertTrue(reference_count_unusable({"referenceCount": 0, "references": [], "citations": []}))
        self.assertTrue(reference_count_unusable({"referenceCount": None, "references": [{"paperId": "r1"}], "citations": [{"paperId": "c1"}], "citationCount": 1}))
        self.assertTrue(not_relevant_disruption_relaxed({"referenceCount": 0, "references": [], "citationCount": 0, "citations": []}, 0.03))

    def test_relaxed_own_ok_without_propagation_would_keep_focal(self):
        focal = _paper("f1", refs=[{"paperId": "r1"}], ref_count=1, citations=[{"paperId": "c1"}], cit_count=1)
        bad_ref = {"paperId": "r1", "referenceCount": 10, "references": [], "citationCount": 0, "citations": []}
        self.assertTrue(not_relevant_disruption_relaxed(bad_ref, 0.03))
        self.assertFalse(not_relevant_disruption_relaxed(focal, 0.03))


if __name__ == "__main__":
    unittest.main()
