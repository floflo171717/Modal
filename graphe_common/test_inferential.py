"""Tests unitaires : inferential (core + papiers + auteurs + domaines)."""
from __future__ import annotations

import unittest

import numpy as np
from scipy.stats import fisher_exact, ttest_1samp, ttest_ind

import networkx as nx

from graphe_common.inferential import collect_paper_metrics, contingency_table, independence_tests, inferential_row, pearson_test, run_anova, run_inferential_stats, run_large_team_centrality_test, run_npl_citation_inferential, run_paper_domain_inferential, run_team_disruption_test, shapiro_test, welch_test


def _paper(pid: str, citations: int, n_authors: int = 2, fields: list[str] | None = None, category: str | None = None) -> dict:
    '''
    En entrée : identifiant, nombre de citations, nombre d'auteurs ; champs ou catégorie optionnels.

    En sortie : dictionnaire papier minimal pour les tests unitaires inférentiels.

    Variables : fos (liste s2FieldsOfStudy construite à partir de fields ou category).
    '''
    fos = [{"category": f} for f in fields] if fields else ([{"category": category}] if category else [])
    return {"paperId": pid, "citationCount": citations, "authors": [{"authorId": str(i)} for i in range(n_authors)], "s2FieldsOfStudy": fos, "fieldsOfStudy": fields or []}


class TestInferentialCore(unittest.TestCase):
    def test_contingency_table(self):
        table, rows, cols = contingency_table(["a", "a", "b"], ["x", "y", "x"])
        self.assertEqual((table, rows, cols), ([[1, 1], [1, 0]], ["a", "b"], ["x", "y"]))

    def test_independence_fisher_and_chi2(self):
        keys_a, keys_b = ["h", "h", "l", "l", "h", "l"] * 5, ["i", "i", "i", "n", "n", "n"] * 5
        rows = independence_tests(keys_a, keys_b, label="test")
        self.assertEqual({r["test"] for r in rows}, {"fisher_exact", "chi2_independence"})
        self.assertAlmostEqual(rows[0]["p_value"], fisher_exact(contingency_table(keys_a, keys_b)[0])[1])

    def test_pearson_welch_shapiro(self):
        xs, ys = list(range(20)), [2 * x for x in range(20)]
        self.assertLess(pearson_test(xs, ys, label="lin")["p_value"], 0.05)
        a, b = np.random.default_rng(0).normal(0, 1, 40), np.random.default_rng(1).normal(3, 1.5, 40)
        row = welch_test(a.tolist(), b.tolist(), label="g")
        t, p = ttest_ind(a, b, equal_var=False)
        self.assertAlmostEqual(row["statistic"], t)
        self.assertAlmostEqual(row["p_value"], p)
        self.assertEqual(shapiro_test([1.0, 2.0, 3.0, 4.0, 5.0], label="x")["test"], "shapiro")

    def test_anova_top2_welch(self):
        groups = {"A": [1.0] * 12, "B": [5.0] * 12, "C": [0.5] * 4}
        stat, p, k, names = run_anova(groups, min_n=10)
        self.assertEqual((k, names[0]), (2, "A"))
        top2 = sorted([(d, v) for d, v in groups.items() if len(v) >= 10], key=lambda kv: -np.mean(kv[1]))[:2]
        row = welch_test([float(x) for x in top2[0][1]], [float(x) for x in top2[1][1]], label="dom")
        self.assertEqual(row["test"], "ttest_ind_welch")


class TestPaperInferential(unittest.TestCase):
    def setUp(self):
        base = [(100, 1, "biology"), (80, 2, "biology"), (60, 3, "physics"), (40, 6, "physics"), (20, 8, "chemistry"), (10, 1, "chemistry"), (5, 2, None), (2, 7, None)]
        self.papers = [_paper(f"p{i}", c, a, category=cat) for i, (c, a, cat) in enumerate(base)]
        self.papers += [_paper(f"e{i}", 30 + i, 2 + (i % 6), category="biology" if i % 2 else "physics") for i in range(24)]

    def test_pipeline_tp4(self):
        rows = run_inferential_stats(self.papers, alpha=0.05)
        tests = {r["test"] for r in rows}
        self.assertTrue({"pearson", "shapiro", "fisher_exact", "chi2_independence", "ttest_ind_welch", "ttest_1samp"} <= tests)
        c, f, a = collect_paper_metrics(self.papers)
        self.assertEqual(len(c), 32)
        t, p = ttest_1samp(c, popmean=float(np.median(c)))
        self.assertEqual(next(r for r in rows if r["test"] == "ttest_1samp")["p_value"], float(p))

    def test_team_disruption(self):
        disruption = {f"p{i}": float(10 - i) for i in range(1, 9)}
        tests = {r["test"] for r in run_team_disruption_test(self.papers, disruption, small_team_max=2, large_team_min=5)}
        self.assertEqual(tests, {"pearson", "ttest_ind_welch"})


class TestDomainInferential(unittest.TestCase):
    def test_domain_anova_pearson_welch(self):
        papers = [_paper("a1", 50, 2, fields=["Biology"]), _paper("a2", 45, 2, fields=["Biology"]), _paper("a3", 40, 2, fields=["Biology"]), _paper("b1", 5, 2, fields=["Physics"]), _paper("b2", 4, 2, fields=["Physics"]), _paper("b3", 3, 2, fields=["Physics"])] + [_paper(f"x{i}", i, 2, fields=["Mathematics"]) for i in range(12)]
        tests = {r["test"] for r in run_paper_domain_inferential(papers, min_group_size=3, alpha=0.05)["rows"]}
        self.assertTrue({"f_oneway", "pearson", "ttest_ind_welch"} <= tests)


@unittest.skipIf(nx is None, "networkx non installé")
class TestAuthorHelpers(unittest.TestCase):
    def test_large_team_centrality(self):
        G = nx.relabel_nodes(nx.karate_club_graph(), {n: f"id:{n}" for n in nx.karate_club_graph().nodes})
        tests = {r["test"] for r in run_large_team_centrality_test(G, {f"id:{n}": float(1 + (n % 6)) for n in range(34)}, {str(n): float(n % 20) for n in range(34)}, small_team_max=2.0, large_team_min=4.0)}
        self.assertEqual(tests, {"pearson", "ttest_ind_welch"})


class TestNplCitationInferential(unittest.TestCase):
    def test_npl_citation_tests(self):
        titles = [f"A Long Enough Paper Title Number {i} for Testing" for i in range(20)]
        papers = [{**_paper(f"p{i}", 100 - i * 5, 3, category="biology"), "title": titles[i]} for i in range(20)]
        patents = [{"Lens ID": "L1", "Title": "Patent A", "NPL Resolved Citation Count": "2", "NPL Citations": f'See "{titles[0]}" and "{titles[1]}"', "Inventors": ""}, {"Lens ID": "L2", "Title": "Patent B", "NPL Resolved Citation Count": "1", "NPL Citations": f'Ref: "{titles[2]}"', "Inventors": ""}, {"Lens ID": "L3", "Title": "Patent C", "NPL Resolved Citation Count": "0", "NPL Citations": "", "Inventors": ""}]
        from pathlib import Path
        rows = run_npl_citation_inferential(papers, patents=patents, coauthor_graph_path=Path("/nonexistent"), seed=42)
        self.assertEqual(next(r for r in rows if r["test"] == "summary")["statistic"], 3)
        welch = [r for r in rows if r["test"] == "ttest_ind_welch"]
        self.assertEqual(len(welch), 3)
        self.assertEqual(next(r for r in welch if r["variable"].startswith("citationCount"))["n"], 6)

    def test_npl_no_match(self):
        from pathlib import Path
        papers = [_paper(f"p{i}", 10, 2) for i in range(5)]
        patents = [{"Lens ID": "X", "Title": "T", "NPL Resolved Citation Count": "0", "NPL Citations": "", "Inventors": ""}]
        rows = run_npl_citation_inferential(papers, patents=patents, coauthor_graph_path=Path("/nonexistent"))
        self.assertEqual(next(r for r in rows if r["test"] == "summary")["statistic"], 0)
        self.assertFalse(any(r["test"] == "ttest_ind_welch" for r in rows))


class TestRowHelper(unittest.TestCase):
    def test_inferential_row(self):
        self.assertTrue(inferential_row("pearson", "x", 0.9, 0.01, 10, alpha=0.05)["significant"])
        self.assertFalse(inferential_row("pearson", "x", 0.1, float("nan"), 2, alpha=0.05)["significant"])


if __name__ == "__main__":
    unittest.main()
