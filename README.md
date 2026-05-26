# MODAL — Analyse de graphes sur le corpus RAG

Projet d'analyse de réseaux scientifiques et de brevets autour d'un corpus **RAG** (Semantic Scholar + brevets Lens). Les modules construisent des graphes (co-auteurs, citations, co-inventeurs, similarité par domaines), calculent des centralités, des communautés Louvain, des scores de **disruption** (Wu et al.) et des tests inférentiels (corrélations, χ², t-tests, etc.).

## Prérequis

- **Python 3.11+** (testé avec 3.13)
- Fichiers de données **non versionnés** (voir `.gitignore`) :
  - `papers_array.json` — export Semantic Scholar (~12k papiers RAG)
  - `lens-export.csv` — brevets Lens (module brevets)
- Dépendances Python :

```bash
pip install networkx numpy scipy requests
```

Pour le graphe brevets–domaines (`graphe_brevets_domaines`) :

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

## Structure du dépôt

| Dossier | Rôle |
|---------|------|
| [`graphe_common/`](graphe_common/) | Bibliothèque partagée : I/O, construction de graphes, centralités, Louvain, tests inférentiels, disruption |
| [`graphe_coauteurs/`](graphe_coauteurs/) | Graphe de co-auteurs, évaluation Louvain vs champs d'étude |
| [`graphe_papiers_citations/`](graphe_papiers_citations/) | Graphe orienté papier → papier (citations) + disruption |
| [`graphe_auteurs_citations/`](graphe_auteurs_citations/) | Graphe orienté auteur → auteur (agrégation des citations) |
| [`graphe_brevets/`](graphe_brevets/) | Co-inventeurs, rapprochement NPL / auteurs–inventeurs |
| [`graphe_brevets_domaines/`](graphe_brevets_domaines/) | Similarité brevets par domaines extraits des abstracts (spaCy) |

Les rapports de synthèse numérique sont dans les fichiers `RAPPORT*.md` à la racine (corpus, citations, co-auteurs, disruption, brevets-domaines, etc.).

## Données et variables d'environnement

Par défaut, les modules lisent `papers_array.json` à la racine du dépôt. Pour pointer vers un autre fichier :

```bash
# PowerShell
$env:ML_GRAPH_INPUT = "chemin/vers/corpus.json"

# bash
export ML_GRAPH_INPUT=chemin/vers/corpus.json
```

Les caches API Semantic Scholar (h-index, citations auteurs) sont créés à la racine : `author_hindex_cache.json`, `author_citation_count_cache.json`.

## Lancer les analyses

Exécuter depuis la racine du dépôt (`MODAL/`) :

```bash
# Co-auteurs : Louvain + corrélations sur coauthor_graph.graphml
python -m graphe_coauteurs analyze

# Statistiques descriptives + tests inférentiels sur tout le corpus
python -m graphe_coauteurs corpus-stats

# Graphe citations papiers (arête A→B si A ∈ citations(B))
python -m graphe_papiers_citations

# Graphe citations auteurs (même logique, nœuds = auteurs)
python -m graphe_auteurs_citations

# Brevets : co-inventeurs, matching NPL et auteurs
python -m graphe_brevets

# Brevets : graphe de similarité par domaines d'abstract
python -m graphe_brevets_domaines
```

### Construction des arêtes de citation

Dans `graphe_common/builders.py`, le paramètre `edge_source` contrôle le sens des liens (convention **A → B** = « A cite B ») :

| `edge_source` | Définition | Filtre corpus |
|---------------|------------|---------------|
| `references` | B ∈ `references(A)` | bibliographies non élidées (~49 % du corpus) |
| `citations` | A ∈ `citations(B)` | listes de citations non masquées (100 %) |

Les points d'entrée `graphe_papiers_citations` et `graphe_auteurs_citations` utilisent actuellement `citations`. Pour une variante `references`, appeler `run_paper_citation_main` / `run_author_citation_main` depuis `graphe_common.cli` avec un `CitationGraphSpec` adapté.

## Tests unitaires

```bash
python -m unittest graphe_common.test_inferential -v
python -m unittest graphe_common.test_disruption -v
```

## Analyses complémentaires

[`_run_reviewer_checks.py`](_run_reviewer_checks.py) — contrôles relecture (cohorte temporelle, stabilité Louvain, FDR, etc.) ; écrit dans `graphe_coauteurs/data/stats/reviewer_checks/`. Ne modifie pas les graphes GraphML stockés.

## Sorties typiques

- Graphes : `graphe_coauteurs/data/graphs/`, `graphe_brevets_domaines/data/graphs/`
- Statistiques : `graphe_coauteurs/data/stats/`, CSV de corrélations et tests inférentiels
- Brevets-domaines : `graphe_brevets_domaines/data/stats/` (`domains.json`, `synonyms.json`, `summary.json`)

## Licence et contexte

Travaux réalisés dans le cadre du module **MODAL** (analyse de graphes appliquée au domaine RAG). Les PDF de cours sont dans `LECTURES/`.
