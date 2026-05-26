"""Point d'entrée : python -m graphe_coauteurs [analyze|corpus-stats]"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    '''
    En entrée : argv (sous-commande analyze ou corpus-stats, défaut analyze)

    En sortie : aucune (délègue à graphe_coauteurs.graph.main ou run_corpus_stats)

    Variables : parser, sub, args, cmd
    '''
    parser = argparse.ArgumentParser(description="Graphe de co-auteurs MODAL")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("analyze", help="Louvain + ground truth sur coauthor_graph.graphml (défaut)")
    sub.add_parser("corpus-stats", help="Statistiques descriptives du corpus + tests inférentiels")
    args, remaining = parser.parse_known_args(sys.argv[1:] or None)
    cmd = args.command or "analyze"
    sys.argv = [sys.argv[0]] + remaining
    if cmd == "corpus-stats":
        from graphe_coauteurs.graph import run_corpus_stats

        run_corpus_stats()
    else:
        from graphe_coauteurs.graph import main as analyze_main

        analyze_main()


if __name__ == "__main__":
    main()
