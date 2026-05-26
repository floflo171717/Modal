"""Chargement CSV brevets (Lens) — local au module."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

PATENTS_PATH = "lens-export.csv"


def load_patents(path: str | Path | None = None) -> list[dict[str, Any]]:
    resolved = Path(path) if path is not None else Path(__file__).resolve().parents[1] / PATENTS_PATH
    return pd.read_csv(resolved, dtype=str, keep_default_na=False).to_dict("records")


def patent_id(row: dict[str, Any]) -> str:
    return str(row.get("Lens ID") or "")


def patent_abstract(row: dict[str, Any]) -> str:
    for key in ("Abstract", "abstract", "Abstract Text"):
        if text := str(row.get(key) or "").strip():
            return text
    return ""
