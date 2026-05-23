"""
utils.py
========
CLI helpers: banner, result table, logging configuration.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    logging.basicConfig(stream=sys.stderr, level=level, format=fmt)


def print_banner() -> None:
    banner = r"""
╔══════════════════════════════════════════════════════════╗
║         Local Document AI — Classification & Search      ║
║         Open-source only • Runs 100% offline             ║
╚══════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_results_table(results: Dict[str, Any]) -> None:
    """Pretty-print a summary table of classification results."""
    if not results:
        print("No results to display.")
        return

    col_w = [40, 20, 8]
    header = (
        f"{'Filename':<{col_w[0]}}  {'Class':<{col_w[1]}}  {'Fields found':>{col_w[2]}}"
    )
    sep = "─" * (sum(col_w) + 6)

    print(f"\n{sep}")
    print(header)
    print(sep)

    for filename, record in results.items():
        doc_class = record.get("class", "?")
        n_fields = sum(
            1 for k, v in record.items()
            if k not in ("class", "_error", "_confidence") and v is not None
        )
        print(
            f"{filename[:col_w[0]]:<{col_w[0]}}  "
            f"{doc_class:<{col_w[1]}}  "
            f"{n_fields:>{col_w[2]}}"
        )

    print(sep)
    print(f"  Total: {len(results)} document(s)\n")
