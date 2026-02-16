#!/usr/bin/env python3
"""Generate the full query battery CSV from pre-defined drug pairs.

Usage: python scripts/generate_queries.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.drug_pairs import get_offline_pairs
from src.query_generator import generate_queries, save_queries


def main():
    pairs = get_offline_pairs()
    queries = generate_queries(pairs)
    save_queries(queries)


if __name__ == "__main__":
    main()
