#!/usr/bin/env python3
"""Collect PubMed literature data for all drug pairs.

Usage:
    python scripts/collect_pubmed.py
    python scripts/collect_pubmed.py --api-key YOUR_NCBI_KEY
    python scripts/collect_pubmed.py --top-n 10
"""

import argparse
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.drug_pairs import OFFLINE_PAIRS
from src.pubmed_client import collect_pubmed_data


def main():
    parser = argparse.ArgumentParser(
        description="Collect PubMed data for MeQA drug pairs")
    parser.add_argument("--api-key", default=None,
                        help="NCBI API key (optional, raises rate limit)")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of article summaries per search (default: 5)")
    args = parser.parse_args()

    results = collect_pubmed_data(
        pairs=OFFLINE_PAIRS,
        api_key=args.api_key,
        top_n=args.top_n,
    )

    # Print quick summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        brand_count = r["brand_search"]["total_count"] if r["brand_search"] else 0
        inn_count = r["inn_search"]["total_count"] if r["inn_search"] else 0
        bioeq_count = r["bioequivalence_search"]["total_count"] if r["bioequivalence_search"] else 0
        print(f"  {r['pair_id']:16s}  brand={brand_count:>6,}  "
              f"INN={inn_count:>6,}  bioeq={bioeq_count:>5,}")


if __name__ == "__main__":
    main()
