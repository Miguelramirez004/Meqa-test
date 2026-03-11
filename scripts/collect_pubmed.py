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
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  {'Pair ID':16s}  {'Brands':>8s}  {'Generic':>8s}  "
          f"{'INN':>8s}  {'BvG':>6s}  {'Bioeq':>6s}")
    print("-" * 80)
    for r in results:
        total_brand = r.get("total_brand_pubs", 0)
        total_generic = r.get("total_generic_pubs", 0)
        inn_count = r["inn_search"]["total_count"] if r.get("inn_search") else 0
        bvg_count = r["brand_vs_generic_search"]["total_count"] if r.get("brand_vs_generic_search") else 0
        bioeq_count = r["bioequivalence_search"]["total_count"] if r.get("bioequivalence_search") else 0
        print(f"  {r['pair_id']:16s}  {total_brand:>8,}  {total_generic:>8,}  "
              f"{inn_count:>8,}  {bvg_count:>6,}  {bioeq_count:>6,}")


if __name__ == "__main__":
    main()
