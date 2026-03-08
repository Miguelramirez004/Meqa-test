#!/usr/bin/env python3
"""Analyze all collected responses and generate mention-based metrics.

Usage: python scripts/analyze_responses.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.response_analyzer import analyze_all_responses, compute_asymmetry_scores


def main():
    print("=" * 60)
    print("Layer 1: Drug Mention Audit — Response Analysis")
    print("=" * 60)
    metrics = analyze_all_responses()
    if metrics:
        # Summary by query type
        condition = [m for m in metrics if m["query_type"] == "condition"]
        single = [m for m in metrics if m["query_type"] == "single"]
        comparative = [m for m in metrics if m["query_type"] == "comparative"]

        print(f"\nSummary:")
        print(f"  Condition-first responses: {len(condition)}")
        print(f"  Single-drug responses:     {len(single)}")
        print(f"  Comparative responses:     {len(comparative)}")

        # Cross-reference stats for single-drug queries
        if single:
            brand_queried = [m for m in single if m["is_generic"] is False]
            generic_queried = [m for m in single if m["is_generic"] is True]

            if brand_queried:
                cross_ref_brand = sum(1 for m in brand_queried if m["generic_mentioned"])
                print(f"\n  When brand queried → generic mentioned: "
                      f"{cross_ref_brand}/{len(brand_queried)} "
                      f"({cross_ref_brand/len(brand_queried)*100:.0f}%)")

            if generic_queried:
                cross_ref_generic = sum(1 for m in generic_queried if m["brand_mentioned"])
                print(f"  When generic queried → brand mentioned: "
                      f"{cross_ref_generic}/{len(generic_queried)} "
                      f"({cross_ref_generic/len(generic_queried)*100:.0f}%)")

        # Asymmetry ranking
        asymmetry = compute_asymmetry_scores(metrics)
        if asymmetry:
            print(f"\n  Asymmetry ranking (highest CAS first):")
            for a in asymmetry:
                print(f"    {a['pair_id']}: CAS={a['composite_asymmetry_score']:.4f} "
                      f"(mention={a['mention_asymmetry']:+.4f}, "
                      f"cross_ref={a['cross_ref_ratio']})")


if __name__ == "__main__":
    main()
