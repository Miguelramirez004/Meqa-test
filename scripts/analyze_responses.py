#!/usr/bin/env python3
"""Analyze all collected MeQA responses and generate coded metrics.

Usage: python scripts/analyze_responses.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.response_analyzer import analyze_all_responses


def main():
    print("="*60)
    print("MeQA Response Analyzer")
    print("="*60)
    metrics = analyze_all_responses()
    if metrics:
        print(f"\nSummary:")
        brand = [m for m in metrics if m["is_generic"] is False or m["is_generic"] == "False"]
        generic = [m for m in metrics if m["is_generic"] is True or m["is_generic"] == "True"]
        comp = [m for m in metrics if m["is_generic"] == "comparative"]
        print(f"  Brand responses:       {len(brand)}")
        print(f"  Generic responses:     {len(generic)}")
        print(f"  Comparative responses: {len(comp)}")
        if brand:
            avg_b = sum(m["word_count"] for m in brand) / len(brand)
            print(f"  Avg brand word count:  {avg_b:.0f}")
        if generic:
            avg_g = sum(m["word_count"] for m in generic) / len(generic)
            print(f"  Avg generic word count: {avg_g:.0f}")


if __name__ == "__main__":
    main()
