#!/usr/bin/env python3
"""Statistical hypothesis testing for Layer 1: Drug Mention Audit.

Tests asymmetry hypotheses using mention-based metrics.

Usage: python scripts/statistical_analysis.py
Prereq: python scripts/analyze_responses.py (generates response_metrics.csv)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

try:
    import pandas as pd
    from scipy import stats
except ImportError:
    print("Install dependencies: pip install pandas scipy")
    sys.exit(1)

from src.config import ANALYSIS_DIR

ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    metrics_file = ANALYSIS_DIR / "response_metrics.csv"
    if not metrics_file.exists():
        print(f"Not found: {metrics_file}")
        print("Run first: python scripts/analyze_responses.py")
        sys.exit(1)
    return pd.read_csv(metrics_file)


def load_asymmetry() -> pd.DataFrame | None:
    asym_file = ANALYSIS_DIR / "asymmetry_scores.csv"
    if asym_file.exists():
        return pd.read_csv(asym_file)
    return None


def test_mention_asymmetry(df: pd.DataFrame):
    """H1: Brand drugs are mentioned more often than generics across all query types."""
    print("\n" + "=" * 60)
    print("H1: Mention Asymmetry (Brand vs Generic visibility)")
    print("=" * 60)

    pairs = df["pair_id"].unique()
    rows = []
    for pid in pairs:
        p = df[df["pair_id"] == pid]
        rows.append({
            "pair_id": pid,
            "brand_mentions": p["brand_mention_count"].sum(),
            "generic_mentions": p["generic_mention_count"].sum(),
        })

    agg = pd.DataFrame(rows)
    agg["diff"] = agg["brand_mentions"] - agg["generic_mentions"]

    print(f"\n  Per-pair mention counts:")
    for _, r in agg.iterrows():
        print(f"    {r['pair_id']}: brand={r['brand_mentions']}, "
              f"generic={r['generic_mentions']}, diff={r['diff']:+d}")

    if len(agg) >= 5:
        w, p = stats.wilcoxon(agg["brand_mentions"].values, agg["generic_mentions"].values)
        print(f"\n  Wilcoxon signed-rank: W={w:.4f}, p={p:.6f}")
        print(f"  {'*** SIGNIFICANT ***' if p < 0.05 else 'not significant'} (alpha=0.05)")


def test_position_advantage(df: pd.DataFrame):
    """H2: Brand drugs appear earlier in responses than generics."""
    print("\n" + "=" * 60)
    print("H2: Position Advantage (Which drug appears first)")
    print("=" * 60)

    brand_first = (df["first_drug_mentioned"] == "brand").sum()
    generic_first = (df["first_drug_mentioned"] == "generic").sum()
    neither = (df["first_drug_mentioned"] == "neither").sum()
    total = brand_first + generic_first

    print(f"\n  Brand mentioned first:   {brand_first}")
    print(f"  Generic mentioned first: {generic_first}")
    print(f"  Neither mentioned:       {neither}")

    if total > 0:
        ratio = brand_first / total
        print(f"  Brand-first ratio: {ratio:.4f} (0.5 = no advantage)")

        result = stats.binomtest(brand_first, total, p=0.5, alternative="greater")
        print(f"  Binomial test (H0: ratio=0.5): p={result.pvalue:.6f}")
        print(f"  {'*** SIGNIFICANT ***' if result.pvalue < 0.05 else 'not significant'} (alpha=0.05)")


def test_cross_reference(df: pd.DataFrame):
    """H3: Cross-reference asymmetry in single-drug queries."""
    print("\n" + "=" * 60)
    print("H3: Cross-Reference Asymmetry")
    print("=" * 60)

    single = df[df["query_type"] == "single"]
    brand_queried = single[single["is_generic"].astype(str) == "False"]
    generic_queried = single[single["is_generic"].astype(str) == "True"]

    if brand_queried.empty and generic_queried.empty:
        print("  No single-drug queries found.")
        return

    p_brand_given_generic = (
        generic_queried["brand_mentioned"].sum() / len(generic_queried)
        if len(generic_queried) > 0 else 0
    )
    p_generic_given_brand = (
        brand_queried["generic_mentioned"].sum() / len(brand_queried)
        if len(brand_queried) > 0 else 0
    )

    print(f"\n  P(brand mentioned | generic queried): {p_brand_given_generic:.4f}")
    print(f"  P(generic mentioned | brand queried): {p_generic_given_brand:.4f}")

    if p_generic_given_brand > 0:
        ratio = p_brand_given_generic / p_generic_given_brand
        print(f"  Cross-reference ratio: {ratio:.4f}")
        print(f"  (>1 = asking about generic pulls in brand more than reverse)")

    a = generic_queried["brand_mentioned"].sum() if len(generic_queried) > 0 else 0
    b = len(generic_queried) - a
    c = brand_queried["generic_mentioned"].sum() if len(brand_queried) > 0 else 0
    d = len(brand_queried) - c

    if a + b > 0 and c + d > 0:
        odds_ratio, p = stats.fisher_exact([[a, b], [c, d]])
        print(f"\n  Fisher's exact test: OR={odds_ratio:.4f}, p={p:.6f}")
        print(f"  {'*** SIGNIFICANT ***' if p < 0.05 else 'not significant'} (alpha=0.05)")


def test_condition_visibility(df: pd.DataFrame):
    """H4: Condition-first queries — which drug type gets mentioned more."""
    print("\n" + "=" * 60)
    print("H4: Condition-First Query Visibility")
    print("=" * 60)

    condition = df[df["query_type"] == "condition"]
    if condition.empty:
        print("  No condition-first responses found.")
        return

    brand_mentioned = condition["brand_mentioned"].sum()
    generic_mentioned = condition["generic_mentioned"].sum()

    print(f"\n  Total condition-first responses: {len(condition)}")
    print(f"  Brand mentioned:   {brand_mentioned} ({brand_mentioned/len(condition)*100:.0f}%)")
    print(f"  Generic mentioned: {generic_mentioned} ({generic_mentioned/len(condition)*100:.0f}%)")

    if brand_mentioned + generic_mentioned > 0:
        ratio = brand_mentioned / (brand_mentioned + generic_mentioned)
        print(f"  Brand share: {ratio:.4f}")


def test_therapeutic_variation(df: pd.DataFrame):
    """H5: Asymmetry varies by therapeutic area."""
    print("\n" + "=" * 60)
    print("H5: Variation by Therapeutic Area")
    print("=" * 60)

    asym_df = load_asymmetry()
    if asym_df is None or asym_df.empty:
        print("  No asymmetry scores found. Run analyze_responses.py first.")
        return

    print(f"\n  Composite Asymmetry Score by pair:")
    for _, r in asym_df.iterrows():
        print(f"    {r['pair_id']}: CAS={r['composite_asymmetry_score']:.4f} "
              f"(mention={r['mention_asymmetry']:+.4f})")

    cas = asym_df["composite_asymmetry_score"].values
    print(f"\n  Mean CAS:   {np.mean(cas):.4f}")
    print(f"  Median CAS: {np.median(cas):.4f}")
    print(f"  Std CAS:    {np.std(cas):.4f}")

    if len(cas) >= 5:
        w, p = stats.wilcoxon(cas - 0.5)
        print(f"\n  Wilcoxon (H0: median CAS = 0.5): W={w:.4f}, p={p:.6f}")
        print(f"  {'*** SIGNIFICANT ***' if p < 0.05 else 'not significant'}")


def summary(df: pd.DataFrame):
    """Overall summary table."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for label, filt in [("Condition", "condition"), ("Single-drug", "single"), ("Comparative", "comparative")]:
        sub = df[df["query_type"] == filt]
        if sub.empty:
            continue
        print(f"\n  {label} (n={len(sub)}):")
        print(f"    Brand mentioned:   {sub['brand_mentioned'].sum()} ({sub['brand_mentioned'].mean()*100:.0f}%)")
        print(f"    Generic mentioned: {sub['generic_mentioned'].sum()} ({sub['generic_mentioned'].mean()*100:.0f}%)")
        print(f"    Avg brand mentions/response:   {sub['brand_mention_count'].mean():.2f}")
        print(f"    Avg generic mentions/response:  {sub['generic_mention_count'].mean():.2f}")
        print(f"    Avg word count: {sub['response_word_count'].mean():.0f}")

    out = ANALYSIS_DIR / "summary_report.csv"
    df.groupby("query_type").agg({
        "brand_mentioned": ["sum", "mean"],
        "generic_mentioned": ["sum", "mean"],
        "brand_mention_count": ["mean", "std"],
        "generic_mention_count": ["mean", "std"],
        "response_word_count": ["mean", "std", "median"],
    }).to_csv(out)
    print(f"\n  → {out}")


def main():
    print("=" * 60)
    print("Layer 1: Drug Mention Audit — Statistical Analysis")
    print("=" * 60)
    df = load_data()
    print(f"Loaded {len(df)} coded responses.")

    test_mention_asymmetry(df)
    test_position_advantage(df)
    test_cross_reference(df)
    test_condition_visibility(df)
    test_therapeutic_variation(df)
    summary(df)

    print("\n✓ Analysis complete. Results in data/analysis/")


if __name__ == "__main__":
    main()
