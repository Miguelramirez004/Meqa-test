#!/usr/bin/env python3
"""Statistical hypothesis testing for MeQA asymmetry experiment.

Tests H1–H4 using paired non-parametric tests on coded response metrics.

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


def test_h1(df: pd.DataFrame):
    """H1: Brand responses are more complete than generic responses."""
    print("\n" + "="*60)
    print("H1: Response Completeness (Brand vs Generic)")
    print("="*60)

    factual = df[df["query_type"] == "factual"].copy()
    brand = factual[factual["is_generic"].astype(str) == "False"]
    generic = factual[factual["is_generic"].astype(str) == "True"]

    # Aggregate by pair_id + category
    brand_agg = brand.groupby(["pair_id", "query_category"]).agg(
        word_count=("word_count", "mean"),
        completeness=("completeness_score", "mean"),
        ae_named=("adverse_effects_named", "mean"),
    ).reset_index()

    generic_agg = generic.groupby(["pair_id", "query_category"]).agg(
        word_count=("word_count", "mean"),
        completeness=("completeness_score", "mean"),
        ae_named=("adverse_effects_named", "mean"),
    ).reset_index()

    merged = brand_agg.merge(generic_agg, on=["pair_id", "query_category"],
                             suffixes=("_brand", "_generic"))

    for metric in ["word_count", "completeness", "ae_named"]:
        b = merged[f"{metric}_brand"].values
        g = merged[f"{metric}_generic"].values
        if len(b) >= 5:
            w, p = stats.wilcoxon(b, g)
            d = np.median(b - g)
            print(f"\n  {metric}:")
            print(f"    Brand mean: {np.mean(b):.2f}  Generic mean: {np.mean(g):.2f}")
            print(f"    Wilcoxon W={w:.4f}  p={p:.6f}  median diff={d:+.2f}")
            print(f"    {'*** SIGNIFICANT ***' if p < 0.05 else 'not significant'} (α=0.05)")
        else:
            print(f"\n  {metric}: insufficient pairs ({len(b)})")


def test_h2(df: pd.DataFrame):
    """H2: Differential accuracy — generates coding template for manual review."""
    print("\n" + "="*60)
    print("H2: Differential Accuracy (requires manual coding)")
    print("="*60)

    template = df[["query_id", "pair_id", "drug_name", "is_generic",
                    "query_category"]].copy()
    template["accuracy_score"] = ""  # 1=accurate, 2=partial, 3=omission, 4=error
    template["error_type"] = ""      # hallucination | omission | contradiction
    template["coder"] = ""
    template["notes"] = ""

    out = ANALYSIS_DIR / "accuracy_coding_template.csv"
    template.to_csv(out, index=False, encoding="utf-8")
    print(f"  → Template saved: {out}")
    print("  Fill in accuracy_score (1-4) and error_type, then re-run.")


def test_h3(df: pd.DataFrame):
    """H3: Framing bias in comparative queries."""
    print("\n" + "="*60)
    print("H3: Comparative Query Framing Bias")
    print("="*60)

    comp = df[df["query_type"] == "comparative"].copy()
    if comp.empty:
        print("  No comparative responses found.")
        return

    print(f"  Total comparative queries: {len(comp)}")
    print(f"  Mentions bioequivalence: {comp['mentions_bioequivalence'].sum()} "
          f"({comp['mentions_bioequivalence'].mean()*100:.0f}%)")
    print(f"  Refers to professional: {comp['mentions_professional'].sum()} "
          f"({comp['mentions_professional'].mean()*100:.0f}%)")
    print(f"  Avg word count: {comp['word_count'].mean():.0f}")

    for cat in comp["query_category"].unique():
        s = comp[comp["query_category"] == cat]
        print(f"\n    {cat}: n={len(s)}, "
              f"avg words={s['word_count'].mean():.0f}, "
              f"bioequiv={s['mentions_bioequivalence'].mean()*100:.0f}%")


def test_h4(df: pd.DataFrame):
    """H4: Asymmetry varies by therapeutic area."""
    print("\n" + "="*60)
    print("H4: Variation by Therapeutic Area")
    print("="*60)

    factual = df[df["query_type"] == "factual"].copy()
    pairs = factual["pair_id"].unique()
    rows = []

    for pid in pairs:
        p = factual[factual["pair_id"] == pid]
        b_wc = p[p["is_generic"].astype(str) == "False"]["word_count"].mean()
        g_wc = p[p["is_generic"].astype(str) == "True"]["word_count"].mean()
        if not (np.isnan(b_wc) or np.isnan(g_wc)):
            rows.append({"pair_id": pid, "asymmetry": b_wc - g_wc,
                         "brand_wc": b_wc, "generic_wc": g_wc})

    if not rows:
        print("  Insufficient data.")
        return

    asym = pd.DataFrame(rows)
    print(f"\n  Asymmetry (brand − generic word count):")
    print(f"    Mean:   {asym['asymmetry'].mean():+.2f}")
    print(f"    Median: {asym['asymmetry'].median():+.2f}")
    print(f"    Std:    {asym['asymmetry'].std():.2f}")
    print(f"    Range:  [{asym['asymmetry'].min():.0f}, {asym['asymmetry'].max():.0f}]")

    if len(asym) >= 5:
        w, p = stats.wilcoxon(asym["asymmetry"].values)
        print(f"\n    One-sample Wilcoxon (H₀: median = 0): W={w:.4f}, p={p:.6f}")
        print(f"    {'*** SIGNIFICANT ***' if p < 0.05 else 'not significant'}")

    out = ANALYSIS_DIR / "asymmetry_by_pair.csv"
    asym.to_csv(out, index=False)
    print(f"\n  → {out}")


def summary(df: pd.DataFrame):
    """Overall summary table."""
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for label, filt in [("Brand", "False"), ("Generic", "True"), ("Comparative", "comparative")]:
        sub = df[df["is_generic"].astype(str) == filt]
        if sub.empty:
            continue
        print(f"\n  {label} (n={len(sub)}):")
        for col in ["word_count", "completeness_score", "safety_warnings_count",
                     "adverse_effects_named"]:
            print(f"    {col}: mean={sub[col].mean():.2f} ± {sub[col].std():.2f}")

    out = ANALYSIS_DIR / "summary_report.csv"
    df.groupby("is_generic").agg({
        "word_count": ["mean", "std", "median"],
        "completeness_score": ["mean", "std"],
        "safety_warnings_count": ["mean", "std"],
        "adverse_effects_named": ["mean", "std"],
    }).to_csv(out)
    print(f"\n  → {out}")


def main():
    print("="*60)
    print("MeQA Asymmetry — Statistical Analysis")
    print("="*60)
    df = load_data()
    print(f"Loaded {len(df)} coded responses.")

    test_h1(df)
    test_h2(df)
    test_h3(df)
    test_h4(df)
    summary(df)

    print("\n✓ Analysis complete. Results in data/analysis/")


if __name__ == "__main__":
    main()
