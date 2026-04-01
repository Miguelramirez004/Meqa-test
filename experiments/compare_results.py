#!/usr/bin/env python3
"""Compare drug mention rates between original and loose prompt conditions.

Loads response JSONs from data/responses/ and data/responses_loose/,
applies the same BRAND_NAMES_BY_INN detection logic from src/response_analyzer.py,
and prints a side-by-side comparison table.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, RESPONSES_DIR
from src.response_analyzer import (
    BRAND_NAMES_BY_INN,
    GENERIC_LABS,
    _extract_inn,
    _extract_short_name,
    _get_brand_names,
    _find_word_positions,
)

RESPONSES_LOOSE_DIR = DATA_DIR / "responses_loose"


# ═══════════════════════════════════════════════════════════════════════════════
# DETECTION (mirrors src/response_analyzer.py logic)
# ═══════════════════════════════════════════════════════════════════════════════


def detect_mentions(response_text: str, query: dict) -> dict:
    """Detect brand, generic-product, and INN mentions using whole-word
    case-insensitive regex (same logic as response_analyzer)."""
    principio = query.get("principio_activo", "")
    inn = _extract_inn(principio)
    drug_name = query.get("drug_name", "")
    is_generic = query.get("is_generic", "")

    # Brand names for this INN
    brand_names = _get_brand_names(inn, pair_brand=drug_name)

    text_lower = response_text.lower()

    # Brand detection — whole-word case-insensitive
    brand_found = False
    brand_positions = []
    for b in brand_names:
        if len(b) < 3:
            continue
        pat = re.compile(r"\b" + re.escape(b) + r"\b", re.IGNORECASE)
        for m in pat.finditer(response_text):
            brand_found = True
            word_pos = len(response_text[:m.start()].split())
            brand_positions.append(word_pos)

    # Generic product detection (INN + lab)
    generic_found = False
    generic_positions = []
    for lab in GENERIC_LABS:
        pattern = f"{inn} {lab}"
        if len(pattern) < 5:
            continue
        pat = re.compile(r"\b" + re.escape(pattern) + r"\b", re.IGNORECASE)
        for m in pat.finditer(response_text):
            generic_found = True
            word_pos = len(response_text[:m.start()].split())
            generic_positions.append(word_pos)

    # INN detection (bare active ingredient)
    inn_found = False
    inn_positions = []
    if inn and len(inn) >= 3:
        pat = re.compile(r"\b" + re.escape(inn) + r"\b", re.IGNORECASE)
        for m in pat.finditer(response_text):
            inn_found = True
            word_pos = len(response_text[:m.start()].split())
            inn_positions.append(word_pos)

    # Retrieval rank: first drug in chunks
    retrieval_details = query.get("meqa_metadata", {}).get("retrieval_details", [])

    return {
        "brand_mentioned": brand_found,
        "generic_mentioned": generic_found,
        "inn_mentioned": inn_found,
        "both_mentioned": brand_found and generic_found,
        "neither_mentioned": not brand_found and not generic_found and not inn_found,
        "brand_first_pos": min(brand_positions) if brand_positions else None,
        "generic_first_pos": min(generic_positions) if generic_positions else None,
        "inn_first_pos": min(inn_positions) if inn_positions else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD RESPONSES
# ═══════════════════════════════════════════════════════════════════════════════


def load_responses(directory: Path) -> dict[str, dict]:
    """Load all response JSONs keyed by query_id."""
    responses = {}
    for fp in sorted(directory.glob("*.json")):
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
            qid = data.get("query_id", fp.stem)
            responses[qid] = data
        except (json.JSONDecodeError, KeyError):
            continue
    return responses


def retrieval_rank_agreement(response_data: dict, mentions: dict) -> bool | None:
    """Check if the first-mentioned drug type matches retrieval rank order.

    Returns True if the first brand/generic mention aligns with the first
    drug type in retrieval results, False if misaligned, None if can't determine.
    """
    details = response_data.get("meqa_metadata", {}).get("retrieval_details", [])
    if not details:
        return None

    # First drug in retrieval
    first_retrieved = details[0].get("drug_name", "")
    is_generic_chunk = any(
        lab in first_retrieved.lower() for lab in GENERIC_LABS
    ) or "efg" in first_retrieved.upper()

    # First mention type in response
    brand_pos = mentions.get("brand_first_pos")
    generic_pos = mentions.get("generic_first_pos")

    if brand_pos is not None and generic_pos is not None:
        response_first = "generic" if generic_pos < brand_pos else "brand"
    elif brand_pos is not None:
        response_first = "brand"
    elif generic_pos is not None:
        response_first = "generic"
    else:
        return None

    retrieval_first = "generic" if is_generic_chunk else "brand"
    return response_first == retrieval_first


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════


def compute_rates(responses: dict[str, dict]) -> dict:
    """Compute mention rates overall and by pair_id."""
    overall = defaultdict(int)
    by_pair: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total = 0
    pair_totals: dict[str, int] = defaultdict(int)
    agreement_total = 0
    agreement_valid = 0

    for qid, data in responses.items():
        text = data.get("response_text", "")
        if not text:
            continue
        m = detect_mentions(text, data)
        pair_id = data.get("pair_id", "unknown")
        total += 1
        pair_totals[pair_id] += 1

        if m["brand_mentioned"]:
            overall["brand"] += 1
            by_pair[pair_id]["brand"] += 1
        if m["generic_mentioned"]:
            overall["generic"] += 1
            by_pair[pair_id]["generic"] += 1
        if m["both_mentioned"]:
            overall["both"] += 1
            by_pair[pair_id]["both"] += 1
        if m["neither_mentioned"]:
            overall["neither"] += 1
            by_pair[pair_id]["neither"] += 1

        agree = retrieval_rank_agreement(data, m)
        if agree is not None:
            agreement_valid += 1
            if agree:
                agreement_total += 1

    return {
        "total": total,
        "overall": dict(overall),
        "by_pair": {k: dict(v) for k, v in by_pair.items()},
        "pair_totals": dict(pair_totals),
        "agreement_rate": agreement_total / agreement_valid if agreement_valid > 0 else None,
        "agreement_valid": agreement_valid,
    }


def print_comparison(orig_rates: dict, loose_rates: dict):
    """Print a clean side-by-side comparison table."""
    def pct(count, total):
        return f"{count}/{total} ({100*count/total:.1f}%)" if total > 0 else "N/A"

    ot = orig_rates["total"]
    lt = loose_rates["total"]
    oo = orig_rates["overall"]
    lo = loose_rates["overall"]

    print("\n" + "=" * 72)
    print("  PROMPT EXPERIMENT: ORIGINAL vs LOOSE — Drug Mention Comparison")
    print("=" * 72)

    # ── Shared query IDs ──
    print(f"\n  Responses analyzed:  Original={ot}   Loose={lt}")

    # ── Overall table ──
    print("\n  ┌─────────────────────────────────┬──────────────────┬──────────────────┐")
    print("  │ Metric                          │ Original         │ Loose            │")
    print("  ├─────────────────────────────────┼──────────────────┼──────────────────┤")

    rows = [
        ("Brand mention rate",   pct(oo.get("brand", 0), ot),   pct(lo.get("brand", 0), lt)),
        ("Generic mention rate", pct(oo.get("generic", 0), ot), pct(lo.get("generic", 0), lt)),
        ("Both mentioned rate",  pct(oo.get("both", 0), ot),    pct(lo.get("both", 0), lt)),
        ("Neither mentioned",    pct(oo.get("neither", 0), ot), pct(lo.get("neither", 0), lt)),
    ]
    for label, oval, lval in rows:
        print(f"  │ {label:<31s} │ {oval:<16s} │ {lval:<16s} │")

    # Agreement rate
    oa = orig_rates["agreement_rate"]
    la = loose_rates["agreement_rate"]
    oa_str = f"{oa:.1%}" if oa is not None else "N/A"
    la_str = f"{la:.1%}" if la is not None else "N/A"
    print(f"  │ {'Retrieval-mention agreement':<31s} │ {oa_str:<16s} │ {la_str:<16s} │")

    print("  └─────────────────────────────────┴──────────────────┴──────────────────┘")

    # ── Breakdown by pair_id ──
    all_pairs = sorted(
        set(list(orig_rates["by_pair"].keys()) + list(loose_rates["by_pair"].keys()))
    )

    print("\n  BREAKDOWN BY PAIR_ID")
    print("  ┌──────────────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐")
    print("  │ pair_id          │ O.brand │ O.genrc │ L.brand │ L.genrc │ O.neith │ L.neith │")
    print("  ├──────────────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤")

    for pid in all_pairs:
        opt = orig_rates["pair_totals"].get(pid, 0)
        lpt = loose_rates["pair_totals"].get(pid, 0)
        ob = orig_rates["by_pair"].get(pid, {}).get("brand", 0)
        og = orig_rates["by_pair"].get(pid, {}).get("generic", 0)
        lb = loose_rates["by_pair"].get(pid, {}).get("brand", 0)
        lg = loose_rates["by_pair"].get(pid, {}).get("generic", 0)
        on = orig_rates["by_pair"].get(pid, {}).get("neither", 0)
        ln = loose_rates["by_pair"].get(pid, {}).get("neither", 0)

        def cell(c, t):
            return f"{c}/{t}" if t > 0 else "-"

        print(
            f"  │ {pid:<16s} │ {cell(ob,opt):>7s} │ {cell(og,opt):>7s} "
            f"│ {cell(lb,lpt):>7s} │ {cell(lg,lpt):>7s} "
            f"│ {cell(on,opt):>7s} │ {cell(ln,lpt):>7s} │"
        )

    print("  └──────────────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘")
    print()


def main():
    print("Loading original responses from:", RESPONSES_DIR)
    orig = load_responses(RESPONSES_DIR)
    print(f"  Found {len(orig)} responses")

    print("Loading loose responses from:", RESPONSES_LOOSE_DIR)
    loose = load_responses(RESPONSES_LOOSE_DIR)
    print(f"  Found {len(loose)} responses")

    if not orig or not loose:
        print("ERROR: Need responses in both directories to compare.")
        sys.exit(1)

    # Restrict to shared query IDs
    shared = sorted(set(orig.keys()) & set(loose.keys()))
    print(f"  Shared query IDs: {len(shared)}")

    orig_shared = {k: orig[k] for k in shared}
    loose_shared = {k: loose[k] for k in shared}

    orig_rates = compute_rates(orig_shared)
    loose_rates = compute_rates(loose_shared)

    print_comparison(orig_rates, loose_rates)


if __name__ == "__main__":
    main()
