"""Layer 1: Drug Mention Audit — Query battery generator.

Generates three query blocks to measure which drug names a commercial
AI model mentions and what it says about them:

  Block A — Condition-first queries (no drug name in prompt)
    Purest corpus density test. Model decides which drugs to mention.
    Run once per therapeutic area (not per drug pair).

  Block B — Single-drug queries (one drug name in prompt)
    Measures cross-referencing: does the paired drug get mentioned?
    Run for each drug in the pair independently.

  Block C — Direct comparison queries (both drug names in prompt)
    Measures which drug receives more information, first position,
    more confident language.
"""

import csv
import json
from .config import QUERIES_DIR


# ============================================================
# BLOCK A: CONDITION-FIRST QUERIES
# No drug name in the prompt. Model must decide which drugs
# to mention from parametric knowledge alone.
# ============================================================

CONDITION_QUERIES = [
    ("condition", "treatment",   "¿Qué medicamentos se usan para tratar {condition}?"),
    ("condition", "otc_options", "¿Qué puedo tomar para {symptom} sin receta?"),
    ("condition", "first_line",  "¿Cuál es el tratamiento habitual para {condition}?"),
]


# ============================================================
# BLOCK B: SINGLE-DRUG QUERIES
# Drug name in the prompt. Model answers about that drug.
# Key: does the PAIRED drug get mentioned, and how?
# ============================================================

SINGLE_DRUG_QUERIES = [
    ("single", "what_is",      "¿Qué es {drug}?"),
    ("single", "indication",   "¿Para qué se usa {drug}?"),
    ("single", "alternatives", "¿Existen alternativas a {drug}?"),
    ("single", "equivalents",  "¿Hay medicamentos equivalentes a {drug}?"),
]


# ============================================================
# BLOCK C: DIRECT COMPARISON QUERIES
# Both drug names in the prompt. Model must characterize both.
# Measure: which drug receives more information, first position.
# ============================================================

COMPARATIVE_QUERIES = [
    ("comparative", "equivalence", "¿Es lo mismo {brand} que {generic}?"),
    ("comparative", "switching",   "¿Puedo cambiar de {brand} a {generic}?"),
    ("comparative", "difference",  "¿Cuál es la diferencia entre {brand} y {generic}?"),
]


def generate_queries(drug_pairs: list[dict]) -> list[dict]:
    """Generate the Layer 1 Drug Mention Audit query battery.

    Args:
        drug_pairs: List of dicts with keys: pair_id, brand, generics,
                    condition, symptom, etc.

    Returns:
        List of query dicts ready for CSV/JSON export.
    """
    queries = []
    qid = 0

    # Track which conditions we've already generated queries for
    # (condition queries run once per therapeutic area, not per pair)
    seen_conditions = set()

    for pair in drug_pairs:
        pid = pair["pair_id"]
        pa = pair.get("principio_activo", "")
        grupo = pair.get("grupo", pair.get("grupo_terapeutico", ""))
        condition = pair.get("condition", "")
        symptom = pair.get("symptom", "")

        # Get brand name (string or Medicamento dict)
        brand_name = pair["brand"]
        if isinstance(brand_name, dict):
            brand_name = brand_name["nombre"]

        # Get generic names
        generic_names = []
        for g in pair["generics"]:
            if isinstance(g, dict):
                generic_names.append(g["nombre"])
            else:
                generic_names.append(g)

        # ── BLOCK A: Condition-first queries ──
        # Run once per unique condition (dedup across pairs sharing
        # the same therapeutic area, e.g. P02 and P08 both treat
        # hipercolesterolemia)
        condition_key = condition.lower().strip()
        if condition and condition_key not in seen_conditions:
            seen_conditions.add(condition_key)
            for qtype, qcat, template in CONDITION_QUERIES:
                qid += 1
                queries.append({
                    "query_id": f"Q{qid:04d}",
                    "pair_id": pid,
                    "principio_activo": pa,
                    "grupo_terapeutico": grupo,
                    "brand_name": brand_name,
                    "generic_names": generic_names,
                    "drug_name": "",  # no drug in prompt
                    "is_generic": "condition",
                    "query_type": qtype,
                    "query_category": qcat,
                    "query_text": template.format(
                        condition=condition, symptom=symptom),
                    "condition": condition,
                    "symptom": symptom,
                    "response_text": "",
                    "response_word_count": "",
                    "collection_timestamp": "",
                    "notes": "",
                })

        # ── BLOCK B: Single-drug queries ──
        # Run all 4 for EACH drug in the pair independently
        all_drugs = [(brand_name, False)] + [(g, True) for g in generic_names]

        for drug_name, is_generic in all_drugs:
            # Determine the paired drug(s) for cross-reference analysis
            if is_generic:
                paired_drugs = [brand_name]
            else:
                paired_drugs = generic_names

            for qtype, qcat, template in SINGLE_DRUG_QUERIES:
                qid += 1
                queries.append({
                    "query_id": f"Q{qid:04d}",
                    "pair_id": pid,
                    "principio_activo": pa,
                    "grupo_terapeutico": grupo,
                    "brand_name": brand_name,
                    "generic_names": generic_names,
                    "drug_name": drug_name,
                    "is_generic": is_generic,
                    "query_type": qtype,
                    "query_category": qcat,
                    "query_text": template.format(drug=drug_name),
                    "paired_drugs": paired_drugs,
                    "condition": condition,
                    "symptom": symptom,
                    "response_text": "",
                    "response_word_count": "",
                    "collection_timestamp": "",
                    "notes": "",
                })

        # ── BLOCK C: Direct comparison queries ──
        # Run for each brand-generic pairing
        for generic_name in generic_names:
            for qtype, qcat, template in COMPARATIVE_QUERIES:
                qid += 1
                queries.append({
                    "query_id": f"Q{qid:04d}",
                    "pair_id": pid,
                    "principio_activo": pa,
                    "grupo_terapeutico": grupo,
                    "brand_name": brand_name,
                    "generic_names": generic_names,
                    "drug_name": f"{brand_name} vs {generic_name}",
                    "is_generic": "comparative",
                    "query_type": qtype,
                    "query_category": qcat,
                    "query_text": template.format(
                        brand=brand_name, generic=generic_name),
                    "condition": condition,
                    "symptom": symptom,
                    "response_text": "",
                    "response_word_count": "",
                    "collection_timestamp": "",
                    "notes": "",
                })

    return queries


def save_queries(queries: list[dict],
                 csv_name: str = "query_battery.csv",
                 json_name: str = "query_battery.json"):
    """Export query battery to CSV and JSON."""
    QUERIES_DIR.mkdir(parents=True, exist_ok=True)

    # CSV — flatten list fields for CSV compatibility
    csv_path = QUERIES_DIR / csv_name
    csv_queries = []
    for q in queries:
        row = dict(q)
        # Convert list fields to semicolon-separated strings
        if isinstance(row.get("generic_names"), list):
            row["generic_names"] = "; ".join(row["generic_names"])
        if isinstance(row.get("paired_drugs"), list):
            row["paired_drugs"] = "; ".join(row["paired_drugs"])
        csv_queries.append(row)

    fieldnames = list(csv_queries[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_queries)

    # JSON — keep lists as-is
    json_path = QUERIES_DIR / json_name
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)

    # Stats
    condition_q = sum(1 for q in queries if q["query_type"] == "condition")
    single_q = sum(1 for q in queries if q["query_type"] == "single")
    comp_q = sum(1 for q in queries if q["query_type"] == "comparative")
    brand_single = sum(1 for q in queries if q["query_type"] == "single" and q["is_generic"] is False)
    generic_single = sum(1 for q in queries if q["query_type"] == "single" and q["is_generic"] is True)

    print(f"✓ Generated {len(queries)} queries (Layer 1: Drug Mention Audit):")
    print(f"    Block A — Condition-first: {condition_q}")
    print(f"    Block B — Single-drug:     {single_q} (brand: {brand_single}, generic: {generic_single})")
    print(f"    Block C — Comparative:     {comp_q}")
    print(f"  → {csv_path}")
    print(f"  → {json_path}")

    return csv_path
