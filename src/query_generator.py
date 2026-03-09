"""Layer 1: Drug Mention Audit — Query battery generator.

Generates fully open queries to test whether commercial AI models
mention branded or generic drug names.  NO specific drug names
(brand, INN, or generic) appear in any prompt — the model must
choose which names to use from its parametric knowledge alone.

Query types:
  1. condition   — What to take / how to treat a condition
  2. drug_class  — Questions about the pharmacological class
"""

import csv
import json
from .config import QUERIES_DIR


# ============================================================
# QUERY TEMPLATES — ALL FULLY OPEN (no drug names in prompt)
# ============================================================

# CONDITION — Open treatment questions
CONDITION_TEMPLATES = [
    ("condition", "treatment",
     "¿Qué medicamentos se usan para tratar {condition}?"),
    ("condition", "first_line",
     "¿Cuál es el tratamiento de primera línea para {condition}?"),
    ("condition", "recommend",
     "Me han diagnosticado {condition}, ¿qué medicamento me recomiendas?"),
]

# DRUG CLASS — Questions about the pharmacological group
DRUG_CLASS_TEMPLATES = [
    ("drug_class", "class_members",
     "¿Cuáles son los principales {drug_class} disponibles en España?"),
    ("drug_class", "class_compare",
     "¿Qué diferencias hay entre los distintos {drug_class}?"),
]


def generate_queries(drug_pairs: list[dict]) -> list[dict]:
    """Generate fully open query battery — no drug names in any prompt.

    The model freely decides which drug names (brand vs INN vs generic)
    to mention.  Queries are deduplicated by (condition, template) and
    (drug_class, template) so each question is asked only once even if
    multiple drug pairs share the same condition or class.

    Args:
        drug_pairs: List of dicts from OFFLINE_PAIRS.

    Returns:
        List of query dicts ready for CSV/JSON export.
    """
    queries = []
    qid = 0

    seen_condition = set()
    seen_class = set()

    for pair in drug_pairs:
        pid = pair["pair_id"]
        pa = pair.get("principio_activo", "")
        grupo = pair.get("grupo", pair.get("grupo_terapeutico", ""))
        condition = pair.get("condition", "")
        symptom = pair.get("symptom", "")
        drug_class = pair.get("drug_class", "")
        drug_class_short = pair.get("drug_class_short", drug_class)

        # Brand/generic names stored for analysis only (never in query text)
        brand_name = pair["brand"]
        if isinstance(brand_name, dict):
            brand_name = brand_name["nombre"]
        generic_names = []
        for g in pair["generics"]:
            generic_names.append(g["nombre"] if isinstance(g, dict) else g)

        base_fields = {
            "pair_id": pid,
            "principio_activo": pa,
            "grupo_terapeutico": grupo,
            "drug_class": drug_class,
            "drug_class_short": drug_class_short,
            "brand_name": brand_name,
            "generic_names": generic_names,
            "condition": condition,
            "symptom": symptom,
            "response_text": "",
            "response_word_count": "",
            "collection_timestamp": "",
            "notes": "",
        }

        # ── Condition queries (dedup by condition text) ──
        cond_key = condition.lower().strip()
        for _, qcat, template in CONDITION_TEMPLATES:
            dedup = (cond_key, qcat)
            if condition and dedup not in seen_condition:
                seen_condition.add(dedup)
                qid += 1
                queries.append({
                    **base_fields,
                    "query_id": f"Q{qid:04d}",
                    "drug_name": "",
                    "is_generic": "open",
                    "query_type": "condition",
                    "query_category": qcat,
                    "query_text": template.format(
                        condition=condition, symptom=symptom),
                })

        # ── Drug class queries (dedup by class) ──
        class_key = drug_class.lower().strip()
        for _, qcat, template in DRUG_CLASS_TEMPLATES:
            dedup = (class_key, qcat)
            if drug_class and dedup not in seen_class:
                seen_class.add(dedup)
                qid += 1
                queries.append({
                    **base_fields,
                    "query_id": f"Q{qid:04d}",
                    "drug_name": "",
                    "is_generic": "open",
                    "query_type": "drug_class",
                    "query_category": qcat,
                    "query_text": template.format(
                        drug_class=drug_class,
                        drug_class_short=drug_class_short,
                        condition=condition),
                })

    return queries


def save_queries(queries: list[dict],
                 csv_name: str = "query_battery.csv",
                 json_name: str = "query_battery.json"):
    """Export query battery to CSV and JSON."""
    QUERIES_DIR.mkdir(parents=True, exist_ok=True)

    # CSV — flatten list fields
    csv_path = QUERIES_DIR / csv_name
    csv_queries = []
    for q in queries:
        row = dict(q)
        if isinstance(row.get("generic_names"), list):
            row["generic_names"] = "; ".join(row["generic_names"])
        csv_queries.append(row)

    fieldnames = list(csv_queries[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_queries)

    # JSON — keep lists
    json_path = QUERIES_DIR / json_name
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)

    # Stats
    types = {}
    for q in queries:
        t = q["query_type"]
        types[t] = types.get(t, 0) + 1

    print(f"✓ Generated {len(queries)} queries (Layer 1: Drug Mention Audit):")
    for t, n in sorted(types.items()):
        print(f"    {t}: {n}")
    print(f"  → {csv_path}")
    print(f"  → {json_path}")

    return csv_path
