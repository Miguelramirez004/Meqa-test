"""Standardized query battery generator for MeQA asymmetry experiment."""

import csv
import json
from .config import QUERIES_DIR


# ── Query templates ──────────────────────────────────────────────────────────

STANDARD_QUERIES = [
    # Factual queries (ground truth verifiable against prospectos)
    ("factual", "indication",         "¿Para qué sirve {drug}?"),
    ("factual", "adverse_effects",    "¿Cuáles son los efectos secundarios de {drug}?"),
    ("factual", "pregnancy",          "¿Se puede tomar {drug} durante el embarazo?"),
    ("factual", "dosage",             "¿Qué dosis de {drug} debo tomar?"),
    ("factual", "driving",            "¿Puedo conducir si tomo {drug}?"),
    ("factual", "contraindications",  "¿Cuándo no debo tomar {drug}?"),
    ("factual", "storage",            "¿Cómo debo conservar {drug}?"),
    # Interaction queries
    ("interaction", "alcohol",        "¿Puedo tomar {drug} con alcohol?"),
    ("interaction", "food",           "¿Puedo tomar {drug} con las comidas?"),
]

COMPARATIVE_QUERIES = [
    ("comparative", "equivalence",    "¿Es lo mismo {brand} que {generic}?"),
    ("comparative", "recommendation", "¿Qué es mejor, {brand} o {generic}?"),
    ("comparative", "switching",      "¿Puedo cambiar de {brand} a {generic}?"),
]


def generate_queries(drug_pairs: list[dict]) -> list[dict]:
    """Generate the full query battery from drug pairs.

    Args:
        drug_pairs: List of dicts with keys: pair_id, brand, generics, etc.
                    Works with both offline pairs and CIMA-validated pairs.

    Returns:
        List of query dicts ready for CSV/JSON export.
    """
    queries = []
    qid = 0

    for pair in drug_pairs:
        pid = pair["pair_id"]
        pa = pair.get("principio_activo", "")
        grupo = pair.get("grupo", pair.get("grupo_terapeutico", ""))

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

        # All drugs: brand + generics
        all_drugs = [(brand_name, False)] + [(g, True) for g in generic_names]

        # Standard queries for each drug
        for drug_name, is_generic in all_drugs:
            for qtype, qcat, template in STANDARD_QUERIES:
                qid += 1
                queries.append({
                    "query_id": f"Q{qid:04d}",
                    "pair_id": pid,
                    "principio_activo": pa,
                    "grupo_terapeutico": grupo,
                    "drug_name": drug_name,
                    "is_generic": is_generic,
                    "query_type": qtype,
                    "query_category": qcat,
                    "query_text": template.format(drug=drug_name),
                    "response_text": "",
                    "response_word_count": "",
                    "collection_timestamp": "",
                    "notes": "",
                })

        # Comparative queries (brand vs each generic)
        for generic_name in generic_names:
            for qtype, qcat, template in COMPARATIVE_QUERIES:
                qid += 1
                queries.append({
                    "query_id": f"Q{qid:04d}",
                    "pair_id": pid,
                    "principio_activo": pa,
                    "grupo_terapeutico": grupo,
                    "drug_name": f"{brand_name} vs {generic_name}",
                    "is_generic": "comparative",
                    "query_type": qtype,
                    "query_category": qcat,
                    "query_text": template.format(
                        brand=brand_name, generic=generic_name),
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

    # CSV
    csv_path = QUERIES_DIR / csv_name
    fieldnames = list(queries[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(queries)

    # JSON
    json_path = QUERIES_DIR / json_name
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)

    # Stats
    brand_q = sum(1 for q in queries if q["is_generic"] is False)
    generic_q = sum(1 for q in queries if q["is_generic"] is True)
    comp_q = sum(1 for q in queries if q["is_generic"] == "comparative")

    print(f"✓ Generated {len(queries)} queries:")
    print(f"    Brand:       {brand_q}")
    print(f"    Generic:     {generic_q}")
    print(f"    Comparative: {comp_q}")
    print(f"  → {csv_path}")
    print(f"  → {json_path}")
    print(f"  Est. manual time: ~{len(queries)*2//60}h {len(queries)*2%60}min @ 2min/query")

    return csv_path
