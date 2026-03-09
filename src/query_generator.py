"""Layer 1: Drug Mention Audit — Query battery generator.

Generates queries organized by therapeutic area to test whether
commercial AI models mention branded or generic drug names.

All queries are OPEN (no specific product names in the prompt) so the
model must choose which names to use from its parametric knowledge.

Query types:
  1. condition   — What to take / how to treat a condition
  2. drug_class  — Questions about the pharmacological class
  3. inn         — Questions using only the INN (active ingredient)
  4. recommend   — Asking for a specific product recommendation
  5. access      — OTC/prescription/availability questions
"""

import csv
import json
from .config import QUERIES_DIR


# ============================================================
# QUERY TEMPLATES BY TYPE
# ============================================================

# TYPE 1: CONDITION — Open treatment questions (no drug/INN in prompt)
# Model freely decides which drugs/brands to mention.
CONDITION_TEMPLATES = [
    ("condition", "treatment",
     "¿Qué medicamentos se usan para tratar {condition}?"),
    ("condition", "otc_options",
     "¿Qué puedo tomar para {symptom} sin receta?"),
    ("condition", "first_line",
     "¿Cuál es el tratamiento de primera línea para {condition}?"),
    ("condition", "recommend_condition",
     "Me han diagnosticado {condition}, ¿qué medicamento me recomiendas?"),
    ("condition", "options",
     "¿Cuáles son las opciones farmacológicas para {condition}?"),
]

# TYPE 2: DRUG CLASS — Questions about the pharmacological group
# Tests whether models use brand names or INNs when describing the class.
DRUG_CLASS_TEMPLATES = [
    ("drug_class", "class_members",
     "¿Cuáles son los principales {drug_class} disponibles en España?"),
    ("drug_class", "class_best",
     "¿Cuál es el mejor {drug_class_short} para {condition}?"),
    ("drug_class", "class_compare",
     "¿Qué diferencias hay entre los distintos {drug_class}?"),
    ("drug_class", "class_generic",
     "¿Existen genéricos de los {drug_class} en España?"),
]

# TYPE 3: INN — Questions using ONLY the active ingredient name
# Tests cross-referencing: does the model mention brands when asked about INN?
INN_TEMPLATES = [
    ("inn", "what_is",
     "¿Qué es {inn}?"),
    ("inn", "indication",
     "¿Para qué sirve {inn}?"),
    ("inn", "brands",
     "¿Con qué nombres comerciales se vende {inn} en España?"),
    ("inn", "generic_available",
     "¿Hay genéricos de {inn}?"),
    ("inn", "alternatives",
     "¿Qué alternativas existen a {inn}?"),
]

# TYPE 4: RECOMMEND — Asks model for a specific product recommendation
# Most direct test: does it recommend a brand or generic by name?
RECOMMEND_TEMPLATES = [
    ("recommend", "pharmacy_ask",
     "Voy a la farmacia a por algo para {symptom}, ¿qué nombre pido?"),
    ("recommend", "brand_or_generic",
     "¿Es mejor comprar un {inn} de marca o genérico?"),
    ("recommend", "specific_product",
     "¿Qué {inn} concreto me recomiendas comprar?"),
]

# TYPE 5: ACCESS — OTC, prescription, and cost questions
# Tests whether accessibility framing changes brand/generic mention patterns.
ACCESS_TEMPLATES = [
    ("access", "prescription_needed",
     "¿Necesito receta para comprar {inn}?"),
    ("access", "cost",
     "¿Cuánto cuesta {inn} en la farmacia?"),
    ("access", "cheapest",
     "¿Cuál es la opción más barata para tratar {condition}?"),
]


def _extract_inn_name(principio_activo: str) -> str:
    """Extract just the INN from 'Omeprazol 20mg' → 'omeprazol'."""
    if not principio_activo:
        return ""
    import re
    words = principio_activo.strip().split()
    inn_words = []
    for w in words:
        if re.match(r"^\d", w) or w.upper() in ("MG", "MCG", "ML"):
            break
        inn_words.append(w)
    return " ".join(inn_words).lower()


def generate_queries(drug_pairs: list[dict]) -> list[dict]:
    """Generate open queries organized by therapeutic area.

    All queries avoid specific product names to let the model freely
    choose which drug names (brand vs INN vs generic) to mention.
    Queries are deduplicated by (condition, template) and (drug_class, template).

    Args:
        drug_pairs: List of dicts from OFFLINE_PAIRS.

    Returns:
        List of query dicts ready for CSV/JSON export.
    """
    queries = []
    qid = 0

    # Track dedup keys
    seen_condition = set()
    seen_class = set()

    for pair in drug_pairs:
        pid = pair["pair_id"]
        pa = pair.get("principio_activo", "")
        inn = _extract_inn_name(pa)
        grupo = pair.get("grupo", pair.get("grupo_terapeutico", ""))
        condition = pair.get("condition", "")
        symptom = pair.get("symptom", "")
        drug_class = pair.get("drug_class", "")
        drug_class_short = pair.get("drug_class_short", drug_class)

        # Get brand/generic names for analysis reference (not used in query text)
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

        # ── TYPE 1: Condition queries (dedup by condition text) ──
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
                    "is_generic": "condition",
                    "query_type": "condition",
                    "query_category": qcat,
                    "query_text": template.format(
                        condition=condition, symptom=symptom),
                })

        # ── TYPE 2: Drug class queries (dedup by class) ──
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
                    "is_generic": "drug_class",
                    "query_type": "drug_class",
                    "query_category": qcat,
                    "query_text": template.format(
                        drug_class=drug_class,
                        drug_class_short=drug_class_short,
                        condition=condition),
                })

        # ── TYPE 3: INN queries (one set per INN) ──
        for _, qcat, template in INN_TEMPLATES:
            qid += 1
            queries.append({
                **base_fields,
                "query_id": f"Q{qid:04d}",
                "drug_name": inn,
                "is_generic": "inn",
                "query_type": "inn",
                "query_category": qcat,
                "query_text": template.format(inn=inn),
            })

        # ── TYPE 4: Recommend queries (per INN) ──
        for _, qcat, template in RECOMMEND_TEMPLATES:
            qid += 1
            queries.append({
                **base_fields,
                "query_id": f"Q{qid:04d}",
                "drug_name": inn,
                "is_generic": "recommend",
                "query_type": "recommend",
                "query_category": qcat,
                "query_text": template.format(
                    inn=inn, symptom=symptom, condition=condition),
            })

        # ── TYPE 5: Access queries (per INN) ──
        for _, qcat, template in ACCESS_TEMPLATES:
            qid += 1
            queries.append({
                **base_fields,
                "query_id": f"Q{qid:04d}",
                "drug_name": inn,
                "is_generic": "access",
                "query_type": "access",
                "query_category": qcat,
                "query_text": template.format(
                    inn=inn, condition=condition),
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
