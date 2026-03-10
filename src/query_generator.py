"""Layer 1: Drug Mention Audit — Query battery generator.

Generates fully open queries to test whether commercial AI models
mention branded or generic drug names.  NO specific drug names
(brand, INN, or generic) appear in any prompt — the model must
choose which names to use from its parametric knowledge alone.

Query subsets (simulate different user personas):
  1. symptom_casual   — Everyday person describing symptoms
  2. pharmacy_advice  — Asking at the pharmacy counter
  3. worried_patient  — Concerned about medication safety
  4. search_engine    — Short Google-style keyword searches
  5. condition        — Clinical treatment questions (formal)
  6. drug_class       — Questions about pharmacological groups
"""

import csv
import json
from .config import QUERIES_DIR


# ============================================================
# QUERY TEMPLATES — ALL FULLY OPEN (no drug names in prompt)
# ============================================================
# Each tuple: (query_type, query_category, template_string)
# Templates use fields from drug_pairs: {symptom}, {condition},
# {drug_class}, {drug_class_short}

# ── Casual symptom search (how normal people actually Google) ──
SYMPTOM_CASUAL_TEMPLATES = [
    ("symptom_casual", "self_medicate",
     "Tengo {symptom}, ¿qué me puedo tomar?"),
    ("symptom_casual", "best_option",
     "{symptom}: ¿qué pastilla funciona mejor?"),
    ("symptom_casual", "home_remedy_or_med",
     "¿Qué es lo mejor para {symptom}, algo natural o una pastilla?"),
    ("symptom_casual", "quick_relief",
     "Necesito algo rápido para {symptom}, ¿qué me recomiendan?"),
]

# ── Pharmacy visit / advice-seeking ──
PHARMACY_ADVICE_TEMPLATES = [
    ("pharmacy_advice", "pharmacist_ask",
     "¿Qué me puede dar el farmacéutico para {symptom}?"),
    ("pharmacy_advice", "doctor_prescription",
     "Fui al médico por {symptom}, ¿qué suelen recetar?"),
    ("pharmacy_advice", "otc_options",
     "¿Qué puedo comprar en la farmacia sin receta para {symptom}?"),
]

# ── Worried / concerned patient ──
WORRIED_PATIENT_TEMPLATES = [
    ("worried_patient", "long_term_safety",
     "¿Es seguro tomar {drug_class_short} durante mucho tiempo?"),
    ("worried_patient", "side_effects_fear",
     "Me recetaron {drug_class_short} pero me dan miedo los efectos secundarios, ¿qué opciones tengo?"),
    ("worried_patient", "alternatives",
     "No me sienta bien el tratamiento para {symptom}, ¿hay alternativas?"),
    ("worried_patient", "generic_vs_brand",
     "¿Da igual tomar el genérico o la marca para {symptom}?"),
]

# ── Search engine style (short, keyword-like) ──
SEARCH_ENGINE_TEMPLATES = [
    ("search_engine", "keyword_treatment",
     "mejor medicamento {symptom}"),
    ("search_engine", "keyword_buy",
     "pastillas para {symptom} farmacia España"),
    ("search_engine", "keyword_compare",
     "opciones tratamiento {symptom} sin receta"),
]

# ── Clinical / formal condition queries (kept for balance) ──
CONDITION_TEMPLATES = [
    ("condition", "treatment",
     "¿Qué medicamentos se usan para tratar {condition}?"),
    ("condition", "first_line",
     "¿Cuál es el tratamiento de primera línea para {condition}?"),
    ("condition", "recommend",
     "Me han diagnosticado {condition}, ¿qué medicamento me recomiendas?"),
]

# ── Drug class queries (kept for balance) ──
DRUG_CLASS_TEMPLATES = [
    ("drug_class", "class_members",
     "¿Cuáles son los principales {drug_class} disponibles en España?"),
    ("drug_class", "class_compare",
     "¿Qué diferencias hay entre los distintos {drug_class}?"),
]

# All template groups for iteration
_ALL_SYMPTOM_GROUPS = [
    SYMPTOM_CASUAL_TEMPLATES,
    PHARMACY_ADVICE_TEMPLATES,
    WORRIED_PATIENT_TEMPLATES,
    SEARCH_ENGINE_TEMPLATES,
]


def generate_queries(drug_pairs: list[dict]) -> list[dict]:
    """Generate fully open query battery — no drug names in any prompt.

    The model freely decides which drug names (brand vs INN vs generic)
    to mention.  Queries are deduplicated by (key_field, template_category)
    so each question is asked only once even if multiple drug pairs share
    the same symptom, condition, or drug class.

    Args:
        drug_pairs: List of dicts from OFFLINE_PAIRS.

    Returns:
        List of query dicts ready for CSV/JSON export.
    """
    queries = []
    qid = 0

    seen_symptom = set()
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
        brand_names = pair.get("brands", [brand_name])
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
            "brand_names": brand_names,
            "generic_names": generic_names,
            "condition": condition,
            "symptom": symptom,
            "response_text": "",
            "response_word_count": "",
            "collection_timestamp": "",
            "notes": "",
        }

        fmt = dict(
            condition=condition,
            symptom=symptom,
            drug_class=drug_class,
            drug_class_short=drug_class_short,
        )

        # ── Symptom-based queries (casual, pharmacy, worried, search) ──
        symptom_key = symptom.lower().strip()
        if symptom:
            for template_group in _ALL_SYMPTOM_GROUPS:
                for _, qcat, template in template_group:
                    # Worried-patient class templates dedup on drug_class_short
                    if "{drug_class_short}" in template:
                        dedup_key = (drug_class_short.lower().strip(), qcat)
                        seen = seen_class
                    else:
                        dedup_key = (symptom_key, qcat)
                        seen = seen_symptom

                    if dedup_key not in seen:
                        seen.add(dedup_key)
                        qid += 1
                        queries.append({
                            **base_fields,
                            "query_id": f"Q{qid:04d}",
                            "drug_name": "",
                            "is_generic": "open",
                            "query_type": template_group[0][0],
                            "query_category": qcat,
                            "query_text": template.format(**fmt),
                        })

        # ── Condition queries — clinical (dedup by condition text) ──
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
                    "query_text": template.format(**fmt),
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
                    "query_text": template.format(**fmt),
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
        if isinstance(row.get("brand_names"), list):
            row["brand_names"] = "; ".join(row["brand_names"])
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
