"""NLP-based response coding for MeQA asymmetry analysis.

Codes each MeQA response into quantitative metrics:
- Response length (words, chars, sentences)
- Information completeness (categories covered)
- Safety signal density
- Professional referral frequency
- Bioequivalence awareness
- Adverse effects granularity
"""

import re
import hashlib
import json
import csv
from pathlib import Path
from .config import RESPONSES_DIR, ANALYSIS_DIR


# ── Keyword dictionaries ────────────────────────────────────────────────────

PROFESSIONAL_KEYWORDS = [
    "médico", "farmacéutico", "profesional sanitario",
    "consulte", "acuda", "pregunte a su",
    "prescripción", "receta",
]

BIOEQUIVALENCE_KEYWORDS = [
    "bioequivalente", "bioequivalencia", "mismo principio activo",
    "equivalente terapéutico", "mismo efecto", "misma composición",
    "genérico", "efg", "intercambiable",
]

SAFETY_KEYWORDS = [
    "precaución", "advertencia", "no tome", "no debe",
    "contraindicado", "riesgo", "peligro", "grave",
    "suspenda", "interrumpa", "urgencia", "inmediatamente",
]

ADVERSE_EFFECTS = [
    "dolor de cabeza", "cefalea", "náuseas", "vómitos",
    "diarrea", "mareo", "erupción", "reacción alérgica",
    "fiebre", "insomnio", "fatiga", "cansancio",
    "estreñimiento", "somnolencia", "prurito", "urticaria",
    "dolor abdominal", "flatulencia", "sequedad de boca",
    "ansiedad", "temblor", "sudoración",
]

INFORMATION_CATEGORIES = {
    "mentions_indication": r"indicad|sirve para|tratamiento de|para tratar|se utiliza",
    "mentions_dosage": r"dosis|posología|mg|comprimido|cápsula|sobre|tomar \d",
    "mentions_adverse_effects": r"efecto|adverso|secundario|reacción",
    "mentions_contraindications": r"contraindicac|no tome|no debe tomar|alérgic|hipersensib",
    "mentions_interactions": r"interacción|otros medicamentos|junto con|combinación",
    "mentions_pregnancy": r"embaraz|lactancia|gestación|fértil|anticonceptiv",
}


def analyze_response(response_text: str, query: dict) -> dict:
    """Analyze a single MeQA response and return coded metrics.

    Args:
        response_text: Raw MeQA response text.
        query: Dict with query metadata (query_id, pair_id, etc.).

    Returns:
        Dict of coded metrics for this response.
    """
    text_lower = response_text.lower()
    words = response_text.split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', response_text) if s.strip()]

    metrics = {
        # Identity
        "query_id": query.get("query_id", ""),
        "pair_id": query.get("pair_id", ""),
        "drug_name": query.get("drug_name", ""),
        "is_generic": query.get("is_generic", ""),
        "query_type": query.get("query_type", ""),
        "query_category": query.get("query_category", ""),

        # Response length
        "word_count": len(words),
        "char_count": len(response_text),
        "sentence_count": len(sentences),

        # Professional referral
        "mentions_professional": any(kw in text_lower for kw in PROFESSIONAL_KEYWORDS),
        "professional_referral_count": sum(
            1 for kw in PROFESSIONAL_KEYWORDS if kw in text_lower),

        # Bioequivalence awareness
        "mentions_bioequivalence": any(
            kw in text_lower for kw in BIOEQUIVALENCE_KEYWORDS),

        # Safety signal density
        "safety_warnings_count": sum(
            1 for kw in SAFETY_KEYWORDS if kw in text_lower),

        # Adverse effects granularity
        "adverse_effects_named": sum(
            1 for ae in ADVERSE_EFFECTS if ae in text_lower),

        # Response hash (for test-retest reliability)
        "response_hash": hashlib.md5(response_text.encode()).hexdigest()[:12],
    }

    # Information categories
    for cat_name, pattern in INFORMATION_CATEGORIES.items():
        metrics[cat_name] = bool(re.search(pattern, text_lower))

    # Completeness score (0–6)
    metrics["completeness_score"] = sum(
        1 for cat in INFORMATION_CATEGORIES if metrics[cat])

    return metrics


def analyze_all_responses() -> list[dict]:
    """Analyze all response files in data/responses/.

    Looks for JSON files named Q*.json with 'response_text' field.
    Returns list of metric dicts and saves to CSV.
    """
    all_metrics = []
    response_files = sorted(RESPONSES_DIR.glob("Q*.json"))

    if not response_files:
        print(f"No response files found in {RESPONSES_DIR}")
        print("Expected format: Q0001.json with 'response_text' field")
        return []

    for rf in response_files:
        with open(rf, encoding="utf-8") as f:
            data = json.load(f)

        response_text = data.get("response_text", "")
        if not response_text or "error" in data:
            continue

        metrics = analyze_response(response_text, data)
        all_metrics.append(metrics)

    if all_metrics:
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        metrics_file = ANALYSIS_DIR / "response_metrics.csv"

        fieldnames = list(all_metrics[0].keys())
        with open(metrics_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics)

        print(f"✓ Analyzed {len(all_metrics)} responses → {metrics_file}")

    return all_metrics
