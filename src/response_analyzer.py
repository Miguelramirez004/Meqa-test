"""Layer 1: Drug Mention Audit — Response analysis.

Codes each response into mention-based metrics:
- Which drug names appear (brand, generic, both, neither)
- Position of first mention (word index)
- Mention frequency (count)
- Information density around each mention (sentences)
- Cross-reference detection (does querying one drug pull in the paired drug?)

After collection, computes asymmetry scores per drug pair:
- MENTION_ASYMMETRY: normalized mention count difference
- POSITION_ASYMMETRY: which drug appears earlier
- CROSS_REF_RATIO: directional cross-reference probability
- COMPOSITE_ASYMMETRY_SCORE: mean of normalized scores
"""

import re
import hashlib
import json
import csv
from pathlib import Path
from .config import RESPONSES_DIR, ANALYSIS_DIR


# ── Keyword dictionaries (kept for supplementary analysis) ────────────────

PROFESSIONAL_KEYWORDS = [
    "médico", "farmacéutico", "profesional sanitario",
    "consulte", "acuda", "pregunte a su",
    "prescripción", "receta",
    "doctor", "especialista", "profesional de la salud",
    "atención médica", "servicio de urgencias", "centro de salud",
    "consulta médica", "supervisión médica", "control médico",
    "bajo prescripción", "indicación médica",
]

BIOEQUIVALENCE_KEYWORDS = [
    "bioequivalente", "bioequivalencia", "mismo principio activo",
    "equivalente terapéutico", "mismo efecto", "misma composición",
    "genérico", "efg", "intercambiable",
    "equivalencia", "misma eficacia", "misma seguridad",
    "mismo medicamento", "sustancia activa", "principio activo",
    "medicamento genérico", "especialidad genérica",
    "biodisponibilidad", "misma calidad",
]

SAFETY_KEYWORDS = [
    "precaución", "advertencia", "no tome", "no debe",
    "contraindicado", "riesgo", "peligro", "grave",
    "suspenda", "interrumpa", "urgencia", "inmediatamente",
    "atención inmediata", "reacción grave", "efectos graves",
    "sobredosis", "intoxicación", "toxicidad",
    "evite", "no utilice", "no use", "no administre",
    "tenga cuidado", "especial cuidado", "tenga precaución",
    "informe a su médico", "avise a su médico",
]

ADVERSE_EFFECTS = [
    "dolor de cabeza", "cefalea", "náuseas", "vómitos",
    "diarrea", "mareo", "erupción", "reacción alérgica",
    "fiebre", "insomnio", "fatiga", "cansancio",
    "estreñimiento", "somnolencia", "prurito", "urticaria",
    "dolor abdominal", "flatulencia", "sequedad de boca",
    "ansiedad", "temblor", "sudoración",
    "palpitaciones", "taquicardia", "hipotensión",
    "edema", "hinchazón", "artralgia", "mialgia",
    "dolor muscular", "dolor articular", "visión borrosa",
    "alopecia", "caída del cabello", "aumento de peso",
    "pérdida de apetito", "dispepsia", "acidez",
    "erupción cutánea", "dermatitis", "fotosensibilidad",
    "reacción anafiláctica", "angioedema", "broncoespasmo",
    "hepatotoxicidad", "ictericia", "nefrotoxicidad",
    "parestesia", "vértigo", "acúfenos", "tinnitus",
]

# Phrases that indicate the RAG framework could NOT retrieve data
INCOMPLETE_RESPONSE_INDICATORS = [
    "no se encuentra en el prospecto",
    "no se ha encontrado",
    "no dispongo de información",
    "no tengo información",
    "no puedo proporcionar",
    "información no disponible",
    "no se ha podido recuperar",
    "no se ha podido obtener",
    "consulte el prospecto oficial",
    "no se encuentra disponible",
    "no aparece en el prospecto",
    "no consta en el prospecto",
    "no he encontrado información",
    "no se ha localizado",
]

INFORMATION_CATEGORIES = {
    "mentions_indication": r"indicad|sirve para|tratamiento de|para tratar|se utiliza|está indicado|indicaciones",
    "mentions_dosage": r"dosis|posología|mg|comprimido|cápsula|sobre|tomar \d|administración|pauta|vía oral",
    "mentions_adverse_effects": r"efecto|adverso|secundario|reacción|no deseado|frecuente|poco frecuente|raro",
    "mentions_contraindications": r"contraindicac|no tome|no debe tomar|alérgic|hipersensib|no utilice|prohibido",
    "mentions_interactions": r"interacción|otros medicamentos|junto con|combinación|concomitante|simultáneamente|asociación",
    "mentions_pregnancy": r"embaraz|lactancia|gestación|fértil|anticonceptiv|periodo de lactancia|mujeres en edad",
}


# ── Drug name matching helpers ────────────────────────────────────────────

def _extract_short_name(full_name: str) -> str:
    """Extract the first recognizable token from a full drug name.

    'LOSEC 20 MG CAPSULAS DURAS GASTRORRESISTENTES' → 'losec'
    'OMEPRAZOL CINFA 20 MG ...' → 'omeprazol cinfa'
    """
    words = full_name.strip().split()
    short = []
    for w in words:
        if re.match(r"^\d", w) or w.upper() in ("MG", "MCG", "ML", "EFG"):
            break
        short.append(w)
    return " ".join(short).lower() if short else full_name.lower()


def _find_mentions(text: str, drug_name: str) -> list[int]:
    """Find all word positions where drug_name appears in text.

    Uses the short name (brand word or generic+lab) for matching.
    Returns list of word-level positions (0-indexed).
    """
    short = _extract_short_name(drug_name)
    if not short:
        return []

    text_lower = text.lower()
    first_word = short.split()[0] if short else ""

    positions = []
    for pattern in [short, first_word]:
        if not pattern or len(pattern) < 3:
            continue
        start = 0
        while True:
            idx = text_lower.find(pattern, start)
            if idx == -1:
                break
            word_pos = len(text_lower[:idx].split()) - 1
            if word_pos < 0:
                word_pos = 0
            if word_pos not in positions:
                positions.append(word_pos)
            start = idx + len(pattern)

    positions.sort()
    return positions


def _sentences_containing(text: str, drug_name: str) -> list[str]:
    """Return sentences from text that mention the drug."""
    short = _extract_short_name(drug_name)
    if not short:
        return []
    first_word = short.split()[0] if short else ""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    result = []
    for s in sentences:
        s_lower = s.lower()
        if short in s_lower or (first_word and len(first_word) >= 3 and first_word in s_lower):
            result.append(s)
    return result


# ── Core analysis ────────────────────────────────────────────────────────

def analyze_response(response_text: str, query: dict) -> dict:
    """Analyze a single response for drug mention metrics.

    Args:
        response_text: Full model response text.
        query: Dict with query metadata including brand_name, generic_names.

    Returns:
        Dict of coded metrics for this response.
    """
    text_lower = response_text.lower()
    words = response_text.split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', response_text) if s.strip()]

    brand_name = query.get("brand_name", "")
    generic_names_raw = query.get("generic_names", [])
    if isinstance(generic_names_raw, str):
        generic_names = [g.strip() for g in generic_names_raw.split(";") if g.strip()]
    else:
        generic_names = generic_names_raw or []

    # ── Drug mention detection ──
    brand_positions = _find_mentions(response_text, brand_name) if brand_name else []
    brand_mentioned = len(brand_positions) > 0
    brand_mention_count = len(brand_positions)
    brand_first_position = brand_positions[0] if brand_positions else None

    generic_positions = []
    for gn in generic_names:
        generic_positions.extend(_find_mentions(response_text, gn))
    generic_positions = sorted(set(generic_positions))
    generic_mentioned = len(generic_positions) > 0
    generic_mention_count = len(generic_positions)
    generic_first_position = generic_positions[0] if generic_positions else None

    if brand_first_position is not None and generic_first_position is not None:
        first_drug_mentioned = "brand" if brand_first_position <= generic_first_position else "generic"
    elif brand_first_position is not None:
        first_drug_mentioned = "brand"
    elif generic_first_position is not None:
        first_drug_mentioned = "generic"
    else:
        first_drug_mentioned = "neither"

    # ── Sentences containing each drug ──
    brand_info_sentences = _sentences_containing(response_text, brand_name) if brand_name else []
    generic_info_sentences = []
    for gn in generic_names:
        generic_info_sentences.extend(_sentences_containing(response_text, gn))
    seen = set()
    unique_generic_sentences = []
    for s in generic_info_sentences:
        if s not in seen:
            seen.add(s)
            unique_generic_sentences.append(s)
    generic_info_sentences = unique_generic_sentences

    # ── Supplementary keyword metrics ──
    mentions_professional = any(kw in text_lower for kw in PROFESSIONAL_KEYWORDS)
    mentions_bioequivalence = any(kw in text_lower for kw in BIOEQUIVALENCE_KEYWORDS)
    safety_warnings_count = sum(1 for kw in SAFETY_KEYWORDS if kw in text_lower)

    metrics = {
        # Identification
        "query_id": query.get("query_id", ""),
        "pair_id": query.get("pair_id", ""),
        "drug_name": query.get("drug_name", ""),
        "brand_name": brand_name,
        "is_generic": query.get("is_generic", ""),
        "query_type": query.get("query_type", ""),
        "query_category": query.get("query_category", ""),

        # Core mention data
        "brand_mentioned": brand_mentioned,
        "generic_mentioned": generic_mentioned,
        "brand_mention_count": brand_mention_count,
        "generic_mention_count": generic_mention_count,
        "brand_first_position": brand_first_position,
        "generic_first_position": generic_first_position,
        "first_drug_mentioned": first_drug_mentioned,

        # Content around mentions
        "brand_info_sentences": "; ".join(brand_info_sentences),
        "generic_info_sentences": "; ".join(generic_info_sentences),

        # Response stats
        "response_word_count": len(words),
        "response_char_count": len(response_text),
        "response_sentence_count": len(sentences),

        # Supplementary
        "mentions_professional": mentions_professional,
        "mentions_bioequivalence": mentions_bioequivalence,
        "safety_warnings_count": safety_warnings_count,

        # Response hash (test-retest reliability)
        "response_hash": hashlib.md5(response_text.encode()).hexdigest()[:12],
    }

    return metrics


def is_incomplete_response(response_text: str, data: dict) -> bool:
    """Check if a response is incomplete (RAG framework could not retrieve data).

    A response is considered incomplete when:
    - The RAG context was not available (no prospecto chunks retrieved)
    - The response text contains phrases indicating data was not found
    - The response has zero retrieved chunks

    These incomplete responses should be disregarded before analysis
    to avoid skewing metrics with non-answers.
    """
    metadata = data.get("meqa_metadata", {})
    if metadata:
        if not metadata.get("context_available", True):
            return True
        if metadata.get("chunks_retrieved", -1) == 0:
            return True

    text_lower = response_text.lower()
    for indicator in INCOMPLETE_RESPONSE_INDICATORS:
        if indicator in text_lower:
            return True

    return False


# ── Asymmetry score computation ──────────────────────────────────────────

def compute_asymmetry_scores(metrics_list: list[dict]) -> list[dict]:
    """Compute per-pair asymmetry scores from collected mention metrics.

    Returns one row per drug pair with:
    - MENTION_ASYMMETRY: (brand_count - generic_count) / (brand_count + generic_count)
    - POSITION_ASYMMETRY: generic_first_pos - brand_first_pos
    - CROSS_REF_RATIO: P(brand mentioned | generic queried)
                     / P(generic mentioned | brand queried)
    - COMPOSITE_ASYMMETRY_SCORE: mean of normalized component scores
    """
    from collections import defaultdict

    by_pair = defaultdict(list)
    for m in metrics_list:
        by_pair[m["pair_id"]].append(m)

    results = []
    for pair_id, pair_metrics in by_pair.items():
        # Mention asymmetry
        total_brand = sum(m["brand_mention_count"] for m in pair_metrics)
        total_generic = sum(m["generic_mention_count"] for m in pair_metrics)
        denom = total_brand + total_generic
        mention_asymmetry = (total_brand - total_generic) / denom if denom > 0 else 0.0

        # Position asymmetry
        position_diffs = []
        for m in pair_metrics:
            bp = m.get("brand_first_position")
            gp = m.get("generic_first_position")
            if bp is not None and gp is not None:
                position_diffs.append(gp - bp)
        position_asymmetry = (
            sum(position_diffs) / len(position_diffs) if position_diffs else 0.0
        )

        # Cross-reference ratio
        generic_queried = [m for m in pair_metrics
                          if m["query_type"] == "single" and m["is_generic"] is True]
        brand_queried = [m for m in pair_metrics
                         if m["query_type"] == "single" and m["is_generic"] is False]

        p_brand_given_generic = (
            sum(1 for m in generic_queried if m["brand_mentioned"]) / len(generic_queried)
            if generic_queried else 0.0
        )
        p_generic_given_brand = (
            sum(1 for m in brand_queried if m["generic_mentioned"]) / len(brand_queried)
            if brand_queried else 0.0
        )
        cross_ref_ratio = (
            p_brand_given_generic / p_generic_given_brand
            if p_generic_given_brand > 0
            else float("inf") if p_brand_given_generic > 0
            else 1.0
        )

        # First-mentioned advantage
        brand_first_count = sum(1 for m in pair_metrics if m["first_drug_mentioned"] == "brand")
        generic_first_count = sum(1 for m in pair_metrics if m["first_drug_mentioned"] == "generic")
        total_first = brand_first_count + generic_first_count
        first_mention_ratio = brand_first_count / total_first if total_first > 0 else 0.5

        # Composite: normalize each to [0,1] and average
        norm_mention = (mention_asymmetry + 1) / 2
        norm_position = min(max(position_asymmetry / 50 + 0.5, 0), 1)
        norm_crossref = min(cross_ref_ratio / 2, 1) if cross_ref_ratio != float("inf") else 1.0
        norm_first = first_mention_ratio
        composite = (norm_mention + norm_position + norm_crossref + norm_first) / 4

        results.append({
            "pair_id": pair_id,
            "total_brand_mentions": total_brand,
            "total_generic_mentions": total_generic,
            "mention_asymmetry": round(mention_asymmetry, 4),
            "position_asymmetry": round(position_asymmetry, 2),
            "p_brand_given_generic_queried": round(p_brand_given_generic, 4),
            "p_generic_given_brand_queried": round(p_generic_given_brand, 4),
            "cross_ref_ratio": round(cross_ref_ratio, 4) if cross_ref_ratio != float("inf") else "inf",
            "brand_first_count": brand_first_count,
            "generic_first_count": generic_first_count,
            "first_mention_ratio": round(first_mention_ratio, 4),
            "composite_asymmetry_score": round(composite, 4),
        })

    results.sort(key=lambda r: r["composite_asymmetry_score"], reverse=True)
    return results


# ── Main analysis entry point ────────────────────────────────────────────

def analyze_all_responses() -> list[dict]:
    """Analyze all response files in data/responses/.

    Looks for JSON files named Q*.json with 'response_text' field.
    Disregards incomplete responses where the RAG framework could not
    retrieve prospecto data, as these non-answers would skew metrics.
    Returns list of metric dicts and saves to CSV.
    """
    all_metrics = []
    skipped_incomplete = 0
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

        if is_incomplete_response(response_text, data):
            skipped_incomplete += 1
            continue

        metrics = analyze_response(response_text, data)
        all_metrics.append(metrics)

    if skipped_incomplete:
        print(f"  Skipped {skipped_incomplete} incomplete responses (no RAG data retrieved)")

    if all_metrics:
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

        # Save per-response metrics
        metrics_file = ANALYSIS_DIR / "response_metrics.csv"
        fieldnames = list(all_metrics[0].keys())
        with open(metrics_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"✓ Analyzed {len(all_metrics)} responses → {metrics_file}")

        # Compute and save asymmetry scores
        asymmetry = compute_asymmetry_scores(all_metrics)
        if asymmetry:
            asym_file = ANALYSIS_DIR / "asymmetry_scores.csv"
            asym_fields = list(asymmetry[0].keys())
            with open(asym_file, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=asym_fields)
                writer.writeheader()
                writer.writerows(asymmetry)
            print(f"✓ Computed asymmetry scores for {len(asymmetry)} pairs → {asym_file}")

    return all_metrics
