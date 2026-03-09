"""Layer 1: Drug Mention Audit — Response analysis.

Codes each response into mention-based metrics with broad drug name
recognition:

1. **INN detection** — Identifies the active ingredient (principio activo)
   mentioned by its International Non-proprietary Name (e.g., "omeprazol").
2. **Brand detection** — Recognises all known commercial brand names for
   the active ingredient in the Spanish market and internationally
   (e.g., "Losec", "Mepral", "Prilosec").
3. **Generic-product detection** — Detects specific generic products by
   INN + laboratory name (e.g., "Omeprazol Cinfa", "Omeprazol Normon").

Three-way counting (INN is NOT lumped into generic):
- brand_mention_count: explicit brand names (Losec, Keytruda, …)
- generic_mention_count: explicit generic products (INN + lab name)
- inn_mention_count: bare INN (neutral — reported separately)

Asymmetry scores compare brand vs generic-product only.
"""

import re
import hashlib
import json
import csv
from pathlib import Path
from .config import RESPONSES_DIR, ANALYSIS_DIR


# ── Brand name knowledge base ─────────────────────────────────────────────
# All known commercial (brand) names per INN in the Spanish and
# international markets.  Keys are lowercase INN.

BRAND_NAMES_BY_INN = {
    # ── 20 high-volume traditional drugs ──
    "paracetamol": ["gelocatil", "termalgin", "efferalgan", "apiretal", "dolocatil",
                    "panadol", "tylenol"],
    "ibuprofeno": ["nurofen", "espidifen", "dalsy", "dobupal", "bexistar",
                   "neobrufen", "advil", "motrin", "ibufen", "algidol"],
    "amoxicilina": ["clamoxyl", "amoxicilina beecham", "amoxil", "augmentine"],
    "omeprazol": ["losec", "belfa", "mopral", "mepral", "prilosec"],
    "atorvastatina": ["lipitor", "cardyl", "zarator", "sortis", "torvast", "totalip"],
    "enalapril": ["renitec", "cardiovasil", "dilvas", "acetensil", "naprilene"],
    "metformina": ["dianben", "glucophage"],
    "lorazepam": ["orfidal", "idalprem", "placinoral"],
    "sertralina": ["besitran", "aremis", "zoloft"],
    "salbutamol": ["ventolin", "buto asma", "salbulair", "ventolin accuhaler"],
    "amlodipino": ["norvasc", "astudal", "neotensin", "istin"],
    "azitromicina": ["zithromax", "vinzam", "toraseptol", "sumamed"],
    "simvastatina": ["zocor", "pantok", "algoren"],
    "fluoxetina": ["prozac", "adofen", "reneuron"],
    "ramipril": ["altace", "acovil", "ramicor", "triatec", "tritace"],
    "diclofenaco": ["voltaren", "artrotec"],
    "pantoprazol": ["pantoc", "anagastra", "pantoloc", "pantecta", "zurcal"],
    "levotiroxina": ["eutirox", "levothroid", "synthroid", "levothyrox"],
    "alprazolam": ["trankimazin", "tranquimazin retard", "xanax"],
    "ciprofloxacino": ["baycip", "rigoran", "ciflosin", "ciproxin", "cetraxal"],
}

# Common generic pharmaceutical laboratory names in Spain
GENERIC_LABS = [
    "cinfa", "normon", "kern pharma", "teva", "stada", "mylan",
    "sandoz", "ratiopharm", "pensa", "alter", "vir", "aurovitas",
    "zentiva", "aristo", "bluefish", "accord", "sanofi", "ranbaxy",
    "actavis", "apotex", "biogaran", "davur", "edigen", "farmalider",
    "gedeon richter", "korhispana", "ratio", "sun pharma", "tecnigen",
    "winthrop",
]


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


def _extract_inn(principio_activo: str) -> str:
    """Extract the INN (International Non-proprietary Name) from principio_activo.

    'Omeprazol 20mg' → 'omeprazol'
    'Atorvastatina 20mg' → 'atorvastatina'
    """
    if not principio_activo:
        return ""
    words = principio_activo.strip().split()
    inn_words = []
    for w in words:
        if re.match(r"^\d", w) or w.upper() in ("MG", "MCG", "ML"):
            break
        inn_words.append(w)
    return " ".join(inn_words).lower()


def _get_brand_names(inn: str, pair_brand: str = "",
                     pair_brands: list = None) -> list[str]:
    """Get all known brand names for a given INN.

    Combines the knowledge base with the pair's brand list.
    Returns lowercase list of brand names (shortest first for matching).
    """
    brands = set()

    # Add from knowledge base
    for known_inn, known_brands in BRAND_NAMES_BY_INN.items():
        if known_inn == inn or inn.startswith(known_inn):
            brands.update(b.lower() for b in known_brands)

    # Add all brands from pair metadata
    all_pair_brands = pair_brands or ([pair_brand] if pair_brand else [])
    for b in all_pair_brands:
        if not b:
            continue
        short = _extract_short_name(b)
        if short:
            brands.add(short.lower())
            first = short.split()[0]
            if first and len(first) >= 3:
                brands.add(first.lower())

    # Remove the INN itself from brand names (it's generic)
    brands.discard(inn.lower())

    return sorted(brands, key=len)


def _find_word_positions(text: str, pattern: str) -> list[int]:
    """Find all word-level positions where pattern appears in text.

    Uses word-boundary aware matching. Returns 0-indexed word positions.
    """
    if not pattern or len(pattern) < 3:
        return []

    text_lower = text.lower()
    pattern_lower = pattern.lower()
    positions = []

    # Use regex for word-boundary matching where possible
    escaped = re.escape(pattern_lower)
    try:
        for m in re.finditer(escaped, text_lower):
            idx = m.start()
            word_pos = len(text_lower[:idx].split()) - 1
            if word_pos < 0:
                word_pos = 0
            if word_pos not in positions:
                positions.append(word_pos)
    except re.error:
        # Fallback to simple find
        start = 0
        while True:
            idx = text_lower.find(pattern_lower, start)
            if idx == -1:
                break
            word_pos = len(text_lower[:idx].split()) - 1
            if word_pos < 0:
                word_pos = 0
            if word_pos not in positions:
                positions.append(word_pos)
            start = idx + len(pattern_lower)

    return sorted(positions)


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


def _find_all_brand_mentions(text: str, inn: str, pair_brand: str = "",
                             pair_brands: list = None) -> tuple[list[int], list[str]]:
    """Find all brand name mentions in text for a given INN.

    Returns (positions, brand_names_found).
    """
    brand_names = _get_brand_names(inn, pair_brand, pair_brands=pair_brands)
    all_positions = []
    found_names = []

    for brand in brand_names:
        positions = _find_word_positions(text, brand)
        if positions:
            all_positions.extend(positions)
            if brand not in found_names:
                found_names.append(brand)

    all_positions = sorted(set(all_positions))
    return all_positions, found_names


def _find_inn_mentions(text: str, inn: str) -> list[int]:
    """Find all mentions of the INN (active ingredient) in text."""
    if not inn or len(inn) < 3:
        return []
    return _find_word_positions(text, inn)


def _find_generic_product_mentions(text: str, inn: str) -> tuple[list[int], list[str]]:
    """Find mentions of specific generic products (INN + lab name).

    Detects patterns like 'omeprazol cinfa', 'atorvastatina teva', etc.
    Returns (positions, lab_names_found).
    """
    text_lower = text.lower()
    all_positions = []
    found_labs = []

    for lab in GENERIC_LABS:
        pattern = f"{inn} {lab}"
        if len(pattern) < 5:
            continue
        positions = _find_word_positions(text, pattern)
        if positions:
            all_positions.extend(positions)
            if lab not in found_labs:
                found_labs.append(lab)

    all_positions = sorted(set(all_positions))
    return all_positions, found_labs


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


def _sentences_containing_any(text: str, patterns: list[str]) -> list[str]:
    """Return sentences containing any of the given patterns."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    result = []
    for s in sentences:
        s_lower = s.lower()
        if any(p in s_lower for p in patterns if p and len(p) >= 3):
            if s not in result:
                result.append(s)
    return result


# ── Core analysis ────────────────────────────────────────────────────────

def analyze_response(response_text: str, query: dict) -> dict:
    """Analyze a single response for drug mention metrics.

    Uses broad drug name recognition:
    - INN (active ingredient): e.g., "omeprazol"
    - Brand names: all known commercial names (e.g., "Losec", "Mepral")
    - Generic products: INN + lab name (e.g., "Omeprazol Cinfa")

    Args:
        response_text: Full model response text.
        query: Dict with query metadata including brand_name, generic_names,
               principio_activo.

    Returns:
        Dict of coded metrics for this response.
    """
    text_lower = response_text.lower()
    words = response_text.split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', response_text) if s.strip()]

    brand_name = query.get("brand_name", "")
    brand_names_raw = query.get("brand_names", [])
    if isinstance(brand_names_raw, str):
        pair_brands = [b.strip() for b in brand_names_raw.split(";") if b.strip()]
    else:
        pair_brands = brand_names_raw or ([brand_name] if brand_name else [])

    generic_names_raw = query.get("generic_names", [])
    if isinstance(generic_names_raw, str):
        generic_names = [g.strip() for g in generic_names_raw.split(";") if g.strip()]
    else:
        generic_names = generic_names_raw or []

    principio_activo = query.get("principio_activo", "")
    inn = _extract_inn(principio_activo)

    # ── 1. INN (active ingredient) detection ──
    inn_positions = _find_inn_mentions(response_text, inn)
    inn_mentioned = len(inn_positions) > 0
    inn_mention_count = len(inn_positions)
    inn_first_position = inn_positions[0] if inn_positions else None

    # ── 2. Brand name detection (broad: all known brands for this INN) ──
    brand_positions, brand_names_found = _find_all_brand_mentions(
        response_text, inn, pair_brand=brand_name, pair_brands=pair_brands
    )
    brand_mentioned = len(brand_positions) > 0
    brand_mention_count = len(brand_positions)
    brand_first_position = brand_positions[0] if brand_positions else None

    # ── 3. Generic product detection (INN + lab name) ──
    generic_product_positions, generic_labs_found = _find_generic_product_mentions(
        response_text, inn
    )
    generic_product_mentioned = len(generic_product_positions) > 0
    generic_product_mention_count = len(generic_product_positions)

    # ── Combined generic = ONLY specific generic products (INN + lab) ──
    # INN mentions are tracked separately — they are neutral (the active
    # ingredient name is used by both brand and generic contexts).
    # Only explicit generic products ("Omeprazol Cinfa", "Enalapril Normon")
    # count as "generic" for asymmetry calculation.
    generic_mentioned = generic_product_mentioned
    generic_mention_count = generic_product_mention_count
    generic_first_position = (generic_product_positions[0]
                              if generic_product_positions else None)

    # ── First drug type mentioned ──
    if brand_first_position is not None and generic_first_position is not None:
        first_drug_mentioned = "brand" if brand_first_position < generic_first_position else "generic"
    elif brand_first_position is not None:
        first_drug_mentioned = "brand"
    elif generic_first_position is not None:
        first_drug_mentioned = "generic"
    else:
        first_drug_mentioned = "neither"

    # ── Sentences containing each drug type ──
    brand_patterns = brand_names_found if brand_names_found else []
    brand_info_sentences = _sentences_containing_any(response_text, brand_patterns)

    generic_patterns = [inn] if inn else []
    generic_patterns.extend(f"{inn} {lab}" for lab in generic_labs_found)
    generic_info_sentences = _sentences_containing_any(response_text, generic_patterns)

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
        "principio_activo": principio_activo,
        "grupo_terapeutico": query.get("grupo_terapeutico", ""),
        "drug_class": query.get("drug_class", ""),
        "is_generic": query.get("is_generic", ""),
        "query_type": query.get("query_type", ""),
        "query_category": query.get("query_category", ""),

        # INN (active ingredient) mentions
        "inn_mentioned": inn_mentioned,
        "inn_mention_count": inn_mention_count,
        "inn_first_position": inn_first_position,

        # Brand mentions (all known brand names for this INN)
        "brand_mentioned": brand_mentioned,
        "brand_mention_count": brand_mention_count,
        "brand_first_position": brand_first_position,
        "brand_names_found": "; ".join(brand_names_found),

        # Generic product mentions (INN + lab)
        "generic_product_mentioned": generic_product_mentioned,
        "generic_product_mention_count": generic_product_mention_count,
        "generic_labs_found": "; ".join(generic_labs_found),

        # Combined generic (INN + generic products) for asymmetry
        "generic_mentioned": generic_mentioned,
        "generic_mention_count": generic_mention_count,
        "generic_first_position": generic_first_position,

        # First-mention advantage
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

    Three-way counting:
    - brand_mention_count: explicit brand names (Losec, Keytruda, etc.)
    - generic_mention_count: explicit generic products (INN + lab: "Omeprazol Cinfa")
    - inn_mention_count: bare INN mentions (neutral — neither brand nor generic)

    INN mentions are NOT lumped into generic counts.  The asymmetry score
    compares brand vs generic-product mentions only; INN is reported
    separately as a neutral reference.

    Returns one row per drug pair.
    """
    from collections import defaultdict

    by_pair = defaultdict(list)
    for m in metrics_list:
        by_pair[m["pair_id"]].append(m)

    results = []
    for pair_id, pair_metrics in by_pair.items():
        n = len(pair_metrics)

        grupo = pair_metrics[0].get("grupo_terapeutico", "")
        drug_class = pair_metrics[0].get("drug_class", "")

        # Raw counts (three-way)
        total_brand = sum(m["brand_mention_count"] for m in pair_metrics)
        total_generic = sum(m["generic_mention_count"] for m in pair_metrics)
        total_inn = sum(m.get("inn_mention_count", 0) for m in pair_metrics)

        # Mention asymmetry: brand vs generic-product only (INN excluded)
        denom = total_brand + total_generic
        mention_asymmetry = (total_brand - total_generic) / denom if denom > 0 else 0.0

        # Rates (fraction of responses with at least one mention)
        brand_mention_rate = sum(1 for m in pair_metrics if m["brand_mentioned"]) / n
        generic_mention_rate = sum(1 for m in pair_metrics if m["generic_mentioned"]) / n
        inn_mention_rate = sum(1 for m in pair_metrics if m.get("inn_mentioned", False)) / n

        # Position asymmetry (brand first pos vs generic-product first pos)
        position_diffs = []
        for m in pair_metrics:
            bp = m.get("brand_first_position")
            gp = m.get("generic_first_position")
            if bp is not None and gp is not None:
                position_diffs.append(gp - bp)
        position_asymmetry = (
            sum(position_diffs) / len(position_diffs) if position_diffs else 0.0
        )

        # Per-type breakdown (all queries are open now)
        by_type = {}
        for m in pair_metrics:
            qt = m["query_type"]
            if qt not in by_type:
                by_type[qt] = {"brand": 0, "generic": 0, "inn": 0, "n": 0}
            by_type[qt]["brand"] += m["brand_mention_count"]
            by_type[qt]["generic"] += m["generic_mention_count"]
            by_type[qt]["inn"] += m.get("inn_mention_count", 0)
            by_type[qt]["n"] += 1

        # First-mentioned advantage (brand vs generic-product, ignoring INN-only)
        brand_first_count = sum(1 for m in pair_metrics if m["first_drug_mentioned"] == "brand")
        generic_first_count = sum(1 for m in pair_metrics if m["first_drug_mentioned"] == "generic")
        total_first = brand_first_count + generic_first_count
        first_mention_ratio = brand_first_count / total_first if total_first > 0 else 0.5

        # Composite: normalize each to [0,1] and average
        norm_mention = (mention_asymmetry + 1) / 2
        norm_position = min(max(position_asymmetry / 50 + 0.5, 0), 1)
        norm_brand_rate = brand_mention_rate
        norm_first = first_mention_ratio
        composite = (norm_mention + norm_position + norm_brand_rate + norm_first) / 4

        row = {
            "pair_id": pair_id,
            "grupo_terapeutico": grupo,
            "drug_class": drug_class,
            "n_responses": n,
            "total_brand_mentions": total_brand,
            "total_generic_mentions": total_generic,
            "total_inn_mentions": total_inn,
            "brand_mention_rate": round(brand_mention_rate, 4),
            "generic_mention_rate": round(generic_mention_rate, 4),
            "inn_mention_rate": round(inn_mention_rate, 4),
            "mention_asymmetry": round(mention_asymmetry, 4),
            "position_asymmetry": round(position_asymmetry, 2),
            "brand_first_count": brand_first_count,
            "generic_first_count": generic_first_count,
            "first_mention_ratio": round(first_mention_ratio, 4),
            "composite_asymmetry_score": round(composite, 4),
        }

        # Add per-type breakdowns
        for qt in sorted(by_type):
            row[f"{qt}_brand"] = by_type[qt]["brand"]
            row[f"{qt}_generic"] = by_type[qt]["generic"]
            row[f"{qt}_inn"] = by_type[qt]["inn"]

        results.append(row)

    results.sort(key=lambda r: r["composite_asymmetry_score"], reverse=True)
    return results


def compute_therapeutic_area_summary(metrics_list: list[dict]) -> list[dict]:
    """Compute brand vs generic mention counts grouped by therapeutic area.

    Returns one row per grupo_terapeutico with aggregated mention counts.
    """
    from collections import defaultdict

    by_grupo = defaultdict(list)
    for m in metrics_list:
        grupo = m.get("grupo_terapeutico", "Unknown")
        by_grupo[grupo].append(m)

    results = []
    for grupo, grupo_metrics in sorted(by_grupo.items()):
        n = len(grupo_metrics)
        total_brand = sum(m["brand_mention_count"] for m in grupo_metrics)
        total_generic = sum(m["generic_mention_count"] for m in grupo_metrics)
        total_inn = sum(m.get("inn_mention_count", 0) for m in grupo_metrics)
        denom = total_brand + total_generic
        mention_asymmetry = (total_brand - total_generic) / denom if denom > 0 else 0.0

        brand_rate = sum(1 for m in grupo_metrics if m["brand_mentioned"]) / n
        generic_rate = sum(1 for m in grupo_metrics if m["generic_mentioned"]) / n

        brand_first = sum(1 for m in grupo_metrics if m["first_drug_mentioned"] == "brand")
        generic_first = sum(1 for m in grupo_metrics if m["first_drug_mentioned"] == "generic")

        # Breakdown by query type
        by_type = defaultdict(lambda: {"brand": 0, "generic": 0, "n": 0})
        for m in grupo_metrics:
            qt = m["query_type"]
            by_type[qt]["brand"] += m["brand_mention_count"]
            by_type[qt]["generic"] += m["generic_mention_count"]
            by_type[qt]["n"] += 1

        row = {
            "grupo_terapeutico": grupo,
            "n_responses": n,
            "total_brand_mentions": total_brand,
            "total_generic_mentions": total_generic,
            "total_inn_mentions": total_inn,
            "brand_mention_rate": round(brand_rate, 4),
            "generic_mention_rate": round(generic_rate, 4),
            "mention_asymmetry": round(mention_asymmetry, 4),
            "brand_first": brand_first,
            "generic_first": generic_first,
        }

        # Add per-type counts
        for qt in sorted(by_type):
            row[f"{qt}_brand"] = by_type[qt]["brand"]
            row[f"{qt}_generic"] = by_type[qt]["generic"]
            row[f"{qt}_n"] = by_type[qt]["n"]

        results.append(row)

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

        # Compute and save asymmetry scores per pair
        asymmetry = compute_asymmetry_scores(all_metrics)
        if asymmetry:
            asym_file = ANALYSIS_DIR / "asymmetry_scores.csv"
            asym_fields = list(asymmetry[0].keys())
            with open(asym_file, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=asym_fields)
                writer.writeheader()
                writer.writerows(asymmetry)
            print(f"✓ Computed asymmetry scores for {len(asymmetry)} pairs → {asym_file}")

        # Compute and save therapeutic area summary
        area_summary = compute_therapeutic_area_summary(all_metrics)
        if area_summary:
            area_file = ANALYSIS_DIR / "therapeutic_area_summary.csv"
            area_fields = list(area_summary[0].keys())
            with open(area_file, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=area_fields)
                writer.writeheader()
                writer.writerows(area_summary)
            print(f"✓ Computed therapeutic area summary for {len(area_summary)} areas → {area_file}")

    return all_metrics
