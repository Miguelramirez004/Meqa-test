"""RAG engine for automated MeQA response collection.

Uses downloaded CIMA prospectos as a knowledge base and OpenAI as the LLM
to generate responses that simulate the MeQA system.
"""

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from .config import PROSPECTOS_DIR, RESPONSES_DIR


# ── Prospecto loading & indexing ─────────────────────────────────────────────

def load_prospectos(prospectos_dir: Path = PROSPECTOS_DIR) -> dict[str, dict]:
    """Load all prospecto JSON files and index them by drug name.

    Returns:
        Dict mapping normalised drug name -> prospecto data.
    """
    index: dict[str, dict] = {}
    for fp in prospectos_dir.glob("*.json"):
        if fp.name == ".gitkeep":
            continue
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        name = data.get("nombre", fp.stem)
        index[_normalise(name)] = data
    return index


def _normalise(name: str) -> str:
    """Lower-case, collapse whitespace, strip trailing dots."""
    return re.sub(r"\s+", " ", name.strip().lower()).rstrip(".")


# ── Context retrieval ────────────────────────────────────────────────────────

# Map query categories to prospecto section keywords so we can extract
# the most relevant text instead of sending the whole document.
_SECTION_HINTS: dict[str, list[str]] = {
    "indication": ["qué es", "para qué", "indicacion", "indicado"],
    "adverse_effects": ["efecto", "adverso", "secundario", "reaccion"],
    "pregnancy": ["embaraz", "lactancia", "fertilidad"],
    "dosage": ["dosis", "posología", "cómo tomar", "cantidad"],
    "driving": ["conducir", "conducción", "máquina"],
    "contraindications": ["no tome", "no debe", "contraindicac", "hipersensib"],
    "storage": ["conserv", "almacen", "caducidad"],
    "alcohol": ["alcohol"],
    "food": ["comida", "alimento", "comidas"],
    "equivalence": ["genérico", "bioequival", "mismo principio"],
    "recommendation": ["mejor", "diferencia", "comparar"],
    "switching": ["cambiar", "sustituir", "intercambi"],
}


def _extract_prospecto_text(prospecto: dict) -> str:
    """Extract readable text from a prospecto JSON structure."""
    sections = prospecto.get("prospecto_sections", {})
    if not sections:
        return ""
    return _walk_sections(sections)


def _walk_sections(obj, depth: int = 0) -> str:
    """Recursively extract text from CIMA segmented-document structures."""
    if isinstance(obj, str):
        # Strip HTML tags
        clean = re.sub(r"<[^>]+>", " ", obj)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean
    if isinstance(obj, list):
        parts = [_walk_sections(item, depth) for item in obj]
        return "\n".join(p for p in parts if p)
    if isinstance(obj, dict):
        parts = []
        title = obj.get("titulo", obj.get("title", ""))
        if title:
            parts.append(f"\n{'#' * min(depth + 1, 4)} {title}")
        content = obj.get("contenido", obj.get("content", ""))
        if content:
            parts.append(_walk_sections(content, depth + 1))
        # Recurse into sub-sections
        for key in ("secciones", "sections", "subsecciones"):
            if key in obj:
                parts.append(_walk_sections(obj[key], depth + 1))
        return "\n".join(p for p in parts if p)
    return ""


def _filter_sections(full_text: str, category: str) -> str:
    """Return sections most relevant to the query category, or full text if short."""
    if len(full_text) < 2000:
        return full_text

    hints = _SECTION_HINTS.get(category, [])
    if not hints:
        # Return a truncated version
        return full_text[:3000]

    # Split by headings and keep relevant ones
    parts = re.split(r"(\n#{1,4}\s.+)", full_text)
    relevant: list[str] = []
    current_heading = ""

    for part in parts:
        if re.match(r"\n#{1,4}\s", part):
            current_heading = part.lower()
        else:
            if any(h in current_heading or h in part.lower()[:200] for h in hints):
                relevant.append(current_heading + part)

    if relevant:
        return "\n".join(relevant)[:3000]
    return full_text[:3000]


def find_context(query: dict, prospectos: dict[str, dict]) -> str:
    """Find the best prospecto context for a given query.

    Args:
        query: Query dict with drug_name, query_category, etc.
        prospectos: Index from load_prospectos().

    Returns:
        Relevant prospecto text, or empty string if no match.
    """
    drug_name = query.get("drug_name", "")
    category = query.get("query_category", "")

    # For comparative queries, try both drugs
    if " vs " in drug_name:
        parts = drug_name.split(" vs ", 1)
        ctx_parts = []
        for part in parts:
            ctx = _match_drug(part.strip(), prospectos, category)
            if ctx:
                ctx_parts.append(f"--- Prospecto: {part.strip()} ---\n{ctx}")
        return "\n\n".join(ctx_parts)

    return _match_drug(drug_name, prospectos, category)


def _match_drug(drug_name: str, prospectos: dict[str, dict], category: str) -> str:
    """Match a single drug name to its prospecto and extract relevant text."""
    norm = _normalise(drug_name)

    # Exact match
    if norm in prospectos:
        full = _extract_prospecto_text(prospectos[norm])
        return _filter_sections(full, category)

    # Partial match (drug name contained in prospecto name or vice versa)
    for key, data in prospectos.items():
        if norm in key or key in norm:
            full = _extract_prospecto_text(data)
            return _filter_sections(full, category)

    # Token overlap match
    drug_tokens = set(norm.split())
    best_score, best_key = 0, None
    for key in prospectos:
        key_tokens = set(key.split())
        overlap = len(drug_tokens & key_tokens)
        if overlap > best_score:
            best_score = overlap
            best_key = key
    if best_key and best_score >= 2:
        full = _extract_prospecto_text(prospectos[best_key])
        return _filter_sections(full, category)

    return ""


# ── LLM response generation ─────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
Eres MeQA, el asistente virtual de consulta de medicamentos de la Agencia \
Española de Medicamentos y Productos Sanitarios (AEMPS). Tu función es \
responder preguntas sobre medicamentos comercializados en España basándote \
en la información oficial de los prospectos aprobados.

Instrucciones:
- Responde SIEMPRE en español.
- Basa tu respuesta en la información del prospecto proporcionada como contexto.
- Si no se proporciona contexto del prospecto, responde con tu conocimiento \
  farmacéutico general pero indica que se debe consultar el prospecto oficial.
- Sé preciso, conciso y profesional.
- Cuando sea relevante, recomienda consultar al médico o farmacéutico.
- Incluye información sobre indicaciones, dosis, efectos adversos, \
  contraindicaciones o interacciones según corresponda a la pregunta.
- No inventes información. Si no estás seguro, indícalo.
"""

_USER_TEMPLATE_WITH_CONTEXT = """\
Contexto del prospecto oficial:
{context}

---

Pregunta del usuario: {query}
"""

_USER_TEMPLATE_NO_CONTEXT = """\
Pregunta del usuario sobre el medicamento "{drug}": {query}

Nota: No se dispone del prospecto oficial para este medicamento. \
Responde con tu conocimiento farmacéutico general.
"""


def generate_response(
    query: dict,
    context: str,
    client: OpenAI,
    model: str = "gpt-4o-mini",
) -> str:
    """Call OpenAI to generate a MeQA-style response.

    Args:
        query: Query dict with query_text, drug_name, etc.
        context: Retrieved prospecto text (may be empty).
        client: Initialised OpenAI client.
        model: OpenAI model identifier.

    Returns:
        Generated response text.
    """
    query_text = query.get("query_text", "")
    drug_name = query.get("drug_name", "")

    if context.strip():
        user_msg = _USER_TEMPLATE_WITH_CONTEXT.format(
            context=context, query=query_text
        )
    else:
        user_msg = _USER_TEMPLATE_NO_CONTEXT.format(
            drug=drug_name, query=query_text
        )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=800,
    )
    return resp.choices[0].message.content.strip()


# ── Batch collection ─────────────────────────────────────────────────────────

def collect_all_responses(
    queries: list[dict],
    api_key: str,
    model: str = "gpt-4o-mini",
    prospectos_dir: Path = PROSPECTOS_DIR,
    responses_dir: Path = RESPONSES_DIR,
    delay: float = 0.5,
    progress_callback=None,
    skip_existing: bool = True,
) -> list[dict]:
    """Run RAG collection for all queries.

    Args:
        queries: List of query dicts (from query_battery.json).
        api_key: OpenAI API key.
        model: OpenAI model name.
        prospectos_dir: Path to prospecto JSON files.
        responses_dir: Path to save response files.
        delay: Seconds between API calls.
        progress_callback: Optional callable(current, total) for progress.
        skip_existing: Skip queries that already have response files.

    Returns:
        List of response dicts that were collected.
    """
    client = OpenAI(api_key=api_key)
    prospectos = load_prospectos(prospectos_dir)
    responses_dir.mkdir(parents=True, exist_ok=True)

    collected = []
    total = len(queries)

    for i, query in enumerate(queries):
        qid = query.get("query_id", f"Q{i+1:04d}")

        # Skip if already collected
        fpath = responses_dir / f"{qid}.json"
        if skip_existing and fpath.exists():
            if progress_callback:
                progress_callback(i + 1, total)
            continue

        # Retrieve context
        context = find_context(query, prospectos)

        # Generate response
        try:
            response_text = generate_response(query, context, client, model)
        except Exception as e:
            response_text = ""
            query["error"] = str(e)

        # Build response object
        response_data = {
            "query_id": qid,
            "pair_id": query.get("pair_id", ""),
            "principio_activo": query.get("principio_activo", ""),
            "grupo_terapeutico": query.get("grupo_terapeutico", ""),
            "drug_name": query.get("drug_name", ""),
            "is_generic": query.get("is_generic", ""),
            "query_type": query.get("query_type", ""),
            "query_category": query.get("query_category", ""),
            "query_text": query.get("query_text", ""),
            "response_text": response_text,
            "response_word_count": len(response_text.split()) if response_text else 0,
            "collection_timestamp": datetime.now(timezone.utc).isoformat(),
            "notes": f"RAG-generated (model={model}, context={'yes' if context else 'no'})",
        }

        # Save
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(response_data, f, ensure_ascii=False, indent=2)

        collected.append(response_data)

        if progress_callback:
            progress_callback(i + 1, total)

        if delay > 0 and i < total - 1:
            time.sleep(delay)

    return collected
