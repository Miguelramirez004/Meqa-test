"""RAG engine replicating the MeQA architecture with modern components.

Based on: Santamaría, J. (2021). "Medicines Question Answering System, MeQA"
          AEMPS — arXiv:2111.02760

Adaptation of MEQa's 3-block pipeline to a modern RAG system:

  MEQa Block 1 (Question Processing)    → query normalisation + entity extraction
                                           + section prediction
  MEQa Block 2 (Leaflet & Section Ext.) → in-memory vector retrieval with metadata
                                           filtering (replaces TF-IDF + VSM/LSI)
  MEQa Block 3 (Answer Extraction)      → LLM generation grounded in retrieved
                                           chunks with section context

Key design:
  - Embedding: OpenAI text-embedding-3-small (1536-dim, multilingual)
  - Vector store: in-memory numpy (no external DB needed)
  - Generation: gpt-4o-mini via OpenAI API (temp=0.0 for reproducibility)
  - Section metadata preserved end-to-end (fetch → chunk → embed → retrieve → prompt)
  - Identical pipeline for branded and generic drugs
  - Self-bootstrapping: auto-fetches leaflets from CIMA if store is empty
"""

import hashlib
import json
import logging
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from .config import DATA_DIR, RESPONSES_DIR, PROSPECTOS_DIR, PAIRS_DIR
from .config import LEAFLETS_DIR
from .vector_store import MeQAVectorStore

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 1 — QUESTION PROCESSING
# (Simulates MeQA's Normalization + NER + bi-LSTM section prediction)
# ═══════════════════════════════════════════════════════════════════════════════

# The 6 standard leaflet sections (same structure for ALL Spanish prospectos)
LEAFLET_SECTIONS = {
    1: "Qué es y para qué se utiliza",
    2: "Qué necesita saber antes de empezar a tomar",
    3: "Cómo tomar",
    4: "Posibles efectos adversos",
    5: "Conservación",
    6: "Contenido del envase e información adicional",
}

# Section prediction: maps query_category → predicted leaflet section numbers.
# Simulates the bi-LSTM model trained on 13,989 Doctoralia questions.
SECTION_PREDICTION: dict[str, list[int]] = {
    "indication":          [1],
    "adverse_effects":     [4],
    "pregnancy":           [2],
    "dosage":              [3],
    "driving":             [2],
    "contraindications":   [2],
    "storage":             [5],
    "alcohol":             [2, 3],
    "food":                [2, 3],
    "equivalence":         [1, 6],
    "recommendation":      [1, 2],
    "switching":           [1, 6],
}


def normalise_query(text: str) -> str:
    """MeQA Block 1 — Normalization module."""
    text = text.lower().strip()
    nfkd = unicodedata.normalize("NFKD", text)
    text_no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    text_no_accents = text_no_accents.replace("n\u0303", "ñ")
    text_clean = re.sub(r"[^\w\s¿?.,]", " ", text_no_accents)
    return re.sub(r"\s+", " ", text_clean).strip()


def extract_entities(query: dict) -> dict:
    """MeQA Block 1 — Simplified NER module."""
    drug_name = query.get("drug_name", "")
    principio = query.get("principio_activo", "")

    dose_match = re.search(r"(\d+(?:[.,]\d+)?\s*(?:mg|mcg|g|ml|ui))", drug_name, re.I)
    dose = dose_match.group(1) if dose_match else ""

    form_patterns = [
        "comprimidos recubiertos con película", "comprimidos recubiertos",
        "cápsulas duras gastrorresistentes", "cápsulas duras", "capsulas duras",
        "comprimidos", "cápsulas", "capsulas", "solución oral", "solucion oral",
        "sobres", "inyectable", "gel", "crema", "pomada", "gotas", "jarabe", "polvo",
    ]
    form = ""
    name_lower = drug_name.lower()
    for fp in sorted(form_patterns, key=len, reverse=True):
        if fp in name_lower:
            form = fp
            break

    return {
        "medicine_name": drug_name,
        "active_ingredient": principio,
        "dose": dose,
        "pharma_form": form,
        "is_generic": query.get("is_generic", ""),
    }


def predict_sections(query_category: str) -> list[int]:
    """MeQA Block 1 — Section prediction (simulates bi-LSTM)."""
    return SECTION_PREDICTION.get(query_category, [1, 2, 4])


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 2 — RETRIEVAL (replaces TF-IDF + VSM/LSI with dense vector retrieval)
# ═══════════════════════════════════════════════════════════════════════════════

def retrieve_chunks(
    query_text: str,
    store: MeQAVectorStore,
    pair_id: str,
    drug_name: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    """Retrieve relevant chunks from the vector store."""
    return store.retrieve(
        query=query_text,
        top_k=top_k,
        pair_id=pair_id,
        drug_name=drug_name,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 3 — ANSWER GENERATION (replaces rule-based sentence extraction)
# ═══════════════════════════════════════════════════════════════════════════════

# Fixed system prompt — identical for every single query
_SYSTEM_PROMPT = (
    "Eres un asistente farmacéutico. Responde a la pregunta del paciente "
    "utilizando ÚNICAMENTE la información proporcionada en los fragmentos "
    "del prospecto. Si la información no está en los fragmentos proporcionados, "
    "indica que no se encontró la información en el prospecto. Responde en español."
)

_USER_TEMPLATE = """\
Contexto del prospecto:
---
{context}
---

Pregunta del paciente: {query}"""

_USER_TEMPLATE_NO_CONTEXT = """\
Pregunta sobre el medicamento "{drug}" ({ingredient}): {query}

NOTA: No se ha encontrado el prospecto de este medicamento en la base de datos. \
Responde con tu conocimiento farmacéutico general pero indica claramente que \
se debe consultar el prospecto oficial aprobado por la AEMPS."""


def _format_context(results: list[dict]) -> str:
    """Format retrieved chunks as context for the LLM prompt."""
    if not results:
        return ""
    parts = [r["text"] for r in results]
    return "\n---\n".join(parts)


def generate_response(
    query_text: str,
    context: str,
    client: OpenAI,
    model: str = "gpt-4o-mini",
    drug_name: str = "",
    ingredient: str = "",
) -> str:
    """Generate a response using the LLM with retrieved context."""
    if context.strip():
        user_msg = _USER_TEMPLATE.format(context=context, query=query_text)
    else:
        user_msg = _USER_TEMPLATE_NO_CONTEXT.format(
            drug=drug_name, ingredient=ingredient, query=query_text
        )

    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    return resp.choices[0].message.content.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-BOOTSTRAP: fetch leaflets from CIMA, chunk, and index into vector store
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_drugs_from_queries(queries: list[dict]) -> list[dict]:
    """Extract unique (pair_id, drug_name, is_generic) from queries.

    Handles two query formats:
      - Old format: each query has a specific drug_name + is_generic bool
      - New "open" format: drug_name is empty; brand_names/generic_names lists present

    Returns a list of drug dicts suitable for the bootstrap pipeline.
    """
    seen = set()
    drugs = []

    for q in queries:
        is_generic = q.get("is_generic", "")
        pair_id = q.get("pair_id", "")
        principio = q.get("principio_activo", "")

        # ── Old format: query has a specific drug_name ──
        drug_name = q.get("drug_name", "")
        if drug_name and is_generic != "comparative" and " vs " not in drug_name:
            key = (pair_id, drug_name)
            if key not in seen:
                seen.add(key)
                drugs.append({
                    "pair_id": pair_id,
                    "drug_name": drug_name,
                    "is_generic": is_generic is True or is_generic == "True",
                    "principio_activo": principio,
                })
            continue

        # ── New "open" format: extract from brand_names / generic_names ──
        for name in q.get("brand_names", []):
            if isinstance(name, dict):
                name = name.get("nombre", "")
            key = (pair_id, name)
            if name and key not in seen:
                seen.add(key)
                drugs.append({
                    "pair_id": pair_id,
                    "drug_name": name,
                    "is_generic": False,
                    "principio_activo": principio,
                })

        for name in q.get("generic_names", []):
            if isinstance(name, dict):
                name = name.get("nombre", "")
            key = (pair_id, name)
            if name and key not in seen:
                seen.add(key)
                drugs.append({
                    "pair_id": pair_id,
                    "drug_name": name,
                    "is_generic": True,
                    "principio_activo": principio,
                })

    return drugs


def _lookup_nregistro_from_cima(drug_name: str) -> str | None:
    """Look up nregistro from CIMA using multiple search strategies.

    Tries (in order):
      1. Exact full name search
      2. First word only (brand name, e.g. "LOSEC" from "LOSEC 20 MG ...")
      3. First two words (brand + dose)
    """
    from .cima_client import CIMAClient
    cima = CIMAClient(delay=0.5)

    name_upper = drug_name.upper().strip()
    # Build search terms: full name, first word, first two words
    words = name_upper.split()
    search_terms = [name_upper]
    if len(words) > 2:
        search_terms.append(" ".join(words[:2]))
    if len(words) > 1:
        search_terms.append(words[0])

    for term in search_terms:
        try:
            result = cima.search_medicamentos(nombre=term)
        except Exception as e:
            logger.warning("CIMA search error for '%s': %s", term[:40], e)
            continue

        if not result or "resultados" not in result:
            continue

        # Try exact match first
        for med in result["resultados"]:
            if med.get("nombre", "").upper() == name_upper:
                logger.info("  CIMA exact match: %s → %s", term[:30], med["nregistro"])
                return med["nregistro"]

        # Try prefix match (drug name starts with search term)
        for med in result["resultados"]:
            med_name = med.get("nombre", "").upper()
            if med_name.startswith(name_upper) or name_upper.startswith(med_name):
                logger.info("  CIMA prefix match: %s → %s", term[:30], med["nregistro"])
                return med["nregistro"]

        # Fallback: first result with a matching first word
        first_word = words[0] if words else ""
        for med in result["resultados"]:
            if med.get("nombre", "").upper().startswith(first_word):
                logger.info("  CIMA first-word match: %s → %s", term[:30], med["nregistro"])
                return med["nregistro"]

        # Last resort: first result from the shortest search term
        if result["resultados"] and term == search_terms[-1]:
            nreg = result["resultados"][0]["nregistro"]
            logger.info("  CIMA fallback match: %s → %s", term[:30], nreg)
            return nreg

    logger.warning("CIMA lookup failed for: %s (tried %d strategies)", drug_name[:50], len(search_terms))
    return None


def _try_load_prospecto_json(drug_name: str,
                             prospectos_dir: Path = PROSPECTOS_DIR) -> dict | None:
    """Try to find a matching prospecto JSON in data/prospectos/.

    Matches by exact name first, then by prefix (handles short brand names
    like "Gelocatil" matching "GELOCATIL 650 MG COMPRIMIDOS"), then by
    substring in the filename (handles brand names embedded in filenames).

    Returns the parsed JSON if found, None otherwise.
    """
    if not prospectos_dir.exists():
        return None

    name_upper = drug_name.upper()
    best_match = None

    for fp in prospectos_dir.glob("*.json"):
        if fp.name == ".gitkeep":
            continue
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
            nombre = data.get("nombre", "").upper()
            # Exact match — return immediately
            if nombre == name_upper:
                return data
            # Prefix match (e.g. "Gelocatil" matches "GELOCATIL 650 MG ...")
            if nombre.startswith(name_upper + " ") or name_upper.startswith(nombre + " "):
                best_match = data
            # Filename substring match (e.g. "Lipitor" in "12345_LIPITOR_20MG.json")
            elif name_upper in fp.stem.upper().replace("_", " "):
                best_match = data
        except (json.JSONDecodeError, KeyError):
            continue

    return best_match


def _load_all_prospectos(prospectos_dir: Path = PROSPECTOS_DIR) -> list[dict]:
    """Load all prospecto JSON files from the prospectos directory."""
    if not prospectos_dir.exists():
        return []

    prospectos = []
    for fp in prospectos_dir.glob("*.json"):
        if fp.name == ".gitkeep":
            continue
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("prospecto_sections"):
                prospectos.append(data)
        except (json.JSONDecodeError, KeyError):
            continue

    return prospectos


def _chunk_prospecto_json(
    data: dict,
    pair_id: str,
    is_generic: bool,
) -> list[dict]:
    """Chunk a prospecto JSON file (from CIMA segmented API) into section-aware chunks.

    Handles multiple CIMA API response formats:
      - prospecto_sections: {titulo: ..., secciones: [{titulo, contenido}, ...]}
      - prospecto_sections: [{titulo, contenido}, ...]
      - prospecto_sections: {datos: {secciones: [...]}}
      - contenido can be str (HTML) or list of str
    """
    from .leaflet_chunker import chunk_section

    drug_name = data.get("nombre", "")
    nregistro = data.get("nregistro", "")
    sections_data = data.get("prospecto_sections")

    if not sections_data:
        logger.warning("  %s: prospecto_sections is empty/missing", drug_name[:40])
        return []

    # ── Extract the secciones list from various possible structures ──
    secciones = []
    if isinstance(sections_data, list):
        secciones = sections_data
    elif isinstance(sections_data, dict):
        secciones = sections_data.get("secciones", [])
        # Nested under 'datos'
        if not secciones and "datos" in sections_data:
            datos = sections_data["datos"]
            if isinstance(datos, dict):
                secciones = datos.get("secciones", [])
            elif isinstance(datos, list):
                secciones = datos

    if not secciones:
        logger.warning("  %s: no secciones found (keys=%s)",
                        drug_name[:40],
                        list(sections_data.keys()) if isinstance(sections_data, dict) else type(sections_data).__name__)
        return []

    all_chunks = []
    sec_counter = 0  # Fallback section numbering

    for sec in secciones:
        if not isinstance(sec, dict):
            continue

        titulo = sec.get("titulo", sec.get("title", ""))
        contenido = sec.get("contenido", sec.get("content", ""))

        # contenido can be a list of HTML fragments
        if isinstance(contenido, list):
            contenido = "\n".join(str(c) for c in contenido if c)

        if not contenido or not contenido.strip():
            continue

        sec_counter += 1

        # Detect section number from title (e.g., "1. Qué es...")
        sec_num = 0
        match = re.match(r"^(\d+)\.", titulo)
        if match:
            sec_num = int(match.group(1))

        if sec_num < 1 or sec_num > 6:
            # Try keyword detection
            titulo_lower = titulo.lower()
            if "qué es" in titulo_lower or "para qué" in titulo_lower:
                sec_num = 1
            elif "antes de" in titulo_lower or "necesita saber" in titulo_lower:
                sec_num = 2
            elif "cómo tomar" in titulo_lower or "cómo usar" in titulo_lower:
                sec_num = 3
            elif "efectos adversos" in titulo_lower or "efectos secundarios" in titulo_lower:
                sec_num = 4
            elif "conserv" in titulo_lower:
                sec_num = 5
            elif "contenido" in titulo_lower or "información adicional" in titulo_lower:
                sec_num = 6

        # If still unknown, assign sequentially (1-6)
        if sec_num == 0:
            sec_num = min(sec_counter, 6)

        section_chunks = chunk_section(
            html=contenido,
            section_number=sec_num,
            drug_name=drug_name,
            nregistro=nregistro,
            pair_id=pair_id,
            is_generic=is_generic,
        )
        all_chunks.extend(section_chunks)

    if not all_chunks:
        logger.warning("  %s: secciones=%d but chunk_section produced 0 chunks",
                        drug_name[:40], len(secciones))

    return all_chunks


def _load_nregistro_config(pairs_dir: Path = PAIRS_DIR) -> dict[str, dict]:
    """Load nregistro config saved during prospecto download (Step 3).

    Returns dict mapping drug name (upper) → {nregistro, pair_id, is_generic}.
    """
    config_path = pairs_dir / "drug_pairs_nregistro.json"
    if not config_path.exists():
        return {}

    try:
        with open(config_path, encoding="utf-8") as f:
            pairs = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    lookup = {}
    for pair in pairs:
        pid = pair.get("pair_id", "")
        for role in ("branded", "generic"):
            drug = pair.get(role, {})
            name = drug.get("name", "").upper()
            nreg = drug.get("nregistro", "")
            if name and nreg:
                lookup[name] = {
                    "nregistro": nreg,
                    "pair_id": pid,
                    "is_generic": drug.get("is_generic", role == "generic"),
                }
    return lookup


def _fetch_prospecto_from_cima(nregistro: str, drug_name: str,
                               prospectos_dir: Path = PROSPECTOS_DIR) -> dict | None:
    """Download a prospecto JSON from CIMA's segmented doc API and save to disk.

    Returns the parsed prospecto dict if successful, None otherwise.
    """
    from .cima_client import CIMAClient
    cima = CIMAClient(delay=0.5)

    try:
        doc = cima.get_doc_segmentado(nregistro, tipo_doc=2)
    except Exception as e:
        logger.warning("  CIMA docSegmentado failed for %s: %s", nregistro, e)
        return None

    if not doc:
        return None

    # Check for error responses
    if isinstance(doc, dict) and "error" in doc:
        logger.warning("  CIMA returned error for %s: %s", nregistro, doc["error"][:80])
        return None

    # Save to disk for future runs
    prospectos_dir.mkdir(parents=True, exist_ok=True)
    safe_name = drug_name[:40].replace("/", "_").replace(" ", "_")
    filepath = prospectos_dir / f"{nregistro}_{safe_name}.json"

    prospecto_data = {
        "nregistro": nregistro,
        "nombre": drug_name,
        "es_generico": "EFG" in drug_name.upper(),
        "prospecto_sections": doc,
    }

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(prospecto_data, f, ensure_ascii=False, indent=2)
        logger.info("  Saved prospecto: %s", filepath.name)
    except OSError as e:
        logger.warning("  Could not save prospecto: %s", e)

    return prospecto_data


def _validate_chunks(chunks: list[dict]) -> list[dict]:
    """Filter out chunks with empty or too-short text before embedding."""
    valid = []
    for c in chunks:
        text = c.get("text", "").strip()
        if len(text) < 20:  # Skip chunks shorter than ~6 tokens
            continue
        # Ensure text field is clean
        c["text"] = text
        valid.append(c)
    return valid


def _bootstrap_index(
    queries: list[dict],
    store: MeQAVectorStore,
    prospectos_dir: Path = PROSPECTOS_DIR,
    leaflets_dir: Path = LEAFLETS_DIR,
    progress_callback=None,
) -> int:
    """Auto-bootstrap the vector store index from available data sources.

    Strategy:
    1. Index ALL available prospecto JSON files from data/prospectos/
    2. Index any per-section HTML files from data/leaflets/
    3. For drugs still missing, fetch prospectos from CIMA API
       (uses nregistro config from Step 3 if available, otherwise searches CIMA)

    Returns number of chunks indexed.
    """
    from .leaflet_chunker import chunk_drug_leaflet

    all_chunks: list[dict] = []
    indexed_nregistros: set[str] = set()
    indexed_drug_names: set[str] = set()

    # Build pair_id lookup from queries
    pair_lookup: dict[str, str] = {}
    for q in queries:
        pid = q.get("pair_id", "")
        dn = q.get("drug_name", "")
        if dn:
            pair_lookup[dn.upper()] = pid
        for name_list_key in ("brand_names", "generic_names"):
            for n in q.get(name_list_key, []):
                n_str = n.get("nombre", n) if isinstance(n, dict) else n
                if n_str:
                    pair_lookup[n_str.upper()] = pid

    # Load nregistro config from Step 3 (if available)
    nregistro_config = _load_nregistro_config()
    if nregistro_config:
        logger.info("Loaded nregistro config with %d drugs", len(nregistro_config))

    # ── Step 1: Bulk-index ALL prospecto JSONs on disk ──
    all_prospectos = _load_all_prospectos(prospectos_dir)
    logger.info("Step 1: Found %d prospecto JSON files in %s", len(all_prospectos), prospectos_dir)

    for prospecto in all_prospectos:
        nreg = prospecto.get("nregistro", "")
        nombre = prospecto.get("nombre", "")
        es_generico = prospecto.get("es_generico", False)

        # Determine pair_id — try direct match, then prefix match
        p_id = pair_lookup.get(nombre.upper(), "")
        if not p_id:
            # Check nregistro config
            cfg = nregistro_config.get(nombre.upper(), {})
            p_id = cfg.get("pair_id", "")
        if not p_id:
            for key, pid in pair_lookup.items():
                if (nombre.upper().startswith(key + " ")
                        or key.startswith(nombre.upper() + " ")
                        or nombre.upper().split()[0] == key.split()[0]):
                    p_id = pid
                    break

        chunks = _chunk_prospecto_json(prospecto, pair_id=p_id, is_generic=es_generico)
        if chunks:
            all_chunks.extend(chunks)
            indexed_nregistros.add(nreg)
            indexed_drug_names.add(nombre.upper())
            logger.info("  ✓ %s: %d chunks", nombre[:40], len(chunks))
        else:
            logger.warning("  ✗ %s: no chunks produced", nombre[:40])

    logger.info("Step 1 complete: %d chunks from %d prospectos",
                 len(all_chunks), len(indexed_nregistros))

    # ── Step 2: Index any per-section HTML leaflets ──
    html_chunk_count = 0
    if leaflets_dir.exists():
        for pair_dir in leaflets_dir.iterdir():
            if not pair_dir.is_dir():
                continue
            html_files = list(pair_dir.glob("*_section_*.html"))
            if not html_files:
                continue
            nregistro = html_files[0].stem.split("_section_")[0]
            if nregistro in indexed_nregistros:
                continue
            chunks = chunk_drug_leaflet(
                nregistro=nregistro,
                drug_name=pair_dir.name,
                pair_id=pair_dir.name,
                is_generic=False,
                leaflets_dir=leaflets_dir,
            )
            if chunks:
                all_chunks.extend(chunks)
                indexed_nregistros.add(nregistro)
                html_chunk_count += len(chunks)

    logger.info("Step 2 complete: %d chunks from HTML leaflets (%d total)",
                 html_chunk_count, len(all_chunks))

    # ── Step 3: For drugs not yet covered, fetch from CIMA API ──
    drugs = _extract_drugs_from_queries(queries)
    logger.info("Step 3: %d unique drugs from queries, fetching missing from CIMA...", len(drugs))

    fetched_count = 0
    for drug_info in drugs:
        drug_name = drug_info["drug_name"]
        pair_id = drug_info["pair_id"]
        is_generic = drug_info["is_generic"]

        # Skip if already indexed by name match
        if drug_name.upper() in indexed_drug_names:
            continue
        # Also check partial match
        if any(drug_name.upper().split()[0] == dn.split()[0]
               for dn in indexed_drug_names if dn):
            continue

        # Try nregistro config first (from Step 3 download)
        nregistro = None
        cfg = nregistro_config.get(drug_name.upper(), {})
        if cfg:
            nregistro = cfg.get("nregistro")
            logger.info("  %s: nregistro=%s from config", drug_name[:30], nregistro)

        # Otherwise search CIMA
        if not nregistro:
            logger.info("  %s: searching CIMA API...", drug_name[:40])
            nregistro = _lookup_nregistro_from_cima(drug_name)

        if not nregistro:
            logger.warning("  %s: could not find nregistro, skipping", drug_name[:40])
            continue

        if nregistro in indexed_nregistros:
            continue

        # Try fetching the segmented prospecto JSON (preferred)
        prospecto = _fetch_prospecto_from_cima(nregistro, drug_name, prospectos_dir)
        if prospecto:
            chunks = _chunk_prospecto_json(prospecto, pair_id=pair_id, is_generic=is_generic)
            if chunks:
                all_chunks.extend(chunks)
                indexed_nregistros.add(nregistro)
                indexed_drug_names.add(drug_name.upper())
                fetched_count += 1
                logger.info("  ✓ %s: %d chunks from CIMA JSON", drug_name[:30], len(chunks))
                continue

        # Fallback: fetch HTML leaflet sections
        from .cima_leaflet_fetcher import fetch_and_save_leaflet
        fetch_and_save_leaflet(
            nregistro=nregistro,
            drug_name=drug_name,
            pair_id=pair_id,
            leaflets_dir=leaflets_dir,
            delay=0.5,
            skip_existing=True,
        )
        chunks = chunk_drug_leaflet(
            nregistro=nregistro,
            drug_name=drug_name,
            pair_id=pair_id,
            is_generic=is_generic,
            leaflets_dir=leaflets_dir,
        )
        if chunks:
            all_chunks.extend(chunks)
            indexed_nregistros.add(nregistro)
            indexed_drug_names.add(drug_name.upper())
            fetched_count += 1
            logger.info("  ✓ %s: %d chunks from CIMA HTML", drug_name[:30], len(chunks))

    logger.info("Step 3 complete: fetched %d new drugs from CIMA (%d total chunks)",
                 fetched_count, len(all_chunks))

    # ── Validate and index all chunks ──
    if all_chunks:
        valid_chunks = _validate_chunks(all_chunks)
        logger.info("Validation: %d/%d chunks valid. Embedding via OpenAI...",
                     len(valid_chunks), len(all_chunks))

        if not valid_chunks:
            logger.error("All %d chunks were invalid (empty/too short)", len(all_chunks))
            return 0

        if progress_callback:
            progress_callback(0, 1, f"Embedding {len(valid_chunks)} chunks via OpenAI...")

        try:
            indexed = store.index_chunks(valid_chunks)
            logger.info("Successfully indexed %d chunks in vector store", indexed)
            return indexed
        except Exception as e:
            logger.error("Embedding/indexing failed: %s", e, exc_info=True)
            # Try with smaller batches as fallback
            logger.info("Retrying with individual batch embedding...")
            return _index_chunks_with_fallback(store, valid_chunks)

    logger.warning("No chunks produced during bootstrap — check that prospectos "
                   "were downloaded (Step 3) and contain valid section data")
    return 0


def _index_chunks_with_fallback(store: MeQAVectorStore, chunks: list[dict]) -> int:
    """Try indexing chunks in smaller batches to isolate failures."""
    batch_size = 10
    total_indexed = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            n = store.index_chunks(batch)
            total_indexed += n
        except Exception as e:
            logger.warning("Batch %d-%d failed: %s — trying individually",
                           i, i + len(batch), e)
            for j, chunk in enumerate(batch):
                try:
                    n = store.index_chunks([chunk])
                    total_indexed += n
                except Exception as e2:
                    logger.error("Chunk %d failed: text=%s... error=%s",
                                 i + j, chunk.get("text", "")[:50], e2)

    logger.info("Fallback indexing: %d/%d chunks indexed", total_indexed, len(chunks))
    return total_indexed


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def collect_all_responses(
    queries: list[dict],
    api_key: str,
    model: str = "gpt-4o-mini",
    prospectos_dir: Path | None = None,
    responses_dir: Path = RESPONSES_DIR,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
    top_k: int = 5,
    bm25_weight: float = 0.4,
    delay: float = 0.3,
    progress_callback=None,
    skip_existing: bool = True,
) -> list[dict]:
    """Run the full MeQA-adapted RAG pipeline for all queries.

    Self-bootstrapping: if vector store is empty, automatically fetches leaflets
    from CIMA API, chunks them, and indexes them before proceeding.

    Pipeline per query (following the 3 MeQA blocks):
      Block 1: normalise query → extract entities → predict sections
      Block 2: vector store retrieval filtered by pair_id + drug_name
      Block 3: LLM generation grounded in retrieved chunks
    """
    responses_dir.mkdir(parents=True, exist_ok=True)

    # Initialize in-memory vector store with OpenAI embeddings
    store = MeQAVectorStore(openai_api_key=api_key)
    doc_count = store.count

    # ── AUTO-BOOTSTRAP if store is empty ──
    if doc_count == 0:
        logger.info("Vector store is empty. Auto-bootstrapping from available data...")

        def bootstrap_progress(current, total, msg=""):
            if progress_callback:
                # Use negative progress to signal bootstrap phase
                progress_callback(0, len(queries))

        indexed = _bootstrap_index(
            queries=queries,
            store=store,
            prospectos_dir=prospectos_dir or PROSPECTOS_DIR,
            leaflets_dir=LEAFLETS_DIR,
            progress_callback=bootstrap_progress,
        )
        doc_count = store.count
        logger.info("Bootstrap complete: %d chunks indexed", doc_count)

    if doc_count == 0:
        logger.warning(
            "Vector store is still empty after bootstrap. "
            "Responses will be generated without RAG context."
        )

    # Initialize OpenAI client
    openai_client = OpenAI(api_key=api_key)

    # Process each query
    collected = []
    total = len(queries)

    for i, query in enumerate(queries):
        qid = query.get("query_id", f"Q{i+1:04d}")
        fpath = responses_dir / f"{qid}.json"

        if skip_existing and fpath.exists():
            if progress_callback:
                progress_callback(i + 1, total)
            continue

        # ── BLOCK 1: Question Processing ──
        query_text = query.get("query_text", "")
        normalised = normalise_query(query_text)
        entities = extract_entities(query)
        predicted_secs = predict_sections(query.get("query_category", ""))

        drug_name = query.get("drug_name", "")
        pair_id = query.get("pair_id", "")
        is_comparative = query.get("is_generic") == "comparative"

        # ── BLOCK 2: Retrieval ──
        retrieved = []
        context_str = ""

        if doc_count > 0:
            # For comparative queries: retrieve from both drugs (no drug_name filter)
            # For factual queries: filter by specific drug_name
            filter_drug = None if is_comparative else drug_name

            retrieved = retrieve_chunks(
                query_text=query_text,
                store=store,
                pair_id=pair_id,
                drug_name=filter_drug,
                top_k=top_k,
            )

            # ── BLOCK 3 (part 1): Context assembly ──
            context_str = _format_context(retrieved)

        # ── BLOCK 3 (part 2): Answer generation ──
        full_prompt = (
            f"System: {_SYSTEM_PROMPT}\n\n"
            f"User: {_USER_TEMPLATE.format(context=context_str, query=query_text) if context_str else query_text}"
        )
        prompt_hash = hashlib.md5(full_prompt.encode()).hexdigest()

        try:
            response_text = generate_response(
                query_text=query_text,
                context=context_str,
                client=openai_client,
                model=model,
                drug_name=drug_name,
                ingredient=entities.get("active_ingredient", ""),
            )
        except Exception as e:
            logger.error("Generation failed for %s: %s", qid, e)
            response_text = ""

        response_hash = hashlib.md5(response_text.encode()).hexdigest() if response_text else ""

        # Build retrieval details for logging
        retrieval_details = []
        sections_retrieved = set()
        for r in retrieved:
            meta = r.get("metadata", {})
            sections_retrieved.add(meta.get("section_number", 0))
            retrieval_details.append({
                "section_number": meta.get("section_number", 0),
                "section_name": meta.get("section_name", ""),
                "similarity": round(r.get("similarity", 0.0), 4),
                "chunk_index": meta.get("chunk_index", 0),
                "drug_name": meta.get("drug_name", ""),
            })

        # ── Save response ──
        response_data = {
            "query_id": qid,
            "pair_id": pair_id,
            "principio_activo": query.get("principio_activo", ""),
            "grupo_terapeutico": query.get("grupo_terapeutico", ""),
            "drug_name": drug_name,
            "is_generic": query.get("is_generic", ""),
            "query_type": query.get("query_type", ""),
            "query_category": query.get("query_category", ""),
            "query_text": query_text,
            "response_text": response_text,
            "response_word_count": len(response_text.split()) if response_text else 0,
            "collection_timestamp": datetime.now(timezone.utc).isoformat(),
            "meqa_metadata": {
                "model": model,
                "temperature": 0.0,
                "normalised_query": normalised,
                "entities": entities,
                "predicted_sections": predicted_secs,
                "predicted_section_names": [
                    LEAFLET_SECTIONS.get(s, "") for s in predicted_secs
                ],
                "chunks_retrieved": len(retrieved),
                "total_doc_chunks": doc_count,
                "context_available": bool(context_str),
                "context_length": len(context_str),
                "retrieval_method": "openai_embedding_cosine_numpy",
                "retrieval_details": retrieval_details,
                "sections_retrieved": sorted(sections_retrieved),
                "prompt_hash": prompt_hash,
                "response_hash": response_hash,
            },
            "notes": (
                f"MeQA-RAG (model={model}, temp=0.0, "
                f"retrieval=OpenAI-embedding-3-small, "
                f"top_k={top_k}, "
                f"sections={sorted(sections_retrieved)}, "
                f"chunks={len(retrieved)}, "
                f"ctx_len={len(context_str)})"
            ),
        }

        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(response_data, f, ensure_ascii=False, indent=2)

        collected.append(response_data)

        if progress_callback:
            progress_callback(i + 1, total)

        if delay > 0 and i < total - 1:
            time.sleep(delay)

    return collected
