"""RAG engine replicating the MeQA architecture with modern components.

Based on: Santamaría, J. (2021). "Medicines Question Answering System, MeQA"
          AEMPS — arXiv:2111.02760

Adaptation of MEQa's 3-block pipeline to a modern RAG system:

  MEQa Block 1 (Question Processing)    → query normalisation + entity extraction
                                           + section prediction
  MEQa Block 2 (Leaflet & Section Ext.) → ChromaDB retrieval with metadata
                                           filtering (replaces TF-IDF + VSM/LSI)
  MEQa Block 3 (Answer Extraction)      → LLM generation grounded in retrieved
                                           chunks with section context

Key design:
  - Embedding: paraphrase-multilingual-MiniLM-L12-v2 (local, Spanish-native)
  - Vector store: ChromaDB (persistent, local)
  - Generation: gpt-4o-mini via OpenAI API (temp=0.0 for reproducibility)
  - Section metadata preserved end-to-end (fetch → chunk → embed → retrieve → prompt)
  - Identical pipeline for branded and generic drugs
  - Self-bootstrapping: auto-fetches leaflets from CIMA if ChromaDB is empty
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
from .config import LEAFLETS_DIR, CHROMA_DIR
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
# AUTO-BOOTSTRAP: fetch leaflets from CIMA, chunk, and index into ChromaDB
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_drugs_from_queries(queries: list[dict]) -> list[dict]:
    """Extract unique (pair_id, drug_name, is_generic) from queries.

    Skips comparative queries (those have compound drug names like 'A vs B').
    Returns a list of drug dicts suitable for the pipeline config.
    """
    seen = set()
    drugs = []

    for q in queries:
        is_generic = q.get("is_generic", "")
        # Skip comparative queries — they reference two drugs
        if is_generic == "comparative" or " vs " in q.get("drug_name", ""):
            continue

        pair_id = q.get("pair_id", "")
        drug_name = q.get("drug_name", "")
        key = (pair_id, drug_name)

        if key not in seen and drug_name:
            seen.add(key)
            drugs.append({
                "pair_id": pair_id,
                "drug_name": drug_name,
                "is_generic": bool(is_generic),
                "principio_activo": q.get("principio_activo", ""),
            })

    return drugs


def _lookup_nregistro_from_cima(drug_name: str) -> str | None:
    """Look up nregistro from CIMA by exact drug name search."""
    from .cima_client import CIMAClient
    cima = CIMAClient(delay=0.5)

    result = cima.search_medicamentos(nombre=drug_name)
    if not result or "resultados" not in result:
        logger.warning("CIMA lookup failed for: %s", drug_name[:50])
        return None

    # Try exact match first
    for med in result["resultados"]:
        if med.get("nombre", "").upper() == drug_name.upper():
            return med["nregistro"]

    # Fallback: first result
    if result["resultados"]:
        return result["resultados"][0]["nregistro"]

    return None


def _try_load_prospecto_json(drug_name: str,
                             prospectos_dir: Path = PROSPECTOS_DIR) -> dict | None:
    """Try to find a matching prospecto JSON in data/prospectos/.

    Returns the parsed JSON if found, None otherwise.
    """
    if not prospectos_dir.exists():
        return None

    # Search through existing files
    for fp in prospectos_dir.glob("*.json"):
        if fp.name == ".gitkeep":
            continue
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("nombre", "").upper() == drug_name.upper():
                return data
        except (json.JSONDecodeError, KeyError):
            continue

    return None


def _chunk_prospecto_json(
    data: dict,
    pair_id: str,
    is_generic: bool,
) -> list[dict]:
    """Chunk a prospecto JSON file (from CIMA segmented API) into section-aware chunks.

    This handles the existing data format from data/prospectos/*.json files.
    """
    from .leaflet_chunker import chunk_section, SECTION_NAMES

    drug_name = data.get("nombre", "")
    nregistro = data.get("nregistro", "")
    sections_data = data.get("prospecto_sections")

    if not sections_data:
        return []

    all_chunks = []

    # The prospecto_sections structure has nested secciones
    secciones = []
    if isinstance(sections_data, dict):
        secciones = sections_data.get("secciones", [])
    elif isinstance(sections_data, list):
        secciones = sections_data

    for sec in secciones:
        titulo = sec.get("titulo", "")
        contenido = sec.get("contenido", "")

        if not contenido:
            continue

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
            else:
                sec_num = 0

        if sec_num == 0:
            continue

        # contenido is HTML — pass it to chunk_section
        section_chunks = chunk_section(
            html=contenido,
            section_number=sec_num,
            drug_name=drug_name,
            nregistro=nregistro,
            pair_id=pair_id,
            is_generic=is_generic,
        )
        all_chunks.extend(section_chunks)

    return all_chunks


def _bootstrap_index(
    queries: list[dict],
    store: MeQAVectorStore,
    prospectos_dir: Path = PROSPECTOS_DIR,
    leaflets_dir: Path = LEAFLETS_DIR,
    progress_callback=None,
) -> int:
    """Auto-bootstrap the ChromaDB index from available data sources.

    Strategy (in order of preference):
    1. Use existing per-section HTML files in data/leaflets/
    2. Use existing prospecto JSON files in data/prospectos/
    3. Fetch from CIMA API: look up nregistro by name, fetch per-section HTML

    Returns number of chunks indexed.
    """
    from .leaflet_chunker import chunk_section, chunk_drug_leaflet

    drugs = _extract_drugs_from_queries(queries)
    if not drugs:
        logger.warning("No drugs found in queries to bootstrap from")
        return 0

    logger.info("Auto-bootstrapping index for %d unique drugs...", len(drugs))

    all_chunks = []
    total = len(drugs)

    for i, drug_info in enumerate(drugs):
        pair_id = drug_info["pair_id"]
        drug_name = drug_info["drug_name"]
        is_generic = drug_info["is_generic"]

        if progress_callback:
            progress_callback(i, total, f"Indexing: {drug_name[:40]}...")

        # --- Strategy 1: Check for existing per-section HTML ---
        pair_dir = leaflets_dir / pair_id
        html_files = list(pair_dir.glob("*_section_*.html")) if pair_dir.exists() else []

        if html_files:
            # Find the nregistro from filenames
            nregistro = html_files[0].stem.split("_section_")[0]
            chunks = chunk_drug_leaflet(
                nregistro=nregistro,
                drug_name=drug_name,
                pair_id=pair_id,
                is_generic=is_generic,
                leaflets_dir=leaflets_dir,
            )
            if chunks:
                all_chunks.extend(chunks)
                logger.info("  %s: %d chunks from HTML leaflets", drug_name[:30], len(chunks))
                continue

        # --- Strategy 2: Check for existing prospecto JSON ---
        prospecto = _try_load_prospecto_json(drug_name, prospectos_dir)
        if prospecto:
            chunks = _chunk_prospecto_json(prospecto, pair_id, is_generic)
            if chunks:
                all_chunks.extend(chunks)
                logger.info("  %s: %d chunks from prospecto JSON", drug_name[:30], len(chunks))
                continue

        # --- Strategy 3: Fetch from CIMA API ---
        logger.info("  %s: fetching from CIMA API...", drug_name[:40])
        nregistro = _lookup_nregistro_from_cima(drug_name)
        if not nregistro:
            logger.warning("  %s: could not find nregistro, skipping", drug_name[:40])
            continue

        # Fetch per-section HTML
        from .cima_leaflet_fetcher import fetch_and_save_leaflet
        fetch_results = fetch_and_save_leaflet(
            nregistro=nregistro,
            drug_name=drug_name,
            pair_id=pair_id,
            leaflets_dir=leaflets_dir,
            delay=0.5,
            skip_existing=True,
        )

        # Chunk the fetched HTML
        chunks = chunk_drug_leaflet(
            nregistro=nregistro,
            drug_name=drug_name,
            pair_id=pair_id,
            is_generic=is_generic,
            leaflets_dir=leaflets_dir,
        )
        if chunks:
            all_chunks.extend(chunks)
            logger.info("  %s: %d chunks from CIMA fetch", drug_name[:30], len(chunks))
        else:
            logger.warning("  %s: no chunks produced after CIMA fetch", drug_name[:30])

    # Index all chunks
    if all_chunks:
        logger.info("Indexing %d total chunks into ChromaDB...", len(all_chunks))
        if progress_callback:
            progress_callback(total, total, "Embedding and indexing chunks...")
        indexed = store.index_chunks(all_chunks)
        logger.info("Indexed %d chunks into ChromaDB", indexed)
        return indexed

    logger.warning("No chunks produced during bootstrap")
    return 0


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

    Self-bootstrapping: if ChromaDB is empty, automatically fetches leaflets
    from CIMA API, chunks them, and indexes them before proceeding.

    Pipeline per query (following the 3 MeQA blocks):
      Block 1: normalise query → extract entities → predict sections
      Block 2: ChromaDB retrieval filtered by pair_id + drug_name
      Block 3: LLM generation grounded in retrieved chunks
    """
    responses_dir.mkdir(parents=True, exist_ok=True)

    # Initialize ChromaDB vector store
    store = MeQAVectorStore(chroma_dir=CHROMA_DIR)
    doc_count = store.count

    # ── AUTO-BOOTSTRAP if ChromaDB is empty ──
    if doc_count == 0:
        logger.info("ChromaDB is empty. Auto-bootstrapping from CIMA...")

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
            "ChromaDB is still empty after bootstrap. "
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
                "retrieval_method": "chromadb_cosine_multilingual_minilm",
                "retrieval_details": retrieval_details,
                "sections_retrieved": sorted(sections_retrieved),
                "prompt_hash": prompt_hash,
                "response_hash": response_hash,
            },
            "notes": (
                f"MeQA-RAG (model={model}, temp=0.0, "
                f"retrieval=ChromaDB+MiniLM-L12-v2, "
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
