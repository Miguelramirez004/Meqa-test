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

from .config import DATA_DIR, RESPONSES_DIR
from .cima_leaflet_fetcher import LEAFLETS_DIR
from .vector_store import MeQAVectorStore, CHROMA_DIR

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
    """MeQA Block 1 — Normalization module.

    Lowercase, strip accents, remove irrelevant punctuation, collapse whitespace.
    Used for section prediction and entity extraction (NOT for retrieval).
    """
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
    """Retrieve relevant chunks from the vector store.

    For factual queries: filter by pair_id AND drug_name.
    For comparative queries: filter by pair_id only (drug_name=None).

    Returns list of result dicts with 'text', 'metadata', 'similarity'.
    """
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
    """Format retrieved chunks as context for the LLM prompt.

    Each chunk already has its section header prepended (from chunker),
    so the LLM knows which section the information comes from.
    This replicates MEQa's "context addition" step.
    """
    if not results:
        return ""
    # Join chunks with --- separators
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
    """Generate a response using the LLM with retrieved context.

    Uses temperature=0.0 for reproducibility.
    """
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
# PIPELINE ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def _load_drug_pairs_config() -> list[dict]:
    """Load drug pairs config with nregistro from disk, or return empty list."""
    config_path = DATA_DIR / "drug_pairs" / "drug_pairs_nregistro.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    return []


def _find_drug_in_pairs(drug_name: str, pair_id: str,
                        drug_pairs: list[dict]) -> dict | None:
    """Find a drug's config (with nregistro) from the pairs config."""
    for pair in drug_pairs:
        if pair["pair_id"] != pair_id:
            continue
        for role in ("branded", "generic"):
            drug = pair.get(role, {})
            if drug.get("name", "").upper() == drug_name.upper():
                return drug
    return None


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

    Pipeline per query (following the 3 MeQA blocks):
      Block 1: normalise query → extract entities → predict sections
      Block 2: ChromaDB retrieval filtered by pair_id + drug_name
      Block 3: LLM generation grounded in retrieved chunks

    Args:
        queries: List of query dicts from query_battery.json.
        api_key: OpenAI API key.
        model: Chat model for generation.
        prospectos_dir: Unused (kept for backward compatibility with app.py).
        responses_dir: Directory to save response JSONs.
        chunk_size: Unused (kept for backward compatibility).
        chunk_overlap: Unused (kept for backward compatibility).
        top_k: Chunks to retrieve per query (default 5).
        bm25_weight: Unused (kept for backward compatibility).
        delay: Seconds between generation API calls.
        progress_callback: Optional callable(current, total).
        skip_existing: Skip queries with existing response files.

    Returns:
        List of newly collected response dicts.
    """
    responses_dir.mkdir(parents=True, exist_ok=True)

    # Load the drug pairs config with nregistro
    drug_pairs = _load_drug_pairs_config()

    # Initialize ChromaDB vector store
    store = MeQAVectorStore(chroma_dir=CHROMA_DIR)
    doc_count = store.count

    if doc_count == 0:
        logger.warning(
            "ChromaDB is empty. Run leaflet fetch + index first. "
            "The pipeline will generate responses without RAG context."
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
