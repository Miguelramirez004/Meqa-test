"""RAG engine for automated MeQA response collection.

Implements the MeQA RAG architecture:

  ┌─────────────┐
  │  CIMA        │   1. INGESTION: Load prospecto JSONs, extract text,
  │  Prospectos  │      clean HTML, split into semantic chunks
  └──────┬───────┘
         ▼
  ┌─────────────┐
  │  Chunking   │   2. CHUNKING: Recursive character splitting with
  │  + Metadata  │      overlap; each chunk keeps source drug metadata
  └──────┬───────┘
         ▼
  ┌─────────────┐
  │  Embedding  │   3. EMBEDDING: OpenAI text-embedding-3-small encodes
  │  Index      │      every chunk into a dense vector
  └──────┬───────┘
         ▼
  ┌─────────────┐   4. QUERY PROCESSING: Classify query intent, pre-filter
  │  Query      │      by drug name, embed the query
  │  Processor  ├─┐
  └─────────────┘ │
         ▼        │
  ┌─────────────┐ │ 5. RETRIEVAL: Cosine similarity search over the
  │  Retriever  │◀┘    pre-filtered chunk index → top-k relevant chunks
  └──────┬───────┘
         ▼
  ┌─────────────┐
  │  Context    │   6. CONTEXT ASSEMBLY: Deduplicate, rank by score,
  │  Assembly   │      format into prompt with source citations
  └──────┬───────┘
         ▼
  ┌─────────────┐
  │  LLM        │   7. GENERATION: System prompt (MeQA persona) +
  │  Generation │      assembled context + user query → response
  └──────┬───────┘
         ▼
  ┌─────────────┐
  │  Response   │   8. POST-PROCESSING: Validate non-empty, add metadata
  │  Formatting │      (model, timestamp, context used), save as JSON
  └─────────────┘
"""

import json
import math
import pickle
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from openai import OpenAI

from .config import PROSPECTOS_DIR, RESPONSES_DIR, DATA_DIR


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DOCUMENT INGESTION — Load and clean prospecto text
# ═══════════════════════════════════════════════════════════════════════════════

def _strip_html(text: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    clean = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", clean).strip()


def _extract_sections(obj, depth: int = 0) -> list[dict]:
    """Recursively extract titled sections from CIMA segmented documents.

    Returns a flat list of {title, content, depth} dicts so each section
    can be chunked independently while keeping its heading as metadata.
    """
    sections: list[dict] = []

    if isinstance(obj, str):
        text = _strip_html(obj)
        if text:
            sections.append({"title": "", "content": text, "depth": depth})
        return sections

    if isinstance(obj, list):
        for item in obj:
            sections.extend(_extract_sections(item, depth))
        return sections

    if isinstance(obj, dict):
        title = _strip_html(obj.get("titulo", obj.get("title", "")))
        content = obj.get("contenido", obj.get("content", ""))

        if content:
            text = _strip_html(content) if isinstance(content, str) else ""
            if text:
                sections.append({"title": title, "content": text, "depth": depth})
            elif isinstance(content, (list, dict)):
                sub = _extract_sections(content, depth + 1)
                if sub and title:
                    sub[0]["title"] = f"{title} > {sub[0]['title']}" if sub[0]["title"] else title
                sections.extend(sub)

        for key in ("secciones", "sections", "subsecciones"):
            if key in obj:
                sub = _extract_sections(obj[key], depth + 1)
                sections.extend(sub)

    return sections


def load_prospectos(prospectos_dir: Path = PROSPECTOS_DIR) -> list[dict]:
    """Load all prospecto JSONs and return structured document list.

    Each document: {nombre, nregistro, es_generico, sections: [{title, content}]}
    """
    documents: list[dict] = []
    for fp in sorted(prospectos_dir.glob("*.json")):
        if fp.name == ".gitkeep":
            continue
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)

        raw_sections = data.get("prospecto_sections", {})
        sections = _extract_sections(raw_sections)

        documents.append({
            "nombre": data.get("nombre", fp.stem),
            "nregistro": data.get("nregistro", ""),
            "es_generico": data.get("es_generico", False),
            "labtitular": data.get("labtitular", ""),
            "sections": sections,
            "filepath": str(fp),
        })
    return documents


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CHUNKING — Split sections into overlapping text chunks
# ═══════════════════════════════════════════════════════════════════════════════

def _recursive_split(text: str, max_chars: int, overlap: int,
                     separators: list[str] | None = None) -> list[str]:
    """Split text recursively by trying progressively finer separators."""
    if len(text) <= max_chars:
        return [text]

    if separators is None:
        separators = ["\n\n", "\n", ". ", ", ", " "]

    sep = separators[0] if separators else " "
    remaining_seps = separators[1:] if len(separators) > 1 else []

    parts = text.split(sep)
    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = f"{current}{sep}{part}" if current else part
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # If a single part is still too big, split with finer separator
            if len(part) > max_chars and remaining_seps:
                sub_chunks = _recursive_split(part, max_chars, overlap, remaining_seps)
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = part

    if current:
        chunks.append(current)

    # Add overlap: prepend the tail of the previous chunk
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            overlapped.append(prev_tail + " " + chunks[i])
        chunks = overlapped

    return chunks


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[dict]:
    """Split documents into chunks with metadata.

    Returns list of:
        {text, drug_name, nregistro, es_generico, section_title, chunk_idx}
    """
    all_chunks: list[dict] = []

    for doc in documents:
        drug_name = doc["nombre"]
        chunk_idx = 0

        for section in doc["sections"]:
            title = section.get("title", "")
            content = section.get("content", "")
            if not content.strip():
                continue

            # Prefix each chunk with its section heading for context
            prefixed = f"{title}: {content}" if title else content
            splits = _recursive_split(prefixed, chunk_size, chunk_overlap)

            for split_text in splits:
                all_chunks.append({
                    "text": split_text.strip(),
                    "drug_name": drug_name,
                    "drug_name_norm": _normalise(drug_name),
                    "nregistro": doc.get("nregistro", ""),
                    "es_generico": doc.get("es_generico", False),
                    "section_title": title,
                    "chunk_idx": chunk_idx,
                })
                chunk_idx += 1

    return all_chunks


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EMBEDDING & VECTOR INDEX — Encode chunks, store, and search
# ═══════════════════════════════════════════════════════════════════════════════

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536
EMBED_BATCH_SIZE = 100  # OpenAI allows up to 2048 inputs per call


def embed_texts(texts: list[str], client: OpenAI) -> np.ndarray:
    """Embed a list of texts using OpenAI embeddings API.

    Returns an (N, D) numpy array of normalised vectors.
    """
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        for item in resp.data:
            all_embeddings.append(item.embedding)

    matrix = np.array(all_embeddings, dtype=np.float32)
    # L2-normalise so dot product = cosine similarity
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class VectorIndex:
    """In-memory vector index with cosine-similarity search.

    Lightweight alternative to FAISS/Chroma — sufficient for the ~200-500
    chunks that come from 20-30 drug prospectos.
    """

    def __init__(self, chunks: list[dict], embeddings: np.ndarray):
        self.chunks = chunks
        self.embeddings = embeddings  # (N, D), L2-normalised

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        drug_filter: str | None = None,
    ) -> list[dict]:
        """Return top-k most similar chunks.

        Args:
            query_embedding: (D,) normalised vector.
            top_k: Number of results.
            drug_filter: If set, only search chunks whose drug_name_norm
                         contains this string (pre-filter by drug).

        Returns:
            List of {chunk, score} dicts, highest score first.
        """
        if drug_filter:
            drug_norm = _normalise(drug_filter)
            mask = np.array([
                _drug_matches(c["drug_name_norm"], drug_norm)
                for c in self.chunks
            ])
            if not mask.any():
                # Fall back to full search if no drug matches
                mask = np.ones(len(self.chunks), dtype=bool)
        else:
            mask = np.ones(len(self.chunks), dtype=bool)

        subset_embeddings = self.embeddings[mask]
        subset_indices = np.where(mask)[0]

        if len(subset_embeddings) == 0:
            return []

        # Cosine similarity via dot product (vectors are already normalised)
        scores = subset_embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            original_idx = subset_indices[idx]
            results.append({
                "chunk": self.chunks[original_idx],
                "score": float(scores[idx]),
            })
        return results

    @property
    def size(self) -> int:
        return len(self.chunks)

    def save(self, path: Path):
        """Persist index to disk so it doesn't need re-embedding."""
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "embeddings": self.embeddings}, f)

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["chunks"], data["embeddings"])


def build_index(
    prospectos_dir: Path,
    client: OpenAI,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    progress_callback=None,
) -> VectorIndex:
    """Full ingestion pipeline: load → chunk → embed → index."""
    documents = load_prospectos(prospectos_dir)
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)

    if not chunks:
        return VectorIndex([], np.zeros((0, EMBEDDING_DIMS), dtype=np.float32))

    texts = [c["text"] for c in chunks]

    if progress_callback:
        progress_callback("embedding", 0, len(texts))

    embeddings = embed_texts(texts, client)

    if progress_callback:
        progress_callback("embedding", len(texts), len(texts))

    return VectorIndex(chunks, embeddings)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. QUERY PROCESSING — Classify intent and map to drug
# ═══════════════════════════════════════════════════════════════════════════════

def _normalise(name: str) -> str:
    """Lower-case, collapse whitespace, strip trailing dots."""
    return re.sub(r"\s+", " ", name.strip().lower()).rstrip(".")


def _drug_matches(chunk_name: str, query_drug: str) -> bool:
    """Check if a chunk's drug name matches the query drug (fuzzy)."""
    if query_drug in chunk_name or chunk_name in query_drug:
        return True
    # Token overlap: at least 2 significant tokens match
    q_tokens = {t for t in query_drug.split() if len(t) > 2}
    c_tokens = {t for t in chunk_name.split() if len(t) > 2}
    return len(q_tokens & c_tokens) >= 2


# Section-hint keywords for query-category-aware retrieval boost
_CATEGORY_BOOST_TERMS: dict[str, str] = {
    "indication": "indicación para qué sirve tratamiento",
    "adverse_effects": "efectos adversos secundarios reacciones",
    "pregnancy": "embarazo lactancia fertilidad",
    "dosage": "dosis posología cómo tomar cantidad",
    "driving": "conducir conducción máquinas",
    "contraindications": "contraindicaciones no tome no debe",
    "storage": "conservación almacenar caducidad",
    "alcohol": "alcohol bebidas alcohólicas",
    "food": "comida alimentos comidas",
    "equivalence": "genérico bioequivalente mismo principio",
    "recommendation": "mejor diferencia comparar",
    "switching": "cambiar sustituir intercambiar",
}


def build_retrieval_query(query: dict) -> str:
    """Augment the raw query text with category-specific boost terms.

    This makes the embedding more likely to match the right prospecto section.
    """
    query_text = query.get("query_text", "")
    category = query.get("query_category", "")
    drug_name = query.get("drug_name", "")

    boost = _CATEGORY_BOOST_TERMS.get(category, "")
    return f"{drug_name} {query_text} {boost}".strip()


# ═══════════════════════════════════════════════════════════════════════════════
# 5 & 6. RETRIEVAL + CONTEXT ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════

def retrieve_context(
    query: dict,
    index: VectorIndex,
    client: OpenAI,
    top_k: int = 5,
    min_score: float = 0.25,
) -> list[dict]:
    """Retrieve the top-k relevant chunks for a query.

    Steps:
        1. Build augmented retrieval query
        2. Embed the query
        3. Pre-filter index by drug name
        4. Cosine similarity search
        5. Filter by minimum score threshold
    """
    retrieval_text = build_retrieval_query(query)
    query_emb = embed_texts([retrieval_text], client)[0]

    drug_name = query.get("drug_name", "")
    results = index.search(query_emb, top_k=top_k, drug_filter=drug_name)

    # Filter low-confidence results
    return [r for r in results if r["score"] >= min_score]


def assemble_context(results: list[dict], max_context_chars: int = 3000) -> str:
    """Format retrieved chunks into a prompt-ready context block.

    Deduplicates, orders by score, and includes source section titles.
    """
    if not results:
        return ""

    seen_texts: set[str] = set()
    parts: list[str] = []
    total_chars = 0

    for r in results:
        text = r["chunk"]["text"]
        # Simple dedup: skip near-identical chunks
        text_key = text[:100]
        if text_key in seen_texts:
            continue
        seen_texts.add(text_key)

        section = r["chunk"].get("section_title", "")
        drug = r["chunk"].get("drug_name", "")
        score = r["score"]
        header = f"[{drug} — {section}] (relevancia: {score:.2f})" if section else f"[{drug}] (relevancia: {score:.2f})"

        block = f"{header}\n{text}"
        if total_chars + len(block) > max_context_chars:
            # Add truncated final block
            remaining = max_context_chars - total_chars
            if remaining > 100:
                parts.append(block[:remaining] + "...")
            break
        parts.append(block)
        total_chars += len(block) + 2  # +2 for separators

    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. LLM GENERATION — MeQA persona system prompt + context + query
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
Eres MeQA, el asistente virtual de consulta de medicamentos de la Agencia \
Española de Medicamentos y Productos Sanitarios (AEMPS). Tu función es \
responder preguntas sobre medicamentos comercializados en España basándote \
en la información oficial de los prospectos aprobados por la AEMPS.

Instrucciones estrictas:
- Responde SIEMPRE en español.
- Basa tu respuesta EXCLUSIVAMENTE en la información del prospecto \
  proporcionada como contexto. No añadas información que no esté en el contexto.
- Si el contexto no contiene información suficiente para responder, indícalo \
  claramente y recomienda consultar el prospecto oficial completo.
- Sé preciso, conciso y profesional en tono farmacéutico.
- Cuando sea relevante, recomienda consultar al médico o farmacéutico.
- Incluye información sobre indicaciones, dosis, efectos adversos, \
  contraindicaciones o interacciones según corresponda a la pregunta.
- NO inventes información. Si no estás seguro, dilo explícitamente.
- Responde como si fueras el sistema MeQA oficial de la AEMPS.
"""

_USER_TEMPLATE_WITH_CONTEXT = """\
A continuación se presentan fragmentos relevantes del prospecto oficial \
del medicamento, recuperados de la base de datos CIMA de la AEMPS:

{context}

---

Pregunta del usuario: {query}

Responde basándote únicamente en la información del prospecto anterior.
"""

_USER_TEMPLATE_NO_CONTEXT = """\
Pregunta del usuario sobre el medicamento "{drug}": {query}

NOTA: No se dispone del prospecto oficial de este medicamento en la base de \
datos. Responde con tu conocimiento farmacéutico general sobre este principio \
activo, pero indica claramente que se debe consultar el prospecto oficial \
aprobado por la AEMPS para información definitiva.
"""


def generate_response(
    query: dict,
    context: str,
    client: OpenAI,
    model: str = "gpt-4o-mini",
) -> str:
    """Call the LLM with MeQA persona, context, and query."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# 8. RESPONSE POST-PROCESSING & BATCH ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

INDEX_CACHE_FILE = DATA_DIR / "vector_index.pkl"


def collect_all_responses(
    queries: list[dict],
    api_key: str,
    model: str = "gpt-4o-mini",
    prospectos_dir: Path = PROSPECTOS_DIR,
    responses_dir: Path = RESPONSES_DIR,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    top_k: int = 5,
    delay: float = 0.3,
    progress_callback=None,
    skip_existing: bool = True,
) -> list[dict]:
    """Run the full MeQA RAG pipeline for all queries.

    Pipeline per query:
        Query → classify + embed → retrieve chunks → assemble context
        → LLM generation → post-process → save JSON

    Args:
        queries: List of query dicts from query_battery.json.
        api_key: OpenAI API key.
        model: Chat model for generation.
        prospectos_dir: Directory with prospecto JSONs.
        responses_dir: Directory to save response JSONs.
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap characters between chunks.
        top_k: Number of chunks to retrieve per query.
        delay: Seconds between generation API calls.
        progress_callback: Optional callable(current, total) for UI progress.
        skip_existing: Skip queries that already have response files.

    Returns:
        List of newly collected response dicts.
    """
    client = OpenAI(api_key=api_key)
    responses_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage 1-3: Ingest, chunk, embed (or load cached index) ──
    has_prospectos = any(
        f.suffix == ".json" and f.name != ".gitkeep"
        for f in prospectos_dir.iterdir()
    ) if prospectos_dir.exists() else False

    index = None
    if has_prospectos:
        # Try loading cached index
        if INDEX_CACHE_FILE.exists():
            try:
                index = VectorIndex.load(INDEX_CACHE_FILE)
            except Exception:
                index = None

        if index is None:
            index = build_index(
                prospectos_dir, client,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            if index.size > 0:
                index.save(INDEX_CACHE_FILE)

    # ── Stage 4-8: Process each query through the RAG pipeline ──
    collected = []
    total = len(queries)

    for i, query in enumerate(queries):
        qid = query.get("query_id", f"Q{i+1:04d}")

        # Skip existing
        fpath = responses_dir / f"{qid}.json"
        if skip_existing and fpath.exists():
            if progress_callback:
                progress_callback(i + 1, total)
            continue

        # 4-5. Retrieve relevant context
        context_str = ""
        retrieval_results = []
        if index is not None and index.size > 0:
            retrieval_results = retrieve_context(
                query, index, client, top_k=top_k
            )
            # 6. Assemble context
            context_str = assemble_context(retrieval_results)

        # 7. Generate response
        try:
            response_text = generate_response(query, context_str, client, model)
        except Exception as e:
            response_text = ""
            query["error"] = str(e)

        # 8. Post-process and save
        chunks_used = len(retrieval_results)
        avg_score = (
            sum(r["score"] for r in retrieval_results) / chunks_used
            if chunks_used > 0 else 0.0
        )

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
            "rag_metadata": {
                "model": model,
                "chunks_retrieved": chunks_used,
                "avg_similarity_score": round(avg_score, 4),
                "context_available": bool(context_str),
                "embedding_model": EMBEDDING_MODEL,
                "chunk_size": chunk_size,
            },
            "notes": (
                f"RAG-generated (model={model}, "
                f"chunks={chunks_used}, avg_score={avg_score:.3f})"
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
