"""RAG engine simulating the MeQA architecture.

Based on: Santamaría, J. (2021). "Medicines Question Answering System, MeQA"
          AEMPS — arXiv:2111.02760

Real MeQA has three processing blocks:

  Block 1 — Question Processing
    Normalization → NER (medicine names, doses, pharma forms via CIMA/UMLS)
    → bi-LSTM section prediction (6 leaflet sections, sigmoid, 6 outputs)

  Block 2 — Leaflets & Sections Extraction
    TF-IDF leaflet selection → VSM/LSI additional-section extraction

  Block 3 — Answer Extraction
    Sentence extraction from predicted sections → dedup → add context
    → rank by relevance → main answer + additional information

Our simulation maps these blocks onto a modern LangChain RAG pipeline:

  Block 1 → query_normalise + extract_entities + predict_sections
  Block 2 → Hybrid retrieval: BM25 (sparse, ≈ TF-IDF) + FAISS (dense)
            with section-aware filtering
  Block 3 → Context assembly (extract, dedup, section headers)
            + LLM generation with extractive instructions
"""

import json
import os
import pickle
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever

from .config import PROSPECTOS_DIR, RESPONSES_DIR, DATA_DIR


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
# This simulates the bi-LSTM model trained on 13,989 Doctoralia questions
# that predicts which sections contain the answer (sigmoid over 6 outputs).
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

# Keywords for section identification in prospecto text
_SECTION_PATTERNS: dict[int, list[str]] = {
    1: ["qué es", "para qué se utiliza", "para qué sirve", "indicacion", "indicado"],
    2: ["antes de tomar", "antes de usar", "antes de que le administren",
        "no tome", "no use", "advertencia", "precaucion", "embarazo",
        "lactancia", "conduccion", "conducir", "otros medicamentos"],
    3: ["cómo tomar", "cómo usar", "dosis", "posología",
        "si toma más", "si olvidó", "si olvida"],
    4: ["efectos adversos", "efectos secundarios", "efectos no deseados",
        "reacciones adversas", "frecuentes", "poco frecuentes", "raros"],
    5: ["conservación", "conservar", "almacenar", "caducidad",
        "no utilice después", "desechar"],
    6: ["contenido del envase", "composición", "aspecto del producto",
        "información adicional", "titular de la autorización"],
}


def normalise_query(text: str) -> str:
    """MeQA Block 1 — Normalization module.

    Lowercase, strip accents, remove irrelevant punctuation, collapse whitespace.
    Mirrors the real MeQA normalization step.
    """
    text = text.lower().strip()
    # Remove accents (as the real MeQA does)
    nfkd = unicodedata.normalize("NFKD", text)
    text_no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Keep ñ (restore it since it's not an accent)
    text_no_accents = text_no_accents.replace("n\u0303", "ñ")
    # Remove irrelevant punctuation but keep ? and basic structure
    text_clean = re.sub(r"[^\w\s¿?.,]", " ", text_no_accents)
    return re.sub(r"\s+", " ", text_clean).strip()


def extract_entities(query: dict) -> dict:
    """MeQA Block 1 — Simplified NER module.

    Extracts medicine name, active ingredient, dose, and pharma form
    from the structured query metadata. The real MeQA uses maximum
    coincidence with n-grams + CIMA REST + SpaCy dependency parsing.
    """
    drug_name = query.get("drug_name", "")
    principio = query.get("principio_activo", "")

    # Extract dose pattern (e.g., "20 mg", "600 mg", "100 mcg")
    dose_match = re.search(r"(\d+(?:[.,]\d+)?\s*(?:mg|mcg|g|ml|ui))", drug_name, re.I)
    dose = dose_match.group(1) if dose_match else ""

    # Extract pharmaceutical form
    form_patterns = [
        "comprimidos", "cápsulas", "capsulas", "solución oral", "solucion oral",
        "sobres", "inyectable", "gel", "crema", "pomada", "gotas", "jarabe",
        "comprimidos recubiertos", "cápsulas duras", "capsulas duras",
        "comprimidos recubiertos con película", "polvo",
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
    """MeQA Block 1 — Section prediction (simulates bi-LSTM).

    The real MeQA uses a bi-LSTM with learned embeddings → 6 sigmoid
    outputs, trained on 13,989 annotated questions from Doctoralia.
    We approximate this with a deterministic mapping from query category
    to likely leaflet sections.

    Returns list of predicted section numbers (1-6).
    """
    return SECTION_PREDICTION.get(query_category, [1, 2, 4])


def classify_section(text: str) -> int:
    """Classify a text chunk into the most likely leaflet section number."""
    text_lower = text.lower()[:500]
    best_section = 0
    best_score = 0
    for sec_num, patterns in _SECTION_PATTERNS.items():
        score = sum(1 for p in patterns if p in text_lower)
        if score > best_score:
            best_score = score
            best_section = sec_num
    return best_section


# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENT PROCESSING — Load prospectos, extract text, build LangChain Documents
# ═══════════════════════════════════════════════════════════════════════════════

def _strip_html(text: str) -> str:
    clean = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", clean).strip()


def _walk_and_extract(obj, depth: int = 0) -> list[dict]:
    """Recursively extract {title, content} from CIMA segmented documents."""
    results: list[dict] = []

    if isinstance(obj, str):
        text = _strip_html(obj)
        if text:
            results.append({"title": "", "content": text, "depth": depth})
        return results

    if isinstance(obj, list):
        for item in obj:
            results.extend(_walk_and_extract(item, depth))
        return results

    if isinstance(obj, dict):
        title = _strip_html(obj.get("titulo", obj.get("title", "")))
        content = obj.get("contenido", obj.get("content", ""))

        if content:
            if isinstance(content, str):
                text = _strip_html(content)
                if text:
                    results.append({"title": title, "content": text, "depth": depth})
            elif isinstance(content, (list, dict)):
                sub = _walk_and_extract(content, depth + 1)
                if sub and title:
                    sub[0]["title"] = (
                        f"{title} > {sub[0]['title']}" if sub[0]["title"] else title
                    )
                results.extend(sub)

        for key in ("secciones", "sections", "subsecciones"):
            if key in obj:
                results.extend(_walk_and_extract(obj[key], depth + 1))

    return results


def load_prospecto_documents(
    prospectos_dir: Path = PROSPECTOS_DIR,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[Document]:
    """Load prospectos, extract text, chunk, and return LangChain Documents.

    Each Document has metadata: drug_name, nregistro, es_generico,
    section_title, section_number (1-6).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", ", ", " "],
        length_function=len,
    )

    all_docs: list[Document] = []

    for fp in sorted(prospectos_dir.glob("*.json")):
        if fp.name == ".gitkeep":
            continue

        with open(fp, encoding="utf-8") as f:
            data = json.load(f)

        drug_name = data.get("nombre", fp.stem)
        nregistro = data.get("nregistro", "")
        es_generico = data.get("es_generico", False)

        raw_sections = data.get("prospecto_sections", {})
        extracted = _walk_and_extract(raw_sections)

        for section in extracted:
            title = section.get("title", "")
            content = section.get("content", "")
            if not content.strip():
                continue

            # Prefix text with section title for better retrieval
            full_text = f"{title}: {content}" if title else content
            section_num = classify_section(full_text)

            chunks = splitter.split_text(full_text)
            for chunk_text in chunks:
                doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "drug_name": drug_name,
                        "drug_name_lower": drug_name.lower(),
                        "nregistro": nregistro,
                        "es_generico": es_generico,
                        "section_title": title,
                        "section_number": section_num,
                    },
                )
                all_docs.append(doc)

    return all_docs


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 2 — LEAFLETS & SECTIONS EXTRACTION
# (Hybrid retrieval: BM25 sparse ≈ TF-IDF + FAISS dense + section filtering)
# ═══════════════════════════════════════════════════════════════════════════════

FAISS_INDEX_DIR = DATA_DIR / "faiss_index"


def build_hybrid_retriever(
    documents: list[Document],
    api_key: str,
    top_k: int = 8,
    bm25_weight: float = 0.4,
    dense_weight: float = 0.6,
) -> tuple:
    """Build BM25 + FAISS ensemble retriever.

    This simulates MeQA's Block 2:
    - BM25 ≈ TF-IDF-based leaflet/section matching (keyword precision)
    - FAISS ≈ dense semantic matching (captures paraphrases)
    - Ensemble combines both (like MeQA combining TF-IDF + VSM/LSI)

    Returns (ensemble_retriever, faiss_store).
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
    )

    # Dense retriever: FAISS
    faiss_store = FAISS.from_documents(documents, embeddings)
    dense_retriever = faiss_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    # Sparse retriever: BM25 (simulates TF-IDF from real MeQA)
    bm25_retriever = BM25Retriever.from_documents(documents, k=top_k)

    # Ensemble (like MeQA combining TF-IDF + VSM/LSI)
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[bm25_weight, dense_weight],
    )

    return ensemble, faiss_store


def filter_by_drug_and_section(
    results: list[Document],
    drug_name: str,
    predicted_sections: list[int],
    max_results: int = 6,
) -> list[Document]:
    """MeQA Block 2 — Filter results by drug name and predicted sections.

    Simulates:
    - Leaflet selection (filter by medicine name/dose/form)
    - Section extraction (prioritise bi-LSTM predicted sections,
      keep VSM/LSI additional sections at lower priority)
    """
    drug_lower = drug_name.lower()
    drug_tokens = {t for t in drug_lower.split() if len(t) > 2}

    # Score each result by drug match + section match
    scored: list[tuple[float, Document]] = []
    for doc in results:
        doc_drug = doc.metadata.get("drug_name_lower", "")
        doc_section = doc.metadata.get("section_number", 0)

        # Drug match score
        drug_score = 0.0
        if drug_lower in doc_drug or doc_drug in drug_lower:
            drug_score = 1.0
        else:
            doc_tokens = {t for t in doc_drug.split() if len(t) > 2}
            overlap = len(drug_tokens & doc_tokens)
            if overlap >= 2:
                drug_score = 0.7
            elif overlap >= 1:
                drug_score = 0.3

        # Section match score (bi-LSTM predicted sections get higher priority)
        section_score = 0.0
        if doc_section in predicted_sections:
            section_score = 1.0   # Main answer section (like bi-LSTM prediction)
        elif doc_section > 0:
            section_score = 0.4   # Additional info (like VSM/LSI sections)

        total = drug_score * 0.6 + section_score * 0.4
        scored.append((total, doc))

    # Sort by score descending, take top results
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:max_results]]


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 3 — ANSWER EXTRACTION
# (Context assembly + LLM generation with extractive focus)
# ═══════════════════════════════════════════════════════════════════════════════

def assemble_context(
    documents: list[Document],
    max_chars: int = 3500,
) -> str:
    """MeQA Block 3 — Assemble answer context from retrieved passages.

    Simulates MeQA's answer extraction:
    - Extract sentences from selected leaflets for predicted sections
    - Remove duplicate sentences
    - Add context (section headers) for understandability
    - Group by sections
    """
    if not documents:
        return ""

    # Deduplicate by content prefix
    seen: set[str] = set()
    unique_docs: list[Document] = []
    for doc in documents:
        key = doc.page_content[:120]
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    # Group by section and format with headers (like MeQA adds context)
    parts: list[str] = []
    total_chars = 0

    for doc in unique_docs:
        section_title = doc.metadata.get("section_title", "")
        drug = doc.metadata.get("drug_name", "")
        sec_num = doc.metadata.get("section_number", 0)
        sec_label = LEAFLET_SECTIONS.get(sec_num, "")

        # Build header with section context (MeQA adds context to sentences)
        if sec_label:
            header = f"[{drug} — Sección {sec_num}: {sec_label}]"
        elif section_title:
            header = f"[{drug} — {section_title}]"
        else:
            header = f"[{drug}]"

        block = f"{header}\n{doc.page_content}"
        if total_chars + len(block) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 100:
                parts.append(block[:remaining] + "...")
            break
        parts.append(block)
        total_chars += len(block) + 2

    return "\n\n".join(parts)


# ── LLM Generation ──

_SYSTEM_PROMPT = """\
Eres MeQA, el sistema de consulta de medicamentos de la Agencia Española de \
Medicamentos y Productos Sanitarios (AEMPS). Tu función es responder preguntas \
sobre medicamentos para uso humano basándote en la información de los prospectos \
oficiales aprobados por la AEMPS.

Siguiendo la arquitectura del sistema MeQA real:

1. RESPONDE basándote EXCLUSIVAMENTE en los fragmentos del prospecto \
   proporcionados como contexto. No añadas información externa.
2. EXTRAE la información relevante del prospecto — tu respuesta debe ser \
   lo más fiel posible al texto oficial del prospecto.
3. Si la información del contexto no es suficiente, indica: "Esta información \
   no se encuentra en el prospecto consultado."
4. AÑADE contexto de sección cuando sea útil (p.ej., "Según la sección de \
   contraindicaciones del prospecto...").
5. Responde SIEMPRE en español, de forma precisa y profesional.
6. Cuando sea pertinente, recomienda consultar al médico o farmacéutico.
7. ELIMINA información duplicada en tu respuesta.
"""

_USER_TEMPLATE_WITH_CONTEXT = """\
Se han recuperado los siguientes fragmentos relevantes del prospecto oficial \
del medicamento, extraídos de la base de datos CIMA de la AEMPS:

{context}

---

Pregunta: {query}

Responde basándote ÚNICAMENTE en la información del prospecto anterior. \
Extrae y cita la información relevante de las secciones proporcionadas.
"""

_USER_TEMPLATE_NO_CONTEXT = """\
Pregunta sobre el medicamento "{drug}" ({ingredient}): {query}

NOTA: No se ha encontrado el prospecto de este medicamento en la base de datos. \
Responde con tu conocimiento farmacéutico general pero indica claramente que \
se debe consultar el prospecto oficial aprobado por la AEMPS.
"""


def generate_response(
    query: dict,
    context: str,
    client: ChatOpenAI,
) -> str:
    """MeQA Block 3 — Generate response with extractive focus."""
    query_text = query.get("query_text", "")
    drug_name = query.get("drug_name", "")
    ingredient = query.get("principio_activo", "")

    if context.strip():
        user_msg = _USER_TEMPLATE_WITH_CONTEXT.format(
            context=context, query=query_text
        )
    else:
        user_msg = _USER_TEMPLATE_NO_CONTEXT.format(
            drug=drug_name, ingredient=ingredient, query=query_text
        )

    messages = [
        ("system", _SYSTEM_PROMPT),
        ("human", user_msg),
    ]
    resp = client.invoke(messages)
    return resp.content.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def collect_all_responses(
    queries: list[dict],
    api_key: str,
    model: str = "gpt-4o-mini",
    prospectos_dir: Path = PROSPECTOS_DIR,
    responses_dir: Path = RESPONSES_DIR,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    top_k: int = 8,
    bm25_weight: float = 0.4,
    delay: float = 0.3,
    progress_callback=None,
    skip_existing: bool = True,
) -> list[dict]:
    """Run the full MeQA-simulated RAG pipeline for all queries.

    Pipeline per query (following the 3 MeQA blocks):
      Block 1: normalise query → extract entities → predict sections
      Block 2: hybrid retrieve (BM25 + FAISS) → filter by drug + sections
      Block 3: assemble context → LLM generation → save response

    Args:
        queries: List of query dicts from query_battery.json.
        api_key: OpenAI API key.
        model: Chat model for generation.
        prospectos_dir: Directory with prospecto JSONs.
        responses_dir: Directory to save response JSONs.
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap between chunks.
        top_k: Chunks to retrieve per query.
        bm25_weight: Weight for BM25 in ensemble (1-bm25_weight = dense weight).
        delay: Seconds between generation API calls.
        progress_callback: Optional callable(current, total).
        skip_existing: Skip queries with existing response files.

    Returns:
        List of newly collected response dicts.
    """
    responses_dir.mkdir(parents=True, exist_ok=True)

    # ── Build retriever (Blocks 1-2 preparation) ──
    has_prospectos = prospectos_dir.exists() and any(
        f.suffix == ".json" and f.name != ".gitkeep"
        for f in prospectos_dir.iterdir()
    )

    ensemble_retriever = None
    if has_prospectos:
        documents = load_prospecto_documents(
            prospectos_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        if documents:
            ensemble_retriever, _ = build_hybrid_retriever(
                documents,
                api_key=api_key,
                top_k=top_k,
                bm25_weight=bm25_weight,
                dense_weight=1.0 - bm25_weight,
            )

    # ── LLM client ──
    llm = ChatOpenAI(
        model=model,
        temperature=0.3,
        max_tokens=800,
        openai_api_key=api_key,
    )

    # ── Process each query through the 3 MeQA blocks ──
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
        normalised = normalise_query(query.get("query_text", ""))
        entities = extract_entities(query)
        predicted_secs = predict_sections(query.get("query_category", ""))

        # ── BLOCK 2: Leaflets & Sections Extraction ──
        context_str = ""
        chunks_used = 0
        if ensemble_retriever is not None:
            # Build retrieval query with drug name + normalised query
            retrieval_query = (
                f"{entities['medicine_name']} {normalised}"
            )
            raw_results = ensemble_retriever.invoke(retrieval_query)

            # Filter by drug name and predicted sections
            filtered = filter_by_drug_and_section(
                raw_results,
                drug_name=entities["medicine_name"],
                predicted_sections=predicted_secs,
                max_results=top_k,
            )
            chunks_used = len(filtered)

            # ── BLOCK 3 (part 1): Context assembly ──
            context_str = assemble_context(filtered)

        # ── BLOCK 3 (part 2): Answer generation ──
        try:
            response_text = generate_response(query, context_str, llm)
        except Exception as e:
            response_text = ""
            query["error"] = str(e)

        # ── Save response ──
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
            "meqa_metadata": {
                "model": model,
                "normalised_query": normalised,
                "entities": entities,
                "predicted_sections": predicted_secs,
                "predicted_section_names": [
                    LEAFLET_SECTIONS.get(s, "") for s in predicted_secs
                ],
                "chunks_retrieved": chunks_used,
                "context_available": bool(context_str),
                "retrieval_method": "hybrid_bm25_faiss",
                "bm25_weight": bm25_weight,
            },
            "notes": (
                f"MeQA-RAG (model={model}, "
                f"retrieval=BM25+FAISS, "
                f"sections={predicted_secs}, "
                f"chunks={chunks_used})"
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
