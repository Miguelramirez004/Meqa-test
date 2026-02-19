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
import logging
import os
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

from .config import PROSPECTOS_DIR, RESPONSES_DIR, DATA_DIR

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
    1: ["qué es", "para qué se utiliza", "para qué sirve", "indicacion",
        "indicado", "grupo de medicamentos", "grupo farmacoterapéutico",
        "pertenece a un grupo"],
    2: ["antes de tomar", "antes de usar", "antes de que le administren",
        "no tome", "no use", "advertencia", "precaucion", "embarazo",
        "lactancia", "conduccion", "conducir", "otros medicamentos",
        "qué necesita saber", "tenga especial cuidado", "interaccion"],
    3: ["cómo tomar", "cómo usar", "dosis", "posología", "como tomar",
        "si toma más", "si olvidó", "si olvida", "como usar",
        "siga exactamente", "modo de empleo", "vía oral", "vía de administración"],
    4: ["efectos adversos", "efectos secundarios", "efectos no deseados",
        "reacciones adversas", "frecuentes", "poco frecuentes", "raros",
        "muy raros", "posibles efectos"],
    5: ["conservación", "conservar", "almacenar", "caducidad",
        "no utilice después", "desechar", "cómo conservar", "como conservar"],
    6: ["contenido del envase", "composición", "aspecto del producto",
        "información adicional", "titular de la autorización",
        "excipientes", "responsable de la fabricación"],
}

# Number-based section detection from CIMA titles (e.g., "1.", "2.", etc.)
_SECTION_NUMBER_RE = re.compile(r"^\s*(\d)\s*[.\-–—:]")


def normalise_query(text: str) -> str:
    """MeQA Block 1 — Normalization module.

    Lowercase, strip accents, remove irrelevant punctuation, collapse whitespace.
    Mirrors the real MeQA normalization step.
    Used for section prediction and entity extraction (NOT for retrieval).
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


def classify_section(title: str, content: str) -> int:
    """Classify a text section into the most likely leaflet section number.

    Checks the title first (most reliable), then falls back to content patterns.
    Also detects section numbers from CIMA-style numbered titles.
    """
    # First: try to detect section number from title (e.g., "1. Qué es...")
    if title:
        m = _SECTION_NUMBER_RE.match(title)
        if m:
            num = int(m.group(1))
            if 1 <= num <= 6:
                return num

    # Second: check title text against keyword patterns
    title_lower = title.lower() if title else ""
    if title_lower:
        best_section = 0
        best_score = 0
        for sec_num, patterns in _SECTION_PATTERNS.items():
            score = sum(1 for p in patterns if p in title_lower)
            if score > best_score:
                best_score = score
                best_section = sec_num
        if best_section > 0:
            return best_section

    # Third: check content text against keyword patterns
    content_lower = content.lower()[:1000] if content else ""
    best_section = 0
    best_score = 0
    for sec_num, patterns in _SECTION_PATTERNS.items():
        score = sum(1 for p in patterns if p in content_lower)
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
    """Recursively extract {title, content} from CIMA segmented documents.

    Handles various CIMA JSON structures:
    - Top-level: {titulo, contenido, secciones}
    - Nested sections with subsecciones
    - Plain strings and lists
    """
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
        elif title and not content:
            # Some sections have title but no direct content — just record the title
            # so child sections can inherit it
            pass

        for key in ("secciones", "sections", "subsecciones"):
            if key in obj:
                child_results = _walk_and_extract(obj[key], depth + 1)
                # Prepend parent title to children if they lack context
                if title:
                    for cr in child_results:
                        if cr["title"] and title not in cr["title"]:
                            cr["title"] = f"{title} > {cr['title']}"
                        elif not cr["title"]:
                            cr["title"] = title
                results.extend(child_results)

    return results


def _extract_principio_from_name(nombre: str) -> str:
    """Try to extract the active ingredient from a drug's official name.

    Generic names typically start with the active ingredient:
      'OMEPRAZOL CINFA 20 MG CAPSULAS...' → 'omeprazol'
    Brand names don't contain the ingredient directly:
      'LOSEC 20 MG CAPSULAS...' → ''
    """
    # Common patterns: the first word is often the active ingredient for generics
    # We return it lowercased for matching purposes
    words = nombre.lower().split()
    if not words:
        return ""
    # If it looks like a generic (has EFG), the first word is likely the ingredient
    if "efg" in (w.lower() for w in words):
        return words[0]
    return ""


def load_prospecto_documents(
    prospectos_dir: Path = PROSPECTOS_DIR,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> list[Document]:
    """Load prospectos, extract text, chunk, and return LangChain Documents.

    Each Document has metadata: drug_name, nregistro, es_generico,
    section_title, section_number (1-6), principio_activo.
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

        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load %s: %s", fp, e)
            continue

        drug_name = data.get("nombre", fp.stem)
        nregistro = data.get("nregistro", "")
        es_generico = data.get("es_generico", False)
        principio = _extract_principio_from_name(drug_name)

        # Try multiple keys for the prospecto content
        raw_sections = (
            data.get("prospecto_sections")
            or data.get("secciones")
            or data.get("sections")
            or data
        )
        extracted = _walk_and_extract(raw_sections)

        if not extracted:
            # Fallback: try to extract text from any string values in the JSON
            logger.warning("No sections extracted from %s, trying fallback", fp.name)
            for key, val in data.items():
                if isinstance(val, str) and len(val) > 100:
                    text = _strip_html(val)
                    if text:
                        extracted.append({"title": key, "content": text, "depth": 0})

        for section in extracted:
            title = section.get("title", "")
            content = section.get("content", "")
            if not content.strip():
                continue

            # Prefix text with section title for better retrieval
            full_text = f"{title}: {content}" if title else content
            section_num = classify_section(title, content)

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
                        "principio_activo": principio,
                    },
                )
                all_docs.append(doc)

    logger.info("Loaded %d chunks from %s", len(all_docs), prospectos_dir)
    return all_docs


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 2 — LEAFLETS & SECTIONS EXTRACTION
# (Hybrid retrieval: BM25 sparse ≈ TF-IDF + FAISS dense + section filtering)
# ═══════════════════════════════════════════════════════════════════════════════

FAISS_INDEX_DIR = DATA_DIR / "faiss_index"


def _reciprocal_rank_fusion(
    results_lists: list[list[Document]],
    weights: list[float],
    k: int = 60,
) -> list[Document]:
    """Combine multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score for document d = Σ weight_i / (k + rank_i)
    This is the standard approach used by many production RAG systems.
    """
    doc_scores: dict[str, tuple[float, Document]] = {}

    for results, weight in zip(results_lists, weights):
        for rank, doc in enumerate(results):
            # Use first 300 chars of content as dedup key
            key = doc.page_content[:300]
            current_score = doc_scores.get(key, (0.0, doc))[0]
            rrf_score = weight / (k + rank + 1)
            doc_scores[key] = (current_score + rrf_score, doc)

    sorted_results = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in sorted_results]


def build_hybrid_retriever(
    documents: list[Document],
    api_key: str,
    top_k: int = 12,
    bm25_weight: float = 0.4,
    dense_weight: float = 0.6,
) -> tuple:
    """Build BM25 + FAISS retriever components.

    This simulates MeQA's Block 2:
    - BM25 ≈ TF-IDF-based leaflet/section matching (keyword precision)
    - FAISS ≈ dense semantic matching (captures paraphrases)

    Returns (bm25_retriever, faiss_store, weights).
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
    )

    # Dense retriever: FAISS
    faiss_store = FAISS.from_documents(documents, embeddings)

    # Sparse retriever: BM25 (simulates TF-IDF from real MeQA)
    bm25_retriever = BM25Retriever.from_documents(documents, k=top_k)

    return bm25_retriever, faiss_store, [bm25_weight, dense_weight]


def hybrid_retrieve(
    query: str,
    bm25_retriever: BM25Retriever,
    faiss_store: FAISS,
    weights: list[float],
    top_k: int = 12,
) -> list[Document]:
    """Run hybrid retrieval using BM25 + FAISS with Reciprocal Rank Fusion.

    This replaces the EnsembleRetriever to avoid import issues.
    """
    try:
        bm25_results = bm25_retriever.invoke(query)
    except Exception as e:
        logger.warning("BM25 retrieval failed: %s", e)
        bm25_results = []

    try:
        faiss_results = faiss_store.similarity_search(query, k=top_k)
    except Exception as e:
        logger.warning("FAISS retrieval failed: %s", e)
        faiss_results = []

    if not bm25_results and not faiss_results:
        return []

    combined = _reciprocal_rank_fusion(
        [bm25_results, faiss_results],
        weights,
    )
    return combined[:top_k * 2]  # Return extra for filtering


def filter_by_drug_and_section(
    results: list[Document],
    drug_name: str,
    principio_activo: str,
    predicted_sections: list[int],
    max_results: int = 10,
) -> list[Document]:
    """MeQA Block 2 — Filter results by drug name and predicted sections.

    Simulates:
    - Leaflet selection (filter by medicine name/active ingredient)
    - Section extraction (prioritise bi-LSTM predicted sections,
      keep VSM/LSI additional sections at lower priority)

    Uses both drug_name AND principio_activo for robust matching.
    """
    drug_lower = drug_name.lower()
    principio_lower = principio_activo.lower().strip() if principio_activo else ""

    # Extract the core drug tokens (skip very short words and generic terms)
    _skip_words = {
        "mg", "mcg", "ml", "efg", "con", "de", "del", "las", "los",
        "comprimidos", "capsulas", "cápsulas", "duras", "recubiertos",
        "película", "pelicula", "polvo", "solución", "solucion", "oral",
        "gastrorresistentes", "microgramos",
    }
    drug_tokens = {
        t for t in drug_lower.split()
        if len(t) > 2 and t not in _skip_words
    }
    # Add principio_activo tokens
    principio_tokens = set()
    if principio_lower:
        principio_tokens = {
            t for t in principio_lower.split()
            if len(t) > 2 and t not in _skip_words and not re.match(r"^\d+", t)
        }

    scored: list[tuple[float, Document]] = []
    for doc in results:
        doc_drug = doc.metadata.get("drug_name_lower", "")
        doc_section = doc.metadata.get("section_number", 0)
        doc_principio = doc.metadata.get("principio_activo", "")
        doc_content_lower = doc.page_content.lower()

        # Drug match score — multiple strategies for robustness
        drug_score = 0.0

        # Strategy 1: Exact substring match (drug name in doc or vice versa)
        if drug_lower and doc_drug:
            if drug_lower in doc_drug or doc_drug in drug_lower:
                drug_score = 1.0

        # Strategy 2: Active ingredient match
        if drug_score < 1.0 and principio_lower:
            # Check if principio matches doc metadata
            if doc_principio and (
                principio_lower in doc_principio or doc_principio in principio_lower
            ):
                drug_score = max(drug_score, 0.9)
            # Check if principio_activo appears in the chunk text
            for pt in principio_tokens:
                if pt in doc_content_lower:
                    drug_score = max(drug_score, 0.8)
                    break

        # Strategy 3: Token overlap between drug names
        if drug_score < 0.8 and drug_tokens:
            doc_tokens = {t for t in doc_drug.split() if len(t) > 2 and t not in _skip_words}
            overlap = len(drug_tokens & doc_tokens)
            if overlap >= 2:
                drug_score = max(drug_score, 0.7)
            elif overlap >= 1:
                drug_score = max(drug_score, 0.4)

        # Strategy 4: Check if drug name tokens appear in chunk text
        if drug_score < 0.4 and drug_tokens:
            text_matches = sum(1 for t in drug_tokens if t in doc_content_lower)
            if text_matches >= 2:
                drug_score = max(drug_score, 0.5)
            elif text_matches >= 1:
                drug_score = max(drug_score, 0.3)

        # Section match score (bi-LSTM predicted sections get higher priority)
        section_score = 0.0
        if doc_section in predicted_sections:
            section_score = 1.0   # Main answer section (like bi-LSTM prediction)
        elif doc_section > 0:
            section_score = 0.3   # Additional info (like VSM/LSI sections)

        total = drug_score * 0.6 + section_score * 0.4
        scored.append((total, doc))

    # Sort by score descending, take top results
    scored.sort(key=lambda x: x[0], reverse=True)

    # Always return results even if scores are low (better some context than none)
    top_results = [doc for _, doc in scored[:max_results]]

    # Log diagnostics
    if scored:
        top_score = scored[0][0]
        logger.info(
            "Filter: top_score=%.2f, drug=%s, principio=%s, results=%d",
            top_score, drug_name[:30], principio_activo, len(top_results),
        )

    return top_results


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 3 — ANSWER EXTRACTION
# (Context assembly + LLM generation with extractive focus)
# ═══════════════════════════════════════════════════════════════════════════════

def assemble_context(
    documents: list[Document],
    max_chars: int = 4500,
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

def _build_retrieval_query(query: dict, entities: dict) -> str:
    """Build a focused retrieval query.

    Instead of concatenating the full drug name + full question (too noisy),
    we use the active ingredient + the core question. This produces better
    BM25 and FAISS matches.
    """
    principio = entities.get("active_ingredient", "")
    query_text = query.get("query_text", "")

    # For comparative queries, include both drug names
    if query.get("is_generic") == "comparative":
        return query_text

    # Use active ingredient (short, matches content) + original question text
    # Keep the original accented text (critical for BM25 matching Spanish)
    if principio:
        return f"{principio} {query_text}"
    else:
        # Fallback: use drug name but trim the long form details
        drug = entities.get("medicine_name", "")
        # Take just the first 2-3 words of the drug name
        drug_short = " ".join(drug.split()[:3])
        return f"{drug_short} {query_text}"


def collect_all_responses(
    queries: list[dict],
    api_key: str,
    model: str = "gpt-4o-mini",
    prospectos_dir: Path = PROSPECTOS_DIR,
    responses_dir: Path = RESPONSES_DIR,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
    top_k: int = 12,
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

    bm25_retriever = None
    faiss_store = None
    retriever_weights = [bm25_weight, 1.0 - bm25_weight]
    doc_count = 0

    if has_prospectos:
        documents = load_prospecto_documents(
            prospectos_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        doc_count = len(documents)
        if documents:
            bm25_retriever, faiss_store, retriever_weights = build_hybrid_retriever(
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
        if bm25_retriever is not None and faiss_store is not None:
            # Build focused retrieval query (NOT accent-stripped)
            retrieval_query = _build_retrieval_query(query, entities)

            raw_results = hybrid_retrieve(
                retrieval_query,
                bm25_retriever,
                faiss_store,
                retriever_weights,
                top_k=top_k,
            )

            # Filter by drug name and predicted sections
            filtered = filter_by_drug_and_section(
                raw_results,
                drug_name=entities["medicine_name"],
                principio_activo=entities.get("active_ingredient", ""),
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
                "total_doc_chunks": doc_count,
                "context_available": bool(context_str),
                "context_length": len(context_str),
                "retrieval_method": "hybrid_bm25_faiss_rrf",
                "bm25_weight": bm25_weight,
            },
            "notes": (
                f"MeQA-RAG (model={model}, "
                f"retrieval=BM25+FAISS+RRF, "
                f"sections={predicted_secs}, "
                f"chunks={chunks_used}, "
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
