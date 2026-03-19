"""Clean and chunk CIMA leaflet HTML into section-aware text chunks.

Follows the MEQa methodology:
  - Preserve section structure (6 standard sections)
  - Preserve paragraph breaks and list items
  - Prepend section header to each chunk (MEQa's "context addition")
  - Split by section first, then by paragraph, then by sentence if needed
"""

import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup

from .config import LEAFLETS_DIR

SECTION_NAMES = {
    1: "Qué es y para qué se utiliza",
    2: "Qué necesita saber antes de tomar",
    3: "Cómo tomar",
    4: "Posibles efectos adversos",
    5: "Conservación",
    6: "Contenido del envase e información adicional",
}

logger = logging.getLogger(__name__)

# Approximate tokens per character for Spanish text
CHARS_PER_TOKEN = 3.3
DEFAULT_MAX_TOKENS = 400
DEFAULT_OVERLAP_TOKENS = 50


def _html_to_text(html: str) -> str:
    """Convert HTML to clean text preserving paragraph breaks and list items."""
    soup = BeautifulSoup(html, "html.parser")

    # Replace <br> tags with newlines
    for br in soup.find_all("br"):
        br.replace_with("\n")

    # Replace <li> items with "- " prefix
    for li in soup.find_all("li"):
        li.insert(0, "\n- ")

    # Replace <p> tags with double newlines
    for p in soup.find_all("p"):
        p.insert(0, "\n\n")

    # Get text
    text = soup.get_text()

    # Clean up whitespace while preserving paragraph breaks
    # Collapse multiple spaces on the same line
    text = re.sub(r"[ \t]+", " ", text)
    # Normalize line breaks: collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def _estimate_tokens(text: str) -> int:
    """Approximate token count for Spanish text."""
    return int(len(text) / CHARS_PER_TOKEN)


def _split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs (double newline separated)."""
    paragraphs = re.split(r"\n\n+", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _split_into_sentences(text: str) -> list[str]:
    """Split text at sentence boundaries (period/question/exclamation followed by space)."""
    # Split at sentence boundaries, keeping the punctuation with the sentence
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if s.strip()]


def _chunk_text_with_overlap(
    paragraphs: list[str],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[str]:
    """Chunk paragraphs with token limit and overlap.

    Strategy:
    1. Group consecutive paragraphs until max_tokens is reached
    2. If a single paragraph exceeds max_tokens, split at sentence boundaries
    3. Add overlap_tokens of overlap between consecutive chunks
    """
    max_chars = int(max_tokens * CHARS_PER_TOKEN)
    overlap_chars = int(overlap_tokens * CHARS_PER_TOKEN)

    chunks = []
    current_parts = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        # If single paragraph exceeds limit, split at sentences
        if para_len > max_chars:
            # Flush current accumulator first
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_len = 0

            sentences = _split_into_sentences(para)
            sent_parts = []
            sent_len = 0
            for sent in sentences:
                if sent_len + len(sent) + 1 > max_chars and sent_parts:
                    chunks.append(" ".join(sent_parts))
                    # Keep overlap from end of previous chunk
                    overlap_text = " ".join(sent_parts)
                    if len(overlap_text) > overlap_chars:
                        overlap_text = overlap_text[-overlap_chars:]
                    sent_parts = [overlap_text, sent] if overlap_chars > 0 else [sent]
                    sent_len = len(" ".join(sent_parts))
                else:
                    sent_parts.append(sent)
                    sent_len += len(sent) + 1
            if sent_parts:
                chunks.append(" ".join(sent_parts))
            continue

        # Check if adding this paragraph would exceed limit
        new_len = current_len + para_len + (2 if current_parts else 0)
        if new_len > max_chars and current_parts:
            # Flush current chunk
            chunks.append("\n\n".join(current_parts))

            # Create overlap from end of previous chunk
            if overlap_chars > 0:
                prev_text = "\n\n".join(current_parts)
                if len(prev_text) > overlap_chars:
                    overlap = prev_text[-overlap_chars:]
                    current_parts = [overlap, para]
                else:
                    current_parts = [prev_text, para]
            else:
                current_parts = [para]
            current_len = sum(len(p) for p in current_parts) + 2 * (len(current_parts) - 1)
        else:
            current_parts.append(para)
            current_len = new_len

    # Flush remaining
    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


def chunk_section(
    html: str,
    section_number: int,
    drug_name: str,
    nregistro: str,
    pair_id: str,
    is_generic: bool,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[dict]:
    """Clean and chunk a single leaflet section.

    Returns list of chunk dicts with text and metadata.
    """
    section_name = SECTION_NAMES.get(section_number, f"Sección {section_number}")
    section_header = f"Sección {section_number} - {section_name}"

    # Convert HTML to clean text
    text = _html_to_text(html)
    if not text.strip():
        return []

    # Split into paragraphs
    paragraphs = _split_into_paragraphs(text)
    if not paragraphs:
        return []

    # Chunk with overlap
    raw_chunks = _chunk_text_with_overlap(paragraphs, max_tokens, overlap_tokens)

    # Build chunk dicts with section header prepended
    chunks = []
    for idx, chunk_text in enumerate(raw_chunks):
        # Prepend section header (MEQa's "context addition")
        full_text = f"{section_header}:\n{chunk_text}"

        chunks.append({
            "text": full_text,
            "metadata": {
                "drug_name": drug_name,
                "nregistro": nregistro,
                "is_generic": is_generic,
                "pair_id": pair_id,
                "section_number": section_number,
                "section_name": section_name,
                "chunk_index": idx,
                "char_count": len(chunk_text),
                "token_count_approx": _estimate_tokens(chunk_text),
            },
        })

    return chunks


def chunk_drug_leaflet(
    nregistro: str,
    drug_name: str,
    pair_id: str,
    is_generic: bool,
    leaflets_dir: Path = LEAFLETS_DIR,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[dict]:
    """Chunk all sections for a single drug's leaflet.

    Reads the per-section HTML files from disk and produces chunks.
    """
    pair_dir = leaflets_dir / pair_id
    all_chunks = []

    for sec_num in range(1, 7):
        fpath = pair_dir / f"{nregistro}_section_{sec_num}.html"
        if not fpath.exists() or fpath.stat().st_size == 0:
            logger.warning("Missing section %d for nregistro=%s", sec_num, nregistro)
            continue

        html = fpath.read_text(encoding="utf-8")
        section_chunks = chunk_section(
            html=html,
            section_number=sec_num,
            drug_name=drug_name,
            nregistro=nregistro,
            pair_id=pair_id,
            is_generic=is_generic,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        all_chunks.extend(section_chunks)

    logger.info("Chunked %s: %d chunks from %s", drug_name[:30], len(all_chunks),
                pair_dir)
    return all_chunks


def chunk_all_leaflets(
    drug_pairs: list[dict],
    leaflets_dir: Path = LEAFLETS_DIR,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[dict]:
    """Chunk all leaflets for all drug pairs.

    Args:
        drug_pairs: List of pair dicts with 'pair_id', 'branded', 'generic' keys.
        leaflets_dir: Base directory containing leaflet HTML files.
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Token overlap between consecutive chunks.

    Returns:
        List of all chunk dicts with text and metadata.
    """
    all_chunks = []

    for pair in drug_pairs:
        pair_id = pair["pair_id"]
        for role in ("branded", "generic"):
            drug = pair.get(role)
            if not drug or not drug.get("nregistro"):
                continue

            chunks = chunk_drug_leaflet(
                nregistro=drug["nregistro"],
                drug_name=drug["name"],
                pair_id=pair_id,
                is_generic=drug.get("is_generic", role == "generic"),
                leaflets_dir=leaflets_dir,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )
            all_chunks.extend(chunks)

    logger.info("Total chunks across all pairs: %d", len(all_chunks))
    return all_chunks
