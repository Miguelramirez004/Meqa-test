"""In-memory vector store using OpenAI embeddings + numpy cosine similarity.

No external vector DB needed.  Designed for Streamlit Cloud where the
filesystem is ephemeral.

Embeds chunks via OpenAI text-embedding-3-small (1536-dim, excellent
multilingual / Spanish support).  Retrieval is a simple numpy dot-product
on the (small) set of leaflet chunks — fast enough for ~1000 chunks.
"""

import hashlib
import logging
import time
from typing import Optional

import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
BATCH_SIZE = 50  # Conservative batch size to avoid API limits


class MeQAVectorStore:
    """In-memory vector store backed by OpenAI embeddings."""

    def __init__(self, openai_api_key: str | None = None, **_kwargs):
        self._client: OpenAI | None = None
        if openai_api_key:
            self._client = OpenAI(api_key=openai_api_key)

        # Parallel lists — kept in sync
        self._texts: list[str] = []
        self._metadatas: list[dict] = []
        self._ids: list[str] = []
        self._embeddings: np.ndarray | None = None  # shape (N, dim)

    # ── public properties ─────────────────────────────────────────────────

    @property
    def count(self) -> int:
        return len(self._texts)

    # ── embedding helpers ─────────────────────────────────────────────────

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed texts using OpenAI API with retry logic.  Returns (N, dim) array."""
        if not self._client:
            raise RuntimeError("No OpenAI API key — cannot compute embeddings")

        all_vecs: list[list[float]] = []
        total = len(texts)

        for i in range(0, total, BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

            # Retry with exponential backoff
            for attempt in range(4):
                try:
                    resp = self._client.embeddings.create(
                        input=batch,
                        model=EMBEDDING_MODEL,
                    )
                    all_vecs.extend([d.embedding for d in resp.data])
                    logger.info("  Embedded batch %d/%d (%d texts)",
                                batch_num, total_batches, len(batch))
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    logger.warning("  Embedding batch %d failed (attempt %d/4): %s — retrying in %ds",
                                    batch_num, attempt + 1, e, wait)
                    if attempt == 3:
                        logger.error("  Embedding batch %d failed after 4 attempts: %s", batch_num, e)
                        raise
                    time.sleep(wait)

            # Small delay between batches to respect rate limits
            if i + BATCH_SIZE < total:
                time.sleep(0.2)

        return np.array(all_vecs, dtype=np.float32)

    # ── chunk ID ──────────────────────────────────────────────────────────

    @staticmethod
    def _chunk_id(chunk: dict) -> str:
        meta = chunk["metadata"]
        key = (
            f"{meta.get('drug_name', '')}_{meta.get('nregistro', '')}"
            f"_{meta.get('section_number', 0)}_{meta.get('chunk_index', 0)}"
        )
        return hashlib.md5(key.encode()).hexdigest()

    # ── indexing ──────────────────────────────────────────────────────────

    def index_chunks(self, chunks: list[dict]) -> int:
        """Embed and store chunks in memory.

        Returns number of chunks indexed.
        """
        if not chunks:
            return 0

        # Deduplicate
        seen: set[str] = set(self._ids)
        unique: list[dict] = []
        for c in chunks:
            cid = self._chunk_id(c)
            if cid not in seen:
                seen.add(cid)
                unique.append(c)
        if not unique:
            logger.info("All %d chunks already indexed", len(chunks))
            return 0
        if len(unique) < len(chunks):
            logger.info("Deduplicated %d → %d chunks", len(chunks), len(unique))

        texts = [c["text"] for c in unique]
        ids = [self._chunk_id(c) for c in unique]
        metadatas = [c["metadata"].copy() for c in unique]

        logger.info("Embedding %d chunks via OpenAI %s …", len(texts), EMBEDDING_MODEL)
        try:
            new_embeddings = self._embed_texts(texts)
        except Exception as e:
            logger.error("Failed to embed chunks: %s", e)
            raise

        # Append to existing store
        self._texts.extend(texts)
        self._ids.extend(ids)
        self._metadatas.extend(metadatas)

        if self._embeddings is None:
            self._embeddings = new_embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])

        logger.info("Indexed %d new chunks (total: %d)", len(unique), self.count)
        return len(unique)

    # ── retrieval ─────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        pair_id: Optional[str] = None,
        drug_name: Optional[str] = None,
    ) -> list[dict]:
        """Retrieve top-k chunks by cosine similarity.

        Returns list of dicts with 'text', 'metadata', 'similarity' keys.
        """
        if self._embeddings is None or self.count == 0:
            return []

        # Embed query
        try:
            q_vec = self._embed_texts([query])[0]  # (dim,)
        except Exception as e:
            logger.error("Failed to embed query: %s", e)
            return []

        # Cosine similarity (embeddings are already normalised by OpenAI)
        scores = self._embeddings @ q_vec  # (N,)

        # Apply metadata filters
        mask = np.ones(len(scores), dtype=bool)
        if pair_id:
            mask &= np.array([m.get("pair_id") == pair_id for m in self._metadatas])
        if drug_name:
            mask &= np.array([m.get("drug_name") == drug_name for m in self._metadatas])

        scores[~mask] = -1.0  # exclude filtered-out chunks

        # Top-k
        k = min(top_k, int(mask.sum()))
        if k == 0:
            return []
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        return [
            {
                "text": self._texts[i],
                "metadata": self._metadatas[i],
                "similarity": float(scores[i]),
            }
            for i in top_idx
        ]

    # ── management ────────────────────────────────────────────────────────

    def clear(self):
        """Wipe the in-memory store."""
        self._texts.clear()
        self._metadatas.clear()
        self._ids.clear()
        self._embeddings = None
        logger.info("Cleared vector store")
