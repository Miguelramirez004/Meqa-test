"""ChromaDB vector store with OpenAI or local ONNX embeddings.

Uses OpenAI text-embedding-3-small when an API key is provided (1536-dim,
excellent multilingual support).  Falls back to ChromaDB's built-in ONNX
model (all-MiniLM-L6-v2, 384-dim) when no key is available.

Supports both persistent (local dev) and ephemeral (Streamlit Cloud) modes.
One shared ChromaDB collection for all drugs; filtered at query time by metadata.
"""

import hashlib
import logging
import os
from pathlib import Path

import chromadb
from chromadb.config import Settings

from .config import DATA_DIR

logger = logging.getLogger(__name__)

CHROMA_DIR = DATA_DIR / "chromadb"
COLLECTION_NAME = "meqa_leaflets"

# Detect Streamlit Cloud: ephemeral filesystem, no persistent ChromaDB
_ON_STREAMLIT_CLOUD = bool(os.environ.get("STREAMLIT_SERVER_HEADLESS"))


def _make_embedding_fn(openai_api_key: str | None = None):
    """Create the best available embedding function."""
    if openai_api_key:
        try:
            from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
            logger.info("Using OpenAI text-embedding-3-small for embeddings")
            return OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name="text-embedding-3-small",
            )
        except Exception as e:
            logger.warning("OpenAI embedding init failed (%s), falling back to ONNX", e)

    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    logger.info("Using local ONNX embedding model")
    return DefaultEmbeddingFunction()


class MeQAVectorStore:
    """ChromaDB-based vector store for MEQa leaflet chunks."""

    def __init__(self, chroma_dir: Path | str = CHROMA_DIR,
                 openai_api_key: str | None = None,
                 ephemeral: bool | None = None):
        """Initialize the vector store.

        Args:
            chroma_dir: Directory for persistent storage (ignored in ephemeral mode).
            openai_api_key: OpenAI API key for embeddings (optional).
            ephemeral: Force ephemeral (in-memory) mode. Auto-detected on
                       Streamlit Cloud if not specified.
        """
        self._ephemeral = ephemeral if ephemeral is not None else _ON_STREAMLIT_CLOUD
        self._embedding_fn = _make_embedding_fn(openai_api_key)

        if self._ephemeral:
            logger.info("Using ephemeral (in-memory) ChromaDB client")
            self._client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self.chroma_dir = Path(chroma_dir)
            self.chroma_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.chroma_dir),
                settings=Settings(anonymized_telemetry=False),
            )

        try:
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self._embedding_fn,
            )
        except ValueError:
            # Embedding function conflict — recreate the collection
            logger.info("Embedding function changed, recreating collection")
            self._client.delete_collection(COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self._embedding_fn,
            )

    @property
    def count(self) -> int:
        return self._collection.count()

    def _chunk_id(self, chunk: dict) -> str:
        """Generate a deterministic ID for a chunk."""
        meta = chunk["metadata"]
        key = f"{meta.get('drug_name', '')}_{meta['nregistro']}_{meta['section_number']}_{meta['chunk_index']}"
        return hashlib.md5(key.encode()).hexdigest()

    def index_chunks(self, chunks: list[dict], batch_size: int = 100) -> int:
        """Embed and index all chunks into ChromaDB.

        Args:
            chunks: List of chunk dicts with 'text' and 'metadata' keys.
            batch_size: Number of chunks to process at once.

        Returns:
            Number of chunks indexed.
        """
        if not chunks:
            return 0

        # Deduplicate by chunk ID (keep first occurrence)
        seen_ids: set[str] = set()
        unique_chunks: list[dict] = []
        for c in chunks:
            cid = self._chunk_id(c)
            if cid not in seen_ids:
                seen_ids.add(cid)
                unique_chunks.append(c)
        if len(unique_chunks) < len(chunks):
            logger.info("Deduplicated %d → %d chunks", len(chunks), len(unique_chunks))
        chunks = unique_chunks

        texts = [c["text"] for c in chunks]
        ids = [self._chunk_id(c) for c in chunks]

        # Prepare metadata (ChromaDB requires flat string/int/float values)
        metadatas = []
        for c in chunks:
            meta = c["metadata"].copy()
            # Convert bool to int for ChromaDB compatibility
            meta["is_generic"] = 1 if meta.get("is_generic") else 0
            metadatas.append(meta)

        logger.info("Indexing %d chunks (embeddings computed by ChromaDB)...", len(texts))

        # Upsert in batches — ChromaDB computes embeddings automatically
        indexed = 0
        for i in range(0, len(texts), batch_size):
            end = min(i + batch_size, len(texts))
            self._collection.upsert(
                ids=ids[i:end],
                documents=texts[i:end],
                metadatas=metadatas[i:end],
            )
            indexed += end - i
            logger.info("Indexed %d/%d chunks", indexed, len(texts))

        return indexed

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        pair_id: str | None = None,
        drug_name: str | None = None,
    ) -> list[dict]:
        """Retrieve top-k chunks by cosine similarity with metadata filtering.

        Args:
            query: The query text to embed and search.
            top_k: Number of results to return.
            pair_id: Filter by drug pair ID (required).
            drug_name: Filter by specific drug name (None = both drugs in pair).

        Returns:
            List of result dicts with 'text', 'metadata', 'similarity' keys.
        """
        # Build ChromaDB where filter
        where_filter = {}
        if pair_id and drug_name:
            where_filter = {
                "$and": [
                    {"pair_id": {"$eq": pair_id}},
                    {"drug_name": {"$eq": drug_name}},
                ]
            }
        elif pair_id:
            where_filter = {"pair_id": {"$eq": pair_id}}
        elif drug_name:
            where_filter = {"drug_name": {"$eq": drug_name}}

        # Query ChromaDB — embedding computed automatically from query_texts
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error("ChromaDB query failed: %s", e)
            return []

        # Convert distances to similarity scores (ChromaDB cosine distance = 1 - similarity)
        output = []
        if results and results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                output.append({
                    "text": doc,
                    "metadata": meta,
                    "similarity": 1.0 - dist,  # cosine similarity
                })

        return output

    def clear(self):
        """Delete all data from the collection."""
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embedding_fn,
        )
        logger.info("Cleared ChromaDB collection")
