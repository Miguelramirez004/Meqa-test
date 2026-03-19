"""ChromaDB vector store with built-in ONNX embeddings.

Uses ChromaDB's default embedding function (all-MiniLM-L6-v2 ONNX)
for 384-dimensional vectors that run locally without external downloads.

One shared ChromaDB collection for all drugs; filtered at query time by metadata.
"""

import hashlib
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from .config import DATA_DIR

logger = logging.getLogger(__name__)

CHROMA_DIR = DATA_DIR / "chromadb"
COLLECTION_NAME = "meqa_leaflets"


class MeQAVectorStore:
    """ChromaDB-based vector store for MEQa leaflet chunks."""

    def __init__(self, chroma_dir: Path = CHROMA_DIR):
        self.chroma_dir = chroma_dir
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._embedding_fn = DefaultEmbeddingFunction()
        self._client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
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
        key = f"{meta['nregistro']}_{meta['section_number']}_{meta['chunk_index']}"
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
