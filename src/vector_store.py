"""ChromaDB vector store with sentence-transformers embeddings.

Uses paraphrase-multilingual-MiniLM-L12-v2 for Spanish-native embeddings
(384-dimensional vectors, free, runs locally).

One shared ChromaDB collection for all drugs; filtered at query time by metadata.
"""

import hashlib
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .config import DATA_DIR

logger = logging.getLogger(__name__)

CHROMA_DIR = DATA_DIR / "chromadb"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "meqa_leaflets"

# Singleton for the embedding model (avoid reloading)
_model_cache: dict = {}


def get_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """Get or load the sentence-transformers model (cached singleton)."""
    if model_name not in _model_cache:
        logger.info("Loading embedding model: %s", model_name)
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


class MeQAVectorStore:
    """ChromaDB-based vector store for MEQa leaflet chunks."""

    def __init__(self, chroma_dir: Path = CHROMA_DIR,
                 model_name: str = EMBEDDING_MODEL):
        self.chroma_dir = chroma_dir
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._model = None
        self._client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = get_embedding_model(self.model_name)
        return self._model

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

        # Generate embeddings in batches
        texts = [c["text"] for c in chunks]
        ids = [self._chunk_id(c) for c in chunks]

        # Prepare metadata (ChromaDB requires flat string/int/float values)
        metadatas = []
        for c in chunks:
            meta = c["metadata"].copy()
            # Convert bool to int for ChromaDB compatibility
            meta["is_generic"] = 1 if meta.get("is_generic") else 0
            metadatas.append(meta)

        logger.info("Embedding %d chunks...", len(texts))
        embeddings = self.model.encode(texts, batch_size=batch_size,
                                       show_progress_bar=True)
        embeddings_list = embeddings.tolist()

        # Upsert in batches
        indexed = 0
        for i in range(0, len(texts), batch_size):
            end = min(i + batch_size, len(texts))
            self._collection.upsert(
                ids=ids[i:end],
                documents=texts[i:end],
                embeddings=embeddings_list[i:end],
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

        # Embed query
        query_embedding = self.model.encode([query])[0].tolist()

        # Query ChromaDB
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
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
        )
        logger.info("Cleared ChromaDB collection")
