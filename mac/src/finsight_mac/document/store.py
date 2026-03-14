"""ChromaDB fallback vector store.

Used as a secondary retrieval mechanism when PageIndex tree navigation
doesn't find relevant sections. Chunks document text and stores embeddings
in a local ChromaDB instance.

This is a FALLBACK — PageIndex tree navigation is the primary retrieval method.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB-based vector store for fallback retrieval.

    Only used when PageIndex tree navigation doesn't find
    sufficient relevant sections for a query.

    Example:
        >>> store = VectorStore()
        >>> store.add_document("AAPL_10K_2024", page_texts)
        >>> results = store.search("supply chain risks", top_k=5)
    """

    def __init__(self, persist_dir: str = "./data/chromadb") -> None:
        self._client = None
        self._collection = None
        self.persist_dir = persist_dir

    def _ensure_initialized(self) -> None:
        """Lazy initialization of ChromaDB."""
        if self._client is not None:
            return

        try:
            import chromadb

            self._client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = self._client.get_or_create_collection(
                name="finsight_documents",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"ChromaDB initialized at {self.persist_dir}")
        except ImportError:
            logger.warning(
                "chromadb not installed. Vector store fallback unavailable. "
                "Install with: pip install chromadb"
            )

    def add_document(
        self,
        doc_name: str,
        page_texts: dict[int, str],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> int:
        """Add a document's pages to the vector store.

        Args:
            doc_name: Document identifier.
            page_texts: Mapping of page number to text.
            chunk_size: Characters per chunk.
            chunk_overlap: Overlap between consecutive chunks.

        Returns:
            Number of chunks added.
        """
        self._ensure_initialized()
        if self._collection is None:
            return 0

        chunks = []
        ids = []
        metadatas = []

        for page_num, text in sorted(page_texts.items()):
            # Simple chunking by character count
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunk = text[i : i + chunk_size]
                if len(chunk.strip()) < 50:
                    continue

                chunk_id = f"{doc_name}_p{page_num}_c{i}"
                chunks.append(chunk)
                ids.append(chunk_id)
                metadatas.append({"doc_name": doc_name, "page_num": page_num})

        if chunks:
            self._collection.add(
                documents=chunks,
                ids=ids,
                metadatas=metadatas,
            )
            logger.info(f"Added {len(chunks)} chunks for {doc_name}")

        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_name: str | None = None,
    ) -> list[dict]:
        """Search the vector store.

        Args:
            query: Search query text.
            top_k: Number of results to return.
            doc_name: Optional filter to search within a specific document.

        Returns:
            List of result dicts with text, metadata, and distance.
        """
        self._ensure_initialized()
        if self._collection is None:
            return []

        where = {"doc_name": doc_name} if doc_name else None

        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where,
        )

        output = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                output.append(
                    {
                        "text": doc,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else 0,
                    }
                )

        return output
