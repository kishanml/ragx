import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from .schemas import Chunk
except ImportError:  # pragma: no cover - supports running this file directly
    from schemas import Chunk


class FaissVectorDB:
    def __init__(
        self,
        chunks: list[Chunk] | None = None,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the vector DB.

        Chunks are already generated outside this class. This class only:
        - Stores chunk data
        - Embeds chunk summaries
        - Embeds chunk questions
        - Builds FAISS indexes
        """
        self.model = SentenceTransformer(model_name)
        self.chunks: list[Chunk] = []

        self.summary_texts: list[str] = []
        self.summary_chunk_ids: list[int] = []
        self.summary_index = None

        self.question_texts: list[str] = []
        self.question_chunk_ids: list[int] = []
        self.question_index = None

        if chunks:
            self.store(chunks)

    def store(self, chunks: list[Chunk]):
        """
        Store chunks and build separate FAISS indexes for summaries and questions.
        """
        self.chunks = chunks
        self.summary_texts = []
        self.summary_chunk_ids = []
        self.question_texts = []
        self.question_chunk_ids = []

        for chunk_id, chunk in enumerate(chunks):
            if chunk.summary:
                self.summary_texts.append(chunk.summary)
                self.summary_chunk_ids.append(chunk_id)

            for question in chunk.questions or []:
                self.question_texts.append(question)
                self.question_chunk_ids.append(chunk_id)

        self.summary_index = self._build_index(self.summary_texts)
        self.question_index = self._build_index(self.question_texts)

        return self

    def _build_index(self, texts: list[str]):
        if not texts:
            return None

        embeddings = self.model.encode(texts)
        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        return index

    def query(self, query_text: str, k: int = 5):
        """
        Query summary and question indexes, then return top-k matched chunks.
        """
        query_embedding = self.model.encode([query_text]).astype("float32")
        results_by_chunk = {}

        self._search_index(
            index=self.summary_index,
            query_embedding=query_embedding,
            k=k,
            chunk_ids=self.summary_chunk_ids,
            results_by_chunk=results_by_chunk,
            source="summary",
        )
        self._search_index(
            index=self.question_index,
            query_embedding=query_embedding,
            k=k,
            chunk_ids=self.question_chunk_ids,
            results_by_chunk=results_by_chunk,
            source="question",
            matched_texts=self.question_texts,
        )

        results = list(results_by_chunk.values())
        results.sort(key=lambda result: result["score"])

        return results[:k]

    def _search_index(
        self,
        index,
        query_embedding,
        k,
        chunk_ids,
        results_by_chunk,
        source,
        matched_texts=None,
    ):
        if index is None:
            return

        search_k = min(k, index.ntotal)
        distances, indices = index.search(query_embedding, search_k)

        for i, idx in enumerate(indices[0]):
            chunk_id = chunk_ids[idx]
            score = float(distances[0][i])

            current_result = results_by_chunk.get(chunk_id)
            if current_result and current_result["score"] <= score:
                continue

            result = {
                "chunk": self.chunks[chunk_id],
                "score": score,
                "source": source,
            }

            if matched_texts:
                result["matched_question"] = matched_texts[idx]

            results_by_chunk[chunk_id] = result


    
    