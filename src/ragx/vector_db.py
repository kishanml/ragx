import sqlite3
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


class VectorDB:

    def __init__(self,
                 index_path: str,
                 db_path: str,
                 model_name: str = "all-MiniLM-L6-v2",
                 ):

        # Initializing Model
        self.model = SentenceTransformer(model_name)

        # Initializing Index Path
        self.index_path: str = index_path

        # Initializing SQLite database
        self.db_path = db_path
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash TEXT,
                title   TEXT,
                source    TEXT,
                cleaned_text      TEXT,
                embedding_text   TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _load_faiss(self):
        print('Loading FAISS index')
        if self.index_path is not None and Path(self.index_path).exists():
            index = faiss.read_index(self.index_path)
        else:
            dim = self.model.get_sentence_embedding_dimension()
            index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
        return index

    def _save_faiss(self, index):
        faiss.write_index(index, self.index_path)
        print(f'FAISS saved: {self.index_path}')

    def _append_chunks_to_db(self, file_hash, chunks):
        conn = sqlite3.connect(self.db_path)
        conn.executemany(
            "INSERT INTO chunks (file_hash, title, source, cleaned_text, embedding_text) VALUES (?,?,?,?,?)",
            [(file_hash, chunk['title'], chunk['source'], chunk['cleaned_text'], chunk['embedding_text']) for chunk in
             chunks]
        )
        conn.commit()
        conn.close()

    def _check_if_file_processed(self, file_hash):
        conn = sqlite3.connect(self.db_path)
        (count,) = conn.execute("SELECT COUNT(*) FROM chunks WHERE file_hash=?", (file_hash,)).fetchone()
        conn.close()
        print(f'Existing records for file: {count}')
        return count > 0

    def _load_embedding_text_from_db(self, file_hash):
        print('Loading embedding texts from DB')
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT id, embedding_text FROM chunks WHERE file_hash=?", (file_hash,)).fetchall()
        conn.close()
        ids = [r[0] for r in rows]
        embedding_texts = [r[1] for r in rows]
        print(f'Loaded {len(embedding_texts)} embedding texts')
        return ids, embedding_texts

    def _fetch_chunks_by_ids(self, chunk_ids):
        """Fetch full chunk rows for a list of row ids."""
        conn = sqlite3.connect(self.db_path)
        placeholders = ','.join('?' * len(chunk_ids))
        rows = conn.execute(
            f"SELECT id, title, source, cleaned_text FROM chunks WHERE id IN ({placeholders})",
            chunk_ids
        ).fetchall()
        conn.close()
        # preserve caller-supplied order
        id_to_row = {r[0]: r for r in rows}
        return [
            {'id': id_to_row[cid][0], 'heading': id_to_row[cid][1],
             'page': id_to_row[cid][2], 'text': id_to_row[cid][3]}
            for cid in chunk_ids if cid in id_to_row
        ]

    def _append_chunks_to_index(self, ids, embedding_texts):
        index = self._load_faiss()
        embeddings = self.model.encode(embedding_texts)
        embeddings = np.array(embeddings).astype("float32")
        index.add_with_ids(embeddings, np.array(ids, dtype=np.int32))
        self._save_faiss(index)

    def append_chunks(self, file_hash, chunks):
        if not self._check_if_file_processed(file_hash):
            self._append_chunks_to_db(file_hash, chunks)
            ids, embedding_texts = self._load_embedding_text_from_db(file_hash)
            self._append_chunks_to_index(ids, embedding_texts)

    def _search_faiss(self, text, k=3):
        index = self._load_faiss()
        search_k = min(k, index.ntotal)
        text_embedding = self.model.encode([text])[0]
        distances, indices = index.search(text_embedding, search_k)
        return distances, indices

    def retrieve_chunks(self, text, k=3):
        distances, indices = self._search_faiss(text, k)
        return self._fetch_chunks_by_ids(indices), distances
