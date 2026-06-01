import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import uuid
from copy import deepcopy
from sqlite_crud import SQLiteCRUD


class VectorDB:

    def __init__(self,
                 path: str,
                 table_name: str = "chunks",
                 model_name: str = "all-MiniLM-L6-v2",
                 ):

        # Initializing Model
        self.model = SentenceTransformer(model_name)

        # create path if not exists
        os.makedirs(path, exist_ok=True)

        # Initializing Index Path
        self.index_path: str = os.path.join(path, "index.faiss")

        # Initializing SQLite database
        self.db_path = os.path.join(path, "db.sqlite")
        self.table_name = table_name
        if not Path(self.db_path).exists():
            sqlite_obj = SQLiteCRUD(self.db_path)
            sqlite_obj.create_table(self.table_name, {
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "text": "TEXT",
                "chunk_id": "INTEGER",
                "file_hash": "TEXT"
            })
            sqlite_obj.close()

    def _load_faiss(self):
        # print('Loading FAISS index')
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
        sqlite_obj = SQLiteCRUD(self.db_path)
        data = deepcopy(chunks)
        for obj in data:
            obj["file_hash"] = file_hash
        sqlite_obj.insert_many(self.table_name, data)
        sqlite_obj.close()

    def _load_embedding_text_from_db(self, file_hash):
        print('Loading embedding texts from DB')

        sqlite_obj = SQLiteCRUD(self.db_path)
        rows = sqlite_obj.get_where(self.table_name, "file_hash", file_hash)
        sqlite_obj.close()
        print(rows)
        ids = [r["id"] for r in rows]
        embedding_texts = [r["text"] for r in rows]
        print(f'Loaded {len(embedding_texts)} embedding texts')
        return ids, embedding_texts

    def _fetch_chunks_by_ids(self, chunk_ids):
        """Fetch full chunk rows for a list of row ids."""
        sqlite_obj = SQLiteCRUD(self.db_path)
        rows = sqlite_obj.get_by_ids(self.table_name, chunk_ids)
        sqlite_obj.close()
        return rows

    def _check_if_file_processed(self, file_hash):
        sqlite_obj = SQLiteCRUD(self.db_path)
        rows = sqlite_obj.get_where(self.table_name, "file_hash", file_hash)
        sqlite_obj.close()
        count = len(rows)
        print(f'Existing records for file: {count}')
        return count > 0

    def _append_chunks_to_index(self, ids, embedding_texts):
        index = self._load_faiss()
        embeddings = self.model.encode(embedding_texts)
        embeddings = np.array(embeddings).astype("float32")
        index.add_with_ids(embeddings, np.array(ids, dtype=np.int32))
        self._save_faiss(index)

    def append_chunks(self, chunks, file_hash=None):
        if file_hash is None:
            file_hash = str(uuid.uuid4())
        if not self._check_if_file_processed(file_hash):
            self._append_chunks_to_db(file_hash, chunks)
            ids, embedding_texts = self._load_embedding_text_from_db(file_hash)
            self._append_chunks_to_index(ids, embedding_texts)
            return ids
        return []

    def _search_faiss(self, text, k=3):
        index = self._load_faiss()
        text_embedding = self.model.encode([text])
        distances, indices = index.search(text_embedding, k)
        return distances.tolist()[0], indices.tolist()[0]

    def retrieve_chunks(self, text, k=3):
        distances, indices = self._search_faiss(text, k)
        return self._fetch_chunks_by_ids(indices), distances
