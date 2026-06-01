import time

from utils import Timer
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from tqdm import tqdm
from store import chunk_entities_extraction_prompt
from sentence_transformers import CrossEncoder
import numpy as np
from sqlite_crud import SQLiteCRUD
from vector_db import VectorDB
from tqdm import tqdm
import os
from copy import deepcopy
from schemas import Output
from dotenv import load_dotenv
load_dotenv()


class VectorFramework:

    def __init__(self, framework_path: str, embedding_model_name: str = "all-MiniLM-L6-v2",
                 chat_model_name: str = "gpt-5.4-mini",
                 reranker_model_name: str = 'cross-encoder/ms-marco-MiniLM-L-12-v2'):
        # define models
        self._llm = ChatOpenAI(model=chat_model_name, temperature=0)
        self._reranker = CrossEncoder(reranker_model_name)

        # define vars
        summary_path_key = "summary"
        question_path_key = "question"
        self.main_db_path = os.path.join(framework_path, "main_db.sqlite")
        self.main_table_name = "main_table"

        # create folders if not exists
        os.makedirs(framework_path, exist_ok=True)

        # create Vector DB Objects
        self.summary_vector_db = VectorDB(
            path=os.path.join(framework_path, summary_path_key),
            model_name=embedding_model_name
        )

        self.question_vector_db = VectorDB(
            path=os.path.join(framework_path, question_path_key),
            model_name=embedding_model_name
        )

        # create tables
        sqlite_obj = SQLiteCRUD(self.main_db_path)
        sqlite_obj.create_table(self.main_table_name, {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "text": "TEXT",
            "file_hash": "TEXT"
        })
        sqlite_obj.close()

    def _update_chunk_entities(self, chunk: dict):
        response = self._llm.with_structured_output(Output).invoke(
            [SystemMessage(content=chunk_entities_extraction_prompt.format(text_chunk=chunk["text"]))])
        chunk["summary"] = response.summary
        chunk["questions"] = response.questions
        chunk["keywords"] = response.keywords
        time.sleep(0.1)
        return chunk

    def _insert_to_main_table(self, file_hash, chunks):
        sqlite_obj = SQLiteCRUD(self.main_db_path)
        data = deepcopy(chunks)
        for obj in data:
            obj["file_hash"] = file_hash
        sqlite_obj.insert_many(self.main_table_name, data)
        rows = sqlite_obj.get_where(self.main_table_name, "file_hash", file_hash)
        sqlite_obj.close()
        return rows

    def _check_if_file_exists(self, file_hash):
        sqlite_obj = SQLiteCRUD(self.main_db_path)
        rows = sqlite_obj.get_where(self.main_table_name, "file_hash", file_hash)
        sqlite_obj.close()
        count = len(rows)
        print(f'Existing records for file: {count}')
        return count > 0

    @staticmethod
    def _prepare_summary_chunks(chunks):
        updated_chunks = []
        for chunk in chunks:
            updated_chunk = dict()
            updated_chunk["chunk_id"] = chunk["id"]
            updated_chunk["text"] = chunk["summary"]
            updated_chunks.append(updated_chunk)
        return updated_chunks

    @staticmethod
    def _preprare_question_chunks(chunks):
        updated_chunks = []
        for chunk in chunks:
            for question in chunk["questions"]:
                updated_chunk = dict()
                updated_chunk["chunk_id"] = chunk["id"]
                updated_chunk["text"] = question
                updated_chunks.append(updated_chunk)
        return updated_chunks

    def append_chunks(self, file_hash, chunks):
        if not self._check_if_file_exists(file_hash):
            rows = self._insert_to_main_table(file_hash, chunks)
            # with Timer("Generating chunk's entities"):
            rows_updated = list(map(self._update_chunk_entities, tqdm(rows)))

            summary_chunks = self._prepare_summary_chunks(rows_updated)
            question_chunks = self._preprare_question_chunks(rows_updated)

            self.summary_vector_db.append_chunks(summary_chunks, file_hash)
            self.question_vector_db.append_chunks(question_chunks, file_hash)

    def retrieve_chunks(self, text, k=5, use_reranker=False):
        retreived_summary_chunks, retreived_summary_distances = self.summary_vector_db.retrieve_chunks(text, k)
        retreived_question_chunks, retreived_question_distances = self.question_vector_db.retrieve_chunks(text, k)

        all_chunks = retreived_summary_chunks + retreived_question_chunks
        all_distances = retreived_summary_distances + retreived_question_distances

        if use_reranker is False:
            sorted_distances_idx = np.argsort(all_distances)[::-1]
            filtered_chunks = dict()

            for idx in sorted_distances_idx:
                chunk = all_chunks[idx]
                if chunk["chunk_id"] not in filtered_chunks.keys():
                    filtered_chunks[chunk["chunk_id"]] = all_distances[idx]

                if len(filtered_chunks) >= k:
                    break

            sqlite_obj = SQLiteCRUD(self.main_db_path)
            rows = sqlite_obj.get_by_ids(self.main_table_name, list(filtered_chunks.keys()))
            sqlite_obj.close()
            return [row["text"] for row in rows]

        else:
            unique_chunk_ids = list(set(map(lambda x: x["chunk_id"], all_chunks)))
            sqlite_obj = SQLiteCRUD(self.main_db_path)
            rows = sqlite_obj.get_by_ids(self.main_table_name, unique_chunk_ids)
            sqlite_obj.close()
            filtered_rows = self._rerank(text, rows, k)
            return [filtered_row["text"] for filtered_row in filtered_rows]

    def _rerank(self, query, chunks, top_n):
        """Score (query, chunk-text) pairs with a cross-encoder and return the top-n chunks sorted by descending reranker score"""
        # print(f'\nReranking {len(chunks)} candidates -> top {top_n}')
        if not chunks:
            return chunks

        pairs = [(query, ch['text']) for ch in chunks]
        scores = self._reranker.predict(pairs)

        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        top = [ch for _, ch in ranked[:top_n]]
        # print('Reranking done')
        return top
