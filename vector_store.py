# vector_store.py

import os
import pickle
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from config import Config
from models import SearchResult

class VectorStore:
    """A unified interface for vector storage and retrieval operations."""
    def __init__(self):
        self.config = Config()
        self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.index = None
        self.doc_metadata_store = []
        self._initialize_vector_db()

    def _initialize_vector_db(self):
        db_type = self.config.VECTOR_DB_TYPE.lower()
        if db_type == "pinecone" and self.config.PINECONE_API_KEY:
            try:
                self._initialize_pinecone()
                print("[VectorStore] Pinecone initialized successfully.")
            except Exception as e:
                print(f"❌ Failed to initialize Pinecone: {e}. Falling back to FAISS.")
                self.config.VECTOR_DB_TYPE = "faiss"
                self._initialize_faiss()
        else:
            self._initialize_faiss()
            print("[VectorStore] FAISS initialized successfully.")

    def _initialize_pinecone(self):
        pc = Pinecone(api_key=self.config.PINECONE_API_KEY)
        index_name = self.config.PINECONE_INDEX_NAME
        if index_name not in pc.list_indexes().names():
            from pinecone import ServerlessSpec
            pc.create_index(
                name=index_name,
                dimension=self.embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        self.index = pc.Index(index_name)

    def _initialize_faiss(self):
        store_dir = self.config.VECTOR_STORE_DIR
        os.makedirs(store_dir, exist_ok=True)
        index_path = os.path.join(store_dir, "faiss_index.bin")
        meta_path = os.path.join(store_dir, "doc_metadata.pkl")
        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path, 'rb') as f:
                self.doc_metadata_store = pickle.load(f)
        else:
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embedding_dim))
            self.doc_metadata_store = []

    def _save_faiss_index(self):
        store_dir = self.config.VECTOR_STORE_DIR
        faiss.write_index(self.index, os.path.join(store_dir, "faiss_index.bin"))
        with open(os.path.join(store_dir, "doc_metadata.pkl"), 'wb') as f:
            pickle.dump(self.doc_metadata_store, f)

    def add_documents(self, chunks: List[Dict[str, Any]]):
        if not chunks: return
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        if self.config.VECTOR_DB_TYPE == "pinecone":
            self._add_to_pinecone(chunks, embeddings)
        else:
            self._add_to_faiss(chunks, embeddings)

    def _add_to_pinecone(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        vectors_to_upsert = []
        for chunk, embedding in zip(chunks, embeddings):
            clean_metadata = {k: v for k, v in chunk["metadata"].items() if isinstance(v, (str, int, float, bool, list))}
            vectors_to_upsert.append((chunk["id"], embedding.tolist(), clean_metadata))
        for i in range(0, len(vectors_to_upsert), 100):
            self.index.upsert(vectors=vectors_to_upsert[i:i + 100])

    def _add_to_faiss(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        start_index = len(self.doc_metadata_store)
        ids_to_add = np.array(range(start_index, start_index + len(chunks)), dtype=np.int64)
        faiss.normalize_L2(embeddings)
        self.index.add_with_ids(embeddings, ids_to_add)
        self.doc_metadata_store.extend(chunks)
        self._save_faiss_index()

    def search(self, query: str, top_k: int, document_ids: Optional[List[str]] = None) -> List[SearchResult]:
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        if self.config.VECTOR_DB_TYPE == "pinecone":
            return self._search_pinecone(query_embedding[0], top_k, document_ids)
        else:
            return self._search_faiss(query_embedding, top_k, document_ids)

    def _search_pinecone(self, query_embedding: np.ndarray, top_k: int, document_ids: Optional[List[str]]) -> List[SearchResult]:
        filter_dict = {"document_id": {"$in": document_ids}} if document_ids else None
        results = self.index.query(vector=query_embedding.tolist(), top_k=top_k, include_metadata=True, filter=filter_dict)
        return [SearchResult(content=match.metadata.get("text", ""), metadata=match.metadata, score=match.score) for match in results.matches]

    def _search_faiss(self, query_embedding: np.ndarray, top_k: int, document_ids: Optional[List[str]]) -> List[SearchResult]:
        if self.index.ntotal == 0: return []
        faiss.normalize_L2(query_embedding)
        search_k = top_k * 10 if document_ids and self.index.ntotal > top_k else top_k
        scores, indices = self.index.search(query_embedding, min(search_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or len(results) >= top_k: continue
            doc = self.doc_metadata_store[idx]
            if not document_ids or doc["metadata"]["document_id"] in document_ids:
                results.append(SearchResult(content=doc["text"], metadata=doc["metadata"], score=float(score)))
        return results
    
    def delete_document(self, document_id: str) -> bool:
        # Simplified for brevity; production would need robust implementation
        return False
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        return self.doc_metadata_store