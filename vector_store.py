import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import faiss
from sentence_transformers import SentenceTransformer
import pinecone
from openai import OpenAI

from config import Config
from models import SearchResult

class VectorStore:
    """Handles document embeddings and similarity search"""
    
    def __init__(self):
        self.config = Config()
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.document_ids = []
        
        # Initialize embedding model
        self._initialize_embedding_model()
        
        # Initialize vector database
        self._initialize_vector_db()
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model"""
        if self.config.EMBEDDING_MODEL.startswith("text-embedding-"):
            # Use OpenAI embeddings
            self.openai_client = OpenAI(api_key=self.config.OPENAI_API_KEY)
            self.embedding_model = "openai"
        else:
            # Use sentence transformers
            self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
    
    def _initialize_vector_db(self):
        """Initialize the vector database (FAISS or Pinecone)"""
        if self.config.VECTOR_DB_TYPE == "pinecone":
            self._initialize_pinecone()
        else:
            self._initialize_faiss()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone vector database"""
        if not self.config.PINECONE_API_KEY:
            raise ValueError("Pinecone API key is required for Pinecone vector database")
        
        pinecone.init(
            api_key=self.config.PINECONE_API_KEY,
            environment=self.config.PINECONE_ENVIRONMENT
        )
        
        # Create index if it doesn't exist
        if self.config.PINECONE_INDEX_NAME not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.config.PINECONE_INDEX_NAME,
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine"
            )
        
        self.index = pinecone.Index(self.config.PINECONE_INDEX_NAME)
    
    def _initialize_faiss(self):
        """Initialize FAISS vector database"""
        # Create directory if it doesn't exist
        os.makedirs(self.config.VECTOR_STORE_DIR, exist_ok=True)
        
        # Try to load existing index
        index_path = os.path.join(self.config.VECTOR_STORE_DIR, "faiss_index")
        documents_path = os.path.join(self.config.VECTOR_STORE_DIR, "documents.pkl")
        
        if os.path.exists(index_path) and os.path.exists(documents_path):
            self.index = faiss.read_index(index_path)
            with open(documents_path, 'rb') as f:
                self.documents = pickle.load(f)
                self.document_ids = [doc["id"] for doc in self.documents]
        else:
            # Create new index
            dimension = 1536 if self.embedding_model == "openai" else self.embedding_model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add document chunks to the vector store
        
        Args:
            chunks: List of document chunks with text and metadata
        """
        if not chunks:
            return
        
        # Generate embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self._get_embeddings(texts)
        
        # Add to vector database
        if self.config.VECTOR_DB_TYPE == "pinecone":
            self._add_to_pinecone(chunks, embeddings)
        else:
            self._add_to_faiss(chunks, embeddings)
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks"""
        if self.embedding_model == "openai":
            # Use OpenAI embeddings
            response = self.openai_client.embeddings.create(
                model=self.config.EMBEDDING_MODEL,
                input=texts
            )
            embeddings = [embedding.embedding for embedding in response.data]
            return np.array(embeddings)
        else:
            # Use sentence transformers
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            return embeddings
    
    def _add_to_pinecone(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """Add documents to Pinecone"""
        vectors = []
        for i, chunk in enumerate(chunks):
            vector = {
                "id": chunk["id"],
                "values": embeddings[i].tolist(),
                "metadata": chunk["metadata"]
            }
            vectors.append(vector)
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
    
    def _add_to_faiss(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """Add documents to FAISS"""
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store documents
        self.documents.extend(chunks)
        self.document_ids.extend([chunk["id"] for chunk in chunks])
        
        # Save to disk
        self._save_faiss_index()
    
    def _save_faiss_index(self):
        """Save FAISS index and documents to disk"""
        index_path = os.path.join(self.config.VECTOR_STORE_DIR, "faiss_index")
        documents_path = os.path.join(self.config.VECTOR_STORE_DIR, "documents.pkl")
        
        faiss.write_index(self.index, index_path)
        with open(documents_path, 'wb') as f:
            pickle.dump(self.documents, f)
    
    def search(self, query: str, top_k: Optional[int] = None, 
               document_ids: Optional[List[str]] = None) -> List[SearchResult]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            document_ids: Filter by specific document IDs
            
        Returns:
            List of search results
        """
        if top_k is None:
            top_k = self.config.TOP_K_RESULTS
        
        # Generate query embedding
        query_embedding = self._get_embeddings([query])
        
        # Search in vector database
        if self.config.VECTOR_DB_TYPE == "pinecone":
            results = self._search_pinecone(query_embedding[0], top_k, document_ids)
        else:
            results = self._search_faiss(query_embedding[0], top_k, document_ids)
        
        return results
    
    def _search_pinecone(self, query_embedding: np.ndarray, top_k: int, 
                        document_ids: Optional[List[str]] = None) -> List[SearchResult]:
        """Search in Pinecone"""
        # Prepare filter
        filter_dict = None
        if document_ids:
            filter_dict = {"document_id": {"$in": document_ids}}
        
        # Search
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
        # Convert to SearchResult objects
        search_results = []
        for match in results.matches:
            search_results.append(SearchResult(
                content=match.metadata.get("text", ""),
                metadata=match.metadata,
                score=match.score,
                source_document=match.metadata.get("document_id", "")
            ))
        
        return search_results
    
    def _search_faiss(self, query_embedding: np.ndarray, top_k: int,
                     document_ids: Optional[List[str]] = None) -> List[SearchResult]:
        """Search in FAISS"""
        # Normalize query embedding
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        
        # Search
        scores, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        
        # Convert to SearchResult objects
        search_results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                doc = self.documents[idx]
                
                # Apply document filter if specified
                if document_ids and doc["metadata"]["document_id"] not in document_ids:
                    continue
                
                search_results.append(SearchResult(
                    content=doc["text"],
                    metadata=doc["metadata"],
                    score=float(score),
                    source_document=doc["metadata"]["document_id"]
                ))
        
        return search_results
    
    def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        if self.config.VECTOR_DB_TYPE == "pinecone":
            # Pinecone doesn't support direct document retrieval by ID
            # This would need to be implemented with a separate document store
            return None
        else:
            for doc in self.documents:
                if doc["id"] == document_id:
                    return doc
            return None
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents in the vector store"""
        if self.config.VECTOR_DB_TYPE == "pinecone":
            # This would need to be implemented with a separate document store
            return []
        else:
            return self.documents
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document from the vector store"""
        if self.config.VECTOR_DB_TYPE == "pinecone":
            # Delete from Pinecone
            self.index.delete(ids=[document_id])
            return True
        else:
            # Remove from FAISS and documents list
            indices_to_remove = []
            for i, doc in enumerate(self.documents):
                if doc["metadata"]["document_id"] == document_id:
                    indices_to_remove.append(i)
            
            if indices_to_remove:
                # Remove from documents list
                for i in reversed(indices_to_remove):
                    del self.documents[i]
                
                # Rebuild FAISS index
                self._rebuild_faiss_index()
                return True
            
            return False
    
    def _rebuild_faiss_index(self):
        """Rebuild FAISS index from documents"""
        if not self.documents:
            return
        
        # Generate embeddings for all documents
        texts = [doc["text"] for doc in self.documents]
        embeddings = self._get_embeddings(texts)
        
        # Create new index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize and add embeddings
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        # Update document IDs
        self.document_ids = [doc["id"] for doc in self.documents]
        
        # Save to disk
        self._save_faiss_index()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        if self.config.VECTOR_DB_TYPE == "pinecone":
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_type": "pinecone"
            }
        else:
            return {
                "total_vectors": self.index.ntotal,
                "dimension": self.index.d,
                "index_type": "faiss"
            } 