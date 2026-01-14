# src/rag/core.py
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import logging
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from dotenv import load_dotenv
import shutil

load_dotenv()

class ProductionRAGSystem:
    """
    Production RAG system with advanced features, explicit embedding control,
    and robust error handling.
    """
    def __init__(self,
                 persist_directory: str = "./data/vector_stores/financial_data",
                 embedding_type: str = "huggingface",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200
                 ):
        
        self.logger = logging.getLogger(__name__)
        
        # Resolve absolute path for safety
        self.persist_directory = str(Path(os.getcwd()) / persist_directory)
        self.embedding_type = embedding_type.lower()
        
        # Initialize Embeddings
        self.embeddings = self._initialize_embeddings()

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize vector store
        try:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=f"financial_data_{self.embedding_type}"
            )
            self.logger.info(f" [INFO] RAG System initialized at {self.persist_directory}")
            
        except Exception as e:
            self.logger.critical(f"Failed to initialize ChromaDB: {e}")
            
            # --- AUTO-FIX LOGIC ---
            # Check the Exception TYPE name, not just the message
            error_name = type(e).__name__
            error_msg = str(e).lower()
            
            if "PanicException" in error_name or "sqlite" in error_msg or "range" in error_msg:
                self.logger.warning(f"[WARN] Database corruption detected ({error_name}). Wiping and rebuilding...")
                
                # Close any lingering connections (if possible)
                if hasattr(self, 'vector_store'):
                    del self.vector_store
                
                # Hard Reset
                if os.path.exists(self.persist_directory):
                    try:
                        shutil.rmtree(self.persist_directory)
                    except PermissionError:
                        self.logger.error("[ERROR] Could not delete corrupted data. Windows is locking the file.")
                        self.logger.error(f"👉 ACTION REQUIRED: Manually delete folder: {self.persist_directory}")
                        raise e

                # Retry initialization
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name=f"financial_data_{self.embedding_type}"
                )
                self.logger.info(" [SUCCESS] Database successfully rebuilt.")
            else:
                raise e

    def _initialize_embeddings(self):
        """Helper to initialize embeddings with fallback logic"""
        if self.embedding_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                self.logger.warning("OPENAI_API_KEY not found. Falling back to HuggingFace.")
                return self._get_hf_embeddings()
            try:
                return OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    api_key=api_key
                )
            except Exception as e:
                self.logger.error(f"OpenAI init failed: {e}. Falling back to HuggingFace.")
                return self._get_hf_embeddings()
        else:
            return self._get_hf_embeddings()

    def _get_hf_embeddings(self):
        """Standardized HF Embeddings configuration"""
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def add_company_data(self, documents: List[Document]) -> List[str]:
        """Add documents with intelligent chunking and error handling"""
        if not documents:
            self.logger.warning("No documents provided to add_company_data")
            return []
        
        try:
            # Split documents
            all_chunks = []
            for doc in documents:
                chunks = self.text_splitter.split_documents([doc])
                all_chunks.extend(chunks)
            
            self.logger.info(f"Split {len(documents)} documents into {len(all_chunks)} chunks")
            
            # Add to vector store
            # Batching could be added here for massive datasets, but this is fine for standard use
            doc_ids = self.vector_store.add_documents(all_chunks)
            self.logger.info(f" [SUCCESS] Added {len(doc_ids)} chunks to vector store")
            return doc_ids
        
        except Exception as e:
            self.logger.error(f" [ERROR] Failed to add documents: {e}")
            return []

    def query_with_similarity_scores(self, 
                                     question: str,
                                     k: int = 5,
                                     company: Optional[str] = None,
                                     metric_type: Optional[str] = None,
                                     score_threshold: float = 2.0) -> List[Tuple[Document, float]]:
        """
        Query with similarity scores and specific metadata filtering.
        """
        search_kwargs = {"k": k}
        filters = []

        # Build filters (Your exact logic)
        if company:
            filters.append({"company": company.upper()})

        if metric_type:
            filters.append({"doc_type": metric_type})

        if len(filters) == 1:
            search_kwargs["filter"] = filters[0]
        elif len(filters) > 1:
            search_kwargs["filter"] = {"$and": filters}
             
        self.logger.info(f" [QUERY] '{question}' | Filters: {filters}")

        try:
            # ChromaDB returns L2 distance (lower is better)
            scores_with_results = self.vector_store.similarity_search_with_score(
                question,
                **search_kwargs
            )

            # Log raw results before filtering
            self.logger.info(f" [RAW] Found {len(scores_with_results)} docs before threshold filter")
            if scores_with_results and len(scores_with_results) > 0:
                best_score = min(score for _, score in scores_with_results)
                worst_score = max(score for _, score in scores_with_results)
                self.logger.info(f" [SCORES] Best: {best_score:.4f}, Worst: {worst_score:.4f}, Threshold: {score_threshold}")
            
            # Filter by score threshold
            filtered_results = [
                (doc, score) for doc, score in scores_with_results
                if score <= score_threshold
            ]

            if len(filtered_results) == 0 and len(scores_with_results) > 0:
                self.logger.warning(f" [FILTERED OUT] All {len(scores_with_results)} docs exceeded threshold {score_threshold}")

            self.logger.info(f" [SUCCESS] Found {len(filtered_results)} relevant docs (distance <= {score_threshold})")
            return filtered_results
        
        except Exception as e:
            self.logger.error(f" [ERROR] Query failed: {e}")
            return []

    def query(self,
              question: str,
              k: int = 5,
              company: Optional[str] = None,
              metric_type: Optional[str] = None,
              score_threshold: float = 2.0) -> List[Document]:        
        """Wrapper that strips scores and returns documents only"""  
        try:
            scores_with_results = self.query_with_similarity_scores(
                question=question,
                k=k,
                company=company,
                metric_type=metric_type,
                score_threshold=score_threshold
            )
            return [doc for doc, score in scores_with_results]
        except Exception as e:
            self.logger.error(f" [ERROR] Query wrapper failed: {e}")
            return []

    def get_company_metrics(self, company: str) -> List[str]:
        """Get all available metrics for a company using direct metadata scanning"""
        try:
            company = company.upper()
            
            # Direct access to Chroma Collection (Your original logic)
            # Wrapped in checks to prevent crashes if library changes
            if hasattr(self.vector_store, '_collection'):
                results = self.vector_store._collection.get(
                    where={"company": company},
                    include=['metadatas'] # Optimization: Don't fetch embeddings/documents
                )
                
                all_metrics = set()
                if results and 'metadatas' in results:
                    for metadata in results['metadatas']:
                        if metadata and 'metric' in metadata and metadata['metric']:
                            all_metrics.add(metadata['metric'])

                if all_metrics:
                    self.logger.info(f"Found {len(all_metrics)} unique metrics for {company}")
                    return sorted(list(all_metrics))

            # Fallback
            self.logger.info("Falling back to similarity search for metrics...")
            return self._get_metrics_via_similarity(company)
        
        except Exception as e:
            self.logger.error(f" [ERROR] Failed to get company metrics: {e}")
            return []

    def _get_metrics_via_similarity(self, company: str) -> List[str]:
        """Fallback method using similarity search"""
        try:
            queries = [f"{company} financial", company]
            all_metrics = set()
            
            for query in queries:
                results = self.query(query, k=50, company=company)
                for doc in results:
                    if doc.metadata.get("metric"):
                        all_metrics.add(doc.metadata["metric"])

            return sorted(list(all_metrics))
        except Exception as e:
            self.logger.error(f"Similarity search fallback failed: {e}")
            return []

    def clear_company_data(self, company: str) -> bool:
        """Remove all data for a specific company"""
        try:
            # Note: Chroma requires a non-empty filter
            self.vector_store._collection.delete(
                where={"company": company.upper()}
            )
            self.logger.info(f" [SUCCESS] Cleared all data for {company}")
            return True
        except Exception as e:
            self.logger.error(f" [ERROR] Failed to clear company data: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            count = 0
            if hasattr(self.vector_store, '_collection'):
                count = self.vector_store._collection.count()
            return {
                "total_documents": count,
                "embedding_type": self.embedding_type,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            self.logger.error(f" [ERROR] Failed to get stats: {e}")
            return {}