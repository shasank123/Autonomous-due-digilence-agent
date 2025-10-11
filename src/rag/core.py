from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os
from typing import List, Optional

class Ragsystem:
    def __init__(self,
                persist_directory: str,
                embedding_type: str = "openai",# "huggingface" or "openai"
                model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
                ):
        self.persist_directory = persist_directory
        self.embedding_type = embedding_type

    
        if embedding_type == "openai":
            self.embeddings = OpenAIEmbeddings(
                model_name="text-embedding-3-small",
            )
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

    
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
            )
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        return self.vector_store.add_documents(documents)
    
    def query(self,
              question: str,
              k: int = 5,
              filter_metadata: Optional[dict] = None
              ) -> List[Document]:
        search_kwargs = {"k": k}
        if filter_metadata:
            search_kwargs["filter"] = filter_metadata
            return self.vector_store.similarity_search(question, **search_kwargs)
        
    def query_with_similarity_scores(
            self,
            question: str,
            k: int = 5) -> List[tuple[Document, float]]:
        return self.vector_store.similarity_search_with_score(question, k=k)

    def document_count(self) -> int:
        return self.vector_store._collection.count()

    def persist(self):
        self.vector_store.persist()


    
              
              

        
       