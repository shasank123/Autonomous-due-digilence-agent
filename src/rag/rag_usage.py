# src/rag/rag_usage.py
import os
import sys
from pathlib import Path

# Add project root to python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.rag.core import ProductionRAGSystem
from langchain_core.documents import Document

def Huggingface_rag():
    print("--- HuggingFace RAG Demo ---")
    
    # Ensure directory exists
    persist_dir = "./data/vector_stores/financial_hf"
    os.makedirs(persist_dir, exist_ok=True)

    rag_system = ProductionRAGSystem(
        persist_directory=persist_dir,
        embedding_type="huggingface"
    )

    documents = [
        Document(
            page_content="Apple Inc. reported revenue of $100 billion in 2024.",
            metadata={"source": "sec", "company": "AAPL", "year": 2024, "doc_type": "financial_metric"}
        ),
        Document(
            page_content="Microsoft cloud revenue grew 25% year-over-year.",
            metadata={"source": "news", "company": "MSFT", "year": 2024, "doc_type": "market_analysis"}
        )
    ]

    doc_ids = rag_system.add_company_data(documents)
    print(f"Added {len(doc_ids)} documents")

    results = rag_system.query(question="what is apple's revenue in 2024?", company="AAPL")
    for i, doc in enumerate(results):
        print(f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})")

    return rag_system

def Openai_rag():
    print("\n--- OpenAI RAG Demo ---")
    
    # Check for key before running to prevent crash
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping OpenAI demo: OPENAI_API_KEY not found in environment.")
        return None

    persist_dir = "./data/vector_stores/financial_openai"
    os.makedirs(persist_dir, exist_ok=True)

    rag_system = ProductionRAGSystem(
        persist_directory=persist_dir,
        embedding_type="openai"
    )

    documents = [
        Document(
            page_content="Tesla delivered 500,000 vehicles in Q4 2024.",
            metadata={"source": "sec", "company": "TSLA", "quarter": "Q4", "doc_type": "operational_metric"}
        ),
        Document(
            page_content="Microsoft cloud revenue grew 25% year-over-year.",        
            metadata={"source": "news", "company": "MSFT", "year": 2024}
        )
    ]

    rag_system.add_company_data(documents)
    
    # Using similarity scores to see confidence
    results = rag_system.query_with_similarity_scores(
        question="how many vehicles did tesla deliver in Q4 2024?", 
        company="TSLA"
    )
    
    for doc, score in results:
        print(f"OpenAI Result (Score: {score:.4f}): {doc.page_content}")

    return rag_system

if __name__ == "__main__":
    # 1. Run HF Demo
    hf_rag = Huggingface_rag()
    
    # 2. Run OpenAI Demo
    oa_rag = Openai_rag()
    
    print("\n[DONE] RAG Usage Tests Completed.")