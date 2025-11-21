# src/rag/rag_usage.py
from rag.core import ProductionRAGSystem
from langchain_core.documents import Document

def Huggingface_rag():
    print("huggingface demo")
    rag_system = ProductionRAGSystem(
        persist_directory="./data/vector_stores/financial_hf",
        embedding_type="huggingface"
    )

    documents = [
        Document(
            page_content="Apple Inc. reported revenue of $100 billion in 2024.",
            metadata= {"source": "sec", "company": "AAPL", "year": 2024}
            ),
        Document(
            page_content="Microsoft cloud revenue grew 25% year-over-year.",
            metadata= {"source": "news", "company": "MSFT", "year": 2024}
            )
        ]

    doc_ids = rag_system.add_company_data(documents)
    print(f" added {len(doc_ids)} documents")

    results = rag_system.query(question="what is apple's revenue in 2024?")
    for i, doc in enumerate(results):
        print(f" result {i+1}: {doc.page_content}, (source: {doc.metadata['source']})")

    return rag_system

def Openai_rag():
    print("open ai demo")

    rag_system = ProductionRAGSystem(
        persist_directory="./data/vector_stores/financial_openai",
        embedding_type="openai"
    )

    documents = [
        Document(
            page_content="Tesla delivered 500,000 vehicles in Q4 2024.",
            metadata= {"source": "sec", "company": "TSLA", "quarter": "Q4"}
            ),
        Document(
            page_content="Microsoft cloud revenue grew 25% year-over-year.",        
            metadata= {"source": "news", "company": "MSFT", "year": 2024}
        )
    ]

    rag_system.add_company_data(documents)
    results = rag_system.query(question="how many vehicles did tesla deliver in Q4 2024?")
    for doc in results:
        print(f" openai_result: {doc.page_content}, (source: {doc.metadata['source']})")

    return rag_system

if __name__ == "__main__":
    # Ensure directories exist
    import os
    os.makedirs("./data/vector_stores/financial_hf", exist_ok=True)
    os.makedirs("./data/vector_stores/financial_openai", exist_ok=True)
    
    hf_rag = Huggingface_rag()
    # oa_rag = Openai_rag() # Requires API key