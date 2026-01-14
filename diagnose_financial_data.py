"""
Diagnostic script to check RAG data for AAPL
"""
import sys
import os
sys.path.append('src')

from rag.core import ProductionRAGSystem
from tools.financial_tools import FinancialTools

def diagnose():
    # Initialize RAG
    rag = ProductionRAGSystem()
    print('RAG initialized')

    # Check what metrics exist for AAPL
    metrics = rag.get_company_metrics('AAPL')
    print(f'Metrics for AAPL: {len(metrics) if metrics else 0}')
    if metrics:
        for m in metrics[:5]:
            print(f'  - {m}')

    # Test structured data extraction
    tools = FinancialTools(rag, None)
    structured = tools.extract_structured_metrics('AAPL')
    
    print(f'\nStructured Data:')
    print(f'  Metrics: {list(structured.get("metrics", {}).keys())}')
    print(f'  Trends: {structured.get("trends", {})}')
    print(f'  Ratios: {structured.get("ratios", {})}')
    print(f'  Latest Period: {structured.get("latest_period")}')

    # Show sample metric data
    if structured.get('metrics'):
        for metric_name, data in structured['metrics'].items():
            print(f'\nData for {metric_name}:')
            for item in data[:3]:
                print(f'  {item}')
    else:
        print("\n[WARN] No metrics extracted!")
        
    # Test RAG query directly
    print("\n--- Direct RAG Query Test ---")
    docs = rag.query_with_similarity_scores(
        question="AAPL Revenue 2024 2023",
        k=5,
        score_threshold=2.5
    )
    print(f"Found {len(docs)} documents for Revenue query")
    for doc, score in docs[:3]:
        print(f"\n  Score: {score:.3f}")
        print(f"  Content: {doc.page_content[:200]}...")
        print(f"  Metadata: {doc.metadata}")

if __name__ == "__main__":
    diagnose()
