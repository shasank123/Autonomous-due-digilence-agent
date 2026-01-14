#!/usr/bin/env python3
"""Check what metadata is actually in the documents"""

import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag.core import ProductionRAGSystem

def check_metadata():
    base_path = Path(os.getcwd()) / "data" / "vector_stores" / "financial_data"
    rag = ProductionRAGSystem(persist_directory=str(base_path))
    
    print("=" * 60)
    print("Checking Document Metadata")
    print("=" * 60)
    
    # Query for Liabilities
    results = rag.query_with_similarity_scores(
        question="AAPL Liabilities",
        company="AAPL",
        k=10,
        score_threshold=2.0
    )
    
    print(f"\nFound {len(results)} documents for Liabilities query")
    print("\nDocument metadata and content samples:")
    
    for i, (doc, score) in enumerate(results[:10]):
        print(f"\n--- Document {i+1} (score: {score:.3f}) ---")
        print(f"Metadata: {doc.metadata}")
        print(f"Content (first 150 chars): {doc.page_content[:150]}")

if __name__ == "__main__":
    check_metadata()
