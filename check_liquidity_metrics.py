#!/usr/bin/env python3
"""Check what liquidity-related metrics are available for AAPL"""

import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag.core import ProductionRAGSystem

def check_liquidity_metrics():
    base_path = Path(os.getcwd()) / "data" / "vector_stores" / "financial_data"
    rag = ProductionRAGSystem(persist_directory=str(base_path))
    
    print("=" * 60)
    print("Checking Liquidity Metrics for AAPL")
    print("=" * 60)
    
    # Get all metrics
    metrics = rag.get_company_metrics("AAPL")
    
    # Filter for liquidity-related metrics
    liquidity_keywords = ['current', 'cash', 'liquid', 'receivable', 'payable', 'inventory']
    
    print(f"\nTotal metrics available: {len(metrics)}")
    print("\nLiquidity-related metrics:")
    
    liquidity_metrics = []
    for metric in sorted(metrics):
        if any(keyword in metric.lower() for keyword in liquidity_keywords):
            liquidity_metrics.append(metric)
            print(f"  - {metric}")
    
    print(f"\nFound {len(liquidity_metrics)} liquidity-related metrics")
    
    # Query for specific liquidity metrics
    print("\n" + "=" * 60)
    print("Querying for specific metrics")
    print("=" * 60)
    
    for query in ["Current Assets", "Current Liabilities", "Cash and Cash Equivalents"]:
        print(f"\n--- Query: {query} ---")
        results = rag.query_with_similarity_scores(
            question=f"AAPL {query}",
            company="AAPL",
            k=5,
            score_threshold=2.0
        )
        
        print(f"Found {len(results)} documents")
        for i, (doc, score) in enumerate(results[:3]):
            metric = doc.metadata.get('metric', 'N/A')
            period = doc.metadata.get('period', 'N/A')
            print(f"  {i+1}. Metric: {metric} | Period: {period} | Score: {score:.3f}")

if __name__ == "__main__":
    check_liquidity_metrics()
