#!/usr/bin/env python3
"""
Debug script to test structured data extraction for UI visualization.
This mimics exactly what the UI expects to see.
"""
import sys
import os
import logging
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from src.rag.core import ProductionRAGSystem
from src.tools.financial_tools import FinancialTools
from src.data.collectors.sec_edgar import SECDataCollector

def test_structured_data(company: str = "AAPL"):
    """Test the exact data extraction the UI uses"""
    print("=" * 60)
    print(f"DEBUGGING STRUCTURED DATA FOR: {company}")
    print("=" * 60)
    
    # Initialize components
    base_path = Path(os.getcwd()) / "data" / "vector_stores" / "financial_data"
    print(f"\n1. RAG Path: {base_path}")
    
    rag = ProductionRAGSystem(persist_directory=str(base_path))
    sec_collector = SECDataCollector()
    tools = FinancialTools(rag, sec_collector)
    
    # Check RAG stats
    stats = rag.get_stats()
    print(f"\n2. RAG Stats: {stats}")
    
    # Check company metrics available
    metrics_available = rag.get_company_metrics(company)
    print(f"\n3. Metrics in RAG for {company}: {len(metrics_available) if metrics_available else 0}")
    if metrics_available:
        print(f"   First 10: {metrics_available[:10]}")
    
    print("\n4. Extracting structured data (this is what UI uses)...")
    structured = tools.extract_structured_metrics(company)
    
    print("\n" + "=" * 60)
    print("STRUCTURED DATA RESULTS:")
    print("=" * 60)
    
    # Metrics (ASCII-safe)
    metrics = structured.get('metrics', {})
    print(f"\nMETRICS ({len(metrics)} found):")
    for metric_name, data_points in metrics.items():
        if data_points:
            latest = data_points[0]
            print(f"   - {metric_name}: {latest.get('value', 'N/A')} (period: {latest.get('period', 'N/A')})")
        else:
            print(f"   - {metric_name}: EMPTY []")
    
    # Trends (ASCII-safe)
    trends = structured.get('trends', {})
    print(f"\nTRENDS ({len(trends)} found):")
    for trend_name, value in trends.items():
        print(f"   - {trend_name}: {value}%")
    
    # Ratios (ASCII-safe)
    ratios = structured.get('ratios', {})
    print(f"\nRATIOS ({len(ratios)} found):")
    for ratio_name, value in ratios.items():
        print(f"   - {ratio_name}: {value}")
    
    # Latest period
    print(f"\nLatest Period: {structured.get('latest_period', 'N/A')}")
    
    # UI DIAGNOSTICS
    print("\n" + "=" * 60)
    print("UI DIAGNOSTIC (What the UI will display):")
    print("=" * 60)
    
    # Check specific keys UI looks for
    ui_checks = {
        'Revenue': metrics.get('Revenue', []),
        'Net Income': metrics.get('Net Income', []),
        'Total Assets': metrics.get('Total Assets', []),
        'Cash and Cash Equivalents': metrics.get('Cash and Cash Equivalents', []),
    }
    
    for key, data in ui_checks.items():
        if data:
            value = data[0]['value'] / 1_000_000_000
            print(f"   [OK] {key}: ${value:.2f}B")
        else:
            print(f"   [MISSING] {key}: N/A (no data)")
    
    # Ratio checks
    ratio_checks = ['profit_margin', 'roe', 'current_ratio', 'debt_to_equity']
    for r in ratio_checks:
        val = ratios.get(r)
        if val is not None:
            print(f"   [OK] {r}: {val}")
        else:
            print(f"   [MISSING] {r}: N/A")

    return structured

if __name__ == "__main__":
    company = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    result = test_structured_data(company)
