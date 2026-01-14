#!/usr/bin/env python3
"""Test liquidity metrics retrieval with new aliases"""

import sys
import os
import asyncio
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag.core import ProductionRAGSystem
from src.tools.financial_tools import FinancialTools
from src.data.collectors.sec_edgar import SECDataCollector

async def test_liquidity_retrieval():
    base_path = Path(os.getcwd()) / "data" / "vector_stores" / "financial_data"
    rag = ProductionRAGSystem(persist_directory=str(base_path))
    sec_collector = SECDataCollector()
    tools = FinancialTools(rag, sec_collector)
    
    print("=" * 60)
    print("Testing Liquidity Metrics Retrieval")
    print("=" * 60)
    
    # Test metrics
    liquidity_metrics = ["Current Assets", "Current Liabilities", "Cash and Cash Equivalents"]
    
    for metric in liquidity_metrics:
        print(f"\n--- Testing: {metric} ---")
        result = await tools.retrieve_financial_metrics("AAPL", [metric])
        print(result[:500] if len(result) > 500 else result)

if __name__ == "__main__":
    asyncio.run(test_liquidity_retrieval())
