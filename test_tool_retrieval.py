#!/usr/bin/env python3
"""Test the financial tools directly to see what they're retrieving"""

import sys
import os
import asyncio
import logging
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag.core import ProductionRAGSystem
from src.data.collectors.sec_edgar import SECDataCollector
from src.tools.financial_tools import FinancialTools

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def test_tool_retrieval():
    logger.info("=" * 70)
    logger.info("TESTING FINANCIAL TOOLS DATA RETRIEVAL")
    logger.info("=" * 70)
    
    # Initialize RAG and tools
    base_path = Path(os.getcwd()) / "data" / "vector_stores" / "financial_data"
    rag = ProductionRAGSystem(persist_directory=str(base_path))
    sec_collector = SECDataCollector()
    tools = FinancialTools(rag, sec_collector)
    
    logger.info("\n[Test 1] Retrieve Financial Metrics")
    logger.info("-" * 70)
    
    metrics = ["Revenue", "NetIncomeLoss", "Assets", "Liabilities"]
    result = await tools.retrieve_financial_metrics("AAPL", metrics)
    
    logger.info("\nRESULT:")
    logger.info(result)
    
    logger.info("\n" + "=" * 70)
    logger.info("\n[Test 2] Get Company Overview")
    logger.info("-" * 70)
    
    overview = await tools.get_company_overview("AAPL")
    logger.info("\nRESULT:")
    logger.info(overview)
    
    logger.info("\n" + "=" * 70)
    logger.info("\n[Test 3] Calculate Trends")
    logger.info("-" * 70)
    
    trends = await tools.calculate_trends("AAPL", ["Revenue", "Assets"])
    logger.info("\nRESULT:")
    logger.info(trends)
    
    logger.info("\n" + "=" * 70)
    logger.info("TESTS COMPLETE")
    logger.info("=" * 70)

if __name__ == "__main__":
    asyncio.run(test_tool_retrieval())
