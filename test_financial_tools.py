"""
Test script for Financial Tools
Verifies that SEC data retrieval and investment summary generation work correctly.
"""
import asyncio
import os
from dotenv import load_dotenv
from src.tools.financial_tools import FinancialTools
from src.rag.core import ProductionRAGSystem

# Load env vars
load_dotenv()

from src.data.collectors.sec_edgar import SECDataCollector

async def test_tools():
    print("1. Initializing FinancialTools...")
    try:
        rag = ProductionRAGSystem(persist_directory="./data/vector_stores/financial_data_test")
        sec = SECDataCollector()
        tools = FinancialTools(rag_system=rag, sec_collector=sec)
        print("SUCCESS: FinancialTools initialized")
    except Exception as e:
        print(f"ERROR: Initialization failed: {e}")
        return

    company = "AAPL"
    print(f"\n2. Testing retrieve_financial_metrics for {company}...")
    try:
        # Test getting revenue
        result = await tools.retrieve_financial_metrics(company, ["Revenue"])
        print(f"Result type: {type(result)}")
        print(f"Result preview: {str(result)[:200]}...")
        
        if "Error" in str(result) or "Failed" in str(result):
            print("WARNING: Tool returned error message")
        else:
            print("SUCCESS: Tool execution successful")
            
    except Exception as e:
        print(f"ERROR: retrieve_financial_metrics failed: {e}")

    print(f"\n3. Testing get_company_overview for {company}...")
    try:
        result = await tools.get_company_overview(company)
        print(f"Result preview: {str(result)[:200]}...")
    except Exception as e:
        print(f"ERROR: get_company_overview failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_tools())
