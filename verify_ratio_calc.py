
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag.core import ProductionRAGSystem
from src.tools.financial_tools import FinancialTools
from src.data.collectors.sec_edgar import SECDataCollector

async def verify():
    load_dotenv()
    print("Initializing Financial Tools...")
    
    # Initialize RAG
    base_path = Path(os.getcwd()) / "data" / "vector_stores" / "financial_data"
    rag = ProductionRAGSystem(persist_directory="c:/Users/polam/OneDrive/Desktop/agentic projects/autonomous due diligence agent/data/vector_stores/financial_data_v2", embedding_type="openai")
    
    # Initialize Tools
    tools = FinancialTools(rag, SECDataCollector())
    
    print("\n--- DEBUG: STRUCTURED DATA ---")
    data = tools.extract_structured_metrics("AAPL")
    print(f"Metrics Found: {list(data.get('metrics', {}).keys())}")
    print(f"Ratios Calculated: {data.get('ratios', {})}")
    
    print("\nTesting Ratio Calculation for AAPL...")
    result = await tools.analyze_financial_ratios("AAPL")
    
    print("\n--- RESULT ---")
    print(result)
    print("--------------\n")
    
    if "[CALCULATED]" in result:
        print("[OK] SUCCESS: Dynamic calculation working.")
    elif "[TREND]" in result:
        print("[NOTE] Found pre-calculated ratios (Dynamic calc not needed).")
    else:
        print("[FAIL] FAILURE: Could not calculate ratios.")

if __name__ == "__main__":
    asyncio.run(verify())
