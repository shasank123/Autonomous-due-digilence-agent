import asyncio
import logging
from src.tools.financial_tools import FinancialTools
from src.rag.core import ProductionRAGSystem
from src.data.collectors.sec_edgar import SECDataCollector
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_retrieval():
    load_dotenv()
    
    logger.info("Initializing components...")
    rag_system = ProductionRAGSystem()
    sec_collector = SECDataCollector()
    financial_tools = FinancialTools(rag_system, sec_collector)
    
    company = "AAPL"
    metrics = ["Revenue", "NetIncomeLoss", "Assets", "Liabilities", "StockholdersEquity"]
    
    logger.info(f"Retrieving metrics for {company}...")
    result = await financial_tools.retrieve_financial_metrics(company, metrics)
    
    print("\n--- Retrieval Result ---")
    print(result)
    
    if "[DATA]" in result:
        print("\n[OK] SUCCESS: Data retrieved successfully.")
    else:
        print("\n[ERROR] FAILURE: No data retrieved.")

if __name__ == "__main__":
    asyncio.run(test_retrieval())
