# seed_msft.py
import asyncio
import os
import sys
import logging
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.collectors.sec_edgar import SECDataCollector
from src.data.processors.document_parser import DocumentProcessor
from src.rag.core import ProductionRAGSystem

logging.basicConfig(level=logging.INFO)

async def seed_msft():
    load_dotenv()
    ticker = "MSFT"
    
    print(f"\n[LAUNCH] Starting Data Seed for {ticker}...")
    
    try:
        collector = SECDataCollector()
        processor = DocumentProcessor()
        rag = ProductionRAGSystem()  # Uses HuggingFace by default
        print("   [OK] Components Initialized")
    except Exception as e:
        print(f"   [ERROR] Component Init Failed: {e}")
        return
    
    print("   [FETCH] Fetching SEC Data...")
    raw_data = collector.company_facts(ticker)
    
    if not raw_data:
        print("   [ERROR] SEC Fetch Failed.")
        return
    print(f"   [OK] Data Received (Entity: {raw_data.get('entityName')})")
    
    print("   [PROCESS] Processing into Documents...")
    documents = processor.process_sec_facts(raw_data, ticker)
    
    if not documents:
        print("   [ERROR] No documents created.")
        return
    print(f"   [OK] Generated {len(documents)} Documents")
    
    print("   [SAVE] Indexing to Vector Database...")
    ids = rag.add_company_data(documents)
    
    if ids:
        print(f"   [SUCCESS] {len(ids)} chunks stored for MSFT!")
    else:
        print("   [ERROR] Vector Storage Failed.")

if __name__ == "__main__":
    asyncio.run(seed_msft())
