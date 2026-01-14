# seed_data.py
import os
import sys
import asyncio
import logging
from dotenv import load_dotenv

# Setup paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Project Modules
from src.data.collectors.sec_edgar import SECDataCollector
from src.data.processors.document_parser import DocumentProcessor
from src.rag.core import ProductionRAGSystem

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataSeeder")

async def seed_ticker(ticker: str):
    load_dotenv()
    
    print(f"\n[LAUNCH] Starting Data Seed for {ticker}...")
    
    # 1. Initialize Components
    try:
        collector = SECDataCollector()
        processor = DocumentProcessor()
        rag = ProductionRAGSystem()
        print("   [OK] Components Initialized")
    except Exception as e:
        print(f"   [ERROR] Component Init Failed: {e}")
        return

    # 2. Fetch Data (The part that was failing silently)
    print("   📥 Fetching SEC Data...")
    raw_data = collector.company_facts(ticker)
    
    if not raw_data:
        print("   [ERROR] SEC Fetch Failed. Check your Internet/VPN or Rate Limits.")
        return
    print(f"   [OK] Data Received (Entity: {raw_data.get('entityName')})")

    # 3. Process Data
    print("   ⚙️  Processing into Documents...")
    documents = processor.process_sec_facts(raw_data, ticker)
    
    if not documents:
        print("   [ERROR] Document Processing Failed (No documents created).")
        return
    print(f"   [OK] Generated {len(documents)} Documents")

    # 4. Ingest into RAG
    print("   [SAVE] Indexing to Vector Database...")
    ids = rag.add_company_data(documents)
    
    if ids:
        print(f"   🎉 SUCCESS! {len(ids)} text chunks stored in RAG.")
        print("   👉 You can now run the UI Analysis for this company.")
    else:
        print("   [ERROR] Vector Storage Failed.")

if __name__ == "__main__":
    # You can add more tickers here
    asyncio.run(seed_ticker("AAPL"))