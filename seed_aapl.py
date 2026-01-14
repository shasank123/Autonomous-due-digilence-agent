# seed_aapl.py - Clean version without emojis
import os
import sys
import asyncio
import logging
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.collectors.sec_edgar import SECDataCollector
from src.data.processors.document_parser import DocumentProcessor
from src.rag.core import ProductionRAGSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataSeeder")

async def seed_ticker(ticker: str):
    load_dotenv()
    
    print(f"\n>>> Starting Data Seed for {ticker}...")
    
    # 1. Initialize Components
    try:
        collector = SECDataCollector()
        processor = DocumentProcessor()
        # Use OpenAI embeddings (more stable on Windows)
        from pathlib import Path
        import os
        base_path = Path(os.getcwd()) / "data" / "vector_stores" / "financial_data_v2"
        rag = ProductionRAGSystem(persist_directory=str(base_path), embedding_type="openai")
        print("   [OK] Components Initialized with OpenAI embeddings")
    except Exception as e:
        print(f"   [ERROR] Component Init Failed: {e}")
        return

    # 2. Fetch Data
    print("   [FETCH] Getting SEC Data...")
    raw_data = collector.company_facts(ticker)
    
    if not raw_data:
        print("   [ERROR] SEC Fetch Failed. Check your Internet/VPN or Rate Limits.")
        return
    print(f"   [OK] Data Received (Entity: {raw_data.get('entityName')})")

    # 3. Process Data
    print("   [PROCESS] Creating Documents...")
    documents = processor.process_sec_facts(raw_data, ticker)
    
    if not documents:
        print("   [ERROR] Document Processing Failed (No documents created).")
        return
    print(f"   [OK] Generated {len(documents)} Documents")

    # 4. Ingest into RAG
    print("   [STORE] Indexing to Vector Database...")
    ids = rag.add_company_data(documents)
    
    if ids:
        print(f"   [SUCCESS] {len(ids)} text chunks stored in RAG.")
        print("   >>> You can now run the Analysis for this company.")
    else:
        print("   [ERROR] Vector Storage Failed.")

if __name__ == "__main__":
    asyncio.run(seed_ticker("AAPL"))
