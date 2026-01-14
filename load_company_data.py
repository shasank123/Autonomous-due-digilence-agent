"""
Quick script to load company data into the RAG system
Usage: python load_company_data.py MSFT AAPL GOOGL
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.collectors.sec_edgar import SECDataCollector
from src.data.processors.document_parser import DocumentProcessor
from src.rag.core import ProductionRAGSystem

def load_company(ticker: str):
    """Load a single company's data into the RAG system"""
    print(f"\n{'='*60}")
    print(f"Loading data for {ticker}")
    print(f"{'='*60}")
    
    # Initialize components
    collector = SECDataCollector()
    processor = DocumentProcessor()
    rag_system = ProductionRAGSystem()
    
    try:
        # 1. Fetch SEC data
        print(f"[1/3] Fetching SEC data for {ticker}...")
        company_data = collector.company_facts(ticker)
        
        if not company_data:
            print(f"[ERROR] Failed to fetch data for {ticker}")
            return False
        
        print(f"[OK] Fetched data for {company_data.get('entityName', ticker)}")
        
        # 2. Process into documents
        print(f"[2/3] Processing documents...")
        documents = processor.process_sec_facts(company_data, ticker)
        
        if not documents:
            print(f"[ERROR] No documents created for {ticker}")
            return False
        
        print(f"[OK] Created {len(documents)} documents")
        
        # 3. Add to RAG system
        print(f"[3/3] Adding to vector database...")
        doc_ids = rag_system.add_company_data(documents)
        
        if doc_ids:
            print(f"[SUCCESS] Added {len(doc_ids)} document chunks for {ticker}")
            
            # Show available metrics
            metrics = rag_system.get_company_metrics(ticker)
            print(f"[INFO] Available metrics: {len(metrics)}")
            if metrics:
                print(f"       Sample: {', '.join(metrics[:5])}...")
            
            return True
        else:
            print(f"[ERROR] Failed to add documents to database")
            return False
            
    except Exception as e:
        print(f"[ERROR] Failed to load {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Load multiple companies"""
    # Get tickers from command line, or use defaults
    if len(sys.argv) > 1:
        tickers = [t.upper() for t in sys.argv[1:]]
    else:
        # Default companies
        tickers = ["MSFT", "AAPL", "GOOGL"]
        print(f"No tickers specified. Loading defaults: {', '.join(tickers)}")
    
    print(f"\n{'#'*60}")
    print(f"# Company Data Loader")
    print(f"# Loading {len(tickers)} companies into RAG system")
    print(f"{'#'*60}")
    
    success_count = 0
    for ticker in tickers:
        if load_company(ticker):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {success_count}/{len(tickers)} companies loaded successfully")
    print(f"{'='*60}")
    
    # Show database stats
    try:
        rag = ProductionRAGSystem()
        stats = rag.get_stats()
        print(f"\nDatabase Stats:")
        print(f"  Total documents: {stats.get('total documents', 0)}")
        print(f"  Location: {stats.get('persist_directory', 'Unknown')}")
    except Exception as e:
        print(f"[WARN] Could not fetch stats: {e}")

if __name__ == "__main__":
    main()
