#!/usr/bin/env python3
"""
Seed RAG system with data for any company ticker.
Usage: python seed_company_data.py TICKER
Example: python seed_company_data.py MSFT
"""

import sys
import os
import logging
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag.core import ProductionRAGSystem
from src.data.collectors.sec_edgar import SECDataCollector
from src.data.processors.document_parser import DocumentProcessor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def seed_company_data(ticker: str):
    """Fetch and seed financial data for a given company ticker"""
    
    ticker = ticker.upper().strip()
    logger.info("=" * 70)
    logger.info(f"SEEDING RAG SYSTEM WITH {ticker} DATA")
    logger.info("=" * 70)
    
    # Initialize components
    logger.info("\n[1/4] Initializing components...")
    collector = SECDataCollector()
    processor = DocumentProcessor()
    base_path = Path(os.getcwd()) / "data" / "vector_stores" / "financial_data"
    rag = ProductionRAGSystem(persist_directory=str(base_path))
    logger.info(f"   RAG system path: {base_path}")
    
    # Fetch company data
    logger.info(f"\n[2/4] Fetching {ticker} data from SEC...")
    try:
        company_facts = collector.company_facts(ticker)
        if not company_facts:
            logger.error(f"   FAILED: Could not fetch data for {ticker}")
            logger.error("   This could mean:")
            logger.error("   - Invalid ticker symbol")
            logger.error("   - Company not registered with SEC")
            logger.error("   - SEC API temporarily unavailable")
            return False
            
        entity_name = company_facts.get('entityName', ticker)
        metrics_count = len(company_facts.get('facts', {}).get('us-gaap', {}))
        logger.info(f"   SUCCESS: Fetched SEC data for {entity_name}")
        logger.info(f"   Available metrics: {metrics_count} US-GAAP items")
        
    except Exception as e:
        logger.error(f"   FAILED: Error fetching data: {e}")
        return False
    
    # Process into documents
    logger.info("\n[3/4] Processing data into documents...")
    try:
        documents = processor.process_sec_facts(company_facts, ticker)
        if not documents:
            logger.error("   FAILED: Could not process SEC facts")
            return False
            
        logger.info(f"   SUCCESS: Created {len(documents)} documents")
        logger.info(f"\n   Sample document types:")
        doc_types = {}
        for doc in documents[:50]:  # Sample first 50
            dtype = doc.metadata.get('doc_type', 'unknown')
            doc_types[dtype] = doc_types.get(dtype, 0) + 1
        for dtype, count in doc_types.items():
            logger.info(f"     - {dtype}: {count}")
            
    except Exception as e:
        logger.error(f"   FAILED: Processing error: {e}")
        return False
    
    # Add to RAG
    logger.info("\n[4/4] Adding documents to RAG system...")
    try:
        doc_ids = rag.add_company_data(documents)
        if not doc_ids:
            logger.error("   FAILED: RAG system did not return document IDs")
            return False
            
        logger.info(f"   SUCCESS: Added {len(doc_ids)} documents to RAG")
        
    except Exception as e:
        logger.error(f"   FAILED: Error adding to RAG: {e}")
        return False
    
    # Verify
    logger.info("\n[5/5] Verifying data ingestion...")
    try:
        # Test specific queries
        test_queries = [
            ("Assets", "Assets financial data"),
            ("Revenue", "Revenue financial data"),
            ("Net Income", "NetIncomeLoss profit income")
        ]
        
        found_any = False
        for metric_name, query in test_queries:
            results = rag.query_with_similarity_scores(
                question=f"{ticker} {query}",
                company=ticker,
                k=2,
                score_threshold=5.0
            )
            if results:
                found_any = True
                logger.info(f"   ✓ {metric_name}: {len(results)} results (best score: {results[0][1]:.3f})")
            else:
                logger.warning(f"   ✗ {metric_name}: No results found")
        
        # Get all metrics
        metrics = rag.get_company_metrics(ticker)
        logger.info(f"\n   Total unique metrics available: {len(metrics)}")
        
        if not found_any:
            logger.warning("\n   WARNING: Verification queries returned no results.")
            logger.warning("   Data was added but similarity scores might be too high.")
            logger.warning("   The system should still work, but consider checking thresholds.")
        
    except Exception as e:
        logger.warning(f"   Verification error: {e}")
    
    logger.info("\n" + "=" * 70)
    logger.info("SEED COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nThe {ticker} data has been successfully loaded into the RAG system.")
    logger.info("If you're using Docker, remember to rebuild the container:")
    logger.info("  docker-compose build api")
    logger.info("  docker-compose restart api")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python seed_company_data.py TICKER")
        print("Example: python seed_company_data.py MSFT")
        sys.exit(1)
    
    ticker = sys.argv[1]
    success = seed_company_data(ticker)
    sys.exit(0 if success else 1)
