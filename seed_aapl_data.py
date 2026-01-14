#!/usr/bin/env python3
"""Seed RAG system with AAPL data to resolve data conflicts"""

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

def seed_aapl_data():
    logger.info("=" * 70)
    logger.info("SEEDING RAG SYSTEM WITH AAPL DATA")
    logger.info("=" * 70)
    
    # 1. Initialize components
    logger.info("\n[1/4] Initializing components...")
    collector = SECDataCollector()
    processor = DocumentProcessor()
    
    # Initialize RAG with the correct path
    base_path = Path(os.getcwd()) / "data" / "vector_stores" / "financial_data"
    rag = ProductionRAGSystem(persist_directory=str(base_path))
    logger.info(f"   RAG system initialized at: {base_path}")
    
    # 2. Fetch AAPL data from SEC
    logger.info("\n[2/4] Fetching AAPL data from SEC...")
    company_facts = collector.company_facts("AAPL")
    
    if not company_facts:
        logger.error("   FAILED: Could not fetch AAPL data from SEC")
        return False
    
    logger.info(f"   SUCCESS: Fetched SEC data for {company_facts.get('entityName', 'AAPL')}")
    logger.info(f"   Metrics available: {len(company_facts.get('facts', {}).get('us-gaap', {}))} US-GAAP items")
    
    # 3. Process into documents
    logger.info("\n[3/4] Processing data into documents...")
    documents = processor.process_sec_facts(company_facts, "AAPL")
    
    if not documents:
        logger.error("   FAILED: Could not process SEC facts into documents")
        return False
    
    logger.info(f"   SUCCESS: Created {len(documents)} documents")
    
    # Show sample
    logger.info("\n   Sample documents:")
    for i, doc in enumerate(documents[:3]):
        logger.info(f"     Doc {i+1}: {doc.metadata.get('metric', doc.metadata.get('doc_type'))}")
        logger.info(f"          Content: {doc.page_content[:80]}...")
    
    # 4. Add to RAG system
    logger.info("\n[4/4] Adding documents to RAG system...")
    try:
        doc_ids = rag.add_company_data(documents)
        
        if not doc_ids:
            logger.error("   FAILED: RAG system did not return document IDs")
            return False
        
        logger.info(f"   SUCCESS: Added {len(doc_ids)} documents to RAG system")
        logger.info(f"   Document IDs: {doc_ids[:5]}... (showing first 5)")
        
    except Exception as e:
        logger.error(f"   FAILED: Error adding to RAG: {e}")
        return False
    
    # 5. Verification
    logger.info("\n[5/5] Verifying data ingestion...")
    try:
        # Test query
        test_results = rag.query_with_similarity_scores(
            question="AAPL Assets financial data",
            company="AAPL",
            k=3,
            score_threshold=5.0  # Very relaxed for testing
        )
        
        logger.info(f"   Test query found {len(test_results)} results")
        if test_results:
            logger.info(f"   Best match score: {test_results[0][1]:.3f}")
            logger.info(f"   Content preview: {test_results[0][0].page_content[:100]}...")
        
        # Get metrics
        metrics = rag.get_company_metrics("AAPL")
        logger.info(f"   Available metrics: {len(metrics)} unique metrics found")
        if metrics:
            logger.info(f"   Sample: {metrics[:5]}")
        
    except Exception as e:
        logger.error(f"   Verification error: {e}")
    
    logger.info("\n" + "=" * 70)
    logger.info("SEED COMPLETE")
    logger.info("=" * 70)
    logger.info("\nThe AAPL data has been successfully loaded into the RAG system.")
    logger.info("The financial agents should now be able to access this data.")
    
    return True

if __name__ == "__main__":
    success = seed_aapl_data()
    sys.exit(0 if success else 1)
