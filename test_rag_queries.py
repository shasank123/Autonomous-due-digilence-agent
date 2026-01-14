#!/usr/bin/env python3
"""Test RAG System queries for AAPL to debug data conflicts"""

import sys
import os
import logging
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag.core import ProductionRAGSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rag_queries():
    # Initialize RAG
    base_path = Path(os.getcwd()) / "data" / "vector_stores" / "financial_data"
    rag = ProductionRAGSystem(persist_directory=str(base_path))
    
    logger.info("=" * 60)
    logger.info("Testing RAG System Queries for AAPL")
    logger.info("=" * 60)
    
    # Test 1: Revenue query
    logger.info("\n--- Test 1: Revenue Query ---")
    try:
        results = rag.query_with_similarity_scores(
            question="AAPL Revenue financial data values",
            company="AAPL",
            metric_type="financial_metric",
            k=5,
            score_threshold=3.0  # Very relaxed
        )
        logger.info(f"Found {len(results)} results for Revenue")
        for i, (doc, score) in enumerate(results[:3]):
            logger.info(f"\n  Result {i+1} (score: {score:.3f}):")
            logger.info(f"    Content: {doc.page_content[:200]}...")
    except Exception as e:
        logger.error(f"Revenue query failed: {e}")
    
    # Test 2: Get company metrics
    logger.info("\n--- Test 2: Get Company Metrics ---")
    try:
        metrics = rag.get_company_metrics("AAPL")
        logger.info(f"Found {len(metrics)} metrics for AAPL")
        logger.info(f"Sample metrics: {metrics[:10]}")
    except Exception as e:
        logger.error(f"Get metrics failed: {e}")
    
    # Test 3: Net Income query with different thresholds
    logger.info("\n--- Test 3: Net Income with Different Thresholds ---")
    for threshold in [1.0, 2.0, 3.0, 5.0]:
        try:
            results = rag.query_with_similarity_scores(
                question="AAPL NetIncomeLoss profit income",
                company="AAPL",
                k=3,
                score_threshold=threshold
            )
            logger.info(f"  Threshold {threshold}: {len(results)} results")
            if results:
                logger.info(f"    Best score: {results[0][1]:.3f}")
        except Exception as e:
            logger.error(f"  Threshold {threshold} failed: {e}")
    
    # Test 4: Assets query
    logger.info("\n--- Test 4: Assets Query ---")
    try:
        results = rag.query_with_similarity_scores(
            question="AAPL Assets financial data",
            company="AAPL",
            k=5,
            score_threshold=2.0
        )
        logger.info(f"Found {len(results)} results for Assets")
        for i, (doc, score) in enumerate(results[:2]):
            logger.info(f"\n  Result {i+1} (score: {score:.3f}):")
            logger.info(f"    Metadata: {doc.metadata}")
            logger.info(f"    Content: {doc.page_content[:150]}...")
    except Exception as e:
        logger.error(f"Assets query failed: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("RAG Query Tests Complete")
    logger.info("=" * 60)

if __name__ == "__main__":
    test_rag_queries()
