#!/usr/bin/env python3
"""Test trend calculation logic for Net Income and Liabilities"""

import sys
import os
import logging
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag.core import ProductionRAGSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_trend_calculations():
    base_path = Path(os.getcwd()) / "data" / "vector_stores" / "financial_data"
    rag = ProductionRAGSystem(persist_directory=str(base_path))
    
    logger.info("=" * 60)
    logger.info("Testing Trend Calculations for AAPL")
    logger.info("=" * 60)
    
    # Test NetIncomeLoss trend
    logger.info("\n--- NetIncomeLoss Trend ---")
    results = rag.query_with_similarity_scores(
        question="AAPL NetIncomeLoss profit income values",
        company="AAPL",
        k=20,
        score_threshold=2.0
    )
    
    logger.info(f"Found {len(results)} documents")
    
    # Extract period/value pairs
    period_values = {}
    for doc, score in results:
        content = doc.page_content
        if "NetIncomeLoss" in content:
            lines = content.split('\n')
            period = value = None
            for line in lines:
                if 'Period:' in line:
                    period = line.split(':')[1].strip()
                if 'Value:' in line:
                    value = line.split(':')[1].strip()
            if period and value:
                try:
                    val_clean = value.replace('USD', '').replace(',', '').strip().split()[0]
                    period_values[period] = float(val_clean)
                except:
                    pass
    
    logger.info(f"\nExtracted {len(period_values)} period/value pairs")
    sorted_periods = sorted(period_values.items(), key=lambda x: x[0], reverse=True)
    logger.info("Most recent 10 periods:")
    for period, value in sorted_periods[:10]:
        logger.info(f"  {period}: ${value:,.0f}")
    
    if len(sorted_periods) >= 2:
        latest_p, latest_v = sorted_periods[0]
        prev_p, prev_v = sorted_periods[1]
        growth = ((latest_v - prev_v) / abs(prev_v)) * 100
        logger.info(f"\nTrend: {growth:+.1f}%")
        logger.info(f"Latest: ${latest_v:,.0f} ({latest_p})")
        logger.info(f"Previous: ${prev_v:,.0f} ({prev_p})")

    # Test Liabilities trend
    logger.info("\n\n--- Liabilities Trend ---")
    results = rag.query_with_similarity_scores(
        question="AAPL Liabilities debt obligations values",
        company="AAPL",
        k=20,
        score_threshold=2.0
    )
    
    logger.info(f"Found {len(results)} documents")
    
    period_values = {}
    for doc, score in results:
        content = doc.page_content
        if "Liabilities" in content:
            lines = content.split('\n')
            period = value = None
            for line in lines:
                if 'Period:' in line:
                    period = line.split(':')[1].strip()
                if 'Value:' in line:
                    value = line.split(':')[1].strip()
            if period and value:
                try:
                    val_clean = value.replace('USD', '').replace(',', '').strip().split()[0]
                    period_values[period] = float(val_clean)
                except:
                    pass
    
    logger.info(f"\nExtracted {len(period_values)} period/value pairs")
    sorted_periods = sorted(period_values.items(), key=lambda x: x[0], reverse=True)
    logger.info("Most recent 10 periods:")
    for period, value in sorted_periods[:10]:
        logger.info(f"  {period}: ${value:,.0f}")
    
    if len(sorted_periods) >= 2:
        latest_p, latest_v = sorted_periods[0]
        prev_p, prev_v = sorted_periods[1]
        growth = ((latest_v - prev_v) / abs(prev_v)) * 100
        logger.info(f"\nTrend: {growth:+.1f}%")
        logger.info(f"Latest: ${latest_v:,.0f} ({latest_p})")
        logger.info(f"Previous: ${prev_v:,.0f} ({prev_p})")

if __name__ == "__main__":
    test_trend_calculations()
