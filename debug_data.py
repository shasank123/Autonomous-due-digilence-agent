import asyncio
import os
import logging
from dotenv import load_dotenv
from src.data.collectors.sec_edgar import SECDataCollector
from src.data.processors.document_parser import DocumentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def debug_data_pipeline(ticker: str):
    load_dotenv()
    
    logger.info(f"--- Debugging Data Pipeline for {ticker} ---")
    
    # 1. Initialize Collector
    logger.info("Initializing SECDataCollector...")
    try:
        collector = SECDataCollector()
        logger.info("SECDataCollector initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize SECDataCollector: {e}")
        return

    # 2. Fetch Data
    logger.info(f"Fetching company facts for {ticker}...")
    try:
        company_facts = collector.company_facts(ticker)
        if company_facts:
            logger.info(f"Successfully fetched data. Keys: {company_facts.keys()}")
            if 'facts' in company_facts:
                 logger.info(f"Facts keys: {company_facts['facts'].keys()}")
                 if 'us-gaap' in company_facts['facts']:
                     logger.info(f"US-GAAP metrics count: {len(company_facts['facts']['us-gaap'])}")
                     # Print some sample keys
                     sample_keys = list(company_facts['facts']['us-gaap'].keys())[:10]
                     logger.info(f"Sample US-GAAP keys: {sample_keys}")
                     
                     if 'Assets' in company_facts['facts']['us-gaap']:
                         assets_data = company_facts['facts']['us-gaap']['Assets']
                         if 'units' in assets_data and 'USD' in assets_data['units']:
                             values = assets_data['units']['USD']
                             logger.info(f"Total Assets entries: {len(values)}")
                             logger.info("First 5 entries (dates):")
                             for v in values[:5]:
                                 logger.info(f"  {v.get('end')} (val: {v.get('val')})")
                             logger.info("Last 5 entries (dates):")
                             for v in values[-5:]:
                                 logger.info(f"  {v.get('end')} (val: {v.get('val')})")
        else:
            logger.error("Fetched data is empty or None.")
            return
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        return

    # 3. Process Data
    logger.info("Initializing DocumentProcessor...")
    try:
        processor = DocumentProcessor()
        logger.info("DocumentProcessor initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize DocumentProcessor: {e}")
        return

    logger.info("Processing SEC facts...")
    try:
        documents = processor.process_sec_facts(company_facts, ticker)
        logger.info(f"Processed {len(documents)} documents.")
        
        if documents:
            logger.info("--- Sample Document Content ---")
            for i, doc in enumerate(documents[:3]):
                logger.info(f"Doc {i+1} Metadata: {doc.metadata}")
                logger.info(f"Doc {i+1} Content Preview: {doc.page_content[:200]}...")
        else:
            logger.warning("No documents generated from facts.")
            
    except Exception as e:
        logger.error(f"Failed to process facts: {e}")
        return

if __name__ == "__main__":
    asyncio.run(debug_data_pipeline("AAPL"))
