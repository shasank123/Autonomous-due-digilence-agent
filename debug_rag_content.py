# debug_rag_content.py
"""
Debug script to check what documents are actually in the RAG system
and their metadata fields.
"""
import sys
import os
import logging
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.rag.core import ProductionRAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAGDebug")

def debug_rag():
    try:
        load_dotenv()
        
        # Initialize RAG
        rag = ProductionRAGSystem()
        logger.info(f"RAG Initialized at: {rag.persist_directory}")
        
        # Check collection stats
        try:
            # For Chroma wrapper, access underlying collection for stats
            count = rag.vector_store._collection.count()
            logger.info(f"Total documents in collection: {count}")
        except:
            logger.info("Could not get direct count, skipping.")

        # Helper to print docs
        def print_docs(docs, label):
            logger.info(f"\n--- {label} ({len(docs)} found) ---")
            for i, doc in enumerate(docs[:3]): # Show top 3
                logger.info(f"Doc {i+1}:")
                logger.info(f"  Content: {doc.page_content[:150]}...")
                logger.info(f"  Metadata: {doc.metadata}")

        # 1. Search for AAPL Contracts
        logger.info("\n1. Searching for AAPL Contracts...")
        results = rag.vector_store.similarity_search(
            "contract agreement", 
            k=5,
            filter={"company": "AAPL"}
        )
        print_docs(results, "AAPL Contract Results")
        
        # 2. Search for AAPL Market Docs
        logger.info("\n2. Searching for AAPL Market Docs...")
        results = rag.vector_store.similarity_search(
            "market competition",
            k=5,
            filter={"company": "AAPL"}
        )
        print_docs(results, "AAPL Market Results")
        
        # 3. Check AAPL Doc Types Summary
        logger.info("\n3. Inspecting Metadata types for AAPL docs...")
        sample_results = rag.vector_store.similarity_search("AAPL", k=50, filter={"company": "AAPL"})
        
        doc_types = {}
        for doc in sample_results:
            dt = doc.metadata.get('doc_type', 'UNKNOWN')
            doc_types[dt] = doc_types.get(dt, 0) + 1
            
        logger.info(f"\nFound AAPL Doc Types: {doc_types}")

    except Exception as e:
        logger.error(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_rag()
