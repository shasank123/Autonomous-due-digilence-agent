# diagnose_vector_search.py
"""
Diagnostic script to identify why vector search returns empty results.
"""
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Setup paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Project Modules
from src.rag.core import ProductionRAGSystem

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("VectorSearchDiagnostics")

def main():
    load_dotenv()
    
    print("\n" + "="*60)
    print("[SEARCH] VECTOR SEARCH DIAGNOSTICS")
    print("="*60 + "\n")
    
    # Initialize RAG System
    try:
        base_path = Path(os.getcwd()) / "data" / "vector_stores" / "financial_data"
        rag = ProductionRAGSystem(persist_directory=str(base_path), embedding_type="openai")
        print("[OK] RAG System Initialized\n")
    except Exception as e:
        print(f"[ERROR] RAG Init Failed: {e}")
        return
    
    # 1. Check Database Statistics
    print("-" * 60)
    print("[1]  DATABASE STATISTICS")
    print("-" * 60)
    stats = rag.get_stats()
    print(f"Total Documents: {stats.get('total_documents', 0)}")
    print(f"Embedding Type: {stats.get('embedding_type', 'Unknown')}")
    print(f"Persist Directory: {stats.get('persist_directory', 'Unknown')}")
    
    if stats.get('total_documents', 0) == 0:
        print("\n[WARN]  WARNING: Database is empty!")
        print("   Run 'python seed_data.py' to populate data.\n")
        return
    
    # 2. List Companies in Database
    print("\n" + "-" * 60)
    print("[2]  COMPANIES IN DATABASE")
    print("-" * 60)
    
    try:
        # Get all documents to extract unique companies
        if hasattr(rag.vector_store, '_collection'):
            results = rag.vector_store._collection.get(
                include=['metadatas']
            )
            
            if results and 'metadatas' in results:
                companies = set()
                doc_types = {}
                
                for metadata in results['metadatas']:
                    if metadata and 'company' in metadata:
                        companies.add(metadata['company'])
                        
                        # Count doc types
                        doc_type = metadata.get('doc_type', 'unknown')
                        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                
                print(f"Found {len(companies)} companies:")
                for company in sorted(companies):
                    print(f"  • {company}")
                
                print(f"\nDocument type distribution:")
                for doc_type, count in sorted(doc_types.items()):
                    print(f"  • {doc_type}: {count} documents")
            else:
                print("[WARN]  No metadata found")
    except Exception as e:
        print(f"[ERROR] Error listing companies: {e}")
    
    # 3. Test Queries with Different Thresholds
    print("\n" + "-" * 60)
    print("[3]  TESTING QUERIES WITH DIFFERENT THRESHOLDS")
    print("-" * 60)
    
    test_company = "AAPL"  # Change this based on what's in your DB
    test_query = f"{test_company} Revenue financial data values"
    
    print(f"Query: \"{test_query}\"")
    print(f"Company Filter: {test_company}")
    print(f"doc_type Filter: financial_metric\n")
    
    thresholds = [0.5, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 999.0]
    
    for threshold in thresholds:
        try:
            results = rag.query_with_similarity_scores(
                question=test_query,
                company=test_company,
                metric_type="financial_metric",
                k=20,
                score_threshold=threshold
            )
            
            if results:
                print(f"Threshold {threshold:5.1f}: [OK] {len(results)} results")
                # Show first result's score
                if len(results) > 0:
                    doc, score = results[0]
                    print(f"              Best score: {score:.4f}")
                    print(f"              Content preview: {doc.page_content[:80]}...")
            else:
                print(f"Threshold {threshold:5.1f}: [ERROR] No results")
                
        except Exception as e:
            print(f"Threshold {threshold:5.1f}: [ERROR] Error: {e}")
    
    # 4. Test Query WITHOUT Filters
    print("\n" + "-" * 60)
    print("[4]  TESTING QUERY WITHOUT FILTERS")
    print("-" * 60)
    
    try:
        results = rag.query_with_similarity_scores(
            question=test_query,
            k=20,
            score_threshold=999.0  # Accept everything
        )
        
        print(f"Total results (no filters): {len(results)}")
        
        if results:
            print("\nTop 5 results:")
            for i, (doc, score) in enumerate(results[:5], 1):
                company = doc.metadata.get('company', 'N/A')
                doc_type = doc.metadata.get('doc_type', 'N/A')
                metric = doc.metadata.get('metric', 'N/A')
                print(f"\n  {i}. Score: {score:.4f}")
                print(f"     Company: {company} | Type: {doc_type} | Metric: {metric}")
                print(f"     Content: {doc.page_content[:100]}...")
    except Exception as e:
        print(f"[ERROR] Error: {e}")
    
    # 5. Sample Document Metadata
    print("\n" + "-" * 60)
    print("[5]  SAMPLE DOCUMENT METADATA")
    print("-" * 60)
    
    try:
        if hasattr(rag.vector_store, '_collection'):
            results = rag.vector_store._collection.get(
                limit=3,
                include=['metadatas', 'documents']
            )
            
            if results and 'metadatas' in results:
                for i, (metadata, doc) in enumerate(zip(results['metadatas'], results['documents']), 1):
                    print(f"\nDocument {i}:")
                    print(f"  Metadata: {metadata}")
                    print(f"  Content: {doc[:100]}...")
    except Exception as e:
        print(f"[ERROR] Error: {e}")
    
    print("\n" + "="*60)
    print("[OK] DIAGNOSTICS COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
