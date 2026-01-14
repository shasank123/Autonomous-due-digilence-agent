"""
Diagnostic script to check all companies in vector store
"""
import sys
sys.path.append('src')

from rag.core import ProductionRAGSystem

def check_all_companies():
    rag = ProductionRAGSystem()
    print(f"RAG initialized at: {rag.persist_directory}")
    
    # Get stats
    stats = rag.get_stats()
    print(f"Total documents: {stats.get('total_documents', 0)}")
    
    # Check specific companies
    companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']
    
    print("\nCompany metrics count:")
    for company in companies:
        metrics = rag.get_company_metrics(company)
        count = len(metrics) if metrics else 0
        print(f"  {company}: {count} metrics")
    
    # Also try to get all unique companies from the collection
    print("\n--- Checking collection directly ---")
    if hasattr(rag.vector_store, '_collection'):
        try:
            # Get a sample of documents to see what companies exist
            results = rag.vector_store._collection.get(
                limit=100,
                include=['metadatas']
            )
            
            if results and 'metadatas' in results:
                companies_found = set()
                for metadata in results['metadatas']:
                    if metadata and 'company' in metadata:
                        companies_found.add(metadata['company'])
                
                print(f"Companies found in sample: {sorted(companies_found)}")
                print(f"Total unique companies: {len(companies_found)}")
        except Exception as e:
            print(f"Error accessing collection: {e}")

if __name__ == "__main__":
    check_all_companies()
