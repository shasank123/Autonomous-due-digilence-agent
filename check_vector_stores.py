"""
Check all vector store directories for company data
"""
import sys
sys.path.append('src')

from rag.core import ProductionRAGSystem

def check_vector_store(path):
    print(f"\n=== Checking: {path} ===")
    try:
        rag = ProductionRAGSystem(persist_directory=path)
        stats = rag.get_stats()
        print(f"Total documents: {stats.get('total_documents', 0)}")
        
        # Check for companies
        if hasattr(rag.vector_store, '_collection'):
            results = rag.vector_store._collection.get(
                limit=500,  # Get more to find all companies
                include=['metadatas']
            )
            
            if results and 'metadatas' in results:
                companies_found = set()
                for metadata in results['metadatas']:
                    if metadata and 'company' in metadata:
                        companies_found.add(metadata['company'])
                
                print(f"Unique companies: {sorted(companies_found)}")
                print(f"Total companies: {len(companies_found)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    paths = [
        "./data/vector_stores/financial_data",
        "./data/vector_stores/financial_data_v2",
        "./data/vector_stores/financial_data_test",
    ]
    
    for path in paths:
        check_vector_store(path)
