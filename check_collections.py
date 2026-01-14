"""
Check all collections in a vector store
"""
import chromadb
from chromadb.config import Settings

def check_collections(path):
    print(f"\n=== Checking collections in: {path} ===")
    try:
        client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        collections = client.list_collections()
        print(f"Found {len(collections)} collections:")
        
        for coll in collections:
            print(f"\n  Collection: {coll.name}")
            count = coll.count()
            print(f"  Documents: {count}")
            
            if count > 0:
                # Get sample to see companies
                sample = coll.get(limit=100, include=['metadatas'])
                companies = set()
                for metadata in sample['metadatas']:
                    if metadata and 'company' in metadata:
                        companies.add(metadata['company'])
                print(f"  Companies: {sorted(companies)}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    paths = [
        "./data/vector_stores/financial_data",
        "./data/vector_stores/financial_data_v2",
    ]
    
    for path in paths:
        check_collections(path)
