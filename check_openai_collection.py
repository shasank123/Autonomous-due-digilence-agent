"""
Check OpenAI collection for all companies
"""
import chromadb
from chromadb.config import Settings

def check_openai_collection():
    print("=== Checking financial_data_openai collection ===")
    
    client = chromadb.PersistentClient(
        path="./data/vector_stores/financial_data",
        settings=Settings(anonymized_telemetry=False)
    )
    
    coll = client.get_collection("financial_data_openai")
    count = coll.count()
    print(f"Total documents: {count}")
    
    # Get larger sample to find all companies
    sample = coll.get(limit=1000, include=['metadatas'])
    companies = set()
    for metadata in sample['metadatas']:
        if metadata and 'company' in metadata:
            companies.add(metadata['company'])
    
    print(f"\nCompanies found: {sorted(companies)}")
    print(f"Total unique companies: {len(companies)}")
    
    # Count per company
    print("\nDocuments per company:")
    company_counts = {}
    
    # Get all metadatas
    all_data = coll.get(include=['metadatas'])
    for metadata in all_data['metadatas']:
        if metadata and 'company' in metadata:
            company = metadata['company']
            company_counts[company] = company_counts.get(company, 0) + 1
    
    for company, cnt in sorted(company_counts.items()):
        print(f"  {company}: {cnt}")

if __name__ == "__main__":
    check_openai_collection()
