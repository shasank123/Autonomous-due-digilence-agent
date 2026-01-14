# debug_agent_view.py
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.getcwd()))
from src.rag.core import ProductionRAGSystem

def check_agent_vision():
    print("--- AGENT VISION TEST ---")
    rag = ProductionRAGSystem()
    
    company = "AAPL"
    
    # 1. Check Collection Direct Access (What ensure_company_data uses)
    print(f"\n1. Testing Metadata Lookup for '{company}'...")
    try:
        results = rag.vector_store._collection.get(
            where={"company": company},
            include=['metadatas']
        )
        count = len(results['ids'])
        print(f"   Found {count} documents via Metadata.")
        
        if count > 0:
            print(f"   Sample Metadata: {results['metadatas'][0]}")
        else:
            print("   [ERROR] Agent sees 0 documents via Metadata lookup.")
            print("   (This is why ensure_company_data is failing)")
            
    except Exception as e:
        print(f"   [ERROR] Lookup Crashed: {e}")

    # 2. Check Semantic Search (What the Tools use)
    print(f"\n2. Testing Semantic Search for '{company}'...")
    docs = rag.query(f"{company} revenue", company=company, k=1)
    if docs:
        print(f"   [OK] Agent CAN find data via search.")
    else:
        print(f"   [ERROR] Agent CANNOT find data via search.")

if __name__ == "__main__":
    load_dotenv()
    check_agent_vision()