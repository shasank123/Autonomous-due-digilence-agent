import os
import sys
from dotenv import load_dotenv

# Setup paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.rag.core import ProductionRAGSystem

load_dotenv()

def inspect():
    print("--- Inspecting Vector Database ---")
    rag = ProductionRAGSystem()
    
    # 1. Check Total Count
    stats = rag.get_stats()
    print(f"Database Stats: {stats}")
    
    if stats.get('total_documents', 0) == 0:
        print("[ERROR] CRITICAL: Database is empty! Re-run seed_data.py")
        return

    # 2. Test Financial Query
    print("\nTesting Query: 'AAPL Revenue'")
    # We ask for raw results to see the actual scores
    results = rag.vector_store.similarity_search_with_score(
        "AAPL Revenue", k=3, filter={"company": "AAPL"}
    )
    
    for doc, score in results:
        print(f"\n[Score: {score:.4f}] content: {doc.page_content[:100]}...")
        if score > 0.6:
            print("   [WARN]  NOTE: This would be hidden by the current 0.6 threshold!")
        else:
            print("   [OK] This is visible to agents.")

if __name__ == "__main__":
    inspect()