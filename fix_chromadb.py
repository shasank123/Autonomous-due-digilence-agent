"""
Alternative ChromaDB fix - Rename instead of delete
"""
import os
import shutil
from pathlib import Path
from datetime import datetime

def fix_chromadb_rename():
    """Rename corrupted ChromaDB to backup and create fresh"""
    
    # Path to the vector store
    vector_store_path = Path("./data/vector_stores/financial_data_v2")
    
    if vector_store_path.exists():
        # Create backup name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = Path(f"./data/vector_stores/financial_data_v2_corrupted_{timestamp}")
        
        print(f"[INFO] Found existing vector store at: {vector_store_path}")
        print(f"[INFO] Renaming to: {backup_path.name}")
        
        try:
            vector_store_path.rename(backup_path)
            print("[OK] Successfully renamed corrupted database")
            print(f"[INFO] Backup saved at: {backup_path}")
            print("[INFO] A fresh database will be created on next startup")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to rename: {e}")
            print("\n[SOLUTION] Please close all applications that might be using the database:")
            print("  1. Stop the API server if it's running")
            print("  2. Close any Python processes")
            print("  3. Run this script again")
            return False
    else:
        print(f"[INFO] No existing vector store found")
        print("[INFO] A new database will be created on startup")
        return True

if __name__ == "__main__":
    print("=" * 60)
    print("ChromaDB Fix Utility (Rename Method)")
    print("=" * 60)
    
    if fix_chromadb_rename():
        print("\n[SUCCESS] ChromaDB has been reset!")
        print("\n[NEXT STEPS]:")
        print("  1. Restart your API server")
        print("  2. The system will create a fresh database")
        print("  3. You'll need to reload your data")
    else:
        print("\n[FAILED] Please follow the solution above")
