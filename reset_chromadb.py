# reset_chromadb.py - Forcefully resets corrupted ChromaDB
import os
import shutil
import time
from pathlib import Path

def force_delete(path):
    """Try multiple methods to delete the directory"""
    if not os.path.exists(path):
        print(f"[SKIP] {path} doesn't exist")
        return True
    
    try:
        # Method 1: Direct delete
        shutil.rmtree(path)
        print(f"[OK] Deleted {path}")
        return True
    except PermissionError:
        print(f"[LOCKED] {path} is locked by a process")
        print("       Please:")
        print("       1. Close any running API servers")
        print("       2. Close any Python processes using ChromaDB")
        print("       3. Try again")
        return False
    except Exception as e:
        print(f"[ERROR] Could not delete: {e}")
        return False

if __name__ == "__main__":
    print("\n=== ChromaDB Reset ===\n")
    
    paths = [
        "data/vector_stores/financial_data",
        "data/vector_stores/financial_data_v2"
    ]
    
    success = True
    for p in paths:
        if not force_delete(p):
            success = False
    
    if success:
        print("\n[SUCCESS] All databases cleared!")
        print("[NEXT STEP] Run: python seed_aapl.py")
    else:
        print("\n[FAILED] Some databases could not be deleted")
        print("[ACTION] Stop the API server and try again")
