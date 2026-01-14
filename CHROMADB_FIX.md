# QUICK FIX: Use in-memory ChromaDB to avoid corruption
# src/use_memory_chromadb.py

# This is a temporary workaround for Windows ChromaDB corruption
# Add this to orchestrator.py temporarily

# Instead of:
# self.rag_system = ProductionRAGSystem(persist_directory=str(base_path), embedding_type="openai")

# Use in-memory (data won't persist but won't corrupt):
# from chromadb.config import Settings
# Settings(anonymized_telemetry=False, allow_reset=True, is_persistent=False)

print("To use in-memory ChromaDB (temporary fix):")
print("1. Data won't persist between restarts")
print("2. But API will start without corruption")
print("3. Need to reseed data each time")
print("")
print("Better solution: Use Docker which has a stable ChromaDB environment")
