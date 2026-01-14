import sys
import os
import asyncio
import logging

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Mock environment variables
os.environ["OPENAI_API_KEY"] = "test_key"
os.environ["REDIS_HOST"] = "localhost"
os.environ["REDIS_PORT"] = "6379"
os.environ["SEC_EDGAR_EMAIL"] = "test@example.com"

try:
    from agents.orchestrator import DueDiligenceOrchestrator
    print("Successfully imported DueDiligenceOrchestrator")
except Exception as e:
    print(f"Failed to import DueDiligenceOrchestrator: {e}")
    sys.exit(1)

async def main():
    try:
        orchestrator = DueDiligenceOrchestrator()
        print("Successfully initialized DueDiligenceOrchestrator")
    except Exception as e:
        print(f"Failed to initialize DueDiligenceOrchestrator: {e}")
        # Print full traceback
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
