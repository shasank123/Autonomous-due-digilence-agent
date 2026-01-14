
import asyncio
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agents.financial_analyst import create_financial_team
from src.rag.core import ProductionRAGSystem

# Configure logging to see tool output
logging.basicConfig(level=logging.INFO)

async def debug_agent_flow():
    load_dotenv()
    print("Initializing Financial Team...")
    
    # 1. Initialize Real Team
    team = await create_financial_team()
    
    print("Running Analysis for AAPL...")
    # Using a simpler prompt to speed it up, but enough to trigger tools
    result = await team.analyze_company("AAPL", "Analyze profitability and growth.")
    
    print("\n\n=== DEBUG REPORT ===")
    
    # 2. Inspect Conversation History
    messages = result.get('messages', []) # Access internal messages if we can, otherwise use the return dict
    # Wait, analyze_company returns a Dict from _process_analysis_result
    # The raw messages should be in there if I modify _process_analysis_result to include them?
    # Actually _process_analysis_result lines:
    # return { "status": "success", "analysis": summary, "structured_data": structured_data, "messages_count": ... }
    # It does NOT return the raw messages list in the final dict.
    
    # However, for this debug script, I want to see the messages. 
    # The 'result' variable here is the OUTPUT of analyze_company.
    
    print(f"Status: {result.get('status')}")
    print(f"Summary extracted: \n{result.get('analysis')}")
    
    # 3. Inspect Structured Data
    s_data = result.get('structured_data', {})
    metrics = s_data.get('metrics', {})
    print(f"\nStructured Metrics Keys: {list(metrics.keys())}")
    if 'Revenue' in metrics:
        print(f"Revenue Data Points: {len(metrics['Revenue'])}")
    else:
        print("❌ Revenue Metric MISSING")

    # 4. Check if we can access the raw history from the team object?
    # The team object wraps AutoGen agents. hard to get history from outside unless we returned it.
    
    # HACK: I will mock the _extract_meaningful_summary method on the team instance 
    # to print the messages it sees before running analyze_company.
    
    await team.close()

if __name__ == "__main__":
    asyncio.run(debug_agent_flow())
