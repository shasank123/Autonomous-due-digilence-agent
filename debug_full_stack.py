
import asyncio
import os
import sys
import json
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agents.orchestrator import DueDiligenceOrchestrator

async def debug_full_stack():
    load_dotenv()
    print("Initializing Orchestrator...")
    orchestrator = DueDiligenceOrchestrator()
    
    print("Running START-TO-FINISH Analysis for AAPL...")
    try:
        request_id = str(uuid.uuid4())
        
        result = await orchestrator.execute_analysis(
            request_id=request_id,
            company_ticker="AAPL",
            analysis_type="comprehensive",
            questions=["Debug Visualizations"]
        )
        
        print("\n--- INTEGRATION TEST RESULT ---")
        
        fin_data = result.get('financial_analysis', {})
        print(f"Financial Analysis Keys: {list(fin_data.keys())}")
        
        structured = fin_data.get('structured_data', {})
        metrics = structured.get('metrics', {})
        
        print(f"Structured Data Present: {'YES' if structured else 'NO'}")
        
        if 'Revenue' in metrics:
            print(f"\n[OK] Revenue Data Found: {len(metrics['Revenue'])} points")
            print(json.dumps(metrics['Revenue'], indent=2))
        else:
            print(f"[MISSING] Revenue Data MISSING. Keys found: {list(metrics.keys())}")
            
        if 'Net Income' in metrics:
             print(f"\n[OK] Net Income Data Found: {len(metrics['Net Income'])} points")
             print(json.dumps(metrics['Net Income'], indent=2))

        ratios = structured.get('ratios', {})
        print(f"\nRatios: {json.dumps(ratios, indent=2)}")

    except Exception as e:
        print(f"\n[FATAL EXCEPTION] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_full_stack())
