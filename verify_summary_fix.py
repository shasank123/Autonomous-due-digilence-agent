
import logging
from src.agents.financial_analyst import FinancialAgentTeam
from typing import List, Any
from dataclasses import dataclass

@dataclass
class MockMessage:
    source: str
    content: Any 

def test_summary_extraction():
    print("Testing Summary Extraction Logic...")
    
    class MockTeam(FinancialAgentTeam):
        def __init__(self):
            self.logger = logging.getLogger(__name__)

    team = MockTeam()
    
    # Case 4: FunctionCall object (The newest culprit)
    print(f"\nCase 4 (FunctionCall Object + Real Report):")
    
    # Mocking what AutoGen might store for a function call
    class FunctionCallMock:
        def __init__(self):
            self.name = 'validate_with_source_data'
        def __str__(self):
            return "FunctionCall(id='call_123', arguments='{...}', name='validate_with_source_data')"
            
    messages_4 = [
        MockMessage("financial_analyst", "Draft report..."),
        MockMessage("financial_reviewer", "FINAL REPORT: Invest in AAPL. Strong financial health.\nTERMINATE"),
        MockMessage("financial_reviewer", "[VALIDATION] Summary for AAPL..."),
        MockMessage("financial_reviewer", [FunctionCallMock()]) # List containing function call obj
    ]
    
    # Logic should skip FunctionCall AND Validation log
    summary_4 = team._extract_meaningful_summary(messages_4)
    print(f"Result: {summary_4[:50]}...")
    
    assert "FunctionCall" not in summary_4
    assert "[VALIDATION]" not in summary_4
    assert "FINAL REPORT" in summary_4
    print("Case 4 Passed")

if __name__ == "__main__":
    try:
        test_summary_extraction()
        print("\nALL TESTS PASSED")
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
    except Exception as e:
        print(f"\nERROR: {e}")
