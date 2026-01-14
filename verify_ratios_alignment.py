
import logging
from src.tools.financial_tools import FinancialTools

# Mock RAG system (not needed for this specific test, but required for init)
class MockRAG:
    def __init__(self): pass

def test_ratio_alignment():
    print("Testing Ratio Alignment Logic...")
    
    # Setup
    tools = FinancialTools(MockRAG(), None)
    tools.logger.setLevel(logging.INFO)
    
    # Case 1: Perfect Alignment (2025)
    metrics_aligned = {
        "Revenue": [{"period": "2025", "value": 100}],
        "Net Income": [{"period": "2025", "value": 20}]
    }
    ratios = tools._calculate_ratios_from_metrics(metrics_aligned)
    assert ratios.get("profit_margin") == 20.0
    print(f"[OK] Case 1 (Aligned): Passed. Margin={ratios.get('profit_margin')}%")

    # Case 2: Mismatch (Revenue 2018, Income 2025) -> Should FAIL to find common period or use fallback 
    # Logic: if no common period with both, it picks 'most recent' (2025). 
    # 2025 has Income (20) but NO Revenue. So margin should be None (or not calculated).
    metrics_mismatch = {
        "Revenue": [{"period": "2018", "value": 100}],
        "Net Income": [{"period": "2025", "value": 20}]
    }
    ratios_mismatch = tools._calculate_ratios_from_metrics(metrics_mismatch)
    
    # We expect NO profit margin because 2025 (latest) has no Revenue, and 2018 (oldest) has no Income.
    margin = ratios_mismatch.get("profit_margin")
    if margin is None:
        print(f"[OK] Case 2 (Mismatch): Passed. Margin is correctly None (avoided {20/100*100}% nonsense).")
    else:
        print(f"[FAIL] Case 2 (Mismatch): FAILED. Calculated {margin}% using mismatched data!")

    # Case 3: Partial Alignment (Have 2025 and 2018 for both)
    metrics_partial = {
        "Revenue": [{"period": "2025", "value": 200}, {"period": "2018", "value": 100}],
        "Net Income": [{"period": "2025", "value": 40}, {"period": "2018", "value": 10}]
    }
    ratios_partial = tools._calculate_ratios_from_metrics(metrics_partial)
    assert ratios_partial.get("profit_margin") == 20.0 # 40/200
    print(f"[OK] Case 3 (Full History): Passed. Used 2025 data. Margin={ratios_partial.get('profit_margin')}%")

if __name__ == "__main__":
    test_ratio_alignment()
