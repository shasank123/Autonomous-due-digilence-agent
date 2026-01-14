import logging
from langchain_core.documents import Document
from src.tools.financial_tools import FinancialTools
from src.rag.core import ProductionRAGSystem
from src.data.collectors.sec_edgar import SECDataCollector

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_parsing():
    # Mock dependencies
    rag = ProductionRAGSystem()
    sec = SECDataCollector()
    tools = FinancialTools(rag, sec)

    # Mock Document from DocumentProcessor
    content = """
        Company: AAPL
        Financial Metric: Total Revenue
        Value: 331495000000 USD
        Period End: 2025-06-28
        Filed Date: 2025-08-01
        Form Type: 10-Q
        Context: As Reported
    """
    
    metadata = {
        "source": "sec",
        "company": "AAPL",
        "metric": "Revenue",
        "unit": "USD",
        "period": "2025-06-28",
        "doc_type": "financial_metric"
    }
    
    doc = Document(page_content=content, metadata=metadata)
    scored_docs = [(doc, 0.5)] # doc, score

    print("\n--- Testing Revenue Parsing ---")
    result = tools._extract_metric_data(scored_docs, "Revenue", "AAPL")
    print(f"Result: {result}")

    # Test NetIncomeLoss
    content_ni = """
        Company: AAPL
        Financial Metric: Net Income/Loss
        Value: 112010000000 USD
        Period End: 2025-06-28
        Filed Date: 2025-08-01
        Form Type: 10-Q
        Context: As Reported
    """
    metadata_ni = {
        "source": "sec",
        "company": "AAPL",
        "metric": "NetIncomeLoss",
        "unit": "USD",
        "period": "2025-06-28",
        "doc_type": "financial_metric"
    }
    doc_ni = Document(page_content=content_ni, metadata=metadata_ni)
    scored_docs_ni = [(doc_ni, 0.5)]

    print("\n--- Testing NetIncomeLoss Parsing ---")
    result_ni = tools._extract_metric_data(scored_docs_ni, "NetIncomeLoss", "AAPL")
    print(f"Result: {result_ni}")

if __name__ == "__main__":
    test_parsing()
