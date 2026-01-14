# seed_legal_docs_simple.py
"""
Creates legal-analysis-friendly documents from existing financial data.
This allows the legal agent to analyze financial metrics for legal implications
(debt obligations, liabilities, compliance indicators).
"""
import asyncio
import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag.core import ProductionRAGSystem
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LegalDocSeeder")


def create_legal_context_documents(ticker: str) -> list:
    """
    Creates legal-context documents that help the legal agent
    analyze financial data for legal implications.
    """
    documents = []
    
    # Risk and Compliance Overview Document
    documents.append(Document(
        page_content=f"""
Company: {ticker}
Document Type: Legal Risk Assessment Framework
Category: Legal Due Diligence

LEGAL AND REGULATORY RISK ANALYSIS FRAMEWORK FOR {ticker}

1. FINANCIAL OBLIGATION ANALYSIS
   - Total Liabilities: Review for debt covenant compliance
   - Long-term Debt: Assess refinancing risk and interest obligations
   - Current Liabilities: Evaluate short-term liquidity and payment obligations

2. REGULATORY COMPLIANCE INDICATORS
   - Revenue Recognition: SEC compliance with ASC 606
   - Asset Valuation: Fair value measurement compliance
   - Disclosure Requirements: 10-K and 10-Q filing compliance

3. LITIGATION AND CONTINGENCY ASSESSMENT
   - Contingent Liabilities: Review for pending legal matters
   - Accrued Expenses: Identify potential legal reserves
   - Related Party Transactions: Governance compliance review

4. CORPORATE GOVERNANCE REVIEW
   - Audit Committee Oversight: Internal control effectiveness
   - Executive Compensation: Proxy disclosure compliance
   - Shareholder Rights: Voting and dividend policies

5. MATERIAL CONTRACTS ANALYSIS
   - Debt Agreements: Covenant compliance status
   - Lease Obligations: Operating and finance lease commitments
   - Supply Agreements: Contractual obligation risks

[This framework is used to analyze {ticker}'s financial metrics for legal implications.]
""",
        metadata={
            'company': ticker,
            'doc_type': 'legal_framework',
            'category': 'legal_due_diligence',
            'source': 'Legal Analysis Framework'
        }
    ))
    
    # Compliance Assessment Document
    documents.append(Document(
        page_content=f"""
Company: {ticker}
Document Type: Compliance Assessment
Category: Regulatory Compliance

REGULATORY COMPLIANCE ASSESSMENT FOR {ticker}

SEC FILINGS COMPLIANCE:
- Form 10-K Annual Reports: Standard compliance expected
- Form 10-Q Quarterly Reports: Standard compliance expected
- Form 8-K Current Reports: Event-driven disclosure compliance

FINANCIAL REPORTING STANDARDS:
- US GAAP Compliance: Required for SEC registrants
- Revenue Recognition (ASC 606): Applicable to all contracts
- Lease Accounting (ASC 842): Operating and finance lease treatment

KEY LEGAL CONSIDERATIONS:
- Sarbanes-Oxley Act Compliance: Internal controls certification
- Securities Act Compliance: Disclosure and anti-fraud provisions
- Exchange Act Compliance: Periodic reporting requirements

RISK FACTORS TO REVIEW:
- Analyze total liabilities for debt service coverage
- Review contingent liabilities for pending litigation
- Assess related party transactions for conflicts

COMPLIANCE STATUS: Review financial metrics to assess compliance.
""",
        metadata={
            'company': ticker,
            'doc_type': 'legal_compliance',
            'category': 'regulatory_compliance',
            'source': 'Compliance Assessment'
        }
    ))
    
    # Legal Risk Summary Document
    documents.append(Document(
        page_content=f"""
Company: {ticker}
Document Type: Legal Risk Summary
Category: Legal Due Diligence

LEGAL DUE DILIGENCE SUMMARY FOR {ticker}

OVERVIEW:
This legal due diligence analysis reviews {ticker}'s financial filings 
and metrics to identify potential legal risks and compliance matters.

KEY AREAS OF LEGAL REVIEW:
1. Debt and Financing Obligations
   - Long-term debt covenants
   - Short-term borrowing facilities
   - Interest coverage ratios

2. Litigation and Legal Reserves
   - Accrued litigation costs
   - Pending legal matters
   - Insurance coverage adequacy

3. Regulatory Compliance
   - SEC filing timeliness
   - Accounting standard compliance
   - Internal control effectiveness

4. Contractual Obligations
   - Material contracts summary
   - Lease commitments
   - Purchase obligations

5. Governance and Ethics
   - Board composition
   - Executive compensation
   - Related party transactions

RECOMMENDATION:
Based on available financial data, conduct detailed review of:
- Total liabilities trend
- Contingent liability disclosures
- Revenue recognition policies

For comprehensive legal analysis, review in conjunction with 10-K risk factors.
""",
        metadata={
            'company': ticker,
            'doc_type': 'legal_summary',
            'category': 'legal_due_diligence',
            'source': 'Legal Risk Summary'
        }
    ))
    
    return documents


async def seed_legal_docs(ticker: str):
    """Seed legal context documents for a ticker"""
    load_dotenv()
    
    print(f"\n[LAUNCH] Creating Legal Context Documents for {ticker}...")
    
    # Initialize RAG
    rag = ProductionRAGSystem()
    print("   [OK] RAG initialized")
    
    # Create documents
    documents = create_legal_context_documents(ticker)
    print(f"   [OK] Created {len(documents)} legal context documents")
    
    # Ingest
    print("   [SAVE] Indexing to Vector Database...")
    ids = rag.add_company_data(documents)
    
    if ids:
        print(f"   [SUCCESS] {len(ids)} legal documents stored for {ticker}!")
    else:
        print("   [ERROR] Storage failed")


if __name__ == "__main__":
    ticker = "MSFT"
    asyncio.run(seed_legal_docs(ticker))
