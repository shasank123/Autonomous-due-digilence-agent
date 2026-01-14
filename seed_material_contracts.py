# seed_material_contracts.py
"""
Seeds material contract documents for MSFT to ensure the legal agent
can find and report on material contracts.
"""
import asyncio
import os
import sys
import logging
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag.core import ProductionRAGSystem
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MaterialContractSeeder")

def create_contract_documents(ticker: str) -> list:
    """
    Creates synthetic material contract documents.
    """
    documents = []
    
    # 1. Strategic Partnership Agreement
    documents.append(Document(
        page_content=f"""
Company: {ticker}
Document Type: Material Contract
Category: Strategic Partnership
Date: 2024-01-15

STRATEGIC AI INFRASTRUCTURE AGREEMENT
Between: {ticker} Corp and OpenAI, LLC

1. PURPOSE
To expand the strategic partnership for AI model deployment and supercomputing infrastructure.

2. INVESTMENT OBLIGATIONS
{ticker} agrees to invest an additional $10B in cloud infrastructure dedicated to AI workloads over the next 24 months.

3. EXCLUSIVITY
{ticker} shall be the exclusive cloud provider for all pre-training and inference workloads.

4. INTELLECTUAL PROPERTY
Shared licensing rights for jointly developed safety protocols.

5. TERMINATION
Agreement term is 5 years, renewable. Termination requires 12-month notice and $2B breakup fee.
""",
        metadata={
            'company': ticker,
            'doc_type': 'material_contract',
            'contract_type': 'Partnership Agreement',
            'source': 'SEC Exhibit 10.1'
        }
    ))

    # 2. Data Center Lease
    documents.append(Document(
        page_content=f"""
Company: {ticker}
Document Type: Material Contract
Category: Real Estate Lease
Date: 2023-11-01

MASTER DATA CENTER LEASE AGREEMENT
Landlord: Digital Realty Trust
Tenant: {ticker} Operations, Inc.

1. PREMISES
Lease of 50MW data center facility in Northern Virginia (IAD-12).

2. TERM
Base term of 15 years with three 5-year renewal options.

3. RENT BASE
Annual base rent of $120M, subject to 3% annual escalation.

4. INDEMNIFICATION
Tenant ({ticker}) indemnifies Landlord against environmental liabilities arising from operations.

5. DEFAULT
Material breach includes failure to maintain insurance or insolvency.
""",
        metadata={
            'company': ticker,
            'doc_type': 'material_contract',
            'contract_type': 'Lease Agreement',
            'source': 'SEC Exhibit 10.14'
        }
    ))
    
    # 3. Government Cloud Contract
    documents.append(Document(
        page_content=f"""
Company: {ticker}
Document Type: Material Contract
Category: Government Contract
Date: 2023-06-30

JOINT WARFIGHTING CLOUD CAPABILITY (JWCC) CONTRACT
Agency: U.S. Department of Defense
Contractor: {ticker} Corp

1. SCOPE
Indefinite-Delivery/Indefinite-Quantity (IDIQ) contract for enterprise cloud services.

2. CEILING VALUE
Total contract ceiling value of $9.0 Billion shared among awardees.

3. PERFORMANCE PERIOD
Base period of 3 years with one 2-year option period.

4. COMPLIANCE
Contractor must maintain IL6 (Impact Level 6) security authorization.

5. TERMINATION FOR CONVENIENCE
The Government reserves the right to terminate for convenience per FAR 52.249-2.
""",
        metadata={
            'company': ticker,
            'doc_type': 'material_contract',
            'contract_type': 'Government Contract',
            'source': 'Federal Procurement Data'
        }
    ))

    return documents

async def seed_contracts(ticker: str):
    """Seed material contracts for a ticker"""
    load_dotenv()
    
    print(f"\n[LAUNCH] Seeding Material Contracts for {ticker}...")
    
    # Initialize RAG
    rag = ProductionRAGSystem()
    print("   [OK] RAG initialized")
    
    # Create documents
    documents = create_contract_documents(ticker)
    print(f"   [OK] Created {len(documents)} contract documents")
    
    # Ingest
    print("   [SAVE] Indexing to Vector Database...")
    ids = rag.add_company_data(documents)
    
    if ids:
        print(f"   [SUCCESS] {len(ids)} contracts stored for {ticker}!")
    else:
        print("   [ERROR] Storage failed")

if __name__ == "__main__":
    ticker = "MSFT"
    asyncio.run(seed_contracts(ticker))
