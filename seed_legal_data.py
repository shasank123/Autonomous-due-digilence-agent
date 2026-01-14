# seed_legal_data.py
"""
Seeds legal-specific documents from SEC filings (10-K, 8-K) for a company.
These documents contain risk factors, legal proceedings, and compliance info.
"""
import asyncio
import os
import sys
import logging
import requests
import time
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag.core import ProductionRAGSystem
from src.data.collectors.company_resolver import CompanyResolver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LegalDataSeeder")

class SECFilingsFetcher:
    """Fetches SEC filing documents (10-K, 10-Q, 8-K) from EDGAR"""
    
    def __init__(self, email: str = None):
        load_dotenv()
        self.email = email or os.getenv('SEC_EDGAR_EMAIL', 'sasishasank2@gmail.com')
        self.resolver = CompanyResolver()
        self.base_url = "https://data.sec.gov"
        self.headers = {
            'User-Agent': f"SasiPersonalResearch/1.0 ({self.email})",
            'Accept-Encoding': 'gzip, deflate',
        }
    
    def get_filings_list(self, ticker: str, form_types: List[str] = None) -> List[Dict]:
        """Get list of recent filings for a company"""
        if form_types is None:
            form_types = ['10-K', '10-Q', '8-K']
        
        cik = self.resolver.get_cik(ticker)
        if not cik:
            logger.error(f"CIK not found for {ticker}")
            return []
        
        # CIK needs to be 10 digits with leading zeros for submissions endpoint
        cik_padded = cik.zfill(10)
        
        # Fetch submissions
        url = f"{self.base_url}/submissions/CIK{cik_padded}.json"
        time.sleep(0.2)  # Rate limit
        
        try:
            resp = requests.get(url, headers=self.headers, timeout=30)
            if resp.status_code != 200:
                logger.error(f"Failed to get filings for {ticker}: {resp.status_code}")
                return []
            
            data = resp.json()
            filings = []
            
            recent = data.get('filings', {}).get('recent', {})
            forms = recent.get('form', [])
            accession_numbers = recent.get('accessionNumber', [])
            filing_dates = recent.get('filingDate', [])
            primary_docs = recent.get('primaryDocument', [])
            
            for i, form in enumerate(forms[:50]):  # Check first 50 filings
                if form in form_types:
                    filings.append({
                        'form': form,
                        'accession': accession_numbers[i].replace('-', ''),
                        'date': filing_dates[i],
                        'doc': primary_docs[i] if i < len(primary_docs) else None,
                        'cik': cik
                    })
            
            logger.info(f"Found {len(filings)} relevant filings for {ticker}")
            return filings[:10]  # Return up to 10 most recent
            
        except Exception as e:
            logger.error(f"Error fetching filings list: {e}")
            return []
    
    def fetch_filing_content(self, filing: Dict) -> str:
        """Fetch the actual content of a filing"""
        cik = filing['cik']
        accession = filing['accession']  # Without dashes
        accession_with_dashes = filing.get('accession_with_dashes', '')
        if not accession_with_dashes:
            # Convert to format with dashes: XXXXXXXXXX-XX-XXXXXX
            accession_with_dashes = f"{accession[:10]}-{accession[10:12]}-{accession[12:]}"
        doc = filing.get('doc', 'primary_doc.htm')
        
        # CIK can be with or without leading zeros
        cik_int = int(cik)
        
        # Try multiple URL formats
        urls_to_try = [
            # Format 1: CIK as int, accession without dashes
            f"{self.base_url}/Archives/edgar/data/{cik_int}/{accession}/{doc}",
            # Format 2: CIK as int, accession with dashes
            f"{self.base_url}/Archives/edgar/data/{cik_int}/{accession_with_dashes}/{doc}",
            # Format 3: Try to get index.json instead to find the actual file
            f"{self.base_url}/Archives/edgar/data/{cik_int}/{accession}/index.json",
        ]
        
        for url in urls_to_try:
            time.sleep(0.2)  # Rate limit
            try:
                resp = requests.get(url, headers=self.headers, timeout=60)
                if resp.status_code == 200:
                    if url.endswith('.json'):
                        # This is index, extract actual document URL
                        data = resp.json()
                        # Just use the first document that seems relevant
                        logger.info(f"Found index, extracting docs for {filing['form']}")
                        continue  # Skip for now, try other URLs
                    
                    # Extract text content (strip HTML if needed)
                    content = resp.text
                    import re
                    text = re.sub('<[^<]+?>', ' ', content)
                    text = ' '.join(text.split())  # Normalize whitespace
                    logger.info(f"Fetched content from {url[:80]}...")
                    return text[:50000]  # Limit size
            except Exception as e:
                logger.debug(f"URL failed: {url[:50]}... - {e}")
                continue
        
        logger.warning(f"All URL formats failed for {filing['form']} - {accession}")
        return ""


def create_legal_documents(ticker: str, filings: List[Dict], fetcher: SECFilingsFetcher) -> List[Dict]:
    """Create document objects for RAG ingestion"""
    documents = []
    
    for filing in filings:
        content = fetcher.fetch_filing_content(filing)
        if not content:
            continue
        
        # Extract relevant sections (risk factors, legal proceedings)
        doc = {
            'page_content': f"""
Company: {ticker}
Filing Type: {filing['form']}
Filing Date: {filing['date']}
Document Type: Legal/Regulatory Filing

Content Summary:
{content[:10000]}

[This document contains {filing['form']} filing information including risk factors, 
legal proceedings, and regulatory compliance disclosures for {ticker}.]
""",
            'metadata': {
                'company': ticker,
                'doc_type': 'legal_filing',
                'form_type': filing['form'],
                'filing_date': filing['date'],
                'source': 'SEC EDGAR'
            }
        }
        documents.append(doc)
        logger.info(f"Created document for {ticker} {filing['form']} ({filing['date']})")
    
    return documents


async def seed_legal_data(ticker: str):
    """Main function to seed legal documents for a ticker"""
    load_dotenv()
    
    print(f"\n[LAUNCH] Seeding Legal Data for {ticker}...")
    
    # Initialize
    fetcher = SECFilingsFetcher()
    rag = ProductionRAGSystem()
    
    print("   [OK] Components initialized")
    
    # Fetch filings list
    print("   [FETCH] Getting SEC filings list...")
    filings = fetcher.get_filings_list(ticker, ['10-K', '10-Q', '8-K'])
    
    if not filings:
        print("   [ERROR] No filings found")
        return
    
    print(f"   [OK] Found {len(filings)} filings")
    
    # Create documents
    print("   [PROCESS] Fetching and processing filings...")
    documents = create_legal_documents(ticker, filings, fetcher)
    
    if not documents:
        print("   [ERROR] No documents created")
        return
    
    print(f"   [OK] Created {len(documents)} legal documents")
    
    # Ingest into RAG
    print("   [SAVE] Indexing to Vector Database...")
    
    # Convert to format expected by RAG
    from langchain.schema import Document
    langchain_docs = [
        Document(page_content=d['page_content'], metadata=d['metadata'])
        for d in documents
    ]
    
    ids = rag.add_company_data(langchain_docs)
    
    if ids:
        print(f"   [SUCCESS] {len(ids)} legal documents stored for {ticker}!")
    else:
        print("   [ERROR] Vector storage failed")


if __name__ == "__main__":
    ticker = "MSFT"  # Change this for other companies
    print(f"Seeding legal data for {ticker}...")
    asyncio.run(seed_legal_data(ticker))
