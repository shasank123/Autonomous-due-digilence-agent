# src/data/processors/document_parser.py
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional
import logging
import re

class DocumentProcessor:
    """
    Production Document Processor.
    Robustly handles SEC XBRL tag variations (e.g. Revenue vs SalesRevenueNet).
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def process_sec_facts(self, company_facts: Dict, ticker: str) -> List[Document]:
        documents = []
        if not company_facts or 'facts' not in company_facts:
            return documents
        
        try:
            # 1. Base Info
            documents.extend(self._extract_company_info(company_facts, ticker))
            # 2. Financial Metrics (Raw)
            documents.extend(self._process_financial_metrics(company_facts, ticker))
            # 3. Ratios (Calculated)
            documents.extend(self._compute_financial_ratios(company_facts, ticker))
            # 4. Derived Context (Market/Legal)
            documents.extend(self._generate_derived_market_context(company_facts, ticker))
            documents.extend(self._generate_derived_legal_context(company_facts, ticker))

            self.logger.info(f"Created {len(documents)} documents for {ticker}")
            return documents
        except Exception as e:
            self.logger.error(f"Error processing {ticker}: {e}")
            return documents

    def _extract_company_info(self, company_facts: Dict, ticker: str) -> List[Document]:
        entity = company_facts.get('entityName', 'Unknown')
        sic = company_facts.get('sic', 'Unknown')
        cat = company_facts.get('category', 'Unknown')
        
        content = f"Company: {entity} ({ticker})\nSIC: {sic}\nSector: {cat}\nSource: SEC EDGAR"
        return [Document(page_content=content, metadata={"company": ticker, "doc_type": "company_info"})]

    def _process_financial_metrics(self, facts: Dict, ticker: str) -> List[Document]:
        docs = []
        us_gaap = facts.get('facts', {}).get('us-gaap', {})
        
        # Iterate through ALL gaap keys to find relevant ones
        for tag, data in us_gaap.items():
            # Filter for interesting tags (Revenue, Income, Assets, etc.)
            # Note: Include both 'Liability' and 'Liabilities' because Python substring check is case-sensitive
            if any(x in tag for x in ['Revenue', 'Income', 'Asset', 'Liabilities', 'Liability', 'Equity', 'Stockholders', 'Cash', 'Profit']):
                docs.extend(self._create_metric_documents(tag, data, ticker))
        return docs

    def _create_metric_documents(self, tag: str, data: Dict, ticker: str) -> List[Document]:
        docs = []
        if 'units' not in data or 'USD' not in data['units']: return []
        
        # Sort by date (newest first)
        values = sorted(data['units']['USD'], key=lambda x: x.get('end', ''), reverse=True)
        
        # Take top 3 unique periods
        seen_periods = set()
        for val in values:
            p = val.get('end')
            if p in seen_periods: continue
            seen_periods.add(p)
            if len(seen_periods) > 3: break
            
            content = f"Company: {ticker}\nMetric: {tag}\nValue: {val.get('val', 0)} USD\nPeriod: {p}"
            docs.append(Document(
                page_content=content,
                metadata={
                    "company": ticker, 
                    "metric": tag, 
                    "period": p, 
                    "value": val.get('val', 0),
                    "doc_type": "financial_metric"
                }
            ))
        return docs

    def _extract_latest_values(self, facts: Dict) -> Dict[str, float]:
        """
        Robust extraction that checks multiple possible XBRL tags for each concept.
        """
        values = {}
        us_gaap = facts.get('facts', {}).get('us-gaap', {})
        
        # Priority list of tags for each concept (Order matters!)
        mappings = {
            'revenue': [
                'RevenueFromContractWithCustomerExcludingAssessedTax', 
                'SalesRevenueNet', 
                'Revenues', 
                'Revenue',
                'SalesRevenueGoodsNet'
            ],
            'net_income': [
                'NetIncomeLoss', 
                'NetIncomeLossAvailableToCommonStockholdersBasic', 
                'ProfitLoss'
            ],
            'total_assets': ['Assets'],
            'total_liabilities': ['Liabilities'],
            'equity': [
                'StockholdersEquity', 
                'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'
            ],
            'current_assets': ['AssetsCurrent'],
            'current_liabilities': ['LiabilitiesCurrent']
        }

        for key, potential_tags in mappings.items():
            for tag in potential_tags:
                if tag in us_gaap and 'units' in us_gaap[tag] and 'USD' in us_gaap[tag]['units']:
                    # Get the latest value
                    items = sorted(us_gaap[tag]['units']['USD'], key=lambda x: x.get('end', ''), reverse=True)
                    if items:
                        values[key] = float(items[0]['val'])
                        break # Stop looking for this concept once found
        
        return values

    def _compute_financial_ratios(self, facts: Dict, ticker: str) -> List[Document]:
        data = self._extract_latest_values(facts)
        ratios = {}
        
        # Safe Math Helpers
        def safe_div(n, d): return n / d if d else 0

        if 'net_income' in data and 'total_assets' in data:
            ratios['ROA'] = safe_div(data['net_income'], data['total_assets']) * 100
        
        if 'net_income' in data and 'equity' in data:
            ratios['ROE'] = safe_div(data['net_income'], data['equity']) * 100
            
        if 'total_liabilities' in data and 'equity' in data:
            ratios['Debt to Equity'] = safe_div(data['total_liabilities'], data['equity'])

        if 'current_assets' in data and 'current_liabilities' in data:
            ratios['Current Ratio'] = safe_div(data['current_assets'], data['current_liabilities'])

        docs = []
        for k, v in ratios.items():
            content = f"Ratio: {k}\nValue: {v:.2f}\nInterpretation: {self._get_ratio_interpretation(k, v)}"
            docs.append(Document(page_content=content, metadata={"company": ticker, "metric": k, "doc_type": "financial_ratio"}))
        return docs

    def _generate_derived_market_context(self, facts: Dict, ticker: str) -> List[Document]:
        data = self._extract_latest_values(facts)
        rev = data.get('revenue', 0)
        
        # Robust Market Cap Logic
        pos = "Small Cap/Niche"
        if rev > 100_000_000_000: pos = "Mega Cap/Market Leader"
        elif rev > 10_000_000_000: pos = "Large Cap"
        elif rev > 1_000_000_000: pos = "Mid Cap"
        
        # Determine sector from SIC code or category
        sector = self._get_sector_from_sic(facts.get('sic', '')) or facts.get('category') or 'Technology'
        
        content = (
            f"Market Positioning for {ticker}:\n"
            f"Based on annual revenue of ${rev:,.0f}, {ticker} is a {pos}.\n"
            f"Sector: {sector}."
        )
        return [Document(page_content=content, metadata={"company": ticker, "sector": sector, "doc_type": "market_analysis"})]
    
    def _get_sector_from_sic(self, sic: str) -> Optional[str]:
        """Map SIC code to sector name"""
        if not sic:
            return None
        
        # SIC code categories - first 2 digits determine major sector
        sic_prefix = str(sic)[:2] if sic else ''
        
        sic_mapping = {
            # Technology & Electronics
            '35': 'Computer & Office Equipment',
            '36': 'Electronic Equipment',
            '73': 'Business Services / Software',
            '48': 'Communications',
            # Finance
            '60': 'Banking',
            '61': 'Credit Institutions',
            '62': 'Securities & Investments',
            '63': 'Insurance',
            '67': 'Holding Companies',
            # Healthcare
            '28': 'Chemicals & Pharmaceuticals',
            '38': 'Medical Instruments',
            '80': 'Health Services',
            # Energy
            '13': 'Oil & Gas Extraction',
            '29': 'Petroleum Refining',
            '49': 'Electric & Gas Utilities',
            # Manufacturing
            '37': 'Transportation Equipment',
            '34': 'Fabricated Metal',
            '33': 'Primary Metal',
            # Retail
            '52': 'Building Materials Retail',
            '53': 'General Merchandise',
            '54': 'Food Stores',
            '56': 'Apparel Retail',
            '57': 'Furniture & Home',
            '59': 'Miscellaneous Retail',
        }
        
        return sic_mapping.get(sic_prefix)

    def _generate_derived_legal_context(self, facts: Dict, ticker: str) -> List[Document]:
        data = self._extract_latest_values(facts)
        debt = data.get('total_liabilities', 0)
        eq = data.get('equity', 1)
        lev = debt / eq if eq else 0
        
        risk = "High" if lev > 2.0 else "Low/Moderate"
        content = f"Risk Assessment for {ticker}:\nFinancial Leverage Risk: {risk} (Debt/Equity: {lev:.2f})."
        return [Document(page_content=content, metadata={"company": ticker, "doc_type": "legal_risk"})]

    def _get_ratio_interpretation(self, name, val):
        if name == 'ROA': return "Good" if val > 5 else "Weak"
        if name == 'ROE': return "Strong" if val > 15 else "Average"
        if name == 'Current Ratio': return "Healthy" if val > 1.5 else "Tight"
        if name == 'Debt to Equity': return "Risky" if val > 2.0 else "Safe"
        return "N/A"