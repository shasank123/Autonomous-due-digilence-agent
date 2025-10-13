import json
import requests
import time
from typing import Dict,List,Optional
import logging
from company_resolver import CompanyResolver

class SECDataCollector:
    """
    Production-grade SEC data collector with error handling & rate limiting
    """
    def __init__(self, email: str):
        self.email = email
        self.base_url = "https://data.sec.gov/api/xbrl"
        self.headers = {
            'user_agent': email,
            'Accept-Encoding': 'gzip, deflate'
        },
        self.company_resolver = CompanyResolver(),
        self.logger = logging.getLogger(__name__)

    def company_facts(self, ticker: str) -> Optional[Dict]:
        """
        Production: Fetches data for ANY public company with proper error handling
        """
        cik = self.company_resolver.get_cik(ticker)
        if not cik:
            self.logger.error(F"❌ CIK not found for ticker: {ticker}")
            return None
        
        url = f"{self.base_url}/companyfacts/CIK/{cik}.json"

        try:
            time.sleep(0.2)  # Rate limiting: 2 requests per second
            response = requests.get(url, headers=self.headers, timeout=30)
            if response.status_code == 200:
                self.logger.info(f"✅ Successfully fetched SEC data for {ticker}")
                return response.json()
            elif response.status_code == 404:
                self.logger.warning(f"No SEC data found for {ticker}")
                return None
            else:
                self.logger.error(f"❌ SEC API error {response.status_code} for {ticker}")
                return None
            
        except requests.exceptions.Timeout:
            self.logger.error(f"❌ SEC API timeout for {ticker}")
            return None
        
        except Exception as e:
            self.logger.error(f"❌ Unexpected error for {ticker}: {e}")
            return None


              
    def get_available_metrics(self, ticker: str) -> List[str]:

        facts = self.company_facts(ticker)
        if not facts:
            return []
        
        metrics = []
        
        # 1. US-GAAP Financial Metrics (500+ metrics)
        if 'facts' in facts and 'us-gaap' in facts['facts']:
            metrics.extend(list(facts['facts']['us-gaap'].keys()))

        # 2. DEI - Company Entity Information
        if 'facts' in facts and 'dei' in facts['facts']:
            metrics.extend(list(facts['facts']['dei'].keys()))

        # 3. Other namespaces (if available)
        if 'facts' in facts:
            for namespace in facts['facts']:
                if namespace not in ['us-gaap', 'dei']:
                    metrics.extend(list(facts['facts'][namespace].keys()))

        return sorted(metrics)

