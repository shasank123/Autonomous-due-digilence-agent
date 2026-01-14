# src/data/collectors/sec_edgar.py
import json
import requests
import time
from typing import Dict, List, Optional
import logging
from .company_resolver import CompanyResolver
from dotenv import load_dotenv
import os

load_dotenv() # Load .env variables

class SECDataCollector:
    """
    Production-grade SEC data collector with error handling & rate limiting
    """
    def __init__(self, email: str = None):
        # Fallback to os.getenv if email arg is None, else default
        env_email = os.getenv('SEC_EDGAR_EMAIL')
        self.email = email or env_email or 'sasishasank2@gmail.com'
        self.logger = logging.getLogger(__name__)
        
        self.logger.debug(f"Using email for SEC headers: {self.email}")
        
        self.base_url = "https://data.sec.gov/api/xbrl"
        self.headers = {
            'User-Agent': f"SasiPersonalResearch/1.0 ({self.email})",
            'Accept-Encoding': 'gzip, deflate',
        }
        self.company_resolver = CompanyResolver()

    def company_facts(self, ticker: str) -> Optional[Dict]:
        """
        Fetches company financial data from SEC
        """
        cik = self.company_resolver.get_cik(ticker)
        if not cik:
            self.logger.error(f"CIK not found for ticker: {ticker}")
            return None
        
        url = f"{self.base_url}/companyfacts/CIK{cik}.json"

        try:
            # Restored: Your exact rate limit of 0.2s (5 requests/sec)
            time.sleep(0.2) 
            
            response = requests.get(url, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                self.logger.info(f"Successfully fetched SEC data for {ticker}")
                return response.json()
            elif response.status_code == 404:
                self.logger.warning(f"No SEC data found for {ticker}")
                return None
            else:
                self.logger.error(f"SEC API error {response.status_code} for {ticker}")
                return None
            
        except requests.exceptions.Timeout:
            self.logger.error(f"SEC API timeout for {ticker}")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error fetching SEC data for {ticker}: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error for {ticker}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error for {ticker}: {e}")
            return None
            
    def get_available_metrics(self, ticker: str) -> List[str]:
        """
        Extracts all available XBRL tags (metrics) for a company.
        """
        facts = self.company_facts(ticker)
        if not facts:
            return []
        
        metrics = []
        
        # Restored: Your exact logic for iterating namespaces
        
        # 1. US-GAAP Financial Metrics
        if 'facts' in facts and 'us-gaap' in facts['facts']:
            metrics.extend(list(facts['facts']['us-gaap'].keys()))

        # 2. DEI - Company Entity Information
        if 'facts' in facts and 'dei' in facts['facts']:
            metrics.extend(list(facts['facts']['dei'].keys()))

        # 3. Other namespaces (if available - e.g. IFRS or custom)
        if 'facts' in facts:
            for namespace in facts['facts']:
                if namespace not in ['us-gaap', 'dei']:
                    metrics.extend(list(facts['facts'][namespace].keys()))

        return sorted(metrics)

if __name__ == "__main__":
    # Test Block
    collector = SECDataCollector()
    print("Testing SEC Data Collector...")
    test_tickers = ["AAPL", "TSLA"]
    for t in test_tickers:
        m = collector.get_available_metrics(t)
        print(f"{t}: Found {len(m)} metrics")