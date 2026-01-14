# src/data/collectors/company_resolver.py
import os
import requests
import json
import logging
from dotenv import load_dotenv
from typing import Optional, Dict, List

load_dotenv()
logger = logging.getLogger(__name__)

class CompanyResolver:
    """
    Resolves company tickers to SEC CIK numbers.
    Implements Singleton pattern to load the ticker map only once per session.
    """
    _instance = None
    _ticker_map = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CompanyResolver, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Only load if not already loaded (Singleton check)
        if not self._ticker_map:
            self._ticker_map = self.load_ticker_map()

    def load_ticker_map(self) -> Dict[str, str]:
        """
        Load all 10,000+ companies from SEC.
        """
        try:
            # SEC official company tickers JSON
            url = "https://www.sec.gov/files/company_tickers.json"
            
            # Use environment email or fallback to prevent crash, but warn user
            email = os.getenv('SEC_EDGAR_EMAIL', 'sasishasank2@gmail.com')
            headers = {
                'User-Agent': f"CompanyResolver/1.0 ({email})",
                'Accept-Encoding': 'gzip, deflate',
                'Host': 'www.sec.gov'
            }

            logger.info("Downloading SEC Ticker Map...")
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Convert to {ticker: cik} mapping
            ticker_map = {}
            for company in data.values():
                ticker = company.get('ticker')
                # Restored: Your zfill(10) logic is required for valid SEC URLs
                cik_str = str(company.get('cik_str', '')).zfill(10)
                
                if ticker and cik_str:
                    ticker_map[ticker.upper()] = cik_str
            
            logger.info(f"Loaded {len(ticker_map)} companies from SEC")
            return ticker_map
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error loading SEC tickers: {e}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error loading SEC tickers: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error loading SEC tickers: {e}")
            return {}

    def get_cik(self, ticker: str) -> Optional[str]:
        if not ticker: return None
        return self._ticker_map.get(ticker.upper())
    
    def search_companies(self, query: str) -> list:
        """Search companies by name or ticker"""
        if not query: return []
        
        results = []
        query = query.upper()
        
        # Restored: Your exact search implementation
        for ticker, cik in self._ticker_map.items():
            if query in ticker:
                results.append({'ticker': ticker, 'cik': cik})

        return results[:10]  # Return top 10 matches

if __name__ == "__main__":
    # Test Block
    resolver = CompanyResolver()
    print(f"Loaded: {len(resolver._ticker_map)}")
    print(f"AAPL CIK: {resolver.get_cik('AAPL')}")