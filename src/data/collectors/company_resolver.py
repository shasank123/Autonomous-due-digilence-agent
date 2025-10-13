import requests
import json
from typing import Optional, Dict

class CompanyResolver:

    def __init__(self):
        self.ticker_map = self.load_ticker_map()

    def load_ticker_map(self) -> Dict[str, str]:
        
        """Load all 10,000+ companies from SEC"""
        try:
            # SEC official company tickers JSON
            url = "https://www.sec.gov/files/company_tickers.json"
            response = requests.get(url, headers={'user-agent': 'sasishasank2@gmail.com'})
            data = response.json()

            # Convert to {ticker: cik} mapping
            ticker_map = {}
            for company in data.values():
                ticker = company.get('ticker')
                cik_str = str(company.get('cik_str', '')).zfill(10)
                if ticker and cik_str:
                    ticker_map[;ticker.upper()] = cik_str
            print(f"✅ Loaded {len(ticker_map)} companies from SEC")
            return ticker_map
        
        except Exception as e:
            print(f"❌ Failed to load SEC tickers: {e}")
            return {}
        
    def get_cik(self, ticker: str) -> Optional[str]:
        return self.ticker_map.get(ticker.upper())
    
    def search_companies(self, query: str) -> list:
        """Search companies by name or ticker"""
        results = []
        query = query.lower()
        
        for ticker, cik in self.ticker_map.items():
            if query in ticker:
                results.append({'ticker': ticker, 'cik': cik})

        return results[:10]  # Return top 10 matches
