import json
import requests
import time
from typing import Dict,List,Optional

class SECDataCollector:
    """
    REAL SEC API integration - fetches actual company financial data
    """
    def __init__(self, email: str):
        self.email = email
        self.base_url = "https://data.sec.gov/api/xbrl"
        self.headers = {
            'user_agent': f'{email}',
            'Accept-Encoding': 'gzip, deflate'
        }
    
            
        # real CIK mappings for major companies
        ticker_to_cik = {
            "AAPL": "0000320193",    # Apple
            "MSFT": "0000789019",    # Microsoft  
            "TSLA": "0001318605",    # Tesla
            "GOOGL": "0001652044",   # Alphabet (Google)
            "AMZN": "0001018724",    # Amazon
            "META": "0001326801",    # Meta (Facebook)
            "NFLX": "0001065280",    # Netflix
            "NVDA": "0001045810",    # NVIDIA
            "JPM": "0000019617",     # JPMorgan Chase
            "JNJ": "0000200406"      # Johnson & Johnson
        }


    def get_company_ticker_cik(self, ticker: str) -> Optional[str]:
        # convert stock ticker to SEC CIK
        return self.ticker_to_cik.get(ticker.upper())

        
    def company_facts(self, ticker: str) -> Optional[Dict]:
        """
        REAL SEC API call: fetches actual financial data
        """
        cik = self.get_company_ticker_cik(ticker)
        if not cik:
            print(f" unknown ticker symbol: {ticker}")
            return None
        
        url = f"{self.base_url}/companyfacts/cik/{cik}.json"

        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                print(f" ✅ Successfully fetched SEC data for {ticker}")
                return response.json()

            else:
                print(f" Failed to fetch SEC data for {ticker}, status code: {response.status_code}")
                return None
            
        except Exception as e:
            print(f"❌ Network error: {e}")
            return None
        
    def extract_key_metrics(self, company_facts: Dict, ticker: str) -> Dict:
        """
        Extract key financial metrics from SEC data
        """
        if not company_facts or 'facts' not in company_facts:
            print("No financial data found to extract")
            return {}
        
        metrics = {}
        facts = company_facts['facts']

        # US-GAAP financial metrics
        if 'us-gaap' in facts:
            gaap = facts['us-gaap']

            # Revenue
            if 'Revenue' in gaap and 'units' in gaap['Revenue']:
                revenue_data = gaap['Revenue']['units'].get('USD', [])
                if revenue_data:
                    metrics['Revenue'] = revenue_data[0]['val']
                    metrics['Revenue_currency'] = 'USD'

        # Assets
        if 'Assets' in gaap and 'units' in gaap['Assets']:
            assets_data = gaap['Assets']['units'].get('USD', [])
            if assets_data:
                metrics['assets'] = assets_data[0]['val']

        # Net Income
        if 'NetIncomeLoss' in gaap and 'units' in gaap['NetIncomeLoss']:
            net_income_data = gaap['NetIncomeLoss']['units'].get('USD', [])
            if net_income_data:
                metrics['net_income'] = net_income_data[0]['val']

        print(f"📊 Extracted {len(metrics)} key metrics for {ticker}")
        return metrics
    

    
# TEST WITH REAL DATA        
if __name__ == "__main__":
    collector = SECDataCollector(email="sasishasank2@gmail.com")

    # Test with Apple - REAL API CALL
    print("🔄 Fetching REAL Apple financial data from SEC...")
    apple_data = collector.company_facts("AAPL")

    if apple_data:
        metrics = collector.extract_key_metrics(apple_data, "AAPL")
        print("f💰 Apple Financial Metrics:")
        for metric, value in metrics.items():
            print(f" {metric}: {value}")

    else:
        print("❌ Failed to fetch Apple data")
        
        
            


            