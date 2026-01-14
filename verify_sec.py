import requests
import os
from dotenv import load_dotenv

load_dotenv()

email = os.getenv('SEC_EDGAR_EMAIL', 'sasishasank2@gmail.com')
headers = {
    'User-Agent': f"SasiResearch/1.0 ({email})",
    'Accept-Encoding': 'gzip, deflate'
}

print(f"Testing connection with: {headers['User-Agent']}")

# Test URL (Apple's Ticker Data)
url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"

try:
    r = requests.get(url, headers=headers, timeout=10)
    print(f"Status Code: {r.status_code}")
    
    if r.status_code == 200:
        print("[OK] SUCCESS! Data received.")
        print(f"Keys found: {list(r.json().keys())}")
    elif r.status_code == 403:
        print("[ERROR] FAILED: 403 Forbidden (Blocked).")
    else:
        print(f"[ERROR] FAILED: {r.status_code}")
except Exception as e:
    print(f"[ERROR] ERROR: {e}")