import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

print("Importing rag.core...")
from rag.core import ProductionRAGSystem
print("✅ rag.core imported")

print("Importing financial_analyst...")
from agents.financial_analyst import FinancialAgentTeam
print("✅ financial_analyst imported")

print("Importing market_analyst...")
from agents.market_analyst import MarketAgentTeam
print("✅ market_analyst imported")

print("Importing legal_reviewer...")
from agents.legal_reviewer import LegalAgentTeam
print("✅ legal_reviewer imported")
