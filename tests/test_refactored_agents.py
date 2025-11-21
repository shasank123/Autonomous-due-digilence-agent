import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.financial_analyst import FinancialAgentTeam
from agents.market_analyst import MarketAgentTeam
from agents.legal_reviewer import LegalAgentTeam
from tools.financial_tools import FinancialTools
from tools.market_tools import MarketTools
from tools.legal_tools import LegalTools

class TestRefactoredAgents(unittest.TestCase):
    def setUp(self):
        self.mock_model_client = MagicMock()
        self.mock_rag_system = MagicMock()

    def test_financial_agent_initialization(self):
        agent = FinancialAgentTeam(self.mock_model_client, self.mock_rag_system)
        self.assertIsInstance(agent.tools, FinancialTools)
        self.assertTrue(hasattr(agent, 'financial_researcher'))
        self.assertTrue(hasattr(agent, 'financial_analyst'))
        self.assertTrue(hasattr(agent, 'financial_reviewer'))
        print("✅ FinancialAgentTeam initialized successfully with FinancialTools")

    def test_market_agent_initialization(self):
        agent = MarketAgentTeam(self.mock_model_client, self.mock_rag_system)
        self.assertIsInstance(agent.tools, MarketTools)
        self.assertTrue(hasattr(agent, 'industry_analyst'))
        self.assertTrue(hasattr(agent, 'market_researcher'))
        self.assertTrue(hasattr(agent, 'competitive_analyst'))
        print("✅ MarketAgentTeam initialized successfully with MarketTools")

    def test_legal_agent_initialization(self):
        agent = LegalAgentTeam(self.mock_model_client, self.mock_rag_system)
        self.assertIsInstance(agent.tools, LegalTools)
        self.assertTrue(hasattr(agent, 'compliance_analyst'))
        self.assertTrue(hasattr(agent, 'risk_assessor'))
        self.assertTrue(hasattr(agent, 'contract_reviewer'))
        print("✅ LegalAgentTeam initialized successfully with LegalTools")

if __name__ == '__main__':
    unittest.main()
