
import unittest
from unittest.mock import MagicMock, AsyncMock
import sys
import os
import asyncio

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock external dependencies
sys.modules['langchain_core'] = MagicMock()
sys.modules['langchain_core.documents'] = MagicMock()
sys.modules['autogen_agentchat'] = MagicMock()
sys.modules['autogen_agentchat.agents'] = MagicMock()
sys.modules['autogen_agentchat.teams'] = MagicMock()
sys.modules['autogen_agentchat.conditions'] = MagicMock()
sys.modules['autogen_ext'] = MagicMock()
sys.modules['autogen_ext.models'] = MagicMock()
sys.modules['autogen_ext.models.openai'] = MagicMock()
sys.modules['autogen_core'] = MagicMock()
sys.modules['autogen_core.model_context'] = MagicMock()
sys.modules['rag'] = MagicMock()
sys.modules['rag.core'] = MagicMock()
sys.modules['data'] = MagicMock()
sys.modules['data.processors'] = MagicMock()
sys.modules['data.processors.document_parser'] = MagicMock()
sys.modules['data.collectors'] = MagicMock()
sys.modules['data.collectors.sec_edgar'] = MagicMock()

from agents.market_analyst import MarketAgentTeam

class TestMarketAgentTeam(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.mock_client = MagicMock()
        self.mock_rag = MagicMock()
        # Mock _create_agents and _create_team to avoid side effects during init
        with unittest.mock.patch('agents.market_analyst.MarketAgentTeam._create_agents'), \
             unittest.mock.patch('agents.market_analyst.MarketAgentTeam._create_team'), \
             unittest.mock.patch('agents.market_analyst.MarketAgentTeam._setup_logging'):
            self.agent = MarketAgentTeam(self.mock_client, self.mock_rag)
            # Manually set logger since we mocked setup_logging
            self.agent.logger = MagicMock()

    async def test_identify_company_industry_fallback(self):
        # Test fix for industry_mapping.get(company)
        # Force rag query to return nothing so it hits fallback
        self.mock_rag.query_with_similarity_scores.return_value = []
        
        industry = await self.agent._identify_company_industry('AAPL')
        self.assertEqual(industry, 'Technology') # Should get from mapping, not 'Unknown Industry'

    def test_assess_risk_severity_typo(self):
        # Test fix for _assess_risk_severity name
        severity = self.agent._assess_risk_severity("This is a high risk investment")
        self.assertEqual(severity, "HIGH")

    async def test_analyze_competitive_advantages_loop(self):
        # Test fix for advantage_types loop variable
        # Mock RAG response
        mock_doc = MagicMock()
        mock_doc.page_content = "We have superior technology and innovation."
        self.mock_rag.query_with_similarity_scores.return_value = [(mock_doc, 0.9)]
        
        advantages = await self.agent._analyze_competitive_advantages('TEST')
        # If loop is fixed, it should find the advantage
        self.assertTrue(len(advantages) > 0)
        self.assertEqual(advantages[0]['type'], 'technology')

    def test_format_competitive_analysis_report_key_error(self):
        # Test fix for 'weaknesses' key error
        analysis = {
            'direct_competitors': [{
                'competitor': 'COMP',
                'strengths': [],
                'weaknesses': ['weakness1'] # Key is 'weaknesses'
            }],
            'competitive_advantages': []
        }
        # Should not raise KeyError
        report = self.agent._format_competitive_analysis_report(analysis, 'TEST', ['COMP'])
        self.assertIn("Weaknesses: 1 identified", report)

    async def test_assess_market_opportunities_extend(self):
        # Test fix for extend vs append
        # Mock _identify_market_segments
        self.agent._identify_market_segments = AsyncMock(return_value=['Seg1'])
        # Mock _analyze_market_segment
        self.agent._analyze_market_segment = MagicMock(return_value={'growth_potential': 'HIGH', 'segment': 'Seg1', 'key_insights': [], 'barriers': []})
        # Mock _analyze_innovation_opportunities
        self.agent._analyze_innovation_opportunities = MagicMock(return_value=[{'area': 'tech', 'content': 'stuff', 'score': 0.9}])
        # Mock format report to avoid errors there
        self.agent._format_opportunity_assessment_report = MagicMock(return_value="Report")

        await self.agent.assess_market_opportunities('TEST')
        
        # Check if innovation_areas is flat list (implied by successful execution and logic)
        # We can't easily check internal state unless we inspect the call to format report
        call_args = self.agent._format_opportunity_assessment_report.call_args
        opportunity_analysis = call_args[0][0]
        self.assertIsInstance(opportunity_analysis['innovation_areas'], list)
        # It should contain the dict directly, not a list
        self.assertIsInstance(opportunity_analysis['innovation_areas'][0], dict)

    async def test_analyze_market_segment_assignment(self):
        # Test fix for == vs =
        mock_doc = MagicMock()
        mock_doc.page_content = "high growth opportunity"
        self.mock_rag.query_with_similarity_scores.return_value = [(mock_doc, 0.9)]

        analysis = await self.agent._analyze_market_segment('TEST', 'Seg1')
        self.assertEqual(analysis['growth_potential'], 'HIGH')

if __name__ == '__main__':
    unittest.main()
