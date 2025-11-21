# src/agents/market_analyst.py
import os
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Tuple
from langchain_core.documents import Document
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.model_context import BufferedChatCompletionContext
import sys

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from rag.core import ProductionRAGSystem
from data.processors.document_parser import DocumentProcessor
from data.collectors.sec_edgar import SECDataCollector
from tools.market_tools import MarketTools

class MarketAgentTeam:
    """Market Analysis Agent Team integrated with our RAG system"""
    
    def __init__(self, model_client: OpenAIChatCompletionClient, rag_system: ProductionRAGSystem):
        self.model_client = model_client
        self.rag_system = rag_system
        self.sec_collector = SECDataCollector()
        self.document_parser = DocumentProcessor()
        self.tools = MarketTools(self.rag_system, self.sec_collector)
        self.team = None
        self.logger = self._setup_logging()
        self._create_agents()
        self._create_team()

    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for production"""
        logger = logging.getLogger("MarketAgentTeam")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _create_agents(self):
        """Create market analysis agents with specialized roles"""

        # Industry Analyst - Market trends and competitive landscape
        self.industry_analyst = AssistantAgent(
            name="industry_analyst",
            model_client=self.model_client,
            model_context=BufferedChatCompletionContext(buffer_size=15),
            tools = [self.tools.analyze_industry_trends, self.tools.research_competitive_landscape],
            system_message="""You are an Industry Analysis Specialist. Your responsibilities:

            1. **Market Trend Analysis**: Use analyze_industry_trends() to examine industry dynamics
            2. **Competitive Landscape**: Use research_competitive_landscape() to analyze competitors
            3. **Growth Pattern Analysis**: Identify market growth drivers and patterns
            4. **Sector Performance**: Evaluate industry sector performance and outlook

            **Key Focus Areas:**
            - Industry growth rates and market size
            - Competitive positioning and market share
            - Emerging trends and disruptions
            - Regulatory and macroeconomic impacts

            **Tools Available:**
            - analyze_industry_trends(company, industry): Analyze market trends and positioning
            - research_competitive_landscape(company, competitors): Research competitive environment

            Use 'INDUSTRY_ANALYSIS_COMPLETE' when industry analysis is finished.
            """
        )

        # Market Researcher - Industry reports and market intelligence
        self.market_researcher = AssistantAgent(
            name = "market_researcher",
            model_client=self.model_client,
            model_context=BufferedChatCompletionContext(buffer_size=12),
            tools = [self.tools.assess_market_opportunities],
            system_message="""You are a Market Research Specialist. Your responsibilities:

            1. **Opportunity Assessment**: Use assess_market_opportunities() to identify growth areas
            2. **Market Intelligence**: Gather and analyze market data and reports
            3. **Segment Analysis**: Evaluate specific market segments and niches
            4. **Growth Forecasting**: Project market growth and opportunity sizing

            **Key Analysis Areas:**
            - Market opportunity identification and sizing
            - Segment growth potential and attractiveness
            - Customer needs and market gaps
            - Innovation and emerging market trends

            **Tools Available:**
            - assess_market_opportunities(company, segments): Evaluate growth opportunities

            Use 'MARKET_RESEARCH_COMPLETE' when market research is finished.
            """
        )

        # Competitive Analyst - Peer comparison and positioning
        self.competitive_analyst = AssistantAgent(
            name = "competitive_analyst",
            model_client = self.model_client,
            model_context=BufferedChatCompletionContext(buffer_size=10),
            tools = [self.tools.research_competitive_landscape, self.tools.analyze_industry_trends],
            system_message="""You are a Competitive Intelligence Specialist. Your responsibilities:

            1. **Competitive Benchmarking**: Compare company performance against peers
            2. **Positioning Analysis**: Evaluate competitive positioning and differentiation
            3. **Strategic Assessment**: Analyze competitor strategies and capabilities
            4. **Threat Analysis**: Identify competitive threats and market pressures

            **Review Checklist:**
            - Comprehensive competitor profiling
            - SWOT analysis (Strengths, Weaknesses, Opportunities, Threats)
            - Market share dynamics and trends
            - Strategic implications and recommendations

            **Tools Available:**
            - research_competitive_landscape(company, competitors): Analyze competitors
            - analyze_industry_trends(company, industry): Contextual industry analysis

            Use 'COMPETITIVE_ANALYSIS_COMPLETE' when competitive analysis is finished.
            """
        )

    def _create_team(self):
        """Create the market analysis team with robust termination conditions"""

        termination_conditions = (
            TextMentionTermination("MARKET_ANALYSIS_COMPLETE") |
            TextMentionTermination("INDUSTRY_ANALYSIS_COMPLETE") |
            TextMentionTermination("COMPETITIVE_ANALYSIS_COMPLETE") |
            TextMentionTermination("MARKET_RESEARCH_COMPLETE") |
            TextMentionTermination("TERMINATE") |
            MaxMessageTermination(max_messages=25)
        )

        self.team = RoundRobinGroupChat(
            [self.industry_analyst, self.market_researcher, self.competitive_analyst],
            termination_condition=termination_conditions,
            max_turns=20
        )

    async def analyze_company_market(self, company_ticker: str, additional_context: str = "") -> Dict[str, Any]:
        """Run comprehensive market analysis with production-grade error handling"""

        try:
            # Input validation
            if not company_ticker or not company_ticker.strip():
                return self._create_error_result("Company ticker is required")
            
            company_ticker = company_ticker.upper().strip()

            # Ensure data availabilit
            data_available =  await self._ensure_market_data(company_ticker)
            if not data_available:
                return self._create_error_result(f"Insufficient market data available for {company_ticker}")
            
            self.logger.info(f"Starting market analysis for {company_ticker}")

            # Build comprehensive task with context
            task = self._build_market_analysis_task(company_ticker, additional_context)

            # Execute analysis with timeout protection
            try:
                result = await asyncio.wait_for(
                    self.team.run(task=task),
                    timeout=300 # 5-minute timeout
                )

            except asyncio.TimeoutError:
                self.logger.error(f"Market analysis timeout for {company_ticker}")
                return self._create_error_result("Market analysis timeout - process took too long")
            
            # Process and validate results
            analysis_result = self._process_market_result(result, company_ticker)
            self.logger.info(f"Successfully completed market analysis for {company_ticker}")
            return analysis_result

        except Exception as e:
            self.logger.error(f"Market analysis failed for {company_ticker}: {e}")
            return self._create_error_result(f"Market analysis failed: {str(e)}")
        
    def _build_market_analysis_task(self, company: str, additional_context: str) -> str:
        """Build comprehensive market analysis task with context"""

        base_task = f"""
        Perform comprehensive market analysis for {company} using our market intelligence pipeline.

        **Market Data Sources Available:**
        - Industry trends and growth analysis
        - Competitive landscape intelligence
        - Market opportunity assessments
        - Peer comparison and benchmarking
        - Innovation and emerging trend analysis

        **Market Analysis Framework:**
        1. INDUSTRY: Analyze market trends, growth drivers, and sector dynamics
        2. COMPETITIVE: Research competitive landscape and positioning
        3. OPPORTUNITY: Assess market opportunities and growth potential
        4. SYNTHESIS: Integrate findings into strategic market assessment

        **Analysis Quality Requirements:**
        - Cross-validate market data from multiple sources
        - Consider both current market position and future potential
        - Evaluate competitive threats and market opportunities
        - Provide specific, data-driven market recommendations

        **Additional Context:** {additional_context or "Standard comprehensive market analysis"}

        Coordinate as a market analysis team and ensure thorough documentation of the analysis process.
        """
        return base_task.strip()
    
    async def _ensure_market_data(self, company: str) -> bool:
        """Ensure market data is available with production-grade error handling"""

        try:
            if not company or not company.strip():
                self.logger.warning("Company ticker required for market data assurance")
                return False
            
            company = company.upper().strip()

            # Check existing market data with quality assessment
            market_queries = [
                f"{company} market analysis",
                f"{company} competitive landscape",
                f"{company} industry trends",
                f"{company} growth opportunities"
            ]

            market_docs_found = 0
            
            for query in market_queries:
                docs = self.rag_system.query(query, company=company, k=2)
                if docs:
                    market_docs_found += 1

            if market_docs_found >= 4:  # Require minimum market documents
                self.logger.info(f"Market data verified for {company}: {market_docs_found} documents")
                return True
            
            self.logger.warning(f"Insufficient market data for {company}: {market_docs_found} documents")
            return False
        
        except Exception as e:
            self.logger.error(f"Market data assurance failed for {company}: {e}")
            return False
        
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result for market operations"""

        return {
            'company': 'UNKNOWN',
            'error': error_message,
            'market_integration': False,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'success': False
        }
    
    def _process_market_result(self, result, company: str) -> Dict[str, Any]:
        """Process and validate market analysis results"""

        try:
            summary = self._extract_market_summary(result.messages)

            return {
                'company': company,
                'messages': [msg.to_dict() for msg in result.messages],
                'stop_reason': result.stop_reason,
                'summary': summary,
                'market_integration': True,
                'data_source': 'Market Intelligence + Production RAG',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message_count': len(result.messages),
                'success': True
            }
        
        except Exception as e:
            self.logger.error(f"Market result processing failed for {company}: {e}")
            return self._create_error_result(f"Market result processing failed: {str(e)}")

    def _extract_market_summary(self, messages: str) -> str:
        """Extract meaningful market summary from team messages"""

        try:
            # Prioritize competitive analyst messages, then industry analyst, then market researcher
            for source in ['competitive_analyst', 'industry_analyst', 'market_researcher']:
                for msg in reversed(messages):
                    if hasattr(msg, 'source') and msg.source == source and hasattr(msg, 'content'):
                        content = msg.content.strip()

                    if len(content) > 50 and any(keyword in content.lower() for keyword in ['recommend', 'conclusion', 'summary', 'assessment', 'market', 'competitive']):
                        return f" {source}: {content[:500]}..."

            # Fallback: find most substantial market message   
            substantial_messages = []
            for msg in messages:
                if hasattr(msg, 'content') and len(msg.content.strip()) > 100:
                    substantial_messages.append((msg.source, msg.content))

            if substantial_messages:
                source, content = substantial_messages[:-1]
                return f" {source}: {content[:400]}..."

            return "Market analysis completed - review detailed messages for complete assessment"

        except Exception as e:
            self.logger.warning(f"Market summary extraction failed: {e}")
            return "Market analysis completed - summary extraction unavailable"

    async def close(self):
        """Clean up market resources with proper error handling"""

        try:
            if hasattr(self, 'model_client'):
                await self.model_client.close()
                self.logger.info("Market agent team resources cleaned up")
        
        except Exception as e:
            self.logger.error(f"Error during market cleanup: {e}")

# Production-grade factory function for Market Agent Team
async def create_market_team(
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        rag_system: Optional[ProductionRAGSystem] = None,
        timeout: int = 30
) -> MarketAgentTeam:
    """Create production-grade market agent team with comprehensive setup"""

    try:
        # Environment variable fallback with validation
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OpenAI API key required. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )
        
        # Initialize RAG system with production settings
        if not rag_system:
            rag_system = ProductionRAGSystem(
                persist_directory="./data/vector_stores/market_data", # Separate store for market docs
                embedding_type="huggingface",
                chunk_size=800,
                chunk_overlap=100
            )

        # Configure model client for production use
        model_client = OpenAIChatCompletionClient(
            model=model,
            api_key=api_key,
            seed=42,  # For reproducibility
            temperature=0.1,
            timeout=timeout,
            max_retries=3
        )

        return MarketAgentTeam(model_client, rag_system)
    
    except Exception as e:
        logging.error(f"Failed to create market agent team: {e}")
        raise

# Production-ready main execution
async def main():
    """Production-grade example usage"""

    # Configure logging for production
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    market_team = None
    try:
        # Create market agent team
        market_team = await create_market_team()  

        # Test with a company
        company = "AAPL"

        # Ensure data availability
        data_ready = await market_team._ensure_market_data(company)
        if not data_ready:
            print(f"❌ Cannot analyze {company} - data unavailable")
            return

        # Run comprehensive analysis
        result = await market_team.analyze_company_market(
            company_ticker=company,
            additional_context="Focus on competitive positioning and emerging opportunities"
        ) 

        # Display results
        if result.get('success'):
            print(f"✅ Analysis completed for {result['company']}")
            print(f"📊 Summary: {result['summary']}")
            print(f"⏱️  Timestamp: {result['timestamp']}")
            print(f"📝 Messages: {result['message_count']}")

        else:
            print(f"❌ Analysis failed: {result.get('error', 'unknown error')}")   

    except Exception as e:
        logging.error(f"Main execution failed: {e}")

    finally:
        if market_team:
            await market_team.close()      

if __name__ == "__main__":
    asyncio.run(main())
