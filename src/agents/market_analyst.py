# src/agents/market_analyst.py
import os
import sys
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.model_context import BufferedChatCompletionContext

# Local Imports
from src.rag.core import ProductionRAGSystem
from src.data.processors.document_parser import DocumentProcessor
from src.data.collectors.sec_edgar import SECDataCollector
from src.tools.market_tools import MarketTools

load_dotenv()

class MarketAgentTeam:
    """
    Market Analysis Agent Team.
    Orchestrates specialized agents for Industry, Competitor, and Opportunity analysis.
    """
    
    def __init__(self, model_client: OpenAIChatCompletionClient, rag_system: ProductionRAGSystem):
        self.model_client = model_client
        self.rag_system = rag_system
        self.logger = logging.getLogger(__name__)
        
        # Initialize Tools
        self.sec_collector = SECDataCollector()
        self.document_parser = DocumentProcessor()
        self.tools = MarketTools(self.rag_system, self.sec_collector)
        
        # Initialize Agents
        self._create_agents()
        self._create_team()

    def _create_agents(self):
        """Create market analysis agents with specialized roles"""

        # 1. Industry Analyst
        self.industry_analyst = AssistantAgent(
            name="industry_analyst",
            model_client=self.model_client,
            model_context=BufferedChatCompletionContext(buffer_size=15),
            tools=[
                self.tools.analyze_industry_trends, 
                self.tools.research_competitive_landscape
            ],
            system_message="""You are an Industry Analysis Specialist. Your responsibilities:
            1. **Market Trend Analysis**: Use analyze_industry_trends() to examine dynamics.
            2. **Growth Pattern Analysis**: Identify market growth drivers (CAGR, sector size).
            3. **Sector Performance**: Evaluate industry outlook.
            
            Focus on macro trends. Use 'INDUSTRY_ANALYSIS_COMPLETE' when done."""
        )

        # 2. Market Researcher
        self.market_researcher = AssistantAgent(
            name="market_researcher",
            model_client=self.model_client,
            model_context=BufferedChatCompletionContext(buffer_size=12),
            tools=[self.tools.assess_market_opportunities],
            system_message="""You are a Market Research Specialist. Your responsibilities:
            1. **Opportunity Assessment**: Use assess_market_opportunities() to identify growth areas.
            2. **Innovation Analysis**: Look for "forward-looking" statements in data regarding new products.
            3. **Gap Analysis**: Where is the company expanding?
            
            Focus on future potential. Use 'MARKET_RESEARCH_COMPLETE' when done."""
        )

        # 3. Competitive Analyst (The Closer)
        self.competitive_analyst = AssistantAgent(
            name="competitive_analyst",
            model_client=self.model_client,
            model_context=BufferedChatCompletionContext(buffer_size=10),
            tools=[
                self.tools.research_competitive_landscape, 
                self.tools.analyze_industry_trends
            ],
            system_message="""You are a Competitive Intelligence Specialist. Your responsibilities:
            1. **Competitive Benchmarking**: Compare company performance against peers using research_competitive_landscape().
            2. **SWOT Synthesis**: Combine Industry and Market findings into a Strategic Assessment.
            3. **Final Report**: Provide the final Market Strategy Report.
            
            Synthesize the findings of the Industry Analyst and Market Researcher.
            Use 'MARKET_ANALYSIS_COMPLETE' when the final report is ready."""
        )

    def _create_team(self):
        """Create the market analysis team with robust termination conditions"""
        termination_conditions = (
            TextMentionTermination("MARKET_ANALYSIS_COMPLETE") |
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

            # Ensure data availability (Auto-ingest if missing)
            data_available = await self._ensure_market_data(company_ticker)
            if not data_available:
                return self._create_error_result(f"Insufficient market data available for {company_ticker} and fetch failed.")
            
            self.logger.info(f"Starting market analysis for {company_ticker}")

            # Build task
            task = self._build_market_analysis_task(company_ticker, additional_context)

            # Execute with timeout
            try:
                # Reset team state before running (fixes "team already running" error)
                await self.team.reset()
                result = await asyncio.wait_for(
                    self.team.run(task=task),
                    timeout=300 # 5-minute timeout
                )
            except asyncio.TimeoutError:
                self.logger.error(f"Market analysis timeout for {company_ticker}")
                return self._create_error_result("Market analysis timeout - process took too long")
            
            # Process results
            analysis_result = self._process_market_result(result, company_ticker)
            self.logger.info(f"Successfully completed market analysis for {company_ticker}")
            return analysis_result

        except Exception as e:
            self.logger.error(f"Market analysis failed for {company_ticker}: {e}")
            return self._create_error_result(f"Market analysis failed: {str(e)}")
        
    def _build_market_analysis_task(self, company: str, additional_context: str) -> str:
        """Builds the prompt for the market team."""
        return f"""
        Perform comprehensive market analysis for {company}.

        **Context:** {additional_context or "Standard competitive landscape analysis"}

        **Process:**
        1. INDUSTRY: Analyze sector trends, growth rates, and regulatory impacts.
        2. RESEARCH: Identify specific growth opportunities and new market entries.
        3. COMPETITIVE: Compare against peers and synthesize a Market Strategy Report.

        Coordinate as a team. Do not fabricate data.
        """
    
    async def _ensure_market_data(self, company: str) -> bool:
        """
        Ensures market data exists in RAG. If not, fetches from SEC and ingests it.
        Uses broader queries that work with financial_metric data.
        """
        try:
            if not company: return False
            
            # 1. Check existing RAG data with BROADER queries
            # These work with financial_metric data that exists for AAPL
            market_queries = [
                f"{company} revenue",
                f"{company} assets",
                f"{company} liabilities",
                f"{company} financial",
            ]

            for query in market_queries:
                docs = self.rag_system.query(query, company=company, k=1)
                if docs:
                    # If ANY data found, proceed with analysis
                    self.logger.info(f"Market data verified for {company}")
                    return True
            
            self.logger.info(f"Market data missing for {company}. Initiating Fetch...")

            # 2. Fetch from SEC (Blocking -> Thread)
            company_data = await asyncio.to_thread(self.sec_collector.company_facts, company)
            if not company_data: return False

            # 3. Process (Blocking -> Thread)
            documents = await asyncio.to_thread(
                self.document_parser.process_sec_facts, company_data, company
            )
            if not documents: return False

            # 4. Ingest
            doc_ids = self.rag_system.add_company_data(documents)
            if not doc_ids: return False

            self.logger.info(f"Ingested {len(doc_ids)} documents for market analysis.")
            return True
        
        except Exception as e:
            self.logger.error(f"Market data assurance failed: {e}")
            return False
        
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        return {
            'company': 'UNKNOWN',
            'error': error_message,
            'success': False,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _process_market_result(self, result, company: str) -> Dict[str, Any]:
        """Extracts the final summary from the team interaction."""
        try:
            # Handle AutoGen result object
            messages = result.messages if hasattr(result, 'messages') else []
            summary = self._extract_market_summary(messages)

            return {
                'company': company,
                'summary': summary,
                'messages': [msg.to_dict() if hasattr(msg, 'to_dict') else str(msg) for msg in messages],
                'success': True,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            self.logger.error(f"Result processing failed: {e}")
            return self._create_error_result(f"Result processing failed: {str(e)}")

    def _extract_market_summary(self, messages: List[Any]) -> str:
        """
        Iterates backwards through messages to find the final conclusion.
        """
        try:
            if not messages: return "No analysis generated."

            target_order = ['competitive_analyst', 'industry_analyst', 'market_researcher']
            
            for source in target_order:
                for msg in reversed(messages):
                    msg_source = getattr(msg, 'source', '') or (msg.get('source') if isinstance(msg, dict) else '')
                    msg_content = getattr(msg, 'content', '') or (msg.get('content') if isinstance(msg, dict) else '')

                    # FIX: Use substring matching to handle AutoGen's appended UUIDs
                    if source in msg_source and msg_content:
                        content = str(msg_content).strip()
                        if len(content) > 50:
                            return content

            return "Market analysis completed (Summary extraction failed)"
        except Exception as e:
            self.logger.warning(f"Summary extraction error: {e}")
            return "Market analysis completed (Error extracting summary)"

    async def close(self):
        """Cleanup resources."""
        try:
            if hasattr(self.model_client, 'close'):
                await self.model_client.close()
        except: pass

# --- Factory Function ---
async def create_market_team(
        model: str = "gpt-4-turbo", 
        api_key: str = None,
        rag_system: ProductionRAGSystem = None,
        timeout: int = 30
) -> MarketAgentTeam:
    """Factory to create the Market Team."""
    
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key: raise ValueError("OPENAI_API_KEY required.")

    if not rag_system:
        # Use a distinct path or shared path depending on architecture
        base_path = Path(os.getcwd()) / "data" / "vector_stores" / "market_data"
        rag_system = ProductionRAGSystem(persist_directory=str(base_path), embedding_type="huggingface")

    client = OpenAIChatCompletionClient(
        model=model, 
        api_key=api_key, 
        temperature=0.1,
        timeout=timeout
    )
    
    return MarketAgentTeam(client, rag_system)

# --- Main Entry Point ---
if __name__ == "__main__":
    async def main():
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        
        print("Initializing Market Agent Team...")
        market_team = None
        try:
            market_team = await create_market_team()
            
            ticker = "AAPL"
            print(f"Analyzing Market Strategy for {ticker}...")
            
            result = await market_team.analyze_company_market(
                ticker, 
                "Focus on competitive positioning vs. Microsoft and Google."
            )
            
            if result['success']:
                print(f"\n--- MARKET REPORT FOR {ticker} ---\n")
                print(result['summary'])
                print("\n----------------------------------")
            else:
                print(f"Error: {result['error']}")
                
        except Exception as e:
            print(f"Fatal Error: {e}")
        finally:
            if market_team: await market_team.close()

    asyncio.run(main())