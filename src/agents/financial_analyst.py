# src/agents/financial_analyst.py
import os
import asyncio
import logging
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
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
from tools.financial_tools import FinancialTools

class FinancialAgentTeam:
    """Financial Due Diligence Agent Team integrated with our RAG system"""
    def __init__(self, model_client: OpenAIChatCompletionClient, rag_system: ProductionRAGSystem):
        self.model_client = model_client
        self.rag_system = rag_system
        self.sec_collector = SECDataCollector()
        self.document_parser = DocumentProcessor()
        self.tools = FinancialTools(self.rag_system, self.sec_collector)
        self.team = None
        self.logger = self._setup_logging()
        self._create_agents()
        self._create_team()

    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for production"""
        logger = logging.getLogger(f"FinancialAgentTeam")
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
        """Financial Due Diligence Agent Team integrated with our RAG system"""

        # Financial Researcher - Uses our RAG system for data retrieval
        self.financial_researcher = AssistantAgent(
            name="financial_researcher",
            model_client=self.model_client,
            model_context=BufferedChatCompletionContext(buffer_size=15),
            tools = [self.tools.retrieve_financial_metrics, self.tools.get_company_overview],
            system_message="""You are a Financial Research Specialist. Your responsibilities:

        1. **Financial Data Retrieval**: Use retrieve_financial_metrics() to get SEC financial data
        2. **Company Overview**: Use get_company_overview() for business context
        3. **Trend Analysis**: Analyze revenue, profit, and growth trends
        4. **Data Quality**: Verify data completeness and reliability

        **Key Focus Areas:**
        - Revenue trends and growth drivers from SEC data
        - Profitability metrics analysis
        - Financial statement completeness
        - Data consistency across periods

        **Tools Available:**
        - retrieve_financial_metrics(company, metrics): Get specific financial metrics
        - get_company_overview(company): Get company business information

        Use 'RESEARCH_COMPLETE' when data collection is finished.
        """       
        )

        # Financial Analyst - Uses RAG data for quantitative analysis
        self.financial_analyst = AssistantAgent(
            name="financial_analyst",
            model_client=self.model_client,
            model_context=BufferedChatCompletionContext(buffer_size=12),
            tools=[self.tools.analyze_financial_ratios, self.tools.calculate_trends],
            system_message="""You are a Quantitative Financial Analyst. Your responsibilities:

        1. **Ratio Analysis**: Use analyze_financial_ratios() to examine profitability, liquidity, efficiency
        2. **Trend Calculation**: Use calculate_trends() for growth rate analysis
        3. **Benchmarking**: Compare ratios against industry standards
        4. **Performance Assessment**: Evaluate financial health and stability

        **Key Analysis Areas:**
        - Profitability: ROA, ROE, Margins
        - Liquidity: Current Ratio, Quick Ratio
        - Efficiency: Asset Turnover, Inventory Turnover
        - Solvency: Debt-to-Equity, Interest Coverage

        **Tools Available:**
        - analyze_financial_ratios(company): Get and analyze financial ratios
        - calculate_trends(company, metrics): Calculate growth trends

        Use 'ANALYSIS_COMPLETE' when quantitative analysis is finished.
        """
        )
        
        # Financial Reviewer - Validates and synthesizes findings
        self.financial_reviewer = AssistantAgent(
            name="financial_reviewer",
            model_client=self.model_client,
            model_context=BufferedChatCompletionContext(buffer_size=10),
            tools=[self.tools.validate_with_source_data, self.tools.generate_investment_summary],
            system_message="""You are a Senior Financial Reviewer. Your responsibilities:

        1. **Validation**: Use validate_with_source_data() to cross-check findings
        2. **Synthesis**: Integrate research and analysis into comprehensive assessment
        3. **Risk Assessment**: Evaluate financial risks and opportunities
        4. **Recommendations**: Provide data-driven investment recommendations

        **Review Checklist:**
        - Data consistency with SEC source documents
        - Logical reasoning in ratio analysis
        - Comprehensive risk assessment
        - Clear investment thesis supported by data

        **Tools Available:**
        - validate_with_source_data(company, findings): Cross-check with original SEC data
        - generate_investment_summary(company): Create final investment recommendation

        Use 'REVIEW_COMPLETE' when review and synthesis are finished.
        """
        )

    def _create_team(self):
        """Create the financial team with robust termination conditions"""

        # Comprehensive termination conditions
        termination_conditions = (
            TextMentionTermination("ANALYSIS_COMPLETE")|
            TextMentionTermination("REVIEW_COMPLETE")|
            TextMentionTermination("RESEARCH_COMPLETE")|
            TextMentionTermination("TERMINATE")|
            MaxMessageTermination(max_messages=30)
        )

        self.team = RoundRobinGroupChat(
            [self.financial_researcher, self.financial_analyst, self.financial_reviewer],
            termination_condition=termination_conditions,
            max_turns=25           
        )

    async def analyze_company(self, company_ticker: str, additional_context: str = "") -> Dict[str, Any]:
        """Run comprehensive financial due diligence with production-grade error handling"""
        try:
            # Input validation
            if not company_ticker or not company_ticker.strip():
                return self._create_error_result("Company ticker is required")
            
            company_ticker = company_ticker.upper().strip()

            # Ensure data availability
            data_available = await self.ensure_company_data(company_ticker)
            if not data_available:
                return self._create_error_result(f"Insufficient data available for {company_ticker}")
            
            self.logger.info(f" Starting financial analysis for {company_ticker}")

            # Build comprehensive task with context
            task = self._build_analysis_task(company_ticker, additional_context)

            # Execute analysis with timeout protection
            try:
                result = await asyncio.wait_for(
                    self.team.run(task=task),
                    timeout=300 # 5-minute timeout
                )
            except asyncio.TimeoutError:
                self.logger.error(f"Analysis timeout for {company_ticker}")
                return self._create_error_result(f"Analysis timeout - process took too long")
            
            # Process and validate results
            analysis_result = self._process_analysis_result(result, company_ticker)
            self.logger.info(f"Successfully completed analysis for {company_ticker}")
            return analysis_result
        
        except Exception as e:
            self.logger.error(f"Financial analysis failed for {company_ticker}: {e}")
            return self._create_error_result(f"Analysis failed: {str(e)}")
        
    def _build_analysis_task(self, company: str, additional_context: str) -> str:
        """Build comprehensive analysis task with context"""

        base_task = f"""
        Perform comprehensive financial due diligence for {company} using our SEC data pipeline.

        **Data Sources Available:**
        - SEC financial metrics (Revenue, Net Income, Assets, Liabilities, Equity)
        - Calculated financial ratios (ROA, ROE, Current Ratio, Debt-to-Equity)
        - Company business information and entity data
        - Multi-period financial data for trend analysis

        **Analysis Framework:**
        1. RESEARCH: Retrieve and validate SEC financial metrics
        2. ANALYSIS: Calculate and interpret financial ratios and trends
        3. REVIEW: Validate findings and generate data-driven recommendations

        **Quality Requirements:**
        - Cross-validate all findings with source SEC data
        - Consider multiple periods for trend analysis
        - Assess both strengths and risks
        - Provide specific, actionable recommendations

        **Additional Context:** {additional_context or "Standard comprehensive analysis"}

        Coordinate as a team and ensure thorough documentation of the analysis process.
        """
        return base_task.strip()

    def _create_error_result(self, message: str) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            "status": "error",
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def _process_analysis_result(self, result: Any, company: str) -> Dict[str, Any]:
        """Process and format the analysis result"""
        # Extract the final message from the conversation
        messages = result.messages
        summary = self._extract_meaningful_summary(messages)
        
        return {
            "status": "success",
            "company": company,
            "analysis": summary,
            "messages": [msg.to_dict() if hasattr(msg, 'to_dict') else str(msg) for msg in messages],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "messages_count": len(messages)
        }

    def _extract_meaningful_summary(self, messages) -> str:
         """Extract meaningful summary from team messages with quality checks"""
         try:
             # Prioritize reviewer messages, then analyst, then researcher
             for source in ['financial_reviewer', 'financial_analyst', 'financial_researcher']:
                 for msg in reversed(messages):
                     if hasattr(msg, 'source') and msg.source == source and hasattr(msg, 'content'):
                        content = msg.content.strip()
                        if len(content) > 50 and any(keyword in content.lower() for keyword in ['recommend', 'conclusion', 'summary', 'assessment']):
                            return f" {source}: {content[:500]}..."
                        
             # Fallback: find most substantial message
             substantial_messages = []
             for msg in messages:
                 if hasattr(msg, 'content') and len(msg.content.strip()) > 100:
                     substantial_messages.append((msg.source, msg.content))

             if substantial_messages:
                 source, content = substantial_messages[-1] # Get last substantial message
                 return f" {source}: {content[:400]}"
             
             return "Analysis completed - review detailed messages for complete assessment"
         
         except Exception as e:
             self.logger.warning(f"Summary extraction failed: {e}")
             return "Analysis completed - summary extraction unavailable"

    async def ensure_company_data(self, company: str) -> bool:
        """Ensure company data is available with production-grade error handling"""
        try:
            if not company or not company.strip():
                self.logger.error(f"Company ticker required for data assurance")
                return False
            
            company = company.upper().strip()

            # Check existing data with quality assessment
            metrics = self.rag_system.get_company_metrics(company)
            if metrics and len(metrics) > 5:
                self.logger.info(f"Company data verified for {company}: {len(metrics)} metrics")
                return True
            
            # Fetch and process new data
            self.logger.info(f"Fetching SEC data for {company}")

            company_data = self.sec_collector.company_facts(company)
            if not company_data:
                self.logger.error(f"SEC data fetch failed for {company}")
                return False
            
            documents = self.document_parser.process_sec_facts(company_data, company)
            if not documents or len(documents) < 3: # Require minimum documents
                self.logger.error(f"Insufficient documents processed for {company}")
                return False
            
            doc_ids = self.rag_system.add_company_data(documents)
            if not doc_ids or len(doc_ids) < 3:
                self.logger.error(f"RAG storage failed for {company}")
                return False

            self.logger.info(f"Successfully added {len(doc_ids)} documents for {company}")
            return True
        
        except Exception as e:
            self.logger.error(f"Data assurance failed for {company}")
            return False

    async def close(self):
        """Clean up resources with proper error handling"""
        try:
            if hasattr(self, 'model_client'):
                await self.model_client.close()
            self.logger.info("Financial agent team resources cleaned up")
        
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

# Production-grade factory function
async def create_financial_team(
        model: str = "gpt-4o",
                api_key: Optional[str] = None,
                rag_system: Optional[ProductionRAGSystem] = None,
                timeout: int = 30
) -> FinancialAgentTeam:
    """Create production-grade financial agent team with comprehensive setup"""
    try:
        # Environment variable fallback with validation
        if not api_key:
            api_key = os.getenv("OPEN_AI_KEY")
        
        if not api_key:
            raise ValueError(
                "OpenAI API key required. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )
        
        # Initialize RAG system with production settings
        if not rag_system:
            rag_system = ProductionRAGSystem(
                persist_directory="./data/vector_stores/financial_data",
                embedding_type="huggingface",
                chunk_size=800,
                chunk_overlap=100
            )

        # Configure model client for production use
        model_client = OpenAIChatCompletionClient(
            model=model,
            api_key=api_key,
            seed=42, # For reproducibility
            temperature=0.1,
            timeout=timeout,
            max_retries=3
        )

        return FinancialAgentTeam(model_client, rag_system)

    except Exception as e:
        logging.error(f"Failed to create financial agent team: {e}")
        raise

# Production-ready main execution
async def main():
    """Production-grade example usage"""

    # Configure logging for production
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    financial_team = None
    try:
        # Create financial agent team
        financial_team = await create_financial_team()  

        # Test with a company
        company = "AAPL"

        # Ensure data availability
        data_ready = await financial_team.ensure_company_data(company)
        if not data_ready:
            print(f"❌ Cannot analyze {company} - data unavailable")
            return

        # Run comprehensive analysis
        result = await financial_team.analyze_company(
            company_ticker=company,
            additional_context="Focus on profitability, growth trends, and risk assessment"
        ) 

        # Display results
        if result.get('status') == 'success':
            print(f"✅ Analysis completed for {result['company']}")
            print(f"📊 Summary: {result['analysis']}")
            print(f"⏱️  Timestamp: {result['timestamp']}")
            print(f"📝 Messages: {result['messages_count']}")

        else:
            print(f"❌ Analysis failed: {result.get('message', 'unknown error')}")   

    except Exception as e:
        logging.error(f"Main execution failed: {e}")

    finally:
        if financial_team:
            await financial_team.close()      

if __name__ == "__main__":
    asyncio.run(main())