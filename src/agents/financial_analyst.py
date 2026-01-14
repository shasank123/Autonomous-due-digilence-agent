# src/agents/financial_analyst.py
import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# AutoGen Imports
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination

# Local Imports
from src.rag.core import ProductionRAGSystem
from src.tools.financial_tools import FinancialTools
from src.data.collectors.sec_edgar import SECDataCollector
from src.data.processors.document_parser import DocumentProcessor

load_dotenv()

class FinancialAgentTeam:
    """
    Orchestrator Class.
    Manages the lifecycle of the Financial Analyst Team, Tool Access, and RAG Integration.
    """
    def __init__(self, model_client: OpenAIChatCompletionClient, rag_system: ProductionRAGSystem):
        self.model_client = model_client
        self.rag_system = rag_system
        self.logger = logging.getLogger(__name__)
        
        # Initialize Tools
        # Note: We create instances here. 
        self.sec_collector = SECDataCollector()
        self.financial_tools = FinancialTools(rag_system, self.sec_collector)
        self.document_parser = DocumentProcessor()

        # 1. Financial Researcher Agent
        self.financial_researcher = AssistantAgent(
            name="financial_researcher",
            model_client=model_client,
            tools=[
                self.financial_tools.retrieve_financial_metrics,
                self.financial_tools.get_company_overview,
                self.financial_tools.calculate_trends
            ],
            system_message="""You are an expert financial researcher.
            Your role is to:
            1. Retrieve accurate financial data from SEC filings using the provided tools.
            2. Verify data availability and quality.
            3. Provide raw financial metrics and trend analysis.
            
            Always cite your data sources. If data is missing, report it clearly."""
        )

        # 2. Financial Analyst Agent
        self.financial_analyst = AssistantAgent(
            name="financial_analyst",
            model_client=model_client,
            tools=[
                self.financial_tools.analyze_financial_ratios,
                self.financial_tools.generate_investment_summary,
                self.financial_tools.validate_with_source_data
            ],
            system_message="""You are a senior financial analyst.
            Your role is to:
            1. Analyze financial ratios (Profitability, Liquidity, Solvency).
            2. Evaluate company performance based on the metrics provided by the researcher.
            3. Identify key risks and opportunities.
            
            Base your analysis STRICTLY on the provided data. Do not hallucinate numbers."""
        )

        # 3. Compliance Reviewer Agent (The Guardrail)
        self.financial_reviewer = AssistantAgent(
            name="financial_reviewer",
            model_client=model_client,
            # Reviewer typically relies on context, but can call validation tools if needed
            tools=[self.financial_tools.validate_with_source_data],
            system_message="""You are a risk manager and compliance reviewer.
            Your role is to:
            1. Review the Analyst's output for accuracy and bias.
            2. Ensure all claims are supported by the retrieved data.
            3. Flag inconsistencies.
            
            If the analysis is solid, output the Final Investment Recommendation.
            FORMAT:
            ### Executive Summary
            [Text]
            ### Key Metrics
            [Text]
            ### Risks
            [Text]
            
            DO NOT add conversational filler (e.g., "Great job", "Here is the report").
            Just output the report content.
            
            When you are satisfied, end your message with "TERMINATE"."""
        )

        # Termination Conditions
        # 1. "TERMINATE" in text (Reviewer sign-off)
        # 2. Max 40 messages (Increased to handle complex cases like liquidity analysis)
        #    This prevents premature termination when agents need to retry missing data
        termination_conditions = (
            TextMentionTermination("TERMINATE") |
            MaxMessageTermination(max_messages=40)
        )

        # Team Structure
        self.team = RoundRobinGroupChat(
            [self.financial_researcher, self.financial_analyst, self.financial_reviewer],
            termination_condition=termination_conditions,
            max_turns=30  # Increased from 20 to align with message limit
        )

    async def analyze_company(self, company_ticker: str, additional_context: str = "") -> Dict[str, Any]:
        """Run comprehensive financial due diligence with production-grade error handling"""
        try:
            # Input validation
            if not company_ticker or not company_ticker.strip():
                return self._create_error_result("Company ticker is required")
            
            company_ticker = company_ticker.upper().strip()

            # --- FORCE RUN FIX START ---
            # We perform the check, but we DO NOT abort if it fails.
            # This assumes the operator (you) has verified data exists via other means (seed_data.py).
            data_available = await self.ensure_company_data(company_ticker)
            
            if not data_available:
                self.logger.warning(f" [FORCE RUN] Data check failed for {company_ticker}, but proceeding anyway as requested.")
            else:
                self.logger.info(f" [OK] Data check passed for {company_ticker}.")
            # --- FORCE RUN FIX END ---
            
            self.logger.info(f"Starting financial analysis for {company_ticker}")

            # Build comprehensive task with context
            task = self._build_analysis_task(company_ticker, additional_context)

            # Execute analysis with timeout protection
            try:
                # Reset team state before running (fixes "team already running" error)
                await self.team.reset()
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

    async def ensure_company_data(self, company: str) -> bool:
        """
        Ensures data exists in RAG. If not, fetches from SEC and processes it.
        Uses asyncio.to_thread to prevent blocking the agent loop during downloads.
        """
        try:
            # 1. Check existing RAG data
            # Use 'await asyncio.to_thread' if the vector store call is blocking, 
            # but LangChain calls are usually fast enough.
            metrics = self.rag_system.get_company_metrics(company)
            
            if metrics and len(metrics) > 3:
                self.logger.info(f"Data verified for {company}: {len(metrics)} metrics found.")
                return True
            
            self.logger.info(f"Data missing for {company}. Initiating SEC Fetch...")

            # 2. Fetch from SEC (Blocking Operation -> Run in Thread)
            company_data = await asyncio.to_thread(self.sec_collector.company_facts, company)
            
            if not company_data:
                self.logger.error(f"SEC data fetch failed for {company}")
                return False
            
            # 3. Process Data (CPU Bound -> Run in Thread)
            documents = await asyncio.to_thread(
                self.document_parser.process_sec_facts, company_data, company
            )
            
            if not documents:
                self.logger.error(f"Document processing failed for {company}")
                return False
            
            # 4. Add to RAG
            doc_ids = self.rag_system.add_company_data(documents)
            
            if not doc_ids:
                self.logger.error(f"RAG storage failed for {company}")
                return False

            self.logger.info(f"Successfully ingested {len(doc_ids)} documents for {company}")
            return True
        
        except Exception as e:
            self.logger.error(f"Data assurance failed for {company}: {e}")
            return False

    def _build_analysis_task(self, company: str, additional_context: str) -> str:
        """Constructs the prompt for the agent team."""
        return f"""
        Perform comprehensive financial due diligence for {company}.

        **Context:** {additional_context or "Standard fundamental analysis"}

        **Process:**
        1. RESEARCHER: Fetch 'Revenue', 'Net Income', 'Assets', 'Liabilities' and trends.
        2. ANALYST: Calculate Ratios (ROA, ROE, Debt/Equity) and interpret them.
        3. REVIEWER: Validate the findings. If valid, provide your final summary and complete the review.

        Use the tools provided. Do not fabricate data.
        """

    def _process_analysis_result(self, result: Any, company: str) -> Dict[str, Any]:
        """Extracts the final answer from the chat history and structured metrics."""
        # Handle AutoGen result object
        messages = result.messages if hasattr(result, 'messages') else []
        
        summary = self._extract_meaningful_summary(messages)
        
        # Extract structured financial data for visualization
        structured_data = None
        try:
            structured_data = self.financial_tools.extract_structured_metrics(company)
            self.logger.info(f"Extracted structured data with {len(structured_data.get('metrics', {}))} metrics")
        except Exception as e:
            self.logger.warning(f"Could not extract structured metrics: {e}")
            structured_data = {"metrics": {}, "trends": {}, "ratios": {}}
        
        return {
            "status": "success",
            "company": company,
            "analysis": summary,
            "structured_data": structured_data,
            "messages_count": len(messages),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def _extract_meaningful_summary(self, messages: List[Any]) -> str:
        """Extract meaningful summary, ensuring we don't return the User Prompt."""
        try:
            if not messages: return "No analysis generated."

            # 1. Filter out the initial USER prompt (usually the first message)
            # We only want messages from the Assistant agents
            # FIX: Use substring matching to handle AutoGen's appended UUIDs
            target_agents = ['financial_reviewer', 'financial_analyst', 'financial_researcher']
            agent_messages = []
            
            for msg in messages:
                source = getattr(msg, 'source', '') or msg.get('source', '')
                if any(agent_name in source for agent_name in target_agents):
                    agent_messages.append(msg)

            if not agent_messages:
                return "[WARN] Financial Analysis Failed: Agents did not generate a response. Check logs for tool errors."

            # 2. Prioritize Reviewer -> Analyst -> Researcher
            target_order = ['financial_reviewer', 'financial_analyst', 'financial_researcher']
            
            for source in target_order:
                for msg in reversed(agent_messages):
                    msg_source = getattr(msg, 'source', '') or msg.get('source', '')
                    msg_content = getattr(msg, 'content', '') or msg.get('content', '')
                    
                    if isinstance(msg_content, list):
                        msg_content = " ".join([str(c) for c in msg_content])

                    if source in msg_source and isinstance(msg_content, str):
                        # FIX: Ignore validation tool outputs AND tool calls
                        if "[VALIDATION]" in msg_content:
                            continue
                        if "FunctionCall" in msg_content and "arguments=" in msg_content:
                            continue

                        clean = msg_content.replace("TERMINATE", "").strip()
                        if len(clean) > 50:
                            return clean
            
            # Fallback
            return str(agent_messages[-1].content).replace("TERMINATE", "").strip()

        except Exception as e:
            self.logger.warning(f"Summary extraction error: {e}")
            return "Analysis completed (Summary extraction failed)"

    def _create_error_result(self, message: str) -> Dict[str, Any]:
        return {
            "status": "error",
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def close(self):
        """Cleanup resources."""
        if hasattr(self.model_client, 'close'):
            # Some clients might not need await or might not have close
            # Wrapped in try/except just in case
            try:
                await self.model_client.close()
            except: pass

# --- Factory Function ---
async def create_financial_team(model: str = "gpt-4-turbo", api_key: str = None) -> FinancialAgentTeam:
    """
    Factory to spin up the team.
    """
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required.")

    # Initialize RAG
    # Note: Using absolute path relative to execution
    base_path = Path(os.getcwd()) / "data" / "vector_stores" / "financial_data"
    rag = ProductionRAGSystem(persist_directory=str(base_path), embedding_type="huggingface")

    # Initialize Model Client
    client = OpenAIChatCompletionClient(
        model=model,
        api_key=api_key,
        temperature=0.1
    )

    return FinancialAgentTeam(client, rag)

# --- Main Entry Point ---
if __name__ == "__main__":
    async def main():
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        
        print("Initializing Financial Agent Team...")
        try:
            team = await create_financial_team()
            
            ticker = "AAPL"
            print(f"Analyzing {ticker}...")
            
            result = await team.analyze_company(ticker, "Focus on recent growth trends.")
            
            if result['status'] == 'success':
                print(f"\n--- ANALYSIS FOR {ticker} ---\n")
                print(result['analysis'])
                print("\n-----------------------------")
            else:
                print(f"Error: {result['message']}")
                
            await team.close()
            
        except Exception as e:
            print(f"Fatal Error: {e}")

    asyncio.run(main())