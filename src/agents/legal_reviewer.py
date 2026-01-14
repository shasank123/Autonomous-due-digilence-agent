# src/agents/legal_reviewer.py
import os
import sys
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
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
from src.tools.legal_tools import LegalTools

load_dotenv()

class LegalAgentTeam:
    """
    Legal Due Diligence Agent Team.
    Orchestrates specialized agents for Compliance, Risk, and Contract analysis.
    """
    def __init__(self, model_client: OpenAIChatCompletionClient, rag_system: ProductionRAGSystem):
        self.model_client = model_client
        self.rag_system = rag_system
        self.logger = logging.getLogger(__name__)
        
        # Initialize Tools
        self.sec_collector = SECDataCollector()
        # Note: DocumentProcessor is initialized but mostly used by the ingestion pipeline
        self.document_parser = DocumentProcessor() 
        self.tools = LegalTools(self.rag_system, self.sec_collector)
        
        # Initialize Agents
        self._create_agents()
        self._create_team()

    def _create_agents(self):
        """Create legal due diligence agents with specialized roles"""

        # 1. Legal Compliance Analyst
        self.compliance_analyst = AssistantAgent(
            name="compliance_analyst",
            model_client=self.model_client,
            model_context=BufferedChatCompletionContext(buffer_size=15),
            tools=[
                self.tools.retrieve_legal_filings, 
                self.tools.check_regulatory_compliance
            ],
            system_message="""You are a Legal Compliance Specialist. Your responsibilities:
            1. **Regulatory Filings**: Use retrieve_legal_filings() to access SEC legal documents.
            2. **Compliance Checks**: Use check_regulatory_compliance() to verify legal adherence.
            3. **Regulation Analysis**: Analyze SEC rules and disclosure requirements.
            
            Focus on SEC filing compliance (10-K, 10-Q, 8-K) and specific regulatory violations.
            Use 'COMPLIANCE_REVIEW_COMPLETE' when compliance analysis is finished."""
        )
        
        # 2. Legal Risk Assessor
        self.risk_assessor = AssistantAgent(
            name="risk_assessor",
            model_client=self.model_client,
            model_context=BufferedChatCompletionContext(buffer_size=12),
            tools=[
                self.tools.analyze_litigation_history, 
                self.tools.assess_legal_risks
            ],
            system_message="""You are a Legal Risk Assessment Specialist. Your responsibilities:
            1. **Litigation Analysis**: Use analyze_litigation_history() to examine legal disputes.
            2. **Risk Assessment**: Use assess_legal_risks() to evaluate material legal exposures.
            3. **Contingency Analysis**: Assess financial impact of legal contingencies.
            
            Focus on pending litigation, regulatory enforcement, and IP disputes.
            Use 'RISK_ASSESSMENT_COMPLETE' when risk analysis is finished."""
        )
        
        # 3. Contract Reviewer (The Closer)
        self.contract_reviewer = AssistantAgent(
            name="contract_reviewer",
            model_client=self.model_client,
            model_context=BufferedChatCompletionContext(buffer_size=10),
            tools=[
                self.tools.review_material_contracts, 
                self.tools.validate_legal_findings
            ],
            system_message="""You are a Senior Contract Review Specialist. Your responsibilities:
            1. **Contract Analysis**: Use review_material_contracts() to examine key agreements.
            2. **Findings Validation**: Use validate_legal_findings() to cross-check legal conclusions.
            3. **Recommendation**: Integrate findings into a final Legal Due Diligence Report.
            
            Review the work of the Compliance Analyst and Risk Assessor.
            Ensure all claims are supported by source documents.
            Use 'LEGAL_REVIEW_COMPLETE' when the final report is ready."""
        )

    def _create_team(self):
        """Create the legal analysis team with robust termination conditions"""
        termination_conditions = (
            TextMentionTermination("LEGAL_REVIEW_COMPLETE") |
            MaxMessageTermination(max_messages=25)
        )

        self.team = RoundRobinGroupChat(
            [self.compliance_analyst, self.risk_assessor, self.contract_reviewer],
            termination_condition=termination_conditions,
            max_turns=20
        )
        
    async def analyze_company_legal(self, company_ticker: str, additional_context: str = "") -> Dict[str, Any]:
        """Run comprehensive legal due diligence with production-grade error handling"""
        try:
            # Input validation
            if not company_ticker or not company_ticker.strip():
                return self._create_error_result("Company ticker is required")
            
            company_ticker = company_ticker.upper().strip()

            # Ensure data availability
            data_available = await self._ensure_legal_data(company_ticker)
            if not data_available:
                return self._create_error_result(f"Insufficient legal data available for {company_ticker}")
            
            self.logger.info(f"Starting legal analysis for {company_ticker}")

            # Build task
            task = self._build_legal_analysis_task(company_ticker, additional_context)

            # Execute with timeout
            try:
                # Reset team state before running (fixes "team already running" error)
                await self.team.reset()
                result = await asyncio.wait_for(
                    self.team.run(task=task),
                    timeout=300 # 5-minute timeout
                )
            except asyncio.TimeoutError:
                self.logger.error(f"Legal analysis timeout for {company_ticker}")
                return self._create_error_result("Legal analysis timeout - process took too long")
            
            # Process results
            analysis_result = self._process_legal_result(result, company_ticker)
            self.logger.info(f"Successfully completed legal analysis for {company_ticker}")
            return analysis_result
        
        except Exception as e:
            self.logger.error(f"Legal analysis failed for {company_ticker}: {e}")
            return self._create_error_result(f"Legal analysis failed: {str(e)}")
        
    def _build_legal_analysis_task(self, company: str, additional_context: str) -> str:
        """Constructs the prompt for the legal team."""
        return f"""
        Perform comprehensive legal due diligence for {company}.

        **Context:** {additional_context or "Standard legal risk assessment"}

        **Process:**
        1. COMPLIANCE: Check SEC filings (10-K, 8-K) and regulatory status.
        2. RISK: Analyze litigation history and 'Item 1A' risk factors.
        3. CONTRACTS: Review material agreements.
        4. SYNTHESIS: Validate findings and provide a Legal Due Diligence Report.

        Coordinate as a team. Do not fabricate legal data.
        """
    
    async def _ensure_legal_data(self, company: str) -> bool:
        """
        Verifies that sufficient data exists in the RAG system for legal analysis.
        Note: We check for any company data since financial metrics can inform
        legal analysis (liabilities, debt, contingencies).
        """
        try:
            if not company: return False
            
            # Broader queries that work with financial data
            # Financial data can inform legal analysis (liabilities, debt, compliance)
            queries = [
                f"{company} liabilities",
                f"{company} assets",
                f"{company} debt",
                f"{company} revenue",
                f"{company} legal",
                f"{company} risk",
            ]

            found_count = 0
            for q in queries:
                # Lightweight query check
                docs = self.rag_system.query(q, company=company, k=1)
                if docs: 
                    found_count += 1
                    # If we found any data, that's enough to proceed
                    if found_count >= 1:
                        self.logger.info(f"Data verified for legal analysis of {company}")
                        return True

            self.logger.warning(f"No data found for {company}")
            return False
        
        except Exception as e:
            self.logger.error(f"Legal data assurance check failed: {e}")
            return False
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        return {
            'company': 'UNKNOWN',
            'error': error_message,
            'success': False,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _process_legal_result(self, result: Any, company: str) -> Dict[str, Any]:
        """Extracts the final summary from the team interaction."""
        try:
            # Handle AutoGen result object structure
            messages = result.messages if hasattr(result, 'messages') else []
            summary = self._extract_meaningful_summary(messages)

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
        
    def _extract_meaningful_summary(self, messages: List[Any]) -> str:
        """
        Iterates backwards through messages to find the final conclusion
        from the Contract Reviewer or Risk Assessor.
        """
        try:
            if not messages: return "No analysis generated."

            target_order = ['contract_reviewer', 'risk_assessor', 'compliance_analyst']
            
            for source in target_order:
                for msg in reversed(messages):
                    # Defensive attribute access
                    msg_source = getattr(msg, 'source', '') or (msg.get('source') if isinstance(msg, dict) else '')
                    msg_content = getattr(msg, 'content', '') or (msg.get('content') if isinstance(msg, dict) else '')

                    # FIX: Use substring matching to handle AutoGen's appended UUIDs
                    if source in msg_source and msg_content:
                        content = str(msg_content).strip()
                        
                        # Skip validation messages and tool outputs
                        skip_patterns = [
                            "[SUCCESS]", "[VALIDATION]", "[REPORT]", 
                            "FunctionCall", "arguments=", "Findings corroborated"
                        ]
                        if any(pattern in content for pattern in skip_patterns):
                            continue
                        
                        # Heuristic: A good summary is usually substantial
                        if len(content) > 50:
                            return content
            
            return "Analysis completed (Summary extraction failed)"
        except Exception as e:
            self.logger.warning(f"Summary extraction error: {e}")
            return "Analysis completed (Error extracting summary)"
        
    async def close(self):
        """Cleanup resources."""
        try:
            if hasattr(self.model_client, 'close'):
                await self.model_client.close()
        except: pass

# --- Factory Function ---
async def create_legal_team(
        model: str = "gpt-4-turbo", 
        api_key: str = None,
        rag_system: ProductionRAGSystem = None
) -> LegalAgentTeam:
    """Factory to create the Legal Team."""
    
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key: raise ValueError("OPENAI_API_KEY required.")

    # Initialize RAG if not provided
    if not rag_system:
        base_path = Path(os.getcwd()) / "data" / "vector_stores" / "legal_data"
        rag_system = ProductionRAGSystem(persist_directory=str(base_path), embedding_type="huggingface")

    client = OpenAIChatCompletionClient(model=model, api_key=api_key, temperature=0.1)
    
    return LegalAgentTeam(client, rag_system)

# --- Main Entry Point ---
if __name__ == "__main__":
    async def main():
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        
        print("Initializing Legal Agent Team...")
        legal_team = None
        try:
            legal_team = await create_legal_team()
            
            ticker = "AAPL"
            print(f"Analyzing Legal Risks for {ticker}...")
            
            # Note: This assumes data is already in RAG. 
            # In a full flow, the Financial Agent usually ingests the data first.
            result = await legal_team.analyze_company_legal(
                ticker, 
                "Focus on recent antitrust litigation."
            )
            
            if result['success']:
                print(f"\n--- LEGAL REPORT FOR {ticker} ---\n")
                print(result['summary'])
                print("\n---------------------------------")
            else:
                print(f"Error: {result['error']}")
                
        except Exception as e:
            print(f"Fatal Error: {e}")
        finally:
            if legal_team: await legal_team.close()

    asyncio.run(main())