# src/agents/legal_reviewer.py
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
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from rag.core import ProductionRAGSystem
from data.processors.document_parser import DocumentProcessor
from data.collectors.sec_edgar import SECDataCollector
from tools.legal_tools import LegalTools

class LegalAgentTeam:
    """Legal Due Diligence Agent Team integrated with our RAG system"""
    def __init__(self, model_client: OpenAIChatCompletionClient, rag_system: ProductionRAGSystem):
        self.model_client = model_client
        self.rag_system = rag_system
        self.sec_collector = SECDataCollector()
        self.document_parser = DocumentProcessor()
        self.tools = LegalTools(self.rag_system, self.sec_collector)
        self.team = None
        self.logger = self._setup_logging()
        self._create_agents()
        self._create_team()

    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for production"""
        logger = logging.getLogger("LegalAgentTeam")
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
        """Create legal due diligence agents with specialized roles"""

        # Legal Compliance Analyst - Regulatory compliance and filings
        self.compliance_analyst = AssistantAgent(
            name = "compliance_analyst",
            model_client=self.model_client,
            model_context=BufferedChatCompletionContext(buffer_size=15),
            tools=[self.tools.retrieve_legal_filings, self.tools.check_regulatory_compliance],
            system_message="""You are a Legal Compliance Specialist. Your responsibilities:

            1. **Regulatory Filings**: Use retrieve_legal_filings() to access SEC legal documents
            2. **Compliance Checks**: Use check_regulatory_compliance() to verify legal adherence
            3. **Regulation Analysis**: Analyze SEC rules, disclosure requirements, and compliance standards
            4. **Document Validation**: Ensure legal documents are current and properly filed

            **Key Focus Areas:**
            - SEC filing compliance (10-K, 10-Q, 8-K)
            - Regulatory disclosure requirements
            - Legal entity structure and governance
            - Compliance history and violations

            **Tools Available:**
            - retrieve_legal_filings(company, filing_types): Get specific legal documents
            - check_regulatory_compliance(company, regulations): Verify compliance status

            Use 'COMPLIANCE_REVIEW_COMPLETE' when compliance analysis is finished.
            """
         )
        
        # Legal Risk Assessor - Litigation and risk analysis
        self.risk_assessor = AssistantAgent(
            name = "risk_assessor",
            model_client= self.model_client,
            model_context=BufferedChatCompletionContext(buffer_size=12),
            tools = [self.tools.analyze_litigation_history, self.tools.assess_legal_risks],
            system_message="""You are a Legal Risk Assessment Specialist. Your responsibilities:

            1. **Litigation Analysis**: Use analyze_litigation_history() to examine legal disputes
            2. **Risk Assessment**: Use assess_legal_risks() to evaluate potential legal exposures
            3. **Contingency Analysis**: Assess legal contingencies and their financial impact
            4. **Risk Quantification**: Evaluate likelihood and impact of legal risks

            **Key Analysis Areas:**
            - Pending litigation and legal disputes
            - Regulatory investigations and enforcement actions
            - Contract disputes and arbitration cases
            - Intellectual property risks and claims

            **Tools Available:**
            - analyze_litigation_history(company): Get litigation and dispute information
            - assess_legal_risks(company): Evaluate overall legal risk exposure

            Use 'RISK_ASSESSMENT_COMPLETE' when risk analysis is finished.
            """
         )
        
        # Contract Reviewer - Material contracts and obligations
        self.contract_reviewer = AssistantAgent(
            name="contract_reviewer",
            model_client=self.model_client,
            model_context=BufferedChatCompletionContext(buffer_size=10),
            tools=[self.tools.review_material_contracts, self.tools.validate_legal_findings],
            system_message="""You are a Senior Contract Review Specialist. Your responsibilities:

            1. **Contract Analysis**: Use review_material_contracts() to examine key agreements
            2. **Findings Validation**: Use validate_legal_findings() to cross-check legal conclusions
            3. **Obligation Synthesis**: Integrate compliance and risk findings into legal assessment
            4. **Legal Recommendations**: Provide data-driven legal due diligence recommendations

            **Review Checklist:**
            - Material contract obligations and commitments
            - Legal finding consistency with source documents
            - Comprehensive legal risk assessment
            - Clear legal due diligence conclusions

            **Tools Available:**
            - review_material_contracts(company): Analyze material contracts and agreements
            - validate_legal_findings(company, findings): Cross-check with original legal documents

            Use 'LEGAL_REVIEW_COMPLETE' when legal due diligence is finished.
            """
         )

    def _create_team(self):
        """Create the legal analysis team with robust termination conditions"""
        termination_conditions = (
            TextMentionTermination("LEGAL_REVIEW_COMPLETE") |
            TextMentionTermination("COMPLIANCE_REVIEW_COMPLETE") |
            TextMentionTermination("RISK_ASSESSMENT_COMPLETE") |
            TextMentionTermination("TERMINATE") |
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

            # Build comprehensive task with context
            task = self._build_legal_analysis_task(company_ticker, additional_context)

            # Execute analysis with timeout protection
            try:
                result = await asyncio.wait_for(
                    self.team.run(task=task),
                    timeout=300 # 5-minute timeout
                )
            
            except asyncio.TimeoutError:
                self.logger.error(f"Legal analysis timeout for {company_ticker}")
                return self._create_error_result("Legal analysis timeout - process took too long")
            
            # Process and validate results
            analysis_result = self._process_legal_result(result, company_ticker)
            self.logger.info(f"Successfully completed legal analysis for {company_ticker}")
            return analysis_result
        
        except Exception as e:
            self.logger.error(f"Legal analysis failed for {company_ticker}: {e}")
            return self._create_error_result(f"Legal analysis failed: {str(e)}")
        
    def _build_legal_analysis_task(self, company: str, additional_context: str) -> str:
        """Build comprehensive legal analysis task with context"""

        base_task = f"""
        Perform comprehensive legal due diligence for {company} using our SEC legal data pipeline.

        **Legal Data Sources Available:**
        - SEC legal filings (10-K, 10-Q, 8-K, DEF 14A)
        - Regulatory compliance documents
        - Litigation history and legal disputes
        - Material contracts and agreements
        - Risk factor disclosures

        **Legal Analysis Framework:**
        1. COMPLIANCE: Review regulatory filings and compliance status
        2. RISK: Analyze litigation history and legal risk exposure  
        3. CONTRACTS: Review material contracts and legal obligations
        4. SYNTHESIS: Validate findings and provide legal recommendations

        **Legal Quality Requirements:**
        - Cross-validate all legal findings with source documents
        - Assess both current compliance and historical patterns
        - Evaluate materiality of legal risks and obligations
        - Provide specific, actionable legal recommendations

        **Additional Context:** {additional_context or "Standard comprehensive legal due diligence"}

        Coordinate as a legal team and ensure thorough documentation of the legal analysis process.
        """
        return base_task.strip()
    
    # Helper methods for production legal operations
    async def _ensure_legal_data(self, company: str) -> bool:
        """Ensure legal data is available with production-grade error handling"""
        try:
            if not company or not company.strip():
                self.logger.error(f"Company ticker required for legal data assurance")
                return False
            
            company = company.upper().strip()

            # Check existing legal data with quality assessment
            legal_queries = [
                f"{company} legal filings",
                f"{company} litigation",
                f"{company} contracts",
                f"{company} compliance"
            ]

            legal_docs_found = 0
            for query in legal_queries:
                docs = self.rag_system.query(query,company=company,k=2)
                if docs:
                    legal_docs_found += len(docs)

            if legal_docs_found >= 4:
                self.logger.info(f"Legal data verified for {company}: {legal_docs_found} documents")
                return True
            
            self.logger.warning(f"Insufficient legal data for {company}: {legal_docs_found} documents")
            return False
        
        except Exception as e:
            self.logger.error(f"Legal data assurance failed for {company}: {e}")
            return False
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result for legal operations"""
        
        return {
            'company': 'UNKNOWN',
            'error': error_message,
            'legal_integration': False,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'success': False
        }
    
    def _process_legal_result(self, result, company: str) -> Dict[str, Any]:
        """Process and validate legal analysis results"""
        try:
            summary = self._extract_meaningful_summary(result.messages)

            return {
                'company': company,
                'messages': [msg.to_dict() for msg in result.messages],
                'summary': summary,
                'legal_integration': True,
                'data_source': 'SEC Legal Documents + Production RAG',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message_count': len(result.messages),
                'success': True
            }
        
        except Exception as e:
            self.logger.error(f"Legal result processing failed for {company}: {e}")
            return self._create_error_result(f"Legal result processing failed: {str(e)}")
        
    def _extract_meaningful_summary(self, messages) -> str:
        """Extract meaningful legal summary from team messages"""
        try:
            # Prioritize contract reviewer messages, then risk assessor, then compliance analyst
            for source in ['contract_reviewer', 'risk_assessor', 'compliance_analyst']:
                for msg in reversed(messages):
                    if hasattr(msg, 'source') and msg.source == source and hasattr(msg, 'content'):
                        content = msg.content.strip()
                        if len(content) > 50 and any(keyword in content.lower() for keyword in ['recommend', 'conclusion', 'summary', 'assessment', 'legal', 'compliance']):
                            return f"{source}: {content[:500]}..."

            # Fallback: find most substantial legal message
            substantial_messages = []
            for msg in messages:
                if hasattr(msg, 'content') and len(msg.content.strip()) > 100:
                    substantial_messages.append((msg.source, msg.content))  

            if substantial_messages:
                source, content = substantial_messages[-1]
                return f"{source}: {content[:400]}..." 
            
            return "Legal analysis completed - review detailed messages for complete assessment"
        
        except Exception as e:
            self.logger.warning(f"Legal summary extraction failed: {e}")
            return "Legal analysis completed - summary extraction unavailable"
        
    async def close(self):
        """Clean up legal resources with proper error handling"""
        try:
            if hasattr(self, 'model_client'):
                await self.model_client.close()
            self.logger.info("Legal agent team resources cleaned up")

        except Exception as e:
            self.logger.error(f"Error during legal cleanup: {e}")

# Production-grade factory function for Legal Agent Team
async def create_legal_team(
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        rag_system: Optional[ProductionRAGSystem] = None,
        timeout: int = 30
) -> LegalAgentTeam:
    
    """Create production-grade legal agent team with comprehensive setup"""

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
                persist_directory="./data/vector_stores/legal_data",
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

        return LegalAgentTeam(model_client, rag_system)
    
    except Exception as e:
        logging.error(f"Failed to create legal agent team: {e}")
        raise

# Production-ready main execution
async def main():
    """Production-grade example usage"""

    # Configure logging for production
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    legal_team = None
    try:
        # Create legal agent team
        legal_team = await create_legal_team()  

        # Test with a company
        company = "AAPL"

        # Ensure data availability
        data_ready = await legal_team._ensure_legal_data(company)
        if not data_ready:
            print(f"❌ Cannot analyze {company} - data unavailable")
            return

        # Run comprehensive analysis
        result = await legal_team.analyze_company_legal(
            company_ticker=company,
            additional_context="Focus on regulatory risks and pending litigation"
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
        if legal_team:
            await legal_team.close()      

if __name__ == "__main__":
    asyncio.run(main())
