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

class LegalAgentTeam:
    """Legal Due Diligence Agent Team integrated with our RAG system"""
    def __init__(self, model_client: OpenAIChatCompletionClient, rag_system: ProductionRAGSystem):
        self.model_client = model_client
        self.rag_system = rag_system
        self.sec_collector = SECDataCollector()
        self.document_parser = DocumentProcessor()
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
            tools=[self.retrieve_legal_filings, self.check_regulatory_compliance],
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
            tools = [self.analyze_litigation_history, self.assess_legal_risks],
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
            tools=[self.review_material_contracts, self.validate_legal_findings],
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

    # RAG-Integrated Legal Tool Functions
    async def retrieve_legal_filings(self, company: str, filing_types: List[str] = None) -> str:
        """Retrieve and validate legal filings with robust error handling"""
        try:
            # Input validation
            if not company or not company.strip():
                return "❌ Company ticker is required"
            
            company = company.upper().strip()

            if not filing_types:
                filing_types = ["10-K", "10-Q", "8-K", "S-1", "DEF 14A"]

            valid_filings = [ft for ft in filing_types if ft and ft.strip()]
            if not valid_filings:
                return "❌ No valid filing types provided"
            
            self.logger.info(f"Retrieving legal filings for {company}: {valid_filings}")

            results = []
            filings_found = 0
            filings_failed = 0

            for filing_type in valid_filings:
                try:
                    # Use similarity search for legal documents
                    scored_documents = self.rag_system.query_with_similarity_scores(
                        question=f"{company}{filing_type} legal filing regulatory document",
                        company=company,
                        metric_type="legal_filing",
                        k=5,
                        score_threshold=0.5
                    )

                    if not scored_documents:
                        self.logger.warning(f"No legal filings found for{company} {filing_type}")
                        results.append(f"❌ {filing_type}: No legal filings available")
                        filings_failed+=1
                        continue

                    # Extract and validate filing data
                    filing_results = self._extract_legal_filing_data(scored_documents, filing_type, company)

                    if filing_results:
                        results.append(f"📄 {filing_type}:\n{filing_results}")
                        filings_found += 1
                    else:
                        results.append(f"⚠️ {filing_type}: Data available but parsing failed")
                        filings_failed += 1

                except Exception as e:
                    self.logger.error(f"Legal filing retrieval failed for {filing_type}: {e}")
                    results.append(f"⚠️ {filing_type}: Processing error")
                    continue

            # Build final response with summary
            summary = self._build_legal_filings_summary(company, filings_found, filings_failed, len(valid_filings))

            if results:
                return f"{summary}\n\n" + "\n\n".join(results)
            else:
                return f"{summary}\n\n ❌ No legal filings could be retrieved for {company}"
            
        except Exception as e:
            self.logger.error(f"Legal filings retrieval failed for {company}: {e}")
            return f"❌ System error retrieving legal filings: {str(e)}"
        
    def _extract_legal_filing_data(self, scored_documents: List[Tuple], filing_type: str, company: str) -> str:
        """Extract and validate legal filing data from documents"""
        try:
            filing_data = []
            for doc, score in scored_documents:
                try:
                    # Skip if document doesn't contain our target filing type
                    if filing_type not in doc.page_content:
                        continue

                    # Parse structured legal document
                    parsed_data = self._parse_legal_document(doc.page_content, filing_type)
                    if not parsed_data:
                        continue

                    # Extract key legal information
                    filing_date = parsed_data.get('filing_date', 'unknown')
                    document_date = parsed_data.get('document_date', 'Unknown')
                    sections = parsed_data.get('sections', [])
                    risk_factors = parsed_data.get('risk_factors', [])

                    # Format filing information
                    filing_info = f"• Filed: {filing_date} | Document Date:{document_date} | Confidence: {score:.2f}"

                    if sections:
                        filing_info += f"\n Key Sections: {','.join(sections[:3])}"
                    
                    if risk_factors:
                        filing_info += f"\n Risk Factors: {len(risk_factors)} identified"

                    filing_data.append(filing_info)

                except Exception as doc_error:
                    self.logger.warning(f"Legal document processing failed: {doc_error}")
                    continue

            if not filing_data:
                return None

            return "\n".join(filing_data[:3])
        
        except Exception as e:
            self.logger.error(f"Legal filing data extraction failed: {e}")
            return None

    def _parse_legal_document(self, content: str, filing_type: str) -> Dict[str, Any]:
        """Parse structured legal document with validation"""
        try:
            lines = content.split()
            data = {}

            for line in lines:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()

                if key == 'Filing Type':
                    data['filing_type'] = value
                elif key == 'Filing Date':
                    data['filing_date'] = value
                elif key == 'Document Date':
                    data['document_date'] = value
                elif key == 'Sections':
                    data['sections'] = [s.strip() for s in value.split(',')]
                elif key == 'Risk Factors':
                    data['risk_factors'] = [rf.strip() for rf in value.strip(';')]

            # Validate we have the target filing type and essential data
            if(data.get('filing_type') == filing_type
               and data.get('filing_date')):
                return data

            return None

        except Exception as e:
            self.logger.warning(f"Legal document parsing failed: {e}")
            return None
    
    

        

                
                    
        
                    
                        
                        





                
        
    





