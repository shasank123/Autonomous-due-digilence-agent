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
    
    async def check_regulatory_compliance(self, company: str, regulations: List[str] = None) -> str:
        """Check regulatory compliance status with validation"""
        try:
            if not company or not company.strip():
                return "❌ Company ticker is required"
            
            company = company.upper().strip()

            if not regulations:
                regulations = ["SOX", "Dodd-Frank", "SEC Disclosure", "GAAP Compliance"]

            self.logger.info(f"Checking regulatory compliance for {company}: {regulations}")

            compliance_results = []
            for regulation in regulations:
                try:

                    scored_documents = self.rag_system.query_with_similarity_scores(
                        question=f"{company}{regulation} compliance regulatory requirement",
                        company=company,
                        metric_type="compliance_doc"
                        k=3,
                        score_threshold=0.6
                    )

                    compliance_status = self._assess_compliance_status(scored_documents, regulation, company)
                    compliance_results.append(f"📋 {regulation}: {compliance_status}")

                except Exception as e:
                    self.logger.error(f"Compliance check failed for {regulation}: {e}")
                    compliance_results.append(f"⚠️{regulation}: Compliance check failed")

            return f"Regulatory Compliance Assessment for {company}\n\n" + "\n\n".join(compliance_results)
        
        except Exception as e:
            self.logger.error(f"Regulatory compliance check failed for {company}: {e}")
            return f"❌ System error checking regulatory compliance:{str(e)}"
        
    def _assess_compliance_status(self, scored_documents: List[Tuple], regulation: str, company: str) -> str:
        """Assess compliance status based on document analysis"""

        if not scored_documents:
            return f"❌ No compliance data available"
        
        # Analyze document content for compliance indicators
        compliance_indicators = 0
        violation_indicators =0

        for doc, score in scored_documents:
            content_lower = doc.page_content.lower()
            regulation_lower = regulation.lower()

            # Positive compliance indicators
            if any(term in content_lower for term in ['compliant', 'in compliance', 'meets requirements', 'satisfies']):
                compliance_indicators += 1
            
            # Negative compliance indicators
            if any(term in content_lower for term in ['violation', 'deficiency', 'non-compliant', 'investigation', 'enforcement']):
                violation_indicators += 1

        # Determine compliance status
        if compliance_indicators > violation_indicators:
            return "✅ Likely Compliant"
        elif violation_indicators > compliance_indicators:
            return "❌ Potential Compliance Issues"
        else:
            return "⚠️ Insufficient Information"
        
    async def analyze_litigation_history(self, company: str) -> str:
        """Analyze litigation history and legal disputes"""
        
        try:
            if not company or not company.strip():
                return "❌ Company ticker is required"
            
            company = company.upper().strip()

            self.logger.info(f"Analyzing litigation history for {company}")

            # Query for litigation-related documents
            scored_documents = self.rag_system.query_with_similarity_scores(
                question=f"{company} litigation legal disputes lawsuits claims",
                company=company,
                metric_type="legal_risk",
                k=10,
                score_threshold=0.4
            )
            
            if not scored_documents:
                return f"📊 No litigation history found for {company}"
            
            litigation_analysis = self._categorize_litigation(scored_documents, company)

            return self._format_litigation_report(litigation_analysis, company)
        
        except Exception as e:
            self.logger.error(f"Litigation analysis failed for {company}: {e}")
            return f"❌ System error analyzing litigation history: {str(e)}"
        
    def _categorize_litigation(self, scored_documents: List[Tuple], company: str) -> Dict[str, Any]:
        """Categorize litigation by type and severity"""

        litigation_categories = {
            'securities_litigation': [],
            'contract_disputes': [],
            'intellectual_property': [],
            'employment_law': [],
            'regulatory_enforcement': []
        }

        for doc, score in scored_documents:
            try:
                content_lower = doc.page_content.lower()
                doc_info = {
                    'content_preview' : content_lower[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    'score' : score,
                    'doc_type' : doc.metadata.get('doc_type', 'unknown')
                }

                # Categorize based on content
                if any(term in content_lower for term in ['securities', 'stock', 'shareholder', 'class action']):
                    litigation_categories['securities_litigation'].append(doc_info)
                
                elif any(term in content_lower for term in ['contract', 'agreement', 'breach', 'dispute']):
                    litigation_categories['contract_disputes'].append(doc_info)

                elif any(term in content_lower for term in ['patent', 'trademark', 'copyright', 'intellectual property']):
                    litigation_categories['intellectual_property'].append(doc_info)

                elif any(term in content_lower for term in ['employment', 'labor', 'discrimination', 'wrongful termination']):
                    litigation_categories['employment_law'].append(doc_info)

                elif any(term in content_lower for term in ['regulatory', 'enforcement', 'investigation', 'sec', 'doj']):
                    litigation_categories['regulatory_enforcement'].append(doc_info)

            except Exception as doc_error:
                self.logger.warning(f"Litigation categorization failed: {doc_error}")
                continue

        return litigation_categories
    
    def _format_litigation_report(self, litigation_analysis: Dict[str, Any], company: str) -> str:
        """Format comprehensive litigation report"""

        report_parts = [f"⚖️ Litigation History Analysis for {company}"]

        for category, docs in litigation_analysis.items():
            if docs:
                category_name = category.replace('_', '').title()
                report_parts.append(f"\n 🔹 {category_name} ({len(docs)} documents):")

                for doc in docs[:2]:
                    report_parts.append(f" • {doc['content_preview']} [confidence: {doc['score']:.2f}]")

        if not any(docs for docs in litigation_analysis.values()):
            report_parts.append(f"\n ✅ No significant litigation history identified")

        report_parts.append("n🎯 Legal Risk Assessment:")
        total_docs = sum(len(docs) for docs in litigation_analysis.values())

        if total_docs > 10:
            report_parts.append(" ⚠️ High legal risk exposure - multiple litigation matters")
        elif total_docs > 5:
            report_parts.append(" 📊 Moderate legal risk exposure - review recommended")
        else:
            report_parts.append(" ✅ Low legal risk exposure - standard monitoring sufficient")

        return "\n".join(report_parts)
    
    async def assess_legal_risks(self, company: str) -> str:
        """Comprehensive legal risk assessment with quantitative scoring"""
        try:
            if not company or not company.strip():
                return "❌ Company ticker is required for legal risk assessment"
            
            company = company.upper().strip()
            self.logger.info(f"Assessing legal risks for {company}")

            # Multi-faceted risk assessment
            risk_queries = [
                f"{company} litigation legal disputes lawsuits",
                f"{company} regulatory investigations enforcement",
                f"{company} contract disputes breaches",
                f"{company} intellectual property risks claims",
                f"{company} compliance violations penalties"
            ]

            all_risk_documents = []
            for query in risk_queries:
                try:

                    scored_docs = self.rag_system.query_with_similarity_scores(
                        question=query,
                        company=company,
                        metric_type="legal_risk",
                        k=5,
                        score_threshold=0.4
                    )
                    all_risk_documents.append(scored_docs)

                except Exception as risk_error:
                    self.logger.error(f"Risk query failed:{risk_error}")
                    continue
            
            if not all_risk_documents:
                return f"📊 No legal risk documents found for {company}"
            
            # Quantitative risk scoring
            risk_assessment = self._quantify_legal_risks(all_risk_documents, company)
            return self._format_risk_assessment_report(risk_assessment, company)
        
        except Exception as e:
            self.logger.error(f"Legal risk assessment failed for {company}: {e}")
            return f"❌ System error in legal risk assessment: {str(e)}"
        
    def _quantify_legal_risks(self, risk_documents: List[Tuple], company: str) -> Dict[str, Any]:
        """Quantify legal risks with severity scoring"""

        risk_categories = {
            'litigation_risk': {'score': 0, 'documents': [], 'indicators': 0},
            'regulatory_risk': {'score': 0, 'documents': [], 'indicators': 0},
            'contract_risk': {'score': 0, 'documents': [], 'indicators': 0},
            'compliance_risk': {'score': 0, 'documents': [], 'indicators': 0},
            'ip_risk': {'score': 0, 'documents': [], 'indicators': 0}
        }

        for doc, score in risk_documents:
            try:
                content_lower = doc.page_content.lower()
                doc_info = {
                    'content': doc.page_content[:150] if len(doc.page_content) > 150 else doc.page_content,
                    'score': score,
                    'doc_type': doc.metadata.get('doc_type', 'unknown')
                }

                # Litigation risk indicators
                litigation_terms = ['lawsuit', 'litigation', 'class action', 'dispute', 'claim', 'settlement']

                if any(term in content_lower for term in litigation_terms):
                    risk_categories['litigation_risk']['indicators'] += 1
                    risk_categories['litigation_risk']['score'] = score
                    risk_categories['litigation_risk']['documents'].append(doc_info)

                # Regulatory risk indicators
                regulatory_terms = ['investigation', 'enforcement', 'sec', 'doj', 'regulatory', 'subpoena']

                if any(term in content_lower for term in regulatory_terms):
                    risk_categories['regulatory_risk']['indicators'] += 1
                    risk_categories['regulatory_risk']['score'] = score
                    risk_categories['regulatory_risk']['documents'].append(doc_info)

                # Contract risk indicators
                contract_terms = ['contract', 'agreement', 'breach', 'termination', 'obligation']

                if any(term in content_lower for term in contract_terms):
                    risk_categories['contract_risk']['indicators'] += 1
                    risk_categories['contract_risk']['score'] += score
                    risk_categories['contract_risk']['documents'].append(doc_info)

                # Compliance risk indicators
                compliance_terms = ['violation', 'non-compliant', 'deficiency', 'penalty', 'fine']

                if any(term in content_lower for term in compliance_terms):
                    risk_categories['compliance_risk']['indicators'] += 1
                    risk_categories['compliance_risk']['score'] += score
                    risk_categories['compliance_risk']['documents'].append(doc_info)

                # IP risk indicators
                ip_terms = ['patent', 'trademark', 'copyright', 'intellectual property', 'infringement']

                if any(term in content_lower for term in ip_terms):
                    risk_categories['ip_risk']['indicators'] += 1
                    risk_categories['ip_risk']['score'] += score
                    risk_categories['ip_risk']['documents'].append(doc_info)
            
            except Exception as doc_error:
                self.logger.warning(f"Risk quantification failed for document: {doc_error}")
                continue

        # Normalize scores and calculate overall risk
        total_documents = len(risk_documents)
        overall_risk_score = 0
        active_categories = 0

        for category in risk_categories:
            if risk_categories[category]['indicators'] > 0:
                # Normalize score (0-10 scale)
                risk_categories[category]['normalized_score'] = min(
                    (risk_categories[category]['score'] / risk_categories[category]['indicators']) * 10, 10
                ) if risk_categories[category]['indicators'] > 0 else 0
                overall_risk_score += risk_categories[category]['normalized_score']
                active_categories += 1

        overall_risk_score = overall_risk_score / active_categories if active_categories > 0 else 0

        return {
            'risk_categories': risk_categories,
            'overall_risk_score': overall_risk_score,
            'total_risk_documents': total_documents,
            'active_risk_categories': active_categories
            }
    
    def _format_risk_assessment_report(self, risk_assessment: Dict[str, Any], company: str) -> str:
        """Format comprehensive legal risk assessment report"""

        risk_categories = risk_assessment['risk_categories']
        overall_score = risk_assessment['overall_score']

        report_parts = [
            f"⚖️ Legal Risk Assessment Report: {company}",
            f"Overall Risk Score: {overall_score:.1f}/10.0",
            f"Risk Documents Analyzed: {risk_assessment['total_risk_documents']}"
            f"Active Risk Categories: {risk_assessment['active_risk_categories']}",
            ""
        ]

        # Risk category breakdown
        report_parts.append(f"📊 Risk Category Breakdown:")
        for category, data in risk_categories.items():
            if data['indicators'] > 0:
                category_name = category.replace('_', '').title()
                score = data['normalized_score']
                indicators = data['indicators']

                # Risk level assessment
                if score >= 7:
                    risk_level = "🔴 HIGH"
                elif score >= 4:
                    risk_level = "🟡 MEDIUM"
                else:
                    risk_level = "🟢 LOW"

                report_parts.append(f" • {category_name}: {risk_level} (Score: {score:.1f}, Indicators: {indicators})")

        # Key risk findings
        report_parts.append("")
        report_parts.append("🔍 Key Risk Findings:")

        high_risk_categories = [
            (cat, data) for cat, data in risk_categories.items()
            if data.get('normalized_score', 0) > 5 and data['documents']
        ]

        if high_risk_categories:
            for category, data in high_risk_categories[:3]:
                category_name = category.replace('_', '').title()
                top_doc = data['documents'][0] # Highest confidence document
                report_parts.append(f" • {category_name}: {top_doc['content']}")
        else:
            report_parts.append(f"• No high-risk categories identified")

        # Risk mitigation recommendations
        report_parts.append("")
        report_parts.append("🛡️ Risk Mitigation Recommendations:")

        if overall_score >= 7:
            report_parts.append("• 🔴 IMMEDIATE ACTION REQUIRED: Engage legal counsel for high-risk matters")
            report_parts.append("  • Conduct comprehensive legal audit")
            report_parts.append("  • Implement enhanced compliance monitoring")

        elif overall_score >= 4:
            report_parts.append("  • 🟡 ENHANCED MONITORING: Review high-medium risk categories")
            report_parts.append("  • Document risk mitigation strategies")
            report_parts.append("  • Regular legal compliance reviews")

        else:
            report_parts.append("  • 🟢 STANDARD MONITORING: Maintain current legal compliance programs")
            report_parts.append("  • Periodic risk assessment updates")
            report_parts.append("  • Continue standard legal due diligence")

        return "\n".join(report_parts)
    
    async def review_material_contracts(self, company: str) -> str:
        """Review material contracts and obligations with comprehensive analysis"""
        try:
            
            if not company or not company.strip():
                return "❌ Company ticker is required for contract review"
            
            company = company.upper().strip()
            self.logger.info(f"Reviewing material contracts for {company}")

            # Query for contract-related documents
            scored_documents = self.rag_system.query_with_similarity_scores(
                question=f"{company} material contracts agreements obligations commitments",
                company=company,
                metric_type="contract_doc",
                k=15,
                score_threshold=0.5
            )

            if not scored_documents:
                return f"📄 No material contract documents found for {company}"
            
            # Analyze contract obligations and commitments
            contract_analysis = self._analyze_material_contracts(scored_documents, company)
            return self._format_contract_review_report(contract_analysis, company)
        
        except Exception as e:
            self.logger.error(f"Material contract review failed for {company}: {e}")
            return f"❌ System error in contract review: {str(e)}"
        
    def _analyze_material_contracts(self, contract_documents: List[Tuple], company: str) -> Dict[str, Any]:
        """Analyze material contracts for obligations and commitments"""

        contract_categories = {
            'employment_agreements': [],
            'customer_contracts': [],
            'supplier_agreements': [],
            'debt_instruments': [],
            'lease_agreements': [],
            'intellectual_property': [],
            'joint_ventures': []
        }
        total_obligations = 0
        high_value_contracts = 0

        for doc, score in contract_documents:
            try:
                content_lower = doc.page_content.lower()
                contract_info = {
                    'content': doc.page_content,
                    'score': score,
                    'contract_type': self._identify_contract_type(content_lower),
                    'obligations': self._extract_contract_obligations(content_lower),
                    'term': self._extract_contract_term(content_lower),
                    'value_indicator': self._extract_contract_value(content_lower)
                }

                # Categorize contract
                category = contract_info['contract_type']
                if category in contract_categories:
                    contract_categories[category].append(contract_info)

                # Count obligations and high-value contracts
                total_obligations += len(contract_info['obligations'])
                if contract_info['value_indicator'] == 'high':
                    high_value_contracts += 1

            except Exception as doc_error:
                self.logger.warning(f"Contract analysis failed: {doc_error}")
                continue

        return {
            'contract_categories': contract_categories,
            'total_contracts': len(contract_documents),
            'total_obligations': total_obligations,
            'high_value_contracts': high_value_contracts,
            'active_categories': sum(1 for docs in contract_categories.values() if docs)
        }
    
    def _identify_contract_type(self, content: str) -> str:
        """Identify specific contract type from content"""

        content_lower = content.lower()

        if any(term in content_lower for term in ['employment', 'executive', 'compensation']):
            return 'employment_agreements'
        elif any(term in content_lower for term in ['customer', 'sale', 'revenue', 'service']):
            return 'customer_contracts'
        elif any(term in content_lower for term in ['supplier', 'vendor', 'purchase', 'procurement']):
            return 'supplier_agreements'
        elif any(term in content_lower for term in ['debt', 'loan', 'credit', 'financing']):
            return 'debt_instruments'
        elif any(term in content_lower for term in ['lease', 'rental', 'real estate', 'property']):
            return 'lease_agreements'
        elif any(term in content_lower for term in ['patent', 'trademark', 'license', 'intellectual']):
            return 'intellectual_property'
        elif any(term in content_lower for term in ['joint venture', 'partnership', 'alliance']):
            return 'joint_ventures'
        else:
            return 'other_agreements'
        
    def _extract_contract_obligations(self, content: str) -> List[str]:
        """Extract specific contractual obligations"""

        obligations = []
        content_lower = content.lower()

        obligation_indicators = [
            ('must', 'obligation to perform')
            ('shall', 'contractual requirement'),
            ('required to', 'mandatory action'),
            ('obligated', 'legal obligation'),
            ('commitment', 'ongoing commitment'),
            ('guarantee', 'performance guarantee'),
            ('warranty', 'product/service warranty')
        ]

        for term, obligation_type in obligation_indicators:
            if term in content_lower:
                obligations.append(obligation_type)

        return list(set(obligations))
    
    def _extract_contract_term(self, content: str) -> str:
        """Extract contract term information"""

        content_lower = content.lower()

        if any(term in content_lower for term in ['perpetual', 'evergreen']):
            return 'perpetual'
        elif any(term in content_lower for term in ['5 years', 'five years']):
            return '5 years'
        elif any(term in content_lower for term in ['3 years', 'three years']):
            return '3 years'
        elif any(term in content_lower for term in ['1 year', 'one year']):
            return '1 year'
        else:
            return 'unknown'
        
    def _extract_contract_value(self, content: str) -> str:
        """Extract contract value indicator"""
        content_lower = content.lower()
        
        # Look for monetary indicators
        if any(term in content_lower for term in ['$1,000,000', '$1m', 'million']):
            return 'high'
        elif any(term in content_lower for term in ['$100,000', '$100k']):
            return 'medium'
        elif any(term in content_lower for term in ['$10,000', '$10k']):
            return 'low'
        else:
            return 'unknown'
        
    def _format_contract_review_report(self, contract_analysis: Dict[str, Any], company: str) -> str:
        """Format comprehensive contract review report"""
        categories = contract_analysis['contract_categories']

        report_parts = [
            f"📄 Material Contract Review: {company}",
            f"Total Contracts Analyzed: {contract_analysis['total_contracts']}",
            f"Total Obligations Identified: {contract_analysis['total_obligations']}",
            f"High-Value Contracts: {contract_analysis['high_value_contracts']}",
            f"Contract Categories: {contract_analysis['active_categories']}",
            ""
        ]

        # Contract category breakdown
        report_parts.append("📊 Contract Category Analysis:")
        for category, contracts in categories.items():
            if contracts:
                category_name = category.replace('_', ' ').title()
                report_parts.append(f"  • {category_name}: {len(contracts)} contracts")

                # Show key obligations for each category
                all_obligations = []
                for contract in contracts[:2]:  # Top 2 contracts per category
                    all_obligations.extend(contract['obligations'])
                
                unique_obligations = list(set(all_obligations))[:3]  # Top 3 unique obligations
                if unique_obligations:
                    report_parts.append(f" Key Obligations: {', '.join(unique_obligations)}")

        # Key contract findings
        report_parts.append("")
        report_parts.append("🔍 Key Contract Findings:")
        
        if contract_analysis['high_value_contracts'] > 0:
            report_parts.append(f"  • {contract_analysis['high_value_contracts']} high-value contracts identified") 
        
        if contract_analysis['total_obligations'] > 50:
            report_parts.append("  • Significant contractual obligations portfolio")
        elif contract_analysis['total_obligations'] > 20:
            report_parts.append("  • Moderate contractual obligations")
        else:
            report_parts.append("  • Limited contractual obligations")

        # Contract risk assessment
        report_parts.append("")
        report_parts.append("⚖️ Contract Risk Assessment:")

        high_risk_categories = [
            cat for cat, contracts in categories.items() 
            if contracts and any(c['value_indicator'] == 'high' for c in contracts)
        ]
        
        if high_risk_categories:
            report_parts.append("  • 🔴 High-risk contracts in: " + ", ".join(
                [c.replace('_', ' ').title() for c in high_risk_categories]
            ))
        else:
            report_parts.append("  • 🟢 No high-risk contracts identified")

        return "\n".join(report_parts)
    
    async def validate_legal_findings(self, company: str, findings: str) -> str:
        """Validate legal findings against source documents with confidence scoring"""
        try:
            # Input validation
            if not company or not company.strip():
                return "❌ Company ticker is required for validation"
            if not findings or len(findings) < 10:
                return "❌ Findings must contain meaningful content for validation"
            
            company = company.upper().strip()
            self.logger.info(f"Validating legal findings for {company}: {findings[:100]}... ")

            # Use similarity search with scores for legal validation
            scored_documents = self.rag_system.query_with_similarity_scores(
                question=findings,
                company=company,
                k=10,
                score_threshold=0.3
            )

            if not scored_documents:
                self.logger.warning(f"No source legal documents found for validation of {company}")
                return f"⚠️ No source legal documents found for {company} to validate findings"
            
            # Analyze document relevance and legal conflicts
            validation_results = self._analyze_legal_validation(scored_documents, findings, company)
            return self._format_legal_validation_report(validation_results, company, len(scored_documents))
        
        except Exception as e:
            self.logger.error(f"Legal validation failed for {company}: {e}")
            return f"❌ System error during legal validation: {str(e)}"
    
    def _analyze_legal_validation(self, scored_documents: List[Tuple], findings: str, company: str) -> Dict[str, Any]:
        """Analyze legal document relevance and conflicts with findings"""

        supporting_docs = []
        conflicting_docs = []
        neutral_docs = []

        legal_terms = self._extract_legal_terms(findings)

        for doc, score in scored_documents:
            try:
                relevance_score = self._calculate_legal_relevance_score(doc.page_content, legal_terms)
                doc_type = doc.metadata.get('doc_type', 'unknown')
                legal_category = doc.metadata.get('legal_category', 'unknown')

                validation_doc = {
                    'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    'similarity_score': score,
                    'relevance_score': relevance_score,
                    'doc_type': doc_type,
                    'legal_category': legal_category,
                    'confidence_score': min(score, relevance_score) # Combined confidence
                }

                # Categorize based on legal relevance
                if relevance_score >= 0.7:
                    supporting_docs.append(validation_doc)
                elif relevance_score <= 0.3:
                    conflicting_docs.append(validation_doc)
                else:
                    neutral_docs.append(validation_doc)

            except Exception as doc_error:
                self.logger.warning(f" Legal document analysis failed: {doc_error}")
                continue

        return {
            'supporting': sorted(supporting_docs, key=lambda x:x['confidence'], reverse=True),
            'conflicting': sorted(conflicting_docs, key=lambda x:x['confidence'], reverse=True),
            'neutral': neutral_docs,
            'total_analyzed': len(scored_documents)
        }
    
    def _extract_legal_terms(self, text: str) -> List[str]:
        """Extract meaningful legal terms from findings"""
        try:            
            # Legal-specific stop words and important terms
            legal_stop_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for']
            legal_key_terms = [
                'compliance', 'regulation', 'litigation', 'contract', 'obligation',
                'risk', 'violation', 'enforcement', 'disclosure', 'governance',
                'liability', 'dispute', 'settlement', 'investigation', 'penalty'
            ]

            words = text.lower().split()
            key_terms = [
                word.strip('.,!?;:()[]{}"\'')
                for word in words
                if len(word) > 3 and word not in legal_stop_words and word in legal_key_terms
            ]
            return list(set(key_terms))[:10] # Return top 10 unique legal terms
        
        except Exception as e:
            self.logger.warning(f"Legal term extraction failed: {e}")
            return []
        
    def _calculate_legal_relevance_score(self, content: str, legal_terms: List[str]) -> float:
        """Calculate legal relevance score between content and legal terms"""

        if not legal_terms:
            return 0.0
        
        content_lower = content.lower()
        matches = sum(1 for term in legal_terms if term in content_lower)
        # Normalize score between 0 and 1 with legal weighting
        return min(matches/ len(legal_terms), 1.0)
    
    def _format_legal_validation_report(self, validation_results: Dict[str, Any], company: str, total_docs: int) -> str:
        """Format comprehensive legal validation report"""

        supporting = validation_results['supporting']
        conflicting = validation_results['conflicting']
        neutral = validation_results['neutral']

        report_parts = [f"⚖️ Legal Validation Report for {company}"]
        report_parts.append(f"Analyzed {total_docs} source legal documents")
        report_parts.append("")

        # Supporting legal evidence
        if supporting:
            report_parts.append(f"✅ Supporting Legal Evidence ({len(supporting)} documents):")
            for doc in supporting[:3]:
                report_parts.append(
                    f" • {doc['legal_category']} (confidence: {doc['confidence']:.2f})"
                )
        else:
            report_parts.append(f"✅ No strong supporting legal evidence found")

        # Potential legal conflicts
        if conflicting:
            report_parts.append(f"")
            report_parts.append(f"⚠️ Potential Legal Conflicts ({len(conflicting)} documents):")
            for doc in conflicting[:2]:
                report_parts.append(
                    f" • {doc['legal_category']} (confidence: {doc['confidence']:.2f})"
                )
        else:
            report_parts.append(f"")
            report_parts.append(f"✅ No legal conflicts identified")

        # Legal validation assessment
        report_parts.append("")

        if supporting and not conflicting:
            report_parts.append(f"🎯 Legal Assessment: Findings are well-supported by legal documents")
        elif conflicting and not supporting:
            report_parts.append(f"🎯 Legal Assessment: Findings conflict with legal documents - legal review required")
        elif supporting and conflicting:
            report_parts.append(f"🎯 Legal Assessment: Mixed legal evidence - further legal analysis recommended")
        else:
            report_parts.append(f"🎯 Legal Assessment: Insufficient legal evidence for validation")

        return "\n".join(report_parts)
    
    # Helper methods for production legal operations
    async def _ensure_legal_data(self, )
        



        

        
    

            
        


          

        

                         
                 

    

            

                







    


    
    
            
        
                                                          






        

                
                    
        
                    
                        
                        





                
        
    





