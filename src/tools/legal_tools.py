import logging
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

class LegalTools:
    """Legal Analysis Tools"""

    def __init__(self, rag_system, sec_collector):
        self.rag_system = rag_system
        self.sec_collector = sec_collector
        self.logger = logging.getLogger("LegalTools")

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
        
    def _build_legal_filings_summary(self, company: str, found: int, failed: int, total: int) -> str:
        """Build summary of legal filings retrieval"""
        if found == total:
            return f"✅ Successfully retrieved all {total} legal filings for {company}"
        elif found > 0:
            return f"📊 Retrieved {found}/{total} filings for {company}"
        else:
            return f"❌ Failed to retrieve legal filings for {company}"

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
                        metric_type="compliance_doc",
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
            'regulatory_actions': [],
            'other': []
        }

        for doc, score in scored_documents:
            try:
                content_lower = doc.page_content.lower()
                
                # Categorize based on keywords
                if 'securities' in content_lower or 'shareholder' in content_lower:
                    litigation_categories['securities_litigation'].append(doc.page_content)
                elif 'contract' in content_lower or 'breach' in content_lower:
                    litigation_categories['contract_disputes'].append(doc.page_content)
                elif 'patent' in content_lower or 'trademark' in content_lower or 'copyright' in content_lower:
                    litigation_categories['intellectual_property'].append(doc.page_content)
                elif 'employment' in content_lower or 'labor' in content_lower or 'discrimination' in content_lower:
                    litigation_categories['employment_law'].append(doc.page_content)
                elif 'regulatory' in content_lower or 'sec' in content_lower or 'doj' in content_lower:
                    litigation_categories['regulatory_actions'].append(doc.page_content)
                else:
                    litigation_categories['other'].append(doc.page_content)

            except Exception as e:
                self.logger.warning(f"Litigation categorization failed: {e}")
                continue

        return litigation_categories

    def _format_litigation_report(self, litigation_analysis: Dict[str, Any], company: str) -> str:
        """Format litigation analysis report"""
        
        report_parts = [f"⚖️ Litigation History Analysis: {company}"]
        
        has_litigation = False
        for category, cases in litigation_analysis.items():
            if cases:
                has_litigation = True
                report_parts.append(f"\n🔹 {category.replace('_', ' ').title()} ({len(cases)} cases):")
                for case in cases[:2]: # Show top 2 cases per category
                    preview = case[:150] + "..." if len(case) > 150 else case
                    report_parts.append(f"  • {preview}")

        if not has_litigation:
            report_parts.append("\n✅ No significant litigation history identified in available records.")

        return "\n".join(report_parts)
