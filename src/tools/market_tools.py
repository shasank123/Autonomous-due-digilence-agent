import logging
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

class MarketTools:
    """Market Analysis Tools"""

    def __init__(self, rag_system, sec_collector):
        self.rag_system = rag_system
        self.sec_collector = sec_collector
        self.logger = logging.getLogger("MarketTools")

    async def analyze_industry_trends(self, company: str, industry: Optional[str] = None) -> str:
        """Analyze industry trends and market positioning with comprehensive data validation"""

        try:
            if not company or not company.strip():
                return "❌ Company ticker is required for industry analysis"
            
            company = company.upper().strip()
            self.logger.info(f"Analyzing industry trends for {company}")

            # Determine industry if not provided
            if not industry:
                industry = await self._identify_company_industry(company)

            # Multi-faceted industry analysis
            analysis_queries = [
                f"{industry} market trends growth forecast",
                f"{company} competitive positioning {industry}",
                f"{industry} market share competitive landscape",
                f"{company} industry performance benchmarks"
            ]

            all_industry_docs = []
            for query in analysis_queries:
                try:

                    scored_documents = self.rag_system.query_with_similarity_scores(
                        question=query,
                        company=company,
                        metric_type="market_analysis",
                        k=5,
                        score_threshold=0.5
                    )
                    all_industry_docs.append(scored_documents)
                
                except Exception as query_error:
                    self.logger.warning(f"Industry query failed: {query_error}")
                    continue
            
            if not all_industry_docs:
                return f"📊 No industry analysis data found for {company} in {industry}"
            
            # Comprehensive trend analysis
            trend_analysis = self._analyze_market_trends(all_industry_docs, company, industry)
            return self._format_industry_analysis_report(trend_analysis, company, industry)
        
        except Exception as e:
            self.logger.error(f"Industry trend analysis failed for {company}: {e}")
            return f"❌ System error in industry analysis: {str(e)}"
        
    async def _identify_company_industry(self, company: str) -> str:
        """Identify company's primary industry from available data"""

        try:
            # Query for company business information
            company_docs = self.rag_system.query_with_similarity_scores(
                question=f"{company} business industry sector SIC",
                company=company,
                k=3,
                score_threshold=0.6
            )
           
            if company_docs:
                # Extract industry from documents
                for doc, score in company_docs:
                    industry = self._extract_industry_from_content(doc.page_content)
                    if industry:
                        self.logger.info(f"Identified industry for {company}: {industry}")
                        return industry
                    
            # Fallback to common industry mapping
            industry_mapping = {
                'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
                'TSLA': 'Automotive', 'F': 'Automotive', 'GM': 'Automotive',
                'JPM': 'Financial Services', 'BAC': 'Financial Services', 'WFC': 'Financial Services',
                'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
                'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'MRK': 'Healthcare'
            }

            return industry_mapping.get(company, 'Unknown Industry')
        
        except Exception as e:
            self.logger.warning(f"Industry identification failed for {company}: {e}")
            return "Unknown Industry"
        
    def _extract_industry_from_content(self, content: str) -> Optional[str]:
        """Extract industry information from document content"""

        try:
            content_lower = content.lower()

            # Industry keyword mapping
            industry_keywords = {
                'technology': ['technology', 'software', 'hardware', 'semiconductor', 'tech'],
                'healthcare': ['healthcare', 'pharmaceutical', 'medical', 'biotech', 'health'],
                'financial': ['financial', 'banking', 'insurance', 'investment', 'finance'],
                'energy': ['energy', 'oil', 'gas', 'renewable', 'petroleum'],
                'automotive': ['automotive', 'auto', 'vehicle', 'car', 'automobile'],
                'retail': ['retail', 'consumer', 'ecommerce', 'merchandise'],
                'industrial': ['industrial', 'manufacturing', 'machinery', 'equipment']
            }

            for industry, keywords in industry_keywords.items():
                if any(keyword in content_lower for keyword in keywords):
                    return industry.title()
                
            return None
        
        except Exception as e:
            self.logger.warning(f"Industry extraction failed: {e}")
            return None
        
    def _analyze_market_trends(self, industry_docs_list: List[List[Tuple]], company: str, industry: str) -> Dict[str, Any]:
        """Analyze comprehensive market trends from documents"""

        trend_analysis = {
            'growth_indicators': [],
            'competitive_position': [],
            'market_share_data': [],
            'risk_factors': [],
            'opportunities': []
        }

        # Flatten list of lists
        all_docs = [item for sublist in industry_docs_list for item in sublist]

        for doc, score in all_docs:
            try:
                content = doc.page_content
                content_lower = content.lower()

                # Growth indicators
                if any(term in content_lower for term in ['growth', 'expanding', 'increasing', 'rising']):
                    trend_analysis['growth_indicators'].append({
                        'content': content[:200] + "..." if len(content) > 200 else content,
                        'score': score,
                        'confidence': self._calculate_market_confidence(content)
                    })
        
                # Competitive positioning
                if any(term in content_lower for term in ['competitive', 'leader', 'position', 'market share']):
                    trend_analysis['competitive_position'].append({
                        'content': content[:200] + "..." if len(content) > 200 else content,
                        'score': score,
                        'company_mentioned': company.lower() in content_lower
                    })

                # Market share data
                if any(term in content_lower for term in ['market share', '%', 'percent', 'dominant']):
                    trend_analysis['market_share_data'].append({
                        'content': content[:200] + "..." if len(content) > 200 else content,
                        'score': score,
                        'metrics': self._extract_market_metric(content)
                    })

                # Risk factors
                if any(term in content_lower for term in ['risk', 'challenge', 'threat', 'competition']):
                    trend_analysis['risk_factors'].append({
                        'content': content[:200] + "..." if len(content) > 200 else content,
                        'score': score,
                        'severity': self._assess_risk_severity(content)
                    })

                # Opportunities
                if any(term in content_lower for term in ['opportunity', 'potential', 'growth area', 'emerging']):
                    trend_analysis['opportunities'].append({
                        'content': content[:200] + "..." if len(content) > 200 else content,
                        'score': score,
                        'potential': self._assess_opportunity_potential(content)
                    })

            except Exception as doc_error:
                self.logger.warning(f"Market trend analysis failed for document: {doc_error}")
                continue

        return trend_analysis
    
    def _calculate_market_confidence(self, content: str) -> float:
        """Calculate confidence score for market analysis"""

        try:
            confidence_factors = 0
            total_factors = 0

            # Quantitative data presence
            if any(char.isdigit() for char in content):
                confidence_factors += 1
            total_factors += 1

            # Specific metrics mentioned
            if any(term in content.lower() for term in ['%', 'growth', 'increase', 'decrease']):
                confidence_factors += 1
            total_factors += 1

            # Time references
            if any(term in content.lower() for term in ['202', 'q1', 'q2', 'q3', 'q4']):
                confidence_factors += 1
            total_factors += 1

            return confidence_factors / total_factors
        
        except Exception as e:
            self.logger.error(f"Market confidence calculation failed: {e}")
            return 0.5
        
    def _extract_market_metric(self, content: str) -> List[str]:
        """Extract market metrics from content"""

        metrics = []
        try:
            lines = content.split('\n')
            for line in lines:
                line_lower = line.lower()
                if any(term in line_lower for term in ['market share', 'growth rate', 'cagr']):
                    metrics.append(line.strip())
            return metrics[:3]

        except Exception as e:
            self.logger.warning(f"Market metrics extraction failed: {e}")
            return []

    def _assess_risk_severity(self, content: str) -> str:
        """Assess risk severity from content"""

        content_lower = content.lower()

        if any(term in content_lower for term in ['high risk', 'significant', 'major', 'severe']):
            return "HIGH"
        elif any(term in content_lower for term in ['moderate', 'medium', 'some risk']):
            return "MEDIUM"
        elif any(term in content_lower for term in ['low risk', 'minor', 'limited']):
            return "LOW"
        else:
            return "UNKNOWN"

    def _assess_opportunity_potential(self, content: str) -> str:
        """Assess opportunity potential from content"""
        content_lower = content.lower()
        
        if any(term in content_lower for term in ['significant', 'substantial', 'major opportunity']):
            return "HIGH"
        elif any(term in content_lower for term in ['moderate', 'potential', 'emerging']):
            return "MEDIUM"
        elif any(term in content_lower for term in ['limited', 'small', 'niche']):
            return "LOW"
        else:
            return "UNKNOWN"

    def _format_industry_analysis_report(self, trend_analysis: Dict[str, Any], company: str, industry: str) -> str:
        """Format comprehensive industry analysis report"""

        report_parts = [
            f"🏭 Industry Analysis Report: {company}",
            f"Industry: {industry}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]

        # Add sections based on analysis
        if trend_analysis['growth_indicators']:
            report_parts.append("📈 Growth Indicators:")
            for item in trend_analysis['growth_indicators'][:3]:
                report_parts.append(f"• {item['content']} (Confidence: {item['confidence']:.2f})")
            report_parts.append("")

        if trend_analysis['competitive_position']:
            report_parts.append("🏆 Competitive Position:")
            for item in trend_analysis['competitive_position'][:3]:
                report_parts.append(f"• {item['content']}")
            report_parts.append("")

        if trend_analysis['risk_factors']:
            report_parts.append("⚠️ Risk Factors:")
            for item in trend_analysis['risk_factors'][:3]:
                report_parts.append(f"• [{item['severity']}] {item['content']}")
            report_parts.append("")

        return "\n".join(report_parts)
