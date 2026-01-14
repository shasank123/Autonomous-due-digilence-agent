# src/tools/market_tools.py
import logging
import re
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

class MarketTools:
    """Market Analysis Tools"""

    def __init__(self, rag_system, sec_collector):
        self.rag_system = rag_system
        self.sec_collector = sec_collector
        self.logger = logging.getLogger("MarketTools")

    def _format_metric_name(self, name: str) -> str:
        """Convert CamelCase to readable format"""
        if not name:
            return "Unknown"
        readable = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        readable = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', readable)
        return readable

    def _format_currency(self, value) -> str:
        """Format large numbers as readable currency"""
        try:
            num = float(value)
            if num >= 1_000_000_000:
                return f"${num/1_000_000_000:.1f}B"
            elif num >= 1_000_000:
                return f"${num/1_000_000:.1f}M"
            elif num >= 1_000:
                return f"${num/1_000:.1f}K"
            else:
                return f"${num:,.0f}"
        except:
            return str(value)

    async def analyze_industry_trends(self, company: str, industry: Optional[str] = None) -> str:
        """Analyze industry trends and market positioning with comprehensive data validation"""
        try:
            if not company or not company.strip():
                return " [ERROR] Company ticker is required for industry analysis"
            
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
                        metric_type="industry_classification", # Corrected (seeded doc_type)
                        k=5,
                        score_threshold=1.2
                    )

                    if not scored_documents:
                         scored_documents = self.rag_system.query_with_similarity_scores(
                            question=query,
                            company=company,
                            metric_type=None, # Fallback to broad search
                            k=5,
                            score_threshold=1.0
                        )

                    all_industry_docs.extend(scored_documents)
                
                except Exception as query_error:
                    self.logger.warning(f"Industry query failed: {query_error}")
                    continue
            
            if not all_industry_docs:
                return f" [SUMMARY] No industry analysis data found for {company} in {industry}"
            
            # Comprehensive trend analysis
            trend_analysis = self._analyze_market_trends(all_industry_docs, company, industry)
            return self._format_industry_analysis_report(trend_analysis, company, industry)
        
        except Exception as e:
            self.logger.error(f"Industry trend analysis failed for {company}: {e}")
            return f" [ERROR] System error in industry analysis: {str(e)}"

    async def _identify_company_industry(self, company: str) -> str:
        """Identify company's primary industry from available data"""
        try:
            # Query for company business information
            company_docs = self.rag_system.query_with_similarity_scores(
                question=f"{company} business industry sector SIC",
                company=company,
                metric_type="industry_classification", # Corrected (seeded doc_type)
                k=3,
                score_threshold=1.2
            )
            
            if not company_docs:
                company_docs = self.rag_system.query_with_similarity_scores(
                    question=f"{company} business industry sector SIC",
                    company=company,
                    metric_type=None, # Fallback
                    k=3,
                    score_threshold=1.0
                )
            
            if company_docs:
                # First check metadata for sector/industry
                for doc, score in company_docs:
                    sector = doc.metadata.get('sector')
                    if sector and sector != 'Unknown':
                        self.logger.info(f"Found sector for {company} from metadata: {sector}")
                        return sector
                    
                    industry = doc.metadata.get('industry')
                    if industry and industry != 'Unknown':
                        self.logger.info(f"Found industry for {company} from metadata: {industry}")
                        return industry
                
                # Fallback: Extract industry from document content
                for doc, score in company_docs:
                    industry = self._extract_industry_from_content(doc.page_content)
                    if industry:
                        self.logger.info(f"Identified industry for {company}: {industry}")
                        return industry
                    
            # Fallback to common industry mapping
            industry_mapping = {
                'AAPL': 'Consumer Electronics', 'MSFT': 'Software & Cloud', 'GOOGL': 'Internet Services',
                'AMZN': 'E-Commerce & Cloud', 'META': 'Social Media', 'NVDA': 'Semiconductors',
                'TSLA': 'Electric Vehicles', 'F': 'Automotive', 'GM': 'Automotive',
                'JPM': 'Banking', 'BAC': 'Banking', 'WFC': 'Banking', 'GS': 'Investment Banking',
                'XOM': 'Oil & Gas', 'CVX': 'Oil & Gas', 'COP': 'Oil & Gas',
                'JNJ': 'Pharmaceuticals', 'PFE': 'Pharmaceuticals', 'MRK': 'Pharmaceuticals',
                'UNH': 'Health Insurance', 'WMT': 'Retail', 'TGT': 'Retail'
            }

            return industry_mapping.get(company, 'Technology')
        
        except Exception as e:
            self.logger.warning(f"Industry identification failed for {company}: {e}")
            return "Technology"
        
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
        
    def _analyze_market_trends(self, all_docs: List[Tuple], company: str, industry: str) -> Dict[str, Any]:
        """Analyze comprehensive market trends from documents"""
        
        trend_analysis = {
            'growth_indicators': [],
            'competitive_position': [],
            'market_share_data': [],
            'risk_factors': [],
            'opportunities': []
        }

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
            total_factors = 3 # Fixed total factors to prevent division by zero

            # Quantitative data presence
            if any(char.isdigit() for char in content):
                confidence_factors += 1

            # Specific metrics mentioned
            if any(term in content.lower() for term in ['%', 'growth', 'increase', 'decrease']):
                confidence_factors += 1

            # Time references
            if any(term in content.lower() for term in ['202', 'q1', 'q2', 'q3', 'q4']):
                confidence_factors += 1

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
            f" [REPORT] Industry Analysis Report: {company}",
            f"Industry: {industry}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]

        # Add sections based on analysis
        if trend_analysis['growth_indicators']:
            report_parts.append(" [GROWTH] Growth Indicators:")
            for item in trend_analysis['growth_indicators'][:3]:
                report_parts.append(f"• {item['content']} (Confidence: {item['confidence']:.2f})")
            report_parts.append("")

        if trend_analysis['competitive_position']:
            report_parts.append(" [COMP] Competitive Position:")
            for item in trend_analysis['competitive_position'][:3]:
                report_parts.append(f"• {item['content']}")
            report_parts.append("")

        if trend_analysis['risk_factors']:
            report_parts.append(" [RISK] Risk Factors:")
            for item in trend_analysis['risk_factors'][:3]:
                report_parts.append(f"• [{item['severity']}] {item['content']}")
            report_parts.append("")

        # Add Opportunity Section
        if trend_analysis['opportunities']:
            report_parts.append(" [OPPORTUNITY] Market Opportunities:")
            for item in trend_analysis['opportunities'][:3]:
                report_parts.append(f"• [{item['potential']}] {item['content']}")
            report_parts.append("")

        return "\n".join(report_parts)

    async def research_competitive_landscape(self, company: str, competitors: List[str] = None) -> str:
        """Research competitive environment"""
        try:
            if not company or not company.strip():
                return " [ERROR] Company ticker is required"
            
            company = company.upper().strip()
            self.logger.info(f"Researching competitive landscape for {company}")

            query = f"{company} competitors competitive landscape market share"
            if competitors:
                query += f" vs {' '.join(competitors)}"

            scored_documents = self.rag_system.query_with_similarity_scores(
                question=query,
                company=company,
                metric_type="competitive_analysis", # Corrected (seeded doc_type)
                k=5,
                score_threshold=1.2
            )

            if not scored_documents:
                scored_documents = self.rag_system.query_with_similarity_scores(
                    question=query,
                    company=company,
                    metric_type=None, # Fallback
                    k=5,
                    score_threshold=1.0
                )

            if not scored_documents:
                return f" [SUMMARY] No competitive landscape data found for {company}"

            # Deduplicate and format results
            seen_metrics = set()
            results = []
            for doc, score in scored_documents:
                metric = doc.metadata.get('metric', '')
                
                if metric and metric not in seen_metrics:
                    seen_metrics.add(metric)
                    
                    # Try to get value from metadata first, then parse from content
                    value = doc.metadata.get('value')
                    if value is None:
                        # Fallback: parse value from page_content
                        value = self._extract_value_from_content(doc.page_content)
                    
                    period = doc.metadata.get('period', '')
                    
                    # Format nicely
                    readable_name = self._format_metric_name(metric)
                    formatted_value = self._format_currency(value) if value else "N/A"
                    formatted_period = period[:10] if period else "N/A"
                    
                    results.append(f"• **{readable_name}**: {formatted_value} ({formatted_period})")
            
            if not results:
                return f" [SUMMARY] No distinct competitive data found for {company}"

            return f" [REPORT] Competitive Landscape for {company}:\n" + "\n".join(results[:5])
        
        except Exception as e:
            self.logger.error(f"Competitive research failed for {company}: {e}")
            return f" [ERROR] System error researching competitors: {str(e)}"
    
    def _extract_value_from_content(self, content: str) -> Optional[float]:
        """Extract numeric value from document content"""
        try:
            # Look for "Value: X USD" pattern
            import re
            match = re.search(r'Value:\s*([\d,]+(?:\.\d+)?)\s*USD', content)
            if match:
                return float(match.group(1).replace(',', ''))
            
            # Look for any large number
            numbers = re.findall(r'\b(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\b', content)
            if numbers:
                # Return the largest number found
                parsed = [float(n.replace(',', '')) for n in numbers]
                return max(parsed) if parsed else None
            
            return None
        except Exception:
            return None

    async def assess_market_opportunities(self, company: str, segments: List[str] = None) -> str:
        """Evaluate growth opportunities"""
        try:
            if not company or not company.strip():
                return " [ERROR] Company ticker is required"
            
            company = company.upper().strip()
            self.logger.info(f"Assessing market opportunities for {company}")

            query = f"{company} market opportunities growth potential expansion"
            if segments:
                query += f" in {' '.join(segments)}"

            scored_documents = self.rag_system.query_with_similarity_scores(
                question=query,
                company=company,
                metric_type="market_opportunities", # Corrected (seeded doc_type)
                k=5,
                score_threshold=1.2
            )

            if not scored_documents:
                 scored_documents = self.rag_system.query_with_similarity_scores(
                    question=query,
                    company=company,
                    metric_type=None, # Fallback
                    k=5,
                    score_threshold=1.0
                )

            if not scored_documents:
                return f" [SUMMARY] No market opportunity data found for {company}"

            # Deduplicate and format results
            seen_metrics = set()
            results = []
            for doc, score in scored_documents:
                metric = doc.metadata.get('metric', '')
                
                if metric and metric not in seen_metrics:
                    seen_metrics.add(metric)
                    value = doc.metadata.get('value', '')
                    period = doc.metadata.get('period', '')
                    
                    # Format nicely
                    readable_name = self._format_metric_name(metric)
                    formatted_value = self._format_currency(value) if value else "N/A"
                    formatted_period = period[:10] if period else "N/A"
                    
                    results.append(f"• **{readable_name}**: {formatted_value} ({formatted_period})")
            
            if not results:
                return f" [SUMMARY] No distinct opportunity data found for {company}"

            return f" [REPORT] Market Opportunities for {company}:\n" + "\n".join(results[:5])
        
        except Exception as e:
            self.logger.error(f"Opportunity assessment failed for {company}: {e}")
            return f" [ERROR] System error assessing opportunities: {str(e)}"