# src/agents/market_analyst.py
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
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from rag.core import ProductionRAGSystem
from data.processors.document_parser import DocumentProcessor
from data.collectors.sec_edgar import SECDataCollector

class MarketAgentTeam:
    """Market Analysis Agent Team integrated with our RAG system"""
    
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
        logger = logging.getLogger("MarketAgentTeam")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    # Market Analysis Tool Functions
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

            return industry_mapping.get('company', 'Unknown Industry')
        
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
        
    def _analyze_market_trends(self, industry_docs: List[Tuple], company: str, industry: str) -> Dict[str, Any]:
        """Analyze comprehensive market trends from documents"""

        trend_analysis = {
            'growth_indicators': [],
            'competitive_position': [],
            'market_share_data': [],
            'risk_factors': [],
            'opportunities': []
        }

        for doc, score in industry_docs:
            try:
                content = doc.page_content
                content_lower = content.lower()

                # Growth indicators
                if any(term in content_lower for term in ['growth', 'expanding', 'increasing', 'rising']):
                    trend_analysis['growth_indicators'].append(
                        'content': content[:200] + "..." if len(content) > 200 else content,
                        'score': score,
                        'confidence': self._calculate_market_confidence(content)
                    )
        
                # Competitive positioning
                if any(term in content_lower for term in ['competitive', 'leader', 'position', 'market share']):
                    trend_analysis['competitive_position'].append(
                        'content': content[:200] + "..." if len(content) > 200 else content,
                        'score': score,
                        'company_mentioned': company.lower() in content_lower
                    )

                # Market share data
                if any(term in content_lower for term in ['market share', '%', 'percent', 'dominant']):
                    trend_analysis['market_share_data'].append(
                        'content': content[:200] + "..." if len(content) > 200 else content,
                        'score': score,
                        'metrics': self._extract_market_metrics(content)
                    )

                # Risk factors
                if any(term in content_lower for term in ['risk', 'challenge', 'threat', 'competition']):
                    trend_analysis['risk_factors'].append(
                        'content': content[:200] + "..." if len(content) > 200 else content,
                        'score': score,
                        'severity': self._assess_risk_severity(content)
                    )

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

        metrics =[]
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

    def _asses_risk_severity(self, content: str) -> str:
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

        # Growth Analysis
        growth_indicators = trend_analysis['growth_indicators']
        if growth_indicators:
            report_parts.append("📈 Growth Indicators:")
            for indicator in growth_indicators:
                report_parts.append(f"• {indicator['content']} (confidence: {indicator['confidence']:.2f})")

        else:
            report_parts.append("📈 Growth Indicators: Limited data available")

        # Competitive Position
        competitive_data = trend_analysis['competitive_position']
        if competitive_data:
            report_parts.append("")
            report_parts.append("🥊 Competitive Positioning:")

            company_specific = [d for d in competitive_data if d['company_mentioned']]
            if company_specific:
                report_parts.append(f"• Company-specific insights: {len(company_specific)} documents")

            for position in competitive_data[:2]:
                report_parts.append(f"• {position['content']}")  

        # Market Share
        market_share = trend_analysis['market_share_data']
        if market_share:
            report_parts.append("")
            report_parts.append("📊 Market Share Analysis:")

            for share_data in market_share[:2]:
                if share_data['metrics']:
                    report_parts.append(f" • Metrics: {','.join(share_data['metrics'])}")

        # Risk Assessment
        risks = trend_analysis['risk_factors']
        if risks:
            report_parts.append("")
            report_parts.append("⚠️  Risk Factors:")

            high_risks = [r for r in risks if r['severity'] == 'HIGH']
            if high_risks:
                report_parts.append(f" • High severity risks: {len(high_risks)} identified")
            
            for risk in risks[:2]:
                report_parts.append(f" • {risk['content']} (Severity: {risk['severity']})")

        # Opportunities
        opportunities = trend_analysis['opportunities']
        if opportunities:
            report_parts.append("")
            report_parts.append("🎯 Growth Opportunities:")

            high_potential = [o for o in opportunities if o['potential'] == 'HIGH']
            if high_potential:
                report_parts.append(f" • High-potential opportunities: {len(high_potential)} identified")
            
            for opportunity in opportunities[:2]:
                report_parts.append(f" • {opportunity['content']} (Potential: {opportunity['potential']})")

        # Summary
        report_parts.append("")
        report_parts.append("🎯 Market Assessment Summary:")

        total_insights = sum(len(data) for data in trend_analysis.values())

        if total_insights > 15:
            report_parts.append("  • 📊 Comprehensive market intelligence available")
        elif total_insights > 8:
            report_parts.append("  • 📈 Good market insights for strategic planning")
        else:
            report_parts.append("  • 📋 Limited market data - consider additional research")

        return "\n".join(report_parts)
    
    async def research_competitive_landscape(self, company: str, competitors: Optional[List[str]] = None) -> str:
        """Research competitive landscape with peer analysis"""

        try:
            if not company or not company.strip():
                return "❌ Company ticker is required for competitive analysis"
            
            company = company.upper().strip()
            self.logger.info(f"Researching competitive landscape for {company}")

            # Identify competitors if not provided
            if not competitors:
                competitors = await self._identify_competitors(company)

            competitive_analysis = {
                'direct_competitors': [],
                'competitive_advantages': [],
                'market_differentiators': [],
                'peer_performance': []
            }

            # Analyze each competitor
            for competitor in competitors[:5]: # Limit to top 5 competitors
                try:
                    competitor_analysis = await self._analyze_competitor(company, competitor)
                    if competitor_analysis:
                        competitive_analysis['direct_competitors'].append(competitor_analysis)

                except Exception as comp_error:
                    self.logger.warning(f"Competitor analysis failed for {competitor}: {comp_error}")
                    continue
            
            # Analyze company's competitive position
            company_advantages = self._analyze_competitive_advantages(company)
            competitive_analysis['competitive_advantages'] = company_advantages

            return self._format_competitive_analysis_report(competitive_analysis, company, competitors)
        
        except Exception as e:
            self.logger.error(f"Competitive landscape research failed for {company}: {e}")
            return f"❌ System error in competitive analysis: {str(e)}"
        
    async def _identify_competitors(self, company: str) -> List[str]:
        """Identify company's main competitors"""

        try:
            # Common competitor mappings
            competitor_map = {
                'AAPL': ['MSFT', 'GOOGL', 'Samsung'],
                'MSFT': ['AAPL', 'GOOGL', 'AMZN', 'ORCL'],
                'GOOGL': ['MSFT', 'AAPL', 'META', 'AMZN'],
                'TSLA': ['F', 'GM', 'RIVN', 'NIO'],
                'JPM': ['BAC', 'WFC', 'C', 'GS'],
                'XOM': ['CVX', 'COP', 'BP', 'SHEL'],
                'JNJ': ['PFE', 'MRK', 'ABT', 'GILD']
            }

            return competitor_map.get(company, [])

        except Exception as e:
            self.logger.warning(f"Competitor identification failed for {company}: {e}")
            return []

    async def _analyze_competitor(self, company: str, competitor: str) -> Dict[str, Any]:
        """Analyze individual competitor"""

        try:
            # Query for competitor comparison data
            scored_docs = self.rag_system.query_with_similarity_scores(
                question=f"{company} vs {competitor} comparison competitive",
                company=company,
                metric_type="competitive_analysis",
                k=3,
                score_threshold=0.4
            )

            if not scored_docs:
                return None

            analysis = {
                'competitor': competitor,
                'comparison_points': [],
                'strengths': [],
                'weaknesses': []
            }

            for doc, score in scored_docs:
                content_lower = doc.page_content.lower()

                # Extract comparison points
                if 'vs' in content_lower or 'compar' in content_lower:
                    analysis['comparison_points'].append({
                        'content': doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                        'score': score
                    })

                # Identify strengths/weaknesses
                if any(term in content_lower for term in ['advantage', 'strength', 'better', 'superior']):
                    analysis['strengths'].append({
                        'content': doc.page_content[:100],
                        'score': score
                    })

                if any(term in content_lower for term in ['weakness', 'challenge', 'lagging', 'trailing']):
                        analysis['weaknesses'].append({
                            'content': doc.page_content[:100],
                            'score': score
                        })

            return analysis if analysis['comparison_points'] else None
        
        except Exception as e:
            self.logger.warning(f"Individual competitor analysis failed for {competitor}: {e}")
            return None
        
    async def _analyze_competitive_advantages(self, company: str) -> List[Dict[str, Any]]:
        """Analyze company's competitive advantages"""

        try:
            scored_docs = self.rag_system.query_with_similarity_scores(
                question=f"{company} competitive advantage strength differentiation",
                company=company,
                metric_type="business_analysis",
                k=5,
                score_threshold=0.5
            )

            advantages = []
            
            for doc, score in scored_docs:
                content_lower = doc.page_content.lower()

                advantage_types = {
                    'technology': ['technology', 'innovation', 'patent', 'proprietary'],
                    'brand': ['brand', 'reputation', 'recognition', 'loyalty'],
                    'cost': ['cost', 'efficiency', 'margin', 'pricing'],
                    'distribution': ['distribution', 'network', 'reach', 'channel']
                }

                for advantage_type, keywords in advantage_type.items():
                    if any(keyword in content_lower for keyword in keywords):
                        advantages.append({
                            'type': advantage_type,
                            'content': doc.page_content[:120] if len(doc.page_content) > 120 else doc.page_content,
                            'score': score
                        })
                        break
            
            return advantages
        
        except Exception as e:
            self.logger.warning(f"Competitive advantages analysis failed for {company}: {e}")
            return []
        
    def _format_competitive_analysis_report(self, competitive_analysis: Dict[str, Any], company: str, competitors: List[str]) -> str:
        """Format comprehensive competitive analysis report"""

        report_parts = [
            f"🥊 Competitive Landscape Analysis: {company}",
            f"Key Competitors: {', '.join(competitors[:5])}",
            f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]

        # Competitor Analysis
        direct_competitors = competitive_analysis['direct_competitors']
        if direct_competitors:
            report_parts.append(f"🎯 Direct Competitor Analysis:")

            for competitor in direct_competitors:
                report_parts.append(f" • {competitor['competitor']}:")

                if competitor['strengths']:
                    report_parts.append(f" - Strengths: {len(competitor['strengths'])} identified")

                if competitor['weakness']:
                    report_parts.append(f" - Weaknesses: {len(competitor['weaknesses'])} identified")

        else:
            report_parts.append("🎯 Direct Competitor Analysis: Limited comparison data available")

        # Competitive Advantages
        advantages = competitive_analysis['competitive_advantages']
        if advantages:
            report_parts.append("")
            report_parts.append("🏆 Competitive Advantages:")

            advantage_type = {}
            for advantage in advantages:
                adv_type = advantage['type']
                
                if adv_type not in advantage_type:
                    advantage_type[adv_type] = 0

                advantage_type[adv_type] += 1

            for adv_type, count in advantage_type.items():
                report_parts.append(f"  • {adv_type.title()}: {count} advantage(s) identified")

            # Show top advantages
            for advantage in advantages[:3]:
                report_parts.append(f" - {advantage['content']}")

        # Strategic Recommendations
        report_parts.append("")
        report_parts.append("💡 Strategic Recommendations:")

        if advantages and direct_competitors:
            report_parts.append("  • Leverage identified competitive advantages in market positioning")
            report_parts.append("  • Monitor key competitors for emerging threats")
            report_parts.append("  • Focus on differentiation in competitive segments")

        elif advantages:
            report_parts.append("  • Capitalize on competitive advantages for market growth")
            report_parts.append("  • Conduct deeper competitor research for complete analysis")

        else:
            report_parts.append("  • Invest in competitive intelligence gathering")
            report_parts.append("  • Focus on building sustainable competitive advantages")   

        return "\n".join(report_parts)

    async def assess_market_opportunities(self, company: str, market_segments: Optional[List[str]] = None) -> str:
        """Assess market opportunities and growth potential"""
        try:
            if not company or not company.strip():
                return "❌ Company ticker is required for market opportunity assessment"

            company = company.upper().strip()
            self.logger.info(f"Assessing market opportunities for {company}") 

            # Identify relevant market segments
            if not market_segments:
                market_segments = await self._identify_market_segments(company)

            opportunity_analysis = {
                'growth_segments': [],
                'emerging_markets': [],
                'innovation_areas': [],
                'partnership_opportunities': []
            }

            # Analyze each market segment
            for segment in market_segments:
                try:
                    segment_analysis = self._analyze_market_segment(company, segment)
                    if segment_analysis:
                        if segment_analysis['growth_potential'] == 'HIGH':
                            opportunity_analysis['growth_segments'].append(segment_analysis)
                        elif 'emerging' in segment.lower() or 'new' in segment.lower():
                            opportunity_analysis['emerging_markets'].append(segment_analysis)

                except Exception as segment_error:
                    self.logger.warning(f"Market segment analysis failed for {segment}: {segment_error}")
                    continue

            # Innovation opportunities  
            innovations_ops = self._analyze_innovation_opportunities(company)
            opportunity_analysis['innovation_areas'].append(innovations_ops)

            return self._format_opportunity_assessment_report(opportunity_analysis, company, market_segments)
        
        except Exception as e:
            self.logger.error(f"Market opportunity assessment failed for {company}: {e}")
            return f"❌ System error in opportunity assessment: {str(e)}"
        
    async def _identify_market_segments(self, company: str) -> List[str]:
         """Identify relevant market segments for the company"""

         try:
             # Common market segment mappings
             segment_map = {
                'AAPL': ['Smartphones', 'Wearables', 'Services', 'Personal Computing'],
                'MSFT': ['Cloud Computing', 'Enterprise Software', 'Gaming', 'AI'],
                'TSLA': ['Electric Vehicles', 'Energy Storage', 'Autonomous Driving'],
                'JNJ': ['Pharmaceuticals', 'Medical Devices', 'Consumer Health'],
                'AMZN': ['E-commerce', 'Cloud Services', 'Digital Advertising', 'Logistics']
             }

             return segment_map.get(company, ['Core Business', 'Adjacent Markets', 'New Ventures'])
         
         except Exception as e:
            self.logger.warning(f"Market segment identification failed for {company}: {e}")
            return ['Core Business', 'Growth Areas']
         
    async def _analyze_market_segment(self, company: str, segment: str) -> Optional[Dict[str, Any]]:
        """Analyze specific market segment opportunity"""

        try:
            scored_docs = self.rag_system.query_with_similarity_scores(
                question=f"{company} {segment} market opportunity growth potential",
                company=company,
                metric_type="market_opportunity",
                k=3,
                score_threshold=0.4
            )

            if not scored_docs:
                return None

            analysis = {
                'segment': segment,
                'growth_potential': 'MEDIUM',
                'key_insights': [],
                'barriers': []
            }

            for doc, score in scored_docs:
                content_lower = doc.page_content.lower()

                # Assess growth potential
                if any(term in content_lower for term in ['high growth', 'rapidly expanding', 'significant opportunity']):
                    analysis['growth_potential'] == 'HIGH'
                
                elif any(term in content_lower for term in ['limited', 'saturated', 'mature']):
                    analysis['growth_potential'] == 'LOW'

                # Extract key insights
                analysis['key_insights'].append({
                    'content': doc.page_content[:120] + "..." if len(doc.page_content) > 120 else doc.page_content,
                    'score': score
                })

                # Identify barriers
                if any(term in content_lower for term in ['barrier', 'challenge', 'obstacle', 'competition']):
                    analysis['barriers'].append({
                        'content': doc.page_content[:100],
                        'score': score
                    })

            return analysis
        
        except Exception as e:
            self.logger.error(f"Market segment analysis failed for {segment}: {e}")
            return None

    async def _analyze_innovation_opportunities(self, company: str) -> List[Dict[str, Any]]:
        """Analyze innovation and R&D opportunities"""

        try:
            scored_docs = self.rag_system.query_with_similarity_scores(
                question=f"{company} innovation research development new technology",
                company=company,
                metric_type="innovation",
                k=4,
                score_threshold=0.5
            )

            innovations = []
            for doc, score in scored_docs:
                content_lower = doc.page_content.lower()

                innovation_areas = {
                    'technology': ['ai', 'machine learning', 'blockchain', 'iot'],
                    'product': ['new product', 'feature', 'platform', 'service'],
                    'process': ['efficiency', 'automation', 'optimization', 'workflow']
                }

                for area, keywords in innovation_areas.items():
                    if any(keyword in content_lower for keyword in keywords):
                        innovations.append({
                            'area': area,
                            'content': doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                            'score': score
                        })
                        break
                
            return innovations
        
        except Exception as e:
            self.logger.warning(f"Innovation opportunities analysis failed for {company}: {e}")
            return []
        
    def _format_opportunity_assessment_report(self, opportunity_analysis: Dict[str, Any], company: str, market_segments: List[str]) -> str:
        """Format comprehensive opportunity assessment report"""

        report_parts = [
            f"🎯 Market Opportunity Assessment: {company}",
            f"Analyzed Segments: {', '.join(market_segments)}",
            f"Assessment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]

        # Growth Segments
        growth_segments = opportunity_analysis['growth_segments']
        if growth_segments:
            report_parts.append("🚀 High-Growth Market Segments:")

            for segment in growth_segments:
                report_parts.append(f" •{segment['segment']}:")

                if segment['key_insights']:
                    report_parts.append(f" - {segment['key_insights'][0]['content']}")

                if segment['barriers']:
                    report_parts.append(f" - Barriers: {len(segment['barriers'])} identified")

        else:
            report_parts.append("🚀 High-Growth Segments: No high-growth segments identified")

        # Emerging Markets
        emerging_markets = opportunity_analysis['emerging_markets']
        if emerging_markets:
            report_parts.append("")
            report_parts.append("🌱 Emerging Market Opportunities:")

            for market in emerging_markets:
                report_parts.append(f"• {market['segment']} (Early-stage opportunity)")

        # Innovation Areas
        innovation_areas = opportunity_analysis['innovation_areas']
        if innovation_areas:
            report_parts.append("")
            report_parts.append("💡 Innovation & R&D Opportunities:")

            innovation_by_area = {}
            for innovation in innovation_areas:
                area = innovation['area']
                if area not in innovation_by_area:
                    innovation_by_area[area] = []
                innovation_by_area[area].append(innovation)

            for area, innovations in innovation_by_area.items():
                report_parts.append(f" •  {area.title()}: {len(innovations)} opportunity(s)")

        # Strategic Recommendations
        report_parts.append("")
        report_parts.append("📈 Strategic Growth Recommendations:")

        total_opportunities = (len(growth_segments) + len(emerging_markets)) + len(innovation_areas)

        if total_opportunities > 8:
            report_parts.append("  • 🎯 Multiple high-potential opportunities identified")
            report_parts.append("  • Prioritize based on strategic alignment and resources")
            report_parts.append("  • Consider phased market entry approach")

        elif total_opportunities > 3:
            report_parts.append("  • 📊 Solid opportunity portfolio available")
            report_parts.append("  • Focus on 2-3 most promising segments")
            report_parts.append("  • Build capabilities for selected opportunities")

        else:
            report_parts.append("  • 🔍 Limited opportunity data available")
            report_parts.append("  • Conduct deeper market research")
            report_parts.append("  • Explore adjacent market expansion")

        return "\n".join(report_parts)
    
    async def analyze_company_market(self, company_ticker: str, additional_context: str = "") -> Dict[str, Any]:
        """Run comprehensive market analysis with production-grade error handling"""

        try:
            # Input validation
            if not company_ticker or not company_ticker.strip():
                return self._create_error_result("Company ticker is required")
            
            company_ticker = company_ticker.upper().strip()

            # Ensure data availabilit
            data_available =  await self._ensure_market_data(company_ticker)
            if not data_available:
                return self._create_error_result(f"Insufficient market data available for {company_ticker}")
            
            self.logger.info(f"Starting market analysis for {company_ticker}")

            # Build comprehensive task with context
            task = self._build_market_analysis_task(company_ticker, additional_context)

            # Execute analysis with timeout protection
            try:
                result = await asyncio.wait_for(
                    self.team.run(task=task),
                    timeout=300 # 5-minute timeout
                )

            except asyncio.TimeoutError:
                self.logger.error(f"Market analysis timeout for {company_ticker}")
                return self._create_error_result("Market analysis timeout - process took too long")
            
            # Process and validate results
            analysis_result = self._process_market_result(result, company_ticker)
            self.logger.info(f"Successfully completed market analysis for {company_ticker}")
            return analysis_result

        except Exception as e:
            self.logger.error(f"Market analysis failed for {company_ticker}: {e}")
            return self._create_error_result(f"Market analysis failed: {str(e)}")
        
    def _build_market_analysis_task(self, company: str, additional_context: str) -> str:
        """Build comprehensive market analysis task with context"""

        base_task = f"""
        Perform comprehensive market analysis for {company} using our market intelligence pipeline.

        **Market Data Sources Available:**
        - Industry trends and growth analysis
        - Competitive landscape intelligence
        - Market opportunity assessments
        - Peer comparison and benchmarking
        - Innovation and emerging trend analysis

        **Market Analysis Framework:**
        1. INDUSTRY: Analyze market trends, growth drivers, and sector dynamics
        2. COMPETITIVE: Research competitive landscape and positioning
        3. OPPORTUNITY: Assess market opportunities and growth potential
        4. SYNTHESIS: Integrate findings into strategic market assessment

        **Analysis Quality Requirements:**
        - Cross-validate market data from multiple sources
        - Consider both current market position and future potential
        - Evaluate competitive threats and market opportunities
        - Provide specific, data-driven market recommendations

        **Additional Context:** {additional_context or "Standard comprehensive market analysis"}

        Coordinate as a market analysis team and ensure thorough documentation of the analysis process.
        """
        return base_task.strip()
    
    async def _ensure_market_data(self, company: str) -> bool:
        """Ensure market data is available with production-grade error handling"""

        try:
            if not company or not company.strip():
                self.logger.warning("Company ticker required for market data assurance")
                return False
            
            company = company.upper().strip()

            # Check existing market data with quality assessment
            market_queries = [
                f"{company} market analysis",
                f"{company} competitive landscape",
                f"{company} industry trends",
                f"{company} growth opportunities"
            ]

            market_docs_found = 0
            
            for query in market_queries:
                docs = self.rag_system.query(query, company=company, k=2)
                if docs:
                    market_docs_found += 1

            if market_docs_found >= 4:  # Require minimum market documents
                self.logger.info(f"Market data verified for {company}: {market_docs_found} documents")
                return True
            
            self.logger.warning(f"Insufficient market data for {company}: {market_docs_found} documents")
            return False
        
        except Exception as e:
            self.logger.error(f"Market data assurance failed for {company}: {e}")
            return False
        
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result for market operations"""

        return {
            'company': 'UNKNOWN',
            'error': error_message,
            'market_integration': False,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'success': False
        }
    
    def _process_market_summary(self, result, company: str) -> Dict[str, Any]:
        """Process and validate market analysis results"""

        try:
            summary = self._extract_market_summary(result.messages)

            return {
                'company': company,
                'messages': [msg.to_dict() for msg in result.messages],
                'stop_reason': result.stop_reason,
                'summary': summary,
                'market_integration': True,
                'data_source': 'Market Intelligence + Production RAG',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message_count': len(result.messages),
                'success': True
            }
        
        except Exception as e:
            self.logger.error(f"Market result processing failed for {company}: {e}")
            return self._create_error_result(f"Market result processing failed: {str(e)}")

    def _extract_market_summary(self, messages: str) -> str:
        """Extract meaningful market summary from team messages"""

        try:
            # Prioritize competitive analyst messages, then industry analyst, then market researcher
            for source in ['competitive_analyst', 'industry_analyst', 'market_researcher']:
                for msg in reversed(messages):
                    if hasattr(msg, 'source') and msg.source == source and hasattr(msg, 'content'):
                        content = msg.content.strip()

                    if len(content) > 50 and any(keyword in content.lower() for keyword in ['recommend', 'conclusion', 'summary', 'assessment', 'market', 'competitive']):
                        return f" {source}: {content[:500]}..."

            # Fallback: find most substantial market message   
            substantial_messages = []
            for msg in messages:
                if hasattr(msg, 'content') and len(msg.content.strip()) > 100:
                    substantial_messages.append((msg.source, msg.content))

            if substantial_messages:
                source, content = substantial_messages[:-1]
                return f" {source}: {content[:400]}..."

            return "Market analysis completed - review detailed messages for complete assessment"

        except Exception as e:
            self.logger.warning(f"Market summary extraction failed: {e}")
            return "Market analysis completed - summary extraction unavailable"

    async def close(self):
        """Clean up market resources with proper error handling"""

        try:
            if hasattr(self, 'model_client'):
                await self.model_client.close()
                self.logger.info("Market agent team resources cleaned up")
        
        except Exception as e:
            self.logger.error(f"Error during market cleanup: {e}")
            
    def _create_agents(self):
        """Create market analysis agents with specialized roles"""

        # Industry Analyst - Market trends and competitive landscape
        self.industry_analyst = AssistantAgent(
            name="industry_analyst",
            model_client=self.model_client,
            model_context=BufferedChatCompletionContext(buffer_size=15),
            tools = [self.analyze_industry_trends, self.research_competitive_landscape],
            system_message="""You are an Industry Analysis Specialist. Your responsibilities:

            1. **Market Trend Analysis**: Use analyze_industry_trends() to examine industry dynamics
            2. **Competitive Landscape**: Use research_competitive_landscape() to analyze competitors
            3. **Growth Pattern Analysis**: Identify market growth drivers and patterns
            4. **Sector Performance**: Evaluate industry sector performance and outlook

            **Key Focus Areas:**
            - Industry growth rates and market size
            - Competitive positioning and market share
            - Emerging trends and disruptions
            - Regulatory and macroeconomic impacts

            **Tools Available:**
            - analyze_industry_trends(company, industry): Analyze market trends and positioning
            - research_competitive_landscape(company, competitors): Research competitive environment

            Use 'INDUSTRY_ANALYSIS_COMPLETE' when industry analysis is finished.
            """
        )

        # Market Researcher - Industry reports and market intelligence
        self.market_researcher = AssistantAgent(
            name = "market_researcher",
            model_client=self.model_client,
            model_context=BufferedChatCompletionContext(buffer_size=12),
            tools = [self.assess_market_opportunities],
            system_message="""You are a Market Research Specialist. Your responsibilities:

            1. **Opportunity Assessment**: Use assess_market_opportunities() to identify growth areas
            2. **Market Intelligence**: Gather and analyze market data and reports
            3. **Segment Analysis**: Evaluate specific market segments and niches
            4. **Growth Forecasting**: Project market growth and opportunity sizing

            **Key Analysis Areas:**
            - Market opportunity identification and sizing
            - Segment growth potential and attractiveness
            - Customer needs and market gaps
            - Innovation and emerging market trends

            **Tools Available:**
            - assess_market_opportunities(company, segments): Evaluate growth opportunities

            Use 'MARKET_RESEARCH_COMPLETE' when market research is finished.
            """
        )

        # Competitive Analyst - Peer comparison and positioning
        self.competitive_analyst = AssistantAgent(
            name = "competitive_analyst",
            model_client = self.model_client,
            model_context=BufferedChatCompletionContext(buffer_size=10),
            tools = [self.research_competitive_landscape, self.analyze_industry_trends],
            system_message="""You are a Competitive Intelligence Specialist. Your responsibilities:

            1. **Competitive Benchmarking**: Compare company performance against peers
            2. **Positioning Analysis**: Evaluate competitive positioning and differentiation
            3. **Strategic Assessment**: Analyze competitor strategies and capabilities
            4. **Threat Analysis**: Identify competitive threats and market pressures

            **Review Checklist:**
            - Comprehensive competitor profiling
            - SWOT analysis (Strengths, Weaknesses, Opportunities, Threats)
            - Market share dynamics and trends
            - Strategic implications and recommendations

            **Tools Available:**
            - research_competitive_landscape(company, competitors): Analyze competitors
            - analyze_industry_trends(company, industry): Contextual industry analysis

            Use 'COMPETITIVE_ANALYSIS_COMPLETE' when competitive analysis is finished.
            """
        )

    def _create_team(self):
        """Create the market analysis team with robust termination conditions"""

        termination_conditions = (
            TextMentionTermination("MARKET_ANALYSIS_COMPLETE") |
            TextMentionTermination("INDUSTRY_ANALYSIS_COMPLETE") |
            TextMentionTermination("COMPETITIVE_ANALYSIS_COMPLETE") |
            TextMentionTermination("MARKET_RESEARCH_COMPLETE") |
            TextMentionTermination("TERMINATE") |
            MaxMessageTermination(max_messages=25)
        )

        self.team = RoundRobinGroupChat(
            [self.industry_analyst, self.market_researcher, self.competitive_analyst],
            termination_condition=termination_conditions,
            max_turns=20
        )

# Production-grade factory function for Market Agent Team
async def create_market_team(
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        rag_system: Optional[ProductionRAGSystem] = None,
        timeout: int = 30
) -> MarketAgentTeam:
    """Create production-grade market agent team with comprehensive setup"""

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
                persist_directory="./data/vector_stores/market_data", # Separate store for market docs
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

        return MarketAgentTeam(model_client, rag_system)
    
    except Exception as e:
        logging.error(f"Failed to create market agent team: {e}")
        raise




        










        
        
            







        


        
  


       


            



                


        



    
        
