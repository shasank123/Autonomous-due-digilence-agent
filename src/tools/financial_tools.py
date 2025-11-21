import logging
import math
from typing import List, Dict, Optional, Tuple, Any
from langchain_core.documents import Document

class FinancialTools:
    """Financial Analysis Tools"""
    
    def __init__(self, rag_system, sec_collector):
        self.rag_system = rag_system
        self.sec_collector = sec_collector
        self.logger = logging.getLogger("FinancialTools")

    async def retrieve_financial_metrics(self, company: str, metrics: List[str]) -> str:
        """Retrieve and validate financial metrics with robust error handling"""
        try:
            # validate input
            if not company or not company.strip():
                return f"❌ Company ticker is required"
            
            company = company.upper().strip()

            if not metrics:
                metrics = ["Revenue", "NetIncomeLoss", "Assets", "Liabilities", "StockholdersEquity"]

            else:
                # Validate metrics list
                metrics = [m for m in metrics if m or m.strip()]
                if not metrics:
                    return f"❌ No valid metrics provided"
                
            self.logger.info(f"Retrieving financial metrics for {company}: {metrics}")

            results = []
            metrics_found = 0
            metrics_failed = 0

            for metric in metrics:
                try:
                    # Use similarity search with scores for quality control
                    scored_documents = self.rag_system.query_with_similarity_scores(
                        question=f"{company} {metric} financial data values",
                        company=company,
                        metric_type="financial_metric",
                        k=5,
                        score_threshold=0.4
                    )

                    if not scored_documents:
                        self.logger.warning(f"No documents found for {company} {metric}")
                        results.append(f"❌ {metric}: No data available in system")
                        metrics_failed=+1
                        continue

                    # Extract and validate metric data
                    metric_results = self._extract_metric_data(scored_documents, metric, company)
                    
                    if metric_results:
                        results.append(f"📊 {metric}\n: {metric_results}")
                        metrics_found +=1

                    else:
                        results.append(f"⚠️ {metric}: Data available but parsing failed")
                        metrics_failed +=1

                except Exception as e:
                    self.logger.warning(f"⚠️ {metric}: Data available but parsing failed")
                    results.append(f"⚠️ {metric}: Processing error")
                    continue
            
            # Build final response with summary
            summary = self._build_metrics_summary(company, metrics_found, metrics_failed, len(metrics))

            if results:
                return f"{summary}\n\n" + "\n\n".join(results)
            else:
                return f"{summary}\n\n❌ No financial metrics could be retrieved for {company}"
            
        except Exception as e:
            self.logger.error(f"Financial metrics retrieval failed for {company}: {e}")
            return f"❌ System error retrieving financial metrics: {str(e)}"
        
    def _extract_metric_data(self, scored_documents: List[Tuple], metric: str, company: str) -> str:
        """Extract and validate metric data from documents"""
        try:
            period_data = {}
            for doc, score in scored_documents:
                try:
                    # Skip if document doesn't contain our target metric
                    if metric not in doc.page_content:
                        continue

                    # Parse structured financial data
                    parsed_data = self._parse_financial_document(doc.page_content, metric)
                    if not parsed_data:
                        continue

                    period = parsed_data.get('period')
                    value_str = parsed_data.get('value')
                    unit = parsed_data.get('unit', 'USD')

                    # Validate required fields
                    if not period or not value_str:
                        continue

                    # Clean and convert value
                    try:
                        clean_value = float(value_str.replace(',',''))

                        # Validate value is reasonable
                        if not self._is_valid_financial_value(clean_value,metric):
                            self.logger.warning(f"Invalid value for {metric}: {clean_value}")
                            continue

                        # Use the highest confidence value for each period
                        if period not in period_data or score > period_data[period]['confidence']:
                            period_data[period] = {
                                'value': clean_value,
                                'unit': unit,
                                'confidence': score,
                                'form': parsed_data.get('form', 'Unknown')
                            }

                    except (ValueError, TypeError) as conversion_error:
                        self.logger.warning(f"Value conversion failed for {metric}: {value_str}")
                        continue
                
                except Exception as doc_error:
                    self.logger.warning(f"Document processing failed for {metric}: {doc_error}")
                    continue

            if not period_data:
                return None
            
            # Sort by period (most recent first) and format output
            return self._format_metric_data(period_data, metric)
        
        except Exception as e:
            self.logger.error(f"Metric data extraction failed for {metric}: {e}")
            return None
        
    def _parse_financial_document(self, content: str, target_metric: str) -> Dict[str, str]:
        """Parse structured financial document with validation"""
        try:
            lines = content.split('\n')
            data = {}

            for line in lines:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()

                    if key == 'company':
                        data['company'] = value
                    elif key == 'financial_metric':
                        data['metric'] = value
                    elif key == 'value':
                        # Extract numeric value and unit
                        value_parts = value.split()
                        if value_parts:
                            data['value'] = value_parts[0]
                            if len(value_parts) > 1 and value_parts[1] in ['USD', 'EUR', 'GBP']:
                                data['unit'] = value_parts[1]
                            elif key == 'Period End':
                                data['period'] = value
                            elif key == 'Filed Date':
                                data['filed'] = value
                            elif key == 'Form Type':
                                data['form'] = value

            # Validate we have the target metric and essential data
            if(data.get('metric') == target_metric and
               data.get('value') and
               data.get('period')):
               return data
            
            return None
        
        except Exception as e:
            self.logger.warning(f"Document parsing failed: {e}")
            return None

    def _is_valid_financial_value(self, value: float, metric: str) -> bool:
        """Validate financial values are within reasonable ranges"""
        
        if math.isnan(value) or math.isinf(value):
            return False
        
        # Metric-specific validation
        validation_rules = {
             'Revenue' : (0, 10_000_000_000_000),
             'NetIncomeLoss': (-1_000_000_000_000, 1_000_000_000_000),  # Can be negative
             'Assets': (0, 10_000_000_000_000),
             'Liabilities': (0, 10_000_000_000_000),
             'StockholdersEquity': (-1_000_000_000_000, 10_000_000_000_000)  # Can be negative

         }
        
        min_value, max_value = validation_rules.get('metric', (0, 10_000_000_000_000))
        return min_value <= value <= max_value
    
    def _format_metric_data(self, period_data: Dict, metric: str) -> str:
        """Format metric data for readable output"""
        try:
            # Sort periods chronologically (most recent first)
            sorted_periods = sorted(period_data.items(), key=lambda x:x[0], reverse=True)

            formatted_lines = []
            for period, data in sorted_periods[:3]:
                value = data['value']
                unit = data['unit']
                confidence = data['confidence']
                form = data['form']

                # Format value based on size
                if value >= 1_000_000_000:
                    display_value = f"${value/1_000_000_000:,.1f}B"
                elif value >= 1_000_000:
                    display_value = f"${value/1_000_000:,.1f}M"
                else:
                    display_value = f"${value:,.0f}"

                formatted_lines.append(
                     f" • {display_value}{unit} (period: {period},"
                     f"Form: {form} Confidence: {confidence:.2f}"
                )

            return "\n".join(formatted_lines)

        except Exception as e:
            self.logger.error(f"Metric formatting failed: {e}")   
            return "• [Data formatting error]" 
        
    def _build_metrics_summary(self, company: str, found: int, failed: int, total: int) -> str:
        """Build summary of metrics retrieval results"""
        success_rate = (found/total) * 100 if total > 0 else 0

        if found == total:
            return f"✅ Successfully retrieved all {total} financial metrics for {company}"
        elif found > 0:
            return f"📊 Retrieved {found}/{total} metrics for {company} ({success_rate:.2f}% success)"
        else:
            return f"❌ Failed to retrieve financial metrics for {company}"
    
    async def get_company_overview(self, company: str) -> str:
        """Get validated company business overview"""
        try:
            # Use similarity search with scores for quality control
            scored_documents = self.rag_system.query_with_similarity_scores(
                question=f"{company} business overview entity information",
                company=company,
                metric_type="company_overview",
                score_threshold=0.6 # Only high-confidence matches
            )

            if not scored_documents:
                return await self._fallback_company_info(company)
            
            # Process and validate documents
            validated_info = []
            for doc, score in scored_documents[:3]:
                try:
                    # Validate document has meaningful content
                    if self._is_company_valid_info(doc.page_content):
                        clean_data = self._clean_company_info(doc.page_content)
                        validated_info.append(f"📋 {clean_data} (confidence: {score:.2f})")
                
                except Exception as doc_error:
                    self.logger.warning(f"Invalid company info document: {doc_error}")
                    continue

            if validated_info:
                return f"Company Overview for {company}\n\n" + "\n\n".join(validated_info)    
            else:
                return await self._fallback_company_info(company)
            
        except Exception as e:
            self.logger.warning(f"Company overview failed for {company}: {e}")
            return f"❌ System error retrieving company information"

    async def analyze_financial_ratios(self, company: str) -> str:
        """Analyze financial ratios with data validation"""
        try:
             # Get ratios with confidence scores
             scored_documents = self.rag_system.query_with_similarity_scores(
                 question=f"{company} financial ratios profitability liquidity",
                 company=company,
                 metric_type="financial_ratio",
                 score_threshold=0.5
             )
             
             if not scored_documents:
                 return f"📊 No validated ratio data found for {company}. Consider calculating from raw metrics."
             
             # Categorize and validate ratios
             ratio_categories = {
                 'profitability': [],
                'liquidity': [],
                'efficiency': [],
                'solvency': []
                }
             
             for doc, score in scored_documents:
                 try:
                     ratio_data = self._parse_ratio_document(doc.page_content)
                     if ratio_data and self._validate_ratio_value(ratio_data['value']):
                         category = self._categorize_ratio(ratio_data['name'])
                         if category in ratio_categories:
                             ratio_categories[category].append((ratio_data, score))

                 except Exception as ratio_error:
                     self.logger.warning(f"Invalid ratio document: {ratio_error}")
                     continue
                 
             # Build structured analysis
             analysis_parts = [f"📈 Financial Ratio Analysis for {company}:"]

             for category, ratios in ratio_categories.items():
                 if ratios:
                     analysis_parts.append(f"\n🔹 {category.title()}:")
                     for ratio_data, score in ratios[:3]:
                         analysis_parts.append(
                             f" • {ratio_data['name']}: {ratio_data['value']:.2f}"
                             f"({ratio_data.get('interpretation', 'N/A')}) "
                             f"[confidence: {score:.2f}]"
                         )
                        
             if len(analysis_parts) > 1:
                 return "\n".join(analysis_parts)
             
             else:
                return f"📊 No validated ratios available for {company}"
             
        except Exception as e:
            self.logger.warning(f"Ratio analysis failed for {company}: {e}")
            return f"❌ System error in ratio analysis"  

    def _is_company_valid_info(self, content: str) -> bool:
        """Validate company info document has meaningful data"""
        required_keywords = ['Company:', 'SIC', 'Business', 'Industry']
        return any(keyword in content for keyword in required_keywords)
        
    def _clean_company_info(self, content: str) -> str:
        """Clean and format company information"""
        lines = content.split('\n')
        clean_lines = [line.strip() for line in lines if line.strip()]
        return '\n'.join(clean_lines[:6]) # Return first 6 meaningful lines
    
    def _parse_ratio_document(self, content: str) -> Optional[Dict]:
        """Parse ratio document with validation"""
        try:
            lines = content.split('\n')
            ratio_data = {}

            for line in lines:
                if 'Financial Ratio:' in line:
                    ratio_data['name'] = line.split('Financial Ratio:')[1].strip()
                 
                elif 'value:' in line:
                    value_str = line.split('value:')[1].strip().split()[0]
                    ratio_data['value'] = float(value_str)

                elif 'interpretation:' in line:
                    ratio_data['interpretation'] = line.split('interpretation')[1].strip()

            return ratio_data if ratio_data.get('name') and 'value' in ratio_data else None
        
        except:
            return None
        
    def _validate_ratio_value(self, value: float) -> bool:
        """Validate ratio values are within reasonable ranges"""
        return not (math.isnan(value) or math.isinf(value) or abs(value) > 1000)
    
    def _categorize_ratio(self, ratio_name: str) -> str:
        """Categorize ratios for better organization"""

        profitability = ['ROA', 'ROE', 'Margin', 'Return']
        liquidity = ['Current', 'Quick', 'Liquidity']
        efficiency = ['Turnover', 'Efficiency']
        solvency = ['Debt', 'Equity', 'Solvency', 'Leverage']

        ratio_lower = ratio_name.lower()

        if any(term in ratio_lower for term in profitability):
            return 'profitability'
        if any(term in ratio_lower for term in liquidity):
            return 'liquidity'
        if any(term in ratio_lower for term in efficiency):
            return 'efficiency'
        if any(term in ratio_lower for term in solvency):
            return 'solvency'
        else:
            return 'other'
        
    async def _fallback_company_info(self, company: str) -> str:
        """Fallback method with proper error handling"""
        try:
            company_data = self.sec_collector.company_facts(company)
            if company_data and company_data.get('entityName'):
                return (
                    f"company: {company_data['entityName']}\n"
                    f"Ticker: {company}\n"
                    f"Souce: SEC EDGAR\n"
                    f"Note: Limited information available"
                )
            
        except Exception as e:
            self.logger.warning(f"Fallback company info failed: {e}")

        return f"❌ No reliable company information found for {company}"
    
    async def calculate_trends(self, company: str, metrics: List[str]) -> str:
         """Calculate growth trends with robust error handling and data validation"""
         try:
             if not metrics:
                 metrics = ["Revenue", "NetIncomeLoss", "Assets"]    

             trend_analysis = []
             for metric in metrics:
                try:
                    documents = self.rag_system.query_with_similarity_scores(
                        question=f"{company}{metric} financial document analysis",
                        company=company,
                        k=10
                    )

                    if not documents:
                        trend_analysis.append(f"❌ {metric}: No data available in system")
                    
                    # Note: query_with_similarity_scores returns (doc, score) tuples
                    # We need just documents for _extract_period_values if it expects list of docs
                    # But looking at _extract_period_values implementation in original file, it iterates over docs
                    # Let's adjust to pass just docs or handle tuples
                    
                    # Extract just documents from tuples
                    docs_only = [doc for doc, score in documents]

                    period_values = self._extract_period_values(docs_only, metric, company)

                    if not period_values:
                        trend_analysis.append(f"⚠️ {metric}: Data available but parsing failed")
                        continue

                    sorted_periods = sorted(period_values.items(), key=lambda x: x[0], reverse=True)
                    if len(sorted_periods) < 2:
                        trend_analysis.append(f"📊 {metric}: Single data point ({sorted_periods[0][1]:,.0f}) - need more periods for trends")
                        continue

                    # Calculate growth between most recent periods
                    latest_period, latest_value = sorted_periods[0]
                    previous_period, previous_value = sorted_periods[1]

                    # Validate data quality
                    if previous_value == 0:
                        trend_analysis.append(f"⚠️ {metric}: Previous period value is zero - cannot calculate growth")
                        continue

                    if latest_value < 0 or previous_value < 0:
                         # Allow negative values for NetIncome but flag it
                         pass

                    # Calculate growth percentage
                    growth_pct = ((latest_value - previous_value) /abs(previous_value)) * 100

                    # Format trend analysis with context
                    trend_info = self._format_trend_analysis(
                        metric, growth_pct,latest_value, previous_value,latest_period, previous_period,len(sorted_periods)
                        )

                    trend_analysis.append(trend_info)

                except Exception as metric_error:
                    self.logger.error(f"Error processing metric {metric} for {company} : {metric_error}")
                    trend_analysis.append(f"⚠️ {metric}: Processing error - check data quality") 
                    continue

             if not trend_analysis:
                 return f"❌ No trend analysis possible for {company} - insufficient or invalid data"
             
             return f"📈 Trend Analysis for {company}\n\n" + "\n\n".join(trend_analysis)
         
         except Exception as e:
             self.logger.error(f"Trend calculation failed for{company}: {e}")  
             return f"❌ System error in trend analysis: {str(e)}"    
         
    def _extract_period_values(self, documents: List[Document], metric: str, company: str) -> Dict[str, float]:
        """Extract period-value pairs with robust parsing and validation"""
        period_values = {}

        for doc in documents:
            try:
                # Skip if document doesn't contain our target metric
                if metric not in doc.page_content:
                    continue
                # Parse the structured financial document
                parsed_data = self._parse_financial_document(doc.page_content, metric)
                if not parsed_data:
                    continue
                
                period = parsed_data.get('period')
                value_str = parsed_data.get('value')
                
                # Validate required fields
                if not period or not value_str:
                    continue
                # Clean and convert value
                try:
                    # Remove commas and convert to float
                    clean_value = float(value_str.replace(',', ''))
                    
                    # Check for duplicate periods - keep the most recent filing
                    if period in period_values:
                        existing_value = period_values[period]
                        # Log data consistency issue
                        if abs(clean_value-existing_value) > abs(existing_value*0.01):
                            self.logger.warning(f"Data inconsistency for {company} {metric}{period}: {existing_value} vs {clean_value}")
                        # Keep the larger value (typically more complete data)
                        period_values[period] = max(existing_value, clean_value)
                    
                    else:
                        period_values[period] = clean_value

                except (ValueError,TypeError) as conversion_error:
                    self.logger.warning(f"Value conversion failed for: {metric}: {value_str} - {conversion_error}")      
                    continue
            
            except Exception as doc_error:
                self.logger.warning(f"Document parsing failed for {metric}: {doc_error}")
                continue

        return period_values
    
    def _format_trend_analysis(self, metric: str, growth_pct: float, latest_value: float, previous_value: float, latest_period: str, previous_period: str, data_points: int) -> str:
        """Format trend analysis with proper financial context and insights"""

        # Human-readable metric name
        metric_display = self._humanize_metric_name(metric)

        # Growth trend indicator
        if growth_pct > 15:
            trend_icon = "🚀"
            trend_desc = "Strong growth"
        elif growth_pct > 5:
            trend_icon = "📈" 
            trend_desc = "Moderate growth"
        elif growth_pct > -5:
            trend_icon = "➡️"
            trend_desc = "Stable"
        elif growth_pct > -15:
            trend_icon = "📉"
            trend_desc = "Moderate decline"
        else:
            trend_icon = "🔻"
            trend_desc = "Significant decline"
        
        # Format values for readability
        def format_currency(value):
            if abs(value) >= 1_000_000_000:
                return f"${value/1_000_000_000:.1f}B"
            elif abs(value) >= 1_000_000:
                return f"${value/1_000_000:.1f}M"
            else:
                return f"${value:,.0f}"
            
        return f"""{trend_icon}{metric_display}:
        • Growth: {growth_pct:+.1f}% ({trend_desc})
        • Latest: {format_currency(latest_value)} {latest_period}
        • Previous: {format_currency(previous_value)} {previous_period}
        • Data Quality: {data_points} periods analyzed"""
    
    def _humanize_metric_name(self, metric: str) -> str:
        """Convert metric names to human-readable format"""

        metric_map = {
            'Revenue': 'Revenue',
            'NetIncomeLoss': 'Net Income',
            'Assets': 'Total Assets', 
            'Liabilities': 'Total Liabilities',
            'StockholdersEquity': "Shareholders' Equity",
            'CurrentAssets': 'Current Assets',
            'CurrentLiabilities': 'Current Liabilities'
        }

        return metric_map.get(metric, metric)
    
    async def validate_with_source_data(self, company: str, findings: str) -> str:
         """Validate analysis findings against source SEC data with confidence scoring"""
         try:
             # Input validation
             if not company or not company.strip():
                 return "❌ Company ticker is required for validation"
             if not findings or len(findings.strip()) < 10:
                 return "❌ Findings must contain meaningful content for validation"
             
             company = company.upper().strip()
             self.logger.info(f"Validating findings for {company}: {findings[:100]}...")

             # Use similarity search with scores for quality validation
             scored_documents = self.rag_system.query_with_similarity_scores(
                 question=findings,
                 company=company,
                 k=10,
                 score_threshold=0.3  # Lower threshold to catch potential conflicts

             )

             if not scored_documents:
                 self.logger.warning(f"No source documents found for validation of {company}")
                 return f"⚠️ No source documents found for {company} to validate findings"
             
             # Analyze document relevance and conflicts
             validation_results = self._analyze_findings_validation(scored_documents, findings, company)
             return self._format_validation_report(validation_results, company, len(scored_documents))
         
         except Exception as e:
             self.logger.error(f"Validation failed for {company}: {e}")
             return f"❌ System error during validation: {str(e)}"
         
    def _analyze_findings_validation(self, scored_documents: List[Tuple], findings: str, company: str) -> Dict[str, Any]:
        """Analyze document relevance and conflicts with findings"""
        supporting_docs = []
        conflicting_docs = []
        neutral_docs = []

        findings_terms = self._extract_key_terms(findings)

        for doc, score in scored_documents:
            try:
                relevance_score = self._calculate_relevance_score(doc.page_content, findings_terms)
                doc_type = doc.metadata.get('doc_type', 'unknown')
                metric = doc.metadata.get('metric', 'unknown')

                validation_doc = {
                    'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    'similarity_score': score,
                    'relevance_score': relevance_score,
                    'doc_type': doc_type,
                    'metric': metric,
                    'confidence': min(score, relevance_score) # Combined confidence

                }

                # Categorize based on relevance
                if relevance_score >= 0.7:
                    supporting_docs.append(validation_doc)
                elif relevance_score <= 0.3:
                    conflicting_docs.append(validation_doc)
                else:
                    neutral_docs.append(validation_doc)

            except Exception as doc_error:
                self.logger.warning(f"Document analysis failed: {doc_error}")
                continue

        return {
            'supporting': sorted(supporting_docs, key= lambda x:x['confidence'], reverse=True),
            'conflicting': sorted(conflicting_docs, key= lambda x:x.get('confidence', 0), reverse=True),
            'neutral': neutral_docs,
            'total_analyzed': len(scored_documents)
        }    
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract meaningful key terms from findings"""
        try:
            # Remove common stop words and short terms
            stop_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
            words = text.lower().split()
            key_terms = [
                word.strip('.,!?;:()[]{}"\'') 
                for word in words
                if len(word) > 3 and word not in stop_words                
            ]
            return list(set(key_terms))[:15] # Return top 15 unique terms
        
        except Exception as e:
            self.logger.warning(f"Key term extraction failed: {e}")
            return []
        
    def _calculate_relevance_score(self, content: str, key_terms: List[str]) -> float:
        """Calculate relevance score between content and key terms"""
        
        if not key_terms:
            return 0.0
        
        content_lower = content.lower()
        matches = sum(1 for term in key_terms if term in content_lower)
        # Normalize score between 0 and 1
        return min(matches/len(key_terms), 1.0)
    
    def _format_validation_report(self, validation_results: Dict[str, Any], company: str, total_docs: int) -> str:
        """Format comprehensive validation report"""
        supporting = validation_results['supporting']
        conflicting = validation_results['conflicting']
        neutral = validation_results['neutral']

        report_parts = [f"📋 Validation Report for {company}"]
        report_parts.append(f"Analyzed {total_docs} source documents")
        report_parts.append("")

        # Supporting evidence
        if supporting:
            report_parts.append(f"✅ Supporting Evidence ({len(supporting)} documents):")
            for doc in supporting[:3]:
                report_parts.append(
                    f"  • {doc['metric']} (Confidence: {doc['confidence']:.2f})"
                )
        else:
            report_parts.append("✅ No strong supporting evidence found")

        # Potential conflicts
        if conflicting:
            report_parts.append(f"")
            report_parts.append(f"⚠️  Potential Conflicts ({len(conflicting)} documents:)")
            
            for doc in conflicting[:2]:
                report_parts.append(
                    f" • {doc['metric']} (Confidence: {doc['confidence']:.2f})"
                )

        # Neutral documents
        if neutral and not supporting and not conflicting:
            report_parts.append(f"")
            report_parts.append(f"📊 Neutral Documents ({len(neutral)}):")
            report_parts.append(f"• Findings cannot be strongly validated or contradicted")

        # Overall assessment
        report_parts.append("")
        if supporting and not conflicting:
            report_parts.append(f"🎯 Assessment: Findings are well-supported by source data")
        elif conflicting and not supporting:
            report_parts.append(f"🎯 Assessment: Findings conflict with source data - review needed")
        elif supporting and conflicting:
            report_parts.append(f"🎯 Assessment: Mixed evidence - further analysis recommended")
        else:
            report_parts.append(f"🎯 Assessment: Insufficient evidence for validation")

        return "\n".join(report_parts)
    
    async def generate_investment_summary(self, company: str) -> str:
        """Generate data-driven investment summary with risk assessment"""
        try:
            if not company or not company.strip():
                return f"❌ Company ticker is required for investment summary"
            
            company = company.upper().strip()
            self.logger.info(f"Generating investment summary for {company}")

            # Get comprehensive data with quality filtering
            scored_documents = self.rag_system.query_with_similarity_scores(
                question=f"{company} financial performance risk assessment outlook",
                company=company,
                k=20, # More documents for comprehensive analysis
                score_threshold=0.4
            )

            if not scored_documents:
                return f"❌ Insufficient data available for {company} investment analysis"   
            
            # Analyze data quality and categories
            analysis_results = self._analyze_investment_data(scored_documents, company)

            if not analysis_results['has_sufficient_data']:
                return f"⚠️ Limited data available for {company}. Consider fetching latest SEC filings."
            
            return self._generate_comprehensive_summary(analysis_results, company)
        
        except Exception as e:
            self.logger.error(f"Investment summary generation failed for {company}: {e}")
            return f"❌ System error generating investment summary: {str(e)}"
        
    def _analyze_investment_data(self, scored_documents: List[Tuple], company: str) -> Dict[str, Any]:
        """Analyze investment data across categories"""
        categories = {
            'profitability': [],
            'liquidity': [],
            'solvency': [],
            'efficiency': [],
            'growth': [],
            'risk': []
        }   

        high_confidence_documents = 0

        for doc, score in scored_documents:
            try:
                doc_type = doc.metadata.get('doc_type')
                metric = doc.metadata.get('metric')

                doc_info = {
                    'content': doc.page_content,
                    'score': score,
                    'doc_type': doc_type,
                    'metric': metric
                }

                # Categorize document
                category = self._categorize_investment_document(doc_info)
                if category:
                    categories[category].append(doc_info)

                # Track high confidence documents
                if score >= 0.7:
                    high_confidence_documents +=1

            except Exception as doc_error:
                self.logger.warning(f"Document categorization failed: {doc_error}")
                continue

        return {
            'categories': categories,
            'total_documents': len(scored_documents),
            'high_confidence_docs': high_confidence_documents,
            'has_sufficient_data': len(high_confidence_documents) >= 5 and any(len(docs) >= 2 for docs in categories.values())
        }
    
    def _categorize_investment_document(self, doc_info: Dict[str, Any]) -> Optional[str]:
        """Categorize document for investment analysis"""
        content_lower = doc_info['content'].lower()
        metric_lower = str(doc_info['metric']).lower()

        # Profitability indicators
        if any(term in content_lower or term in metric_lower for term in ['revenue', 'income', 'profit', 'margin', 'roa', 'roe']):
            return 'profitability'
        
        # Liquidity indicators
        elif any(term in content_lower or term in metric_lower for term in ['current', 'quick', 'liquidity', 'cash']):
            return 'liquidity'
        
        # Solvency indicators
        elif any(term in content_lower or term in metric_lower for term in ['debt', 'equity', 'leverage', 'solvency']):
            return 'solvency'
        
        # Efficiency indicators
        elif any(term in content_lower or term in metric_lower for term in ['turnover', 'efficiency', 'asset']):
            return 'efficiency'
        
        # Growth indicators
        elif any(term in content_lower or term in metric_lower for term in ['growth', 'trend', 'increase', 'decrease']):
            return 'growth'
        
        # Risk indicators
        elif any(term in content_lower or term in metric_lower for term in ['risk', 'volatility', 'uncertainty']):
            return 'risk'
        
        return None
    
    def _generate_comprehensive_summary(self, analysis_results: Dict[str, Any], company: str) -> str:
        """Generate comprehensive investment summary"""
        categories = analysis_results['categories']

        summary_parts = [
            f"🏢 Comprehensive Investment Analysis: {company}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Data Quality: {analysis_results['high_confidence_documents']}/{analysis_results['total_documents']} high-confidence documents",
            ""
        ]

        # Data availability overview
        available_categories = [cat for cat, docs in categories.items() if docs]
        summary_parts.append(f"📊 Data Availability:")
        
        for category in available_categories:
            doc_count = len(categories[category])
            summary_parts.append(f"• {category.title()}: {doc_count} documents")
        
        summary_parts.append("")

        # Analysis recommendations
        summary_parts.append("🎯 Recommended Analysis Focus:")

        if categories['profitability']:
            summary_parts.append(f"• Analyze profitability trends and margins")
        if categories['liquidity']:
            summary_parts.append("  • Assess short-term financial health and liquidity")           
        if categories['solvency']:
            summary_parts.append("  • Evaluate long-term debt and capital structure")
        if categories['growth']:
            summary_parts.append("  • Examine revenue and income growth patterns")
               
        summary_parts.append("")

        # Risk assessment
        risk_docs = len(categories['risk'])
        
        if risk_docs > 0:
            summary_parts.append(f"⚠️  Risk Considerations: {risk_docs} risk-related documents identified")
        else:
            summary_parts.append(f"✅ No specific risk documents identified - conduct standard risk assessment")
       
        summary_parts.append("")
        
        # Next steps
        summary_parts.append("🚀 Next Steps:")
        summary_parts.append("  1. Use financial metrics tools for detailed analysis")
        summary_parts.append("  2. Calculate and interpret key financial ratios")
        summary_parts.append("  3. Validate findings against source SEC data")
        summary_parts.append("  4. Generate final investment recommendation")

        return "\n".join(summary_parts)
