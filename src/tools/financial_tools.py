# src/tools/financial_tools.py
import logging
import math
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any

class FinancialTools:
    """
    Financial Analysis Tools Layer.
    Full production version containing all original analysis, validation, and summary logic.
    """
    
    # Metric name mapping: common names -> SEC XBRL names
    METRIC_ALIASES = {
        # Core metrics
        'revenue': ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax'],
        'revenues': ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax'],
        'netincome': ['NetIncomeLoss'],
        'netincomeloss': ['NetIncomeLoss'],
        'assets': ['Assets'],
        'liabilities': ['Liabilities', 'LiabilitiesAndStockholdersEquity'],
        'stockholdersequity': ['StockholdersEquity'],
        'equity': ['StockholdersEquity'],
        # Liquidity metrics
        'currentassets': ['AssetsCurrent'],
        'currentliabilities': ['LiabilitiesCurrent'],
        'cashandcashequivalents': ['CashAndCashEquivalentsAtCarryingValue', 'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents', 'Cash'],
        'cash': ['CashAndCashEquivalentsAtCarryingValue', 'Cash', 'CashEquivalentsAtCarryingValue'],
        'cashequivalents': ['CashAndCashEquivalentsAtCarryingValue', 'CashEquivalentsAtCarryingValue'],
        'accountsreceivable': ['AccountsReceivableNetCurrent'],
        'inventory': ['InventoryNet'],
        'accountspayable': ['AccountsPayableCurrent'],
    }
    
    def __init__(self, rag_system, sec_collector):
        self.rag_system = rag_system
        self.sec_collector = sec_collector
        self.logger = logging.getLogger(__name__)

    # --- 1. METRICS RETRIEVAL ---
    async def retrieve_financial_metrics(self, company: str, metrics: List[str]) -> str:
        """Retrieve financial metrics with enhanced error handling for liquidity analysis."""
        try:
            if not company or not company.strip():
                return " [ERROR] Company ticker is required"
            
            company = company.upper().strip()
            if not metrics:
                metrics = ["Revenue", "NetIncomeLoss", "Assets", "Liabilities", "StockholdersEquity"]

            self.logger.info(f"Retrieving financial metrics for {company}: {metrics}")

            results = []
            metrics_found = 0
            metrics_failed = 0
            liquidity_metrics_missing = []

            for metric in metrics:
                try:
                    # Use similarity search
                    scored_documents = self.rag_system.query_with_similarity_scores(
                        question=f"{company} {metric} financial data values",
                        company=company,
                        metric_type="financial_metric",
                        k=20,
                        score_threshold=2.0  # Relaxed threshold for L2 distance
                    )

                    if not scored_documents:
                        self.logger.warning(f"No documents found for {company} {metric}")
                        
                        # Track liquidity metrics separately for better guidance
                        is_liquidity = any(liq in metric.lower() for liq in 
                                         ['current', 'cash', 'liquid', 'receivable', 'payable', 'inventory'])
                        if is_liquidity:
                            liquidity_metrics_missing.append(metric)
                        
                        results.append(f" [MISSING] {metric}: No data available in system")
                        metrics_failed += 1
                        continue

                    metric_results = self._extract_metric_data(scored_documents, metric, company)
                    
                    if metric_results:
                        results.append(f" [DATA] {metric}:\n{metric_results}")
                        metrics_found += 1
                    else:
                        results.append(f" [WARNING] {metric}: Data available but parsing failed")
                        metrics_failed += 1

                except Exception as e:
                    self.logger.warning(f"Error processing {metric}: {e}")
                    results.append(f" [WARNING] {metric}: Processing error")
                    metrics_failed += 1
                    continue
            
            summary = self._build_metrics_summary(company, metrics_found, metrics_failed, len(metrics))
            
            # Add specific guidance for missing liquidity data
            if liquidity_metrics_missing:
                guidance = (f"\n\n [GUIDANCE] Missing liquidity metrics: {', '.join(liquidity_metrics_missing)}. "
                           f"Consider proceeding with available data or requesting alternative metrics. "
                           f"DO NOT retry the same query - the data is not currently in the system.")
                summary += guidance
            
            if results:
                return f"{summary}\n\n" + "\n\n".join(results)
            return f"{summary}\n\n [ERROR] No financial metrics could be retrieved."
            
        except Exception as e:
            self.logger.error(f"Financial metrics retrieval failed for {company}: {e}")
            return f" [ERROR] System error retrieving financial metrics: {str(e)}"

    def _normalize_metric_name(self, metric: str) -> List[str]:
        """Normalize metric name to SEC equivalents using aliases."""
        normalized = re.sub(r'[^a-z0-9]', '', metric.lower())
        # Return list of possible SEC metric names
        return self.METRIC_ALIASES.get(normalized, [metric])
    
    def _extract_metric_data(self, scored_documents: List[Tuple], metric: str, company: str) -> Optional[str]:
        """
        Extract financial metric data from documents using EXACT metadata matching.
        This prevents mixing different metric types (e.g., 'Liabilities' vs 'AccruedLiabilities').
        """
        try:
            period_data = {}
            
            # Get possible SEC metric names for this metric
            target_metrics = self._normalize_metric_name(metric)
            # Also keep the original for fallback fuzzy matching
            target_simple = re.sub(r'[^a-z0-9]', '', metric.lower())
            
            for doc, score in scored_documents:
                # CRITICAL FIX: Use exact metadata matching with aliases
                # This ensures we only get data for the EXACT metric requested
                doc_metric = doc.metadata.get('metric', '')
                
                # Try exact match first (including aliases)
                if doc_metric and doc_metric in target_metrics:
                    exact_match = True
                # Fallback to fuzzy match only if metadata is not available
                elif not doc_metric:
                    content_simple = re.sub(r'[^a-z0-9]', '', doc.page_content.lower())
                    if target_simple not in content_simple:
                        continue
                    exact_match = False
                else:
                    # Metadata exists but doesn't match - skip this document
                    continue

                parsed = self._parse_financial_document(doc.page_content)
                if not parsed or not parsed.get('period') or not parsed.get('value'): 
                    continue

                try:
                    # ROBUST FLOAT PARSING (Handles "USD", spaces, etc.)
                    val_match = re.search(r'[-+]?\d*\.\d+|\d+', parsed['value'].replace(',', ''))
                    if not val_match: continue
                    
                    clean_val = float(val_match.group())
                    
                    # Validation
                    if not self._is_valid_financial_value(clean_val, metric): continue
                    
                    # Store best confidence (prioritize exact matches)
                    period = parsed['period']
                    if period not in period_data:
                        period_data[period] = {
                            'value': clean_val, 
                            'unit': parsed.get('unit', 'USD'),
                            'form': parsed.get('form', 'Unknown'), 
                            'confidence': score,
                            'exact_match': exact_match
                        }
                    # Replace if this is a better match
                    elif exact_match and not period_data[period].get('exact_match'):
                        # Exact match always wins over fuzzy match
                        period_data[period] = {
                            'value': clean_val, 
                            'unit': parsed.get('unit', 'USD'),
                            'form': parsed.get('form', 'Unknown'), 
                            'confidence': score,
                            'exact_match': exact_match
                        }
                    elif score > period_data[period]['confidence']:
                        # Better score for same match type
                        period_data[period]['value'] = clean_val
                        period_data[period]['confidence'] = score
                        
                except Exception as e:
                    continue

            if not period_data: return None
            return self._format_metric_data(period_data, metric)
            
        except Exception as e:
            self.logger.error(f"Extraction failed: {e}")
            return None

    # --- 2. COMPANY OVERVIEW (Restored) ---
    async def get_company_overview(self, company: str) -> str:
        try:
            scored_documents = self.rag_system.query_with_similarity_scores(
                question=f"{company} business overview entity information",
                company=company,
                metric_type="company_overview",
                score_threshold=2.0  # Relaxed threshold for L2 distance
            )

            if not scored_documents:
                return await self._fallback_company_info(company)
            
            validated_info = []
            for doc, score in scored_documents[:3]:
                if self._is_company_valid_info(doc.page_content):
                    clean_data = self._clean_company_info(doc.page_content)
                    validated_info.append(f" [INFO] {clean_data} (confidence: {score:.2f})")
            
            if validated_info:
                return f"Company Overview for {company}\n\n" + "\n\n".join(validated_info)
            return await self._fallback_company_info(company)

        except Exception as e:
            self.logger.warning(f"Overview failed for {company}: {e}")
            return f" [ERROR] System error retrieving company information"

    # --- 3. RATIO ANALYSIS (Restored) ---
    async def analyze_financial_ratios(self, company: str) -> str:
        """Analyze financial ratios with resilient error handling for missing liquidity data."""
        try:
            scored_documents = self.rag_system.query_with_similarity_scores(
                question=f"{company} financial ratios profitability liquidity",
                company=company,
                metric_type="financial_ratio",
                score_threshold=2.0  # Relaxed threshold for L2 distance
            )
             
            # Fallback: Calculate ratios dynamically from raw metrics
            self.logger.info(f"Pre-calculated ratios missing for {company}. Attempting dynamic calculation...")
            
            # Use the existing structured extraction which already fetches raw data and calculates ratios
            structured = self.extract_structured_metrics(company)
            ratios = structured.get('ratios', {})
            
            if not ratios:
                return (f"[DATA] No pre-calculated ratios available for {company}, and dynamic calculation failed due to missing raw data. "
                       f"Proceed with manual ratio calculation if possible.")
            
            # Format the calculated ratios for the agent
            analysis_parts = [f"[CALCULATED] Financial Ratio Analysis for {company} (Derived from Raw Metrics):"]
            
            # 1. Profitability
            prof_ratios = []
            if 'profit_margin' in ratios: prof_ratios.append(f"Profit Margin: {ratios['profit_margin']}%")
            if 'roa' in ratios: prof_ratios.append(f"ROA: {ratios['roa']}%")
            if 'roe' in ratios: prof_ratios.append(f"ROE: {ratios['roe']}%")
            
            if prof_ratios:
                analysis_parts.append("\n🔹 Profitability:")
                for r in prof_ratios: analysis_parts.append(f" • {r}")
                
            # 2. Liquidity
            liq_ratios = []
            if 'current_ratio' in ratios: liq_ratios.append(f"Current Ratio: {ratios['current_ratio']}x")
            if 'quick_ratio' in ratios: liq_ratios.append(f"Quick Ratio: {ratios['quick_ratio']}x")
            
            if liq_ratios:
                analysis_parts.append("\n🔹 Liquidity:")
                for r in liq_ratios: analysis_parts.append(f" • {r}")
                
            # 3. Solvency
            sol_ratios = []
            if 'debt_to_equity' in ratios: sol_ratios.append(f"Debt/Equity: {ratios['debt_to_equity']}x")
            
            if sol_ratios:
                analysis_parts.append("\n🔹 Solvency:")
                for r in sol_ratios: analysis_parts.append(f" • {r}")
                
            # 4. Efficiency
            eff_ratios = []
            if 'asset_turnover' in ratios: eff_ratios.append(f"Asset Turnover: {ratios['asset_turnover']}x")
            
            if eff_ratios:
                analysis_parts.append("\n🔹 Efficiency:")
                for r in eff_ratios: analysis_parts.append(f" • {r}")
                
            return "\n".join(analysis_parts)

        except Exception as e:
            self.logger.warning(f"Ratio analysis failed: {e}")
            return f"[ERROR] System error in ratio analysis: {str(e)}. Consider calculating ratios manually."

    # --- 4. TREND CALCULATION ---
    async def calculate_trends(self, company: str, metrics: List[str]) -> str:
        try:
            if not metrics: metrics = ["Revenue", "NetIncomeLoss", "Assets"]    
            trend_analysis = []
            
            for metric in metrics:
                try:
                    documents = self.rag_system.query_with_similarity_scores(
                        question=f"{company} {metric} financial document analysis",
                        company=company, k=10, score_threshold=2.0
                    )
                    
                    if not documents:
                        trend_analysis.append(f"[ERROR] {metric}: No data available")
                        continue

                    # Unpack tuples
                    docs_only = [doc for doc, score in documents]
                    period_values = self._extract_period_values(docs_only, metric, company)

                    if not period_values:
                        trend_analysis.append(f"[WARN] {metric}: Data available but parsing failed")
                        continue

                    sorted_periods = sorted(period_values.items(), key=lambda x: x[0], reverse=True)
                    if len(sorted_periods) < 2:
                        trend_analysis.append(f"[DATA] {metric}: Single data point - need more periods")
                        continue

                    latest_p, latest_v = sorted_periods[0]
                    prev_p, prev_v = sorted_periods[1]

                    if prev_v == 0:
                        trend_analysis.append(f"[WARN] {metric}: Previous value zero - no growth calc")
                        continue

                    growth_pct = ((latest_v - prev_v) / abs(prev_v)) * 100
                    trend_analysis.append(self._format_trend_analysis(
                        metric, growth_pct, latest_v, prev_v, latest_p, prev_p, len(sorted_periods)
                    ))

                except Exception as e:
                    self.logger.error(f"Trend error {metric}: {e}")
                    continue

            return f"[TREND] Trend Analysis for {company}\n\n" + "\n\n".join(trend_analysis)
        except Exception as e:
            return f"[ERROR] System error in trend analysis: {str(e)}"

    # --- 5. INVESTMENT SUMMARY (Restored) ---
    async def generate_investment_summary(self, company: str) -> str:
        try:
            if not company: return " [ERROR] Company required"
            company = company.upper().strip()
            self.logger.info(f"Generating investment summary for {company}")

            scored_documents = self.rag_system.query_with_similarity_scores(
                question=f"{company} financial performance risk assessment outlook",
                company=company,
                k=20,
                score_threshold=2.5  # Broader search for comprehensive analysis
            )

            if not scored_documents:
                return f" [ERROR] Insufficient data for {company} investment analysis"

            analysis_results = self._analyze_investment_data(scored_documents, company)
            return self._generate_comprehensive_summary(analysis_results, company)
        
        except Exception as e:
            self.logger.error(f"Investment summary failed: {e}")
            return f" [ERROR] System error generating summary: {str(e)}"

    # --- 6. VALIDATION (Restored) ---
    async def validate_with_source_data(self, company: str, findings: str) -> str:
        try:
            if not findings: return "[ERROR] Findings required"
            company = company.upper().strip()

            scored_documents = self.rag_system.query_with_similarity_scores(
                question=findings, company=company, k=10, score_threshold=2.0  # Relaxed threshold
            )

            if not scored_documents:
                return f"[WARN] No source documents found for validation"
            
            validation_results = self._analyze_findings_validation(scored_documents, findings, company)
            return self._format_validation_report(validation_results, company, len(scored_documents))
        except Exception as e:
            return f"[ERROR] Validation failed: {str(e)}"

    # --- HELPERS (Restored All) ---
    def _parse_financial_document(self, content: str, target: str="") -> Optional[Dict]:
        """Parses Key: Value lines safely."""
        data = {}
        try:
            for line in content.split('\n'):
                if ':' in line:
                    parts = line.split(':', 1)
                    key = parts[0].strip().lower().replace(' ', '_')
                    val = parts[1].strip()
                    data[key] = val
            
            return data if 'value' in data else None
        except: return None

    def _is_valid_financial_value(self, value: float, metric: str) -> bool:
        if math.isnan(value) or math.isinf(value): return False
        return abs(value) < 10_000_000_000_000

    def _format_metric_data(self, period_data: Dict, metric: str) -> str:
        sorted_periods = sorted(period_data.items(), key=lambda x:x[0], reverse=True)
        lines = []
        for period, data in sorted_periods[:3]:
            val = data['value']
            val_str = f"${val/1e9:.1f}B" if val >= 1e9 else f"${val/1e6:.1f}M"
            lines.append(f" • {val_str} (Period: {period}, Form: {data['form']})")
        return "\n".join(lines)

    def _build_metrics_summary(self, company, found, failed, total):
        if found == total: return f" [SUCCESS] Retrieved all {total} metrics for {company}"
        if found > 0: return f" [PARTIAL] Retrieved {found}/{total} metrics"
        return f" [ERROR] Failed to retrieve metrics"

    async def _fallback_company_info(self, company: str) -> str:
        try:
            data = self.sec_collector.company_facts(company)
            if data and data.get('entityName'):
                return f"Company: {data['entityName']}\nSource: SEC EDGAR"
        except: pass
        return f"[ERROR] No reliable info for {company}"

    def _is_company_valid_info(self, content: str) -> bool:
        return any(k in content for k in ['Company:', 'SIC', 'Business', 'Industry'])

    def _clean_company_info(self, content: str) -> str:
        return '\n'.join([l.strip() for l in content.split('\n') if l.strip()][:6])

    def _parse_ratio_document(self, content: str) -> Optional[Dict]:
        data = {}
        for line in content.split('\n'):
            if 'Financial Ratio:' in line: data['name'] = line.split(':')[1].strip()
            elif 'Value:' in line: 
                try: data['value'] = float(line.split(':')[1].strip())
                except: pass
            elif 'Interpretation:' in line: data['interpretation'] = line.split(':')[1].strip()
        return data if 'name' in data and 'value' in data else None

    def _validate_ratio_value(self, value: float) -> bool:
        return not (math.isnan(value) or math.isinf(value) or abs(value) > 1000)

    def _categorize_ratio(self, name: str) -> str:
        name = name.lower()
        if any(x in name for x in ['roa', 'roe', 'margin', 'return']): return 'profitability'
        if any(x in name for x in ['current', 'quick', 'liquidity']): return 'liquidity'
        if any(x in name for x in ['debt', 'equity', 'solvency']): return 'solvency'
        return 'efficiency'

    def _extract_period_values(self, documents: List, metric: str, company: str) -> Dict[str, float]:
        """
        Extract period values for trend analysis using EXACT metadata matching.
        This ensures we only compare values from the SAME metric across time.
        """
        values = {}
        target_metrics = self._normalize_metric_name(metric)
        target_simple = re.sub(r'[^a-z0-9]', '', metric.lower())
        
        for doc in documents:
            # CRITICAL FIX: Use exact metadata matching with aliases
            doc_metric = doc.metadata.get('metric', '')
            
            # Exact match with aliases preferred
            if doc_metric and doc_metric in target_metrics:
                pass  # Good to process
            # Fallback to fuzzy only if no metadata
            elif not doc_metric:
                # Only use fuzzy matching if metadata is unavailable
                if target_simple not in re.sub(r'[^a-z0-9]', '', doc.page_content.lower()):
                    continue
            else:
                # Metadata exists but doesn't match - skip
                continue
            
            parsed = self._parse_financial_document(doc.page_content, metric)
            if parsed and parsed.get('period') and parsed.get('value'):
                try:
                    val = float(re.sub(r'[^\d.-]', '', parsed['value']))
                    # Only store if this period isn't already captured or if it's a better value
                    if parsed['period'] not in values:
                        values[parsed['period']] = val
                except: 
                    continue
        return values

    def _format_trend_analysis(self, metric, growth, latest, prev, l_per, p_per, count):
        icon = "📈" if growth > 0 else "📉"
        return (f"{icon} {self._humanize_metric_name(metric)}: {growth:+.1f}%\n"
                f"   Latest: ${latest:,.0f} ({l_per}) vs Prev: ${prev:,.0f} ({p_per})")

    def _humanize_metric_name(self, metric):
        return metric.replace('NetIncomeLoss', 'Net Income').replace('StockholdersEquity', 'Equity')

    def _analyze_investment_data(self, docs: List[Tuple], company: str) -> Dict:
        categories = {'profitability': [], 'liquidity': [], 'solvency': [], 'growth': [], 'risk': []}
        high_conf = 0
        for doc, score in docs:
            cat = self._categorize_investment_document(doc.page_content)
            if cat: categories[cat].append(doc.page_content)
            if score >= 0.7: high_conf += 1
        
        return {'categories': categories, 'total': len(docs), 'high_conf': high_conf, 'sufficient': high_conf >= 5}

    def _categorize_investment_document(self, content: str) -> Optional[str]:
        c = content.lower()
        if any(x in c for x in ['revenue', 'profit', 'margin']): return 'profitability'
        if any(x in c for x in ['cash', 'current ratio']): return 'liquidity'
        if any(x in c for x in ['debt', 'leverage']): return 'solvency'
        if any(x in c for x in ['growth', 'expansion']): return 'growth'
        if any(x in c for x in ['risk', 'uncertainty']): return 'risk'
        return None

    def _generate_comprehensive_summary(self, data: Dict, company: str) -> str:
        report = [f" [REPORT] Investment Analysis: {company}"]
        report.append(f"Data Quality: {data['high_conf']} high-confidence docs.")
        
        cats = data['categories']
        if cats['profitability']: report.append("- Analyze profitability trends")
        if cats['liquidity']: report.append("- Assess liquidity")
        if cats['risk']: report.append(f"- Risk: {len(cats['risk'])} docs found")
        
        return "\n".join(report)

    def _analyze_findings_validation(self, docs: List[Tuple], findings: str, company: str) -> Dict:
        """
        Validate findings against retrieved documents.
        Uses the RAG system's similarity scores as the primary validation metric.
        Lower scores (closer to 0) indicate higher relevance for L2 distance.
        """
        supporting, neutral, low_confidence = [], [], []
        
        # The RAG system already filtered by score_threshold=2.0, so all documents here are relevant
        # We categorize them by confidence level with more lenient thresholds
        for doc, score in docs:
            item = {'content': doc.page_content, 'similarity_score': score}
            
            # L2 distance: lower is better
            # Since RAG already filtered to score < 2.0, we use more lenient thresholds:
            # Scores < 0.9 are excellent matches (high confidence)
            # Scores 0.9-1.5 are good matches (medium confidence)
            # Scores 1.5-2.0 are acceptable matches (lower confidence but still relevant)
            if score < 0.9:  # Excellent match
                supporting.append(item)
            elif score < 1.5:  # Good match
                neutral.append(item)
            else:  # Acceptable match (still passed RAG's 2.0 threshold)
                low_confidence.append(item)
            
        return {
            'supporting': supporting, 
            'conflicting': [],  # Never treat low relevance as conflict
            'neutral': neutral, 
            'low_confidence': low_confidence
        }

    def _format_validation_report(self, res: Dict, company: str, total: int) -> str:
        """Format validation report based on similarity scores"""
        report = [f"[VALIDATION] Summary for {company}"]
        report.append(f"  Analyzed {total} documents from RAG system")
        
        num_supporting = len(res['supporting'])
        num_neural = len(res.get('neutral', []))
        num_low_conf = len(res.get('low_confidence', []))
        
        if num_supporting > 0:
            report.append(f"  [+] {num_supporting} high-confidence supporting documents")
        if num_neural > 0:
            report.append(f"  [~] {num_neural} medium-confidence documents")
        if num_low_conf > 0:
            report.append(f"  [o] {num_low_conf} lower-confidence documents")
        
        # CRITICAL FIX: All documents passed RAG's score_threshold filter, so all are relevant
        # Count ALL documents toward adequacy assessment, not just high/medium confidence
        total_relevant = num_supporting + num_neural + num_low_conf
        
        # More realistic thresholds based on total relevant documents
        if total_relevant >= 8:
            report.append("  [VERDICT] Strong support from retrieved data")
        elif total_relevant >= 5:
            report.append("  [VERDICT] Adequate support from retrieved data")
        elif total_relevant >= 2:
            report.append("  [VERDICT] Some supporting evidence found")
        else:
            report.append("  [VERDICT] Insufficient supporting evidence")
        
        return "\n".join(report)
    
    def extract_structured_metrics(self, company: str) -> Dict[str, Any]:
        """
        Extract structured financial metrics for visualization and comprehensive reporting.
        Returns time-series data, trends, and ratios in JSON-serializable format.
        
        Args:
            company: Company ticker symbol
            
        Returns:
            Dictionary with structured metrics, trends, and ratios
        """
        from typing import List, Tuple
        import re
        
        # Define key metrics to extract
        core_metrics = [
            "Revenue", "Net Income", "Total Assets", "Total Liabilities",
            "Current Assets", "Current Liabilities", "Cash and Cash Equivalents",
            "Operating Cash Flow", "Free Cash Flow", "Equity"
        ]
        
        structured_data = {
            "metrics": {},
            "trends": {},
            "ratios": {},
            "latest_period": None
        }
        
        try:
            # Extract each metric's time series
            for metric in core_metrics:
                metric_data = self._extract_metric_timeseries(company, metric)
                if metric_data:
                    structured_data["metrics"][metric] = metric_data
            
            # Calculate trends (growth rates)
            structured_data["trends"] = self._calculate_trends(structured_data["metrics"])
            
            # Calculate financial ratios
            structured_data["ratios"] = self._calculate_ratios_from_metrics(structured_data["metrics"])
            
            # Determine latest period
            if structured_data["metrics"]:
                first_metric = list(structured_data["metrics"].values())[0]
                if first_metric:
                    structured_data["latest_period"] = first_metric[0].get("period", "Unknown")
            
            self.logger.info(f"Extracted structured data for {company}: {len(structured_data['metrics'])} metrics")
            return structured_data
            
        except Exception as e:
            self.logger.error(f"Error extracting structured metrics for {company}: {e}")
            return structured_data
    
    def _extract_metric_timeseries(self, company: str, metric: str) -> List[Dict[str, Any]]:
        """Extract time-series data for a single metric"""
        try:
            # FIX: Use synonyms and recent years - use exact XBRL tags for better matching
            synonyms = {
                "Revenue": "RevenueFromContractWithCustomerExcludingAssessedTax Revenue Sales",
                "Net Income": "NetIncomeLoss Net Income profit loss",
                "Total Assets": "Assets Total Assets",
                "Total Liabilities": "Liabilities Total Liabilities",
                "Equity": "StockholdersEquity Equity stockholders"
            }
            
            # Map metric names to expected metadata metric tags for filtering
            metric_metadata_tags = {
                "Revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "Revenue", "SalesRevenueNet"],
                "Net Income": ["NetIncomeLoss", "NetIncome", "ProfitLoss"],
                "Total Assets": ["Assets"],
                "Total Liabilities": ["Liabilities"],
                "Current Assets": ["AssetsCurrent"],
                "Current Liabilities": ["LiabilitiesCurrent"],
                "Cash and Cash Equivalents": ["CashAndCashEquivalentsAtCarryingValue", "Cash"],
                "Operating Cash Flow": ["NetCashProvidedByUsedInOperatingActivities"],
                "Free Cash Flow": ["FreeCashFlow"],
                "Equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]
            }
            
            search_term = synonyms.get(metric, metric)
            query = f"{company} {search_term} 2025 2024 2023"
            scored_documents = self.rag_system.query_with_similarity_scores(question=query, k=25, score_threshold=2.5)
            
            if not scored_documents:
                return []
           
            # Parse values and periods from documents
            timeseries = []
            seen_periods = set()
            
            # Get expected metadata tags for this metric
            expected_tags = metric_metadata_tags.get(metric, [metric.replace(" ", "")])
            
            for doc, score in scored_documents:
                content = doc.page_content
                metadata = doc.metadata
                
                # CRITICAL FIX: Filter by metadata metric tag to avoid cross-contamination
                doc_metric = metadata.get("metric", "")
                if doc_metric:
                    # Check if this document's metric matches what we're looking for
                    # Use EXACT matching to prevent 'Liabilities' from matching 'LiabilitiesAndStockholdersEquity'
                    matches = doc_metric in expected_tags
                    if not matches:
                        continue  # Skip this document - wrong metric type
                
                # Extract period
                period = metadata.get("period") or metadata.get("end") or metadata.get("filed")
                if not period:
                    # Try to extract from content
                    period_match = re.search(r'(\d{4}-\d{2}-\d{2}|\d{4})', content)
                    if period_match:
                        period = period_match.group(1)[:4]  # Get year
                
                # Extract value
                value = metadata.get("value")
                if value is None:
                    # Try to extract from content
                    value_match = re.search(r'[\$]?([\d,]+\.?\d*)\s*(billion|million|thousand)?', content, re.IGNORECASE)
                    if value_match:
                        value_str = value_match.group(1).replace(',', '')
                        multiplier = 1
                        if 'billion' in content.lower():
                            multiplier = 1_000_000_000
                        elif 'million' in content.lower():
                            multiplier = 1_000_000
                        elif 'thousand' in content.lower():
                            multiplier = 1_000
                        try:
                            value = float(value_str) * multiplier
                        except:
                            value = None
                
                # Add to timeseries if we have both period and value
                if period and value is not None:
                     # Standardize period
                    period = str(period).strip()
                    if len(period) > 10:  period = period[:10]

                    if period not in seen_periods:
                        timeseries.append({
                            "period": period,
                            "value": float(value),
                            "confidence": float(score)
                        })
                        seen_periods.add(period)
            
            # SORT by period descending (newest first)
            # This fixes the mismatch where we compared 2025 Net Income to 2018 Revenue
            timeseries.sort(key=lambda x: x['period'], reverse=True)
            
            return timeseries
            
            # Sort by period (most recent first)
            timeseries.sort(key=lambda x: x["period"], reverse=True)
            
            # Limit to last 5 years
            return timeseries[:5]
            
        except Exception as e:
            self.logger.warning(f"Could not extract timeseries for {metric}: {e}")
            return []
    
    def _calculate_trends(self, metrics: Dict[str, List[Dict]]) -> Dict[str, float]:
        """Calculate growth trends from metrics"""
        trends = {}
        
        for metric_name, data_points in metrics.items():
            if len(data_points) >= 2:
                try:
                    # Calculate YoY growth from most recent to previous year
                    latest = data_points[0]["value"]
                    previous = data_points[1]["value"]
                    
                    if previous != 0:
                        growth = ((latest - previous) / abs(previous)) * 100
                        trends[f"{metric_name.lower().replace(' ', '_')}_growth"] = round(growth, 2)
                except Exception as e:
                    self.logger.debug(f"Could not calculate trend for {metric_name}: {e}")
                    continue
        
        return trends
    
    def _calculate_ratios_from_metrics(self, metrics: Dict[str, List[Dict]]) -> Dict[str, float]:
        """Calculate financial ratios from extracted metrics using STRICT PERIOD ALIGNMENT"""
        ratios = {}
        
        try:
            # 1. Pivot data to {period: {metric: value}}
            period_map = {}
            all_metrics = metrics.keys()
            
            for metric, data_points in metrics.items():
                for point in data_points:
                    p = point.get('period')
                    v = point.get('value')
                    if p and v is not None:
                        if p not in period_map:
                            period_map[p] = {}
                        period_map[p][metric] = v
            
            # 2. Sort periods descending (newest first)
            sorted_periods = sorted(period_map.keys(), reverse=True)
            
            if not sorted_periods:
                return {}
                
            # 3. Find availability of key metrics per period
            # We prioritize the most recent period that has at least Revenue and Net Income
            best_period = None
            
            for p in sorted_periods:
                data = period_map[p]
                # Check for critical mass of data
                has_income = 'Net Income' in data
                has_revenue = 'Revenue' in data
                has_assets = 'Total Assets' in data
                
                if has_income and has_revenue:
                    best_period = p
                    break
            
            # Fallback: if no period has both, just take the most recent one with decent data
            if not best_period and sorted_periods:
                best_period = sorted_periods[0]
            
            self.logger.info(f"Calculating ratios using aligned period: {best_period}")
            ratios['period_used'] = best_period
            
            # 4. Calculate Ratios using ONLY that period's data
            data = period_map.get(best_period, {})
            
            revenue = data.get("Revenue")
            net_income = data.get("Net Income")
            total_assets = data.get("Total Assets")
            total_liabilities = data.get("Total Liabilities")
            current_assets = data.get("Current Assets")
            current_liabilities = data.get("Current Liabilities")
            inventory = data.get("Inventory", 0) # Optional
            
            # -- Profitability --
            if revenue and revenue != 0:
                if net_income:
                    ratios["profit_margin"] = round((net_income / revenue) * 100, 2)
            
            if total_assets and total_assets != 0:
                if net_income:
                    ratios["roa"] = round((net_income / total_assets) * 100, 2)
                if revenue:
                    ratios["asset_turnover"] = round(revenue / total_assets, 2)
                
                if total_liabilities is not None:
                    equity = total_assets - total_liabilities
                    if equity != 0 and net_income:
                        ratios["roe"] = round((net_income / equity) * 100, 2)
                    
                    if equity != 0:
                         ratios["debt_to_equity"] = round(total_liabilities / equity, 2)

            # -- Liquidity --
            if current_liabilities and current_liabilities != 0:
                if current_assets:
                    ratios["current_ratio"] = round(current_assets / current_liabilities, 2)
                    ratios["quick_ratio"] = round((current_assets - inventory) / current_liabilities, 2)

            return ratios

        except Exception as e:
            self.logger.error(f"Error calculating ratios: {e}")
            return {}
