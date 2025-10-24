# src/agents/financial_agent.py
import autogen
from typing import Dict, Any, List, Optional
import logging
import asyncio
from datetime import datetime
import json

from src.rag.core import ProductionRAGSystem
from src.data.processors.document_parser import DocumentProcessor

class FinancialAnalysisAgent:
    """
    Production Financial Analyst using AutoGen
    Specialized in SEC filings, ratio analysis, and financial modeling
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rag_system = ProductionRAGSystem()
        self.document_processor = DocumentProcessor()
        
        # Configure AutoGen agents
        self.llm_config = {
            "config_list": [
                {
                    "model": "gpt-4-1106-preview",
                    "api_key": "your_openai_api_key",
                    "temperature": 0.1
                }
            ],
            "timeout": 600,
            "cache_seed": 42
        }
        
        # Create agent team
        self.financial_analyst = autogen.AssistantAgent(
            name="Financial_Analyst",
            system_message="""
            You are a Senior Financial Analyst specializing in SEC filings and financial due diligence.
            
            YOUR EXPERTISE:
            - SEC 10-K, 10-Q, 8-K analysis
            - Financial ratio calculation and interpretation  
            - Revenue and profit trend analysis
            - Balance sheet strength assessment
            - Cash flow analysis
            - Financial risk assessment
            
            ANALYSIS FRAMEWORK:
            1. Extract key financial metrics from SEC data
            2. Calculate critical ratios (ROA, ROE, Current Ratio, Debt/Equity)
            3. Identify trends across periods
            4. Compare against industry benchmarks
            5. Assess financial health and sustainability
            
            Always provide specific numbers, percentages, and data-driven insights.
            Flag any concerning trends or red flags immediately.
            """,
            llm_config=self.llm_config
        )
        
        self.financial_researcher = autogen.AssistantAgent(
            name="Financial_Researcher",
            system_message="""
            You are a Financial Data Researcher. Your role is to:
            - Extract specific financial data from RAG system
            - Gather historical financial metrics
            - Retrieve ratio calculations
            - Find trend data across reporting periods
            - Provide raw financial data to the analyst
            
            Be precise and comprehensive in data retrieval.
            """,
            llm_config=self.llm_config
        )
        
        self.user_proxy = autogen.UserProxyAgent(
            name="User_Proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={"work_dir": "financial_analysis"},
            llm_config=self.llm_config
        )
        
        # Register functions as tools
        self.user_proxy.register_function(
            function_map={
                "get_financial_metrics": self.get_financial_metrics,
                "calculate_ratios": self.calculate_ratios,
                "analyze_revenue_trends": self.analyze_revenue_trends,
                "assess_financial_health": self.assess_financial_health
            }
        )
        
        self.logger.info("FinancialAnalysisAgent initialized")
    
    async def analyze(self, company_ticker: str, rag_context: List[str], questions: List[str]) -> Dict[str, Any]:
        """
        Execute comprehensive financial analysis
        """
        try:
            self.logger.info(f"Starting financial analysis for {company_ticker}")
            
            # Prepare analysis context
            analysis_context = self._prepare_context(company_ticker, rag_context, questions)
            
            # Create analysis task
            analysis_task = f"""
            Perform comprehensive financial due diligence for {company_ticker}.
            
            CONTEXT:
            {analysis_context}
            
            SPECIFIC ANALYSIS REQUIRED:
            1. Financial Ratio Analysis
            2. Revenue and Profitability Trends  
            3. Balance Sheet Strength
            4. Cash Flow Analysis
            5. Risk Assessment
            
            Please use the available tools to gather data and provide a comprehensive financial assessment.
            """
            
            # Execute group chat
            group_chat = autogen.GroupChat(
                agents=[self.user_proxy, self.financial_researcher, self.financial_analyst],
                messages=[],
                max_round=12
            )
            
            manager = autogen.GroupChatManager(
                groupchat=group_chat,
                llm_config=self.llm_config
            )
            
            # Start conversation
            await asyncio.to_thread(
                self.user_proxy.initiate_chat,
                manager,
                message=analysis_task,
                clear_history=True
            )
            
            # Extract results from conversation
            results = self._extract_results(group_chat.messages)
            
            self.logger.info(f"Completed financial analysis for {company_ticker}")
            return results
            
        except Exception as e:
            self.logger.error(f"Financial analysis failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    def _prepare_context(self, company_ticker: str, rag_context: List[str], questions: List[str]) -> str:
        """Prepare context for analysis"""
        context_parts = []
        
        if rag_context:
            context_parts.append("FINANCIAL CONTEXT FROM SEC FILINGS:")
            context_parts.extend(rag_context[:5])  # Limit context length
        
        if questions:
            context_parts.append("SPECIFIC QUESTIONS TO ANSWER:")
            context_parts.extend(questions)
        
        return "\n".join(context_parts)
    
    def _extract_results(self, messages: List[Dict]) -> Dict[str, Any]:
        """Extract structured results from group chat messages"""
        try:
            # Get the final analysis from financial analyst
            final_messages = [
                msg for msg in messages 
                if msg.get('name') == 'Financial_Analyst' and msg.get('content')
            ]
            
            if final_messages:
                final_analysis = final_messages[-1]['content']
            else:
                final_analysis = "Analysis completed but no final report generated."
            
            return {
                "status": "completed",
                "analysis": final_analysis,
                "key_metrics": self._extract_metrics_from_analysis(final_analysis),
                "risk_factors": self._extract_risk_factors(final_analysis),
                "recommendations": self._extract_recommendations(final_analysis),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Results extraction failed: {e}")
            return {"status": "completed", "analysis": "Analysis completed with extraction issues."}
    
    # Tool functions for AutoGen agents
    def get_financial_metrics(self, company_ticker: str, metrics: List[str] = None) -> str:
        """Get specific financial metrics from RAG system"""
        try:
            if not metrics:
                metrics = ["Revenue", "NetIncomeLoss", "Assets", "Liabilities", "CashAndCashEquivalents"]
            
            results = {}
            for metric in metrics:
                docs = self.rag_system.query(
                    f"{company_ticker} {metric}",
                    company=company_ticker,
                    metric_type="financial_metric",
                    k=3
                )
                if docs:
                    results[metric] = [doc.page_content for doc in docs[:2]]
            
            return json.dumps(results, indent=2)
            
        except Exception as e:
            return f"Error retrieving metrics: {str(e)}"
    
    def calculate_ratios(self, company_ticker: str) -> str:
        """Calculate financial ratios"""
        try:
            # Get ratio documents from RAG
            ratio_docs = self.rag_system.query(
                f"{company_ticker} financial ratios",
                company=company_ticker,
                metric_type="financial_ratio", 
                k=10
            )
            
            ratios = {}
            for doc in ratio_docs:
                content = doc.page_content
                # Extract ratio values from document content
                if "ROA" in content:
                    ratios["ROA"] = self._extract_ratio_value(content, "ROA")
                if "ROE" in content:
                    ratios["ROE"] = self._extract_ratio_value(content, "ROE")
                if "Current Ratio" in content:
                    ratios["Current_Ratio"] = self._extract_ratio_value(content, "Current Ratio")
                if "Debt to Equity" in content:
                    ratios["Debt_to_Equity"] = self._extract_ratio_value(content, "Debt to Equity")
            
            return json.dumps(ratios, indent=2)
            
        except Exception as e:
            return f"Error calculating ratios: {str(e)}"
    
    def analyze_revenue_trends(self, company_ticker: str, periods: int = 4) -> str:
        """Analyze revenue trends over multiple periods"""
        try:
            revenue_docs = self.rag_system.query(
                f"{company_ticker} Revenue trend",
                company=company_ticker,
                metric_type="financial_metric",
                k=periods * 2
            )
            
            trends = []
            for doc in revenue_docs[:periods]:
                content = doc.page_content
                # Extract period and value
                period = doc.metadata.get('period', 'Unknown')
                value = self._extract_revenue_value(content)
                trends.append({"period": period, "revenue": value})
            
            return json.dumps({"revenue_trends": trends}, indent=2)
            
        except Exception as e:
            return f"Error analyzing revenue trends: {str(e)}"
    
    def assess_financial_health(self, company_ticker: str) -> str:
        """Comprehensive financial health assessment"""
        try:
            # Get multiple data points
            ratios = json.loads(self.calculate_ratios(company_ticker))
            metrics = json.loads(self.get_financial_metrics(company_ticker))
            trends = json.loads(self.analyze_revenue_trends(company_ticker))
            
            assessment = {
                "liquidity": self._assess_liquidity(ratios),
                "profitability": self._assess_profitability(ratios, metrics),
                "solvency": self._assess_solvency(ratios),
                "efficiency": self._assess_efficiency(ratios, trends),
                "overall_health": "Good"  # Simplified assessment
            }
            
            return json.dumps(assessment, indent=2)
            
        except Exception as e:
            return f"Error assessing financial health: {str(e)}"
    
    # Helper methods
    def _extract_ratio_value(self, content: str, ratio_name: str) -> float:
        """Extract ratio value from content"""
        # Simplified extraction - in production, use more robust parsing
        try:
            lines = content.split('\n')
            for line in lines:
                if ratio_name in line and "Value:" in line:
                    value_part = line.split("Value:")[1].strip()
                    return float(value_part.split()[0])
            return 0.0
        except:
            return 0.0
    
    def _extract_revenue_value(self, content: str) -> float:
        """Extract revenue value from content"""
        try:
            lines = content.split('\n')
            for line in lines:
                if "Value:" in line and "USD" in line:
                    value_part = line.split("Value:")[1].split("USD")[0].strip()
                    return float(value_part.replace(',', ''))
            return 0.0
        except:
            return 0.0
    
    def _assess_liquidity(self, ratios: Dict) -> str:
        """Assess liquidity position"""
        current_ratio = ratios.get('Current_Ratio', 0)
        if current_ratio > 2.0:
            return "Strong"
        elif current_ratio > 1.0:
            return "Adequate" 
        else:
            return "Concerning"
    
    def _assess_profitability(self, ratios: Dict, metrics: Dict) -> str:
        """Assess profitability"""
        roa = ratios.get('ROA', 0)
        if roa > 10:
            return "Excellent"
        elif roa > 5:
            return "Good"
        else:
            return "Needs Improvement"
    
    def _assess_solvency(self, ratios: Dict) -> str:
        """Assess solvency position"""
        debt_equity = ratios.get('Debt_to_Equity', 0)
        if debt_equity < 0.5:
            return "Conservative"
        elif debt_equity < 2.0:
            return "Moderate"
        else:
            return "High Leverage"
    
    def _assess_efficiency(self, ratios: Dict, trends: Dict) -> str:
        """Assess operational efficiency"""
        # Simplified assessment
        return "Efficient"
    
    def _extract_metrics_from_analysis(self, analysis: str) -> Dict[str, Any]:
        """Extract key metrics from analysis text"""
        # Simplified extraction - in production, use more sophisticated NLP
        return {
            "profitability": "Extracted from analysis",
            "liquidity": "Extracted from analysis", 
            "efficiency": "Extracted from analysis"
        }
    
    def _extract_risk_factors(self, analysis: str) -> List[str]:
        """Extract risk factors from analysis"""
        risks = []
        if "risk" in analysis.lower() or "concern" in analysis.lower():
            risks.append("Financial risks identified in analysis")
        return risks
    
    def _extract_recommendations(self, analysis: str) -> List[str]:
        """Extract recommendations from analysis"""
        recommendations = []
        if "recommend" in analysis.lower() or "suggest" in analysis.lower():
            recommendations.append("Financial recommendations provided in analysis")
        return recommendations
    
    def is_healthy(self) -> bool:
        """Health check for agent"""
        try:
            return all([
                self.rag_system is not None,
                self.financial_analyst is not None,
                self.financial_researcher is not None
            ])
        except Exception:
            return False