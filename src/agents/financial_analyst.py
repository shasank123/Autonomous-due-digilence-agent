# src/agents/financial_agent.py
import os
import asyncio
from typing import Dict, List, Optional, Any
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

class FinancialAgentTeam:
    """Financial Due Diligence Agent Team integrated with our RAG system"""
    def __init__(self, model_client: OpenAIChatCompletionClient, rag_system: ProductionRAGSystem):
        self.model_client = model_client
        self.rag_system = rag_system
        self.sec_collector = SECDataCollector()
        self.document_parser = DocumentProcessor()
        self.team = None
        self._create_agents()
        self._create_team()

    def _create_agents(self):
        """Financial Due Diligence Agent Team integrated with our RAG system"""

        # Financial Researcher - Uses our RAG system for data retrieval
        self.financial_researcher = AssistantAgent(
            name="financial_researcher",
            model_client=self.model_client,
            model_context=BufferedChatCompletionContext(buffer_size=15),
            tools = [self.retrieve_financial_metrics, self.get_company_overview],
            system_message="""You are a Financial Research Specialist. Your responsibilities:

        1. **Financial Data Retrieval**: Use retrieve_financial_metrics() to get SEC financial data
        2. **Company Overview**: Use get_company_overview() for business context
        3. **Trend Analysis**: Analyze revenue, profit, and growth trends
        4. **Data Quality**: Verify data completeness and reliability

        **Key Focus Areas:**
        - Revenue trends and growth drivers from SEC data
        - Profitability metrics analysis
        - Financial statement completeness
        - Data consistency across periods

        **Tools Available:**
        - retrieve_financial_metrics(company, metrics): Get specific financial metrics
        - get_company_overview(company): Get company business information

        Use 'RESEARCH_COMPLETE' when data collection is finished.
        """       
        )

        # Financial Analyst - Uses RAG data for quantitative analysis
        self.financial_analyst = AssistantAgent(
            name="financial_analyst",
            model_client=self.model_client,
            model_context=BufferedChatCompletionContext(buffer_size=12),
            tools=[self.analyze_financial_ratios, self.calculate_trends],
            ssystem_message="""You are a Quantitative Financial Analyst. Your responsibilities:

        1. **Ratio Analysis**: Use analyze_financial_ratios() to examine profitability, liquidity, efficiency
        2. **Trend Calculation**: Use calculate_trends() for growth rate analysis
        3. **Benchmarking**: Compare ratios against industry standards
        4. **Performance Assessment**: Evaluate financial health and stability

        **Key Analysis Areas:**
        - Profitability: ROA, ROE, Margins
        - Liquidity: Current Ratio, Quick Ratio
        - Efficiency: Asset Turnover, Inventory Turnover
        - Solvency: Debt-to-Equity, Interest Coverage

        **Tools Available:**
        - analyze_financial_ratios(company): Get and analyze financial ratios
        - calculate_trends(company, metrics): Calculate growth trends

        Use 'ANALYSIS_COMPLETE' when quantitative analysis is finished.
        """
        )
        
        # Financial Reviewer - Validates and synthesizes findings
        self.financial_reviewer = AssistantAgent(
            name="financial_reviewer",
            model_client=self.model_client,
            model_context=BufferedChatCompletionContext(buffer_size=10),
            tools=[self.validate_with_source_data, self.generate_investment_summary],
            system_message="""You are a Senior Financial Reviewer. Your responsibilities:

        1. **Validation**: Use validate_with_source_data() to cross-check findings
        2. **Synthesis**: Integrate research and analysis into comprehensive assessment
        3. **Risk Assessment**: Evaluate financial risks and opportunities
        4. **Recommendations**: Provide data-driven investment recommendations

        **Review Checklist:**
        - Data consistency with SEC source documents
        - Logical reasoning in ratio analysis
        - Comprehensive risk assessment
        - Clear investment thesis supported by data

        **Tools Available:**
        - validate_with_source_data(company, findings): Cross-check with original SEC data
        - generate_investment_summary(company): Create final investment recommendation

        Use 'REVIEW_COMPLETE' when review and synthesis are finished.
        """
        )

    # RAG-Integrated Tool Functions
    async def retrieve_financial_metrics(self, company: str, metrics: List[str]) -> str:
                                 
             """Retrieve comprehensive financial metrics with full context"""
             try:
                if not metrics:
                    metrics = ["Revenue", "NetIncomeLoss", "Assets", "Liabilities", "StockholdersEquity"]

                results = []
                for metric in metrics:
                    # Query for specific metrics
                    documents = self.rag_system.query(
                        question=f"{company}{metric} financial performance",
                        company=company,
                        metric_type="financial_metric",
                        k=5
                    )

                    if documents:
                        metric_periods = []
                        for doc in documents[:3]:
                            if metric in doc.page_content:
                                parsed_data = self._parse_financial_document(doc.page_content, metric)
                                if parsed_data:
                                    metric_periods.append(parsed_data)

                        if metric_periods:
                            # Sort by period (most recent first)
                            metric_periods.sort(key=lambda x:x.get('period', ''), reverse=True)

                            period_outputs = []
                            for data in metric_periods[:3]:
                                period_outputs.append(
                                    f" - {data['value']} {data.get('unit', 'USD')}",
                                    f"(Period: {data.get('period', 'Unknown')})",
                                    f" Filed: {data.get('filed', 'Unknown')}"
                                )
                            results.append(f"📊 {company}\n" + "\n".join(period_outputs))
                        else:
                            results.append(f"❌ No structured data found for {metric}")
                    
                    else:
                        results.append(f"❌ No documents found for {metric}") 

                return f"Financial Metrics for {company}:\n\n" + "\n\n".join(results)
             
             except Exception as e:
                 return f"Error retrieving financial metrics {str(e)}" 
         
        
    def _parse_financial_document(self, content: str, target_metric: str) -> Dict[str, str]:
        """Parse structured financial document content"""
        try:
            lines = content.split('\n')
            data = {}
            
            for line in lines:
                line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()

                    if key == 'company':
                        data['company'] = value
                    elif key == 'financial_metric':
                        data['metric'] = value
                    elif key == 'value':
                        data['value'] = value.split()[0]
                        if 'USD' in value:
                            data['unit'] = 'USD'
                    elif key == 'Period End':
                        data['period'] = value
                    elif key == 'Filed Date':
                        data['filed'] = value
                    elif key == 'Form Type':
                        data['form'] = value

            # Only return if it matches our target metric and has essential data
            if (data.get('metric') == target_metric and 'value' in data and 'period' in data):
                return data
            
        except Exception as e:
            print(f"Error parsing document: {e}")
            return None

    async def get_company_overview(self, company: str) -> str:
        """Get company business overview and entity information"""
        try:
            documents = self.rag_system.query(
                question=f"{company} business overview company information",
                company=company,
                metric_type="company_info"
            )

            if documents:
                overview = []
                for doc in documents:
                    overview.append(f"📋 {doc.page_content}")
                
                return f"Company Overview for {company}:\n" + "\n".join(overview)           
            else:
                # Fallback: Try to get company info from SEC directly
                company_data = self.sec_collector.company_facts(company)
                if company_data and 'entityName' in company_data:
                    return f"Company: {company_data['entityName']} \nticker: {company} \nData Source: SEC"
                else:
                    return f"No company overview found for {company}"
                
        except Exception as e:
            return f"Error getting company overview: {str(e)}"
        
    async def analyze_financial_ratios(self, company: str) -> str:
        """Analyze financial ratios from our RAG system"""
        try:
            # Get ratio documents
            documents = self.rag_system.query(
                question=f"{company}financial ratios performance metrics",
                company=company,
                metric_type="financial_ratio"
            )
            
            if documents:
                ratio_analysis = []
                for doc in documents:
                    ratio_analysis.append(f"📈 {doc.page_content}")

                return f"Financial Ratio Analysis for {company}\n" + "\n".join(ratio_analysis)
            
            else:
                return f"No financial ratio data found for {company}. Ratios may need to be calculated from raw metrics."
        
        except Exception as e:
            return f"Error analyzing financial ratios: {str(e)}"
        
            

            


        
                           

                        

                
            

                        


        

        