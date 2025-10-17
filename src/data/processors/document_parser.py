from langchain_core.documents import Document
from typing import List, Dict
import json
import logging

class DocumentProcessor:
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
       
    """
    Processes ALL financial metrics and calculates ratios
    """

    def process_sec_facts(self, company_facts: Dict, ticker: str) -> List[Document]:
      
        # Converts ALL SEC financial data into comprehensive documents
        documents = []

        if not company_facts or 'facts' not in company_facts:       
            return documents
        
        # Extract company info
        company_info = self._extract_company_info(company_facts,ticker)
        documents.extend(company_info)

        # Process ALL financial metrics
        financial_docs = self._process_financial_metrics(company_facts,ticker)
        documents.extend(financial_docs)

        # Calculate and add financial ratios
        ratio_docs = self._compute_financial_ratios(company_facts,ticker)
        documents.extend(ratio_docs)

        print(f" created {len(documents)} comprehensive documents for {ticker}")
        return documents
    
    def _extract_company_info(self, company_facts: Dict, ticker: str) -> List[Document]:

        #Extract company entity information
        documents = []

        entity_info = {
            "name": "unknown",
            "sic": "unknown",
            "category": "unknown",
        }

        if 'entityName' in company_facts:
            entity_info['name'] = company_facts['entityName']

        if 'sic' in company_facts:
            entity_info['sic'] = company_facts['sic']

        if 'category' in company_facts:
            entity_info['category'] = company_facts['category']

        content = f"""
        company_name: {entity_info['name']} ({ticker})
        SIC Industry Code: {entity_info['sic']}
        Business Category: {entity_info['category']}
        Data Source: U.S. Securities and Exchange Commission (SEC)
        """
        document = Document(
            page_content=content.strip(),
            metadata={
                "source": "sec",
                "company": ticker,
                "doc_type": "company_info",
                "data_type": "entity_information"
            }
        )

        documents.append(document)
        return documents
    
    def _process_financial_metrics(self, company_facts: Dict, ticker: str) -> List[Document]:

        documents = []
        
        if 'us-gaap' not in company_facts.get('facts', {}):
            return documents
        
        financial_facts = company_facts['facts']['us-gaap']

        key_metrics = ['Revenue', 'Assets', 'Liabilities', 'NetIncomeLoss',
            'CashAndCashEquivalents', 'Inventory', 'PropertyPlantEquipment',
            'LongTermDebt', 'StockholdersEquity', 'EarningsPerShareBasic',
            'CostOfRevenue', 'OperatingIncomeLoss', 'ResearchAndDevelopmentExpense',
            'SalesRevenueGoodsNet', 'SalesRevenueServicesNet', 'GrossProfit',
            'InterestExpense', 'IncomeTaxExpenseBenefit', 'DepreciationDepletionAndAmortization']
        
        for metric in key_metrics:
            if metric in financial_facts:
                metric_data = financial_facts[metric]
                docs = self._create_metric_documents(metric, metric_data, ticker)
                documents.extend(docs)
        return documents
    
    def _create_metric_documents(self, metric: str, metric_data: str, ticker: str) -> List[Document]:
        #Create documents for a specific financial metric
        documents = []

        if 'units' not in metric_data:
            return documents
        
        for unit, values in metric_data['units'].items():

            # Take latest 3 periods for trend analysis
            for value in values[:3]:
                content = self._format_financial_content(metric, ticker, unit, value)

                document = Document(
                    page_content=content,
                    metadata={
                        "source": "sec",
                        "company": ticker,
                        "metric": metric,
                        "unit": unit,
                        "period": value.get('end', ''),
                        "filed": value.get('filed', ''),
                        "form": value.get('form', ''),
                        "doc_type": "financial_metric",
                        "data_type": "raw_financial"
                    } 
                )
                documents.append(document)

        return documents
    
    def _format_financial_content(self, metric: str, ticker: str, unit: str, value: Dict) -> str:
        #Format financial data into readable content
        return f"""
        Company: {ticker}
        Financial Metric: {self._humanize_metric_name(metric)}
        Value: {value.get('val', 'N/A')} {unit}
        Period End: {value.get('end', 'N/A')}
        Filed Date: {value.get('filed', 'N/A')}
        Form Type: {value.get('form', 'N/A')}
        Context: {value.get('frame', 'As Reported')}
        """
    
    def _humanize_metric_name(self, metric: str) -> str:
        #Convert metric names to human-readable format
        replacements = {
            'Revenue': 'Total Revenue',
            'NetIncomeLoss': 'Net Income/Loss',
            'CashAndCashEquivalents': 'Cash & Cash Equivalents',
            'PropertyPlantEquipment': 'Property, Plant & Equipment',
            'StockholdersEquity': "Stockholders' Equity",
            'EarningsPerShareBasic': 'Basic Earnings Per Share (EPS)',
            'ResearchAndDevelopmentExpense': 'Research & Development Expense'
        }

        return replacements.get(metric, metric.replace('_', '').title())

    
    def _compute_financial_ratios(self, company_facts: Dict, ticker: str) -> List[Document]:
        #Calculate and create documents for financial ratios
        documents = []
        
        # Extract latest values for ratio calculation
        financial_data = self._extract_latest_values(company_facts)
        #Calculate key financial ratios
        ratios = self._compute_ratios(financial_data)

        # Create ratio documents
        for ratio_name, ratio_value in ratios.items():
            if ratio_value is not None:
                content = f"""
                company = {ticker},
                Financial Ratio: {ratio_name}
                Value: {ratio_value:.2f}
                Interpretation: {self._get_ratio_interpretation(ratio_name, ratio_value)}
                Calculation Period: Latest Available Data
                """

                document = Document(
                    page_content= content.strip(),
                    meta_data = {
                        "source": "calculated",
                        "company": ticker, 
                        "metric": ratio_name,
                        "doc_type": "financial_ratio",
                        "data_type": "calculated_metric"
                    }
                )
                documents.append(document)

        return documents
    
    def _extract_latest_values(self, company_facts: Dict) -> Dict[str, float]:

        #Extract latest values for ratio calculations
        values = {}

        if 'us-gaap' not in company_facts.get('facts', {}):
            return values
        
        financial_facts = company_facts['facts']['us-gaap']

        # Map of metrics we need for ratios
        metric_map = {
            'Revenue': 'revenue',
            'NetIncomeLoss': 'net_income', 
            'Assets': 'total_assets',
            'Liabilities': 'total_liabilities',
            'StockholdersEquity': 'equity',
            'CashAndCashEquivalents': 'cash',
            'CurrentAssets': 'current_assets',
            'CurrentLiabilities': 'current_liabilities'
        }

        for metric_key, value_key in metric_map.items():
            if metric_key in financial_facts and 'USD' in financial_facts[metric_key].get('units',{}):
                usd_values= financial_facts[metric_key]['units']['USD']

                if usd_values:
                    values[value_key] = usd_values[0]['val']

        return values

    def _compute_ratios(self, financial_data: Dict[str, float]) -> Dict[str, float]:
        
        #Compute financial ratios from extracted data
        ratios = {}

        try:
            # Profitability Ratios
            if 'net_income' in financial_data and 'total_assets' in financial_data:
                if financial_data['total_assets'] != 0:
                    ratios['Return on Assets (ROA)'] = (financial_data['net_income'] / financial_data['total_assets']) * 100
                else:
                    self.logger.warning("Total assets is zero, skipping ROA calculation")
                                  
            
            if 'net_income' in financial_data and 'equity' in financial_data:
                if financial_data['equity'] != 0:
                    ratios['Return on Equity (ROE)'] = (financial_data['net_income'] / financial_data['equity']) * 100
                else:
                    self.logger.warning("Equity is zero, skipping ROE calculation")
        
            # Liquidity Ratios
            if 'current_assets' in financial_data and 'current_liabilities' in financial_data:
                if financial_data['current_liabilities'] != 0:
                    ratios['Current Ratio'] = financial_data['current_assets'] / financial_data['current_liabilities']
                else:
                    self.logger.warning("Current liabilities is zero, skipping Current Ratio calculation")
        
            # Solvency Ratios
            if 'total_liabilities' in financial_data and 'equity' in financial_data:
                if financial_data['equity'] != 0:
                    ratios['Debt to Equity'] = financial_data['total_liabilities'] / financial_data['equity']
                else:
                    self.logger.warning("Equity is zero, skipping Debt to Equity calculation")
        
            # Efficiency Ratios
            if 'revenue' in financial_data and 'total_assets' in financial_data:
                if financial_data['total_assets'] != 0:
                    ratios['Asset Turnover'] = financial_data['revenue'] / financial_data['total_assets']
                else:
                    self.logger.warning("Total assets is zero, skipping Asset Turnover calculation")
        
            for ratio_name, ratio_value in ratios.items():
                if ratio_value == float('inf') or ratio_value == float('-inf'):
                    self.logger.warning(f"invalid ratio value for {ratio_name}: {ratio_value}")
                    del ratios[ratio_name]

        except (ZeroDivisionError, KeyError) as e:
            self.logger.error(f"Unexpected error in ratio calculation: {e}")
            return ratios
        
        except Exception as e:
            self.logger.error(f"Unexpected error in ratio calculation: {e}")
            return {}
        
        self.logger.info(f"calculated {len(ratios)} financial ratios")
        return ratios

             
    def _get_ratio_interpretation(self, ratio_name: str, value: float) -> str:

        interpretations = {
            'Return on Assets (ROA)': "Good > 5%, Excellent > 10%" if value > 5 else "Needs improvement < 5%",
            'Return on Equity (ROE)': "Good > 15%, Excellent > 20%" if value > 15 else "Needs improvement < 15%", 
            'Current Ratio': "Healthy > 1.5, Risk < 1.0" if value > 1.5 else "Potential liquidity risk < 1.0",
            'Debt to Equity': "Conservative < 0.5, High > 2.0" if value < 0.5 else "Moderate leverage 0.5-2.0" if value <= 2 else "High leverage > 2.0"
        }
        
        return interpretations.get(ratio_name, "Industry context needed for full interpretation")



if __name__ == "__main__":
    processor = DocumentProcessor()
    print("🧪 Testing Document Processor...")

    # Test 1: Create sample SEC data
    sample_sec_data = {
        "facts": {
            "us-gaap": {
                "Revenue": {
                    "units": {
                        "USD": [
                            {"val": 1000000000, "end": "2023-12-31", "form": "10-K", "filed": "2024-01-31"},
                            {"val": 900000000, "end": "2022-12-31", "form": "10-K", "filed": "2023-01-31"}
                        ]
                    }
                },
                "Assets": {
                    "units": {
                        "USD": [
                            {"val": 500000000, "end": "2023-12-31", "form": "10-K", "filed": "2024-01-31"}
                        ]
                    }
                }
            },
            "dei": {
                "EntityRegistrantName": {
                    "units": {
                        "USD": [
                            {"val": "TEST COMPANY INC"}
                        ]
                    }
                }
            }
        }
    }

    # Test 2: Process documents
    print("1. Testing document processing...")
    documents = processor.process_sec_facts(sample_sec_data, "TEST")
    print(f"✅ Created {len(documents)} documents")

     # Test 3: Show document samples
    print("\n2. Document samples:")
    for i,doc in enumerate(documents[:3]):
        print(f" document: {i+1}")
        print(f" content: {doc.page_content[:100]}")
        meta_data = {k: doc.metadata[k] for k in list(doc.metadata)[:3]}
        print(f"  Metadata: {meta_data}")  # First 3 metadata items

    # Test 4: Test ratio calculations
    print("3. Testing ratio calculations...")
    financial_data = processor._extract_latest_values(sample_sec_data)
    print(f"   Extracted data: {financial_data}")

    ratios = processor._compute_ratios(financial_data)
    print(f"   Calculated ratios: {ratios}")

    # Test 5: Test individual components
    print("\n4. Testing helper functions...")

    # Test humanize metric name
    test_metrics = ["Revenue", "NetIncomeLoss", "CashAndCashEquivalents"]
    for metric in test_metrics:
        humanized = processor._humanize_metric_name(metric)
        print(f"  {metric} → {humanized}")

    # Test ratio interpretation
    test_ratios = {
        "Return on Assets (ROA)": 8.5,
        "Current Ratio": 1.2,
        "Debt to Equity": 0.3
    }

    for ratio, value in test_ratios.items():
        interpretation = processor._get_ratio_interpretation(ratio, value)
        print(f" {ratio} ({value}): {interpretation}")

    print("\n🎯 Document Processor tests completed!")


        





        








                
    