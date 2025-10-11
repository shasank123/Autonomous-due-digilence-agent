from langchain_core.documents import Document
from typing import List, Dict
import json

class DocumentProcessor:
    """
    Processes REAL SEC data into RAG-ready documents
    """
    def process_sec_facts(self, company_facts: Dict, ticker: str) -> List[Document]:
      
    # Convert REAL SEC financial data into searchable documents
        documents = []

        if not company_facts or 'facts' not in company_facts:       
            return documents
        facts = company_facts['facts']

        #  process us-gaap financial data
        if 'us-gaap' in facts:
            documents.extend(self.process_gaap_facts(facts['us-gaap'], ticker))

        # process company information
        if 'dei' in facts:
            documents.extend(self.process_dei_facts(facts['dei'], ticker))

        print(f" created {len(documents)} searchable documents for {ticker}")
        return documents
    
    def process_gaap_facts(self, gaap_facts: Dict, ticker: str) -> List[Document]:
        
        """Process accounting/financial facts"""
        documents = []

        for fact_name, fact_data in gaap_facts.items():
            if 'units' not in fact_data:
                continue
            for unit, values in fact_data['units'].items():
                for value in values[:5]:
                    doc_content = self._create_financial_content(ticker, fact_name, value, unit)

                    document = Document(page_content = doc_content,
                                        metadata= {
                                            "source" : "SEC",
                                            "company": ticker,
                                            "metric_type": "financial",
                                            "metric": fact_name,
                                            "unit": unit,
                                            "period_end": value.get('end', ''),
                                            "filed_date": value.get('filed', ''),
                                            "form_type": value.get('form', '')
                                   }  
                                )
                    documents.append(document)
                    return documents
                
    def process_dei_facts(self, dei_facts: Dict, ticker: str) -> List[Document]:
        
        """ process company entity information """
        documents = []
        
        company_name = "unknown"

        if 'EntityRegistrationName' in dei_facts:
            name_data = dei_facts['EntityRegistrationName']

            if 'units' in name_data and 'USD' in name_data['units']:
                company_name = name_data['units']['USD'][0]['val']

        # Create company overview document
        overview_content = f"""
        Company Name: {company_name}
        Ticker Symbol: {ticker}
        Business Information: Publicly traded company filing with SEC
        Reporting Status: Active SEC filer
        """

        document = Document(
            page_content=overview_content.strip(),
            metadata={
                "source": "sec", 
                "company": ticker,
                "metric_type": "company_info",
                "metric": "company_overview"
            }
        )

        documents.append(document)
        return documents
    
    def _create_financial_content(self, ticker : str, fact_name: str, value: Dict, unit: str) -> str:

        # Create readable financial statement content
        return f"""
        Company: {ticker}
        Financial Metric: {fact_name.replace('_', ' ').title()}
        Value: {value.get('val', 'N/A')} {unit}
        Reporting Period: {value.get('end', 'N/A')}
        Report Filed: {value.get('filed', 'N/A')}
        Form Type: {value.get('form', 'N/A')}
        Accounting Standard: US-GAAP
        """
    
    # test with real data

if __name__ == "__main__":
    from collectors.sec_edgar import SECDataCollector

    processor = DocumentProcessor()
    collector = SECDataCollector(email= "sasishasank2@gmail.com")

    # REAL TEST: Process Microsoft data
    print("🔄 Testing with REAL Microsoft data...")
    msft_data = collector.company_facts("MSFT")

    if msft_data:
        documents = processor.process_sec_facts(msft_data, "MSFT")
        print(f"📄 Processed {len(documents)} documents for MSFT")

        # show sample documents
        for i,doc in enumerate(documents[:3]):
            print(f"\n -- Document {i+1} --")
            print(f" page_content: {doc.page_content[:200]}")
            print(f" metadata: {doc.metadata}")        
    else:
        print("❌ Failed to fetch Microsoft data")


    



        
        

        
                            
                        



