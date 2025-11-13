# src/agents/memory_manager.py
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import logging
from langgraph.store.memory import InMemoryStore
from langchain_huggingface import HuggingFaceEmbeddings

class MemoryManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Create a "smart storage" that understands text meaning
        self.store = InMemoryStore(
            index={
                "embed": HuggingFaceEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2"),# This converts text to numbers (embeddings) for semantic search
                "dims": 384,
                "fields": ["insights", "company_comtext", "financial_patterns", "$"]
                 # "$" means: embed ALL fields for search
             }
        )
        # Create "folders" for different types of memories
        self.NAMESPACES = {
            "financial_patterns": "financial_insights",
            "legal_risks": "legal_insights",            
            "market_trends": "market_insights",         
            "industry_benchmarks": "industry_data"
        }

    def store_financial_insight(self, user_id: str, company_ticker: str, insight: str,
                                metrics: Dict[str, Any], pattern_type: str = "general") -> str:
        # Step 1: Create unique ID for this memory
        memory_id = str(uuid.uuid4())
        # Step 2: Choose which "folder" to store this in
        name_space = (user_id, self.NAMESPACES["financial_patterns"])# Example: ("user_123", "financial_insights")

        # Step 3: Prepare the memory data
        memory_data = {
            "insight" : insight,
            "company_ticker": company_ticker,
            "pattern_type": pattern_type,
            "metrics": metrics,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "financial_patterns": f" {insight} for {company_ticker} with metrics {metrics}"
        }
        # Step 4: Store in the "smart storage"
        self.store.put(
            name_space,
            memory_id,
            memory_data,
            index = ["insight", "financial_patterns"]# Make these fields searchable
        )

        self.logger.info(f"Stored financial insight for {company_ticker}")
        return memory_id
    
    def store_legal_risk(
            self,
            user_id: str,
            company_ticker: str,
            risk_type: str,
            description: str,
            severity: str,
            context: str
    ) -> str:
         """Store legal risks and compliance patterns"""
         memory_id = str(uuid.uuid4())
         namespace = (user_id, self.NAMESPACES["legal_risks"])

         memory_data = {
             "risk_type": risk_type,
             "description": description,
             "severity": severity,
             "company_ticker": company_ticker,
             "context": context,
             "timestamp": datetime.now(timezone.utc).isoformat(),
             "legal_insights": f"{risk_type} risk for {company_ticker}: {description}"
         }

         self.store.put(
             namespace,
             memory_id,
             memory_data,
             index= ["risk_type","description","legal_insights"]
         )
         self.logger.info(f"Stored legal insight for {company_ticker}")
         return memory_id
    
    def search_similar_companies(self, user_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find companies with similar financial/risk profiles"""

        # Choose which "memory folder" to search in
        namespace = (user_id, self.NAMESPACES["financial_patterns"])
        # Example: ("user_123", "financial_insights")

        #Perform semantic search using natural language
        results = self.store.search(
            namespace,
            query=query,
            limit=limit
        )
        
        insights = []
        for result in results:
            insights.append(
                {
                    "company": result.value.get("company_ticker"),
                    "insight": result.value.get("insight"),
                    "metrics": result.value.get("metrics", {}),
                    "pattern_type": result.value.get("pattern_type"),
                    "timestamp": result.value.get("timestamp"),
                    "score": result.score if hasattr(result, 'score') else 1.0
                }
            )
        return insights

    def search_industry_risks(
            self,
            user_id: str,
            industry_context: str,
            risk_type: Optional[str] = None,
            limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar legal/market risks"""
        namespace = (user_id, self.NAMESPACES["legal_risks"])
        query = f" {risk_type} risks for {industry_context}" if risk_type else industry_context

        results = self.store.search(
            namespace,
            query=query,
            limit=limit
        )

        risks = []
        for result in results:
            risks.append(
                {
                    "company": result.value.get("company_ticker"),
                    "risk_type": result.value.get("risk_type"),
                    "description": result.value.get("description"),
                    "severity": result.value.get("severity"),
                    "context": result.value.get("context"),
                    "timestamp": result.value.get("timestamp")
                    }
            )

        return risks
    
    def get_cross_analysis_insights(
            self,
            user_id: str,
            current_company: str,
            analysis_type: str
    ) -> Dict[str, Any]:
        """Get relevant insights from previous analyses"""

        # Create empty container for insights
        insights = {
            "financial_patterns": [],
            "legal_risks": [],
            "similar_companies": []
        }
        
        try:
            # Find similar financial patterns
            financial_query = f"financial patterns ratios performance {current_company}"
            insights["financial_patterns"] = self.search_similar_companies(
                user_id,
                financial_query,
                limit=3
            )# Example result: [
            #   {"company": "AAPL", "insight": "High R&D = Growth", ...},
            #   {"company": "NIO", "insight": "EV companies need capex", ...}
            # ]

            # Find relevant legal risks
            legal_query = f"legal complaince risks {current_company}"
            insights["legal_risks"] = self.search_industry_risks(
                user_id,
                legal_query,
                limit=3
            )# Example result: [
            #   {"company": "F", "risk_type": "regulatory", "description": "Emissions  standards", ...},
            #   {"company": "GM", "risk_type": "safety", "description": "Recall issues", ...}
            # ]
             

            # Find companies with similar profiles
            similarity_query =f"companies similar to {current_company} financial metrics"
            # Query: "companies similar to TSLA financial metrics"
            insights["similar_companies"] = self.search_similar_companies(
                user_id,
                similarity_query,
                limit=3
            )# Example result: [
            #   {"company": "NIO", "insight": "EV manufacturer growth patterns", ...},
            #   {"company": "RIVN", "insight": "Electric vehicle market entry", ...}
            # ]

        except Exception as e:
            self.logger.warning(f"Cross-analysis insights failed: {e}")

        # Return all gathered insights
        return insights # Example return: {
        #   "financial_patterns": [AAPL insights, NIO insights],
        #   "legal_risks": [F risks, GM risks], 
        #   "similar_companies": [NIO profile, RIVN profile]
        # }
        















    
    