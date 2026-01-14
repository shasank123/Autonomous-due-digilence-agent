# src/agents/memory_manager.py
import uuid
import logging
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from langgraph.store.memory import InMemoryStore
from langchain_huggingface import HuggingFaceEmbeddings

class MemoryManager:
    """
    Manages long-term memory and cross-analysis insights for the agent team.
    Uses semantic search to find patterns across different companies and domains.
    """
    _instance = None

    def __new__(cls):
        # Singleton pattern to prevent reloading embeddings
        if cls._instance is None:
            cls._instance = super(MemoryManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Prevent re-initialization
        if hasattr(self, 'store'):
            return

        self.logger = logging.getLogger(__name__)
        
        try:
            self.logger.info("Initializing Memory Manager (Embeddings)...")
            # Initialize Semantic Store
            self.store = InMemoryStore(
                index={
                    "embed": HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
                    "dims": 384,
                    # Searchable fields
                    "fields": ["insight", "risk_type", "description", "financial_patterns", "legal_insights"] 
                }
            )
        except Exception as e:
            self.logger.critical(f"Failed to initialize Memory Store: {e}")
            raise

        # Namespaces for organizing memory types
        self.NAMESPACES = {
            "financial": "financial_insights",
            "legal": "legal_insights",            
            "market": "market_insights",         
            "industry": "industry_data"
        }

    async def store_financial_insight(self, user_id: str, company_ticker: str, insight: str, 
                                      metrics: Dict[str, Any], pattern_type: str = "general") -> str:
        """Stores a financial insight for future cross-referencing."""
        try:
            memory_id = str(uuid.uuid4())
            # Tuple key for LangGraph InMemoryStore: (User, Namespace)
            namespace = (user_id, self.NAMESPACES["financial"])

            memory_data = {
                "insight": insight,
                "company_ticker": company_ticker,
                "pattern_type": pattern_type,
                "metrics": metrics,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                # Text field for semantic embedding
                "financial_patterns": f"{company_ticker} {pattern_type}: {insight}. Metrics: {metrics}"
            }

            # Store operation (Blocking I/O wrapped for safety)
            # In a real DB, this would be awaitable. InMemory is fast enough.
            self.store.put(
                namespace,
                memory_id,
                memory_data
            )

            self.logger.info(f"Stored financial insight for {company_ticker}")
            return memory_id
        
        except Exception as e:
            self.logger.error(f"Failed to store financial insight: {e}")
            return ""

    async def store_legal_risk(self, user_id: str, company_ticker: str, risk_type: str, 
                               description: str, severity: str, context: str) -> str:
        """Stores identified legal risks."""
        try:
            memory_id = str(uuid.uuid4())
            namespace = (user_id, self.NAMESPACES["legal"])

            memory_data = {
                "risk_type": risk_type,
                "description": description,
                "severity": severity,
                "company_ticker": company_ticker,
                "context": context,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "legal_insights": f"{severity} {risk_type} risk for {company_ticker}: {description}"
            }

            self.store.put(
                namespace,
                memory_id,
                memory_data
            )
            self.logger.info(f"Stored legal risk for {company_ticker}")
            return memory_id

        except Exception as e:
            self.logger.error(f"Failed to store legal risk: {e}")
            return ""

    async def search_similar_companies(self, user_id: str, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Finds companies with similar financial/risk profiles based on the query."""
        try:
            namespace = (user_id, self.NAMESPACES["financial"])
            
            results = self.store.search(
                namespace,
                query=query,
                limit=limit
            )
            
            insights = []
            for result in results:
                insights.append({
                    "company": result.value.get("company_ticker"),
                    "insight": result.value.get("insight"),
                    "metrics": result.value.get("metrics", {}),
                    "pattern_type": result.value.get("pattern_type"),
                    "timestamp": result.value.get("timestamp"),
                    "score": getattr(result, 'score', 0.0)
                })
            return insights

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

    async def search_industry_risks(self, user_id: str, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Searches for past legal/market risks relevant to the current context."""
        try:
            namespace = (user_id, self.NAMESPACES["legal"])
            
            results = self.store.search(
                namespace,
                query=query,
                limit=limit
            )

            risks = []
            for result in results:
                risks.append({
                    "company": result.value.get("company_ticker"),
                    "risk_type": result.value.get("risk_type"),
                    "description": result.value.get("description"),
                    "severity": result.value.get("severity"),
                    "timestamp": result.value.get("timestamp")
                })
            return risks

        except Exception as e:
            self.logger.error(f"Industry risk search failed: {e}")
            return []

    async def get_cross_analysis_insights(self, user_id: str, company: str, analysis_type: str = "general") -> Dict[str, Any]:
        """
        Aggregates insights from previous analyses to provide context for the current company.
        Example: "When analyzing Tesla, look at how Rivian handled supply chain issues (from memory)."
        """
        insights = {
            "financial_patterns": [],
            "legal_risks": [],
            "similar_companies": []
        }
        
        try:
            # 1. Find Similar Financial Patterns
            # Query: "financial patterns similar to {company}"
            fin_query = f"financial patterns performance metrics like {company}"
            insights["financial_patterns"] = await self.search_similar_companies(user_id, fin_query)

            # 2. Find Relevant Legal Risks
            # Query: "legal risks for {company} industry"
            legal_query = f"legal regulatory risks {company} industry"
            insights["legal_risks"] = await self.search_industry_risks(user_id, legal_query)

            # 3. Find Strategic Peers (General similarity)
            peer_query = f"companies strategically similar to {company}"
            insights["similar_companies"] = await self.search_similar_companies(user_id, peer_query)

        except Exception as e:
            self.logger.warning(f"Cross-analysis insight generation partial fail: {e}")

        return insights

# --- Testing Block ---
if __name__ == "__main__":
    async def main():
        logging.basicConfig(level=logging.INFO)
        mem = MemoryManager()
        
        user = "test_user"
        
        # 1. Store some dummy data
        print("Storing Insight...")
        await mem.store_financial_insight(
            user, "TSLA", "High volatility due to regulatory credits", 
            {"volatility": "high", "beta": 2.1}, "risk_pattern"
        )
        
        # 2. Store some legal data
        await mem.store_legal_risk(
            user, "TSLA", "Regulatory", "SEC investigation into disclosures", "High", "10-K Filing"
        )

        # 3. Test Retrieval
        print("\nRetrieving Insights for similar company (RIVN)...")
        insights = await mem.get_cross_analysis_insights(user, "RIVN")
        
        print(f"Financial Patterns Found: {len(insights['financial_patterns'])}")
        for item in insights['financial_patterns']:
            print(f" - Found match from {item['company']}: {item['insight']}")

    asyncio.run(main())