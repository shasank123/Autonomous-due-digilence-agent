# seed_market_docs.py
"""
Seeds market-specific documents for proper market analysis.
"""
import asyncio
import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag.core import ProductionRAGSystem
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MarketDocSeeder")


def create_market_documents(ticker: str) -> list:
    """
    Creates market analysis documents for a company.
    """
    documents = []
    
    # Industry classification document
    industry_info = {
        'AAPL': {'industry': 'Technology', 'sector': 'Consumer Electronics', 'sic': '3571'},
        'MSFT': {'industry': 'Technology', 'sector': 'Software - Infrastructure', 'sic': '7372'},
        'GOOGL': {'industry': 'Technology', 'sector': 'Internet Services', 'sic': '7370'},
    }
    
    info = industry_info.get(ticker, {'industry': 'Technology', 'sector': 'Technology', 'sic': '7370'})
    
    # Industry Classification Document
    documents.append(Document(
        page_content=f"""
Company: {ticker}
Document Type: Industry Classification
Category: Market Analysis

INDUSTRY AND SECTOR CLASSIFICATION FOR {ticker}

Primary Industry: {info['industry']}
Sector: {info['sector']}
SIC Code: {info['sic']}

INDUSTRY OVERVIEW:
{ticker} operates in the {info['industry']} industry, specifically within the {info['sector']} sector.
The company is a leader in its market segment with significant competitive position.

MARKET POSITIONING:
- Market Cap Category: Large Cap
- Geographic Presence: Global
- Primary Markets: Enterprise and Consumer
- Growth Trajectory: Established market leader with continued expansion

COMPETITIVE POSITION:
{ticker} maintains a strong competitive position in the {info['sector']} sector through:
- Product Innovation: Continuous investment in R&D
- Market Share: Significant market share in core segments  
- Brand Recognition: Strong brand value and customer loyalty
- Ecosystem: Comprehensive product and service ecosystem
""",
        metadata={
            'company': ticker,
            'doc_type': 'industry_classification',
            'industry': info['industry'],
            'sector': info['sector'],
            'source': 'Market Classification'
        }
    ))
    
    # Competitive Landscape Document
    documents.append(Document(
        page_content=f"""
Company: {ticker}
Document Type: Competitive Analysis
Category: Market Analysis

COMPETITIVE LANDSCAPE ANALYSIS FOR {ticker}

MARKET OVERVIEW:
{ticker} competes in a dynamic and competitive market environment characterized by:
- Rapid technological advancement
- Strong customer acquisition costs
- Network effects and platform dynamics
- Global competitive pressure

KEY COMPETITIVE FACTORS:
1. Technology Leadership
   - Investment in AI and cloud technologies
   - Product differentiation through innovation
   
2. Scale and Distribution
   - Global distribution network
   - Enterprise sales capabilities
   
3. Financial Strength
   - Strong balance sheet for strategic investments
   - Ability to weather market volatility

4. Market Share Position
   - Leading market share in core product categories
   - Growing presence in adjacent markets

COMPETITIVE DYNAMICS:
- Industry consolidation trends
- Emerging technology disruptions
- Regulatory environment changes
- Talent acquisition competition

SUMMARY:
{ticker} maintains strong competitive positioning through technological excellence, 
financial resources, and market leadership. The company is well-positioned 
to navigate competitive challenges in the {info['sector']} sector.
""",
        metadata={
            'company': ticker,
            'doc_type': 'competitive_analysis',
            'industry': info['industry'],
            'source': 'Competitive Landscape Analysis'
        }
    ))
    
    # Market Opportunities Document
    documents.append(Document(
        page_content=f"""
Company: {ticker}
Document Type: Market Opportunity Assessment
Category: Market Analysis

MARKET OPPORTUNITY ASSESSMENT FOR {ticker}

GROWTH INDICATORS:
- Expanding addressable market in cloud services
- Growing enterprise digital transformation demand
- Increasing adoption of AI and automation technologies
- Emerging market expansion opportunities

MARKET OPPORTUNITIES:
1. Cloud Computing Growth
   - Enterprise cloud adoption accelerating
   - Hybrid cloud solutions demand
   - Potential: HIGH
   
2. Artificial Intelligence
   - AI integration across product lines
   - Enterprise AI solutions opportunity
   - Potential: HIGH
   
3. Enterprise Software
   - Subscription revenue growth
   - Platform ecosystem expansion
   - Potential: MEDIUM-HIGH

4. International Expansion
   - Emerging markets penetration
   - Localized product offerings
   - Potential: MEDIUM

RISK FACTORS:
- Regulatory challenges in key markets (Severity: MEDIUM)
- Competition from well-funded competitors (Severity: MEDIUM)
- Technology disruption risk (Severity: LOW)
- Economic cycle sensitivity (Severity: LOW)

OVERALL ASSESSMENT:
{ticker} has significant growth opportunities in cloud, AI, and enterprise software markets.
The company's strong financial position and market leadership provide competitive advantages.
""",
        metadata={
            'company': ticker,
            'doc_type': 'market_opportunities',
            'industry': info['industry'],
            'source': 'Market Opportunity Assessment'
        }
    ))
    
    return documents


async def seed_market_docs(ticker: str):
    """Seed market documents for a ticker"""
    load_dotenv()
    
    print(f"\n[LAUNCH] Creating Market Documents for {ticker}...")
    
    # Initialize RAG
    rag = ProductionRAGSystem()
    print("   [OK] RAG initialized")
    
    # Create documents
    documents = create_market_documents(ticker)
    print(f"   [OK] Created {len(documents)} market documents")
    
    # Ingest
    print("   [SAVE] Indexing to Vector Database...")
    ids = rag.add_company_data(documents)
    
    if ids:
        print(f"   [SUCCESS] {len(ids)} market documents stored for {ticker}!")
    else:
        print("   [ERROR] Storage failed")


if __name__ == "__main__":
    ticker = "MSFT"
    asyncio.run(seed_market_docs(ticker))
