# src/agents/orchestrator.py
import os
import sys
import asyncio
import logging
import uuid
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, TypedDict
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# LangGraph Imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

# AutoGen & AI Imports
from autogen_ext.models.openai import OpenAIChatCompletionClient
from openai import AsyncOpenAI

# Local Agents & Tools
# We import the TEAMS, not the factories, to allow shared resource injection
from src.agents.financial_analyst import FinancialAgentTeam
from src.agents.legal_reviewer import LegalAgentTeam
from src.agents.market_analyst import MarketAgentTeam
from src.rag.core import ProductionRAGSystem
from src.agents.memory_manager import MemoryManager

load_dotenv()

# --- State Definition ---
class AnalysisState(TypedDict):
    # Inputs
    request_id: str
    company_ticker: str
    analysis_type: str
    questions: List[str]
    user_id: Optional[str]

    # Progress & Metadata
    current_step: str
    progress: float
    start_time: str
    last_update: str

    # Agent Outputs (The Full Dictionaries)
    financial_analysis: Dict[str, Any]
    legal_analysis: Dict[str, Any]
    market_analysis: Dict[str, Any]

    # Context & Memory
    rag_context: Dict[str, Any]
    memory_insights: Dict[str, Any]
    
    # Final Output
    synthesis_report: Dict[str, Any]
    
    # Error Handling
    errors: List[str]
    warnings: List[str]

class DueDiligenceOrchestrator:
    """
    The Brain. Orchestrates the flow between Financial, Legal, and Market agents.
    Manages State, RAG Context, and Memory.
    """
    def __init__(self):
        self.logger = logging.getLogger("Orchestrator")
        self.logger.setLevel(logging.INFO)
        
        # 1. Setup Models
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key: raise ValueError("OPENAI_API_KEY required.")
        
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4-turbo")
        
        # Client for AutoGen Agents (Shared across all teams)
        self.agent_client = OpenAIChatCompletionClient(
            model=self.model_name,
            api_key=api_key,
            temperature=0.1
        )
        
        # Client for Orchestrator Synthesis (Raw OpenAI for JSON control)
        self.synth_client = AsyncOpenAI(api_key=api_key)
        
        # 2. Setup Infrastructure (Shared Resources)
        # FIX 1: Point to the _v2 folder where seed_data.py actually saved the files
        base_path = Path(os.getcwd()) / "data" / "vector_stores" / "financial_data"
        
        # FIX 2: Force 'huggingface' to match the Seeder's behavior
        self.rag_system = ProductionRAGSystem(
            persist_directory=str(base_path), 
            embedding_type="huggingface"  # Match API configuration for consistent collection access
        )
        self.memory_manager = MemoryManager()
        # 3. Initialize Agent Teams
        # We inject the SHARED client and RAG system here
        self.financial_agent = FinancialAgentTeam(self.agent_client, self.rag_system)
        self.legal_agent = LegalAgentTeam(self.agent_client, self.rag_system)
        self.market_agent = MarketAgentTeam(self.agent_client, self.rag_system)
        
        # 4. Build the Graph
        self.graph = self._build_workflow_graph()

    def _build_workflow_graph(self) -> StateGraph:
        """Constructs the LangGraph workflow."""
        workflow = StateGraph(AnalysisState)

        # Nodes
        workflow.add_node("initialize", self._initialize_analysis)
        workflow.add_node("gather_context", self._gather_rag_context)
        workflow.add_node("check_memory", self._gather_memory_insights)
        
        # Agent Nodes
        workflow.add_node("financial_agent", self._execute_financial_analysis)
        workflow.add_node("legal_agent", self._execute_legal_analysis)
        workflow.add_node("market_agent", self._execute_market_analysis)
        
        # Synthesis Nodes
        workflow.add_node("synthesize", self._synthesize_findings)
        workflow.add_node("save_memory", self._store_insights)
        workflow.add_node("error_handler", self._handle_errors)

        # Flow Logic (Linear for Reliability)
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "gather_context")
        workflow.add_edge("gather_context", "check_memory")
        
        # Routing Logic
        workflow.add_conditional_edges(
            "check_memory",
            self._route_analysis,
            {
                "financial_only": "financial_agent",
                "legal_only": "legal_agent",
                "market_only": "market_agent",
                "comprehensive": "financial_agent" # Start the full chain
            }
        )

        # Full Chain Flow: Financial -> Legal -> Market -> Synthesis
        workflow.add_edge("financial_agent", "legal_agent")
        workflow.add_edge("legal_agent", "market_agent")
        workflow.add_edge("market_agent", "synthesize")

        # Single Agent Paths (If routing skipped others)
        # Note: LangGraph allows multiple incoming edges
        
        workflow.add_edge("synthesize", "save_memory")
        workflow.add_edge("save_memory", END)
        workflow.add_edge("error_handler", END)

        return workflow.compile(checkpointer=InMemorySaver())

    # --- Node Implementations ---

    async def _initialize_analysis(self, state: AnalysisState) -> AnalysisState:
        state["progress"] = 0.05
        state["current_step"] = "Initialization"
        ticker = state.get("company_ticker")
        if not ticker or not ticker.isalpha() or len(ticker) > 5:
            state["errors"].append(f"Invalid ticker: {ticker}")
            return state
        self.logger.info(f"Initializing analysis for {ticker} ({state['analysis_type']})")
        return state

    async def _gather_rag_context(self, state: AnalysisState) -> AnalysisState:
        """Fetch RAG context in parallel for speed."""
        state["progress"] = 0.15
        state["current_step"] = "Gathering RAG Context"
        ticker = state["company_ticker"]
        try:
            # Run 3 queries in parallel
            fin_task = asyncio.to_thread(self.rag_system.query, f"{ticker} financial ratios revenue assets", company=ticker, k=5)
            leg_task = asyncio.to_thread(self.rag_system.query, f"{ticker} legal litigation risk compliance", company=ticker, k=3)
            mkt_task = asyncio.to_thread(self.rag_system.query, f"{ticker} market share competitors trends", company=ticker, k=3)

            fin_docs, leg_docs, mkt_docs = await asyncio.gather(fin_task, leg_task, mkt_task)

            state["rag_context"] = {
                "financial": [d.page_content for d in fin_docs],
                "legal": [d.page_content for d in leg_docs],
                "market": [d.page_content for d in mkt_docs]
            }
            self.logger.info(f"Context gathered: {len(fin_docs)} Fin, {len(leg_docs)} Leg, {len(mkt_docs)} Mkt")
        except Exception as e:
            state["warnings"].append(f"RAG Context Error: {e}")
            state["rag_context"] = {}
        return state

    async def _gather_memory_insights(self, state: AnalysisState) -> AnalysisState:
        state["progress"] = 0.20
        try:
            insights = await self.memory_manager.get_cross_analysis_insights(
                user_id=state.get("user_id", "default"),
                company=state["company_ticker"]
            )
            state["memory_insights"] = insights
        except Exception as e:
            state["warnings"].append(f"Memory Fetch Error: {e}")
            state["memory_insights"] = {}
        return state

    async def _execute_financial_analysis(self, state: AnalysisState) -> AnalysisState:
        state["current_step"] = "Financial Analysis"
        state["progress"] = 0.40
        try:
            res = await self.financial_agent.analyze_company(
                state["company_ticker"], 
                f"User Questions: {state.get('questions', [])}"
            )
            state["financial_analysis"] = res
        except Exception as e:
            state["errors"].append(f"Financial Agent Failed: {e}")
        return state

    async def _execute_legal_analysis(self, state: AnalysisState) -> AnalysisState:
        state["current_step"] = "Legal Analysis"
        state["progress"] = 0.60
        try:
            # Share financial context with legal agent
            fin_context = ""
            if state.get("financial_analysis"):
                fin_context = f"Financial Context: {str(state['financial_analysis'])[:500]}..."
            
            res = await self.legal_agent.analyze_company_legal(
                state["company_ticker"],
                f"{fin_context}\nQuestions: {state['questions']}"
            )
            state["legal_analysis"] = res if res else {
                "summary": "Legal analysis completed but no specific findings were identified.",
                "success": True
            }
        except Exception as e:
            state["errors"].append(f"Legal Agent Failed: {e}")
            # Provide fallback content so UI doesn't show empty
            state["legal_analysis"] = {
                "summary": f"Legal analysis encountered an issue: {str(e)[:100]}. Further investigation recommended.",
                "success": False,
                "error": str(e)
            }
        return state

    async def _execute_market_analysis(self, state: AnalysisState) -> AnalysisState:
        state["current_step"] = "Market Analysis"
        state["progress"] = 0.80
        try:
            res = await self.market_agent.analyze_company_market(
                state["company_ticker"],
                f"Questions: {state['questions']}"
            )
            state["market_analysis"] = res if res else {
                "summary": "Market analysis completed but no specific findings were identified.",
                "success": True
            }
        except Exception as e:
            state["errors"].append(f"Market Agent Failed: {e}")
            # Provide fallback content so UI doesn't show empty
            state["market_analysis"] = {
                "summary": f"Market analysis encountered an issue: {str(e)[:100]}. Further investigation recommended.",
                "success": False,
                "error": str(e)
            }
        return state

    async def _synthesize_findings(self, state: AnalysisState) -> AnalysisState:
        """Combines all agent outputs into the final JSON report."""
        state["current_step"] = "Synthesis"
        state["progress"] = 0.90
        
        try:
            context = {
                "company": state["company_ticker"],
                "financial": state.get("financial_analysis", {}),
                "legal": state.get("legal_analysis", {}),
                "market": state.get("market_analysis", {}),
                "rag_count": sum(len(v) for v in state.get("rag_context", {}).values())
            }
            
            # Generate sections in parallel or sequence
            summary = await self._generate_exec_summary(context)
            risks = await self._generate_llm_risk_assessment(context)
            
            # --- RISK SCORE FALLBACK (Fix for "N/A/10") ---
            if risks.get('risk_score') is None or risks.get('score') is None:
                level = str(risks.get('risk_level', 'MEDIUM')).upper()
                if 'HIGH' in level or 'CRITICAL' in level: risks['score'] = 8
                elif 'LOW' in level: risks['score'] = 3
                else: risks['score'] = 5
            # ---------------------------------------------
            
            rec = await self._generate_recommendation(context)
            
            # Construct the final report
            state["synthesis_report"] = {
                "executive_summary": summary,
                "risk_assessment": risks,
                "recommendation": rec,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            self.logger.info("Synthesis complete.")
            
        except Exception as e:
            state["errors"].append(f"Synthesis Failed: {e}")
            state["synthesis_report"] = {"error": str(e)}
            
        return state

    async def _store_insights(self, state: AnalysisState) -> AnalysisState:
        """Saves findings to Memory for future reference."""
        state["progress"] = 1.0
        state["current_step"] = "Complete"
        ticker = state["company_ticker"]
        user_id = state.get("user_id", "default")
        
        try:
            # 1. Store Financial
            fin_data = state.get("financial_analysis", {})
            if fin_data.get("status") == "success":
                await self.memory_manager.store_financial_insight(
                    user_id=user_id,
                    company_ticker=ticker,
                    insight=str(fin_data.get("analysis", ""))[:200],
                    metrics={}, 
                    pattern_type="general_analysis"
                )

            # 2. Store Legal
            legal_data = state.get("legal_analysis", {})
            if legal_data.get("success"):
                await self.memory_manager.store_legal_risk(
                    user_id=user_id,
                    company_ticker=ticker,
                    risk_type="General Legal",
                    description=str(legal_data.get("summary", ""))[:200],
                    severity="Unknown",
                    context="Automated Analysis"
                )
            self.logger.info("Insights stored in memory.")
        except Exception as e:
            state["warnings"].append(f"Memory Storage Failed: {e}")
        return state

    async def _handle_errors(self, state: AnalysisState) -> AnalysisState:
        self.logger.error(f"Workflow Critical Error: {state.get('errors')}")
        return state

    # --- Routing & Helpers ---

    def _route_analysis(self, state: AnalysisState) -> str:
        t = state.get("analysis_type", "comprehensive")
        if t == "financial": return "financial_only"
        if t == "legal": return "legal_only"
        if t == "market": return "market_only"
        return "comprehensive"

    # --- LLM Generators ---

    async def _generate_exec_summary(self, context: Dict) -> str:
        prompt = f"""
        Act as a Lead Investment Banker. Synthesize due diligence for {context['company']}.
        
        FINANCIAL: {str(context['financial'])[:1500]}
        LEGAL: {str(context['legal'])[:1000]}
        MARKET: {str(context['market'])[:1000]}
        
        Write a 3-paragraph Executive Summary. Be decisive and professional.
        """
        res = await self.synth_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content

    async def _generate_llm_risk_assessment(self, context: Dict) -> Dict[str, Any]:
        prompt = f"""
        Perform risk assessment based on this data:
        {str(context)[:3000]}
        
        Return valid JSON with keys:
        - risk_level (LOW/MEDIUM/HIGH/CRITICAL)
        - key_risks (List[str] of top 3 risks)
        - score (Integer 1-10, where 10 is Extreme Risk. MUST BE AN INTEGER.)
        """
        res = await self.synth_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(res.choices[0].message.content)

    async def _generate_recommendation(self, context: Dict) -> Dict:
        prompt = f"""
        Final Investment Recommendation for {context['company']}.
        Based on: {str(context)[:2000]}
        
        Return valid JSON with keys: 
        - verdict (BUY/SELL/HOLD)
        - confidence (High/Med/Low)
        - reasoning (String summary)
        """
        res = await self.synth_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(res.choices[0].message.content)

    def _format_final_response(self, state: AnalysisState) -> Dict[str, Any]:
        """
        Formats the final output for the API. 
        Ensures all nested dictionaries are present.
        """
        self.logger.info(f"Formatting Response. Keys available: {list(state.keys())}")
        
        return {
            "status": "completed",
            "request_id": state.get("request_id"),
            "company": state.get("company_ticker"),
            "analysis_type": state.get("analysis_type"),
            "progress": state.get("progress", 1.0),
            
            # The Critical Payload
            "synthesis_report": state.get("synthesis_report", {}),
            
            # Detailed Reports (for UI Tabs)
            "financial_analysis": state.get("financial_analysis", {}),
            "legal_analysis": state.get("legal_analysis", {}),
            "market_analysis": state.get("market_analysis", {}),
            
            "errors": state.get("errors", []),
            "warnings": state.get("warnings", []),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    # --- Public Entry Point ---
    async def execute_analysis(self, request_id: str, company_ticker: str, analysis_type: str, questions: List[str]) -> Dict:
        """
        Main entry point called by the API.
        Runs the graph and formats the output.
        """
        config = {"configurable": {"thread_id": request_id}}
        
        initial_state = AnalysisState(
            request_id=request_id,
            company_ticker=company_ticker.upper(),
            analysis_type=analysis_type,
            questions=questions,
            user_id="default_user",
            current_step="Start",
            progress=0.0,
            start_time=datetime.now(timezone.utc).isoformat(),
            last_update=datetime.now(timezone.utc).isoformat(),
            financial_analysis={}, legal_analysis={}, market_analysis={},
            rag_context={}, memory_insights={}, synthesis_report={},
            errors=[], warnings=[]
        )
        
        # Run Graph
        final_state = await self.graph.ainvoke(initial_state, config)
        
        # Format & Return
        return self._format_final_response(final_state)

# --- Run Script (Test) ---
if __name__ == "__main__":
    async def run():
        orchestrator = DueDiligenceOrchestrator()
        result = await orchestrator.execute_analysis(
            request_id=str(uuid.uuid4()),
            company_ticker="AAPL",
            analysis_type="comprehensive",
            questions=["Is the growth rate sustainable?"]
        )
        print(json.dumps(result, indent=2))

    asyncio.run(run())