# src/agents/orchestrator.py
from typing import Dict, List, Optional, Any, TypedDict, Annotated
from datetime import datetime, timezone
import logging
import uuid
from operator import add
import json

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from agents.financial_analyst import FinancialAgentTeam
from agents.legal_reviewer import LegalAnalysisAgent
from agents.market_analyst import MarketAnalysisAgent
from rag.core import ProductionRAGSystem
from agents.memory_manager import MemoryManager

class AnalysisState(TypedDict):
    # Inputs - Data coming FROM the user
    request_id: str
    company_ticker: str
    analysis_type: str
    questions: List[str]
    user_id: Optional[str]

    # Progress Tracking - LangGraph internal state
    current_step: str
    progress: float
    start_time: str
    last_update: str

    # Agent Results - Outputs FROM each agent
    financial_analysis: Dict[str, Any]
    legal_analysis: Dict[str, Any]
    market_analysis: Dict[str, Any]

    # Intermediate Data - Temporary data between steps
    rag_context: Dict[str, Any]
    extracted_metrics: Dict[str, Any]

    # Memory Insights
    memory_insights: Dict[str, Any]

    # Final Output - Combined resultsgit
    synthesis_report: Dict[str, Any]

    # Error Handling - Using LangGraph reducers
    errors: Annotated[List[str], add]
    warnings: Annotated[List[str], add]


class DueDiligenceOrchestrator:
    """
    Production orchestrator using latest LangGraph patterns
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rag_system = ProductionRAGSystem()
        self.memory_manager = MemoryManager()

        # Initialize agents
        self.financial_agent = FinancialAnalysisAgent()
        self.legal_agent = LegalAnalysisAgent()
        self.market_agent = MarketAnalysisAgent()

        # Build workflow with persistence
        self.graph = self._build_workflow_graph()
        self.logger.info("DueDiligenceOrchestrator initialized with persistence and semantic memory")
    
    def _build_workflow_graph(self) -> StateGraph:
        """Build workflow using latest LangGraph patterns"""
        workflow = StateGraph[AnalysisState]

        # Add nodes
        workflow.add_node("initialize_analysis", self._initialize_analysis)
        workflow.add_node("gather_rag_context", self._gather_rag_context)
        workflow.add_node("gather memory insights", self._gather_memory_insights)
        workflow.add_node("execute_financial_analysis", self._execute_financial_analysis)
        workflow.add_node("execute_legal_analysis", self._execute_legal_analysis)
        workflow.add_node("execute_market_analysis", self._execute_market_analysis)
        workflow.add_node("synthesize_findings", self._synthesize_findings)
        workflow.add_node("store insights", self._store_insights)
        workflow.add_node("handle_errors", self._handle_errors)

        # Define edges using START constant
        workflow.add_edge(START, "initialize_analysis")
        workflow.add_edge("initialize_analysis","gather_rag_context")
        workflow.add_edge("gather_rag_context", "gather_memory_insights")

        # Conditional routing based on analysis type
        workflow.add_conditional_edges(
            "gather_memory_insights",
            self._route_analysis_type,
            {
                "financial": "execute_financial_analysis",
                "legal": "execute_legal_analysis",
                "market": "execute_market_analysis", 
                "comprehensive": "execute_financial_analysis"
            }

        )

        # Comprehensive analysis flow
        workflow.add_edge("execute_financial_analysis", "execute_legal_analysis")
        workflow.add_edge("execute_legal_analysis", "execute_market_analysis")
        workflow.add_edge("execute_market_analysis", "synthesize_findings")

        # Single analysis flows
        workflow.add_conditional_edges(
            "execute_financial_analysis",
            self._should_continue_to_legal,
            {
                "continue": "execute_legal_analysis",
                "synthesize": "synthesize_findings"
            }
        )

        workflow.add_conditional_edges(
            "execute_legal_anaysis",
            self._should_continue_to_market,
            {
                "continue": "execute_market_analysis",
                "synthesize": "synthesize_findings"
                
            }
        )

        workflow.add_edge("execute_market_analysis", "synthesize_findings")
        workflow.add_edge("synthesize_findings", "store_insights")
        workflow.add_edge("store_insights", END)
        workflow.add_edge("handle_errors", END)

         # Compile with persistence
        checkpointer = InMemorySaver()
        store = InMemoryStore()

        return workflow.compile(
            checkpointer=checkpointer,
            store=store
        )

    async def execute_analysis(
            self,
            request_id: str,
            company_ticker: str,               
            analysis_type: str,                 
            questions: List[str], 
            user_id: Optional[str] = None,
            update_progress_callback: Optional[callable] = None  
    ) -> Dict[str, Any]:
        """
        Execute analysis with proper thread persistence
        """
        try:
            # Create config with thread_id for persistence
            config = {"configurable": {"thread_id": request_id}}

            initial_state = analysis_type(
                request_id=request_id,
                company_ticker=company_ticker.upper(),
                analysis_type=analysis_type,
                questions=questions,
                user_id=user_id,
                current_step="initialized",
                progress=0.0,
                start_time=datetime.now(timezone.utc).isoformat(), # "2024-01-15T10:30:00Z"
                last_update=datetime.now(timezone.utc)().isoformat(),# "2024-01-15T10:30:00Z"
                financial_analysis={},                    
                legal_analysis={},                        
                market_analysis={}, 
                rag_context={},                           
                extracted_metrics={},                     
                synthesis_report={},                      
                errors=[],                                
                warnings=[]  
            )
            # Execute the LangGraph workflow asynchronously
            final_state = await self.graph.ainvoke(
                initial_state,
                config
            )# This runs: initialize → financial → legal → market → synthesize

            return self._format_final_response(final_state)
        
        except Exception as e:
            self.logger.error(f"Analysis execution failed: {e}")
            self._format_error_response(str(e)) # Returns: {"error": "Analysis failed: API timeout", "status": "failed"}

    def get_analysis_state(self, request_id: str) -> Optional[Dict]:
         """Get current analysis state using persistence"""
         try:
             # Create config to identify WHICH analysis to check
             config = {"configurable": {"thread_id": request_id}} 

             # Get the LATEST saved state from LangGraph persistence
             state = self.graph.get_state(config)# Returns StateSnapshot with values, metadata, etc.

             return state.values if state else None  #Returns: {"progress": 0.4, "current_step": "financial_analysis", ...}
            # or None if analysis doesn't exist
         
         except Exception as e:
             self.logger.error(f"Failed to get state for {request_id}: {e}")
             return None

    def get_analysis_history(self, request_id: str) -> List[Dict]:
        """Get analysis execution history - ALL checkpoints"""
        try:
            # Identify which analysis history to retrieve
            config = {"configurable": {"thread_id": request_id}}
            history = list(self.graph.get_state_history(config)) #Returns: [StateSnapshot1, #StateSnapshot2, StateSnapshot3, ...]
            
            # Convert each checkpoint to readable format
            return [
                {
                    "values": snapshot.values,
                    "step": snapshot.metadata.get("step", "unknown"),
                    "timestamp": snapshot.created_at.isoformat() if snapshot.created_at else "unknown"
                } 
                for snapshot in history
            ]
        
        except Exception as e:
            self.logger.error(f"Failed to get history for {request_id}: {e}")
            return []
        

    # Node implementations
    async def _initialize_analysis(self, state: AnalysisState) -> AnalysisState:
        """Initialize analysis with progress tracking and validation"""
        state["progress"] = 0.5
        state["current_step"] = "initialize_analysis"
        state["last_update"] = datetime.now(timezone.utc).isoformat()

        try:
            if not await self._validate_company(state["company_ticker"]):
                state["errors"].append(f"Unsupported company ticker: {state['company_ticker']}")
                return state
            
            validation_types = ["comprehensive", "financial", "legal", "market"]
            if state["analysis_type"] not in validation_types:
                state["errors"].append(f"Invalid analysis type: {state['analysis_type']}")
                return state
            
            for question in state["questions"]:
                if len(question) > 500:
                    state["warnings"].append(f"Question too long, truncating: {question[:100]}")
                    
            self.logger.info(f"✅ Initialized analysis for: {state['company_ticker']}")

        except Exception as e:
            state["errors"].append(f"Initialization failed: {str(e)}")

        return state
    
    async def _gather_rag_context(self, state: AnalysisState) -> AnalysisState:
        """Gather RAG context with progress tracking"""
        state["progress"] = 0.15
        state["current_step"] = "gather_rag_context"
        state["last_update"] = datetime.now(timezone.utc).isoformat() # Example: #"2024-01-15T10:31:00Z"
        
        try:
            rag_context = {}
            # Financial context from SEC filings
            financial_docs = self.rag_system.query(
                # Search query: "AAPL financial statements ratios revenue assets"
                f" {state['company_ticker']} financial statements ratios revenue assets",
                company = state["company_ticker"],
                k = 10 
            ) # Returns: [Document1, Document2, ... Document10] from vector store

            # Extract just the text content from first 5 financial documents
            rag_context["financial"] = [doc.page_content for doc in financial_docs[:5]]
            
            # Legal context 
            legal_docs = self.rag_system.query(
                f" {state['company_ticker']} legal compliance regulatory",
                company= state["company_ticker"],
                k = 5
            )
            rag_context["legal"] = [doc.page_content for doc in legal_docs[:3]]
            
            # Market context
            market_docs = self.rag_system.query(
                f" {state['company_ticker']} market competition industry",
                company= state['company_ticker'],
                k = 5
            )
            rag_context["market"] = [doc.page_content for doc in market_docs[:3]]

            state["rag_context"] = rag_context

            self.logger.info(f"📚 Gathered {sum(len(v) for v in rag_context.values())} RAG documents")

        except Exception as e:
            state["warnings"].append(f"RAG context gathering partial:{str(e)}")
            state["rag_context"] = {}

        return state # State now has: progress=0.15, rag_context={financial_docs}, #current_step="gather_rag_context"
    
    async def _gather_memory_insights(self, state: AnalysisState) -> AnalysisState:
         """Gather insights from previous analyses using semantic memory"""
         state["progress"] = 0.25
         state["current_step"] = "gather_memory_insights"
         state["last_update"] = datetime.now(timezone.utc).isoformat()

         try:
             user_id = state.get("user_id", "default_user")
             company_ticker = state["company_ticker"]
             # Get cross-analysis insights
             memory_insights = self.memory_manager.get_cross_analysis_insights(
                 user_id=user_id,
                 current_company=company_ticker,
                 analysis_type=state["analysis_type"]
             )

             state["memory_insights"] = memory_insights
             self.logger.info(f" Retrieved {len(memory_insights['financial_patterns'])} memory insights for {company_ticker}")

         except Exception as e:
             state["warnings".append(f"Memory insights partial: {str(e)}")]
             state["memory_insights"] = {}

         return state
    
    async def _store_insights(self, state: AnalysisState) -> AnalysisState:
        """Store valuable insights from this analysis for future use"""
        state["progress"] = 0.98
        state["current_step"] = "store_insights"
        state["last_update"] = datetime.now(timezone.utc).isoformat()

        try:
            user_id = state.get("user_id", "default_user")
            company_ticker = state["company_ticker"]
            # Store financial insights
            if state.get("financial_analysis", {}).get("key_metrics"):
                financial_insight = f"Financial performance patterns for {company_ticker}"
                self.memory_manager.store_financial_insight(
                    user_id=user_id,
                    company_ticker=company_ticker,
                    insight=financial_insight,
                    metrics=state["financial_analysis"]["key_metrics"],
                    pattern_type="profitability"
                )

            # Store legal risks
            if state.get("legal_analysis", {}).get("risk_factors"):
                for risk in state["legal_analysis"]["risk_factors"]:
                    severity = self._extract_risk_severity(risk)
                    self.memory_manager.store_legal_risk(
                        user_id=user_id,
                        company_ticker=company_ticker,
                        risk_type="compliance",
                        description=risk,
                        severity=severity,
                        context=f"Legal analysis for {company_ticker}"
                    )
        
            self.logger.info(f"Stored insights from {company_ticker} analysis")

        except Exception as e:
            state["warnings"].append(f"Insight storage failed: {str(e)}")

        return state
   
    async def _execute_financial_analysis(self, state: AnalysisState) -> AnalysisState:
        """Execute financial analysis using Financial Agent"""
        state["progress"] = 0.4
        state["current_step"] = "execute_financial_analysis"
        state["last_update"] = datetime.now(timezone.utc).isoformat()

        try:
            financial_results = await self.financial_agent.analyze(
                company_ticker= state['company_ticker'],
                rag_context= state["rag_context"].get("financial_results", []),
                questions= state["questions"]

            )

            state["financial_analysis"] = financial_results
            self.logger.info(f"💰 Completed financial analysis for {state['company_ticker']}")

        except Exception as e:
            state["errors"].append(f"Financial analysis failed: {str(e)}")
            state["errors"] = {"error": str(e), "status": "failed"}

        return state
    
    async def _execute_legal_analysis(self, state: AnalysisState) -> AnalysisState:
        """Execute legal analysis using Legal Agent"""
        state["progress"] = 0.6
        state["current_step"] = "execute_legal_analysis"
        state["last_update"] = datetime.now(timezone.utc).isoformat() 

        try:
            legal_results = await self.legal_agent.analyze(
                company_ticker= state["company_ticker"],
                rag_context= state["rag_context"].get("legal", []),
                questions= state["questions"]
            )

            state["legal_analysis"] = legal_results
            self.logger.info(f"⚖️ Completed legal analysis: {state['company_ticker']}")     

        except Exception as e:
            state["warnings"].append(f"Legal analysis partial: {str(e)}") 
            state["legal_analysis"] = {"warning": str(e), "status": "partial"}

        return state
    
    async def _execute_market_analysis(self, state:AnalysisState) -> AnalysisState:
        """Execute market analysis using Market Agent"""
        state["progress"] = 0.8
        state["current_step"] ="execute_market_analysis"
        state["last_update"] = datetime.now(timezone.utc).isoformat()

        try:
            legal_results = await self.market_agent.analyze(
                company_ticker= state["company_ticker"],
                rag_context= state["rag_context"].get("market", []),
                questions= state["questions"]
            )

            state["legal_analysis"] = legal_results
            self.logger.info(f"📈 Completed market analysis for {state['company_ticker']}")

        except Exception as e:
            state["warnings"].append(f"Market analysis partial: {str(e)}")
            state["market_analysis"] = {"warning" : str(e), "status": "partial"}

        return state

    async def _synthesize_findings(self, state:AnalysisState) -> AnalysisState:
        """Synthesize all findings into final report using LLM"""
        state["progress"] = 0.95
        state["current_step"] = "synthesize_findings"
        state["last_update"] = datetime.now(timezone.utc).isoformat()

        try:
            # Prepare comprehensive context for LLM synthesis
            synthesis_context = self._prepare_llm_synthesis_context(state)
            # Use LLM to generate professional executive summary
            executive_summary = await self._generate_llm_executive_summary(synthesis_context)
            # Use LLM to extract and validate key metrics
            key_findings = await self._extract_llm_key_findings(synthesis_context)
            # Use LLM for sophisticated risk assessment
            risk_assessment = await self._generate_llm_risk_assessment(synthesis_context)
            # Use LLM for data-driven investment recommendation
            recommendation = await self.generate_llm_recommendation(synthesis_context)

            # Include memory insights in synthesis
            memory_context = ""
            if state.get("memory_insights"):
                memory_context = self._format_memory_context(state["memory_insights"])

            synthesis = {
                "company": state["company_ticker"],
                "analysis_type": state["analysis_type"],
                "executive_summary": executive_summary,
                "key_findings": key_findings,
                "cross_analysis_insights": state.get("memory_insights", {}),
                "risk_assessment": risk_assessment,
                "recommendation": recommendation,
                "supporting_evidence": self._extract_supporting_evidence(state),
                "confidence_score": self._calculate_confidence_score(state),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "analysis_duration": self._calculate_duration(state["start_time"])
            }

            state["synthesis_report"] = synthesis
            self.logger.info(f"🎯 Generated LLM-powered synthesis report for {state['company_ticker']}")

        except Exception as e:
            state["errors"].append(f"Synthesis failed: {str(e)}")
            state["synthesis_report"] = {"error": str(e)}
            
        return state 
       
    async def _generate_llm_executive_summary(self, context: Dict) -> str:
        """Generate professional executive summary using LLM"""
        prompt = f"""
        As a senior investment analyst, create an executive summary for due diligence.

        COMPANY = {context('company')}
        ANALYSIS_TYPE = {context('analysis_type')}

        FINANCIAL FINDINGS:
        {context('financial_analysis')}

        LEGAL FINDINGS:
        {context('legal_analysis')}

        MARKET FINDINGS:
        {context('market_analysis')}

        RAG CONTEXT:
        {context['rag_insights']} documents analyzed from SEC filings

        Create a concise, professional executive summary (3-4 paragraphs) highlighting:
        1. Key financial strengths/weaknesses
        2. Critical risk factors  
        3. Market positioning
        4. Overall investment attractiveness
        
        Format for institutional investors.
        """
        response = await self.OpenAI.chat.completions.create(
            model ="gpt-4",
            messages = [{"role": "user", "content": prompt}],
            temperature = 0.1
        )
        """OpenAI response structure:
        response = {
            "choices": [
                {
                    "message": {
                        "content": "Apple Inc. demonstrates strong financial performance..."
                    }
                },
                # Could have multiple choices if you request n>1
            ]
        }"""
        return response.choices[0].message.content

    async def _extract_llm_key_findings(self, context: Dict) -> Dict[str, Any]:
        """Extract structured key findings using LLM"""
        prompt = f"""
        Extract structured key findings from the analysis data:

        {context}

        Return as JSON with:
        - financial_metrics (key ratios, trends, performance)
        - legal_compliance (status, risks, regulatory issues) 
        - market_position (competitive landscape, growth prospects)
        - red_flags (critical concerns requiring attention)
        - opportunities (potential upside factors)
        """
        response = await self.OpenAI.chat.completions.create(
            model = "gpt-4",
            messages = [{"role": "user", "content": prompt}],
            response_format = {"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)
    
    async def _generate_llm_risk_assessment(self, context: Dict) -> Dict[str, Any]:
        """Generate comprehensive risk assessment using LLM"""
        prompt = f"""
        Perform comprehensive risk assessment for investment due diligence:

        {context}

        Assess and return JSON with:
        - overall_risk_level (LOW/MEDIUM/HIGH/CRITICAL)
        - financial_risks (liquidity, profitability, debt, etc.)
        - legal_risks (compliance, litigation, regulatory)
        - market_risks (competition, industry trends, disruption)
        - operational_risks (management, execution, scalability)
        - risk_mitigation_recommendations
        - risk_score (1-10 scale)
        """

        response = self.OpenAI.chat.completions.create(
            model = "gpt-4",
            messages = [{"role": "user", "content": prompt}],
            response_format = {"type": "json_object"}

        )
        return json.loads(response.choices[0].message.content)
    
    async def _generate_llm_recommendation(self, context: Dict) -> Dict[str, Any]:
        """Generate data-driven investment recommendation using LLM"""
        prompt = f"""
        Provide investment recommendation based on comprehensive due diligence:

        {context}

        Return JSON with:
        - recommendation (STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL)
        - confidence_level (HIGH/MEDIUM/LOW)
        - price_target (if applicable)
        - time_horizon (SHORT/MEDIUM/LONG_TERM)
        - key_investment_thesis (3-4 bullet points)
        - catalysts (potential events that could impact valuation)
        - downside_risks (what could go wrong)
        """
        response = await self.OpenAI.chat.completions.create(
            model = "gpt-4",
            messages = [{"role": "user", "content": prompt}],
            response_format = {"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    
    def _format_memory_context(self, memory_insights: Dict) -> str:
        """Format memory insights for LLM context"""
        context_parts = []

        if memory_insights.get("financial_patterns"):
            context_parts.append(f"SIMILAR FINANCIAL PATTERNS FROM PREVIOUS ANALYSES:")
            for pattern in memory_insights["financial_patterns"]:
                context_parts.append(f"- {pattern['company']}: {pattern['insight']}")

        if memory_insights.get("legal_risks"):
            context_parts.append(f"RELEVANT LEGAL RISKS FROM SIMILAR COMPANIES:")
            for risk in memory_insights["legal_risks"]:
                context_parts.append(f"-{risk['company']}: {risk['risk_type']} - {risk['description']}")
        
        return "\n".join(context_parts) if context_parts else "No relevant historical insights found."
    
    def _extract_risk_severity(self, risk_description: str) -> str:
        """Extract risk severity from risk description text"""
        risk_lower = risk_description.lower()

        if (any(word in risk_lower for word in ["critical", "severe", "major", "high risk", "urgent"])):
            return "high"
        
        elif (any(word in risk_lower for word in ["moderate", "medium", "potential", "could"])):
            return "medium"
        
        elif (any(word in risk_lower for word in ["minor", "low", "minimal", "slight"])):
               return "low"
        
        else:
            return "medium" # Default

    def _prepare_synthesis_llm_context(self, state: AnalysisState) -> Dict[str, Any]:
        """Prepare comprehensive context for LLM synthesis"""
        return {
            "company": state['company_ticker'],
            "analysis_type": state["analysis_type"],
            "financial_analysis": state.get("financial_analysis", {}),
            "legal_analysis": state.get("legal_analysis", {}),
            "market_analysis": state.get("market_analysis", {}),
            "rag_insights": f"analysed {sum(len(v) for v in state.get('rag_context').values())} SEC documents",
            "questions_asked" : state.get("questions", []),
            "analysis_duration": self._calculate_duration(state["start_time"]),
            "data_quality": "HIGH" if not state.get("warnings") else "MEDIUM"
        }
    
    def _extract_supporting_evidence(self, state: AnalysisState) -> List[str]:
        """Extract key evidence supporting the analysis"""
        evidence = []

        if state.get("financial_analysis", {}).get("key_metrics"):
            evidence.append("Financial ratios and performance metrics from SEC filings")

        if state.get("rag_context", {}).get("financial"):
            evidence.append(f"{len(state['rag_context']['financial'])} financial documents analyzed")

        if not state.get("errors"):
            evidence.append("Analysis completed without critical errors")

        return evidence
    
    def _calculate_confidence_score(self, state: AnalysisState) -> float:
        """Calculate confidence score based on data quality and completeness"""
        score = 1.0
        # Deduct for errors
        if state.get("errors"):
            score -= 0.3
        # Deduct for warnings
        if state.get("warnings"):
            score -= len(state["warnings"])*0.1
        # Boost for complete analyses
        if all(
            [
                state.get("financial_analysis"),
                state.get("legal_analysis"),
                state.get("market_analysis")
            ]
        ):
            score += 0.2

        return max(0.0, min(1.0, score))

        
    






