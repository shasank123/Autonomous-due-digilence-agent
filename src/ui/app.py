# src/ui/app.py
import streamlit as st
import requests
import time
import json
import pandas as pd
import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dotenv import load_dotenv
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Frontend")

# Load environment variables
load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page Configuration
st.set_page_config(
    page_title="Autonomous Due Diligence Agent",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium UI
st.markdown("""
    <style>
    .main { padding: 1rem 2rem; }
    .stButton>button { width: 100%; border-radius: 5px; font-weight: 600; }
    .metric-card { background-color: #f0f2f6; border-radius: 10px; padding: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    h1, h2, h3 { color: #1e1e1e; }
    .status-badge { padding: 0.25rem 0.75rem; border-radius: 15px; font-size: 0.85rem; font-weight: 600; }
    div[data-testid="stExpander"] details summary p { font-weight: 600; font-size: 1.1em; }
    </style>
""", unsafe_allow_html=True)

class ApiClient:
    """Client for interacting with the backend API"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()

    def check_health(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check API health status"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                return True, response.json()
            return False, None
        except requests.RequestException as e:
            logger.error(f"Health check failed: {e}")
            return False, None

    def start_analysis(self, ticker: str, analysis_type: str, priority: str = "normal", context: str = "") -> Optional[Dict[str, Any]]:
        """Initiate a new analysis session"""
        payload = {
            "company_ticker": ticker,
            "analysis_type": analysis_type,
            "priority": priority,
            "additional_context": context,
            "timeout_seconds": 600 # 10 mins
        }
        try:
            # Matches src/api/main.py: /analyze endpoint
            response = self.session.post(f"{self.base_url}/analyze", json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                st.error(f"Ticker '{ticker}' not found in SEC database.")
                return None
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")
                return None
        except requests.RequestException as e:
            st.error(f"Connection Error: {e}")
            return None

    def get_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific session"""
        try:
            # Matches src/api/main.py: /analysis/{session_id}
            response = self.session.get(f"{self.base_url}/analysis/{session_id}", timeout=10)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                st.error(f"Session {session_id} not found.")
                return None
            else:
                return None
        except requests.RequestException as e:
            return None

# Initialize Client
api_client = ApiClient(API_BASE_URL)

# --- Components ---

def render_sidebar():
    """Render the sidebar navigation and info"""
    st.sidebar.title("🤖 Autonomous DD")
    st.sidebar.caption("v2.0 | Multi-Agent System")
    st.sidebar.markdown("---")
    
    # Navigation logic
    if 'page_override' in st.session_state:
        page = st.session_state.page_override
        del st.session_state.page_override
    else:
        page = st.sidebar.radio("Navigation", ["New Analysis", "History", "System Status"], index=0)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🕒 Recent History")
    
    if 'history' not in st.session_state:
        st.session_state.history = []
        
    if st.session_state.history:
        for idx, item in enumerate(reversed(st.session_state.history[-5:])): 
            col1, col2 = st.sidebar.columns([3, 1])
            if col1.button(f"{item['ticker']} ({item['type']})", key=f"hist_btn_{idx}"):
                st.session_state.page_override = "History"
                st.session_state.auto_load_session = item['session_id']
                st.rerun()
    else:
        st.sidebar.caption("No analyses run yet.")

    st.sidebar.markdown("---")
    st.sidebar.caption(f"API: `{API_BASE_URL}`")
    
    return page

def render_financial_charts(financial_data: Dict[str, Any]):
    """Render financial charts using real data from structured metrics"""
    st.subheader("[DATA] Financial Visualization")
    
    # Extract structured data
    structured = financial_data.get('structured_data', {})
    metrics = structured.get('metrics', {})
    
    if not metrics:
        st.warning("No structured financial data available for visualization.")
        st.caption("The analysis may have completed, but time-series data could not be extracted.")
        return
    
    # Prepare data for charts
    revenue_data = metrics.get('Revenue', [])
    net_income_data = metrics.get('Net Income', [])
    
    if not revenue_data and not net_income_data:
        st.info("Insufficient time-series data for charting. Showing available metrics below.")
        # Still show what we have
        render_metrics_grid(structured)
        return
    
    # Extract periods and values
    periods = []
    revenue_values = []
    net_income_values = []
    
    # Use revenue periods as baseline (most companies report this)
    for item in reversed(revenue_data):  # Reverse to show oldest to newest
        periods.append(item['period'][:4])  # Get year
        revenue_values.append(item['value'] / 1_000_000_000)  # Convert to billions
    
    # Match net income to same periods
    for period in periods:
        matched = next((item for item in net_income_data if item['period'][:4] == period), None)
        if matched:
            net_income_values.append(matched['value'] / 1_000_000_000)
        else:
            net_income_values.append(None)
    
    # Create interactive Plotly chart
    fig = go.Figure()
    
    if revenue_values:
        fig.add_trace(go.Bar(
            x=periods,
            y=revenue_values,
            name='Revenue',
            marker_color='#4facfe',
            text=[f'${v:.1f}B' for v in revenue_values],
            textposition='outside'
        ))
    
    if any(net_income_values):
        fig.add_trace(go.Scatter(
            x=periods,
            y=net_income_values,
            name='Net Income',
            line=dict(color='#00f2fe', width=4),
            mode='lines+markers',
            marker=dict(size=10),
            text=[f'${v:.1f}B' if v else 'N/A' for v in net_income_values],
            textposition='top center'
        ))
    
    fig.update_layout(
        title='Revenue vs Net Income Trend',
        xaxis_title='Fiscal Year',
        yaxis_title='USD (Billions)',
        template='plotly_white',
        height=450,
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show metrics grid and trends
    col1, col2 = st.columns(2)
    
    with col1:
        render_metrics_grid(structured)
    
    with col2:
        render_trend_analysis(structured)

def render_metrics_grid(structured_data: Dict[str, Any]):
    """Display key financial metrics in a grid"""
    st.markdown("#### 📋 Key Metrics (Latest Period)")
    
    metrics = structured_data.get('metrics', {})
    latest_period = structured_data.get('latest_period', 'N/A')
    
    st.caption(f"Period: {latest_period}")
    
    # Display metrics in 2-column grid
    metric_list = [
        ('Revenue', 'Revenue'),
        ('Net Income', 'Net Income'),
        ('Total Assets', 'Total Assets'),
        ('Cash', 'Cash and Cash Equivalents'),
    ]
    
    for i in range(0, len(metric_list), 2):
        c1, c2 = st.columns(2)
        
        # First metric
        if i < len(metric_list):
            label, key = metric_list[i]
            data = metrics.get(key, [])
            if data:
                value = data[0]['value'] / 1_000_000_000  # Convert to billions
                c1.metric(label, f"${value:.2f}B")
            else:
                c1.metric(label, "N/A")
        
        # Second metric
        if i + 1 < len(metric_list):
            label, key = metric_list[i + 1]
            data = metrics.get(key, [])
            if data:
                value = data[0]['value'] / 1_000_000_000
                c2.metric(label, f"${value:.2f}B")
            else:
                c2.metric(label, "N/A")

def render_trend_analysis(structured_data: Dict[str, Any]):
    """Display growth trends"""
    st.markdown("#### 📈 Growth Trends (YoY)")
    
    trends = structured_data.get('trends', {})
    
    if not trends:
        st.caption("No trend data available")
        return
    
    # Display each trend
    for trend_key, value in trends.items():
        # Format the label
        label = trend_key.replace('_growth', '').replace('_', ' ').title()
        
        # Color code based on positive/negative
        delta_color = "normal" if value >= 0 else "inverse"
        
        st.metric(
            label=label,
            value=f"{value:+.1f}%",
            delta=f"{'Growth' if value >= 0 else 'Decline'}",
            delta_color=delta_color
        )

def render_investment_summary(structured_data: Dict[str, Any], analysis_text: str):
    """Render comprehensive investment summary with scoring"""
    st.markdown("## [TIP] Investment Summary")
    
    ratios = structured_data.get('ratios', {})
    trends = structured_data.get('trends', {})
    
    # Calculate investment score
    score = 50  # Start at neutral
    
    # Positive indicators
    if trends.get('revenue_growth', 0) > 5:
        score += 10
    if trends.get('net_income_growth', 0) > 5:
        score += 10
    if ratios.get('profit_margin', 0) > 15:
        score += 10
    if ratios.get('current_ratio', 0) > 1.5:
        score += 10
    if ratios.get('roe', 0) > 15:
        score += 10
    
    # Negative indicators
    if trends.get('revenue_growth', 0) < -5:
        score -= 15
    if trends.get('net_income_growth', 0) < -10:
        score -= 15
    if ratios.get('debt_to_equity', 10) > 2:
        score -= 10
    
    # Clamp score between 0-100
    score = max(0, min(100, score))
    
    # Determine verdict
    if score >= 75:
        verdict = "🟢 STRONG BUY"
        verdict_color = "green"
    elif score >= 60:
        verdict = "🟢 BUY"
        verdict_color = "green"
    elif score >= 40:
        verdict = "🟡 HOLD"
        verdict_color = "orange"
    elif score >= 25:
        verdict = "🔴 SELL"
        verdict_color = "red"
    else:
        verdict = "🔴 STRONG SELL"
        verdict_color = "red"
    
    # Display summary
    col1, col2, col3 = st.columns(3)
    col1.metric("Investment Score", f"{score}/100")
    col2.metric("Verdict", verdict)
    col3.metric("Confidence", "High" if score > 60 or score < 40 else "Medium")
    
    # Show key ratios
    if ratios:
        st.markdown("### Key Financial Ratios")
        ratio_cols = st.columns(4)
        
        ratio_display = [
            ("Profit Margin", ratios.get('profit_margin'), "%"),
            ("ROE", ratios.get('roe'), "%"),
            ("Current Ratio", ratios.get('current_ratio'), "x"),
            ("Debt/Equity", ratios.get('debt_to_equity'), "x"),
        ]
        
        for i, (label, value, unit) in enumerate(ratio_display):
            if value is not None:
                ratio_cols[i].metric(label, f"{value:.2f}{unit}")
            else:
                ratio_cols[i].metric(label, "N/A")

def render_report_content(result: Dict[str, Any], ticker: str):
    """Parses and renders the complex JSON result from Orchestrator"""
    
    # 1. Synthesis Report (The LLM Summary)
    synthesis = result.get('result', {}).get('synthesis_report', {})
    
    # Fallback logic if synthesis is missing but analysis text exists
    if not synthesis and 'analysis' in result.get('result', {}): 
        synthesis = {'executive_summary': result['result']['analysis']}
    elif not synthesis and 'analysis' in result:
        synthesis = {'executive_summary': result['analysis']}

    if synthesis:
        st.markdown("## [NOTE] Executive Summary")
        st.info(synthesis.get('executive_summary', 'No summary available.'))
        
        # Risk Assessment
        risks = synthesis.get('risk_assessment', {})
        if risks:
            with st.expander("[WARN] Risk Assessment", expanded=False):
                # Handle missing/different key names gracefully
                level = risks.get('risk_level') or risks.get('overall_risk_level', 'Unknown')
                score = risks.get('risk_score') or risks.get('score', 'N/A')
                
                st.markdown(f"**Risk Level:** {level}")
                st.markdown(f"**Score:** {score}/10")
                
                # Render keys list if available
                key_risks = risks.get('key_risks', [])
                if key_risks:
                    st.write("**Identified Risks:**")
                    for r in key_risks:
                        st.markdown(f"- {r}")
                else:
                    st.caption("No specific risk factors identified.")

        # Recommendation
        rec = synthesis.get('recommendation', {})
        if rec:
            with st.expander("[TIP] Investment Verdict", expanded=True):
                col1, col2 = st.columns(2)
                col1.metric("Verdict", rec.get('verdict', 'HOLD'))
                col2.metric("Confidence", rec.get('confidence', 'Medium'))
                st.write(f"**Reasoning:** {rec.get('reasoning', '')}")
    
    # Show Investment Summary with structured data
    st.divider()
    # Handle multiple response formats:
    # 1. Orchestrator direct format: result["financial_analysis"]["structured_data"]
    # 2. Wrapped format: result["result"]["financial_analysis"]["structured_data"]
    
    # Try direct top-level access first (orchestrator format)
    fin_data = result.get('financial_analysis', {})
    
    # Fallback: check if wrapped under 'result' key
    if not fin_data or not fin_data.get('structured_data'):
        inner_result = result.get('result', {})
        fin_data = inner_result.get('financial_analysis', {})
    
    # Final fallback: check if structured_data is at top level of inner_result
    if not fin_data or not fin_data.get('structured_data'):
        inner_result = result.get('result', {})
        if inner_result.get('structured_data'):
            fin_data = inner_result  # Direct financial_agent response format
    
    if fin_data and fin_data.get('structured_data'):
        render_investment_summary(
            fin_data.get('structured_data', {}),
            fin_data.get('analysis', '')
        )

    # 2. Detailed Agent Findings (Tabs)
    st.divider()
    st.subheader("[SEARCH] Deep Dive Analysis")
    
    tab1, tab2, tab3 = st.tabs(["[MONEY] Financial", "⚖️ Legal", "🌍 Market"])
    
    # --- Financial Tab ---
    with tab1:
        # Handle both formats: orchestrator (financial_analysis nested) and direct (at top level)
        inner_result = result.get('result', {})
        fin_data = inner_result.get('financial_analysis', {})
        
        # Fallback: if no nested financial_analysis, use inner_result directly
        if not fin_data and inner_result.get('structured_data'):
            fin_data = inner_result
        
        if fin_data:
            st.markdown(fin_data.get('analysis', 'No detailed financial text.'))
            render_financial_charts(fin_data) # Helper function
            
            # Safe Message Logging
            if 'messages' in fin_data:
                with st.expander("View Financial Team Log"):
                    for msg in fin_data['messages']:
                        # Handle AutoGen message objects
                        if hasattr(msg, 'content') and hasattr(msg, 'source'):
                            content = str(msg.content) if msg.content else ''
                            st.text(f"{msg.source}: {content[:150]}...")
                        elif isinstance(msg, dict):
                            st.text(f"{msg.get('source', 'Agent')}: {str(msg.get('content', ''))[:150]}...")
                        else:
                            pass  # Skip raw objects
        else:
            st.caption("No financial data returned.")

    # --- Legal Tab (CRASH FIX APPLIED HERE) ---
    with tab2:
        # Debug: Check what keys are available
        inner_result = result.get('result', {})
        
        # Try multiple paths for legal data
        leg_data = inner_result.get('legal_analysis', {})
        
        # Debug expander to help troubleshoot
        with st.expander("Debug: Legal Data Structure", expanded=False):
            st.json({
                "inner_result_keys": list(inner_result.keys()) if inner_result else [],
                "legal_analysis_type": str(type(leg_data).__name__),
                "legal_analysis_keys": list(leg_data.keys()) if isinstance(leg_data, dict) and leg_data else [],
                "leg_data_preview": str(leg_data)[:500] if leg_data else "Empty",
            })
        
        if leg_data and isinstance(leg_data, dict):
            # Check if it's an error response
            if leg_data.get('success') == False and leg_data.get('error'):
                st.warning(f"⚠️ Legal Analysis Issue: {leg_data.get('error')}")
                st.caption("Legal due diligence data is limited. This may be because:\n- Insufficient legal filings in the database\n- Company doesn't have relevant SEC legal disclosures\n- Data needs to be seeded with legal-specific documents")
            else:
                # Try to get summary from multiple possible keys
                summary_text = (
                    leg_data.get('summary') or 
                    leg_data.get('analysis') or
                    leg_data.get('content') or
                    leg_data.get('text') or
                    None
                )
                
                if summary_text:
                    st.markdown(summary_text)
                else:
                    # Show what we have if no known summary key
                    st.info("Legal analysis completed but no summary available.")
                    with st.expander("View Raw Data"):
                        st.json(leg_data)
            
            # Safe Message Logging
            if 'messages' in leg_data:
                with st.expander("View Legal Team Log"):
                    for msg in leg_data['messages']:
                        # Handle AutoGen message objects
                        if hasattr(msg, 'content') and hasattr(msg, 'source'):
                            content = str(msg.content) if msg.content else ''
                            st.text(f"{msg.source}: {content[:150]}...")
                        elif isinstance(msg, dict):
                            st.text(f"{msg.get('source', 'Agent')}: {str(msg.get('content', ''))[:150]}...")
                        else:
                            pass  # Skip raw objects
        elif leg_data:
            # leg_data is not a dict but contains something
            st.info(f"Legal analysis returned non-dict data ({type(leg_data).__name__})")
            st.text(str(leg_data)[:1000])
        else:
            st.caption("No legal data returned.")

    # --- Market Tab ---
    with tab3:
        mkt_data = result.get('result', {}).get('market_analysis', {})
        if mkt_data and isinstance(mkt_data, dict):
            # Check if it's an error response
            if mkt_data.get('success') == False and mkt_data.get('error'):
                st.warning(f"⚠️ Market Analysis Issue: {mkt_data.get('error')}")
                st.caption("Market analysis data is limited.")
            else:
                summary_text = mkt_data.get('summary') or mkt_data.get('analysis') or None
                if summary_text:
                    st.markdown(summary_text)
                else:
                    st.info("Market analysis completed but no summary available.")
            
            # Safe Message Logging
            if 'messages' in mkt_data:
                with st.expander("View Market Team Log"):
                    for msg in mkt_data['messages']:
                        # Handle AutoGen message objects
                        if hasattr(msg, 'content') and hasattr(msg, 'source'):
                            content = str(msg.content) if msg.content else ''
                            st.text(f"{msg.source}: {content[:150]}...")
                        elif isinstance(msg, dict):
                            st.text(f"{msg.get('source', 'Agent')}: {str(msg.get('content', ''))[:150]}...")
                        else:
                            pass  # Skip raw objects
        else:
            st.caption("No market data returned.")

    # 3. Download
    st.divider()
    # Handle serialization of non-serializable objects (like datetime)
    try:
        report_text = json.dumps(result, indent=2, default=str)
        st.download_button(
            label="📥 Download Full JSON Report",
            data=report_text,
            file_name=f"{ticker}_DD_Report_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
    except Exception as e:
        st.error(f"Could not prepare download: {e}")

def run_analysis_flow(ticker: str, analysis_type: str, priority: str, context: str):
    """Orchestrates the analysis UI flow with polling"""
    
    # 1. Start Analysis
    with st.status("[LAUNCH] Initializing Agent Swarm...", expanded=True) as status:
        st.write("Handshaking with API...")
        response = api_client.start_analysis(ticker, analysis_type, priority, context)
        
        if not response:
            status.update(label="[ERROR] Connection Failed", state="error")
            return

        session_id = response['session_id']
        st.write(f"[OK] Session Started: `{session_id}`")
        
        # Add to history
        st.session_state.history.append({
            'ticker': ticker, 'session_id': session_id, 'type': analysis_type,
            'timestamp': datetime.now().strftime("%H:%M")
        })
        
        # 2. Polling Loop
        st.write("[REFRESH] Orchestrator running... (This may take 1-2 mins)")
        progress_bar = st.progress(0)
        
        while True:
            time.sleep(3) # Poll every 3 seconds
            
            update = api_client.get_status(session_id)
            if not update:
                continue
                
            status_text = update.get('status', 'unknown')
            prog_val = update.get('progress', 0)
            
            # Update Progress Bar
            progress_bar.progress(min(prog_val, 100))
            
            # Handle States
            if status_text == 'queued':
                st.write("⏳ Analysis in queue...")
            elif status_text == 'processing':
                # Show dynamic updates if available from Redis messages
                if update.get('warnings'):
                    st.warning(f"Agent Warning: {update['warnings'][-1]}")
            elif status_text == 'completed':
                progress_bar.progress(100)
                status.update(label="[OK] Analysis Complete!", state="complete", expanded=False)
                render_report_content(update, ticker)
                break
            elif status_text == 'failed':
                status.update(label="[ERROR] Analysis Failed", state="error")
                st.error(f"Error: {update.get('error')}")
                break

# --- Pages ---

def render_new_analysis():
    st.title("[LAUNCH] New Analysis")
    
    with st.form("analysis_form"):
        col1, col2 = st.columns(2)
        with col1:
            ticker = st.text_input("Company Ticker", placeholder="e.g. AAPL").upper()
            analysis_type = st.selectbox("Analysis Type", ["comprehensive", "financial", "legal", "market"])
        with col2:
            priority = st.selectbox("Priority", ["normal", "high"])
            context = st.text_area("Specific Questions / Context", placeholder="e.g., Focus on recent antitrust risks...")
            
        submitted = st.form_submit_button("Start Due Diligence", type="primary")
        
    if submitted and ticker:
        run_analysis_flow(ticker, analysis_type, priority, context)

def render_history():
    st.title("🗄️ Analysis History")
    
    # Auto-load logic
    if 'auto_load_session' in st.session_state:
        sid = st.session_state.auto_load_session
        del st.session_state.auto_load_session
        
        with st.spinner("Retrieving archived report..."):
            data = api_client.get_status(sid)
            if data:
                st.success(f"Loaded report for {data.get('company')}")
                render_report_content(data, data.get('company'))
            else:
                st.error("Session expired or not found.")
        st.divider()

    # List view
    if not st.session_state.history:
        st.info("No analysis history found.")
        return

    for item in reversed(st.session_state.history):
        with st.container(border=True):
            c1, c2, c3 = st.columns([2, 4, 2])
            c1.markdown(f"**{item['ticker']}**")
            c2.caption(f"Session: {item['session_id']}")
            if c3.button("Load Report", key=item['session_id']):
                st.session_state.auto_load_session = item['session_id']
                st.rerun()

def render_system_status():
    st.title("🖥️ System Status")
    
    if st.button("Refresh Status"):
        st.rerun()
        
    healthy, data = api_client.check_health()
    
    if healthy:
        c1, c2, c3 = st.columns(3)
        c1.metric("Status", data.get('status', 'Unknown'), delta="Online")
        c2.metric("Uptime", f"{data.get('uptime', 0):.0f}s")
        c3.metric("Active Sessions", data.get('active_sessions', 0))
        
        st.subheader("Component Health")
        st.json(data.get('components', {}))
    else:
        st.error("Cannot connect to Backend API.")
        st.warning(f"Ensure `src/api/main.py` is running on {API_BASE_URL}")

# --- Main ---
def main():
    page = render_sidebar()
    if page == "New Analysis": render_new_analysis()
    elif page == "History": render_history()
    elif page == "System Status": render_system_status()

if __name__ == "__main__":
    main()